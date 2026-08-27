"""TAPNext-guided OpenCV replacement-mask service on Modal."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import modal

from mask_stage import (
    clip_polygon_to_frame,
    create_masks,
    parse_s3_url,
    s3_url,
    tracking_corners_to_source,
    validate_request,
    validate_tracking,
)


APP_NAME = "opencv-mask-modal-app"
SECRET_NAME = "studio-s3"
REQUIRED_SECRET_KEYS = (
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY",
    "S3_ENDPOINT",
    "S3_BUCKET",
    "S3_REGION",
)
MASK_FILENAME = "occlusion-mask.mkv"
MASK_CODEC = "ffv1"
MASK_PIXEL_FORMAT = "gray"
MAX_VIDEO_BYTES = 250 * 1024 * 1024
MAX_TRACKING_BYTES = 25 * 1024 * 1024
FUNCTION_TIMEOUT = 20 * 60
MAX_CONTAINERS = 4


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "boto3>=1.40,<2",
        "numpy>=1.26,<3",
        "opencv-python-headless>=4.10,<5",
    )
    .add_local_python_source("mask_stage")
)
app = modal.App(APP_NAME)
studio_s3_secret = modal.Secret.from_name(
    SECRET_NAME, required_keys=list(REQUIRED_SECRET_KEYS)
)


def _s3_settings() -> dict[str, str]:
    missing = [key for key in REQUIRED_SECRET_KEYS if not os.environ.get(key)]
    if missing:
        raise RuntimeError(
            f"Modal secret {SECRET_NAME} is missing required keys: {', '.join(missing)}"
        )
    return {key: os.environ[key] for key in REQUIRED_SECRET_KEYS}


def _s3_client(settings: dict[str, str]):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        aws_access_key_id=settings["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=settings["S3_SECRET_ACCESS_KEY"],
        endpoint_url=settings["S3_ENDPOINT"],
        region_name=settings["S3_REGION"],
        config=Config(s3={"addressing_style": "path"}),
    )


def _download(client: Any, bucket: str, key: str, path: Path, maximum: int) -> Path:
    head = client.head_object(Bucket=bucket, Key=key)
    size = int(head.get("ContentLength", -1))
    if not 0 < size <= maximum:
        raise ValueError(f"s3://{bucket}/{key} exceeds its {maximum}-byte limit")
    partial = path.with_suffix(path.suffix + ".part")
    with partial.open("wb") as stream:
        client.download_fileobj(bucket, key, stream)
    if partial.stat().st_size != size:
        partial.unlink(missing_ok=True)
        raise IOError(f"s3://{bucket}/{key} download size mismatch")
    partial.replace(path)
    return path


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _upload(
    client: Any, bucket: str, key: str, path: Path, content_type: str
) -> str:
    digest, size = _sha256(path)
    client.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"ContentType": content_type, "Metadata": {"sha256": digest}},
    )
    head = client.head_object(Bucket=bucket, Key=key)
    if (
        int(head.get("ContentLength", -1)) != size
        or head.get("Metadata", {}).get("sha256") != digest
    ):
        raise IOError(f"uploaded artifact verification failed for s3://{bucket}/{key}")
    return s3_url(bucket, key)


def _write_image(path: Path, image_data: Any) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_data):
        raise IOError(f"failed to write debug image {path.name}")


def _draw_polygon_locator(image: Any, corners: list[list[float]]) -> Any:
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    inset_width = min(160, max(48, width // 3))
    inset_height = min(110, max(36, height // 4))
    left = max(4, width - inset_width - 8)
    top = max(4, min(48, height - inset_height - 8))
    frame_quad = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float64,
    )
    tracked = np.asarray(corners, dtype=np.float64)
    combined = np.vstack([frame_quad, tracked])
    minimum = np.min(combined, axis=0)
    maximum = np.max(combined, axis=0)
    span = np.maximum(maximum - minimum, 1.0)

    def locate(points: Any) -> Any:
        normalized = (points - minimum) / span
        mapped = normalized * np.array([inset_width - 16, inset_height - 16])
        return np.rint(mapped + np.array([left + 8, top + 8])).astype(np.int32)

    cv2.rectangle(
        image, (left, top), (left + inset_width, top + inset_height), (0, 0, 0), -1
    )
    cv2.rectangle(
        image,
        (left, top),
        (left + inset_width, top + inset_height),
        (255, 255, 255),
        1,
    )
    cv2.polylines(image, [locate(frame_quad)], True, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.polylines(image, [locate(tracked)], True, (0, 255, 255), 2, cv2.LINE_AA)
    return image


def _write_debug_bundle(
    directory: Path,
    frame_index: int,
    frame: Any,
    corners: list[list[float]],
    green_mask: Any,
    hard_replace_mask: Any,
    replace_mask: Any,
    include_hard_anchor: bool,
    offscreen_corner_count: int,
    fully_offscreen: bool,
) -> list[tuple[Path, str]]:
    import cv2
    import numpy as np

    stem = f"frame_{frame_index:06d}"
    original = directory / f"{stem}_original.jpg"
    polygon = directory / f"{stem}_polygon.jpg"
    green = directory / f"{stem}_green-mask.png"
    hard_replace = directory / f"{stem}_hard-replace-mask.png"
    replace = directory / f"{stem}_replace-mask.png"
    overlay = directory / f"{stem}_overlay.jpg"

    polygon_image = frame.copy()
    clipped = clip_polygon_to_frame(corners, frame.shape[1], frame.shape[0])
    if len(clipped) >= 2:
        points = np.rint(np.asarray(clipped, dtype=np.float64)).astype(np.int32)
        cv2.polylines(polygon_image, [points], True, (0, 255, 255), 2, cv2.LINE_AA)
    if offscreen_corner_count > 0:
        polygon_image = _draw_polygon_locator(polygon_image, corners)
    status = (
        "fully off-screen"
        if fully_offscreen
        else f"off-screen corners: {offscreen_corner_count}"
    )
    if offscreen_corner_count > 0 or fully_offscreen:
        cv2.putText(
            polygon_image,
            status,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            polygon_image,
            status,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    overlay_image = frame.astype(np.float32)
    tint = np.zeros_like(frame)
    tint[:, :] = (0, 255, 255)
    weight = (replace_mask.astype(np.float32) / 255.0 * 0.45)[:, :, None]
    overlay_image = np.rint(
        overlay_image * (1.0 - weight) + tint.astype(np.float32) * weight
    ).astype(np.uint8)

    _write_image(original, frame)
    _write_image(polygon, polygon_image)
    _write_image(green, green_mask)
    if include_hard_anchor:
        _write_image(hard_replace, hard_replace_mask)
    _write_image(replace, replace_mask)
    _write_image(overlay, overlay_image)
    artifacts = [
        (original, "image/jpeg"),
        (polygon, "image/jpeg"),
        (green, "image/png"),
        (replace, "image/png"),
        (overlay, "image/jpeg"),
    ]
    if include_hard_anchor:
        artifacts.insert(3, (hard_replace, "image/png"))
    return artifacts


def _start_mask_encoder(path: Path, width: int, height: int, fps: float):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "gray",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:.9f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            MASK_CODEC,
            "-level",
            "3",
            "-pix_fmt",
            MASK_PIXEL_FORMAT,
            "-y",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _finish_mask_encoder(process: subprocess.Popen[bytes]) -> None:
    assert process.stdin is not None
    assert process.stderr is not None
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"FFV1 mask encoding failed: {stderr.strip()}")


def _verify_mask_video(
    path: Path, width: int, height: int, fps: float, frame_count: int
) -> None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("encoded mask must contain exactly one video stream")
    stream = streams[0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    encoded_fps = float(numerator) / float(denominator)
    if (
        stream.get("codec_name") != MASK_CODEC
        or stream.get("pix_fmt") != MASK_PIXEL_FORMAT
        or int(stream.get("width", 0)) != width
        or int(stream.get("height", 0)) != height
        or int(stream.get("nb_read_frames", 0)) != frame_count
        or abs(encoded_fps - fps) > 0.001
    ):
        raise RuntimeError("encoded mask does not match the source media contract")


def _process(request: dict[str, Any]) -> dict[str, Any]:
    import cv2

    normalized = validate_request(request)
    video_bucket, video_key = parse_s3_url(normalized["video_url"], "video_url")
    tracking_bucket, tracking_key = parse_s3_url(
        normalized["tracking_url"], "tracking_url"
    )
    output_bucket, output_prefix = parse_s3_url(
        normalized["output_prefix"], "output_prefix", prefix=True
    )
    settings = _s3_settings()
    configured_bucket = settings["S3_BUCKET"]
    if {video_bucket, tracking_bucket, output_bucket} != {configured_bucket}:
        raise ValueError("all S3 URLs must use the bucket configured by studio-s3")
    client = _s3_client(settings)

    with tempfile.TemporaryDirectory(prefix="opencv-mask-", dir="/tmp") as scratch_name:
        scratch = Path(scratch_name)
        video_path = _download(
            client, video_bucket, video_key, scratch / "input.mp4", MAX_VIDEO_BYTES
        )
        tracking_path = _download(
            client,
            tracking_bucket,
            tracking_key,
            scratch / "tracking.json",
            MAX_TRACKING_BYTES,
        )
        try:
            tracking_payload = json.loads(tracking_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"tracking_url does not contain valid UTF-8 JSON: {error}") from None
        tracking = validate_tracking(tracking_payload)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("video_url is not a decodable video")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0 or fps <= 0:
            capture.release()
            raise ValueError("video_url has invalid dimensions or frame rate")

        tracking = tracking_corners_to_source(
            tracking_payload, tracking, width, height
        )

        mask_path = scratch / MASK_FILENAME
        debug_directory = scratch / "debug"
        debug_requested = set(normalized["debug_frames"])
        debug_artifacts: list[tuple[Path, str]] = []
        coverage: list[float] = []
        soft_coverage: list[float] = []
        partial_replacement_pixels: list[int] = []
        visible_polygon_pixels: list[int] = []
        offscreen_corner_counts: list[int] = []
        partial_offscreen_frames: list[int] = []
        fully_offscreen_frames: list[int] = []
        soft_enabled = normalized["soft_chroma"]["mode"] == "boundary"
        encoder = _start_mask_encoder(mask_path, width, height, fps)
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index >= len(tracking):
                    raise ValueError("video has more frames than tracking data")
                corners = tracking[frame_index]
                (
                    polygon_mask,
                    green_mask,
                    hard_replace_mask,
                    replace_mask,
                    frame_coverage,
                ) = create_masks(
                    frame,
                    corners,
                    normalized["green_hsv"]["lower"],
                    normalized["green_hsv"]["upper"],
                    normalized["soft_chroma"],
                )
                assert encoder.stdin is not None
                encoder.stdin.write(replace_mask.tobytes())
                polygon_pixels = int(cv2.countNonZero(polygon_mask))
                offscreen_corner_count = sum(
                    not (0 <= x < width and 0 <= y < height) for x, y in corners
                )
                fully_offscreen = polygon_pixels == 0
                visible_polygon_pixels.append(polygon_pixels)
                offscreen_corner_counts.append(offscreen_corner_count)
                if fully_offscreen:
                    fully_offscreen_frames.append(frame_index)
                elif offscreen_corner_count > 0:
                    partial_offscreen_frames.append(frame_index)
                if frame_coverage is not None:
                    coverage.append(frame_coverage)
                inside = polygon_mask > 0
                if polygon_pixels > 0:
                    soft_coverage.append(
                        float(replace_mask[inside].astype("float64").mean() / 255.0)
                    )
                partial_replacement_pixels.append(
                    int(((replace_mask > 0) & (replace_mask < 255)).sum())
                )
                if frame_index in debug_requested:
                    debug_artifacts.extend(
                        _write_debug_bundle(
                            debug_directory,
                            frame_index,
                            frame,
                            corners,
                            green_mask,
                            hard_replace_mask,
                            replace_mask,
                            soft_enabled,
                            offscreen_corner_count,
                            fully_offscreen,
                        )
                    )
                frame_index += 1
            if frame_index != len(tracking):
                raise ValueError("tracking data has more frames than the video")
            missing_debug = sorted(debug_requested - set(range(frame_index)))
            if missing_debug:
                raise ValueError(
                    f"debug_frames are outside the video: {', '.join(map(str, missing_debug))}"
                )
            if frame_index == 0:
                raise ValueError("video_url contains no decoded frames")
            _finish_mask_encoder(encoder)
        except BaseException:
            if encoder.poll() is None:
                encoder.kill()
            encoder.wait()
            raise
        finally:
            capture.release()

        _verify_mask_video(mask_path, width, height, fps, frame_index)
        mask_url = _upload(
            client,
            output_bucket,
            f"{output_prefix}{MASK_FILENAME}",
            mask_path,
            "video/x-matroska",
        )
        for path, content_type in debug_artifacts:
            _upload(
                client,
                output_bucket,
                f"{output_prefix}debug/{path.name}",
                path,
                content_type,
            )

        result = {
            "status": "completed",
            "frames": frame_index,
            "width": width,
            "height": height,
            "fps": round(fps, 6),
            "mask_url": mask_url,
            "debug_prefix": s3_url(output_bucket, f"{output_prefix}debug/"),
            "mask_mode": "soft-boundary" if soft_enabled else "hard",
            "soft_chroma": normalized["soft_chroma"],
            "green_coverage_mean": (
                round(sum(coverage) / len(coverage), 6) if coverage else None
            ),
            "green_coverage_min": round(min(coverage), 6) if coverage else None,
            "soft_replacement_coverage_mean": (
                round(sum(soft_coverage) / len(soft_coverage), 6)
                if soft_coverage
                else None
            ),
            "soft_replacement_coverage_min": (
                round(min(soft_coverage), 6) if soft_coverage else None
            ),
            "partial_replacement_pixels": sum(partial_replacement_pixels),
            "partial_replacement_pixels_max": max(partial_replacement_pixels),
            "partial_offscreen_frame_count": len(partial_offscreen_frames),
            "partial_offscreen_frames": partial_offscreen_frames,
            "fully_offscreen_frame_count": len(fully_offscreen_frames),
            "fully_offscreen_frames": fully_offscreen_frames,
            "maximum_offscreen_corner_count": max(offscreen_corner_counts),
            "visible_polygon_pixels_min": min(visible_polygon_pixels),
            "visible_polygon_pixels_max": max(visible_polygon_pixels),
        }
        result_path = scratch / "result.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _upload(
            client,
            output_bucket,
            f"{output_prefix}result.json",
            result_path,
            "application/json",
        )
        return result


@app.function(image=image)
def health() -> dict[str, Any]:
    return {
        "service": APP_NAME,
        "ready": True,
        "processor": "OpenCV fixed HSV inside TAPNext polygons",
        "mask": {"codec": MASK_CODEC, "pixelFormat": MASK_PIXEL_FORMAT},
        "maskModes": ["hard", "soft-boundary"],
        "offscreenPolygons": True,
    }


@app.function(
    image=image,
    secrets=[studio_s3_secret],
    cpu=4,
    memory=8_192,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    timeout=FUNCTION_TIMEOUT,
)
def create_occlusion_mask(request: dict[str, Any]) -> dict[str, Any]:
    """Create and durably commit a lossless replacement mask plus debug bundle."""
    try:
        return _process(request)
    except BaseException as error:
        error_type = f"{type(error).__module__}.{type(error).__name__}"
        raise RuntimeError(f"{error_type}: {error}") from None


@app.local_entrypoint()
def main(request_json: str = "", output_json: str = "") -> None:
    if not request_json:
        print(json.dumps(health.remote(), indent=2))
        return
    request_path = Path(request_json)
    result = create_occlusion_mask.remote(
        json.loads(request_path.read_text(encoding="utf-8"))
    )
    payload = json.dumps(result, indent=2)
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
