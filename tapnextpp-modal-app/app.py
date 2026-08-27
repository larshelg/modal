"""TAPNext++ S3-native tracking and homography stage on Modal."""

from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

from tracking_stage import (
    REQUIRED_SECRET_KEYS,
    artifact_ref,
    create_s3_client,
    download_verified,
    s3_settings_from_env,
    utc_now,
    validate_stage_request,
    write_json,
)


APP_NAME = "tapnextpp-modal-app"
TAPNET_REPOSITORY = "https://github.com/google-deepmind/tapnet.git"
TAPNET_COMMIT = "c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
TAPNET_ROOT = Path("/opt/tapnet")
CHECKPOINT_GENERATION = "1782153793676180"
CHECKPOINT_MD5 = "eea4fffa043f4f28503c130d826b116c"
CHECKPOINT_SIZE = 2_532_283_010
CHECKPOINT_PATH = TAPNET_ROOT / "checkpoints/tapnextpp_512.ckpt"
CHECKPOINT_URL = (
    "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt"
    f"?generation={CHECKPOINT_GENERATION}"
)
MODEL_INPUT_RESOLUTION = 512

GPU_TYPE = "L4"
MAX_CONTAINERS = 1
SCALEDOWN_WINDOW = 5 * 60
STARTUP_TIMEOUT = 30 * 60
FUNCTION_TIMEOUT = 30 * 60
MAX_FRAMES = 300
MIN_POINTS = 4
MAX_POINTS = 128


tracker_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("ca-certificates", "curl", "ffmpeg", "git")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install torch==2.10.0 torchvision==0.25.0 "
        "--index-url https://download.pytorch.org/whl/cu130",
        "python -m pip install 'imageio[ffmpeg]>=2.37,<3' 'numpy>=2,<3' "
        "'einops>=0.8,<1' 'boto3>=1.40,<2' 'opencv-python-headless>=4.12,<5'",
        f"git clone {TAPNET_REPOSITORY} {TAPNET_ROOT}",
        f"cd {TAPNET_ROOT} && git checkout {TAPNET_COMMIT}",
        f"mkdir -p {CHECKPOINT_PATH.parent}",
        f"curl --fail --location '{CHECKPOINT_URL}' --output {CHECKPOINT_PATH}",
        f"test $(stat -c%s {CHECKPOINT_PATH}) -eq {CHECKPOINT_SIZE}",
        f"echo '{CHECKPOINT_MD5}  {CHECKPOINT_PATH}' | md5sum --check -",
    )
    .env(
        {
            "PYTHONPATH": str(TAPNET_ROOT),
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_python_source("tracking_stage", "tracking_geometry")
)

control_image = modal.Image.debian_slim(python_version="3.11").add_local_python_source(
    "tracking_stage", "tracking_geometry"
)
stage_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "boto3>=1.40,<2",
        "imageio[ffmpeg]>=2.37,<3",
        "numpy>=2,<3",
        "opencv-python-headless>=4.12,<5",
    )
    .add_local_python_source("tracking_stage", "tracking_geometry")
)
app = modal.App(APP_NAME)
studio_s3_secret = modal.Secret.from_name(
    "studio-s3", required_keys=list(REQUIRED_SECRET_KEYS)
)
_model: Any | None = None


def _bounded_int(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return value


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    allowed = {
        "analysis_width",
        "analysis_height",
        "queries",
        "backward_tracking",
    }
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise ValueError(f"unsupported request fields: {', '.join(unknown)}")

    width = _bounded_int(request.get("analysis_width"), "analysis_width", 64, 1920)
    height = _bounded_int(
        request.get("analysis_height"), "analysis_height", 64, 1920
    )
    raw_queries = request.get("queries")
    if not isinstance(raw_queries, list):
        raise ValueError("queries must be an array")
    if not MIN_POINTS <= len(raw_queries) <= MAX_POINTS:
        raise ValueError(
            f"queries must contain between {MIN_POINTS} and {MAX_POINTS} points"
        )

    queries: list[list[float]] = []
    for index, query in enumerate(raw_queries):
        if not isinstance(query, list) or len(query) != 3:
            raise ValueError(f"queries[{index}] must be [frame, x, y]")
        frame, x, y = query
        if isinstance(frame, bool) or not isinstance(frame, int) or frame != 0:
            raise ValueError(
                f"queries[{index}][0] must be 0; TAPNext++ is initialized on frame 0"
            )
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError(f"queries[{index}][1] must be numeric")
        if isinstance(y, bool) or not isinstance(y, (int, float)):
            raise ValueError(f"queries[{index}][2] must be numeric")
        x_float = float(x)
        y_float = float(y)
        if not math.isfinite(x_float) or not 0 <= x_float < width:
            raise ValueError(f"queries[{index}][1] is outside the analysis canvas")
        if not math.isfinite(y_float) or not 0 <= y_float < height:
            raise ValueError(f"queries[{index}][2] is outside the analysis canvas")
        queries.append([0.0, x_float, y_float])

    backward_tracking = request.get("backward_tracking", False)
    if not isinstance(backward_tracking, bool):
        raise ValueError("backward_tracking must be a boolean")
    if backward_tracking:
        raise ValueError("TAPNext++ worker supports forward tracking only")
    return {
        "analysis_width": width,
        "analysis_height": height,
        "queries": queries,
        "backward_tracking": False,
    }


def model_metadata() -> dict[str, Any]:
    return {
        "name": "TAPNext++",
        "repository": TAPNET_REPOSITORY,
        "commit": TAPNET_COMMIT,
        "checkpointGeneration": CHECKPOINT_GENERATION,
        "checkpointMd5": CHECKPOINT_MD5,
        "checkpointSize": CHECKPOINT_SIZE,
        "checkpoint": CHECKPOINT_PATH.name,
        "inputResolution": MODEL_INPUT_RESOLUTION,
        "license": "Apache-2.0",
    }


def _load_model():
    global _model
    if _model is None:
        import torch
        from tapnet.tapnextpp.votsp2026.model import TAPNextPP

        _model = TAPNextPP.from_checkpoint(
            CHECKPOINT_PATH,
            device="cuda",
            half_precision=False,
            compile_model=False,
            input_resolution=MODEL_INPUT_RESOLUTION,
        )
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return _model


def _run_tracking_path(video_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    import imageio.v3 as iio
    import numpy as np
    import torch

    normalized = validate_request(request)
    started = time.monotonic()
    metadata = iio.immeta(video_path, plugin="FFMPEG")
    frames = iio.imread(video_path, plugin="FFMPEG")

    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError(f"decoded video has unsupported shape {frames.shape}")
    frames = frames[:, :, :, :3]
    frame_count, source_height, source_width, _ = frames.shape
    if not 2 <= frame_count <= MAX_FRAMES:
        raise ValueError(f"decoded frame count must be between 2 and {MAX_FRAMES}")

    analysis_width = normalized["analysis_width"]
    analysis_height = normalized["analysis_height"]
    analysis_queries = np.asarray(
        [query[1:] for query in normalized["queries"]], dtype=np.float32
    )
    source_queries = analysis_queries.copy()
    source_queries[:, 0] *= source_width / analysis_width
    source_queries[:, 1] *= source_height / analysis_height

    model = _load_model()
    state = None
    tracks: list[list[list[float]]] = []
    visibility: list[list[bool]] = []
    for frame_index, frame_rgb in enumerate(frames):
        frame_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
        positions, visible, state = model.track_frame(
            frame_bgr,
            query_points_xy=source_queries if frame_index == 0 else None,
            state=state,
            autocast=True,
        )
        positions[:, 0] *= analysis_width / source_width
        positions[:, 1] *= analysis_height / source_height
        tracks.append(positions.astype(np.float32).tolist())
        visibility.append(visible.astype(bool).tolist())

    torch.cuda.synchronize()
    return {
        "version": 1,
        "model": model_metadata(),
        "runtime": {
            "device": torch.cuda.get_device_name(0),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "seconds": round(time.monotonic() - started, 3),
        },
        "source": {
            "width": int(source_width),
            "height": int(source_height),
            "fps": float(metadata.get("fps", 0.0)),
            "frameCount": int(frame_count),
        },
        "analysis": {
            "width": analysis_width,
            "height": analysis_height,
        },
        "queries": normalized["queries"],
        "tracks": tracks,
        "visibility": visibility,
    }


def _committed_result(client: Any, bucket: str, key: str, request: dict[str, Any]) -> dict[str, Any] | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except client.exceptions.NoSuchKey:
        return None
    except Exception as error:
        response_code = getattr(error, "response", {}).get("Error", {}).get("Code")
        if response_code in ("NoSuchKey", "404", "NotFound"):
            return None
        raise
    payload = json.loads(response["Body"].read())
    if (
        payload.get("success") is True
        and payload.get("runId") == request["runId"]
        and payload.get("stage") == "tracking"
        and payload.get("inputHash") == request["inputHash"]
    ):
        return payload
    return None


def _run_s3_stage(request: dict[str, Any], tracking_runner: Any) -> dict[str, Any]:
    import imageio.v3 as iio

    from tracking_geometry import build_geometry, render_review_artifacts, seed_queries

    settings = s3_settings_from_env()
    normalized = validate_stage_request(request, settings.bucket)
    client = create_s3_client(settings)
    output = normalized["output"]
    result_key = f"{output['prefix']}result.json"
    committed = _committed_result(client, output["bucket"], result_key, normalized)
    if committed is not None:
        return committed

    try:
        call_id = modal.current_function_call_id() or "local-call"
    except RuntimeError:
        call_id = "local-call"
    with tempfile.TemporaryDirectory(prefix="tapnextpp-stage-", dir="/tmp") as scratch_name:
        scratch = Path(scratch_name)
        video_path = download_verified(
            client, normalized["inputs"]["normalizedVideo"], scratch / "normalized.mp4"
        )
        expected = normalized["expectedMedia"]
        metadata = iio.immeta(video_path, plugin="FFMPEG")
        frames = iio.imread(video_path, plugin="FFMPEG")
        if (
            frames.ndim != 4
            or frames.shape[0] != expected["frames"]
            or frames.shape[1] != expected["height"]
            or frames.shape[2] != expected["width"]
            or abs(float(metadata.get("fps", 0)) - 24.0) > 0.001
        ):
            raise ValueError("downloaded video does not match expectedMedia")

        parameters = normalized["parameters"]
        queries, _ = seed_queries(
            parameters["frameZeroCorners"], expected["width"], expected["height"],
            parameters["analysis"]["width"], parameters["analysis"]["height"],
            include_corner_probes=parameters["qaVersion"] == 2,
        )
        point_request = {
            "analysis_width": parameters["analysis"]["width"],
            "analysis_height": parameters["analysis"]["height"],
            "queries": queries,
            "backward_tracking": False,
        }
        tracking_result = tracking_runner(normalized, point_request)
        if (
            tracking_result["source"]["frameCount"] != expected["frames"]
            or tracking_result["source"]["width"] != expected["width"]
            or tracking_result["source"]["height"] != expected["height"]
        ):
            raise ValueError("tracker source metadata does not match expectedMedia")

        coordinates, metrics, suspects, stable = build_geometry(
            tracking_result, parameters, expected
        )
        coordinates_path = scratch / "coordinates.json"
        metrics_path = scratch / "metrics.json"
        write_json(coordinates_path, coordinates)
        write_json(metrics_path, metrics)
        evidence_enabled = parameters["evidenceMode"] == "full"
        review_video = None
        sheets: list[Path] = []
        suspect_paths: dict[int, Path] = {}
        if evidence_enabled:
            review_video, sheets, suspect_paths = render_review_artifacts(
                frames,
                stable,
                parameters["analysis"],
                suspects,
                scratch / "review",
                coordinates["frames"],
            )
        else:
            suspects = []

        attempt_dir = scratch / "attempts" / call_id / "debug"
        write_json(attempt_dir / "request.json", normalized)
        (attempt_dir / "stdout.log").write_text(
            f"validated {expected['frames']} frames\ntracked {metrics['frameCount']} frames\n"
            f"direct good {metrics['directGood']} recovered {metrics['recovered']}\n"
            f"partial {metrics.get('partialRecovered', 0)} predicted "
            f"{metrics.get('predictionOnly', 0)} bridged {metrics.get('bridged', 0)} "
            f"held {metrics.get('held', 0)}\n",
            encoding="utf-8",
        )
        (attempt_dir / "stderr.log").write_text("", encoding="utf-8")

        bucket, prefix = output["bucket"], output["prefix"]
        artifacts: dict[str, Any] = {}
        artifacts["coordinates"] = artifact_ref(client, bucket, f"{prefix}coordinates.json", coordinates_path, "application/json")
        artifacts["metrics"] = artifact_ref(client, bucket, f"{prefix}metrics.json", metrics_path, "application/json")
        if evidence_enabled:
            assert review_video is not None
            artifacts["reviewVideo"] = artifact_ref(client, bucket, f"{prefix}review/tracking-review.mp4", review_video, "video/mp4")
            artifacts["overviewSheets"] = [
                artifact_ref(client, bucket, f"{prefix}review/{path.name}", path, "image/jpeg")
                for path in sheets
            ]
            for suspect in suspects:
                path = suspect_paths[suspect["frame"]]
                suspect["artifact"] = artifact_ref(
                    client, bucket, f"{prefix}review/suspects/{path.name}", path, "image/jpeg"
                )
        debug: dict[str, Any] = {}
        for name, content_type in (
            ("request.json", "application/json"),
            ("stdout.log", "text/plain"),
            ("stderr.log", "text/plain"),
        ):
            debug[name] = artifact_ref(
                client, bucket, f"{prefix}attempts/{call_id}/debug/{name}",
                attempt_dir / name, content_type,
            )

        summary_metrics = {
            key: metrics[key]
            for key in (
                "frameCount", "directGood", "recovered", "invalidGeometry",
                "maximumCornerDisplacement", "maximumCornerDisplacementFrame",
                "maximumHomographyJump", "maximumHomographyJumpFrame",
                "maximumQuadAreaChange", "maximumQuadAreaChangeFrame",
                "maximumCentroidMovement", "maximumCentroidMovementFrame",
                "p95ReprojectionError",
            )
        }
        for key in (
            "partialRecovered", "predictionOnly", "bridged", "held",
            "fullyOffscreen", "maximumPredictionAge",
        ):
            if key in metrics:
                summary_metrics[key] = metrics[key]
        result = {
            "schemaVersion": 1, "success": True,
            "runId": normalized["runId"], "stage": "tracking",
            "inputHash": normalized["inputHash"],
            "implementation": {
                "name": "tapnextpp-modal-stage",
                "version": f"{TAPNET_COMMIT}:{CHECKPOINT_GENERATION}:qa4-offscreen-v2",
            },
            "metrics": summary_metrics, "suspects": suspects,
            "artifacts": artifacts, "debug": debug, "completedAt": utc_now(),
        }
        result_path = scratch / "result.json"
        write_json(result_path, result)
        artifact_ref(client, bucket, result_key, result_path, "application/json")
        return result


@app.function(image=control_image)
def health() -> dict[str, Any]:
    return {
        "service": APP_NAME,
        "ready": True,
        "gpu": GPU_TYPE,
        "maxContainers": MAX_CONTAINERS,
        "qaVersions": [1, 2],
        "model": model_metadata(),
    }


@app.function(
    image=tracker_image,
    secrets=[studio_s3_secret],
    gpu=GPU_TYPE,
    memory=32_768,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    startup_timeout=STARTUP_TIMEOUT,
    timeout=FUNCTION_TIMEOUT,
)
def track_stage_points_s3(
    request: dict[str, Any], point_request: dict[str, Any]
) -> dict[str, Any]:
    """Private GPU boundary; independently verifies its S3 input in `/tmp`."""
    settings = s3_settings_from_env()
    normalized = validate_stage_request(request, settings.bucket)
    client = create_s3_client(settings)
    with tempfile.TemporaryDirectory(prefix="tapnextpp-gpu-", dir="/tmp") as scratch_name:
        video_path = download_verified(
            client,
            normalized["inputs"]["normalizedVideo"],
            Path(scratch_name) / "normalized.mp4",
        )
        return _run_tracking_path(video_path, point_request)


@app.function(
    image=stage_image,
    secrets=[studio_s3_secret],
    cpu=4,
    memory=8_192,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    timeout=FUNCTION_TIMEOUT,
)
def run_tracking_stage(request: dict[str, Any]) -> dict[str, Any]:
    """Run the complete tracking stage with S3 inputs and durable outputs."""
    try:
        return _run_s3_stage(
            request,
            lambda normalized, point_request: track_stage_points_s3.remote(
                normalized, point_request
            ),
        )
    except BaseException as error:
        error_type = f"{type(error).__module__}.{type(error).__name__}"
        raise RuntimeError(f"{error_type}: {error}") from None


@app.local_entrypoint()
def main(
    stage_request_json: str = "",
    output_json: str = "",
) -> None:
    if stage_request_json:
        request_path = Path(stage_request_json)
        result = run_tracking_stage.remote(json.loads(request_path.read_text(encoding="utf-8")))
        payload = json.dumps(result, indent=2)
        if output_json:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        return
    print(json.dumps(health.remote(), indent=2))
