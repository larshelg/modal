"""SAM 3.1 presenter masks for the weather no-green experiment.

This is deliberately separate from the TAPNext++ Modal app. It performs only
presenter segmentation and video-mask encoding; TV intersection and final
Remotion compositing remain a later, local pipeline stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import modal

from sam_stage import (
    REQUIRED_SECRET_KEYS,
    artifact_ref,
    committed_result,
    create_s3_client,
    download_verified,
    s3_settings_from_env,
    utc_now,
    validate_stage_request,
    write_json,
)


APP_NAME = "sam3-modal-app"
SAM_REPOSITORY = "https://github.com/facebookresearch/sam3.git"
SAM_COMMIT = "8f0b7f4d4e7eda2ed606ebde6702c93359ad01da"
SAM_ROOT = Path("/opt/sam3")

MODEL_REPOSITORY = "facebook/sam3.1"
MODEL_REVISION = "daa63191845a41281374e725f4c9e51c7a824460"
CHECKPOINT_NAME = "sam3.1_multiplex.pt"
MODEL_VOLUME_NAME = "sam3-model-cache"
MODEL_ROOT = Path("/models/facebook-sam3.1")
CHECKPOINT_PATH = MODEL_ROOT / CHECKPOINT_NAME
HF_SECRET_NAME = "huggingface-secret"

PRIMARY_GPU_TYPE = "L40S"
FALLBACK_GPU_TYPE = "A100-80GB"
MAX_CONTAINERS = 1
SCALEDOWN_WINDOW = 5 * 60
STARTUP_TIMEOUT = 45 * 60
FUNCTION_TIMEOUT = 45 * 60
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_FRAMES = 300
DEFAULT_TEXT_PROMPT = "person"
MIN_CONTINUITY_IOU = 0.005
MIN_NONEMPTY_FRAME_FRACTION = 0.90

MASK_CODEC = "ffv1"
MASK_PIXEL_FORMAT = "gray"
MASK_FOREGROUND_VALUE = 255
MASK_BACKGROUND_VALUE = 0


sam_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("ca-certificates", "ffmpeg", "git")
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install torch==2.10.0 torchvision==0.25.0 "
        "--index-url https://download.pytorch.org/whl/cu128",
        "python -m pip install 'imageio[ffmpeg]>=2.37,<3' "
        "'huggingface-hub>=0.34,<2' 'numpy>=1.26,<3' 'pillow>=11,<13' "
        "'boto3>=1.40,<2'",
        f"git clone {SAM_REPOSITORY} {SAM_ROOT}",
        f"cd {SAM_ROOT} && git checkout {SAM_COMMIT}",
        f"python -m pip install --no-build-isolation -e {SAM_ROOT}",
        "python -m pip install 'setuptools>=75,<81'",
    )
    .env(
        {
            "PYTHONPATH": str(SAM_ROOT),
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/models/huggingface",
        }
    )
    .pip_install(
        "einops>=0.8,<1",
        "numpy>=1.26,<2",
        "opencv-python-headless>=4.10,<4.12",
        "pycocotools>=2.0.10,<3",
    )
    .run_commands(
        "python -c \"from sam3.model_builder import "
        "build_sam3_multiplex_video_predictor; print('sam3 import ok')\""
    )
    .add_local_python_source("sam_stage", "sam_review", "display_detection")
)

control_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("huggingface-hub>=0.34,<2")
    .add_local_python_source("sam_stage", "sam_review")
)
stage_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "boto3>=1.40,<2",
        "imageio[ffmpeg]>=2.37,<3",
        "numpy>=1.26,<2",
        "opencv-python-headless>=4.10,<4.12",
        "pillow>=11,<13",
    )
    .add_local_python_source("sam_stage", "sam_review", "display_detection")
)
app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)
studio_s3_secret = modal.Secret.from_name(
    "studio-s3", required_keys=list(REQUIRED_SECRET_KEYS)
)
_predictor: Any | None = None
_image_processor: Any | None = None


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    allowed = {
        "anchor_frame",
        "box_xywh",
        "text_prompt",
        "output_prob_threshold",
        "include_debug_png_zip",
        "include_preview_video",
    }
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise ValueError(f"unsupported request fields: {', '.join(unknown)}")

    anchor_frame = request.get("anchor_frame")
    if isinstance(anchor_frame, bool) or not isinstance(anchor_frame, int):
        raise ValueError("anchor_frame must be an integer")
    if not 0 <= anchor_frame < MAX_FRAMES:
        raise ValueError(f"anchor_frame must be between 0 and {MAX_FRAMES - 1}")

    raw_box = request.get("box_xywh")
    if not isinstance(raw_box, list) or len(raw_box) != 4:
        raise ValueError("box_xywh must be [x, y, width, height]")
    box = [
        _finite_number(value, f"box_xywh[{index}]")
        for index, value in enumerate(raw_box)
    ]
    x, y, width, height = box
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("box_xywh must have a non-negative origin and positive size")
    if x + width > 1 or y + height > 1:
        raise ValueError("box_xywh must stay inside normalized [0, 1] coordinates")

    text_prompt = request.get("text_prompt", DEFAULT_TEXT_PROMPT)
    if not isinstance(text_prompt, str) or not text_prompt.strip():
        raise ValueError("text_prompt must be a non-empty string")
    text_prompt = text_prompt.strip()
    if len(text_prompt) > 200:
        raise ValueError("text_prompt must not exceed 200 characters")

    threshold = _finite_number(
        request.get("output_prob_threshold", 0.5), "output_prob_threshold"
    )
    if not 0 < threshold < 1:
        raise ValueError("output_prob_threshold must be between 0 and 1")

    include_debug_png_zip = request.get("include_debug_png_zip", False)
    if not isinstance(include_debug_png_zip, bool):
        raise ValueError("include_debug_png_zip must be a boolean")
    include_preview_video = request.get("include_preview_video", True)
    if not isinstance(include_preview_video, bool):
        raise ValueError("include_preview_video must be a boolean")

    return {
        "anchor_frame": anchor_frame,
        "box_xywh": box,
        "text_prompt": text_prompt,
        "output_prob_threshold": threshold,
        "include_debug_png_zip": include_debug_png_zip,
        "include_preview_video": include_preview_video,
    }


def validate_video_bytes(video_bytes: bytes) -> None:
    if not isinstance(video_bytes, bytes):
        raise ValueError("video_bytes must be bytes")
    if not video_bytes:
        raise ValueError("video_bytes must not be empty")
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise ValueError(f"video_bytes exceeds the {MAX_VIDEO_BYTES}-byte limit")


def is_cuda_oom(error: BaseException) -> bool:
    message = f"{type(error).__module__}.{type(error).__name__}: {error}".lower()
    markers = (
        "cuda out of memory",
        "cuda error: out of memory",
        "cuda error: memory allocation",
        "outofmemoryerror",
        "cudnn_status_alloc_failed",
    )
    return any(marker in message for marker in markers)


def model_metadata() -> dict[str, Any]:
    return {
        "name": "SAM 3.1 multiplex video predictor",
        "repository": SAM_REPOSITORY,
        "commit": SAM_COMMIT,
        "checkpointRepository": MODEL_REPOSITORY,
        "checkpointRevision": MODEL_REVISION,
        "checkpoint": CHECKPOINT_NAME,
        "license": "SAM License",
        "output": "binary masks exposed by the official predictor API",
    }


def apply_init_state_compat(predictor: Any) -> bool:
    """Bridge an upstream SAM 3.1 wrapper/model signature mismatch.

    At the pinned commit Sam3BasePredictor forwards offload_state_to_cpu, while
    Sam3MultiplexTrackingWithInteractivity.init_state no longer accepts it.
    Our pipeline never requests state offload, so only a false value is safely
    discarded. Return True when the adapter was required.
    """
    import inspect

    original_init_state = predictor.model.init_state
    if "offload_state_to_cpu" in inspect.signature(original_init_state).parameters:
        return False

    def compatible_init_state(*args, offload_state_to_cpu=False, **kwargs):
        if offload_state_to_cpu:
            raise ValueError(
                "the pinned SAM 3.1 multiplex model does not support "
                "offload_state_to_cpu"
            )
        return original_init_state(*args, **kwargs)

    predictor.model.init_state = compatible_init_state
    return True


def _probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("input must contain exactly one video stream")
    stream = streams[0]
    fps_ratio = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    fps = Fraction(fps_ratio)
    if fps <= 0:
        raise ValueError("input video has an invalid frame rate")
    frame_count = int(stream.get("nb_frames") or 0)
    if not 2 <= frame_count <= MAX_FRAMES:
        raise ValueError(f"decoded frame count must be between 2 and {MAX_FRAMES}")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fpsNumerator": fps.numerator,
        "fpsDenominator": fps.denominator,
        "fps": float(fps),
        "fpsRatio": f"{fps.numerator}/{fps.denominator}",
        "frameCount": frame_count,
        "duration": float(stream.get("duration") or frame_count / float(fps)),
    }


def _select_anchor_mask(
    outputs: dict[str, Any], box_xywh: list[float]
) -> tuple[int, Any]:
    import numpy as np

    object_ids = np.asarray(outputs["out_obj_ids"]).reshape(-1)
    masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
    if object_ids.size == 0 or masks.shape[0] == 0:
        raise RuntimeError("SAM 3.1 returned no object for the anchor box")
    if object_ids.size != masks.shape[0]:
        raise RuntimeError("SAM 3.1 returned mismatched object IDs and masks")

    height, width = masks.shape[-2:]
    x, y, box_width, box_height = box_xywh
    x0 = max(0, min(width - 1, int(math.floor(x * width))))
    y0 = max(0, min(height - 1, int(math.floor(y * height))))
    x1 = max(x0 + 1, min(width, int(math.ceil((x + box_width) * width))))
    y1 = max(y0 + 1, min(height, int(math.ceil((y + box_height) * height))))
    scores = masks[:, y0:y1, x0:x1].sum(axis=(1, 2))
    selected_index = int(np.argmax(scores))
    return int(object_ids[selected_index]), masks[selected_index]


def select_continuous_mask(
    outputs: dict[str, Any], reference_mask: Any
) -> tuple[int | None, Any | None, float]:
    import numpy as np

    object_ids = np.asarray(outputs["out_obj_ids"]).reshape(-1)
    masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
    reference = np.asarray(reference_mask, dtype=bool)
    if object_ids.size == 0 or masks.shape[0] == 0:
        return None, None, 0.0
    if object_ids.size != masks.shape[0]:
        raise RuntimeError("SAM 3.1 returned mismatched object IDs and masks")
    if masks.shape[-2:] != reference.shape:
        raise RuntimeError(
            f"SAM mask shape {masks.shape[-2:]} does not match reference "
            f"{reference.shape}"
        )
    intersections = np.logical_and(masks, reference[None, :, :]).sum(axis=(1, 2))
    unions = np.logical_or(masks, reference[None, :, :]).sum(axis=(1, 2))
    ious = intersections / np.maximum(unions, 1)
    selected_index = int(np.argmax(ious))
    selected_iou = float(ious[selected_index])
    if selected_iou < MIN_CONTINUITY_IOU:
        return None, None, selected_iou
    return int(object_ids[selected_index]), masks[selected_index], selected_iou


def _run_ffmpeg_writer(command: list[str], frames: Iterable[Any]) -> None:
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed with status {return_code}: "
            f"{stderr.decode('utf-8', errors='replace')[-4000:]}"
        )


def _encode_mask_video(masks: list[Any], source: dict[str, Any], output: Path) -> None:
    import numpy as np

    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        MASK_PIXEL_FORMAT,
        "-video_size",
        f"{source['width']}x{source['height']}",
        "-framerate",
        source["fpsRatio"],
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        MASK_CODEC,
        "-level",
        "3",
        "-g",
        "1",
        "-pix_fmt",
        MASK_PIXEL_FORMAT,
        str(output),
    ]
    frames = (
        np.asarray(mask, dtype=np.uint8) * MASK_FOREGROUND_VALUE for mask in masks
    )
    _run_ffmpeg_writer(command, frames)


def _encode_preview_video(
    video_path: Path,
    masks: list[Any],
    source: dict[str, Any],
    output: Path,
) -> None:
    import imageio.v3 as iio
    import numpy as np

    def preview_frames():
        decoded_count = 0
        tint = np.asarray([255.0, 36.0, 190.0], dtype=np.float32)
        for frame, mask in zip(
            iio.imiter(video_path, plugin="FFMPEG"), masks, strict=True
        ):
            rgb = np.asarray(frame[:, :, :3], dtype=np.float32)
            alpha = np.asarray(mask, dtype=np.float32)[:, :, None] * 0.58
            yield np.clip(rgb * (1 - alpha) + tint * alpha, 0, 255).astype(
                np.uint8
            )
            decoded_count += 1
        if decoded_count != source["frameCount"]:
            raise RuntimeError(
                f"preview decoded {decoded_count} frames; expected {source['frameCount']}"
            )

    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{source['width']}x{source['height']}",
        "-framerate",
        source["fpsRatio"],
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    _run_ffmpeg_writer(command, preview_frames())


def _encode_debug_png_zip(masks: list[Any], output: Path) -> None:
    import numpy as np
    from PIL import Image

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for frame_index, mask in enumerate(masks):
            with tempfile.NamedTemporaryFile(suffix=".png") as png_file:
                Image.fromarray(
                    np.asarray(mask, dtype=np.uint8) * MASK_FOREGROUND_VALUE,
                    mode="L",
                ).save(png_file.name, format="PNG", optimize=True)
                archive.write(png_file.name, f"masks/{frame_index:06d}.png")


def _mask_stats(mask: Any) -> dict[str, Any]:
    import numpy as np

    y_coords, x_coords = np.nonzero(mask)
    if len(x_coords) == 0:
        return {"foregroundPixels": 0, "bboxXywh": None}
    x0 = int(x_coords.min())
    y0 = int(y_coords.min())
    x1 = int(x_coords.max()) + 1
    y1 = int(y_coords.max()) + 1
    return {
        "foregroundPixels": int(len(x_coords)),
        "bboxXywh": [x0, y0, x1 - x0, y1 - y0],
    }


def _load_predictor():
    global _predictor
    if _predictor is None:
        if not CHECKPOINT_PATH.is_file():
            raise RuntimeError(
                f"missing {CHECKPOINT_PATH}; run `python3 -m modal run app.py "
                "--download-only` first"
            )
        import torch
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        _predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(CHECKPOINT_PATH),
            use_fa3=False,
            compile=False,
            warm_up=False,
            async_loading_frames=True,
        )
        _predictor._modal_init_state_compat = apply_init_state_compat(_predictor)
    return _predictor


def _load_image_processor():
    global _image_processor
    if _image_processor is None:
        if not CHECKPOINT_PATH.is_file():
            raise RuntimeError(
                f"missing {CHECKPOINT_PATH}; run `python3 -m modal run app.py "
                "--download-only` first"
            )
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model = build_sam3_image_model(
            checkpoint_path=str(CHECKPOINT_PATH),
            load_from_HF=False,
            device="cuda",
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )
        _image_processor = Sam3Processor(model)
    return _image_processor


def _run_segmentation(video_bytes: bytes, request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    normalized = validate_request(request)
    validate_video_bytes(video_bytes)
    started = time.monotonic()
    predictor = _load_predictor()
    session_id: str | None = None

    with tempfile.TemporaryDirectory(prefix="sam3-legacy-", dir="/tmp") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        video_path = temp_dir / "input.mp4"
        video_path.write_bytes(video_bytes)
        source = _probe_video(video_path)
        anchor_frame = normalized["anchor_frame"]
        if anchor_frame >= source["frameCount"]:
            raise ValueError(
                f"anchor_frame {anchor_frame} is outside the "
                f"{source['frameCount']}-frame video"
            )

        try:
            session = predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(video_path),
                    "offload_video_to_cpu": True,
                    "offload_state_to_cpu": False,
                }
            )
            session_id = session["session_id"]
            prompt_response = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": anchor_frame,
                    # SAM 3.1 multiplex treats boxes as visual concept prompts,
                    # which matched this presenter in only a few nearby frames.
                    # Text is the official dense video concept prompt; the box
                    # below is retained only to select the intended person from
                    # the anchor-frame detections.
                    "text": normalized["text_prompt"],
                    "output_prob_thresh": normalized["output_prob_threshold"],
                    "rel_coordinates": True,
                }
            )
            anchor_outputs = prompt_response["outputs"]
            anchor_object_id, anchor_mask = _select_anchor_mask(
                anchor_outputs, normalized["box_xywh"]
            )
            masks: list[Any | None] = [None] * source["frameCount"]
            selected_object_ids: list[int | None] = [None] * source["frameCount"]
            selection_ious: list[float | None] = [None] * source["frameCount"]
            masks[anchor_frame] = anchor_mask
            selected_object_ids[anchor_frame] = anchor_object_id
            selection_ious[anchor_frame] = 1.0

            for direction in ("forward", "backward"):
                reference_mask = anchor_mask
                for response in predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": direction,
                        "start_frame_index": anchor_frame,
                        "output_prob_thresh": normalized[
                            "output_prob_threshold"
                        ],
                    }
                ):
                    frame_index = int(response["frame_index"])
                    if frame_index == anchor_frame:
                        continue
                    object_id, mask, continuity_iou = select_continuous_mask(
                        response["outputs"], reference_mask
                    )
                    selection_ious[frame_index] = continuity_iou
                    if mask is None:
                        continue
                    masks[frame_index] = mask
                    selected_object_ids[frame_index] = object_id
                    reference_mask = mask

            missing_frames = [
                frame_index
                for frame_index, mask in enumerate(masks)
                if mask is None or not mask.any()
            ]
            nonempty_fraction = 1 - len(missing_frames) / source["frameCount"]
            if nonempty_fraction < MIN_NONEMPTY_FRAME_FRACTION:
                raise RuntimeError(
                    "presenter continuity selection produced masks for only "
                    f"{nonempty_fraction:.1%} of frames"
                )
            masks = [
                np.zeros((source["height"], source["width"]), dtype=bool)
                if mask is None
                else mask
                for mask in masks
            ]

            mask_path = temp_dir / "presenter-mask.mkv"
            preview_path = temp_dir / "presenter-mask-preview.mp4"
            _encode_mask_video(masks, source, mask_path)
            if normalized["include_preview_video"]:
                _encode_preview_video(video_path, masks, source, preview_path)

            png_zip_bytes = None
            if normalized["include_debug_png_zip"]:
                png_zip_path = temp_dir / "presenter-mask-pngs.zip"
                _encode_debug_png_zip(masks, png_zip_path)
                png_zip_bytes = png_zip_path.read_bytes()

            torch.cuda.synchronize()
            per_frame = [_mask_stats(mask) for mask in masks]
            metadata = {
                "version": 1,
                "purpose": "presenter mask only; TV intersection is not applied",
                "model": model_metadata(),
                "compatibility": {
                    "discardFalseOffloadStateToCpu": bool(
                        getattr(predictor, "_modal_init_state_compat", False)
                    )
                },
                "runtime": {
                    "device": torch.cuda.get_device_name(0),
                    "torch": str(torch.__version__),
                    "cuda": str(torch.version.cuda),
                    "seconds": round(time.monotonic() - started, 3),
                },
                "source": source,
                "request": normalized,
                "tracking": {
                    "anchorObjectId": anchor_object_id,
                    "propagationDirection": "both",
                    "promptMode": "text concept; box selects anchor detection only",
                    "selection": "maximum adjacent-frame mask IoU",
                    "minimumContinuityIou": MIN_CONTINUITY_IOU,
                    "selectedObjectIds": selected_object_ids,
                    "selectionIous": selection_ious,
                    "missingFrames": missing_frames,
                    "nonemptyFrameFraction": nonempty_fraction,
                    "perFrame": per_frame,
                },
                "maskVideo": {
                    "container": "matroska",
                    "codec": MASK_CODEC,
                    "pixelFormat": MASK_PIXEL_FORMAT,
                    "bitDepth": 8,
                    "backgroundValue": MASK_BACKGROUND_VALUE,
                    "foregroundValue": MASK_FOREGROUND_VALUE,
                    "lossless": True,
                    "sourceMasksAreBinary": True,
                    "feathered": False,
                    "dilated": False,
                    "sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                },
                **(
                    {
                        "previewVideo": {
                            "container": "mp4",
                            "codec": "h264",
                            "pixelFormat": "yuv420p",
                        }
                    }
                    if normalized["include_preview_video"] else {}
                ),
            }
            return {
                "mask_video_bytes": mask_path.read_bytes(),
                "preview_video_bytes": (
                    preview_path.read_bytes()
                    if normalized["include_preview_video"] else None
                ),
                "debug_png_zip_bytes": png_zip_bytes,
                "metadata": metadata,
            }
        finally:
            if session_id is not None:
                predictor.handle_request(
                    {
                        "type": "close_session",
                        "session_id": session_id,
                        "run_gc_collect": True,
                    }
                )


@app.function(
    image=control_image,
    secrets=[hf_secret],
    volumes={"/models": model_volume},
    timeout=30 * 60,
)
def download_model() -> dict[str, Any]:
    from huggingface_hub import hf_hub_download, model_info

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(f"Modal secret {HF_SECRET_NAME!r} must define HF_TOKEN")
    info = model_info(MODEL_REPOSITORY, revision=MODEL_REVISION, token=token)
    if info.sha != MODEL_REVISION:
        raise RuntimeError(
            f"resolved checkpoint revision {info.sha} does not match {MODEL_REVISION}"
        )
    downloaded_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPOSITORY,
            filename=CHECKPOINT_NAME,
            revision=MODEL_REVISION,
            token=token,
            local_dir=MODEL_ROOT,
        )
    )
    if downloaded_path.name != CHECKPOINT_NAME or not CHECKPOINT_PATH.is_file():
        raise RuntimeError(f"checkpoint downloaded to unexpected path {downloaded_path}")
    size = CHECKPOINT_PATH.stat().st_size
    model_volume.commit()
    return {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "checkpoint": str(CHECKPOINT_PATH),
        "size": size,
    }


@app.function(image=control_image, volumes={"/models": model_volume})
def health() -> dict[str, Any]:
    return {
        "service": APP_NAME,
        "ready": True,
        "primaryGpu": PRIMARY_GPU_TYPE,
        "fallbackGpu": FALLBACK_GPU_TYPE,
        "maxContainers": MAX_CONTAINERS,
        "checkpointCached": CHECKPOINT_PATH.is_file(),
        "capabilities": ["presenter-video-mask", "display-image-detection"],
        "model": model_metadata(),
    }


def _as_numpy(value: Any):
    if hasattr(value, "detach"):
        value = value.detach()
    if str(getattr(value, "dtype", "")) == "torch.bfloat16":
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    import numpy as np

    return np.asarray(value)


def _run_display_detection_gpu(
    request: dict[str, Any], gpu_attempts: list[str]
) -> dict[str, Any]:
    import torch
    from PIL import Image

    from display_detection import (
        ALLOWED_IMAGE_TYPES,
        DISPLAY_STAGE,
        decode_image,
        public_candidate,
        render_display_overlay,
        select_display_candidate,
        validate_display_request,
    )

    settings = s3_settings_from_env()
    normalized = validate_display_request(request, settings.bucket)
    client = create_s3_client(settings)
    bucket = normalized["output"]["bucket"]
    prefix = normalized["output"]["prefix"]
    result_key = f"{prefix}result.json"
    existing = committed_result(client, bucket, result_key, normalized)
    if existing is not None:
        return existing

    with tempfile.TemporaryDirectory(
        prefix="sam3-display-", dir="/tmp"
    ) as scratch_name:
        scratch = Path(scratch_name)
        image_ref = normalized["inputs"]["image"]
        image_path = download_verified(
            client,
            image_ref,
            scratch / f"input{ALLOWED_IMAGE_TYPES[image_ref['contentType']]}",
        )
        image_rgb = decode_image(image_path)
        height, width = image_rgb.shape[:2]
        processor = _load_image_processor()
        started = time.monotonic()
        prompt_outputs: list[tuple[str, Any, Any]] = []
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            state = processor.set_image(Image.fromarray(image_rgb, mode="RGB"))
            for prompt in normalized["parameters"]["textPrompts"]:
                output = processor.set_text_prompt(state=state, prompt=prompt)
                prompt_outputs.append(
                    (
                        prompt,
                        _as_numpy(output["masks"]),
                        _as_numpy(output["scores"]),
                    )
                )
        torch.cuda.synchronize()
        parameters = normalized["parameters"]
        candidate = select_display_candidate(
            prompt_outputs,
            score_threshold=parameters["scoreThreshold"],
            min_area_ratio=parameters["minAreaRatio"],
            max_area_ratio=parameters["maxAreaRatio"],
        )
        overlay_path = scratch / "overlay.png"
        render_display_overlay(image_rgb, candidate, overlay_path)
        overlay = artifact_ref(
            client, bucket, f"{prefix}overlay.png", overlay_path, "image/png"
        )
        corners = [
            [round(float(x), 3), round(float(y), 3)]
            for x, y in candidate["corners"]
        ]
        selection = public_candidate(candidate)
        for key in ("samScore", "rectangularity", "confidence", "areaRatio"):
            selection[key] = round(float(selection[key]), 6)
        result = {
            "schemaVersion": 1,
            "success": True,
            "runId": normalized["runId"],
            "stage": DISPLAY_STAGE,
            "inputHash": normalized["inputHash"],
            "frameZeroCorners": corners,
            "cornerOrder": "top-left, top-right, bottom-right, bottom-left",
            "source": {"width": width, "height": height},
            "selection": selection,
            "runtime": {
                "device": torch.cuda.get_device_name(0),
                "seconds": round(time.monotonic() - started, 3),
                "gpuAttempts": gpu_attempts,
            },
            "artifacts": {"overlay": overlay},
            "overlayS3Uri": f"s3://{overlay['bucket']}/{overlay['key']}",
            "implementation": {
                "name": "sam31-display-detection",
                "version": f"{SAM_COMMIT}:{MODEL_REVISION}:geometry-v1",
            },
            "completedAt": utc_now(),
        }
        result_path = scratch / "result.json"
        write_json(result_path, result)
        artifact_ref(
            client,
            bucket,
            result_key,
            result_path,
            "application/json",
        )
        return result


@app.function(
    image=sam_image,
    secrets=[hf_secret, studio_s3_secret],
    gpu=PRIMARY_GPU_TYPE,
    memory=65_536,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    startup_timeout=STARTUP_TIMEOUT,
    timeout=FUNCTION_TIMEOUT,
    volumes={"/models": model_volume},
)
def detect_display_gpu(
    request: dict[str, Any], gpu_attempts: list[str]
) -> dict[str, Any]:
    """Private GPU boundary for display segmentation and geometry."""
    try:
        return _run_display_detection_gpu(request, gpu_attempts)
    except BaseException as error:
        error_type = f"{type(error).__module__}.{type(error).__name__}"
        raise RuntimeError(f"{error_type}: {error}") from None


@app.function(
    image=stage_image,
    secrets=[studio_s3_secret],
    cpu=2,
    memory=4_096,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    timeout=FUNCTION_TIMEOUT,
)
def detect_display(request: dict[str, Any]) -> dict[str, Any]:
    """Validate and reuse before dispatching display inference to a GPU."""
    from display_detection import (
        ALLOWED_IMAGE_TYPES,
        decode_image,
        validate_display_request,
    )

    settings = s3_settings_from_env()
    normalized = validate_display_request(request, settings.bucket)
    client = create_s3_client(settings)
    result_key = f"{normalized['output']['prefix']}result.json"
    existing = committed_result(
        client, normalized["output"]["bucket"], result_key, normalized
    )
    if existing is not None:
        return existing
    with tempfile.TemporaryDirectory(
        prefix="sam3-display-preflight-", dir="/tmp"
    ) as scratch_name:
        image_ref = normalized["inputs"]["image"]
        image_path = download_verified(
            client,
            image_ref,
            Path(scratch_name)
            / f"input{ALLOWED_IMAGE_TYPES[image_ref['contentType']]}",
        )
        decode_image(image_path)
    attempts = [PRIMARY_GPU_TYPE]
    try:
        return detect_display_gpu.remote(normalized, attempts)
    except BaseException as error:
        if not is_cuda_oom(error):
            raise
        attempts.append(FALLBACK_GPU_TYPE)
        return detect_display_gpu.with_options(gpu=FALLBACK_GPU_TYPE).remote(
            normalized, attempts
        )


@app.function(
    image=sam_image,
    gpu=PRIMARY_GPU_TYPE,
    memory=65_536,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    startup_timeout=STARTUP_TIMEOUT,
    timeout=FUNCTION_TIMEOUT,
    volumes={"/models": model_volume},
)
def segment_presenter(
    video_bytes: bytes, request: dict[str, Any]
) -> dict[str, Any]:
    try:
        return _run_segmentation(video_bytes, request)
    except BaseException as error:
        error_type = f"{type(error).__module__}.{type(error).__name__}"
        raise RuntimeError(f"{error_type}: {error}") from None


def _validate_expected_media(video_path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    source = _probe_video(video_path)
    if (
        source["frameCount"] != expected["frames"]
        or source["fpsRatio"] != expected["fps"]
        or source["width"] != expected["width"]
        or source["height"] != expected["height"]
    ):
        raise ValueError("downloaded video does not match expectedMedia")
    return source


def _legacy_request(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "anchor_frame": parameters["anchorFrame"],
        "box_xywh": parameters["boxXywh"],
        "text_prompt": parameters["textPrompt"],
        "output_prob_threshold": parameters["outputProbabilityThreshold"],
        "include_debug_png_zip": False,
        "include_preview_video": parameters.get("evidenceMode", "full") == "full",
    }


def _run_s3_sam_gpu(request: dict[str, Any], gpu_attempts: list[str]) -> dict[str, Any]:
    import imageio.v3 as iio
    import numpy as np

    from sam_review import build_metrics, render_review_artifacts, verify_mask_video

    settings = s3_settings_from_env()
    normalized = validate_stage_request(request, settings.bucket)
    client = create_s3_client(settings)
    output = normalized["output"]
    try:
        call_id = modal.current_function_call_id() or "local-call"
    except RuntimeError:
        call_id = "local-call"

    with tempfile.TemporaryDirectory(prefix="sam3-stage-gpu-", dir="/tmp") as scratch_name:
        scratch = Path(scratch_name)
        video_path = download_verified(
            client, normalized["inputs"]["normalizedVideo"], scratch / "normalized.mp4"
        )
        source = _validate_expected_media(video_path, normalized["expectedMedia"])
        segmentation = _run_segmentation(
            video_path.read_bytes(), _legacy_request(normalized["parameters"])
        )
        metadata = segmentation["metadata"]
        metadata["dispatch"] = {"gpuAttempts": gpu_attempts}
        missing = metadata["tracking"].get("missingFrames", [])
        if missing:
            raise ValueError(f"SAM stage produced missing masks for frames {missing}")
        if metadata["source"]["frameCount"] != normalized["expectedMedia"]["frames"]:
            raise ValueError("SAM metadata does not cover expectedMedia.frames")

        mask_path = scratch / "mask.mkv"
        mask_path.write_bytes(segmentation["mask_video_bytes"])
        evidence_enabled = normalized["parameters"]["evidenceMode"] == "full"
        review_video = scratch / "review" / "sam-review.mp4"
        if evidence_enabled:
            review_video.parent.mkdir(parents=True, exist_ok=True)
            preview_bytes = segmentation["preview_video_bytes"]
            if not isinstance(preview_bytes, bytes):
                raise RuntimeError("SAM review preview is missing")
            review_video.write_bytes(preview_bytes)
        verify_mask_video(mask_path, source)
        decoded_masks = iio.imread(mask_path, plugin="FFMPEG")
        if decoded_masks.ndim == 4:
            decoded_masks = decoded_masks[:, :, :, 0]
        masks = [np.asarray(frame, dtype=np.uint8) > 0 for frame in decoded_masks]
        metrics, suspects = build_metrics(masks, metadata)
        sheets: list[Path] = []
        suspect_paths: dict[int, Path] = {}
        if evidence_enabled:
            frames = iio.imread(video_path, plugin="FFMPEG")[:, :, :, :3]
            sheets, suspect_paths = render_review_artifacts(
                frames, masks, metrics, suspects, scratch / "review"
            )
        else:
            suspects = []
        metrics_path = scratch / "metrics.json"
        write_json(metrics_path, metrics)

        attempt_dir = scratch / "attempts" / call_id / "debug"
        write_json(attempt_dir / "request.json", normalized)
        (attempt_dir / "stdout.log").write_text(
            f"validated {normalized['expectedMedia']['frames']} source frames\n"
            f"segmented {metrics['nonemptyFrames']} non-empty masks\n"
            f"minimum adjacent IoU {metrics['minimumAdjacentIou']:.6f}\n",
            encoding="utf-8",
        )
        (attempt_dir / "stderr.log").write_text("", encoding="utf-8")

        bucket, prefix = output["bucket"], output["prefix"]
        artifacts: dict[str, Any] = {}
        artifacts["mask"] = artifact_ref(
            client, bucket, f"{prefix}mask.mkv", mask_path, "video/x-matroska"
        )
        artifacts["metrics"] = artifact_ref(
            client, bucket, f"{prefix}metrics.json", metrics_path, "application/json"
        )
        if evidence_enabled:
            artifacts["reviewVideo"] = artifact_ref(
                client, bucket, f"{prefix}review/sam-review.mp4", review_video, "video/mp4"
            )
            artifacts["overviewSheets"] = [
                artifact_ref(client, bucket, f"{prefix}review/{path.name}", path, "image/jpeg")
                for path in sheets
            ]
            for suspect in suspects:
                path = suspect_paths[int(suspect["frame"])]
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
                "frameCount", "nonemptyFrames", "missingFrames",
                "minimumAdjacentIou", "minimumAdjacentIouFrame",
                "maximumCentroidMovement", "maximumCentroidMovementFrame",
                "maximumBboxChange", "maximumBboxChangeFrame",
                "maximumAreaChangeRatio", "maximumAreaChangeFrame",
                "minimumArea", "minimumAreaFrame", "maximumArea", "maximumAreaFrame",
                "gpuAttempts", "compatibilityAdapter",
            )
        }
        result = {
            "schemaVersion": 1, "success": True,
            "runId": normalized["runId"], "stage": "sam",
            "inputHash": normalized["inputHash"],
            "implementation": {
                "name": "sam31-modal-stage",
                "version": f"{SAM_COMMIT}:{MODEL_REVISION}:qa3-dynamic-media",
            },
            "metrics": summary_metrics, "suspects": suspects,
            "artifacts": artifacts, "debug": debug, "completedAt": utc_now(),
        }
        result_path = scratch / "result.json"
        write_json(result_path, result)
        artifact_ref(client, bucket, f"{prefix}result.json", result_path, "application/json")
        return result


@app.function(
    image=sam_image,
    secrets=[hf_secret, studio_s3_secret],
    gpu=PRIMARY_GPU_TYPE,
    memory=65_536,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    startup_timeout=STARTUP_TIMEOUT,
    timeout=FUNCTION_TIMEOUT,
    volumes={"/models": model_volume},
)
def run_sam_stage_gpu(request: dict[str, Any], gpu_attempts: list[str]) -> dict[str, Any]:
    """Private GPU boundary for the complete strict SAM stage."""
    try:
        return _run_s3_sam_gpu(request, gpu_attempts)
    except BaseException as error:
        error_type = f"{type(error).__module__}.{type(error).__name__}"
        raise RuntimeError(f"{error_type}: {error}") from None


@app.function(
    image=stage_image,
    secrets=[studio_s3_secret],
    cpu=2,
    memory=4_096,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    timeout=FUNCTION_TIMEOUT,
)
def run_sam_stage(request: dict[str, Any]) -> dict[str, Any]:
    """Validate and reuse before dispatching the strict S3-native GPU stage."""
    settings = s3_settings_from_env()
    normalized = validate_stage_request(request, settings.bucket)
    client = create_s3_client(settings)
    result_key = f"{normalized['output']['prefix']}result.json"
    existing = committed_result(
        client, normalized["output"]["bucket"], result_key, normalized
    )
    if existing is not None:
        return existing
    with tempfile.TemporaryDirectory(prefix="sam3-stage-preflight-", dir="/tmp") as scratch_name:
        video_path = download_verified(
            client, normalized["inputs"]["normalizedVideo"],
            Path(scratch_name) / "normalized.mp4",
        )
        _validate_expected_media(video_path, normalized["expectedMedia"])
    attempts = [PRIMARY_GPU_TYPE]
    try:
        return run_sam_stage_gpu.remote(normalized, attempts)
    except BaseException as error:
        if not is_cuda_oom(error):
            raise
        attempts.append(FALLBACK_GPU_TYPE)
        return run_sam_stage_gpu.with_options(gpu=FALLBACK_GPU_TYPE).remote(
            normalized, attempts
        )


def _write_result(
    result: dict[str, Any],
    mask_output: str,
    preview_output: str,
    metadata_output: str,
    debug_png_zip_output: str,
) -> None:
    outputs = {
        "mask output": mask_output,
        "preview output": preview_output,
        "metadata output": metadata_output,
    }
    for label, value in outputs.items():
        if not value:
            raise ValueError(f"--{label.replace(' ', '-')} is required")

    mask_path = Path(mask_output)
    preview_path = Path(preview_output)
    metadata_path = Path(metadata_output)
    for path in (mask_path, preview_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.write_bytes(result["mask_video_bytes"])
    preview_path.write_bytes(result["preview_video_bytes"])
    metadata_path.write_text(
        json.dumps(result["metadata"], indent=2) + "\n", encoding="utf-8"
    )

    debug_bytes = result.get("debug_png_zip_bytes")
    if debug_png_zip_output:
        if debug_bytes is None:
            raise ValueError(
                "request must set include_debug_png_zip=true when "
                "--debug-png-zip-output is supplied"
            )
        debug_path = Path(debug_png_zip_output)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_bytes(debug_bytes)


@app.local_entrypoint()
def main(
    video_path: str = "",
    request_json: str = "",
    mask_output: str = "",
    preview_output: str = "",
    metadata_output: str = "",
    debug_png_zip_output: str = "",
    stage_request_json: str = "",
    display_request_json: str = "",
    output_json: str = "",
    download_only: bool = False,
) -> None:
    if display_request_json:
        request_path = Path(display_request_json)
        result = detect_display.remote(
            json.loads(request_path.read_text(encoding="utf-8"))
        )
        payload = json.dumps(result, indent=2)
        if output_json:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        return
    if stage_request_json:
        request_path = Path(stage_request_json)
        result = run_sam_stage.remote(
            json.loads(request_path.read_text(encoding="utf-8"))
        )
        payload = json.dumps(result, indent=2)
        if output_json:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        return
    if download_only:
        print(json.dumps(download_model.remote(), indent=2))
        return
    if not video_path:
        print(json.dumps(health.remote(), indent=2))
        return
    if not request_json:
        raise ValueError("--request-json is required with --video-path")

    video = Path(video_path)
    request_path = Path(request_json)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    video_bytes = video.read_bytes()
    gpu_attempts = [PRIMARY_GPU_TYPE]
    try:
        result = segment_presenter.remote(video_bytes, request)
    except BaseException as error:
        if not is_cuda_oom(error):
            raise
        gpu_attempts.append(FALLBACK_GPU_TYPE)
        result = segment_presenter.with_options(gpu=FALLBACK_GPU_TYPE).remote(
            video_bytes, request
        )
    result["metadata"]["dispatch"] = {"gpuAttempts": gpu_attempts}
    _write_result(
        result,
        mask_output,
        preview_output,
        metadata_output,
        debug_png_zip_output,
    )
    print(json.dumps(result["metadata"], indent=2))
