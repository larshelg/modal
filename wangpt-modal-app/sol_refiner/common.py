"""Sol-only request contract. No GPU, WanGP, or Modal imports."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

APP_NAME = "sol-refiner"
JOB_DICT_NAME = "sol-refiner-jobs"
MODEL_VOLUME_NAME = "sol-refiner-models"
MODEL_ID = "Efficient-Large-Model/SoL-Refiner-LTX-2.5-for-MiniMax-H3"
MAX_INPUT_BYTES = 1024 * 1024 * 1024
MAX_FRAMES = 241
MAX_PIXELS = 1920 * 1088
TERMINAL_STATES = {"succeeded", "failed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_url(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("input_url must be an HTTPS URL or s3://BUCKET/KEY")
    value = value.strip()
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"https", "s3"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError("input_url must be HTTPS or S3 without embedded credentials or fragments")
    if parsed.scheme == "s3" and (parsed.query or parsed.port or not parsed.path.strip("/")):
        raise ValueError("S3 input must name an object without a query or port")
    if parsed.scheme == "https" and parsed.port not in (None, 443):
        raise ValueError("HTTPS input must use port 443")
    return value


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    allowed = {
        "input_url", "prompt", "output_width", "output_height", "seed",
        "decoder_seed", "generation_id", "asset_id", "frame_policy",
    }
    if set(request) - allowed:
        raise ValueError(f"unknown request fields: {', '.join(sorted(set(request) - allowed))}")
    result = dict(request)
    result["input_url"] = validate_url(result.get("input_url"))
    prompt = result.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16000:
        raise ValueError("prompt must be a non-empty string of at most 16000 characters")
    result["prompt"] = prompt.strip()
    if result.setdefault("frame_policy", "strict") not in {"strict", "pad"}:
        raise ValueError("frame_policy must be strict or pad")
    for key in ("output_width", "output_height"):
        value = result.get(key)
        if type(value) is not int or value < 64 or value > 1920 or value % 2:
            raise ValueError(f"{key} must be an even integer between 64 and 1920")
    canvas_pixels = math.ceil(result["output_width"] / 64) * 64 * math.ceil(result["output_height"] / 64) * 64
    if canvas_pixels > MAX_PIXELS:
        raise ValueError("output exceeds the initial 1920x1088 internal canvas limit")
    for key in ("seed", "decoder_seed"):
        value = result.setdefault(key, 0)
        if type(value) is not int or not 0 <= value < 2**63:
            raise ValueError(f"{key} must be an integer between 0 and 2**63 - 1")
    for key in ("generation_id", "asset_id"):
        if key in result and (not isinstance(result[key], str) or not 1 <= len(result[key]) <= 256):
            raise ValueError(f"{key} must be a string of 1 to 256 characters")
    return result


def validate_frame_count(frames: int, policy: str = "strict") -> int:
    if policy not in {"strict", "pad"}:
        raise ValueError("frame_policy must be strict or pad")
    if frames < 9 or frames > MAX_FRAMES:
        raise ValueError(f"input must contain between 9 and {MAX_FRAMES} frames")
    if (frames - 1) % 8:
        if policy == "pad":
            return 1 + 8 * math.ceil((frames - 1) / 8)
        truncated = 1 + 8 * ((frames - 1) // 8)
        raise ValueError(
            f"input has {frames} frames; Sol requires 8k+1 frames and would truncate "
            f"to {truncated}. Supply a compatible clip; automatic truncation is disabled."
        )
    return frames
