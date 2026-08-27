"""Shared S3 and StageRequest contract for the SAM 3.1 Modal stage."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EXPECTED_STAGE = "sam"
EXPECTED_FPS = "24/1"
RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_SECRET_KEYS = (
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY",
    "S3_ENDPOINT",
    "S3_BUCKET",
    "S3_REGION",
)


@dataclass(frozen=True)
class S3Settings:
    access_key_id: str
    secret_access_key: str
    endpoint: str
    bucket: str
    region: str


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def validate_key(key: Any, prefix: str, field: str = "key") -> str:
    if not isinstance(key, str) or not key:
        raise ValueError(f"{field} must be a non-empty string")
    if key.startswith("/") or "\\" in key or any(ord(char) < 32 for char in key):
        raise ValueError(f"{field} is unsafe")
    parts = key.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{field} is unsafe")
    if not key.startswith(prefix):
        raise ValueError(f"{field} must remain inside {prefix}")
    return key


def validate_stage_request(request: Any, configured_bucket: str) -> dict[str, Any]:
    request = _object(request, "request")
    allowed = {
        "schemaVersion", "runId", "stage", "inputHash", "inputs",
        "parameters", "expectedMedia", "output",
    }
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise ValueError(f"unsupported request fields: {', '.join(unknown)}")
    if request.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("schemaVersion must be 1")
    run_id = request.get("runId")
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("runId is invalid")
    if request.get("stage") != EXPECTED_STAGE:
        raise ValueError("stage must be sam")
    input_hash = request.get("inputHash")
    if not isinstance(input_hash, str) or not SHA256_PATTERN.fullmatch(input_hash):
        raise ValueError("inputHash must be a lowercase SHA-256")

    experiment_prefix = f"studio-experiments/{run_id}/"
    inputs = _object(request.get("inputs"), "inputs")
    if set(inputs) != {"normalizedVideo"}:
        raise ValueError("inputs must contain only normalizedVideo")
    video = _object(inputs["normalizedVideo"], "inputs.normalizedVideo")
    required_ref = {"storage", "bucket", "key", "sha256", "sizeBytes", "contentType"}
    if set(video) != required_ref:
        raise ValueError("normalizedVideo ArtifactRef fields are invalid")
    if video.get("storage") != "s3":
        raise ValueError("normalizedVideo.storage must be s3")
    if video.get("bucket") != configured_bucket:
        raise ValueError("normalizedVideo.bucket does not match configured S3_BUCKET")
    validate_key(video.get("key"), experiment_prefix, "normalizedVideo.key")
    if not isinstance(video.get("sha256"), str) or not SHA256_PATTERN.fullmatch(video["sha256"]):
        raise ValueError("normalizedVideo.sha256 must be a lowercase SHA-256")
    _positive_int(video.get("sizeBytes"), "normalizedVideo.sizeBytes")
    if video.get("contentType") != "video/mp4":
        raise ValueError("normalizedVideo.contentType must be video/mp4")

    media = _object(request.get("expectedMedia"), "expectedMedia")
    if set(media) != {"frames", "fps", "width", "height"}:
        raise ValueError("expectedMedia fields are invalid")
    _positive_int(media.get("frames"), "expectedMedia.frames")
    if media.get("fps") != EXPECTED_FPS:
        raise ValueError("SAM requires 24/1 fps")
    _positive_int(media.get("width"), "expectedMedia.width")
    _positive_int(media.get("height"), "expectedMedia.height")

    parameters = _object(request.get("parameters"), "parameters")
    expected_parameters = {
        "anchorFrame", "boxXywh", "textPrompt",
        "outputProbabilityThreshold", "qaVersion", "evidenceMode",
    }
    if set(parameters) - expected_parameters or not (
        expected_parameters - {"evidenceMode"}
    ).issubset(parameters):
        raise ValueError("parameters fields are invalid")
    anchor = parameters.get("anchorFrame")
    if (
        isinstance(anchor, bool)
        or not isinstance(anchor, int)
        or not 0 <= anchor < media["frames"]
    ):
        raise ValueError(
            f"anchorFrame must be an integer between 0 and {media['frames'] - 1}"
        )
    raw_box = parameters.get("boxXywh")
    if not isinstance(raw_box, list) or len(raw_box) != 4:
        raise ValueError("boxXywh must be [x, y, width, height]")
    box = [_finite_number(value, f"boxXywh[{index}]") for index, value in enumerate(raw_box)]
    x, y, width, height = box
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
        raise ValueError("boxXywh must have positive size inside normalized [0, 1]")
    parameters["boxXywh"] = box
    prompt = parameters.get("textPrompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt.strip()) > 200:
        raise ValueError("textPrompt must be a non-empty string of at most 200 characters")
    parameters["textPrompt"] = prompt.strip()
    threshold = _finite_number(
        parameters.get("outputProbabilityThreshold"),
        "outputProbabilityThreshold",
    )
    if not 0 < threshold < 1:
        raise ValueError("outputProbabilityThreshold must be between 0 and 1")
    parameters["outputProbabilityThreshold"] = threshold
    if parameters.get("qaVersion") != 1:
        raise ValueError("qaVersion must be 1")
    evidence_mode = parameters.get("evidenceMode", "full")
    if evidence_mode not in {"full", "none"}:
        raise ValueError("evidenceMode must be full or none")
    parameters["evidenceMode"] = evidence_mode

    output = _object(request.get("output"), "output")
    if set(output) != {"bucket", "prefix"}:
        raise ValueError("output fields are invalid")
    if output.get("bucket") != configured_bucket:
        raise ValueError("output.bucket does not match configured S3_BUCKET")
    expected_prefix = f"{experiment_prefix}sam/{input_hash}/"
    if output.get("prefix") != expected_prefix:
        raise ValueError("output.prefix does not match runId, stage, and inputHash")
    return request


def s3_settings_from_env(environ: dict[str, str] | None = None) -> S3Settings:
    values = os.environ if environ is None else environ
    missing = [key for key in REQUIRED_SECRET_KEYS if not values.get(key)]
    if missing:
        raise RuntimeError(f"Modal secret studio-s3 is missing required keys: {', '.join(missing)}")
    return S3Settings(
        values["S3_ACCESS_KEY_ID"], values["S3_SECRET_ACCESS_KEY"],
        values["S3_ENDPOINT"], values["S3_BUCKET"], values["S3_REGION"],
    )


def create_s3_client(settings: S3Settings):
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3", aws_access_key_id=settings.access_key_id,
        aws_secret_access_key=settings.secret_access_key,
        endpoint_url=settings.endpoint, region_name=settings.region,
        config=Config(s3={"addressing_style": "path"}),
    )


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def download_verified(client: Any, reference: dict[str, Any], destination: Path) -> Path:
    partial = destination.with_suffix(destination.suffix + ".part")
    with partial.open("wb") as stream:
        client.download_fileobj(reference["bucket"], reference["key"], stream)
    digest, size = sha256_file(partial)
    if size != reference["sizeBytes"] or digest != reference["sha256"]:
        partial.unlink(missing_ok=True)
        raise ValueError("input artifact failed size or SHA-256 verification")
    partial.replace(destination)
    return destination


def artifact_ref(client: Any, bucket: str, key: str, path: Path, content_type: str) -> dict[str, Any]:
    digest, size = sha256_file(path)
    client.upload_file(
        str(path), bucket, key,
        ExtraArgs={"ContentType": content_type, "Metadata": {"sha256": digest}},
    )
    head = client.head_object(Bucket=bucket, Key=key)
    if int(head.get("ContentLength", -1)) != size or head.get("Metadata", {}).get("sha256") != digest:
        raise RuntimeError(f"uploaded artifact verification failed for {key}")
    return {
        "storage": "s3", "bucket": bucket, "key": key, "sha256": digest,
        "sizeBytes": size, "contentType": content_type,
    }


def committed_result(client: Any, bucket: str, key: str, request: dict[str, Any]) -> dict[str, Any] | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except Exception as error:
        code = getattr(error, "response", {}).get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            return None
        raise
    payload = json.loads(response["Body"].read())
    if (
        payload.get("success") is True
        and payload.get("runId") == request["runId"]
        and payload.get("stage") == request["stage"]
        and payload.get("inputHash") == request["inputHash"]
    ):
        return payload
    return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
