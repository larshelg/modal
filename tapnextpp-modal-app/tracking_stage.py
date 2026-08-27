"""S3 contract, homography processing, and review artifacts for TAPNext++."""

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
EXPECTED_STAGE = "tracking"
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
    segments = key.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise ValueError(f"{field} is unsafe")
    if not key.startswith(prefix):
        raise ValueError(f"{field} must remain inside {prefix}")
    return key


def _polygon_crosses(corners: list[list[float]]) -> bool:
    signs: list[float] = []
    for index in range(4):
        first = corners[(index + 1) % 4]
        origin = corners[index]
        second = corners[(index + 2) % 4]
        edge_a = (first[0] - origin[0], first[1] - origin[1])
        edge_b = (second[0] - first[0], second[1] - first[1])
        signs.append(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])
    return not (all(value > 0 for value in signs) or all(value < 0 for value in signs))


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
        raise ValueError("stage must be tracking")
    input_hash = request.get("inputHash")
    if not isinstance(input_hash, str) or not SHA256_PATTERN.fullmatch(input_hash):
        raise ValueError("inputHash must be a lowercase SHA-256")

    experiment_prefix = f"studio-experiments/{run_id}/"
    inputs = _object(request.get("inputs"), "inputs")
    if set(inputs) != {"normalizedVideo"}:
        raise ValueError("inputs must contain only normalizedVideo")
    video = _object(inputs["normalizedVideo"], "inputs.normalizedVideo")
    if set(video) != {
        "storage", "bucket", "key", "sha256", "sizeBytes", "contentType"
    }:
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
        raise ValueError("tracking requires 24/1 fps")
    _positive_int(media.get("width"), "expectedMedia.width")
    _positive_int(media.get("height"), "expectedMedia.height")

    parameters = _object(request.get("parameters"), "parameters")
    allowed_parameters = {
        "frameZeroCorners", "cornerOrder", "surface", "analysis",
        "maxReferenceAreaRatio", "qaVersion", "evidenceMode",
    }
    if set(parameters) - allowed_parameters:
        raise ValueError("parameters contains unsupported fields")
    if parameters.get("cornerOrder") != "top-left, top-right, bottom-right, bottom-left":
        raise ValueError("cornerOrder is invalid")
    if parameters.get("qaVersion") not in {1, 2}:
        raise ValueError("qaVersion must be 1 or 2")
    evidence_mode = parameters.get("evidenceMode", "full")
    if evidence_mode not in {"full", "none"}:
        raise ValueError("evidenceMode must be full or none")
    parameters["evidenceMode"] = evidence_mode
    corners = parameters.get("frameZeroCorners")
    if not isinstance(corners, list) or len(corners) != 4:
        raise ValueError("frameZeroCorners must contain four points")
    normalized_corners: list[list[float]] = []
    for index, corner in enumerate(corners):
        if not isinstance(corner, list) or len(corner) != 2:
            raise ValueError(f"frameZeroCorners[{index}] must be [x, y]")
        x = _finite_number(corner[0], f"frameZeroCorners[{index}][0]")
        y = _finite_number(corner[1], f"frameZeroCorners[{index}][1]")
        if not 0 <= x < media["width"] or not 0 <= y < media["height"]:
            raise ValueError("frameZeroCorners must remain inside the source frame")
        normalized_corners.append([x, y])
    if _polygon_crosses(normalized_corners):
        raise ValueError("frameZeroCorners must be convex and consistently ordered")
    parameters["frameZeroCorners"] = normalized_corners

    for name in ("surface", "analysis"):
        dimensions = _object(parameters.get(name), f"parameters.{name}")
        if set(dimensions) != {"width", "height"}:
            raise ValueError(f"parameters.{name} fields are invalid")
        _positive_int(dimensions.get("width"), f"parameters.{name}.width")
        _positive_int(dimensions.get("height"), f"parameters.{name}.height")
    ratio = _finite_number(
        parameters.get("maxReferenceAreaRatio", 4),
        "parameters.maxReferenceAreaRatio",
    )
    if ratio < 1:
        raise ValueError("maxReferenceAreaRatio must be at least 1")
    parameters["maxReferenceAreaRatio"] = ratio

    output = _object(request.get("output"), "output")
    if set(output) != {"bucket", "prefix"}:
        raise ValueError("output fields are invalid")
    if output.get("bucket") != configured_bucket:
        raise ValueError("output.bucket does not match configured S3_BUCKET")
    expected_prefix = f"{experiment_prefix}tracking/{input_hash}/"
    if output.get("prefix") != expected_prefix:
        raise ValueError("output.prefix does not match runId, stage, and inputHash")
    return request


def s3_settings_from_env(environ: dict[str, str] | None = None) -> S3Settings:
    values = os.environ if environ is None else environ
    missing = [key for key in REQUIRED_SECRET_KEYS if not values.get(key)]
    if missing:
        raise RuntimeError(f"Modal secret studio-s3 is missing required keys: {', '.join(missing)}")
    return S3Settings(
        access_key_id=values["S3_ACCESS_KEY_ID"],
        secret_access_key=values["S3_SECRET_ACCESS_KEY"],
        endpoint=values["S3_ENDPOINT"],
        bucket=values["S3_BUCKET"],
        region=values["S3_REGION"],
    )


def create_s3_client(settings: S3Settings):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        aws_access_key_id=settings.access_key_id,
        aws_secret_access_key=settings.secret_access_key,
        endpoint_url=settings.endpoint,
        region_name=settings.region,
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
        raise ValueError("normalizedVideo failed size or SHA-256 verification")
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
        "storage": "s3", "bucket": bucket, "key": key,
        "sha256": digest, "sizeBytes": size, "contentType": content_type,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
