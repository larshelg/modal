"""Pure validation and job-state helpers for the studio processing REST API."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Callable


STAGES = ("tracking", "sam")
TERMINAL_STATUSES = {"succeeded", "failed"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class RequestError(ValueError):
    """The submitted pipeline request does not satisfy the gateway contract."""


class IdempotencyConflict(ValueError):
    """An idempotency key was reused with a different request."""


class StageExecutionError(RuntimeError):
    def __init__(self, stage: str, cause: BaseException):
        self.stage = stage
        self.cause_type = type(cause).__name__
        super().__init__(f"{stage} stage failed")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def request_digest(request: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()


def idempotency_slot(key: str) -> str:
    return "idempotency:" + hashlib.sha256(key.encode("utf-8")).hexdigest()


def _required_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RequestError(f"{field} must be an object")
    return value


def _stage_request(body: dict[str, Any], stage: str) -> dict[str, Any]:
    request = _required_object(body.get(stage), stage)
    if request.get("schemaVersion") != 1:
        raise RequestError(f"{stage}.schemaVersion must be 1")
    if request.get("stage") != stage:
        raise RequestError(f"{stage}.stage must be {stage!r}")
    run_id = request.get("runId")
    if not isinstance(run_id, str) or not run_id:
        raise RequestError(f"{stage}.runId must be a non-empty string")
    input_hash = request.get("inputHash")
    if not isinstance(input_hash, str) or not SHA256_PATTERN.fullmatch(input_hash):
        raise RequestError(f"{stage}.inputHash must be a lowercase SHA-256")
    inputs = _required_object(request.get("inputs"), f"{stage}.inputs")
    _required_object(
        inputs.get("normalizedVideo"), f"{stage}.inputs.normalizedVideo"
    )
    _required_object(request.get("expectedMedia"), f"{stage}.expectedMedia")
    parameters = _required_object(request.get("parameters"), f"{stage}.parameters")
    evidence_mode = parameters.get("evidenceMode", "full")
    if evidence_mode not in {"full", "none"}:
        raise RequestError(
            f"{stage}.parameters.evidenceMode must be 'full' or 'none'"
        )
    output = _required_object(request.get("output"), f"{stage}.output")
    if not isinstance(output.get("bucket"), str) or not output["bucket"]:
        raise RequestError(f"{stage}.output.bucket must be a non-empty string")
    if not isinstance(output.get("prefix"), str) or not output["prefix"]:
        raise RequestError(f"{stage}.output.prefix must be a non-empty string")
    return request


def validate_pipeline_request(body: Any) -> dict[str, Any]:
    request = _required_object(body, "request")
    unknown = sorted(set(request) - {"schemaVersion", *STAGES})
    if unknown:
        raise RequestError(f"unsupported request fields: {', '.join(unknown)}")
    if request.get("schemaVersion") != 1:
        raise RequestError("schemaVersion must be 1")

    tracking = _stage_request(request, "tracking")
    sam = _stage_request(request, "sam")
    if tracking["runId"] != sam["runId"]:
        raise RequestError("tracking and sam runId values must match")
    if tracking["inputs"]["normalizedVideo"] != sam["inputs"]["normalizedVideo"]:
        raise RequestError("tracking and sam must use the same normalizedVideo")
    if tracking["expectedMedia"] != sam["expectedMedia"]:
        raise RequestError("tracking and sam expectedMedia values must match")
    tracking_evidence = tracking["parameters"].get("evidenceMode", "full")
    sam_evidence = sam["parameters"].get("evidenceMode", "full")
    if tracking_evidence != sam_evidence:
        raise RequestError("tracking and sam evidenceMode values must match")
    if tracking["output"]["bucket"] != sam["output"]["bucket"]:
        raise RequestError("tracking and sam output buckets must match")
    if tracking["output"]["prefix"] == sam["output"]["prefix"]:
        raise RequestError("tracking and sam output prefixes must be distinct")
    return request


def new_job(job_id: str, request: dict[str, Any], digest: str) -> dict[str, Any]:
    timestamp = utc_now()
    return {
        "id": job_id,
        "callId": None,
        "status": "queued",
        "currentStage": "tracking",
        "request": request,
        "requestDigest": digest,
        "runId": request["tracking"]["runId"],
        "stages": {
            "tracking": {"status": "pending"},
            "sam": {"status": "pending"},
        },
        "createdAt": timestamp,
        "updatedAt": timestamp,
    }


def public_job(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"callId", "request", "requestDigest"}
    }


def existing_job_for_request(
    record: dict[str, Any], digest: str
) -> dict[str, Any]:
    if record.get("requestDigest") != digest:
        raise IdempotencyConflict(
            "Idempotency-Key was already used for a different request"
        )
    return record


def sanitized_error(stage: str, error_type: str) -> dict[str, Any]:
    label = "TAPNext++ tracking" if stage == "tracking" else "SAM"
    return {
        "code": "stage_failed",
        "message": f"{label} stage failed; inspect the Modal worker logs.",
        "stage": stage,
        "type": error_type,
        "retryable": True,
    }


def execute_pipeline(
    request: dict[str, Any],
    invoke: Callable[[str, dict[str, Any]], dict[str, Any]],
    update: Callable[[str, str, dict[str, Any] | None], None],
) -> dict[str, dict[str, Any]]:
    """Invoke tracking and then SAM, reporting each state transition."""
    results: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        update(stage, "running", None)
        try:
            result = invoke(stage, request[stage])
        except BaseException as error:
            update(stage, "failed", sanitized_error(stage, type(error).__name__))
            raise StageExecutionError(stage, error) from error
        if not isinstance(result, dict):
            error = TypeError("worker result must be an object")
            update(stage, "failed", sanitized_error(stage, type(error).__name__))
            raise StageExecutionError(stage, error)
        results[stage] = result
        update(stage, "succeeded", result)
    return results
