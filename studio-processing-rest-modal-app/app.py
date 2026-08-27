"""Authenticated asynchronous REST gateway for studio tracking and SAM."""

from __future__ import annotations

import os
import uuid
from typing import Any

import modal

from studio_gateway import (
    TERMINAL_STATUSES,
    IdempotencyConflict,
    RequestError,
    StageExecutionError,
    execute_pipeline,
    existing_job_for_request,
    idempotency_slot,
    new_job,
    public_job,
    request_digest,
    sanitized_error,
    utc_now,
    validate_pipeline_request,
)


APP_NAME = "studio-processing-rest-modal-app"
JOB_DICT_NAME = "studio-processing-rest-jobs"
TRACKING_APP_NAME = os.environ.get("STUDIO_TRACKING_MODAL_APP", "tapnextpp-modal-app")
TRACKING_FUNCTION_NAME = os.environ.get(
    "STUDIO_TRACKING_MODAL_FUNCTION", "run_tracking_stage"
)
SAM_APP_NAME = os.environ.get("STUDIO_SAM_MODAL_APP", "sam3-modal-app")
SAM_FUNCTION_NAME = os.environ.get("STUDIO_SAM_MODAL_FUNCTION", "run_sam_stage")
COORDINATOR_TIMEOUT = 90 * 60


api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi>=0.116,<1", "pydantic>=2.11,<3")
    .add_local_python_source("studio_gateway")
)
control_image = modal.Image.debian_slim(python_version="3.11").add_local_python_source(
    "studio_gateway"
)
app = modal.App(APP_NAME)
job_store = modal.Dict.from_name(JOB_DICT_NAME, create_if_missing=True)


def _persist_stage_update(
    job_id: str,
    stage: str,
    status: str,
    payload: dict[str, Any] | None,
) -> None:
    record = job_store.get(job_id)
    if record is None:
        raise RuntimeError("job record disappeared")
    timestamp = utc_now()
    stage_record: dict[str, Any] = {"status": status, "updatedAt": timestamp}
    if status == "running":
        stage_record["startedAt"] = timestamp
    elif status == "succeeded":
        stage_record.update(result=payload, completedAt=timestamp)
    elif status == "failed":
        stage_record.update(error=payload, completedAt=timestamp)
    record["stages"][stage] = stage_record
    record["status"] = "running" if status != "failed" else "failed"
    record["currentStage"] = stage
    record["updatedAt"] = timestamp
    if status == "failed":
        record["error"] = payload
        record["completedAt"] = timestamp
    job_store.put(job_id, record)


@app.function(image=control_image, timeout=COORDINATOR_TIMEOUT)
def run_pipeline_job(job_id: str, request: dict[str, Any]) -> dict[str, Any]:
    """Call existing S3-native workers sequentially and persist compact results."""

    workers = {
        "tracking": modal.Function.from_name(
            TRACKING_APP_NAME, TRACKING_FUNCTION_NAME
        ),
        "sam": modal.Function.from_name(SAM_APP_NAME, SAM_FUNCTION_NAME),
    }

    def invoke(stage: str, stage_request: dict[str, Any]) -> dict[str, Any]:
        return workers[stage].remote(stage_request)

    def update(
        stage: str, status: str, payload: dict[str, Any] | None
    ) -> None:
        _persist_stage_update(job_id, stage, status, payload)

    try:
        results = execute_pipeline(request, invoke, update)
    except StageExecutionError:
        failed = job_store.get(job_id)
        return public_job(failed) if failed is not None else {"id": job_id, "status": "failed"}

    record = job_store.get(job_id)
    if record is None:
        raise RuntimeError("job record disappeared")
    timestamp = utc_now()
    record.update(
        status="succeeded",
        currentStage="completed",
        result=results,
        completedAt=timestamp,
        updatedAt=timestamp,
    )
    job_store.put(job_id, record)
    return public_job(record)


def create_web_app(store: Any, spawn_job: Any, call_from_id: Any):
    from fastapi import FastAPI, Header, HTTPException

    web = FastAPI(
        title="Studio Processing REST API",
        version="1.0.0",
        description="Sequential TAPNext++ and SAM processing over S3 artifacts.",
    )

    @web.get("/health")
    def health():
        return {
            "service": APP_NAME,
            "ready": True,
            "workers": {
                "tracking": {
                    "app": TRACKING_APP_NAME,
                    "function": TRACKING_FUNCTION_NAME,
                },
                "sam": {"app": SAM_APP_NAME, "function": SAM_FUNCTION_NAME},
            },
            "execution": "sequential",
        }

    @web.post("/v1/jobs", status_code=202)
    def submit_job(
        body: dict[str, Any],
        idempotency_key: str | None = Header(
            default=None, alias="Idempotency-Key"
        ),
    ):
        if idempotency_key is None or not idempotency_key.strip():
            raise HTTPException(status_code=400, detail="Idempotency-Key is required")
        if len(idempotency_key) > 200:
            raise HTTPException(
                status_code=400, detail="Idempotency-Key must be at most 200 characters"
            )
        try:
            request = validate_pipeline_request(body)
        except RequestError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        digest = request_digest(request)
        slot = idempotency_slot(idempotency_key)
        existing_id = store.get(slot)
        if existing_id is not None:
            existing = store.get(existing_id)
            if existing is None:
                raise HTTPException(
                    status_code=409,
                    detail="idempotency record exists but its job has expired",
                )
            try:
                record = existing_job_for_request(existing, digest)
            except IdempotencyConflict as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            return {
                "id": record["id"],
                "status": record["status"],
                "currentStage": record["currentStage"],
            }

        job_id = str(uuid.uuid4())
        record = new_job(job_id, request, digest)
        store.put(job_id, record)
        store.put(slot, job_id)
        try:
            call = spawn_job(job_id, request)
        except BaseException as error:
            timestamp = utc_now()
            failure = sanitized_error("tracking", type(error).__name__)
            failure.update(code="dispatch_failed", message="Modal coordinator dispatch failed.")
            record.update(
                status="failed",
                error=failure,
                completedAt=timestamp,
                updatedAt=timestamp,
            )
            store.put(job_id, record)
            raise HTTPException(
                status_code=503, detail="Modal coordinator is unavailable"
            ) from error
        record["callId"] = call.object_id
        record["updatedAt"] = utc_now()
        store.put(job_id, record)
        return {"id": job_id, "status": "queued", "currentStage": "tracking"}

    @web.get("/v1/jobs/{job_id}")
    def get_job(job_id: str):
        record = store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job not found or expired")
        if record["status"] not in TERMINAL_STATUSES and record.get("callId"):
            call = call_from_id(record["callId"])
            try:
                call.get(timeout=0)
            except (TimeoutError, modal.exception.TimeoutError):
                pass
            except modal.exception.OutputExpiredError:
                current = store.get(job_id)
                if current is not None and current["status"] in TERMINAL_STATUSES:
                    record = current
                else:
                    timestamp = utc_now()
                    failure = {
                        "code": "coordinator_result_expired",
                        "message": "Modal coordinator result expired before a terminal job record was written.",
                        "stage": record.get("currentStage", "tracking"),
                        "type": "OutputExpiredError",
                        "retryable": True,
                    }
                    record.update(
                        status="failed",
                        error=failure,
                        completedAt=timestamp,
                        updatedAt=timestamp,
                    )
                    store.put(job_id, record)
            except BaseException as error:
                current = store.get(job_id) or record
                if current["status"] not in TERMINAL_STATUSES:
                    timestamp = utc_now()
                    stage = current.get("currentStage", "tracking")
                    failure = {
                        "code": "coordinator_failed",
                        "message": "Modal coordinator failed; inspect the Modal logs.",
                        "stage": stage,
                        "type": type(error).__name__,
                        "retryable": True,
                    }
                    current.update(
                        status="failed",
                        error=failure,
                        completedAt=timestamp,
                        updatedAt=timestamp,
                    )
                    store.put(job_id, current)
                record = current
            else:
                record = store.get(job_id) or record
        return public_job(record)

    return web


@app.function(image=api_image, min_containers=0, max_containers=4)
@modal.concurrent(max_inputs=50)
@modal.asgi_app(requires_proxy_auth=True)
def api():
    return create_web_app(
        job_store,
        lambda job_id, request: run_pipeline_job.spawn(job_id, request),
        modal.FunctionCall.from_id,
    )


@app.local_entrypoint()
def main() -> None:
    print(
        {
            "service": APP_NAME,
            "tracking": f"{TRACKING_APP_NAME}/{TRACKING_FUNCTION_NAME}",
            "sam": f"{SAM_APP_NAME}/{SAM_FUNCTION_NAME}",
            "execution": "sequential",
        }
    )
