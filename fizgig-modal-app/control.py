"""Local Modal CLI for the deployed Fizgig worker; no HTTP or GPU-image imports."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import modal

from fizgig_common import (
    APP_NAME,
    JOB_DICT_NAME,
    SUPPORTED_FAMILIES,
    TERMINAL_STATUSES,
    training_intent,
    utc_now,
    validate_component,
)

app = modal.App()
job_store = modal.Dict.from_name(JOB_DICT_NAME, create_if_missing=True)


def _load_job(job_id: str) -> dict[str, Any]:
    job_id = validate_component(job_id, "job_id")
    record = job_store.get(job_id)
    if record is None:
        raise ValueError("training job not found or expired")
    return record


def _metadata(job_id: str) -> dict[str, Any]:
    return job_store.get(f"submission:{job_id}", {})


def _public_record(record: dict[str, Any], *, logs: bool = False) -> dict[str, Any]:
    result = {**record, **_metadata(record["id"])}
    if not logs and isinstance(result.get("progress"), dict):
        result["progress"] = dict(result["progress"])
        result["progress"].pop("log_tail", None)
    return result


def _fail_job(job_id: str, exc: Exception, stage: str) -> dict[str, Any]:
    record = _load_job(job_id)
    if record["status"] not in TERMINAL_STATUSES:
        timestamp = utc_now()
        record.update(
            status="failed",
            error={"message": str(exc), "type": type(exc).__name__, "stage": stage},
            completed_at=timestamp,
            updated_at=timestamp,
        )
        job_store.put(job_id, record)
    return record


def submit_training(request: dict[str, Any]) -> dict[str, Any]:
    """Validate and spawn a job on the stable deployment, then return immediately."""
    return _spawn_training(training_intent(request))


def _spawn_training(request: dict[str, Any], resumed_from: str | None = None) -> dict[str, Any]:
    job_id, timestamp = str(uuid.uuid4()), utc_now()
    metadata = {"request": request, "call_id": None}
    if resumed_from is not None:
        metadata["resumed_from"] = resumed_from
    job_store.put(f"submission:{job_id}", metadata)
    job_store.put(job_id, {
        "id": job_id,
        "status": "queued",
        "request": request,
        "created_at": timestamp,
        "updated_at": timestamp,
    })
    try:
        worker = modal.Function.from_name(APP_NAME, "run_training")
        call = worker.spawn(job_id, request)
    except Exception as exc:
        _fail_job(job_id, exc, "dispatch")
        raise
    # The worker may already be running or finished. Keep client metadata in
    # a separate key so attaching the call ID cannot reset worker progress.
    metadata["call_id"] = call.object_id
    job_store.put(f"submission:{job_id}", metadata)
    return {"id": job_id, "status": "queued", **metadata}


def get_training_job(job_id: str, *, logs: bool = False) -> dict[str, Any]:
    """Poll once and reconcile failures before the GPU worker could start."""
    record = _load_job(job_id)
    job_id = record["id"]
    call_id = _metadata(record["id"]).get("call_id") or record.get("call_id")
    if record["status"] not in TERMINAL_STATUSES and call_id:
        try:
            returned = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            record = _load_job(job_id)
        except modal.exception.OutputExpiredError as exc:
            # Expired output is not evidence that training failed. Keep the
            # persistent job record as-is, unless it became terminal meanwhile.
            record = _load_job(job_id)
            if record["status"] not in TERMINAL_STATUSES:
                raise ValueError("training job result expired; inspect worker logs") from exc
        except Exception as exc:
            record = _fail_job(job_id, exc, "modal")
        else:
            record = _load_job(job_id)
            if record["status"] not in TERMINAL_STATUSES:
                if not isinstance(returned, dict) or returned.get("status") not in TERMINAL_STATUSES:
                    raise RuntimeError("worker returned without a terminal training record")
                record.update(returned, id=job_id)
                job_store.put(job_id, record)
    return _public_record(record, logs=logs)


def pause_training_job(job_id: str) -> dict[str, Any]:
    record = get_training_job(job_id)
    if record["status"] != "running":
        raise ValueError("only a running training job can be paused")
    pause_function = modal.Function.from_name(APP_NAME, "request_pause")
    return _public_record(pause_function.remote(record["id"]))


def resume_training_job(job_id: str) -> dict[str, Any]:
    record = get_training_job(job_id)
    result = record.get("result") or {}
    if record["status"] != "succeeded" or not result.get("paused"):
        raise ValueError("only a successfully paused training job can be resumed")
    # Old worker records contain resolved preset settings. Reconstruct just
    # the intent fields, preserving the family, preset, epoch count and trigger.
    request = {
        key: record["request"][key]
        for key in ("family", "dataset", "output_name", "preset", "trigger_word", "epochs")
        if record["request"].get(key) is not None
    }
    request["resume_from"] = result.get("resume_from") or "latest"
    return _spawn_training(training_intent(request, allow_resume=True), record["id"])


def cancel_training_job(job_id: str) -> dict[str, Any]:
    record = _load_job(job_id)
    job_id = record["id"]
    if record["status"] == "cancelled":
        return _public_record(record)
    if record["status"] in TERMINAL_STATUSES:
        raise ValueError("training job is already terminal")
    call_id = _metadata(record["id"]).get("call_id") or record.get("call_id")
    if not call_id:
        raise RuntimeError("job has no FunctionCall ID; dispatch is incomplete or it predates this client")
    try:
        modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
    except Exception:
        record = _load_job(job_id)
        if record["status"] in TERMINAL_STATUSES:
            return _public_record(record)
        raise
    record = _load_job(job_id)
    # Completion may win the race with cancellation.
    if record["status"] not in TERMINAL_STATUSES:
        timestamp = utc_now()
        record.update(status="cancelled", completed_at=timestamp, updated_at=timestamp)
        job_store.put(job_id, record)
    return _public_record(record)


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


@app.local_entrypoint()
def health() -> None:
    """Read deployed configuration using the small CPU-only health function."""
    print_json(modal.Function.from_name(APP_NAME, "health").remote())


@app.local_entrypoint()
def submit(
    family: str,
    dataset: str,
    output_name: str,
    preset: str,
    trigger_word: str = "",
    epochs: int = 0,
) -> None:
    """Submit typed training intent; zero epochs means the preset default."""
    request = {"family": family, "dataset": dataset, "output_name": output_name, "preset": preset}
    if trigger_word:
        request["trigger_word"] = trigger_word
    if epochs != 0:
        request["epochs"] = epochs
    print_json(submit_training(request))


@app.local_entrypoint()
def submit_file(request_file: str) -> None:
    """Submit the same allowlisted request from a JSON file."""
    print_json(submit_training(json.loads(Path(request_file).expanduser().read_text())))


@app.local_entrypoint()
def status(job_id: str, logs: bool = False) -> None:
    print_json(get_training_job(job_id, logs=logs))


@app.local_entrypoint()
def pause(job_id: str) -> None:
    print_json(pause_training_job(job_id))


@app.local_entrypoint()
def resume(job_id: str) -> None:
    print_json(resume_training_job(job_id))


@app.local_entrypoint()
def cancel(job_id: str) -> None:
    print_json(cancel_training_job(job_id))


@app.local_entrypoint()
def fetch_models(family: str, dry_run: bool = False) -> None:
    """Download model weights through the deployed CPU function."""
    if family not in SUPPORTED_FAMILIES:
        raise ValueError(f"family must be one of: {', '.join(SUPPORTED_FAMILIES)}")
    function = modal.Function.from_name(APP_NAME, "fetch_models")
    print_json(function.remote(family, False, dry_run))
