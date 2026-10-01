"""Sol job execution, kept injectable for CPU lifecycle tests."""

import tempfile
import time
from pathlib import Path

from sol_common import utc_now, validate_request
from sol_storage import download_input, s3_client, upload_output


def run_job(store, runtime, job_id: str, request: dict) -> dict:
    record = store.get(job_id)
    if record is None:
        raise ValueError("job not found")
    started = time.monotonic()

    def update(**fields):
        record.update(fields, updated_at=utc_now())
        store.put(job_id, dict(record))

    try:
        request = validate_request(request)
        update(status="running", phase="download", started_at=utc_now())
        client, bucket = s3_client()
        with tempfile.TemporaryDirectory(prefix="sol-") as temporary:
            directory = Path(temporary)
            original = download_input(request["input_url"], directory / "input.mp4", client, bucket)
            download_ms = round((time.monotonic() - started) * 1000)
            update(phase="refine")
            output, metrics = runtime.refine(original, directory, request)
            update(phase="upload")
            upload_start = time.monotonic()
            artifact = upload_output(output, job_id, client, bucket)
            result = {
                **metrics, "success": True, "output_media": metrics["output"], "output": artifact,
                "download_ms": download_ms,
                "upload_ms": round((time.monotonic() - upload_start) * 1000),
                "worker_total_ms": round((time.monotonic() - started) * 1000),
                **{k: request[k] for k in ("generation_id", "asset_id") if k in request},
            }
        update(status="succeeded", phase="complete", result=result, completed_at=utc_now())
        return result
    except Exception as exc:
        update(status="failed", completed_at=utc_now(), error={
            "type": type(exc).__name__, "message": str(exc), "phase": record.get("phase", "validate"),
        })
        raise
