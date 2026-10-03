"""Local-only client for the deployed Sol app; never imports the GPU image."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import modal

from sol_refiner.common import APP_NAME, JOB_DICT_NAME, TERMINAL_STATES, utc_now, validate_request

app = modal.App()
jobs = modal.Dict.from_name(JOB_DICT_NAME, create_if_missing=True)


def submit_refinement(request: dict) -> dict:
    request = validate_request(request)
    job_id, now = str(uuid.uuid4()), utc_now()
    record = {"id": job_id, "status": "queued", "phase": "queued", "created_at": now, "updated_at": now}
    jobs.put(job_id, record)
    try:
        call = modal.Cls.from_name(APP_NAME, "SolRefiner")().refine.spawn(job_id, request)
    except Exception as exc:
        record.update(status="failed", completed_at=utc_now(), updated_at=utc_now(),
                      error={"type": type(exc).__name__, "message": str(exc), "phase": "dispatch"})
        jobs.put(job_id, record)
        raise
    # Separate entry prevents dispatch from overwriting a worker that already finished.
    jobs.put(f"call:{job_id}", call.object_id)
    return {"id": job_id, "call_id": call.object_id, "status": "queued"}


def get_job(job_id: str) -> dict:
    record = jobs.get(job_id)
    if record is None:
        raise ValueError("job not found or expired")
    call_id = jobs.get(f"call:{job_id}")
    if record["status"] not in TERMINAL_STATES and call_id:
        try:
            result = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            pass
        except Exception as exc:
            record = jobs.get(job_id)
            if record["status"] not in TERMINAL_STATES:
                record.update(status="failed", completed_at=utc_now(), updated_at=utc_now(),
                              error={"type": type(exc).__name__, "message": str(exc), "phase": "modal"})
                jobs.put(job_id, record)
        else:
            record = jobs.get(job_id)
            if record["status"] not in TERMINAL_STATES:
                record.update(status="succeeded", phase="complete", result=result,
                              completed_at=utc_now(), updated_at=utc_now())
                jobs.put(job_id, record)
    return {**record, "call_id": call_id}


@app.local_entrypoint()
def refine(params_file: str):
    """Validate and detach an HTTPS/S3 video refinement request."""
    print(json.dumps(submit_refinement(json.loads(Path(params_file).read_text())), indent=2))


@app.local_entrypoint()
def status(job_id: str):
    print(json.dumps(get_job(job_id), indent=2))


@app.local_entrypoint()
def smoke(input_file: str, prompt: str, output_file: str = "sol-refined.mp4",
          width: int = 1344, height: int = 768, seed: int = 303000, decoder_seed: int = 20260826):
    """Run one local clip against the deployed worker and save the verified MP4."""
    from sol_refiner.runtime import probe_video

    source, target = Path(input_file), Path(output_file)
    metadata_path = target.with_suffix(target.suffix + ".json")
    if target.exists() or metadata_path.exists():
        raise ValueError("output or metrics file already exists; choose a new output path")
    if not source.is_file() or not 0 < source.stat().st_size <= 64 * 1024 * 1024:
        raise ValueError("smoke input must be an existing video no larger than 64 MiB")
    probe_video(source)
    request = validate_request({"input_url": "https://local.invalid/smoke.mp4", "prompt": prompt,
                                "output_width": width, "output_height": height,
                                "seed": seed, "decoder_seed": decoder_seed})
    response = modal.Cls.from_name(APP_NAME, "SolRefiner")().smoke.remote(source.read_bytes(), request)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(response["video"])
    with metadata_path.open("x") as stream:
        json.dump(response["metrics"], stream, indent=2)
    print(json.dumps({"output_file": str(target), **response["metrics"]}, indent=2))


@app.local_entrypoint()
def output_url(job_id: str, expires: int = 3600):
    """Generate a fresh URL locally rather than persisting expiring credentials."""
    from sol_refiner.storage import s3_client

    if not 60 <= expires <= 604800:
        raise ValueError("expires must be between 60 and 604800 seconds")
    record = get_job(job_id)
    if record["status"] != "succeeded":
        raise ValueError("job has no successful output")
    artifact = record["result"]["output"]
    client, bucket = s3_client()
    if artifact["bucket"] != bucket:
        raise ValueError("output belongs to a different configured bucket")
    url = client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": artifact["key"]}, ExpiresIn=expires)
    print(json.dumps({"id": job_id, "url": url, "expires_in_seconds": expires}, indent=2))
