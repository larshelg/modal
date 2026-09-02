"""Local Modal CLI for WanGP generation and artifact inspection."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import modal

from wangpt_common import (
    CATALOG_DICT_NAME,
    JOB_DICT_NAME,
    WAN_COMMIT,
    WORKER_APP_NAME,
    filter_models,
    resolve_generation_kind,
    utc_now,
    validate_data_paths,
    validate_job_request,
)


DATA_VOLUME_NAME = "wangp-data"
KREA_MODEL = "krea2_turbo"
KREA_HELP = {
    "model": KREA_MODEL,
    "kind": "image",
    "submit_command": (
        "python3 -m modal run control.py::krea --params-json '<JSON object>'"
    ),
    "params_json": {
        "required": {
            "prompt": {"type": "string"},
        },
        "optional": {
            "negative_prompt": {"type": "string", "default": ""},
            "resolution": {"type": "string", "default": "1024x1024"},
            "num_inference_steps": {"type": "integer", "default": 8},
            "seed": {"type": "integer", "default": -1},
            "batch_size": {"type": "integer", "default": 1},
            "guidance_scale": {"type": "number", "default": 0},
            "flow_shift": {"type": "number", "default": 5.0},
            "activated_loras": {
                "type": "array[string]",
                "items": "absolute paths under /data",
            },
            "loras_multipliers": {
                "type": "string",
                "description": "weights matching activated_loras in order",
            },
        },
    },
    "example": {
        "prompt": "A red fox walking through fresh snow at golden hour",
        "negative_prompt": "blurry, low quality",
        "resolution": "1024x1024",
        "num_inference_steps": 8,
        "seed": -1,
        "batch_size": 1,
        "guidance_scale": 0,
        "flow_shift": 5.0,
    },
    "lora_example": {
        "prompt": "linda standing in a sunlit photography studio",
        "activated_loras": [
            "/data/loras/krea2/linda_krea2_v1.safetensors",
        ],
        "loras_multipliers": "0.8",
    },
}
# This app only groups local_entrypoints for `modal run`; it is never deployed.
app = modal.App()
job_store = modal.Dict.from_name(JOB_DICT_NAME, create_if_missing=True)
catalog_store = modal.Dict.from_name(CATALOG_DICT_NAME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)


def load_deployed_catalog() -> dict[str, Any]:
    """Read this worker revision's catalog, publishing it on a cache miss."""
    catalog = catalog_store.get(WAN_COMMIT)
    if catalog is not None:
        return catalog

    publisher = modal.Function.from_name(WORKER_APP_NAME, "publish_catalog")
    publisher.remote()
    catalog = catalog_store.get(WAN_COMMIT)
    if catalog is None:
        raise RuntimeError("worker app did not publish its model catalog")
    return catalog


def generation_worker_name(kind: str) -> str:
    """Map catalog output kinds onto deployed GPU worker classes."""
    if kind == "video":
        return "WanGPVideoWorker"
    if kind in {"image", "audio"}:
        return "WanGPImageWorker"
    raise ValueError(f"unsupported generation kind: {kind}")


def submit_generation(
    model: str,
    params: dict[str, Any],
    kind: str | None = None,
) -> dict[str, Any]:
    """Validate, route, and detach one generation job."""
    model = model.strip()
    validate_job_request(model, params)
    validate_data_paths(params)
    resolved_kind = resolve_generation_kind(
        load_deployed_catalog(),
        model,
        kind,
        params,
    )

    job_id = str(uuid.uuid4())
    timestamp = utc_now()
    record = {
        "id": job_id,
        "call_id": None,
        "status": "queued",
        "kind": resolved_kind,
        "model": model,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    job_store.put(job_id, record)

    worker_name = generation_worker_name(resolved_kind)
    worker = modal.Cls.from_name(WORKER_APP_NAME, worker_name)
    try:
        call = worker().run.spawn(job_id, model, params)
    except BaseException as exc:
        failure_time = utc_now()
        record.update(
            status="failed",
            result={
                "success": False,
                "errors": [{"message": str(exc), "stage": "dispatch"}],
            },
            completed_at=failure_time,
            updated_at=failure_time,
        )
        job_store.put(job_id, record)
        raise

    record["call_id"] = call.object_id
    record["updated_at"] = utc_now()
    job_store.put(job_id, record)
    return {"id": job_id, "status": "queued", "kind": resolved_kind}


def get_generation_job(job_id: str) -> dict[str, Any]:
    """Return a job record and reconcile pre-worker Modal failures."""
    record = job_store.get(job_id)
    if record is None:
        raise ValueError("job not found or expired")

    if record["status"] in {"queued", "running"} and record.get("call_id"):
        call = modal.FunctionCall.from_id(record["call_id"])
        try:
            call.get(timeout=0)
        except TimeoutError:
            pass
        except modal.exception.OutputExpiredError as exc:
            raise ValueError("job result expired") from exc
        except Exception as exc:
            record = job_store.get(job_id, record)
            if record["status"] not in {"succeeded", "failed", "cancelled"}:
                timestamp = utc_now()
                record.update(
                    status="failed",
                    result={
                        "success": False,
                        "errors": [{"message": str(exc), "stage": "modal"}],
                    },
                    completed_at=timestamp,
                    updated_at=timestamp,
                )
                job_store.put(job_id, record)
        else:
            record = job_store.get(job_id, record)
    return record


def cancel_generation_job(job_id: str) -> dict[str, Any]:
    """Cancel one queued or running FunctionCall."""
    record = job_store.get(job_id)
    if record is None:
        raise ValueError("job not found or expired")
    if record["status"] == "cancelled":
        return record
    if record["status"] in {"succeeded", "failed"}:
        raise ValueError("job is already terminal")
    if not record.get("call_id"):
        raise RuntimeError("job has not finished dispatching")

    call = modal.FunctionCall.from_id(record["call_id"])
    call.cancel(terminate_containers=True)
    timestamp = utc_now()
    record.update(
        status="cancelled",
        completed_at=timestamp,
        updated_at=timestamp,
    )
    job_store.put(job_id, record)
    return record


def inspect_catalog(
    operation: str,
    model: str = "",
    family: str = "",
    model_type: str = "",
) -> Any:
    """Read model discovery data without starting the WanGP image."""
    catalog = load_deployed_catalog()
    if operation == "models":
        return filter_models(
            catalog["models"],
            family=family or None,
            model_type=model_type or None,
        )
    if operation not in {"defaults", "schema"}:
        raise ValueError(f"unknown catalog operation: {operation}")
    result = catalog[f"{operation}s"].get(model)
    if result is None:
        raise ValueError(f"unknown model: {model}")
    return result


def load_params_file(params_file: str) -> dict[str, Any]:
    """Load a local JSON object for the submit entrypoint."""
    if not params_file:
        return {}
    path = Path(params_file).expanduser()
    try:
        params = json.loads(path.read_text())
    except OSError as exc:
        raise ValueError(f"could not read params file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"params file is not valid JSON: {exc}") from exc
    if not isinstance(params, dict):
        raise ValueError("params file must contain a JSON object")
    return params


def load_params_json(params_json: str) -> dict[str, Any]:
    """Parse one inline JSON object for the Krea entrypoint."""
    try:
        params = json.loads(params_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"params JSON is not valid: {exc}") from exc
    if not isinstance(params, dict):
        raise ValueError("params JSON must be an object")
    return params


def submit_krea_generation(params: dict[str, Any]) -> dict[str, Any]:
    """Submit a full native Krea2 Turbo JSON request."""
    prompt = params.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("params.prompt must be a non-empty string")
    return submit_generation(KREA_MODEL, params, "image")


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


@app.local_entrypoint()
def refresh_catalog() -> None:
    """Publish the catalog from the current worker deployment."""
    publisher = modal.Function.from_name(WORKER_APP_NAME, "publish_catalog")
    print_json(publisher.remote())


@app.local_entrypoint()
def submit(model: str, params_file: str = "", kind: str = "") -> None:
    params = load_params_file(params_file)
    requested_kind = kind.strip() or None
    result = submit_generation(
        model,
        params,
        requested_kind,
    )
    print_json(result)


@app.local_entrypoint()
def krea(params_json: str) -> None:
    """Submit Krea2 Turbo from an inline native WanGP JSON object."""
    print_json(submit_krea_generation(load_params_json(params_json)))


@app.local_entrypoint()
def krea_help() -> None:
    """Print the Krea JSON contract, defaults, and examples."""
    print_json(KREA_HELP)


@app.local_entrypoint()
def status(job_id: str) -> None:
    print_json(get_generation_job(job_id))


@app.local_entrypoint()
def cancel(job_id: str) -> None:
    print_json(cancel_generation_job(job_id))


@app.local_entrypoint()
def models(family: str = "", model_type: str = "") -> None:
    result = inspect_catalog(
        "models",
        "",
        family,
        model_type,
    )
    print_json(result)


@app.local_entrypoint()
def defaults(model: str) -> None:
    result = inspect_catalog("defaults", model)
    print_json(result)


@app.local_entrypoint()
def schema(model: str) -> None:
    result = inspect_catalog("schema", model)
    print_json(result)


@app.local_entrypoint()
def loras(recursive: bool = False) -> None:
    """List LoRAs through the Volume API without starting a container."""
    entries = data_volume.listdir("loras", recursive=recursive)
    print_json(
        [
            {
                "path": entry.path,
                "type": entry.type.value,
                "size": entry.size,
                "mtime": entry.mtime,
            }
            for entry in entries
        ]
    )
