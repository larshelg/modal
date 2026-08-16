"""Asynchronous WanGP REST API on Modal.

WanGP is the internal rendering engine. The public CPU endpoint submits work to
an autoscaling GPU class and returns immediately with a pollable job ID.
"""

from __future__ import annotations

import mimetypes
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import modal


APP_NAME = "wangp-rest"
WAN_ROOT = Path("/opt/Wan2GP")
WAN2AI_ROOT = Path("/opt/Wan2AI")
DATA_ROOT = Path("/data")
CATALOG_PATH = Path("/opt/wangp-catalog.json")
WAN_COMMIT = "a042474d477a741d6b9b60fc6ff304077113cb25"
WAN2AI_COMMIT = "2539c3a87b64fa0f619695f02410fc92c63cba7d"

GPU_TYPE = os.environ.get("WANGP_GPU", "L40S")
MAX_CONTAINERS = int(os.environ.get("WANGP_MAX_CONTAINERS", "3"))
SCALEDOWN_WINDOW = 5 * 60
STARTUP_TIMEOUT = 30 * 60

DATA_VOLUME_NAME = "wangp-data"
JOB_DICT_NAME = "wangp-rest-jobs"

CACHE_DIRS = (
    "ckpts",
    "outputs",
    "config",
    "settings",
    "loras",
    "huggingface/hub",
    "huggingface/transformers",
    "triton",
    "torch-extensions",
    "cache",
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
job_store = modal.Dict.from_name(JOB_DICT_NAME, create_if_missing=True)

gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential",
        "clang",
        "cmake",
        "ffmpeg",
        "git",
        "libgl1",
        "libglib2.0-0",
        "ninja-build",
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install torch==2.10.0 torchvision==0.25.0 "
        "torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130",
        f"git clone https://github.com/deepbeepmeep/Wan2GP.git {WAN_ROOT}",
        f"cd {WAN_ROOT} && git checkout {WAN_COMMIT}",
        f"python -m pip install -r {WAN_ROOT}/requirements.txt",
        f"git clone https://github.com/PrimeEcto/Wan2AI.git {WAN2AI_ROOT}",
        f"cd {WAN2AI_ROOT} && git checkout {WAN2AI_COMMIT}",
        "python -m pip install 'grpclib>=0.4.7,<0.4.10'",
    )
    .env(
        {
            "WAN2GP_ROOT": str(WAN_ROOT),
            "HF_HOME": str(DATA_ROOT / "huggingface"),
            "HF_HUB_CACHE": str(DATA_ROOT / "huggingface/hub"),
            "HUGGINGFACE_HUB_CACHE": str(DATA_ROOT / "huggingface/hub"),
            "TRANSFORMERS_CACHE": str(DATA_ROOT / "huggingface/transformers"),
            "TRITON_CACHE_DIR": str(DATA_ROOT / "triton"),
            "TORCH_EXTENSIONS_DIR": str(DATA_ROOT / "torch-extensions"),
            "XDG_CACHE_HOME": str(DATA_ROOT / "cache"),
            "PYTHONPATH": f"{WAN_ROOT}:{WAN2AI_ROOT}/wangp/scripts",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_file(
        str(Path(__file__).with_name("generate_catalog.py")),
        remote_path="/opt/generate_catalog.py",
        copy=True,
    )
    .run_commands("python /opt/generate_catalog.py")
)

# Discovery is served by a CPU container from the catalog baked into this
# image. Sharing the image with GPU workers avoids maintaining two copies of
# WanGP and guarantees that discovery matches the generation runtime exactly.
api_image = gpu_image

app = modal.App(APP_NAME)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_absolute_path(path: str | os.PathLike[str]) -> Path:
    """Normalize traversal without resolving Modal Volume mount internals."""
    return Path(os.path.abspath(os.path.normpath(path)))


def canonical_output_path(path: str | os.PathLike[str]) -> Path:
    """Map WanGP/Modal internal Volume paths onto the public /data mount."""
    normalized = normalized_absolute_path(path)
    try:
        normalized.relative_to(DATA_ROOT / "outputs")
        return normalized
    except ValueError:
        pass

    parts = normalized.parts
    try:
        output_index = parts.index("outputs")
    except ValueError as exc:
        raise ValueError(f"WanGP output is not inside an outputs directory: {path}") from exc
    relative_parts = parts[output_index + 1 :]
    if not relative_parts or any(part in {"", ".", ".."} for part in relative_parts):
        raise ValueError(f"Invalid WanGP output path: {path}")
    return DATA_ROOT / "outputs" / Path(*relative_parts)


def validate_job_request(model: str, params: dict[str, Any]) -> dict[str, Any]:
    model = model.strip()
    if not model:
        raise ValueError("model must not be empty")
    if "_api" in params:
        raise ValueError("params._api is reserved by the service")
    settings = dict(params)
    settings["model_type"] = model
    return settings


def validate_data_paths(value: Any, key: str = "params") -> None:
    """Reject absolute filesystem paths outside the shared data Volume."""
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            validate_data_paths(child_value, f"{key}.{child_key}")
    elif isinstance(value, list):
        for index, child_value in enumerate(value):
            validate_data_paths(child_value, f"{key}[{index}]")
    elif isinstance(value, str) and value.startswith("/"):
        raw_path = value.split("|", 1)[0]
        try:
            normalized_absolute_path(raw_path).relative_to(DATA_ROOT)
        except ValueError as exc:
            raise ValueError(f"{key} must reference a path under /data") from exc


def serialize_error(error: Any) -> dict[str, Any]:
    return {
        "message": getattr(error, "message", str(error)),
        "stage": getattr(error, "stage", None),
        "task_index": getattr(error, "task_index", None),
        "task_id": getattr(error, "task_id", None),
    }


def serialize_output(path_string: str) -> dict[str, Any]:
    path = canonical_output_path(path_string)
    path.relative_to(DATA_ROOT / "outputs")
    stat = path.stat()
    return {
        "filename": path.name,
        "size_bytes": stat.st_size,
        "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    }


def register_outputs(result: Any, job_id: str) -> list[dict[str, Any]]:
    outputs = []
    for path_string in result.generated_files:
        path = canonical_output_path(path_string)
        metadata = serialize_output(str(path))
        output_id = str(uuid.uuid4())
        job_store.put(
            f"output:{output_id}",
            {
                "id": output_id,
                "job_id": job_id,
                "path": str(path),
                **metadata,
            },
        )
        outputs.append(
            {
                "id": output_id,
                **metadata,
                "url": f"/outputs/{output_id}",
            }
        )
    return outputs


def serialize_result(result: Any, outputs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "outputs": outputs if outputs is not None else [],
        "total_tasks": result.total_tasks,
        "successful_tasks": result.successful_tasks,
        "failed_tasks": result.failed_tasks,
        "errors": [serialize_error(error) for error in result.errors],
    }


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    return __import__("json").loads(path.read_text())


def filter_models(
    models: list[dict[str, Any]],
    family: str | None = None,
    model_type: str | None = None,
) -> list[dict[str, Any]]:
    result = models
    if family:
        result = [item for item in result if item.get("family") == family]
    if model_type:
        result = [item for item in result if item.get("model_type") == model_type]
    return result


def _ensure_cache_layout() -> None:
    for relative_path in CACHE_DIRS:
        (DATA_ROOT / relative_path).mkdir(parents=True, exist_ok=True)


def _force_directory_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink():
        if link_path.resolve() == target_path:
            return
        link_path.unlink()
    elif link_path.exists():
        if link_path.is_dir():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target_path, target_is_directory=True)


def _progress_dict(progress: Any) -> dict[str, Any]:
    return {
        "phase": getattr(progress, "phase", None),
        "status": getattr(progress, "status", None),
        "progress": getattr(progress, "progress", None),
        "current_step": getattr(progress, "current_step", None),
        "total_steps": getattr(progress, "total_steps", None),
    }


class JobCallbacks:
    def __init__(self, job_id: str):
        self.job_id = job_id

    def on_progress(self, progress: Any) -> None:
        record = job_store.get(self.job_id, {})
        record["progress"] = _progress_dict(progress)
        record["updated_at"] = utc_now()
        job_store.put(self.job_id, record)


@app.cls(
    image=gpu_image,
    gpu=GPU_TYPE,
    memory=65_536,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    startup_timeout=STARTUP_TIMEOUT,
    timeout=24 * 60 * 60,
    volumes={str(DATA_ROOT): data_volume},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
class WanGPWorker:
    @modal.enter()
    def initialize(self) -> None:
        _ensure_cache_layout()
        for name in ("ckpts", "outputs", "settings", "loras"):
            _force_directory_symlink(WAN_ROOT / name, DATA_ROOT / name)
        data_volume.commit()
        self.session = None
        self.model_family = None

    def _new_session(self) -> Any:
        from shared.api import init

        return init(
            root=WAN_ROOT,
            output_dir=DATA_ROOT / "outputs",
            cli_args=["--profile", "4", "--attention", "sdpa"],
            console_output=True,
        )

    def _session_for(self, model: str) -> Any:
        family = model.split("_", 1)[0]
        if self.session is None or self.model_family != family:
            if self.session is not None:
                self.session.close()
            self.session = self._new_session()
            self.model_family = family
        return self.session

    @modal.method()
    def run(self, job_id: str, model: str, params: dict[str, Any]) -> dict[str, Any]:
        record = job_store.get(job_id, {})
        if record.get("status") == "cancelled":
            return {"cancelled": True}
        record.update(status="running", started_at=utc_now(), updated_at=utc_now())
        job_store.put(job_id, record)

        try:
            # Fresh containers receive the current Volume snapshot. Warm WanGP
            # workers intentionally keep checkpoint files open, which makes a
            # Modal Volume reload both unnecessary and invalid here.
            session = self._session_for(model)
            settings = session.get_default_settings(model).copy()
            settings.update(validate_job_request(model, params))
            settings["_api"] = {"return_media": False}
            job = session.submit_task(settings, callbacks=JobCallbacks(job_id))
            result = job.result()
            # Make files visible to the CPU download service before publishing a
            # terminal job result that contains their download URLs.
            data_volume.commit()
            outputs = register_outputs(result, job_id)
            payload = serialize_result(result, outputs)
            status = "succeeded" if result.success else "failed"
            record = job_store.get(job_id, record)
            record.update(
                status=status,
                result=payload,
                completed_at=utc_now(),
                updated_at=utc_now(),
            )
            job_store.put(job_id, record)
            return payload
        except BaseException as exc:
            record = job_store.get(job_id, record)
            if record.get("status") != "cancelled":
                record.update(
                    status="failed",
                    result={"success": False, "errors": [{"message": str(exc), "stage": "runtime"}]},
                    completed_at=utc_now(),
                    updated_at=utc_now(),
                )
                job_store.put(job_id, record)
            raise

    @modal.method()
    def list_models(
        self,
        family: str | None = None,
        model_type: str | None = None,
        include_availability: bool = False,
    ) -> list[dict[str, Any]]:
        data_volume.reload()
        session = self._session_for("discovery")
        filters = {}
        if family:
            filters["family"] = family
        if model_type:
            filters["model_type"] = model_type
        return session.list_model_metadata(
            include_availability=include_availability,
            **filters,
        )

    @modal.method()
    def defaults(self, model: str) -> dict[str, Any]:
        data_volume.reload()
        return self._session_for(model).get_default_settings(model)

    @modal.method()
    def schema(self, model: str) -> dict[str, Any] | None:
        data_volume.reload()
        return self._session_for(model).get_model_schema(model)


@app.function(
    image=api_image,
    min_containers=0,
    max_containers=4,
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
    },
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app(requires_proxy_auth=True)
def api():
    import asyncio

    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
    web = FastAPI(
        title="WanGP REST API",
        version="0.1.0",
        description="Asynchronous REST transport for WanGP generation on Modal.",
    )
    catalog = load_catalog()

    class JobRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        model: str = Field(min_length=1)
        params: dict[str, Any] = Field(default_factory=dict)

    @web.get("/health")
    async def health():
        return {
            "service": APP_NAME,
            "ready": True,
            "gpu": GPU_TYPE,
            "max_gpu_containers": MAX_CONTAINERS,
            "wangp_commit": WAN_COMMIT,
            "wan2ai_commit": WAN2AI_COMMIT,
        }

    @web.post("/jobs", status_code=202)
    async def submit_job(body: dict[str, Any]):
        try:
            request = JobRequest.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        try:
            validate_job_request(request.model, request.params)
            validate_data_paths(request.params)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        job_id = str(uuid.uuid4())
        record = {
            "id": job_id,
            "call_id": None,
            "status": "queued",
            "model": request.model.strip(),
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        await job_store.put.aio(job_id, record)
        call = await WanGPWorker().run.spawn.aio(
            job_id,
            request.model.strip(),
            request.params,
        )
        current = await job_store.get.aio(job_id, record)
        current["call_id"] = call.object_id
        await job_store.put.aio(job_id, current)
        return {"id": job_id, "status": "queued"}

    @web.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        record = await job_store.get.aio(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job not found or expired")
        if record["status"] in {"queued", "running"} and record.get("call_id"):
            call = modal.FunctionCall.from_id(record["call_id"])
            try:
                await call.get.aio(timeout=0)
            except TimeoutError:
                pass
            except modal.exception.OutputExpiredError as exc:
                raise HTTPException(status_code=404, detail="job result expired") from exc
            except Exception as exc:
                # Container-start and Modal infrastructure failures can happen
                # before the worker gets a chance to update its own record.
                record = await job_store.get.aio(job_id, record)
                if record["status"] not in {"succeeded", "failed", "cancelled"}:
                    record.update(
                        status="failed",
                        result={
                            "success": False,
                            "errors": [{"message": str(exc), "stage": "modal"}],
                        },
                        completed_at=utc_now(),
                        updated_at=utc_now(),
                    )
                    await job_store.put.aio(job_id, record)
            else:
                record = await job_store.get.aio(job_id, record)
        return record

    @web.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        record = await job_store.get.aio(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job not found or expired")
        if record["status"] == "cancelled":
            return record
        if record["status"] in {"succeeded", "failed"}:
            raise HTTPException(status_code=409, detail="job is already terminal")
        call = modal.FunctionCall.from_id(record["call_id"])
        await call.cancel.aio(terminate_containers=True)
        record.update(status="cancelled", completed_at=utc_now(), updated_at=utc_now())
        await job_store.put.aio(job_id, record)
        return record

    @web.get("/outputs/{output_id}")
    async def get_output(output_id: str):
        output = await job_store.get.aio(f"output:{output_id}")
        if output is None:
            raise HTTPException(status_code=404, detail="output not found or expired")

        record = await job_store.get.aio(output["job_id"])
        if record is None:
            raise HTTPException(status_code=404, detail="job not found or expired")
        if record["status"] != "succeeded":
            raise HTTPException(status_code=409, detail="job has not succeeded")

        path = normalized_absolute_path(output["path"])
        try:
            path.relative_to(DATA_ROOT / "outputs")
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="invalid output path") from exc

        # Reused API containers need an explicit reload to observe commits made
        # by GPU worker containers. Keep this blocking filesystem operation off
        # the ASGI event loop.
        await asyncio.to_thread(data_volume.reload)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="output file not found")
        return FileResponse(
            path,
            media_type=output["media_type"],
            filename=output["filename"],
        )

    @web.get("/models")
    async def models(
        family: str | None = None,
        type: str | None = Query(default=None),
        available: bool = False,
    ):
        if available:
            raise HTTPException(
                status_code=400,
                detail="available=true is not supported by the static CPU catalog",
            )
        return filter_models(catalog["models"], family, type)

    @web.get("/models/{model}/defaults")
    async def defaults(model: str):
        result = catalog["defaults"].get(model)
        if result is None:
            raise HTTPException(status_code=404, detail="unknown model")
        return result

    @web.get("/models/{model}/schema")
    async def schema(model: str):
        result = catalog["schemas"].get(model)
        if result is None:
            raise HTTPException(status_code=404, detail="unknown model")
        return result

    return web
