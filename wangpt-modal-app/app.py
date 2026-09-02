"""GPU worker app for asynchronous WanGP generation on Modal."""

from __future__ import annotations

import faulthandler
import hashlib
import mimetypes
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import modal

from wangpt_common import (
    CATALOG_DICT_NAME,
    DATA_ROOT,
    JOB_DICT_NAME,
    WAN_COMMIT,
    WORKER_APP_NAME,
    normalized_absolute_path,
    utc_now,
    validate_job_request,
)

WAN_ROOT = Path("/opt/Wan2GP")
WAN2AI_ROOT = Path("/opt/Wan2AI")
GENERATED_OUTPUT_ROOT = Path("/tmp/wangp-outputs")
CATALOG_PATH = Path("/opt/wangp-catalog.json")
WAN2AI_COMMIT = "2539c3a87b64fa0f619695f02410fc92c63cba7d"

IMAGE_GPU_TYPE = os.environ.get(
    "WANGP_IMAGE_GPU",
    os.environ.get("WANGP_GPU", "L40S"),
)
VIDEO_GPU_TYPE = os.environ.get("WANGP_VIDEO_GPU", "A100")
IMAGE_MAX_CONTAINERS = int(
    os.environ.get(
        "WANGP_IMAGE_MAX_CONTAINERS",
        os.environ.get("WANGP_MAX_CONTAINERS", "3"),
    )
)
VIDEO_MAX_CONTAINERS = int(os.environ.get("WANGP_VIDEO_MAX_CONTAINERS", "1"))
IMAGE_MEMORY_MB = int(os.environ.get("WANGP_IMAGE_MEMORY_MB", "65536"))
VIDEO_MEMORY_MB = int(os.environ.get("WANGP_VIDEO_MEMORY_MB", "131072"))
IMAGE_WANGP_PROFILE = os.environ.get("WANGP_IMAGE_PROFILE", "4")
VIDEO_WANGP_PROFILE = os.environ.get("WANGP_VIDEO_PROFILE", "4")
MODEL_LOAD_TRACE_INTERVAL_SECONDS = int(
    os.environ.get("WANGP_MODEL_LOAD_TRACE_INTERVAL_SECONDS", "120")
)
SCALEDOWN_WINDOW = 5 * 60
STARTUP_TIMEOUT = 30 * 60

DATA_VOLUME_NAME = "wangp-data"
S3_SECRET_NAME = "studio-s3"
S3_REQUIRED_KEYS = (
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY",
    "S3_ENDPOINT",
    "S3_BUCKET",
    "S3_REGION",
)
S3_OUTPUT_PREFIX = (
    os.environ.get("WANGP_S3_OUTPUT_PREFIX", "runninghub/wangp").strip("/")
    or "runninghub/wangp"
)

CACHE_DIRS = (
    "ckpts",
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
studio_s3_secret = modal.Secret.from_name(
    S3_SECRET_NAME,
    required_keys=list(S3_REQUIRED_KEYS),
)
job_store = modal.Dict.from_name(JOB_DICT_NAME, create_if_missing=True)
catalog_store = modal.Dict.from_name(CATALOG_DICT_NAME, create_if_missing=True)

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
        "python -m pip install 'boto3>=1.40,<2'",
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
    .add_local_python_source("wangpt_common", copy=True)
)

# Share the expensive WanGP build layers while keeping independent Modal images
# for the two worker pools.
image_worker_image = gpu_image.env({"WANGP_WORKER_KIND": "image"})
video_worker_image = gpu_image.env({"WANGP_WORKER_KIND": "video"})

app = modal.App(WORKER_APP_NAME)


def log_runtime_stage(job_id: str, stage: str, **details: Any) -> None:
    """Emit concise, timestamped boundaries around opaque WanGP operations."""
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    message = f"[wangp-runtime] job={job_id} stage={stage} at={utc_now()}"
    if suffix:
        message = f"{message} {suffix}"
    print(message, flush=True)


def canonical_output_path(path: str | os.PathLike[str]) -> Path:
    """Accept only media created in this container's temporary output tree."""
    normalized = normalized_absolute_path(path)
    try:
        normalized.relative_to(GENERATED_OUTPUT_ROOT)
        return normalized
    except ValueError:
        pass

    # WanGP can report its root-relative symlink rather than output_dir. Map
    # that one trusted alias without resolving arbitrary filesystem symlinks.
    try:
        relative = normalized.relative_to(WAN_ROOT / "outputs")
    except ValueError as exc:
        raise ValueError(
            f"WanGP output is not under {GENERATED_OUTPUT_ROOT}: {path}"
        ) from exc
    return GENERATED_OUTPUT_ROOT / relative


def serialize_error(error: Any) -> dict[str, Any]:
    return {
        "message": getattr(error, "message", str(error)),
        "stage": getattr(error, "stage", None),
        "task_index": getattr(error, "task_index", None),
        "task_id": getattr(error, "task_id", None),
    }


def serialize_output(path_string: str) -> dict[str, Any]:
    path = canonical_output_path(path_string)
    stat = path.stat()
    return {
        "filename": path.name,
        "size_bytes": stat.st_size,
        "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    }


def s3_settings_from_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    values = os.environ if environ is None else environ
    missing = [key for key in S3_REQUIRED_KEYS if not values.get(key)]
    if missing:
        raise RuntimeError(
            f"Modal secret {S3_SECRET_NAME} is missing required keys: {', '.join(missing)}"
        )
    return {key: values[key] for key in S3_REQUIRED_KEYS}


def create_s3_client(settings: dict[str, str]):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        aws_access_key_id=settings["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=settings["S3_SECRET_ACCESS_KEY"],
        endpoint_url=settings["S3_ENDPOINT"],
        region_name=settings["S3_REGION"],
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


def upload_output_artifacts(
    result: Any,
    job_id: str,
    client: Any,
    bucket: str,
    prefix: str = S3_OUTPUT_PREFIX,
) -> list[dict[str, Any]]:
    """Upload every generated file and return only verified S3 references."""
    outputs: list[dict[str, Any]] = []
    for index, path_string in enumerate(result.generated_files):
        path = canonical_output_path(path_string)
        metadata = serialize_output(str(path))
        digest, size = sha256_file(path)
        key = f"{prefix}/{job_id}/{index:03d}-{path.name}"
        client.upload_file(
            str(path),
            bucket,
            key,
            ExtraArgs={
                "ContentType": metadata["media_type"],
                "Metadata": {"sha256": digest},
            },
        )
        head = client.head_object(Bucket=bucket, Key=key)
        if (
            int(head.get("ContentLength", -1)) != size
            or head.get("Metadata", {}).get("sha256") != digest
        ):
            raise RuntimeError(f"uploaded output verification failed for {key}")
        outputs.append(
            {
                "storage": "s3",
                "bucket": bucket,
                "key": key,
                "uri": f"s3://{bucket}/{key}",
                **metadata,
                "sha256": digest,
            }
        )
    return outputs


def remove_local_outputs(result: Any) -> None:
    """Remove generated media after every S3 object has been verified."""
    for path_string in result.generated_files:
        canonical_output_path(path_string).unlink(missing_ok=True)


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
    def __init__(
        self,
        job_id: str,
        on_first_progress: Callable[[], None] | None = None,
    ):
        self.job_id = job_id
        self.on_first_progress = on_first_progress

    def on_progress(self, progress: Any) -> None:
        if self.on_first_progress is not None:
            callback = self.on_first_progress
            self.on_first_progress = None
            callback()
        record = job_store.get(self.job_id, {})
        record["progress"] = _progress_dict(progress)
        record["updated_at"] = utc_now()
        job_store.put(self.job_id, record)


class WanGPRuntime:
    def __init__(self, profile: str) -> None:
        self.profile = profile
        self.session = None
        self.model_family = None

    def initialize(self) -> None:
        _ensure_cache_layout()
        GENERATED_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        for name in ("ckpts", "settings", "loras"):
            _force_directory_symlink(WAN_ROOT / name, DATA_ROOT / name)
        # Generated images and videos are deliberately container-local. They
        # are verified in S3 and removed before a terminal result is published.
        _force_directory_symlink(WAN_ROOT / "outputs", GENERATED_OUTPUT_ROOT)
        data_volume.commit()

    def _new_session(self) -> Any:
        from shared.api import init

        return init(
            root=WAN_ROOT,
            output_dir=GENERATED_OUTPUT_ROOT,
            cli_args=["--profile", self.profile, "--attention", "sdpa"],
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
            # Model construction has previously held the Python process long
            # enough to starve Modal's heartbeat thread. Repeating faulthandler
            # dumps make that boundary observable without tracing normal
            # denoising or changing WanGP's execution semantics.
            trace_armed = False

            def cancel_load_trace(reason: str) -> None:
                nonlocal trace_armed
                if not trace_armed:
                    return
                faulthandler.cancel_dump_traceback_later()
                trace_armed = False
                log_runtime_stage(
                    job_id,
                    "load_trace_cancelled",
                    model=model,
                    reason=reason,
                )

            try:
                if MODEL_LOAD_TRACE_INTERVAL_SECONDS > 0:
                    faulthandler.dump_traceback_later(
                        MODEL_LOAD_TRACE_INTERVAL_SECONDS,
                        repeat=True,
                        file=sys.stderr,
                        exit=False,
                    )
                    trace_armed = True
                log_runtime_stage(
                    job_id,
                    "session_start",
                    model=model,
                    profile=self.profile,
                )
                session = self._session_for(model)
                log_runtime_stage(job_id, "session_ready", model=model)
                settings = session.get_default_settings(model).copy()
                settings.update(validate_job_request(model, params))
                settings["_api"] = {"return_media": False}
                log_runtime_stage(job_id, "submit_start", model=model)
                job = session.submit_task(
                    settings,
                    callbacks=JobCallbacks(
                        job_id,
                        on_first_progress=lambda: cancel_load_trace(
                            "first_progress"
                        ),
                    ),
                )
                log_runtime_stage(job_id, "submit_ready", model=model)
                log_runtime_stage(job_id, "result_wait_start", model=model)
                result = job.result()
            finally:
                cancel_load_trace("task_boundary_exit")
            log_runtime_stage(job_id, "result_ready", model=model, success=result.success)
            s3_settings = s3_settings_from_env()
            s3_client = create_s3_client(s3_settings)
            log_runtime_stage(job_id, "s3_upload_start", model=model)
            outputs = upload_output_artifacts(
                result,
                job_id,
                s3_client,
                s3_settings["S3_BUCKET"],
            )
            log_runtime_stage(
                job_id,
                "s3_upload_verified",
                model=model,
                outputs=len(outputs),
            )
            remove_local_outputs(result)
            # Persist downloaded model/cache state only. Generated media never
            # enters the mounted Volume.
            data_volume.commit()
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
                    result={
                        "success": False,
                        "errors": [{"message": str(exc), "stage": "runtime"}],
                    },
                    completed_at=utc_now(),
                    updated_at=utc_now(),
                )
                job_store.put(job_id, record)
            raise


@app.cls(
    image=image_worker_image,
    gpu=IMAGE_GPU_TYPE,
    memory=IMAGE_MEMORY_MB,
    min_containers=0,
    max_containers=IMAGE_MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    startup_timeout=STARTUP_TIMEOUT,
    timeout=24 * 60 * 60,
    volumes={str(DATA_ROOT): data_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),
        studio_s3_secret,
    ],
)
class WanGPImageWorker:
    @modal.enter()
    def initialize(self) -> None:
        self.runtime = WanGPRuntime(IMAGE_WANGP_PROFILE)
        self.runtime.initialize()

    @modal.method()
    def run(self, job_id: str, model: str, params: dict[str, Any]) -> dict[str, Any]:
        return self.runtime.run(job_id, model, params)


@app.cls(
    image=video_worker_image,
    gpu=VIDEO_GPU_TYPE,
    memory=VIDEO_MEMORY_MB,
    min_containers=0,
    max_containers=VIDEO_MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    startup_timeout=STARTUP_TIMEOUT,
    timeout=24 * 60 * 60,
    volumes={str(DATA_ROOT): data_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),
        studio_s3_secret,
    ],
)
class WanGPVideoWorker:
    @modal.enter()
    def initialize(self) -> None:
        self.runtime = WanGPRuntime(VIDEO_WANGP_PROFILE)
        self.runtime.initialize()

    @modal.method()
    def run(self, job_id: str, model: str, params: dict[str, Any]) -> dict[str, Any]:
        return self.runtime.run(job_id, model, params)


@app.function(image=gpu_image, min_containers=0, max_containers=1)
def publish_catalog() -> dict[str, Any]:
    """Publish this worker revision's baked catalog for the local CLI."""
    catalog = load_catalog()
    catalog_store.put(WAN_COMMIT, catalog)
    return {
        "catalog_key": WAN_COMMIT,
        "models": len(catalog.get("models", [])),
        "wan_commit": WAN_COMMIT,
    }
