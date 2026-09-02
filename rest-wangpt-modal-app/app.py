"""Asynchronous WanGP generation and Fizgig training REST API on Modal.

WanGP runs in this app. Fizgig training is dispatched to the independently
deployed ``fizgig-modal-app``. The protected endpoint returns pollable job IDs
for both domains.
"""

from __future__ import annotations

import faulthandler
import hashlib
import mimetypes
import os
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import modal


APP_NAME = "wangp-rest"
WAN_ROOT = Path("/opt/Wan2GP")
WAN2AI_ROOT = Path("/opt/Wan2AI")
DATA_ROOT = Path("/data")
GENERATED_OUTPUT_ROOT = Path("/tmp/wangp-outputs")
CATALOG_PATH = Path("/opt/wangp-catalog.json")
WAN_COMMIT = "92f56e5ee7227d490f6d85281c019e4c4e2dc393"
WAN2AI_COMMIT = "2539c3a87b64fa0f619695f02410fc92c63cba7d"

IMAGE_GPU_TYPE = os.environ.get(
    "WANGP_IMAGE_GPU",
    os.environ.get("WANGP_GPU", "L40S"),
)
VIDEO_GPU_TYPE = os.environ.get("WANGP_VIDEO_GPU", "H100")
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

# Backward-compatible health aliases for clients that predate split workers.
GPU_TYPE = IMAGE_GPU_TYPE
MAX_CONTAINERS = IMAGE_MAX_CONTAINERS

DATA_VOLUME_NAME = "wangp-data"
JOB_DICT_NAME = "wangp-rest-jobs"
TRAINING_JOB_DICT_NAME = "fizgig-rest-jobs"
FIZGIG_JOB_DICT_NAME = "fizgig-modal-jobs"
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

FIZGIG_APP_NAME = os.environ.get("FIZGIG_MODAL_APP_NAME", "fizgig-modal-app")
FIZGIG_RUN_FUNCTION = "run_training"
FIZGIG_PAUSE_FUNCTION = "request_pause"
SUPPORTED_TRAINING_FAMILIES = ("minimax_h3", "krea2")
SUPPORTED_TRAINING_PRESETS = {
    "h3_character_fast": "minimax_h3",
    "h3_character_quality": "minimax_h3",
    "krea2_defaults": "krea2",
    "krea2_ultra_fast": "krea2",
}
TRAINING_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
_SAFE_TRAINING_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ALLOWED_TRAINING_REQUEST_KEYS = {
    "family",
    "dataset",
    "output_name",
    "preset",
    "trigger_word",
    "epochs",
    "resume_from",
}

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
training_job_store = modal.Dict.from_name(
    TRAINING_JOB_DICT_NAME,
    create_if_missing=True,
)
fizgig_job_store = modal.Dict.from_name(
    FIZGIG_JOB_DICT_NAME,
    create_if_missing=True,
)

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
)

# Share the expensive WanGP build layers while keeping independent Modal images
# for the two worker pools.
image_worker_image = gpu_image.env({"WANGP_WORKER_KIND": "image"})
video_worker_image = gpu_image.env({"WANGP_WORKER_KIND": "video"})

# Discovery is served by a CPU container from the catalog baked into this
# image. Sharing the image with GPU workers avoids maintaining two copies of
# WanGP and guarantees that discovery matches the generation runtime exactly.
api_image = gpu_image

app = modal.App(APP_NAME)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_runtime_stage(job_id: str, stage: str, **details: Any) -> None:
    """Emit concise, timestamped boundaries around opaque WanGP operations."""
    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    message = f"[wangp-runtime] job={job_id} stage={stage} at={utc_now()}"
    if suffix:
        message = f"{message} {suffix}"
    print(message, flush=True)


def normalized_absolute_path(path: str | os.PathLike[str]) -> Path:
    """Normalize traversal without resolving Modal Volume mount internals."""
    return Path(os.path.abspath(os.path.normpath(path)))


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


def _training_component(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    value = value.strip()
    if not _SAFE_TRAINING_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            f"{field} must contain only letters, numbers, '.', '_' or '-' and "
            "must not contain a path"
        )
    return value


def _training_int(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return value


def validate_training_request(
    request: dict[str, Any],
    *,
    allow_resume: bool = False,
) -> dict[str, Any]:
    """Validate the public Fizgig contract without translating it to CLI flags."""
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    unknown = sorted(set(request) - _ALLOWED_TRAINING_REQUEST_KEYS)
    if unknown:
        raise ValueError(f"unsupported request fields: {', '.join(unknown)}")
    if "resume_from" in request and not allow_resume:
        raise ValueError("resume_from is controlled by the resume endpoint")

    missing = sorted(
        field
        for field in ("family", "dataset", "output_name", "preset")
        if field not in request
    )
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")

    family = request["family"]
    if family not in SUPPORTED_TRAINING_FAMILIES:
        choices = ", ".join(SUPPORTED_TRAINING_FAMILIES)
        raise ValueError(f"family must be one of: {choices}")
    dataset = _training_component(request["dataset"], "dataset")
    output_name = _training_component(request["output_name"], "output_name")
    preset = request["preset"]
    if preset not in SUPPORTED_TRAINING_PRESETS:
        choices = ", ".join(sorted(SUPPORTED_TRAINING_PRESETS))
        raise ValueError(f"preset must be one of: {choices}")
    if SUPPORTED_TRAINING_PRESETS[preset] != family:
        raise ValueError(f"preset {preset!r} does not support family {family!r}")

    epochs = request.get("epochs")
    if epochs is not None:
        epochs = _training_int(epochs, "epochs", 1, 500)

    trigger_word = request.get("trigger_word")
    if trigger_word is not None:
        trigger_word = _training_component(trigger_word, "trigger_word")

    resume_from = request.get("resume_from")
    if resume_from is not None and resume_from != "latest":
        resume_from = _training_component(resume_from, "resume_from")
        if not resume_from.endswith("-state"):
            raise ValueError(
                "resume_from must be 'latest' or a state-directory basename"
            )

    normalized = {
        "family": family,
        "dataset": dataset,
        "output_name": output_name,
        "preset": preset,
    }
    if trigger_word is not None:
        normalized["trigger_word"] = trigger_word
    if epochs is not None:
        normalized["epochs"] = epochs
    if resume_from is not None:
        normalized["resume_from"] = resume_from
    return normalized


def merge_training_record(
    public_record: dict[str, Any],
    internal_record: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reconcile worker-owned lifecycle fields into the public job record."""
    merged = dict(public_record)
    if not internal_record:
        return merged
    internal_status = internal_record.get("status")
    if internal_status in {"queued", "running"} | TRAINING_TERMINAL_STATUSES:
        merged["status"] = internal_status
    for key in (
        "progress",
        "result",
        "error",
        "pause_requested",
        "started_at",
        "completed_at",
        "updated_at",
    ):
        if key in internal_record:
            merged[key] = internal_record[key]
    return merged


def public_training_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return the stable status resource without internal diagnostic logs."""
    public = dict(record)
    public.pop("call_id", None)
    public.pop("pause_requested", None)
    progress = public.get("progress")
    if isinstance(progress, dict):
        public_progress = dict(progress)
        public_progress.pop("log_tail", None)
        public["progress"] = public_progress
    return public


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


def resolve_generation_kind(
    catalog: dict[str, Any],
    model: str,
    requested_kind: str | None = None,
    params: dict[str, Any] | None = None,
) -> Literal["image", "video", "audio"]:
    """Resolve and validate the worker route from WanGP model metadata."""
    model = model.strip()
    metadata = next(
        (
            item
            for item in catalog.get("models", [])
            if item.get("model_type") == model
        ),
        None,
    )
    if metadata is None:
        raise ValueError(f"unknown model: {model}")

    raw_outputs = metadata.get("main_output", [])
    if isinstance(raw_outputs, str):
        raw_outputs = [raw_outputs]
    outputs = [
        output
        for output in raw_outputs
        if output in {"image", "video", "audio"}
    ]
    if not outputs:
        raise ValueError(f"model {model!r} does not declare a routable output type")

    if requested_kind is not None:
        if requested_kind not in {"image", "video", "audio"}:
            raise ValueError("kind must be one of: image, video, audio")
        if requested_kind not in outputs:
            choices = ", ".join(outputs)
            raise ValueError(
                f"model {model!r} supports {choices}, not {requested_kind}"
            )
        return requested_kind

    if len(outputs) == 1:
        return outputs[0]

    # Models that switch between image and video use image_mode. Overlay the
    # request on baked defaults to preserve WanGP's native behavior.
    settings = dict(catalog.get("defaults", {}).get(model, {}))
    settings.update(params or {})
    if "image" in outputs and "video" in outputs:
        return "image" if settings.get("image_mode", 0) else "video"
    return outputs[0]


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
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
    web = FastAPI(
        title="WanGP and Fizgig REST API",
        version="0.4.0",
        description=(
            "Asynchronous REST transport for WanGP generation and Fizgig "
            "training on Modal."
        ),
    )
    catalog = load_catalog()

    class JobRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        model: str = Field(min_length=1)
        kind: Literal["image", "video", "audio"] | None = None
        params: dict[str, Any] = Field(default_factory=dict)

    async def _load_training_record(job_id: str) -> dict[str, Any]:
        record = await training_job_store.get.aio(job_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail="training job not found or expired",
            )
        internal = await fizgig_job_store.get.aio(job_id)
        merged = merge_training_record(record, internal)
        if merged != record:
            await training_job_store.put.aio(job_id, merged)
        return merged

    async def _spawn_training_job(
        request_body: dict[str, Any],
        *,
        allow_resume: bool = False,
        resumed_from: str | None = None,
    ) -> dict[str, Any]:
        try:
            request = validate_training_request(
                request_body,
                allow_resume=allow_resume,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        job_id = str(uuid.uuid4())
        timestamp = utc_now()
        record = {
            "id": job_id,
            "call_id": None,
            "status": "queued",
            "request": request,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        if resumed_from is not None:
            record["resumed_from"] = resumed_from
        internal_record = {
            "id": job_id,
            "status": "queued",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        await training_job_store.put.aio(job_id, record)
        await fizgig_job_store.put.aio(job_id, internal_record)
        try:
            run_training = modal.Function.from_name(
                FIZGIG_APP_NAME,
                FIZGIG_RUN_FUNCTION,
            )
            call = await run_training.spawn.aio(job_id, request)
        except Exception as exc:
            failure_time = utc_now()
            error = {
                "message": str(exc),
                "type": type(exc).__name__,
                "stage": "dispatch",
            }
            record.update(
                status="failed",
                error=error,
                completed_at=failure_time,
                updated_at=failure_time,
            )
            internal_record.update(
                status="failed",
                error=error,
                completed_at=failure_time,
                updated_at=failure_time,
            )
            await training_job_store.put.aio(job_id, record)
            await fizgig_job_store.put.aio(job_id, internal_record)
            raise HTTPException(
                status_code=503,
                detail=f"Fizgig training app is unavailable: {exc}",
            ) from exc

        record["call_id"] = call.object_id
        record["updated_at"] = utc_now()
        await training_job_store.put.aio(job_id, record)
        return record

    @web.get("/health")
    async def health():
        return {
            "service": APP_NAME,
            "ready": True,
            "gpu": GPU_TYPE,
            "max_gpu_containers": MAX_CONTAINERS,
            "generation_workers": {
                "image": {
                    "gpu": IMAGE_GPU_TYPE,
                    "max_containers": IMAGE_MAX_CONTAINERS,
                    "memory_mb": IMAGE_MEMORY_MB,
                    "wangp_profile": IMAGE_WANGP_PROFILE,
                },
                "video": {
                    "gpu": VIDEO_GPU_TYPE,
                    "max_containers": VIDEO_MAX_CONTAINERS,
                    "memory_mb": VIDEO_MEMORY_MB,
                    "wangp_profile": VIDEO_WANGP_PROFILE,
                },
            },
            "wangp_commit": WAN_COMMIT,
            "wan2ai_commit": WAN2AI_COMMIT,
            "fizgig_app": FIZGIG_APP_NAME,
            "training_families": list(SUPPORTED_TRAINING_FAMILIES),
        }

    @web.post("/training/jobs", status_code=202)
    async def submit_training_job(body: dict[str, Any]):
        record = await _spawn_training_job(body)
        return {"id": record["id"], "status": record["status"]}

    @web.get("/training/jobs/{job_id}")
    async def get_training_job(job_id: str):
        record = await _load_training_record(job_id)
        if record["status"] in {"queued", "running"} and record.get("call_id"):
            call = modal.FunctionCall.from_id(record["call_id"])
            try:
                returned_record = await call.get.aio(timeout=0)
            except TimeoutError:
                pass
            except modal.exception.OutputExpiredError as exc:
                raise HTTPException(
                    status_code=404,
                    detail="training job result expired",
                ) from exc
            except Exception as exc:
                # A worker-side failure normally writes its own record first.
                # If startup failed before that happened, preserve the failure
                # in the public and internal stores here.
                record = await _load_training_record(job_id)
                if record["status"] not in TRAINING_TERMINAL_STATUSES:
                    timestamp = utc_now()
                    error = {
                        "message": str(exc),
                        "type": type(exc).__name__,
                        "stage": "modal",
                    }
                    record.update(
                        status="failed",
                        error=error,
                        completed_at=timestamp,
                        updated_at=timestamp,
                    )
                    internal = await fizgig_job_store.get.aio(job_id, {"id": job_id})
                    internal.update(
                        status="failed",
                        error=error,
                        completed_at=timestamp,
                        updated_at=timestamp,
                    )
                    await fizgig_job_store.put.aio(job_id, internal)
                    await training_job_store.put.aio(job_id, record)
            else:
                internal = await fizgig_job_store.get.aio(job_id)
                record = merge_training_record(record, internal or returned_record)
                await training_job_store.put.aio(job_id, record)
        return public_training_record(record)

    @web.post("/training/jobs/{job_id}/pause")
    async def pause_training_job(job_id: str):
        record = await _load_training_record(job_id)
        if record["status"] != "running":
            raise HTTPException(
                status_code=409,
                detail="only a running training job can be paused",
            )
        try:
            request_pause = modal.Function.from_name(
                FIZGIG_APP_NAME,
                FIZGIG_PAUSE_FUNCTION,
            )
            internal = await request_pause.remote.aio(job_id)
        except Exception as exc:
            current = await _load_training_record(job_id)
            if current["status"] != "running":
                raise HTTPException(
                    status_code=409,
                    detail="training job is no longer running",
                ) from exc
            raise HTTPException(
                status_code=503,
                detail=f"Fizgig pause operation is unavailable: {exc}",
            ) from exc
        record = merge_training_record(record, internal)
        await training_job_store.put.aio(job_id, record)
        return public_training_record(record)

    @web.post("/training/jobs/{job_id}/resume", status_code=202)
    async def resume_training_job(job_id: str):
        record = await _load_training_record(job_id)
        result = record.get("result") or {}
        if record["status"] != "succeeded" or not result.get("paused"):
            raise HTTPException(
                status_code=409,
                detail="only a successfully paused training job can be resumed",
            )
        request = dict(record["request"])
        request["resume_from"] = result.get("resume_from") or "latest"
        resumed = await _spawn_training_job(
            request,
            allow_resume=True,
            resumed_from=job_id,
        )
        return {
            "id": resumed["id"],
            "status": resumed["status"],
            "resumed_from": job_id,
        }

    @web.post("/training/jobs/{job_id}/cancel")
    async def cancel_training_job(job_id: str):
        record = await _load_training_record(job_id)
        if record["status"] == "cancelled":
            return public_training_record(record)
        if record["status"] in {"succeeded", "failed"}:
            raise HTTPException(
                status_code=409,
                detail="training job is already terminal",
            )
        if record.get("call_id"):
            try:
                call = modal.FunctionCall.from_id(record["call_id"])
                await call.cancel.aio(terminate_containers=True)
            except Exception as exc:
                current = await _load_training_record(job_id)
                if current["status"] in TRAINING_TERMINAL_STATUSES:
                    return public_training_record(current)
                raise HTTPException(
                    status_code=503,
                    detail=f"training cancellation is unavailable: {exc}",
                ) from exc

            # Completion can win the race with cancellation. Never overwrite a
            # worker-authored successful or failed terminal record in that case.
            internal = await fizgig_job_store.get.aio(job_id)
            if internal and internal.get("status") in {"succeeded", "failed"}:
                record = merge_training_record(record, internal)
                await training_job_store.put.aio(job_id, record)
                return public_training_record(record)

        timestamp = utc_now()
        record.update(
            status="cancelled",
            completed_at=timestamp,
            updated_at=timestamp,
        )
        internal = await fizgig_job_store.get.aio(job_id, {"id": job_id})
        internal.update(
            status="cancelled",
            completed_at=timestamp,
            updated_at=timestamp,
        )
        await fizgig_job_store.put.aio(job_id, internal)
        await training_job_store.put.aio(job_id, record)
        return public_training_record(record)

    @web.post("/jobs", status_code=202)
    async def submit_job(body: dict[str, Any]):
        try:
            request = JobRequest.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        try:
            model = request.model.strip()
            validate_job_request(model, request.params)
            validate_data_paths(request.params)
            kind = resolve_generation_kind(
                catalog,
                model,
                request.kind,
                request.params,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        job_id = str(uuid.uuid4())
        record = {
            "id": job_id,
            "call_id": None,
            "status": "queued",
            "kind": kind,
            "model": model,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        await job_store.put.aio(job_id, record)
        worker = WanGPVideoWorker if kind == "video" else WanGPImageWorker
        call = await worker().run.spawn.aio(
            job_id,
            model,
            request.params,
        )
        current = await job_store.get.aio(job_id, record)
        current["call_id"] = call.object_id
        await job_store.put.aio(job_id, current)
        return {"id": job_id, "status": "queued", "kind": kind}

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
