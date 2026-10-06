"""GPU worker app for asynchronous WanGP generation on Modal."""

from __future__ import annotations

import faulthandler
import hashlib
import mimetypes
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

import modal

from h3_refmod import REFMOD_COMMIT, REFMOD_ROOT, install_refmod_hooks
from h3_refmod_numbered import REPORTS_ATTR, prepare_numbered_job

from h3_latent import (
    PLUGIN_COMMIT,
    PLUGIN_KEY,
    PLUGIN_ROOT,
    include_latent_outputs,
    install_headless_hooks,
    native_task,
    prepare_latent_job,
)

from wangpt_common import (
    CATALOG_DICT_NAME,
    DATA_ROOT,
    JOB_DICT_NAME,
    SINGULARITY_MODEL,
    WAN_COMMIT,
    WORKER_APP_NAME,
    normalized_absolute_path,
    utc_now,
    validate_job_request,
)

WAN_ROOT = Path("/opt/Wan2GP")
WAN2AI_ROOT = Path("/opt/Wan2AI")
GENERATED_OUTPUT_ROOT = Path("/tmp/wangp-outputs")
S3_INPUT_ROOT = Path("/tmp/wangp-inputs")
CATALOG_PATH = Path("/opt/wangp-catalog.json")
WAN2AI_COMMIT = "2539c3a87b64fa0f619695f02410fc92c63cba7d"

IMAGE_GPU_TYPE = os.environ.get(
    "WANGP_IMAGE_GPU",
    os.environ.get("WANGP_GPU", "H100"),
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
SINGULARITY_MEMORY_MB = int(os.environ.get("WANGP_SINGULARITY_MEMORY_MB", "131072"))
SINGULARITY_MAX_CONTAINERS = int(os.environ.get("WANGP_SINGULARITY_MAX_CONTAINERS", "1"))
SINGULARITY_PROFILE = os.environ.get("WANGP_SINGULARITY_PROFILE", "1")
SINGULARITY_SCALEDOWN_WINDOW = int(os.environ.get("WANGP_SINGULARITY_SCALEDOWN_WINDOW", "300"))

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
S3_INPUT_MAX_BYTES = int(
    os.environ.get("WANGP_S3_INPUT_MAX_BYTES", str(128 * 1024 * 1024))
)
S3_IMAGE_INPUT_KEYS = (
    "image_start",
    "image_end",
    "image_guide",
    "image_mask",
    "image_refs",
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
    .add_local_dir(
        str(Path(__file__).with_name("finetunes")),
        remote_path=str(WAN_ROOT / "finetunes"),
        copy=True,
    )
    .run_commands(
        f"git clone https://github.com/g3n3rativ3/wan2gp-h3-latent-continue.git {PLUGIN_ROOT}",
        f"cd {PLUGIN_ROOT} && git checkout {PLUGIN_COMMIT}",
    )
    .run_commands(
        f"git clone https://github.com/g3n3rativ3/MiniMaxH3Mod-for-WanGP.git {REFMOD_ROOT}",
        f"cd {REFMOD_ROOT} && git checkout {REFMOD_COMMIT}",
    )
    .add_local_file(
        str(Path(__file__).with_name("h3_refmod.py")),
        remote_path=str(WAN_ROOT / "h3_refmod.py"),
        copy=True,
    )
    .add_local_file(
        str(Path(__file__).with_name("h3_refmod_numbered.py")),
        remote_path=str(WAN_ROOT / "h3_refmod_numbered.py"),
        copy=True,
    )
    .add_local_file(
        str(Path(__file__).with_name("patches") / "h3-latent-mux.patch"),
        remote_path="/opt/h3-latent-mux.patch",
        copy=True,
    )
    .run_commands(
        f"cd {WAN_ROOT} && git apply --check /opt/h3-latent-mux.patch && git apply /opt/h3-latent-mux.patch",
    )
    .add_local_python_source("wangpt_common", "h3_latent", "h3_mux", copy=True)
    .add_local_file(
        str(Path(__file__).with_name("patches") / "h3-latent-native.patch"),
        remote_path="/opt/h3-latent-native.patch",
        copy=True,
    )
    .run_commands(
        f"cd {PLUGIN_ROOT} && git apply --check /opt/h3-latent-native.patch && git apply /opt/h3-latent-native.patch",
    )
    .add_local_file(
        str(Path(__file__).with_name("tests") / "test_h3_refmod_numbered_runtime.py"),
        remote_path="/opt/test_h3_refmod_numbered_runtime.py",
        copy=True,
    )
    .run_commands("python /opt/test_h3_refmod_numbered_runtime.py")
    .run_commands("python /opt/generate_catalog.py")
    .add_local_file(
        str(Path(__file__).with_name("patches") / "qwen21-fused-checkpoint.patch"),
        remote_path="/opt/qwen21-fused-checkpoint.patch",
        copy=True,
    )
    .run_commands(
        f"cd {WAN_ROOT} && git apply --check /opt/qwen21-fused-checkpoint.patch && git apply /opt/qwen21-fused-checkpoint.patch",
    )
    .add_local_file(
        str(Path(__file__).with_name("tests") / "test_qwen21_loader_runtime.py"),
        remote_path="/opt/test_qwen21_loader_runtime.py",
        copy=True,
    )
    .run_commands("python /opt/test_qwen21_loader_runtime.py")
)

# Share the expensive WanGP build layers while keeping independent Modal images
# for the two worker pools.
image_worker_image = gpu_image.env({"WANGP_WORKER_KIND": "image"})
video_worker_image = gpu_image.env({"WANGP_WORKER_KIND": "video"})
# Keep the snapshot helper/image layer exclusive to this experimental pool.
singularity_worker_image = gpu_image.add_local_python_source("h3_snapshot", copy=True).env(
    {"WANGP_WORKER_KIND": "singularity-snapshot"}
)

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


def parse_s3_uri(uri: str, configured_bucket: str) -> tuple[str, str]:
    """Parse one bucket-scoped S3 URI without accepting URL decorations."""
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError("S3 image input must use s3://BUCKET/KEY")
    if (
        parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
        or parsed.port
    ):
        raise ValueError(
            "S3 image input URI must not contain credentials, query, or fragment"
        )
    bucket = parsed.hostname or ""
    if bucket != configured_bucket:
        raise ValueError("S3 image input bucket does not match configured S3_BUCKET")
    key = unquote(parsed.path.removeprefix("/"))
    if not key or key.endswith("/") or "\x00" in key:
        raise ValueError("S3 image input URI must contain an object key")
    return bucket, key


def _s3_input_destination(input_dir: Path, index: int, uri: str, key: str) -> Path:
    suffix = Path(key).suffix.lower()
    if not (
        suffix.startswith(".")
        and 1 < len(suffix) <= 12
        and suffix[1:].isascii()
        and suffix[1:].isalnum()
    ):
        suffix = ""
    uri_digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
    return input_dir / f"{index:03d}-{uri_digest}{suffix}"


def download_s3_input(
    client: Any,
    uri: str,
    configured_bucket: str,
    destination: Path,
    max_bytes: int = S3_INPUT_MAX_BYTES,
) -> Path:
    """Download and verify one S3 input before exposing it to WanGP."""
    bucket, key = parse_s3_uri(uri, configured_bucket)
    head = client.head_object(Bucket=bucket, Key=key)
    expected_size = int(head.get("ContentLength", -1))
    if expected_size <= 0:
        raise ValueError("S3 image input must not be empty")
    if expected_size > max_bytes:
        raise ValueError(
            f"S3 image input exceeds the {max_bytes}-byte download limit"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with partial.open("wb") as stream:
            client.download_fileobj(bucket, key, stream)
        digest, actual_size = sha256_file(partial)
        if actual_size != expected_size:
            raise ValueError("S3 image input failed size verification")
        expected_digest = head.get("Metadata", {}).get("sha256")
        if expected_digest and digest != expected_digest:
            raise ValueError("S3 image input failed SHA-256 verification")
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return destination


def materialize_s3_image_inputs(
    params: dict[str, Any],
    job_id: str,
    client: Any,
    configured_bucket: str,
    root: Path = S3_INPUT_ROOT,
) -> tuple[dict[str, Any], Path | None, int]:
    """Download image inputs and the original video/checkpoint for continuation."""
    settings = dict(params)
    input_dir = root / hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:32]
    download_count = 0

    def materialize(value: Any) -> Any:
        nonlocal download_count
        if isinstance(value, list):
            return [materialize(item) for item in value]
        if not (isinstance(value, str) and value.startswith("s3://")):
            return value
        _, key = parse_s3_uri(value, configured_bucket)
        destination = _s3_input_destination(
            input_dir,
            download_count,
            value,
            key,
        )
        download_count += 1
        return str(
            download_s3_input(
                client,
                value,
                configured_bucket,
                destination,
            )
        )

    try:
        for key in S3_IMAGE_INPUT_KEYS:
            if key in settings:
                settings[key] = materialize(settings[key])
        if "video_source" in settings:
            settings["video_source"] = materialize(settings["video_source"])
        if PLUGIN_KEY in settings.get("plugin_data", {}):
            data = dict(settings["plugin_data"])
            options = dict(data[PLUGIN_KEY])
            if "latent_path" in options:
                options["latent_path"] = materialize(options["latent_path"])
            data[PLUGIN_KEY] = options
            settings["plugin_data"] = data
    except BaseException:
        shutil.rmtree(input_dir, ignore_errors=True)
        raise
    return settings, input_dir if download_count else None, download_count


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
        progress_observer: Callable[[Any], None] | None = None,
    ):
        self.job_id = job_id
        self.on_first_progress = on_first_progress
        self.progress_observer = progress_observer

    def on_progress(self, progress: Any) -> None:
        if self.progress_observer is not None:
            self.progress_observer(progress)
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
        S3_INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        for name in ("ckpts", "settings", "loras"):
            _force_directory_symlink(WAN_ROOT / name, DATA_ROOT / name)
        # Generated images and videos are deliberately container-local. They
        # are verified in S3 and removed before a terminal result is published.
        _force_directory_symlink(WAN_ROOT / "outputs", GENERATED_OUTPUT_ROOT)
        data_volume.commit()

    def _new_session(self) -> Any:
        from shared.api import init

        session = init(
            root=WAN_ROOT,
            output_dir=GENERATED_OUTPUT_ROOT,
            cli_args=["--profile", self.profile, "--attention", "sdpa"],
            console_output=True,
        )
        # Install the identity loader before any model is cached by this process,
        # including ordinary jobs followed by a latent-enabled job on a warm worker.
        install_headless_hooks(sys.modules["wgp"])
        install_refmod_hooks(sys.modules["wgp"])
        return session

    def _session_for(self, model: str) -> Any:
        family = model.split("_", 1)[0]
        if self.session is None or self.model_family != family:
            if self.session is not None:
                self.session.close()
            self.session = self._new_session()
            self.model_family = family
        return self.session

    def run(
        self, job_id: str, model: str, params: dict[str, Any], *,
        progress_observer: Callable[[Any], None] | None = None,
    ) -> dict[str, Any]:
        record = job_store.get(job_id, {})
        if record.get("status") == "cancelled":
            return {"cancelled": True}
        record.update(status="running", started_at=utc_now(), updated_at=utc_now())
        job_store.put(job_id, record)

        input_dir: Path | None = None
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
                s3_settings = s3_settings_from_env()
                s3_client = create_s3_client(s3_settings)
                materialized_params, input_dir, input_count = (
                    materialize_s3_image_inputs(
                        validate_job_request(model, params),
                        job_id,
                        s3_client,
                        s3_settings["S3_BUCKET"],
                    )
                )
                if input_count:
                    log_runtime_stage(
                        job_id,
                        "s3_inputs_ready",
                        count=input_count,
                        model=model,
                    )
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
                settings.update(materialized_params)
                prepare_numbered_job(sys.modules["wgp"], settings)
                prepare_latent_job(sys.modules["wgp"], settings)
                log_runtime_stage(job_id, "submit_start", model=model)
                job = session.submit_task(
                    native_task(settings),
                    callbacks=JobCallbacks(
                        job_id,
                        progress_observer=progress_observer,
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
            include_latent_outputs(result, settings)
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
            reference_maps = getattr(sys.modules["wgp"], REPORTS_ATTR, [])
            if reference_maps:
                payload["refmod_reference_maps"] = reference_maps
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
        finally:
            if input_dir is not None:
                shutil.rmtree(input_dir, ignore_errors=True)


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


@app.cls(
    image=singularity_worker_image,
    gpu="H100",
    memory=SINGULARITY_MEMORY_MB,
    min_containers=0,
    max_containers=SINGULARITY_MAX_CONTAINERS,
    scaledown_window=SINGULARITY_SCALEDOWN_WINDOW,
    startup_timeout=STARTUP_TIMEOUT,
    timeout=24 * 60 * 60,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    volumes={str(DATA_ROOT): data_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),
        studio_s3_secret,
    ],
)
class WanGPSingularityWorker:
    @modal.enter(snap=True)
    def prepare_snapshot(self) -> None:
        from h3_snapshot import (
            SNAPSHOT_REVISION, emit_snapshot, make_transformer_resident,
            memory_report, validate_transformer_residency, warm_session,
        )

        started = time.monotonic()
        if float(SINGULARITY_PROFILE) != 1:
            raise ValueError("Transformer-resident snapshot requires WANGP_SINGULARITY_PROFILE=1")
        self.capture_id = str(uuid.uuid4())
        self.snapshot_revision = SNAPSHOT_REVISION
        emit_snapshot("initialize", capture_id=self.capture_id, revision=SNAPSHOT_REVISION)
        self.runtime = WanGPRuntime(SINGULARITY_PROFILE)
        self.runtime.initialize()
        session = self.runtime._session_for(SINGULARITY_MODEL)
        self.warmup = warm_session(session, sys.modules["wgp"], GENERATED_OUTPUT_ROOT, emit_snapshot)
        data_volume.commit()
        self.transformer_residency = make_transformer_resident(session, sys.modules["wgp"], emit_snapshot)
        self.capture_memory = memory_report(sys.modules["wgp"])
        validate_transformer_residency(self.capture_memory)
        self.warmup["initialize_seconds"] = round(time.monotonic() - started, 3)
        emit_snapshot("capture_ready", capture_id=self.capture_id, **self.capture_memory)

    @modal.enter(snap=False)
    def after_restore(self) -> None:
        from h3_snapshot import (
            emit_snapshot, loaded_state, memory_report, reseed_after_restore,
            validate_transformer_residency,
        )

        # Observe restored memory before any placement, warmup, or generation.
        self.restore_memory = memory_report(sys.modules["wgp"])
        validate_transformer_residency(self.restore_memory)
        reseed_after_restore()
        self.boot_id = str(uuid.uuid4())
        self.requests_served = 0
        self.last_request = None
        emit_snapshot("restore_residency", capture_id=self.capture_id, boot_id=self.boot_id,
                      **self.restore_memory)
        emit_snapshot(
            "container_ready", capture_id=self.capture_id, boot_id=self.boot_id,
            revision=self.snapshot_revision, **loaded_state(sys.modules["wgp"]),
        )

    @modal.method()
    def snapshot_info(self) -> dict[str, Any]:
        """Calling this initializes the GPU worker, including warmup if needed."""
        from h3_snapshot import loaded_state, memory_report

        return {
            "revision": self.snapshot_revision,
            "capture_id": self.capture_id,
            "boot_id": self.boot_id,
            "requests_served": self.requests_served,
            "warmup": self.warmup,
            "capture_memory": self.capture_memory,
            "restore_memory": self.restore_memory,
            "transformer_residency": self.transformer_residency,
            "current_memory": memory_report(sys.modules["wgp"]),
            "loaded": loaded_state(sys.modules["wgp"]),
            "last_request": self.last_request,
        }

    @modal.method()
    def run(self, job_id: str, model: str, params: dict[str, Any]) -> dict[str, Any]:
        from h3_snapshot import emit_snapshot, require_singularity, track_model_loads

        require_singularity(model)
        started = time.monotonic()
        wgp = sys.modules["wgp"]
        previous_model = wgp.wan_model
        first_denoising_seconds = None

        def observe(progress: Any) -> None:
            nonlocal first_denoising_seconds
            phase = str(getattr(progress, "phase", ""))
            step = getattr(progress, "current_step", None)
            total = getattr(progress, "total_steps", None)
            expected_steps = params.get("num_inference_steps")
            if (
                first_denoising_seconds is None and phase.startswith("inference")
                and isinstance(step, (int, float)) and step > 0
                and isinstance(expected_steps, int) and total == expected_steps
                and step <= expected_steps
            ):
                first_denoising_seconds = round(time.monotonic() - started, 3)
                emit_snapshot(
                    "first_denoising_progress", job_id=job_id, boot_id=self.boot_id,
                    elapsed_seconds=first_denoising_seconds, step=step,
                )

        emit_snapshot("request_start", job_id=job_id, boot_id=self.boot_id)
        counts = {"load_models_calls": 0}
        try:
            with track_model_loads(wgp) as counts:
                return self.runtime.run(job_id, model, params, progress_observer=observe)
        finally:
            self.requests_served += 1
            self.last_request = {
                "job_id": job_id,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "model_reused": wgp.wan_model is previous_model,
                "first_denoising_seconds": first_denoising_seconds,
                **counts,
            }
            emit_snapshot("request_end", boot_id=self.boot_id, **self.last_request)


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
