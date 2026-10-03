"""Standalone Fizgig training runtime on Modal.

This app exposes stable GPU training and CPU control functions. The local
``control.py`` client calls the deployment directly through Modal's SDK.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import modal

from fizgig_common import (
    APP_NAME,
    DATA_VOLUME_NAME,
    JOB_DICT_NAME,
    SUPPORTED_FAMILY,
    SUPPORTED_FAMILIES,
    SUPPORTED_PRESETS,
    utc_now,
    validate_component,
    validate_training_request,
)

from fizgig_s3 import (
    S3_SECRET_NAME, S3_REQUIRED_KEYS, create_s3_client, dataset_summary,
    plan_dataset, materialize_dataset, load_dataset_snapshot,
)


FIZGIG_REPOSITORY = "https://github.com/shootthesound/Fizgig.git"
FIZGIG_COMMIT = "5d5ced3739f303065ca6d13b51eb95c5ec85c00d"
FIZGIG_ROOT = Path("/opt/Fizgig")
FIZGIG_SCRIPTS = FIZGIG_ROOT / "src/fizgig/scripts"
FIZGIG_FAMILY_SCRIPTS = FIZGIG_ROOT / "src/fizgig/families"
CAPTION_SCRIPT = Path("/opt/fizgig_caption_dataset.py")

DATA_ROOT = Path("/data")
FIZGIG_DATA_ROOT = DATA_ROOT / "fizgig"
MODEL_ROOT = FIZGIG_DATA_ROOT / "models"
DATASET_ROOT = FIZGIG_DATA_ROOT / "datasets"
RUN_ROOT = FIZGIG_DATA_ROOT / "runs"
LORA_ROOT = DATA_ROOT / "loras"

GPU_TYPE = os.environ.get("FIZGIG_GPU", "L40S")
MAX_CONTAINERS = int(os.environ.get("FIZGIG_MAX_CONTAINERS", "1"))
SCALEDOWN_WINDOW = 5 * 60
STARTUP_TIMEOUT = 30 * 60
TRAINING_TIMEOUT = 24 * 60 * 60

MODEL_PATHS = {
    "minimax_h3": {
        "dit": MODEL_ROOT / "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "text_encoder": MODEL_ROOT / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "vae": MODEL_ROOT / "minimax_h3_video_vae_fp16.safetensors",
    },
    "krea2": {
        "dit": MODEL_ROOT / "krea2_raw_bf16.safetensors",
        "text_encoder": MODEL_ROOT / "qwen3vl_4b_fp8_scaled.safetensors",
        "vae": MODEL_ROOT / "qwen_image_vae.safetensors",
    },
}

_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
_EPOCH_PATTERNS = (
    re.compile(r"(?i)epoch\s+(\d+)\s*/\s*(\d+)"),
    re.compile(r"(?i)epoch\s+(\d+)\s+of\s+(\d+)"),
)


data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
job_store = modal.Dict.from_name(JOB_DICT_NAME, create_if_missing=True)

control_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("boto3>=1.40,<2")
    .add_local_python_source("fizgig_common", "fizgig_s3", copy=True)
)

fizgig_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential",
        "cmake",
        "ffmpeg",
        "git",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "ninja-build",
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel uv",
        f"git clone {FIZGIG_REPOSITORY} {FIZGIG_ROOT}",
        f"cd {FIZGIG_ROOT} && git checkout {FIZGIG_COMMIT}",
        # Use Fizgig's own installer helper so its CUDA PyTorch index is scoped
        # only to the torch packages rather than the complete dependency set.
        "UV_SYSTEM_PYTHON=1 "
        f"python {FIZGIG_ROOT}/uv_install_deps.py "
        f"{FIZGIG_ROOT}/requirements.txt /usr/local",
        # Modal injects its runner at /pkg; registry images still need the
        # transport dependency available in the image environment.
        "python -m pip install 'grpclib>=0.4.7,<0.4.10'",
        # Fail the image build if the pinned CLI entrypoints or their runtime
        # imports break. --help exits before loading models or starting training.
        f"python {FIZGIG_FAMILY_SCRIPTS}/cache.py --help > /dev/null",
        f"python {FIZGIG_FAMILY_SCRIPTS}/train.py --help > /dev/null",
        f"python {FIZGIG_SCRIPTS}/minimax_cache_latents.py --help > /dev/null",
        f"python {FIZGIG_SCRIPTS}/minimax_cache_text.py --help > /dev/null",
        f"python {FIZGIG_SCRIPTS}/minimax_train.py --help > /dev/null",
    )
    .pip_install("boto3>=1.40,<2")
    # Copy local code before declaring cache paths under the future Volume
    # mount. Otherwise the copy layer can materialize those cache directories
    # in the image, and Modal refuses to mount a Volume over a non-empty path.
    .add_local_file(
        str(Path(__file__).with_name("caption_dataset.py")),
        remote_path=str(CAPTION_SCRIPT),
        copy=True,
    )
    .add_local_python_source("fizgig_common", "fizgig_s3", copy=True)
    .env(
        {
            "FIZGIG_HOME": str(FIZGIG_DATA_ROOT),
            "HF_HOME": str(DATA_ROOT / "huggingface"),
            "HF_HUB_CACHE": str(DATA_ROOT / "huggingface/hub"),
            "HUGGINGFACE_HUB_CACHE": str(DATA_ROOT / "huggingface/hub"),
            "TRANSFORMERS_CACHE": str(DATA_ROOT / "huggingface/transformers"),
            "TRITON_CACHE_DIR": str(DATA_ROOT / "triton"),
            "TORCH_EXTENSIONS_DIR": str(DATA_ROOT / "torch-extensions"),
            # Keep UV's build/runtime bookkeeping outside the Volume mount;
            # otherwise it materializes /data before Modal can attach it.
            "XDG_CACHE_HOME": "/tmp/fizgig-cache",
            "PYTHONPATH": str(FIZGIG_ROOT / "src"),
            "PYTHONUNBUFFERED": "1",
        }
    )
)

app = modal.App(APP_NAME)


def normalized_absolute_path(path: str | os.PathLike[str]) -> Path:
    """Normalize traversal without resolving Modal Volume mount internals."""
    return Path(os.path.abspath(os.path.normpath(path)))


def require_data_path(path: str | os.PathLike[str], *, root: Path = DATA_ROOT) -> Path:
    normalized = normalized_absolute_path(path)
    try:
        normalized.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path must remain under {root}: {path}") from exc
    return normalized


def paths_for_request(request: dict[str, Any]) -> dict[str, Path]:
    run_dir = require_data_path(RUN_ROOT / request["output_name"], root=FIZGIG_DATA_ROOT)
    dataset_dir = (
        run_dir / "dataset" if request.get("dataset_s3") else
        require_data_path(DATASET_ROOT / request["dataset"], root=FIZGIG_DATA_ROOT)
    )
    return {
        "dataset_dir": dataset_dir,
        "image_dir": dataset_dir / "images",
        "cache_dir": dataset_dir / "cache",
        "config_path": dataset_dir / "dataset.toml",
        "run_dir": run_dir,
        "pause_path": run_dir / ".pause_requested",
        "promoted_lora": require_data_path(LORA_ROOT / f"{request['output_name']}.safetensors"),
    }


def dataset_config_text(image_dir: Path, cache_dir: Path) -> str:
    image_dir = require_data_path(image_dir, root=FIZGIG_DATA_ROOT)
    cache_dir = require_data_path(cache_dir, root=FIZGIG_DATA_ROOT)
    return "\n".join(
        [
            "[general]",
            "resolution = [512, 512]",
            'caption_extension = ".txt"',
            "batch_size = 1",
            "num_repeats = 1",
            "enable_bucket = true",
            "bucket_no_upscale = true",
            "",
            "[[datasets]]",
            f"image_directory = {json.dumps(str(image_dir))}",
            f"cache_directory = {json.dumps(str(cache_dir))}",
            "",
        ]
    )


def _latest_state_dir(run_dir: Path, output_name: str) -> Path | None:
    candidates = sorted(run_dir.glob(f"{output_name}-*-state"))
    return candidates[-1] if candidates else None


def resolve_resume_path(request: dict[str, Any], run_dir: Path) -> Path | None:
    resume_from = request.get("resume_from")
    if resume_from is None:
        return None
    path = (
        _latest_state_dir(run_dir, request["output_name"])
        if resume_from == "latest"
        else run_dir / resume_from
    )
    if path is None or not path.is_dir():
        raise ValueError(f"resume state was not found for {request['output_name']}")
    return require_data_path(path, root=run_dir)


def build_pipeline_commands(
    request: dict[str, Any], paths: dict[str, Path]
) -> list[tuple[str, list[str]]]:
    if request["family"] == "krea2":
        return _krea2_pipeline_commands(request, paths)
    return _h3_pipeline_commands(request, paths)


def _h3_pipeline_commands(
    request: dict[str, Any], paths: dict[str, Path]
) -> list[tuple[str, list[str]]]:
    python = "python"
    config = str(paths["config_path"])
    models = MODEL_PATHS["minimax_h3"]
    commands: list[tuple[str, list[str]]] = [
        (
            "caching_latents",
            [
                python,
                str(FIZGIG_SCRIPTS / "minimax_cache_latents.py"),
                "--dataset_config",
                config,
                "--vae",
                str(models["vae"]),
                "--skip_existing",
            ],
        ),
        (
            "caching_text",
            [
                python,
                str(FIZGIG_SCRIPTS / "minimax_cache_text.py"),
                "--dataset_config",
                config,
                "--text_encoder",
                str(models["text_encoder"]),
                "--skip_existing",
            ],
        ),
    ]

    train = [
        python,
        str(FIZGIG_SCRIPTS / "minimax_train.py"),
        "--dataset_config",
        config,
        "--dit",
        str(models["dit"]),
        "--output_dir",
        str(paths["run_dir"]),
        "--output_name",
        request["output_name"],
        "--network_dim",
        str(request["network_dim"]),
        "--network_alpha",
        str(request["network_alpha"]),
        "--learning_rate",
        str(request["learning_rate"]),
        "--max_train_epochs",
        str(request["epochs"]),
        "--save_every_n_epochs",
        str(request["save_every_n_epochs"]),
        "--save_state",
        "--save_state_on_train_end",
        "--keep_last_n_states",
        "2",
        "--seed",
        str(request["seed"]),
        "--optimizer_type",
        request["optimizer_type"],
        "--caption_dropout",
        "0.05",
        "--base_quant",
        "auto",
        "--blocks_to_swap",
        "auto",
        "--gradient_checkpointing",
        "auto",
        "--no_train_adaln",
    ]
    if request["adapter_ramp"]:
        train.extend(["--adapter_ramp", str(request["adapter_ramp"])])
    if request.get("trigger_word"):
        train.extend(["--metadata_trigger_phrase", request["trigger_word"]])
    resume_path = resolve_resume_path(request, paths["run_dir"])
    if resume_path:
        train.extend(["--resume", str(resume_path)])
    commands.append(("training", train))
    return commands


def _krea2_pipeline_commands(
    request: dict[str, Any], paths: dict[str, Path]
) -> list[tuple[str, list[str]]]:
    python = "python"
    config = str(paths["config_path"])
    models = MODEL_PATHS["krea2"]
    commands: list[tuple[str, list[str]]] = []

    if request["auto_caption"]:
        caption = [
            python,
            str(CAPTION_SCRIPT),
            "--image-dir",
            str(paths["image_dir"]),
            "--text-encoder",
            str(models["text_encoder"]),
            "--seed",
            str(request["seed"]),
        ]
        if request.get("trigger_word"):
            caption.extend(["--trigger-word", request["trigger_word"]])
        commands.append(("captioning", caption))

    commands.extend(
        [
            (
                "caching_latents",
                [
                    python,
                    str(FIZGIG_FAMILY_SCRIPTS / "cache.py"),
                    "--family",
                    "krea2",
                    "--stage",
                    "latents",
                    "--dataset_config",
                    config,
                    "--model",
                    str(models["vae"]),
                    "--skip_existing",
                ],
            ),
            (
                "caching_text",
                [
                    python,
                    str(FIZGIG_FAMILY_SCRIPTS / "cache.py"),
                    "--family",
                    "krea2",
                    "--stage",
                    "text",
                    "--dataset_config",
                    config,
                    "--model",
                    str(models["text_encoder"]),
                    "--skip_existing",
                ],
            ),
        ]
    )

    train = [
        python,
        str(FIZGIG_FAMILY_SCRIPTS / "train.py"),
        "--family",
        "krea2",
        "--precision",
        "auto",
        "--blocks_to_swap",
        "-1",
        "--dataset_config",
        config,
        "--dit",
        str(models["dit"]),
        "--output_dir",
        str(paths["run_dir"]),
        "--output_name",
        request["output_name"],
        "--network_dim",
        str(request["network_dim"]),
        "--network_alpha",
        str(request["network_alpha"]),
        "--learning_rate",
        str(request["learning_rate"]),
        "--max_train_epochs",
        str(request["epochs"]),
        "--save_every_n_epochs",
        str(request["save_every_n_epochs"]),
        "--save_state",
        "--save_state_on_train_end",
        "--keep_last_n_states",
        "2",
        "--seed",
        str(request["seed"]),
        "--optimizer_type",
        request["optimizer_type"],
        "--compile_blocks",
        "auto",
        "--log_per_image_loss",
        "--per_image_lr",
        "--text_encoder",
        str(models["text_encoder"]),
    ]
    if request["adaptive_lr"]:
        train.extend(
            [
                "--adaptive_lr",
                "--adaptive_lr_min",
                str(request["adaptive_lr_min"]),
                "--adaptive_lr_max",
                str(request["adaptive_lr_max"]),
            ]
        )
    if request["auto_recaption"]:
        train.extend(["--auto_recaption", "--captioner", str(models["text_encoder"])])
    if request.get("trigger_word"):
        train.extend(
            [
                "--trigger_word",
                request["trigger_word"],
                "--metadata_trigger_phrase",
                request["trigger_word"],
            ]
        )
    resume_path = resolve_resume_path(request, paths["run_dir"])
    if resume_path:
        train.extend(["--resume", str(resume_path)])
    commands.append(("training", train))
    return commands


def _ensure_layout() -> None:
    for path in (
        MODEL_ROOT,
        DATASET_ROOT,
        RUN_ROOT,
        LORA_ROOT,
        DATA_ROOT / "huggingface/hub",
        DATA_ROOT / "huggingface/transformers",
        DATA_ROOT / "triton",
        DATA_ROOT / "torch-extensions",
        DATA_ROOT / "cache",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _put_job(job_id: str, **updates: Any) -> dict[str, Any]:
    record = job_store.get(job_id, {"id": job_id})
    record.update(updates, updated_at=utc_now())
    job_store.put(job_id, record)
    return record


def _parse_epoch(line: str) -> tuple[int, int] | None:
    for pattern in _EPOCH_PATTERNS:
        match = pattern.search(line)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _run_stage(job_id: str, phase: str, command: list[str], pause_path: Path) -> None:
    _put_job(job_id, status="running", progress={"phase": phase})
    process = subprocess.Popen(
        command,
        cwd=FIZGIG_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    tail: deque[str] = deque(maxlen=50)
    stop_monitor = threading.Event()

    def monitor_pause() -> None:
        while not stop_monitor.wait(2):
            record = job_store.get(job_id, {})
            if record.get("pause_requested"):
                pause_path.touch(exist_ok=True)
                return

    monitor = threading.Thread(target=monitor_pause, daemon=True)
    monitor.start()
    last_update = 0.0
    epoch: tuple[int, int] | None = None
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if line:
                print(line, flush=True)
                tail.append(line)
                epoch = _parse_epoch(line) or epoch
            now = time.monotonic()
            if now - last_update >= 5:
                progress: dict[str, Any] = {"phase": phase, "log_tail": list(tail)}
                if epoch:
                    progress.update(epoch=epoch[0], epochs_total=epoch[1])
                _put_job(job_id, progress=progress)
                last_update = now
        return_code = process.wait()
    finally:
        stop_monitor.set()
        monitor.join(timeout=5)

    if return_code != 0:
        raise RuntimeError(f"Fizgig stage {phase!r} exited with code {return_code}")
    _put_job(job_id, progress={"phase": phase, "log_tail": list(tail)})


def _prepare_run(request: dict[str, Any], paths: dict[str, Path], *, progress=None) -> dict[str, Any] | None:
    _ensure_layout()
    if (
        paths["run_dir"].exists()
        and any(paths["run_dir"].iterdir())
        and request["resume_from"] is None
    ):
        raise ValueError(
            f"run directory already contains data: {paths['run_dir']}; "
            "choose a new output_name or resume"
        )
    dataset = None
    if request.get("dataset_s3"):
        if request["resume_from"]:
            manifest = load_dataset_snapshot(paths["dataset_dir"], request["dataset_s3"])
        else:
            client, bucket = create_s3_client()
            try:
                manifest = materialize_dataset(
                    client, request["dataset_s3"], bucket, paths["dataset_dir"], progress=progress,
                )
            finally:
                client.close()
        dataset = {**dataset_summary(manifest), "snapshot_path": str(paths["dataset_dir"])}
    if not paths["image_dir"].is_dir():
        raise ValueError(f"dataset images directory does not exist: {paths['image_dir']}")
    images = [
        path for path in paths["image_dir"].iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    ]
    if not images:
        raise ValueError(f"dataset images directory contains no supported images: {paths['image_dir']}")
    if request.get("dataset_s3") and request["family"] == "minimax_h3":
        missing = [path for path in images if not path.with_suffix(".txt").is_file()
                   or not path.with_suffix(".txt").read_text(encoding="utf-8").strip()]
        if missing:
            raise ValueError(f"MiniMax H3 requires non-empty .txt captions; {len(missing)} images are missing captions")
    paths["cache_dir"].mkdir(parents=True, exist_ok=True)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["config_path"].write_text(
        dataset_config_text(paths["image_dir"], paths["cache_dir"]), encoding="utf-8",
    )
    paths["pause_path"].unlink(missing_ok=True)
    return dataset


def _verify_models(request: dict[str, Any]) -> None:
    models = MODEL_PATHS[request["family"]]
    missing = [str(path) for path in models.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"required {request['family']} models are missing; run fetch_models first: "
            + ", ".join(missing)
        )


def _finalize_run(request: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    output = paths["run_dir"] / f"{request['output_name']}.safetensors"
    if not output.is_file():
        latest_state = _latest_state_dir(paths["run_dir"], request["output_name"])
        if latest_state is not None:
            return {
                "paused": True,
                "resume_from": latest_state.name,
                "run_path": str(paths["run_dir"]),
            }
        raise FileNotFoundError(f"Fizgig completed without producing {output}")

    paths["promoted_lora"].parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, paths["promoted_lora"])
    return {
        "paused": False,
        "artifact_path": str(paths["promoted_lora"]),
        "run_path": str(paths["run_dir"]),
        "size_bytes": paths["promoted_lora"].stat().st_size,
    }


@app.function(image=control_image)
def health() -> dict[str, Any]:
    """Return deployment metadata without starting a GPU container."""
    return {
        "service": APP_NAME,
        "ready": True,
        "fizgig_commit": FIZGIG_COMMIT,
        "families": list(SUPPORTED_FAMILIES),
        "presets": sorted(SUPPORTED_PRESETS),
        "gpu": GPU_TYPE,
        "max_containers": MAX_CONTAINERS,
        "volume": DATA_VOLUME_NAME,
        "dataset_sources": ["volume", "s3"],
    }


@app.function(
    image=fizgig_image,
    timeout=TRAINING_TIMEOUT,
    volumes={str(DATA_ROOT): data_volume},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def fetch_models(
    family: str = SUPPORTED_FAMILY,
    include_tools: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Download one supported model family into the shared Volume."""
    if family not in SUPPORTED_FAMILIES:
        choices = ", ".join(SUPPORTED_FAMILIES)
        raise ValueError(f"family must be one of: {choices}")
    _ensure_layout()
    upstream_family = "minimax" if family == "minimax_h3" else family
    command = [
        "python",
        "-m",
        "fizgig.scripts.fetch_models",
        "--family",
        upstream_family,
        "--models-dir",
        str(MODEL_ROOT),
    ]
    if include_tools:
        command.extend(["--family", "tools"])
    if dry_run:
        command.append("--dry-run")
    subprocess.run(command, cwd=FIZGIG_ROOT, check=True)
    if not dry_run:
        data_volume.commit()
    return {
        "family": family,
        "models_dir": str(MODEL_ROOT),
        "dry_run": dry_run,
        "expected_models": {
            name: str(path) for name, path in MODEL_PATHS[family].items()
        },
    }


@app.function(
    image=fizgig_image,
    gpu=GPU_TYPE,
    memory=65_536,
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    startup_timeout=STARTUP_TIMEOUT,
    timeout=TRAINING_TIMEOUT,
    volumes={str(DATA_ROOT): data_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),
        modal.Secret.from_name(S3_SECRET_NAME, required_keys=list(S3_REQUIRED_KEYS)),
    ],
)
def run_training(job_id: str, request: dict[str, Any]) -> dict[str, Any]:
    """Run a supported Fizgig cache-and-train pipeline."""
    job_id = validate_component(job_id, "job_id")
    normalized = validate_training_request(request)
    paths = paths_for_request(normalized)
    existing = job_store.get(job_id)
    if existing is not None and existing.get("status") not in {None, "queued"}:
        raise ValueError(f"job already exists with status {existing.get('status')!r}")
    _put_job(
        job_id,
        status="running",
        request=normalized,
        created_at=job_store.get(job_id, {}).get("created_at", utc_now()),
        started_at=utc_now(),
        pause_requested=False,
        progress={"phase": "preparing_dataset"},
    )
    try:
        data_volume.reload()
        _verify_models(normalized)
        last_progress = 0.0

        def download_progress(done, total, downloaded, total_bytes):
            nonlocal last_progress
            now = time.monotonic()
            if done == total or now - last_progress >= 2:
                _put_job(job_id, progress={
                    "phase": "downloading_dataset", "files_downloaded": done,
                    "files_total": total, "bytes_downloaded": downloaded, "bytes_total": total_bytes,
                })
                last_progress = now

        if normalized.get("dataset_s3") and not normalized["resume_from"]:
            _put_job(job_id, progress={"phase": "downloading_dataset"})
        dataset = _prepare_run(normalized, paths, progress=download_progress)
        if dataset:
            data_volume.commit()
            _put_job(job_id, dataset=dataset)
        for phase, command in build_pipeline_commands(normalized, paths):
            _run_stage(job_id, phase, command, paths["pause_path"])
            if phase != "training":
                data_volume.commit()

        _put_job(job_id, progress={"phase": "finalizing"})
        result = _finalize_run(normalized, paths)
        data_volume.commit()
        final_phase = "paused" if result["paused"] else "completed"
        record = _put_job(
            job_id,
            status="succeeded",
            progress={"phase": final_phase},
            result=result,
            completed_at=utc_now(),
        )
        return record
    except BaseException as exc:
        record = job_store.get(job_id, {"id": job_id})
        if record.get("status") != "cancelled":
            _put_job(
                job_id,
                status="failed",
                error={"message": str(exc), "type": type(exc).__name__},
                completed_at=utc_now(),
            )
        raise


@app.function(
    image=control_image,
    timeout=1800,
    secrets=[modal.Secret.from_name(S3_SECRET_NAME, required_keys=list(S3_REQUIRED_KEYS))],
)
def inspect_dataset(dataset_s3: str, verify_download: bool = False) -> dict[str, Any]:
    """Check an S3 prefix on CPU; optional downloads are discarded after verification."""
    client, bucket = create_s3_client()
    try:
        if verify_download:
            with TemporaryDirectory(prefix="fizgig-s3-check-") as temporary:
                manifest = materialize_dataset(client, dataset_s3, bucket, Path(temporary) / "dataset")
        else:
            manifest = plan_dataset(client, dataset_s3, bucket)
        return {**dataset_summary(manifest), "download_verified": verify_download}
    finally:
        client.close()


@app.function(image=control_image)
def request_pause(job_id: str) -> dict[str, Any]:
    job_id = validate_component(job_id, "job_id")
    record = job_store.get(job_id)
    if record is None:
        raise ValueError("job not found")
    if record.get("status") != "running":
        raise ValueError("only a running job can be paused")
    record.update(pause_requested=True, updated_at=utc_now())
    job_store.put(job_id, record)
    return record


@app.local_entrypoint()
def main(
    request_json: str = "",
    fetch: bool = False,
    family: str = SUPPORTED_FAMILY,
    dry_run: bool = False,
) -> None:
    """Small direct-development entrypoint for model fetches and training."""
    if fetch:
        print(fetch_models.remote(family, False, dry_run))
        return
    if not request_json:
        print(health.remote())
        return
    # Keep the legacy entrypoint usable while tracking jobs through the same
    # client as control.py. Calls target the stable deployed worker.
    from control import submit_training

    request = json.loads(Path(request_json).read_text(encoding="utf-8"))
    print(json.dumps(submit_training(request), indent=2))
