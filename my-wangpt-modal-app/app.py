"""Wan2GP MCP server on Modal with persistent model and compiler caches."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "wangp-mcp"
WAN_ROOT = Path("/opt/Wan2GP")
PATCH_ROOT = Path("/opt/wangp-patches")
WAN_COMMIT = "a042474d477a741d6b9b60fc6ff304077113cb25"
DATA_ROOT = Path("/data")
PORT = 8000

GPU_TYPE = os.environ.get("WANGP_GPU", "L40S")
DATA_VOLUME_NAME = "wangp-data"
SCALEDOWN_WINDOW = 20 * 60
STARTUP_TIMEOUT = 30 * 60

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

image = (
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
        # Modal injects its runner package at /pkg, but dependencies still need
        # to exist in registry-based images. Install this after Wan2GP's
        # requirements so its resolver cannot leave the runner without grpclib.
        "python -m pip install 'grpclib>=0.4.7,<0.4.10'",
    )
    .env(
        {
            "HF_HOME": str(DATA_ROOT / "huggingface"),
            "HF_HUB_CACHE": str(DATA_ROOT / "huggingface/hub"),
            "HUGGINGFACE_HUB_CACHE": str(DATA_ROOT / "huggingface/hub"),
            "TRANSFORMERS_CACHE": str(DATA_ROOT / "huggingface/transformers"),
            "TRITON_CACHE_DIR": str(DATA_ROOT / "triton"),
            "TORCH_EXTENSIONS_DIR": str(DATA_ROOT / "torch-extensions"),
            "XDG_CACHE_HOME": str(DATA_ROOT / "cache"),
            # sitecustomize.py in PATCH_ROOT disables the optional persistent
            # GET/SSE MCP stream while preserving stateless JSON POST calls.
            "PYTHONPATH": f"{PATCH_ROOT}:{WAN_ROOT}",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_file(
        str(Path(__file__).with_name("sitecustomize.py")),
        remote_path=str(PATCH_ROOT / "sitecustomize.py"),
    )
)

app = modal.App(APP_NAME, image=image)


def _ensure_cache_layout() -> None:
    for relative_path in CACHE_DIRS:
        (DATA_ROOT / relative_path).mkdir(parents=True, exist_ok=True)


def _force_directory_symlink(link_path: Path, target_path: Path) -> None:
    """Point a Wan2GP data directory at its persistent Volume directory."""
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


@app.function(
    gpu=GPU_TYPE,
    # Keep enough host RAM for model offloading without restricting L40S scheduling.
    memory=65_536,
    min_containers=0,
    max_containers=1,
    scaledown_window=SCALEDOWN_WINDOW,
    startup_timeout=STARTUP_TIMEOUT,
    timeout=24 * 60 * 60,
    volumes={str(DATA_ROOT): data_volume},
    secrets=[
        modal.Secret.from_name(
            "huggingface-secret",
            required_keys=["HF_TOKEN"],
        )
    ],
)
@modal.concurrent(max_inputs=10)
@modal.web_server(PORT, startup_timeout=STARTUP_TIMEOUT, requires_proxy_auth=True)
def serve() -> None:
    _ensure_cache_layout()

    for name in ("ckpts", "outputs", "settings", "loras"):
        _force_directory_symlink(WAN_ROOT / name, DATA_ROOT / name)

    # Publish the initial directory layout immediately. Modal also commits
    # subsequent Volume writes in the background while the server is running.
    data_volume.commit()

    command = [
        "python",
        "wgp.py",
        "--mcp",
        "--mcp-transport",
        "streamable-http",
        "--mcp-host",
        "0.0.0.0",
        "--mcp-port",
        str(PORT),
        "--config",
        str(DATA_ROOT / "config"),
        "--settings",
        str(DATA_ROOT / "settings"),
        "--loras",
        str(DATA_ROOT / "loras"),
        "--output-dir",
        str(DATA_ROOT / "outputs"),
        "--profile",
        "4",
        "--attention",
        "sdpa",
        "--verbose",
        "1",
    ]

    subprocess.Popen(command, cwd=WAN_ROOT)
