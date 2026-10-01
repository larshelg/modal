"""Deploy only SoL-Refiner: modal deploy sol_app.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import modal

from sol_common import APP_NAME, JOB_DICT_NAME, MODEL_ID, MODEL_VOLUME_NAME, validate_request
from sol_versions import MODEL_REVISION, SANA_COMMIT

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
jobs = modal.Dict.from_name(JOB_DICT_NAME, create_if_missing=True)
UPSTREAM = "/opt/Sana/models/sol-refiner/MiniMax-H3"
NATTEN_WHEEL = (
    "https://github.com/SHI-Labs/NATTEN/releases/download/v0.21.5/"
    "natten-0.21.5%2Btorch290cu126-cp312-cp312-linux_x86_64.whl"
)
CACHE_ENV = {"HF_HOME": "/models/huggingface", "HF_HUB_CACHE": "/models/huggingface/hub"}
download_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface-hub==1.33.0"
).env(CACHE_ENV).add_local_python_source("sol_common", "sol_versions")

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .add_local_file(str(Path(__file__).with_name("sol-requirements.lock")),
                    "/opt/sol-requirements.lock", copy=True)
    .run_commands(
        "python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r /opt/sol-requirements.lock",
        f"python -m pip install --no-deps '{NATTEN_WHEEL}'",
        "git init /opt/Sana && git -C /opt/Sana remote add origin https://github.com/NVlabs/Sana.git "
        f"&& git -C /opt/Sana fetch --depth 1 origin {SANA_COMMIT} "
        "&& git -C /opt/Sana checkout --detach FETCH_HEAD",
    )
    .env({**CACHE_ENV, "PYTHONPATH": UPSTREAM, "HF_HUB_OFFLINE": "1",
          "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
    .run_commands(
        "python -m pip check",
        "python -c 'from sol_refiner_h3 import SoLRefinerH3Pipeline; from natten.functional import na3d'",
        f"cd {UPSTREAM} && python -m unittest discover -s tests -v",
    )
    .add_local_python_source("sol_common", "sol_versions", "sol_storage", "sol_runtime", "sol_jobs")
)


@app.function(image=download_image, volumes={"/models": model_volume},
              secrets=[modal.Secret.from_name("huggingface-secret")],
              timeout=7200, max_containers=1)
def prepare_models() -> dict:
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION,
                                     cache_dir=CACHE_ENV["HF_HUB_CACHE"]))
    files = {str(p.relative_to(snapshot)): p.stat().st_size for p in snapshot.rglob("*") if p.is_file()}
    for component in ("vae", "transformer", "latent_upsampler", "diffusion_decoder", "text_encoder", "tokenizer", "connectors", "scheduler"):
        if not any(name.startswith(component + "/") for name in files):
            raise RuntimeError(f"snapshot is missing {component}")
    if "model_index.json" not in files or not all(files.values()):
        raise RuntimeError("snapshot contains missing or empty files")
    manifest = {"model": MODEL_ID, "revision": MODEL_REVISION, "files": files}
    target = Path("/models/manifests") / f"{MODEL_REVISION}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(target)
    model_volume.commit()
    return {"model": MODEL_ID, "revision": MODEL_REVISION, "files": len(files),
            "bytes": sum(files.values()), "volume": MODEL_VOLUME_NAME}


@app.cls(image=gpu_image, gpu="H100", cpu=8, memory=131072,
         volumes={"/models": model_volume}, secrets=[modal.Secret.from_name("studio-s3")],
         min_containers=0, max_containers=1, scaledown_window=300,
         timeout=1800, startup_timeout=1800)
class SolRefiner:
    @modal.enter()
    def load(self):
        from sol_runtime import SolRuntime, snapshot_path

        self.runtime = SolRuntime(snapshot_path())

    @modal.method()
    def refine(self, job_id: str, request: dict) -> dict:
        from sol_jobs import run_job

        return run_job(jobs, self.runtime, job_id, request)

    @modal.method()
    def smoke(self, video: bytes, request: dict) -> dict:
        # Small local-file baseline with no S3 transfer or asynchronous job record.
        request = validate_request(request)
        if not video or len(video) > 64 * 1024 * 1024:
            raise ValueError("smoke input must be between 1 byte and 64 MiB")
        with tempfile.TemporaryDirectory(prefix="sol-smoke-") as temporary:
            directory = Path(temporary)
            original = directory / "input.mp4"
            original.write_bytes(video)
            output, metrics = self.runtime.refine(original, directory, request)
            if output.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("smoke output exceeds 64 MiB; use the S3 service")
            return {"video": output.read_bytes(), "metrics": metrics}


@app.function(image=gpu_image, timeout=300)
def check_environment() -> dict:
    """Build/import check without allocating a GPU or downloading weights."""
    import importlib.metadata
    from sol_refiner_h3 import SoLRefinerH3Pipeline

    return {"pipeline": SoLRefinerH3Pipeline.__name__, "sana_commit": SANA_COMMIT,
            "model_revision": MODEL_REVISION,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "torchvision", "natten", "diffusers", "transformers", "imageio")}}


@app.function(image=gpu_image, secrets=[modal.Secret.from_name("studio-s3")], timeout=900)
def inspect_input(input_url: str, save_video: bool = False) -> dict:
    """Inspect a candidate on CPU before allocating the refinement GPU."""
    import subprocess
    from sol_runtime import _probe, probe_video
    from sol_storage import download_input

    with tempfile.TemporaryDirectory(prefix="sol-inspect-") as temporary:
        directory = Path(temporary)
        source = download_input(input_url, directory / "source.mp4")
        report = {"input_url": input_url, "size_bytes": source.stat().st_size,
                  "container": _probe(source, "-show_streams", "-show_format")}
        try:
            report["validated"] = probe_video(source)
        except ValueError as exc:
            report["validation_error"] = str(exc)
            try:
                report["validated_with_padding"] = probe_video(source, "pad")
            except ValueError as padding_exc:
                report["padding_validation_error"] = str(padding_exc)
        duration = float(report["container"]["format"]["duration"])
        previews = []
        for index, fraction in enumerate((0.1, 0.5, 0.9)):
            preview = directory / f"frame-{index}.jpg"
            subprocess.run([
                "ffmpeg", "-v", "error", "-nostdin", "-ss", str(duration * fraction),
                "-i", str(source), "-frames:v", "1", "-vf", "scale=768:-2", str(preview),
            ], check=True, capture_output=True, timeout=60)
            previews.append(preview.read_bytes())
        result = {"report": report, "previews": previews}
        if save_video:
            if source.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("inspection download is limited to 64 MiB")
            result["video"] = source.read_bytes()
        return result


@app.local_entrypoint()
def inspect(input_url: str, output_dir: str, save_video: bool = False):
    result = inspect_input.remote(input_url, save_video)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for index, preview in enumerate(result["previews"]):
        with (destination / f"source-frame-{index}.jpg").open("xb") as stream:
            stream.write(preview)
    with (destination / "source-inspection.json").open("x") as stream:
        json.dump(result["report"], stream, indent=2)
    if save_video:
        with (destination / "video.mp4").open("xb") as stream:
            stream.write(result["video"])
    # Keep potentially large embedded workflows in the saved inspection file.
    report = {k: v for k, v in result["report"].items() if k != "container"}
    print(json.dumps(report, indent=2))


@app.local_entrypoint()
def check():
    print(json.dumps(check_environment.remote(), indent=2))


@app.local_entrypoint()
def prepare():
    """One-time model download; does not start the GPU class."""
    print(json.dumps(prepare_models.remote(), indent=2))
