"""Singularity snapshot warmup, adapted to the pinned WanGP session API.

Native warmup/cleanup is followed by MMGP-managed transformer-only residency.
Heavy dependencies are imported only inside the GPU container.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from functools import wraps
import json
import random
import secrets
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from h3_latent import native_task
from wangpt_common import SINGULARITY_MODEL

SNAPSHOT_REVISION = "singularity-transformer-resident-v3"
# INT8 transformer is ~20,076 MiB (19.6 GiB), without the task-specific LoRA.
MIN_TRANSFORMER_CUDA_BYTES = 19 * 1024**3
ACCELERATOR = "minimax_h3_lightx2v_ref2v_turbo_4step_alpha8_v0.1_bf16.safetensors"
WARMUP_PARAMS = {
    "model_type": SINGULARITY_MODEL,
    "prompt": (
        "summary: A red ball rests on a blue table.\n"
        "detailed_description: [Shot 1] The red ball in the reference image rolls "
        "slowly across the blue table. A steady camera, one continuous shot.\n"
        "overall_soundscape: A quiet room with a soft rolling sound.\n"
        "non_diegetic_music: None."
    ),
    "resolution": "832x480",
    "video_length": 124,
    "num_inference_steps": 4,
    "sample_solver": "euler",
    "flow_shift": 12.0,
    "guidance_scale": 1.0,
    "guidance_phases": 1,
    "batch_size": 1,
    "repeat_generation": 1,
    "seed": 42,
    "prompt_enhancer": "",
    "activated_loras": [],
    "loras_multipliers": "",
    "video_prompt_type": "I",
    "image_prompt_type": "",
    "audio_prompt_type": "",
}


def require_singularity(model: str) -> None:
    if model != SINGULARITY_MODEL:
        raise ValueError(f"snapshot worker only accepts {SINGULARITY_MODEL}")


def active_lora_files(context: Any) -> list[str]:
    if context is None:
        return []
    transformer = context.offloadobj.models["transformer"]
    active = list(getattr(transformer, "_loras_active_adapters", []))
    adapters = getattr(transformer, "_loras_adapters", {})
    return [Path(str(adapters.get(name, ""))).name for name in active]


def loaded_state(wgp: Any) -> dict[str, Any]:
    context = wgp.get_loaded_model_context()
    if context is None or context.model_type != SINGULARITY_MODEL:
        raise RuntimeError("Singularity warmup did not leave a reusable loaded model")
    definition = context.model_def
    if not any(ACCELERATOR in str(url) for url in definition.get("loras", [])):
        raise RuntimeError("Singularity preset is missing the expected acceleration LoRA")
    return {
        "model": context.model_type,
        "profile": context.profile,
        "config": context.config_id,
        "accelerator": ACCELERATOR,
        "active_loras": active_lora_files(context),
    }


def memory_report(wgp: Any) -> dict[str, Any]:
    import psutil
    import torch

    torch.cuda.synchronize()
    components = {}
    for name, model in wgp.offloadobj.models.items():
        devices: dict[str, int] = {}
        # Logical tensor elements, not physical storage bytes: quantized tensors
        # may wrap packed storage and multiple components can share parameters.
        for tensor in (*model.parameters(), *model.buffers()):
            device = str(tensor.device)
            devices[device] = devices.get(device, 0) + tensor.numel()
        components[name] = devices
    free, total = torch.cuda.mem_get_info()
    return {
        "cpu_rss_bytes": psutil.Process().memory_info().rss,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_free_bytes": free,
        "gpu_total_bytes": total,
        "gpu_name": torch.cuda.get_device_name(),
        "mmgp_active_models": list(wgp.offloadobj.active_models_ids),
        "component_logical_elements_by_device": components,
    }


def create_reference(path: Path) -> None:
    from PIL import Image, ImageDraw

    with Image.new("RGB", (832, 480), (220, 220, 220)) as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 300, 831, 479), fill=(40, 95, 160))
        draw.ellipse((330, 140, 490, 300), fill=(190, 35, 35))
        image.save(path)


def validate_transformer_residency(report: dict[str, Any]) -> None:
    """Fail capture/restore rather than mistake a partial load for residency."""
    if report["cuda_allocated_bytes"] < MIN_TRANSFORMER_CUDA_BYTES:
        raise RuntimeError("Transformer residency requires at least 19 GiB CUDA allocation")
    if report["mmgp_active_models"] != ["transformer"]:
        raise RuntimeError("Only transformer may be active at the snapshot boundary")
    components = report["component_logical_elements_by_device"]
    transformer = components.get("transformer", {})
    if not sum(transformer.values()) or any(
        count and not device.startswith("cuda:") and device != "cuda"
        for device, count in transformer.items()
    ):
        raise RuntimeError("Transformer is not entirely on CUDA")
    for name, devices in components.items():
        if name != "transformer" and any(count and device != "cpu" for device, count in devices.items()):
            raise RuntimeError(f"Snapshot must leave {name} on CPU")


def make_transformer_resident(session: Any, wgp: Any, emit: Callable[..., None]) -> dict[str, Any]:
    if session.active_job is not None:
        raise RuntimeError("Cannot stage transformer while a job is active")
    if loaded_state(wgp)["profile"] != 1:
        raise RuntimeError("Transformer snapshot requires profile 1")
    manager = wgp.offloadobj
    if manager.active_models_ids:
        raise RuntimeError("Native cleanup must finish before staging transformer")
    before = memory_report(wgp)
    started = time.monotonic()
    # Unlike .to('cuda'), this preserves MMGP's model/block bookkeeping.
    manager.ensure_model_loaded("transformer")
    after = memory_report(wgp)  # includes CUDA synchronization
    validate_transformer_residency(after)
    result = {"seconds": round(time.monotonic() - started, 3),
              "before": before, "after": after,
              "cuda_delta_bytes": after["cuda_allocated_bytes"] - before["cuda_allocated_bytes"]}
    emit("transformer_resident", **result)
    return result


@contextmanager
def track_model_loads(wgp: Any):
    """Observe native model construction, not stage-specific CPU/GPU transfers."""
    original = wgp.load_models
    counts = {"load_models_calls": 0}

    @wraps(original)
    def tracked(*args: Any, **kwargs: Any):
        counts["load_models_calls"] += 1
        return original(*args, **kwargs)

    wgp.load_models = tracked
    try:
        yield counts
    finally:
        wgp.load_models = original


def warm_session(
    session: Any,
    wgp: Any,
    output_root: Path,
    emit: Callable[..., None],
    timeout: float = 25 * 60,
) -> dict[str, Any]:
    """Run a real, private task and retain its model, never publishing a job.

    Private session fields here follow shared/api.py and shared/api_cli.py at
    WAN_COMMIT. Join the job thread: result() is signalled before its final exit.
    """
    if session.active_job is not None:
        raise RuntimeError("Cannot snapshot a session with an active job")
    started = time.monotonic()
    original_output = session._output_dir
    job = None
    accelerator_observed = False

    class WarmupCallbacks:
        def on_progress(self, progress: Any) -> None:
            nonlocal accelerator_observed
            # WanGP removes adapters at task completion. Check during inference,
            # not afterward, without changing native LoRA loading or cleanup.
            phase = str(getattr(progress, "phase", ""))
            if phase.startswith("inference") and not accelerator_observed:
                context = wgp.get_loaded_model_context()
                accelerator_observed = (
                    context is not None and context.model_type == SINGULARITY_MODEL
                    and ACCELERATOR in active_lora_files(context)
                )

    with TemporaryDirectory(prefix="snapshot-warmup-", dir=output_root) as directory:
        root = Path(directory)
        reference = root / "reference.png"
        create_reference(reference)
        settings = copy.deepcopy(session.get_default_settings(SINGULARITY_MODEL))
        settings.update(WARMUP_PARAMS)
        settings["image_refs"] = [str(reference)]
        emit("warmup_start", model=SINGULARITY_MODEL)
        try:
            session._output_dir = root / "outputs"
            # Native generation calls wgp.load_models and activates preset LoRAs.
            # The runtime already installed the latent hooks before this point.
            job = session.submit_task(native_task(settings), callbacks=WarmupCallbacks())
            result = job.result(timeout=timeout)
            job._thread.join(timeout=60)
            if job._thread.is_alive() or session.active_job is not None:
                raise RuntimeError("Warmup task did not reach an idle boundary")
            if not result.success:
                errors = "; ".join(str(getattr(e, "message", e)) for e in result.errors)
                raise RuntimeError(f"Singularity snapshot warmup failed: {errors}")
            if not any(
                Path(path).suffix == ".mp4" and Path(path).is_file()
                for path in result.generated_files
            ):
                raise RuntimeError("Singularity warmup produced no MP4")
            if not accelerator_observed:
                raise RuntimeError("Singularity accelerator was not observed during warmup inference")
            state = loaded_state(wgp)
            state["accelerator_verified_during_warmup"] = True
        finally:
            if job is not None:
                if job._thread.is_alive():
                    job.cancel()
                    job._thread.join(timeout=60)
                    if job._thread.is_alive():
                        # Raising prevents capture; never reset a live session.
                        raise RuntimeError("Warmup thread failed to stop; snapshot aborted")
                job.release_input_payload()
                job.release_output_payload()
            session._output_dir = original_output
            session._configure_runtime(session._ensure_runtime())
            with session._job_lock:
                session._state.clear()
                session._state.update(session._create_headless_state())
            wgp.clear_gen_cache()
    state["warmup_seconds"] = round(time.monotonic() - started, 3)
    emit("warmup_complete", **state)
    return state


def reseed_after_restore() -> None:
    """Avoid replaying the captured random-seed sequence on every new container."""
    import numpy as np
    import torch

    random.seed(secrets.randbits(128))
    np.random.seed(secrets.randbits(32))
    torch.manual_seed(secrets.randbits(63))


def emit_snapshot(stage: str, **details: Any) -> None:
    print("[wangp-snapshot] " + json.dumps({"stage": stage, **details}, sort_keys=True), flush=True)
