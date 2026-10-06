"""Native Krea2 Turbo warmup and MMGP residency for the L40S snapshot pool."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from h3_latent import native_task
from snapshot_common import active_lora_files, emit_snapshot, memory_report

KREA_MODEL = "krea2_turbo"
KREA_APP_NAME = "wangpt-krea-modal-app"
SNAPSHOT_REVISION = "krea2-turbo-l40s-transformer-v1"
WARMUP_PARAMS = {
    "model_type": KREA_MODEL,
    "prompt": "A red ceramic teapot on a wooden table, soft window light, detailed photograph",
    "negative_prompt": "",
    "resolution": "1024x1024",
    "num_inference_steps": 8,
    "guidance_scale": 0,
    "flow_shift": 5.0,
    "seed": 42,
    "batch_size": 1,
    "repeat_generation": 1,
    "image_mode": 1,
    "prompt_enhancer": "",
    "activated_loras": [],
    "loras_multipliers": "",
}


def require_krea(model: str) -> None:
    if model != KREA_MODEL:
        raise ValueError(f"Krea snapshot worker only accepts {KREA_MODEL}")


def loaded_state(wgp: Any) -> dict[str, Any]:
    context = wgp.get_loaded_model_context()
    if context is None or context.model_type != KREA_MODEL:
        raise RuntimeError("Krea warmup did not leave a reusable loaded model")
    return {"model": context.model_type, "profile": context.profile,
            "config": context.config_id, "active_loras": active_lora_files(context)}


def validate_residency(report: dict[str, Any]) -> None:
    if "L40S" not in report["gpu_name"]:
        raise RuntimeError("Krea snapshot requires an L40S")
    # Tensor placement is the primary completeness check. This lower bound
    # also rejects a CUDA-context-only capture without assuming H3's weight size.
    if report["cuda_allocated_bytes"] < 1024**3:
        raise RuntimeError("Krea transformer residency requires model-sized CUDA allocation")
    if report["mmgp_active_models"] != ["transformer"]:
        raise RuntimeError("Only the Krea transformer may be active at capture/restore")
    components = report["component_logical_elements_by_device"]
    for name in ("transformer", "text_encoder", "vae"):
        if not sum(components.get(name, {}).values()):
            raise RuntimeError(f"Krea snapshot is missing {name}")
    for name, devices in components.items():
        for device, count in devices.items():
            if count and (not device.startswith("cuda") if name == "transformer" else device != "cpu"):
                raise RuntimeError(f"Unexpected Krea snapshot placement: {name} on {device}")


def make_transformer_resident(session: Any, wgp: Any) -> dict[str, Any]:
    if session.active_job is not None or wgp.offloadobj.active_models_ids:
        raise RuntimeError("Krea native cleanup must finish before staging transformer")
    state = loaded_state(wgp)
    if state["profile"] != 1 or state["active_loras"]:
        raise RuntimeError("Krea base snapshot requires profile 1 without active LoRAs")
    before = memory_report(wgp)
    started = time.monotonic()
    wgp.offloadobj.ensure_model_loaded("transformer")
    after = memory_report(wgp)
    validate_residency(after)
    result = {"seconds": round(time.monotonic() - started, 3), "before": before,
              "after": after, "cuda_delta_bytes": after["cuda_allocated_bytes"] - before["cuda_allocated_bytes"]}
    emit_snapshot("krea_transformer_resident", **result)
    return result


def warm_session(session: Any, wgp: Any, output_root: Path, timeout: float = 25 * 60) -> dict[str, Any]:
    """Generate privately, then join native task cleanup before snapshotting."""
    if session.active_job is not None:
        raise RuntimeError("Cannot snapshot Krea with an active job")
    started = time.monotonic()
    original_output = session._output_dir
    job = None
    with TemporaryDirectory(prefix="krea-snapshot-warmup-", dir=output_root) as directory:
        settings = copy.deepcopy(session.get_default_settings(KREA_MODEL))
        settings.update(WARMUP_PARAMS)
        emit_snapshot("krea_warmup_start", model=KREA_MODEL)
        try:
            session._output_dir = Path(directory) / "outputs"
            job = session.submit_task(native_task(settings))
            result = job.result(timeout=timeout)
            job._thread.join(timeout=60)
            if job._thread.is_alive() or session.active_job is not None:
                raise RuntimeError("Krea warmup did not reach an idle boundary")
            if not result.success:
                errors = "; ".join(str(getattr(e, "message", e)) for e in result.errors)
                raise RuntimeError(f"Krea snapshot warmup failed: {errors}")
            if not any(Path(p).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                       and Path(p).is_file() for p in result.generated_files):
                raise RuntimeError("Krea warmup produced no image")
            state = loaded_state(wgp)
        finally:
            if job is not None:
                if job._thread.is_alive():
                    job.cancel()
                    job._thread.join(timeout=60)
                    if job._thread.is_alive():
                        raise RuntimeError("Krea warmup thread failed to stop; capture aborted")
                job.release_input_payload()
                job.release_output_payload()
            session._output_dir = original_output
            session._configure_runtime(session._ensure_runtime())
            with session._job_lock:
                session._state.clear()
                session._state.update(session._create_headless_state())
            wgp.clear_gen_cache()
    state["warmup_seconds"] = round(time.monotonic() - started, 3)
    emit_snapshot("krea_warmup_complete", **state)
    return state
