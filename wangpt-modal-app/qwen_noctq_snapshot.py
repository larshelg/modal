"""Native Qwen Noct Q V4 reference warmup and L40S GPU residency."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from h3_latent import native_task
from h3_snapshot import active_lora_files, create_reference, emit_snapshot, memory_report

QWEN_NOCTQ_MODEL = "qwen_image_21_noctq_v4"
QWEN_NOCTQ_APP_NAME = "wangpt-qwen-noctq-modal-app"
SNAPSHOT_REVISION = "qwen-noctq-v4-l40s-transformer-v1"
CHECKPOINT = "NoctQ_V4_int8_convrot.safetensors"
CHECKPOINT_REVISION = "a81b9af51120a78e285e57906f2250a2a02080e9"
CHECKPOINT_URL = (
    "https://huggingface.co/Noctaluna/Noct-Q-Uncensored-Qwen-Image-2.1/resolve/"
    f"{CHECKPOINT_REVISION}/{CHECKPOINT}"
)
WARMUP_PARAMS = {
    "model_type": QWEN_NOCTQ_MODEL,
    "prompt": "Change the red ball in <image1> to green; keep the table, framing and background unchanged.",
    "negative_prompt": " ",
    "resolution": "1024x1024",
    "num_inference_steps": 25,
    "sample_solver": "default",
    "guidance_scale": 3.0,
    "flow_shift": 5.0,
    "seed": 42,
    "batch_size": 1,
    "repeat_generation": 1,
    "image_mode": 1,
    "video_prompt_type": "KI",
    "model_mode": 0,
    "remove_background_images_ref": 0,
    "prompt_enhancer": "",
    "activated_loras": [],
    "loras_multipliers": "",
    "custom_settings": {"qwen21_kv_cache": "Disabled", "rgba": "Disabled"},
}


def require_qwen_noctq(model: str) -> None:
    if model != QWEN_NOCTQ_MODEL:
        raise ValueError(f"Qwen Noct Q snapshot worker only accepts {QWEN_NOCTQ_MODEL}")


def loaded_state(wgp: Any) -> dict[str, Any]:
    context = wgp.get_loaded_model_context()
    if context is None or context.model_type != QWEN_NOCTQ_MODEL:
        raise RuntimeError("Qwen Noct Q warmup did not leave a reusable loaded model")
    urls = context.model_def.get("URLs", [])
    if CHECKPOINT_URL not in ([urls] if isinstance(urls, str) else urls):
        raise RuntimeError("Qwen Noct Q preset is missing the pinned V4 checkpoint")
    return {"model": context.model_type, "profile": context.profile,
            "config": context.config_id, "active_loras": active_lora_files(context),
            "checkpoint": CHECKPOINT, "checkpoint_revision": CHECKPOINT_REVISION}


def validate_residency(report: dict[str, Any]) -> None:
    if "L40S" not in report["gpu_name"]:
        raise RuntimeError("Qwen Noct Q snapshot requires an L40S")
    if report["cuda_allocated_bytes"] < 1024**3:
        raise RuntimeError("Qwen transformer residency requires model-sized CUDA allocation")
    if report["mmgp_active_models"] != ["transformer"]:
        raise RuntimeError("Only the Qwen transformer may be active at capture/restore")
    components = report["component_logical_elements_by_device"]
    for name in ("transformer", "text_encoder", "vision_encoder", "vae"):
        if not sum(components.get(name, {}).values()):
            raise RuntimeError(f"Qwen Noct Q snapshot is missing {name}")
    for name, devices in components.items():
        for device, count in devices.items():
            if count and (not device.startswith("cuda") if name == "transformer" else device != "cpu"):
                raise RuntimeError(f"Unexpected Qwen Noct Q snapshot placement: {name} on {device}")


def make_transformer_resident(session: Any, wgp: Any) -> dict[str, Any]:
    if session.active_job is not None or wgp.offloadobj.active_models_ids:
        raise RuntimeError("Qwen native cleanup must finish before staging transformer")
    state = loaded_state(wgp)
    if state["profile"] != 1 or state["active_loras"]:
        raise RuntimeError("Qwen base snapshot requires profile 1 without active task LoRAs")
    before = memory_report(wgp)
    started = time.monotonic()
    wgp.offloadobj.ensure_model_loaded("transformer")
    after = memory_report(wgp)
    validate_residency(after)
    result = {"seconds": round(time.monotonic() - started, 3), "before": before,
              "after": after, "cuda_delta_bytes": after["cuda_allocated_bytes"] - before["cuda_allocated_bytes"]}
    emit_snapshot("qwen_noctq_transformer_resident", **result)
    return result


def warm_session(session: Any, wgp: Any, output_root: Path, timeout: float = 25 * 60) -> dict[str, Any]:
    """Run a private reference-image edit, then join native cleanup."""
    if session.active_job is not None:
        raise RuntimeError("Cannot snapshot Qwen Noct Q with an active job")
    started = time.monotonic()
    original_output = session._output_dir
    job = None
    with TemporaryDirectory(prefix="qwen-noctq-snapshot-warmup-", dir=output_root) as directory:
        root = Path(directory)
        reference = root / "reference.png"
        create_reference(reference)
        settings = copy.deepcopy(session.get_default_settings(QWEN_NOCTQ_MODEL))
        settings.update(WARMUP_PARAMS)
        settings["image_refs"] = [str(reference)]
        emit_snapshot("qwen_noctq_warmup_start", model=QWEN_NOCTQ_MODEL)
        try:
            session._output_dir = root / "outputs"
            job = session.submit_task(native_task(settings))
            result = job.result(timeout=timeout)
            job._thread.join(timeout=60)
            if job._thread.is_alive() or session.active_job is not None:
                raise RuntimeError("Qwen Noct Q warmup did not reach an idle boundary")
            if not result.success:
                errors = "; ".join(str(getattr(e, "message", e)) for e in result.errors)
                raise RuntimeError(f"Qwen Noct Q snapshot warmup failed: {errors}")
            if not any(Path(p).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                       and Path(p).is_file() for p in result.generated_files):
                raise RuntimeError("Qwen Noct Q warmup produced no image")
            state = loaded_state(wgp)
        finally:
            if job is not None:
                if job._thread.is_alive():
                    job.cancel()
                    job._thread.join(timeout=60)
                    if job._thread.is_alive():
                        raise RuntimeError("Qwen warmup thread failed to stop; capture aborted")
                job.release_input_payload()
                job.release_output_payload()
            session._output_dir = original_output
            session._configure_runtime(session._ensure_runtime())
            with session._job_lock:
                session._state.clear()
                session._state.update(session._create_headless_state())
            wgp.clear_gen_cache()
    state["warmup_seconds"] = round(time.monotonic() - started, 3)
    emit_snapshot("qwen_noctq_warmup_complete", **state)
    return state
