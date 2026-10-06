"""Reference-conditioned Krea Turbo Edit warmup for an independent L40S pool."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from h3_latent import native_task
from h3_snapshot import active_lora_files, create_reference, emit_snapshot, memory_report
from krea_snapshot import WARMUP_PARAMS as BASE_WARMUP_PARAMS
from krea_snapshot import validate_residency as validate_base_residency

KREA_EDIT_MODEL = "krea2_turbo_edit"
KREA_EDIT_APP_NAME = "wangpt-krea-edit-modal-app"
EDIT_LORA = "krea2_identity_edit_v1_2.safetensors"
SNAPSHOT_REVISION = "krea2-turbo-edit-l40s-transformer-v1"
WARMUP_PARAMS = {
    **BASE_WARMUP_PARAMS,
    "model_type": KREA_EDIT_MODEL,
    "prompt": "Change the red ball in image 1 to green; keep the table, framing and background unchanged.",
    "video_prompt_type": "KI",
    "model_mode": 0,
    "remove_background_images_ref": 0,
}


def require_krea_edit(model: str) -> None:
    if model != KREA_EDIT_MODEL:
        raise ValueError(f"Krea Edit snapshot worker only accepts {KREA_EDIT_MODEL}")


def loaded_state(wgp: Any) -> dict[str, Any]:
    context = wgp.get_loaded_model_context()
    if context is None or context.model_type != KREA_EDIT_MODEL:
        raise RuntimeError("Krea Edit warmup did not leave a reusable loaded model")
    if not any(EDIT_LORA in str(url) for url in context.model_def.get("loras", [])):
        raise RuntimeError("Krea Edit preset is missing the expected Identity Edit LoRA")
    return {"model": context.model_type, "profile": context.profile,
            "config": context.config_id, "edit_lora": EDIT_LORA,
            "active_loras": active_lora_files(context)}


def validate_residency(report: dict[str, Any]) -> None:
    validate_base_residency(report)
    if not sum(report["component_logical_elements_by_device"].get("vision_encoder", {}).values()):
        raise RuntimeError("Krea Edit snapshot is missing vision_encoder")


def make_transformer_resident(session: Any, wgp: Any) -> dict[str, Any]:
    if session.active_job is not None or wgp.offloadobj.active_models_ids:
        raise RuntimeError("Krea Edit native cleanup must finish before staging transformer")
    state = loaded_state(wgp)
    # Native cleanup unloads task LoRAs, including the preset's edit adapter.
    # Each request activates the preset adapter through the normal WanGP path.
    if state["profile"] != 1 or state["active_loras"]:
        raise RuntimeError("Krea Edit snapshot requires profile 1 without active task LoRAs")
    before = memory_report(wgp)
    started = time.monotonic()
    wgp.offloadobj.ensure_model_loaded("transformer")
    after = memory_report(wgp)
    validate_residency(after)
    result = {"seconds": round(time.monotonic() - started, 3), "before": before,
              "after": after, "cuda_delta_bytes": after["cuda_allocated_bytes"] - before["cuda_allocated_bytes"]}
    emit_snapshot("krea_edit_transformer_resident", **result)
    return result


def warm_session(session: Any, wgp: Any, output_root: Path, timeout: float = 25 * 60) -> dict[str, Any]:
    """Exercise vision conditioning and the edit adapter, then join cleanup."""
    if session.active_job is not None:
        raise RuntimeError("Cannot snapshot Krea Edit with an active job")
    started = time.monotonic()
    original_output = session._output_dir
    job = None
    edit_lora_observed = False

    class WarmupCallbacks:
        def on_progress(self, progress: Any) -> None:
            nonlocal edit_lora_observed
            if str(getattr(progress, "phase", "")).startswith("inference") and not edit_lora_observed:
                context = wgp.get_loaded_model_context()
                edit_lora_observed = (
                    context is not None and context.model_type == KREA_EDIT_MODEL
                    and EDIT_LORA in active_lora_files(context)
                )

    with TemporaryDirectory(prefix="krea-edit-snapshot-warmup-", dir=output_root) as directory:
        root = Path(directory)
        reference = root / "reference.png"
        create_reference(reference)
        settings = copy.deepcopy(session.get_default_settings(KREA_EDIT_MODEL))
        settings.update(WARMUP_PARAMS)
        settings["image_refs"] = [str(reference)]
        emit_snapshot("krea_edit_warmup_start", model=KREA_EDIT_MODEL)
        try:
            session._output_dir = root / "outputs"
            job = session.submit_task(native_task(settings), callbacks=WarmupCallbacks())
            result = job.result(timeout=timeout)
            job._thread.join(timeout=60)
            if job._thread.is_alive() or session.active_job is not None:
                raise RuntimeError("Krea Edit warmup did not reach an idle boundary")
            if not result.success:
                errors = "; ".join(str(getattr(e, "message", e)) for e in result.errors)
                raise RuntimeError(f"Krea Edit snapshot warmup failed: {errors}")
            if not any(Path(p).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                       and Path(p).is_file() for p in result.generated_files):
                raise RuntimeError("Krea Edit warmup produced no image")
            if not edit_lora_observed:
                raise RuntimeError("Identity Edit LoRA was not observed during warmup inference")
            state = loaded_state(wgp)
            state["edit_lora_verified_during_warmup"] = True
        finally:
            if job is not None:
                if job._thread.is_alive():
                    job.cancel()
                    job._thread.join(timeout=60)
                    if job._thread.is_alive():
                        raise RuntimeError("Krea Edit warmup thread failed to stop; capture aborted")
                job.release_input_payload()
                job.release_output_payload()
            session._output_dir = original_output
            session._configure_runtime(session._ensure_runtime())
            with session._job_lock:
                session._state.clear()
                session._state.update(session._create_headless_state())
            wgp.clear_gen_cache()
    state["warmup_seconds"] = round(time.monotonic() - started, 3)
    emit_snapshot("krea_edit_warmup_complete", **state)
    return state
