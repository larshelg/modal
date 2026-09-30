"""Headless wiring for the pinned H3 Latent Continue plugin."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from h3_mux import make_preserving_mux, make_verified_record

PLUGIN_COMMIT = "6b1334823e2acff297d8ab05b0e17eb1ba6a4a3b"
PLUGIN_KEY = "h3_latent_prototype"
PLUGIN_ROOT = Path("/opt/Wan2GP/plugins/wan2gp-h3-latent-continue")
PACKAGE = "_wangp_h3_latent_plugin"
TASK_KEY = "_h3_latent_task_options_v1"


def validate_plugin_data(params: dict[str, Any]) -> None:
    if TASK_KEY in params:
        raise ValueError(f"params.{TASK_KEY} is reserved by the runtime")
    data = params.get("plugin_data", {})
    if not isinstance(data, dict) or set(data) - {PLUGIN_KEY}:
        raise ValueError(f"plugin_data must be an object containing only {PLUGIN_KEY}")
    if PLUGIN_KEY not in data:
        return
    options = data[PLUGIN_KEY]
    if not isinstance(options, dict):
        raise ValueError(f"plugin_data.{PLUGIN_KEY} must be an object")
    allowed = {"save", "continue", "latent_path", "join_mode", "video_context_frames", "audio_context_seconds"}
    if set(options) - allowed:
        raise ValueError("Unknown H3 latent option")
    for key in ("save", "continue"):
        if key in options and type(options[key]) is not bool:
            raise ValueError(f"H3 latent {key} must be a boolean")
    if "latent_path" in options and not isinstance(options["latent_path"], str):
        raise ValueError("H3 latent latent_path must be a string")


def native_task(settings: dict[str, Any]) -> dict[str, Any]:
    """Move public plugin settings out of the flat generation parameters."""
    validate_plugin_data(settings)
    params = copy.deepcopy(settings)
    data = params.pop("plugin_data", {})
    if PLUGIN_KEY in data:
        options = {"save": False, "continue": False, "latent_path": "", **data[PLUGIN_KEY]}
        # Keep a task-owned snapshot: WanGP pops plugin_data while processing its queue.
        params[TASK_KEY] = {"model_type": params["model_type"], "options": options}
    params["_api"] = {"return_media": False}
    return {"params": params, "plugin_data": data}


def install_headless_hooks(wgp: Any) -> Any:
    """Reuse upstream wrappers without constructing UI components or a server."""
    if PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            PACKAGE, PLUGIN_ROOT / "__init__.py",
            submodule_search_locations=[str(PLUGIN_ROOT)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("H3 latent plugin is not installed")
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(PACKAGE, None)
            raise
    integration = importlib.import_module(f"{PACKAGE}.integration")
    if getattr(wgp, "_modal_h3_latent_installed", False):
        return integration
    bridge = importlib.import_module(f"{PACKAGE}.native_bridge")
    bridge.install_loader()
    wgp.generate_media = integration.make_generation_wrapper(wgp.generate_media, wgp.get_base_model_type)
    wgp.save_video = integration.make_save_wrapper(wgp.save_video)
    wgp.combine_and_concatenate_video_with_audio_tracks = make_preserving_mux(
        wgp.combine_and_concatenate_video_with_audio_tracks, integration.JOB,
    )
    wgp.record_file_metadata = integration.make_record_wrapper(make_verified_record(
        wgp.record_file_metadata, integration.JOB, validate_frame_alignment,
    ))
    wgp._modal_h3_latent_installed = True
    print(f"[H3 Latent] Headless integration installed ({PLUGIN_COMMIT[:12]}).", flush=True)
    return integration


def prepare_latent_job(wgp: Any, settings: dict[str, Any]) -> None:
    options = settings.get("plugin_data", {}).get(PLUGIN_KEY, {})
    if not (options.get("save") or options.get("continue")):
        return
    integration = install_headless_hooks(wgp)
    if wgp.get_base_model_type(settings["model_type"]) not in integration.SUPPORTED:
        raise ValueError("H3 latent continuation requires a supported FL2VA or Ref2VA architecture")
    integration.preflight(settings, integration.options({PLUGIN_KEY: options}))
    if options.get("continue"):
        info = integration.inspect_checkpoint(options["latent_path"])
        _, _, _, frames = wgp.get_video_info(settings["video_source"])
        validate_frame_alignment(info, frames)


def validate_frame_alignment(info: dict[str, Any], actual_frames: int) -> None:
    """Byte identity alone cannot detect frames lost during native audio muxing."""
    expected = info.get("output_frames")
    if expected is None:
        raise ValueError("H3 latent checkpoint lacks an exported frame count; recreate the source pair")
    if actual_frames != expected:
        raise ValueError(
            f"H3 latent source has {actual_frames} video frames but its checkpoint records {expected}. "
            "The export may have trimmed frames during audio muxing; use an intact source pair."
        )


def include_latent_outputs(result: Any, settings: dict[str, Any]) -> None:
    """Require and publish each requested checkpoint along with its source MP4."""
    if not result.success or not settings.get("plugin_data", {}).get(PLUGIN_KEY, {}).get("save"):
        return
    videos = [Path(path) for path in result.generated_files if Path(path).suffix.lower() == ".mp4"]
    if not videos:
        raise RuntimeError("H3 latent save returned no MP4")
    sidecars = [path.with_suffix(".safetensors") for path in videos]
    for path in sidecars:
        if not path.is_file():
            raise RuntimeError(f"H3 latent checkpoint was not saved for {path.stem}")
    result.generated_files.extend(str(path) for path in sidecars if str(path) not in result.generated_files)
