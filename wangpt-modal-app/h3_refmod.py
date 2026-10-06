"""Headless activation of the pinned MiniMax H3 RefMods plugin."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from h3_refmod_numbered import install_numbered_hooks

REFMOD_COMMIT = "bff0a8554ae4f6f916e38eaa0ef9a1af03fd437b"
REFMOD_ROOT = Path("/opt/Wan2GP/plugins/wan2gp-minimax-h3-refmod")
REFMOD_DATA_ROOT = Path("/data/refmods")
PACKAGE = "_wangp_h3_refmod_plugin"


def refmods_dir() -> str:
    """Use the same persistent library as Fizgig's exported RefMods."""
    REFMOD_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return str(REFMOD_DATA_ROOT)


def install_refmod_hooks(wgp: Any) -> Any:
    if PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            PACKAGE, REFMOD_ROOT / "__init__.py",
            submodule_search_locations=[str(REFMOD_ROOT)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("MiniMax H3 RefMods plugin is not installed")
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(PACKAGE, None)
            raise
    patches = importlib.import_module(f"{PACKAGE}.patches")
    if getattr(wgp, "_modal_h3_refmod_installed", False):
        return patches
    storage = importlib.import_module(f"{PACKAGE}.storage")
    storage.refmods_dir = refmods_dir
    error = patches.install_patches()
    if error:
        raise RuntimeError(f"MiniMax H3 RefMods initialization failed: {error}")
    install_numbered_hooks(wgp, patches, storage, importlib.import_module("models.minimax_h3.pipeline"))
    # The API initializes model definitions before plugins. Rebuild them so
    # native validation preserves RefMod custom_settings instead of dropping them.
    # shared.api restores the caller's working directory after init(), while
    # refresh_model_defs resolves defaults/ and finetunes/ relative to WanGP.
    previous_directory = Path.cwd()
    try:
        os.chdir(Path(wgp.__file__).parent)
        wgp.refresh_model_defs()
    finally:
        os.chdir(previous_directory)
    definitions = [
        definition for model, definition in wgp.models_def.items()
        if (wgp.get_base_model_type(model) or "").startswith("minimax_h3_ref2va")
    ]
    required = {patches.SETTING_GENERATE, patches.SETTING_EXTRACT}
    # Current H3 already declares four native fields. The plugin adds two;
    # WanGP's five GUI slots would otherwise drop the sixth API setting.
    setting_count = max((len(d.get("custom_settings", [])) for d in definitions), default=0)
    wgp.CUSTOM_SETTINGS_MAX = max(wgp.CUSTOM_SETTINGS_MAX, setting_count)
    extra_settings = importlib.import_module("shared.extra_settings")
    extra_settings._CUSTOM_SETTINGS_MAX = max(extra_settings._CUSTOM_SETTINGS_MAX, setting_count)
    if not definitions or any(
        not required.issubset({item.get("id") for item in wgp.get_model_custom_settings(definition)})
        for definition in definitions
    ):
        raise RuntimeError("MiniMax H3 RefMods settings are missing from the model catalog")
    wgp._modal_h3_refmod_installed = True
    print(f"[H3 RefMods] Headless integration installed ({REFMOD_COMMIT[:12]}).", flush=True)
    return patches
