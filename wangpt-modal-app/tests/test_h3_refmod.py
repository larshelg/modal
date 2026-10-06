from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

import h3_refmod


@pytest.fixture
def plugin(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(h3_refmod, "install_numbered_hooks", lambda *args: None)
    monkeypatch.setitem(sys.modules, "models.minimax_h3.pipeline", SimpleNamespace())
    patches = SimpleNamespace(
        SETTING_GENERATE="h3_refmod_state", SETTING_EXTRACT="h3_refmod_extract",
        install_patches=lambda: calls.append("patch"),
    )
    storage = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "shared.extra_settings", SimpleNamespace(_CUSTOM_SETTINGS_MAX=5))
    monkeypatch.setitem(sys.modules, h3_refmod.PACKAGE, SimpleNamespace())
    monkeypatch.setitem(sys.modules, f"{h3_refmod.PACKAGE}.patches", patches)
    monkeypatch.setitem(sys.modules, f"{h3_refmod.PACKAGE}.storage", storage)
    wgp = SimpleNamespace(
        __file__=str(tmp_path / "wgp.py"),
        CUSTOM_SETTINGS_MAX=5,
        models_def={"custom_h3": {}, "qwen": {}},
        get_base_model_type=lambda model: "minimax_h3_ref2va" if model == "custom_h3" else model,
    )
    wgp.get_model_custom_settings = lambda definition: definition.get("custom_settings", [])[:wgp.CUSTOM_SETTINGS_MAX]

    def refresh():
        assert Path.cwd() == tmp_path
        calls.append("refresh")
        wgp.models_def["custom_h3"]["custom_settings"] = [{"id": f"native_{i}"} for i in range(4)] + [
            {"id": patches.SETTING_GENERATE}, {"id": patches.SETTING_EXTRACT},
        ]

    wgp.refresh_model_defs = refresh
    return wgp, patches, storage, calls


def test_hooks_refresh_cached_custom_models_once_and_share_fizgig_library(plugin, monkeypatch, tmp_path):
    wgp, patches, storage, calls = plugin
    monkeypatch.setattr(h3_refmod, "REFMOD_DATA_ROOT", tmp_path / "refmods")
    previous_directory = Path.cwd()
    assert h3_refmod.install_refmod_hooks(wgp) is patches
    assert Path.cwd() == previous_directory
    h3_refmod.install_refmod_hooks(wgp)
    assert calls == ["patch", "refresh"]
    assert storage.refmods_dir() == str(tmp_path / "refmods")
    assert (tmp_path / "refmods").is_dir()
    assert wgp.models_def["qwen"] == {}
    assert wgp.CUSTOM_SETTINGS_MAX == 6
    assert sys.modules["shared.extra_settings"]._CUSTOM_SETTINGS_MAX == 6
    assert wgp.get_model_custom_settings(wgp.models_def["custom_h3"])[-1]["id"] == patches.SETTING_EXTRACT


def test_plugin_import_failure_is_not_silently_ignored(plugin):
    wgp, patches, _, calls = plugin
    patches.install_patches = lambda: "pipeline unavailable"
    with pytest.raises(RuntimeError, match="pipeline unavailable"):
        h3_refmod.install_refmod_hooks(wgp)
    assert not getattr(wgp, "_modal_h3_refmod_installed", False)
    assert calls == []


def test_missing_catalog_settings_fail_instead_of_silently_discarding_refmods(plugin):
    wgp, _, _, _ = plugin
    wgp.refresh_model_defs = lambda: None
    with pytest.raises(RuntimeError, match="missing from the model catalog"):
        h3_refmod.install_refmod_hooks(wgp)
    assert not getattr(wgp, "_modal_h3_refmod_installed", False)


def test_refresh_failure_restores_working_directory(plugin):
    wgp, _, _, _ = plugin
    previous_directory = Path.cwd()

    def fail():
        raise ValueError("bad definition")

    wgp.refresh_model_defs = fail
    with pytest.raises(ValueError, match="bad definition"):
        h3_refmod.install_refmod_hooks(wgp)
    assert Path.cwd() == previous_directory
