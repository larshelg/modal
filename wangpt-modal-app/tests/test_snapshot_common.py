from types import SimpleNamespace

import pytest

import snapshot_common as common


def test_load_tracking_restores_native_loader_after_failure():
    def original():
        raise ValueError("load failed")
    wgp = SimpleNamespace(load_models=original)
    with pytest.raises(ValueError):
        with common.track_model_loads(wgp) as counts:
            wgp.load_models()
    assert counts["load_models_calls"] == 1
    assert wgp.load_models is original
