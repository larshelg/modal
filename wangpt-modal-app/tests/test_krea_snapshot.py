import copy
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import control
import krea_app
import krea_snapshot as snapshot


def resident_report():
    return {"gpu_name": "NVIDIA L40S", "cuda_allocated_bytes": 12 * 1024**3,
            "mmgp_active_models": ["transformer"],
            "component_logical_elements_by_device": {
                "transformer": {"cuda:0": 12_000_000_000},
                "text_encoder": {"cpu": 4_000_000_000}, "vae": {"cpu": 1000}}}


@pytest.mark.parametrize("fault", ["cpu_tensor", "other_gpu", "no_weights", "vae_gpu", "empty", "inactive"])
def test_residency_rejects_incomplete_or_wrong_gpu_snapshot(fault):
    report = resident_report()
    snapshot.validate_residency(report)
    if fault == "cpu_tensor":
        report["component_logical_elements_by_device"]["transformer"]["cpu"] = 1
    elif fault == "other_gpu":
        report["gpu_name"] = "NVIDIA H100"
    elif fault == "no_weights":
        report["cuda_allocated_bytes"] = 33 * 1024**2
    elif fault == "vae_gpu":
        report["component_logical_elements_by_device"]["vae"] = {"cuda:0": 1000}
    elif fault == "empty":
        del report["component_logical_elements_by_device"]["transformer"]
    else:
        report["mmgp_active_models"] = []
    with pytest.raises(RuntimeError):
        snapshot.validate_residency(report)


@pytest.mark.parametrize("enabled", [False, True])
def test_route_is_opt_in_exact_image_model_and_dispatches_separate_app(monkeypatch, enabled):
    monkeypatch.setenv("WANGP_KREA_SNAPSHOT", "1" if enabled else "0")
    name = "WanGPKreaWorker" if enabled else "WanGPImageWorker"
    assert control.generation_worker_name("image", "krea2_turbo") == name
    assert control.generation_worker_name("image", "krea2_turbo_edit") == "WanGPImageWorker"
    assert control.generation_worker_name("audio", "krea2_turbo") == "WanGPImageWorker"
    assert control.generation_worker_name("video", "krea2_turbo") == "WanGPVideoWorker"
    records = {}
    monkeypatch.setattr(control, "job_store", SimpleNamespace(put=lambda k, v: records.update({k: copy.deepcopy(v)})))
    monkeypatch.setattr(control, "load_deployed_catalog", lambda: {
        "models": [{"model_type": "krea2_turbo", "main_output": ["image"]}]})
    calls = []

    def lookup(app, worker):
        calls.append((app, worker))
        return lambda: SimpleNamespace(run=SimpleNamespace(spawn=lambda *a: SimpleNamespace(object_id="fc-krea")))

    monkeypatch.setattr(control.modal.Cls, "from_name", lookup)
    result = control.submit_krea_generation({"prompt": "a teapot"})
    expected_app = control.KREA_WORKER_APP_NAME if enabled else control.WORKER_APP_NAME
    assert calls == [(expected_app, name)]
    assert records[result["id"]]["call_id"] == "fc-krea"


def test_worker_rejects_edit_preset_before_job_store_access():
    run = krea_app.WanGPKreaWorker._get_user_cls().run._get_raw_f()
    with pytest.raises(ValueError, match="only accepts"):
        run(SimpleNamespace(), "job", "krea2_turbo_edit", {})


def test_restore_validates_before_rng_and_never_loads(monkeypatch):
    monkeypatch.setitem(sys.modules, "wgp", SimpleNamespace())
    report = resident_report()
    monkeypatch.setattr(krea_app, "memory_report", lambda w: copy.deepcopy(report))
    monkeypatch.setattr(krea_app, "loaded_state", lambda w: {})
    monkeypatch.setattr(krea_app, "emit_snapshot", lambda *a, **kw: None)
    seeds = []
    monkeypatch.setattr(krea_app, "reseed_after_restore", lambda: seeds.append(1))
    worker = SimpleNamespace(capture_id="capture", snapshot_revision="v1")
    restore = krea_app.WanGPKreaWorker._get_user_cls().after_restore._get_raw_f()
    restore(worker)
    first_boot = worker.boot_id
    restore(worker)
    assert first_boot != worker.boot_id
    assert worker.capture_id == "capture"
    assert worker.requests_served == 0 and worker.last_request is None
    report["component_logical_elements_by_device"]["transformer"] = {"cpu": 12_000_000_000}
    with pytest.raises(RuntimeError):
        restore(worker)
    assert seeds == [1, 1]


@pytest.mark.parametrize("success", [True, False])
def test_native_image_warmup_releases_payload_and_clears_state(tmp_path, monkeypatch, success):
    monkeypatch.setattr(snapshot, "emit_snapshot", lambda *a, **kw: None)
    monkeypatch.setattr(snapshot, "loaded_state", lambda w: {"model": "krea2_turbo", "profile": 1})
    released = []
    wgp = SimpleNamespace(clear_gen_cache=lambda: released.append("cache"))
    session = SimpleNamespace(active_job=None, _output_dir=tmp_path, _state={"old": "payload"},
        _job_lock=threading.Lock(), _create_headless_state=lambda: {"gen": {}},
        _configure_runtime=lambda r: None, _ensure_runtime=lambda: wgp,
        get_default_settings=lambda model: {"model_type": model})

    def submit(task):
        assert task["params"]["model_type"] == "krea2_turbo"
        assert task["params"]["num_inference_steps"] == 8
        path = session._output_dir / "warmup.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"image")
        thread = threading.Thread(target=lambda: None)
        thread.start()
        return SimpleNamespace(_thread=thread,
            result=lambda timeout: SimpleNamespace(success=success, generated_files=[str(path)],
                errors=[] if success else [SimpleNamespace(message="generation failed")]),
            release_input_payload=lambda: released.append("input"),
            release_output_payload=lambda: released.append("output"))

    session.submit_task = submit
    if success:
        assert snapshot.warm_session(session, wgp, tmp_path)["profile"] == 1
    else:
        with pytest.raises(RuntimeError, match="generation failed"):
            snapshot.warm_session(session, wgp, tmp_path)
    assert session._output_dir == tmp_path
    assert session._state == {"gen": {}}
    assert released == ["input", "output", "cache"]
    assert list(tmp_path.iterdir()) == []
