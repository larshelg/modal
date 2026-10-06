import copy
from importlib import import_module
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import control
krea_edit_app = import_module("wangpt-krea-edit-modal-app.app")
snapshot = import_module("wangpt-krea-edit-modal-app.snapshot")


def resident_report():
    return {"gpu_name": "NVIDIA L40S", "cuda_allocated_bytes": 12 * 1024**3,
            "mmgp_active_models": ["transformer"],
            "component_logical_elements_by_device": {
                "transformer": {"cuda:0": 12_000_000_000},
                "text_encoder": {"cpu": 4_000_000_000}, "vae": {"cpu": 1000},
                "vision_encoder": {"cpu": 1000}}}


@pytest.mark.parametrize("fault", ["missing_vision", "vision_gpu", "transformer_cpu", "wrong_gpu"])
def test_capture_requires_complete_edit_pipeline(fault):
    report = resident_report()
    snapshot.validate_residency(report)
    components = report["component_logical_elements_by_device"]
    if fault == "missing_vision":
        del components["vision_encoder"]
    elif fault == "vision_gpu":
        components["vision_encoder"] = {"cuda:0": 1000}
    elif fault == "transformer_cpu":
        components["transformer"]["cpu"] = 1
    else:
        report["gpu_name"] = "NVIDIA H100"
    with pytest.raises(RuntimeError):
        snapshot.validate_residency(report)


@pytest.mark.parametrize("enabled", [False, True])
def test_edit_dispatch_and_existing_routes(monkeypatch, enabled):
    monkeypatch.setenv("WANGP_KREA_EDIT_SNAPSHOT", "1" if enabled else "0")
    monkeypatch.setenv("WANGP_KREA_SNAPSHOT", "1")
    name = "WanGPKreaEditWorker" if enabled else "WanGPImageWorker"
    assert control.generation_worker_name("image", "krea2_turbo_edit") == name
    assert control.generation_worker_name("image", "krea2_turbo") == "WanGPKreaWorker"
    assert control.generation_worker_name("image", "krea2_raw_edit") == "WanGPImageWorker"
    assert control.generation_worker_name("video", "krea2_turbo_edit") == "WanGPVideoWorker"
    assert control.generation_worker_name("audio", "krea2_turbo_edit") == "WanGPImageWorker"
    records, calls = {}, []
    monkeypatch.setattr(control, "job_store", SimpleNamespace(put=lambda k, v: records.update({k: copy.deepcopy(v)})))
    monkeypatch.setattr(control, "load_deployed_catalog", lambda: {
        "models": [{"model_type": "krea2_turbo_edit", "main_output": ["image"]}]})

    def lookup(app, worker):
        calls.append((app, worker))
        return lambda: SimpleNamespace(run=SimpleNamespace(spawn=lambda *a: SimpleNamespace(object_id="fc-edit")))

    monkeypatch.setattr(control.modal.Cls, "from_name", lookup)
    result = control.submit_generation("krea2_turbo_edit", {"prompt": "make it green", "image_refs": ["/data/input.png"]}, "image")
    expected_app = control.KREA_EDIT_WORKER_APP_NAME if enabled else control.WORKER_APP_NAME
    assert calls == [(expected_app, name)]
    assert records[result["id"]]["call_id"] == "fc-edit"
    assert records[result["id"]]["worker_app"] == expected_app


def test_worker_rejects_base_model_before_runtime_access():
    run = krea_edit_app.WanGPKreaEditWorker._get_user_cls().run._get_raw_f()
    with pytest.raises(ValueError, match="only accepts"):
        run(SimpleNamespace(), "job", "krea2_turbo", {})


def test_restore_validates_vision_before_reseed(monkeypatch):
    monkeypatch.setitem(sys.modules, "wgp", SimpleNamespace())
    report = resident_report()
    monkeypatch.setattr(krea_edit_app, "memory_report", lambda w: copy.deepcopy(report))
    monkeypatch.setattr(krea_edit_app, "loaded_state", lambda w: {})
    monkeypatch.setattr(krea_edit_app, "emit_snapshot", lambda *a, **kw: None)
    seeds = []
    monkeypatch.setattr(krea_edit_app, "reseed_after_restore", lambda: seeds.append(1))
    worker = SimpleNamespace(capture_id="capture", snapshot_revision="v1")
    restore = krea_edit_app.WanGPKreaEditWorker._get_user_cls().after_restore._get_raw_f()
    restore(worker)
    first_boot = worker.boot_id
    restore(worker)
    assert first_boot != worker.boot_id and worker.capture_id == "capture"
    del report["component_logical_elements_by_device"]["vision_encoder"]
    with pytest.raises(RuntimeError, match="vision_encoder"):
        restore(worker)
    assert seeds == [1, 1]


@pytest.mark.parametrize("outcome", ["success", "failed", "missing_lora", "missing_image"])
def test_edit_warmup_exercises_reference_and_lora_and_releases_payload(tmp_path, monkeypatch, outcome):
    monkeypatch.setattr(snapshot, "emit_snapshot", lambda *a, **kw: None)
    monkeypatch.setattr(snapshot, "create_reference", lambda p: p.write_bytes(b"reference"))
    monkeypatch.setattr(snapshot, "loaded_state", lambda w: {"model": snapshot.KREA_EDIT_MODEL, "profile": 1})
    monkeypatch.setattr(snapshot, "active_lora_files", lambda c: [] if outcome == "missing_lora" else [snapshot.EDIT_LORA])
    released = []
    wgp = SimpleNamespace(clear_gen_cache=lambda: released.append("cache"),
                          get_loaded_model_context=lambda: SimpleNamespace(model_type=snapshot.KREA_EDIT_MODEL))
    session = SimpleNamespace(active_job=None, _output_dir=tmp_path, _state={"old": "payload"},
        _job_lock=threading.Lock(), _create_headless_state=lambda: {"gen": {}},
        _configure_runtime=lambda r: None, _ensure_runtime=lambda: wgp,
        get_default_settings=lambda model: {"model_type": model})

    def submit(task, callbacks):
        params = task["params"]
        assert params["model_type"] == "krea2_turbo_edit"
        assert params["video_prompt_type"] == "KI" and params["num_inference_steps"] == 8
        assert len(params["image_refs"]) == 1 and Path(params["image_refs"][0]).is_file()
        assert params["activated_loras"] == []  # Preset adapter is loaded by WanGP.
        callbacks.on_progress(SimpleNamespace(phase="inference"))
        path = session._output_dir / "warmup.png"
        path.parent.mkdir(parents=True)
        if outcome != "missing_image":
            path.write_bytes(b"image")
        thread = threading.Thread(target=lambda: None)
        thread.start()
        return SimpleNamespace(_thread=thread,
            result=lambda timeout: SimpleNamespace(success=outcome != "failed", generated_files=[str(path)],
                errors=[] if outcome != "failed" else [SimpleNamespace(message="generation failed")]),
            release_input_payload=lambda: released.append("input"),
            release_output_payload=lambda: released.append("output"))

    session.submit_task = submit
    if outcome == "success":
        assert snapshot.warm_session(session, wgp, tmp_path)["edit_lora_verified_during_warmup"]
    else:
        with pytest.raises(RuntimeError):
            snapshot.warm_session(session, wgp, tmp_path)
    assert session._output_dir == tmp_path and session._state == {"gen": {}}
    assert released == ["input", "output", "cache"]
    assert list(tmp_path.iterdir()) == []


def test_loaded_state_requires_edit_preset_lora():
    context = SimpleNamespace(model_type=snapshot.KREA_EDIT_MODEL, profile=1, config_id="default",
        model_def={"loras": []})
    wgp = SimpleNamespace(get_loaded_model_context=lambda: context)
    with pytest.raises(RuntimeError, match="Identity Edit LoRA"):
        snapshot.loaded_state(wgp)
