import copy
from importlib import import_module
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import control
qwen_noctq_app = import_module("wangpt-qwen-noctq-modal-app.app")
snapshot = import_module("wangpt-qwen-noctq-modal-app.snapshot")


def resident_report():
    return {"gpu_name": "NVIDIA L40S", "cuda_allocated_bytes": 7 * 1024**3,
            "mmgp_active_models": ["transformer"],
            "component_logical_elements_by_device": {
                "transformer": {"cuda:0": 7_000_000_000},
                "text_encoder": {"cpu": 8_000_000_000}, "vae": {"cpu": 1000},
                "vision_encoder": {"cpu": 1000}}}


@pytest.mark.parametrize("fault", ["missing_vision", "vision_gpu", "transformer_cpu", "wrong_gpu", "empty", "inactive"])
def test_snapshot_requires_complete_qwen_pipeline(fault):
    report = resident_report()
    snapshot.validate_residency(report)
    components = report["component_logical_elements_by_device"]
    if fault == "missing_vision":
        del components["vision_encoder"]
    elif fault == "vision_gpu":
        components["vision_encoder"] = {"cuda:0": 1000}
    elif fault == "transformer_cpu":
        components["transformer"]["cpu"] = 1
    elif fault == "wrong_gpu":
        report["gpu_name"] = "NVIDIA H100"
    elif fault == "empty":
        report["cuda_allocated_bytes"] = 1024
    else:
        report["mmgp_active_models"] = []
    with pytest.raises(RuntimeError):
        snapshot.validate_residency(report)


@pytest.mark.parametrize("enabled", [False, True])
def test_qwen_dispatch_is_exact_and_opt_in(monkeypatch, enabled):
    monkeypatch.setenv("WANGP_QWEN_NOCTQ_SNAPSHOT", "1" if enabled else "0")
    monkeypatch.setenv("WANGP_QWEN_IMAGE_21_SNAPSHOT", "1")
    monkeypatch.setenv("WANGP_KREA_SNAPSHOT", "1")
    monkeypatch.setenv("WANGP_KREA_EDIT_SNAPSHOT", "1")
    name = "WanGPQwenNoctQWorker" if enabled else "WanGPImageWorker"
    assert control.generation_worker_name("image", "qwen_image_21_noctq_v4") == name
    assert control.generation_worker_name("image", "qwen_image_21_7B") == "WanGPQwenImage21Worker"
    assert control.generation_worker_name("image", "krea2_turbo") == "WanGPKreaWorker"
    assert control.generation_worker_name("image", "krea2_turbo_edit") == "WanGPKreaEditWorker"
    assert control.generation_worker_name("audio", "qwen_image_21_noctq_v4") == "WanGPImageWorker"
    assert control.generation_worker_name("video", "qwen_image_21_noctq_v4") == "WanGPVideoWorker"
    records, calls = {}, []
    monkeypatch.setattr(control, "job_store", SimpleNamespace(put=lambda k, v: records.update({k: copy.deepcopy(v)})))
    monkeypatch.setattr(control, "load_deployed_catalog", lambda: {
        "models": [{"model_type": "qwen_image_21_noctq_v4", "main_output": ["image"]}]})

    def lookup(app, worker):
        calls.append((app, worker))
        return lambda: SimpleNamespace(run=SimpleNamespace(spawn=lambda *a: SimpleNamespace(object_id="fc-qwen")))

    monkeypatch.setattr(control.modal.Cls, "from_name", lookup)
    result = control.submit_generation("qwen_image_21_noctq_v4", {"prompt": "a teapot"}, "image")
    expected_app = control.QWEN_NOCTQ_WORKER_APP_NAME if enabled else control.WORKER_APP_NAME
    assert calls == [(expected_app, name)]
    assert records[result["id"]]["call_id"] == "fc-qwen"
    assert records[result["id"]]["worker_app"] == expected_app


def test_worker_rejects_base_model_before_runtime_access():
    run = qwen_noctq_app.WanGPQwenNoctQWorker._get_user_cls().run._get_raw_f()
    with pytest.raises(ValueError, match="only accepts"):
        run(SimpleNamespace(), "job", "qwen_image_21_7B", {})


def test_restore_validates_vision_before_reseed(monkeypatch):
    monkeypatch.setitem(sys.modules, "wgp", SimpleNamespace())
    report = resident_report()
    monkeypatch.setattr(qwen_noctq_app, "memory_report", lambda w: copy.deepcopy(report))
    monkeypatch.setattr(qwen_noctq_app, "loaded_state", lambda w: {})
    monkeypatch.setattr(qwen_noctq_app, "emit_snapshot", lambda *a, **kw: None)
    seeds = []
    monkeypatch.setattr(qwen_noctq_app, "reseed_after_restore", lambda: seeds.append(1))
    worker = SimpleNamespace(capture_id="capture", snapshot_revision="v1")
    restore = qwen_noctq_app.WanGPQwenNoctQWorker._get_user_cls().after_restore._get_raw_f()
    restore(worker)
    first_boot = worker.boot_id
    restore(worker)
    assert first_boot != worker.boot_id and worker.capture_id == "capture"
    del report["component_logical_elements_by_device"]["vision_encoder"]
    with pytest.raises(RuntimeError, match="vision_encoder"):
        restore(worker)
    assert seeds == [1, 1]


@pytest.mark.parametrize("outcome", ["success", "failed", "missing_image"])
def test_reference_warmup_and_payload_cleanup(tmp_path, monkeypatch, outcome):
    monkeypatch.setattr(snapshot, "emit_snapshot", lambda *a, **kw: None)
    monkeypatch.setattr(snapshot, "create_reference", lambda p: p.write_bytes(b"reference"))
    monkeypatch.setattr(snapshot, "loaded_state", lambda w: {"model": snapshot.QWEN_NOCTQ_MODEL, "profile": 1})
    released = []
    wgp = SimpleNamespace(clear_gen_cache=lambda: released.append("cache"))
    session = SimpleNamespace(active_job=None, _output_dir=tmp_path, _state={"old": "payload"},
        _job_lock=threading.Lock(), _create_headless_state=lambda: {"gen": {}},
        _configure_runtime=lambda r: None, _ensure_runtime=lambda: wgp,
        get_default_settings=lambda model: {"model_type": model})

    def submit(task):
        params = task["params"]
        assert params["model_type"] == "qwen_image_21_noctq_v4"
        assert params["video_prompt_type"] == "KI" and params["num_inference_steps"] == 25
        assert len(params["image_refs"]) == 1 and Path(params["image_refs"][0]).is_file()
        assert params["guidance_scale"] == 3 and params["activated_loras"] == []
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
        assert snapshot.warm_session(session, wgp, tmp_path)["profile"] == 1
    else:
        with pytest.raises(RuntimeError):
            snapshot.warm_session(session, wgp, tmp_path)
    assert session._output_dir == tmp_path and session._state == {"gen": {}}
    assert released == ["input", "output", "cache"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("profile,loras", [(4, []), (1, ["user.safetensors"])])
def test_staging_rejects_wrong_profile_and_active_loras(monkeypatch, profile, loras):
    monkeypatch.setattr(snapshot, "loaded_state", lambda w: {"profile": profile, "active_loras": loras})
    wgp = SimpleNamespace(offloadobj=SimpleNamespace(active_models_ids=[]))
    with pytest.raises(RuntimeError, match="profile 1"):
        snapshot.make_transformer_resident(SimpleNamespace(active_job=None), wgp)


@pytest.mark.parametrize("urls", [[], ["https://example.com/base.safetensors"],
    [snapshot.CHECKPOINT_URL.replace(snapshot.CHECKPOINT_REVISION, "main")]])
def test_loaded_state_rejects_missing_or_unpinned_checkpoint(urls):
    context = SimpleNamespace(model_type=snapshot.QWEN_NOCTQ_MODEL, model_def={"URLs": urls})
    with pytest.raises(RuntimeError, match="pinned V4 checkpoint"):
        snapshot.loaded_state(SimpleNamespace(get_loaded_model_context=lambda: context))


def test_loaded_state_accepts_existing_finetune_definition(monkeypatch):
    import json
    model_def = json.loads(Path("finetunes/qwen_image_21_noctq_v4.json").read_text())["model"]
    context = SimpleNamespace(model_type=snapshot.QWEN_NOCTQ_MODEL, model_def=model_def, profile=1, config_id="")
    monkeypatch.setattr(snapshot, "active_lora_files", lambda c: [])
    state = snapshot.loaded_state(SimpleNamespace(get_loaded_model_context=lambda: context))
    assert state["checkpoint"] == "NoctQ_V4_int8_convrot.safetensors"
    assert state["checkpoint_revision"] == snapshot.CHECKPOINT_REVISION
