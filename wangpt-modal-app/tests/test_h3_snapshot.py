import copy
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import app
import control
import h3_snapshot as snapshot
import snapshot_common as common
from wangpt_common import SINGULARITY_MODEL


@pytest.mark.parametrize("flag", [None, "0", "false", "true", "", "1"])
def test_snapshot_route_requires_explicit_opt_in_and_exact_video_model(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("WANGP_SINGULARITY_SNAPSHOT", raising=False)
    else:
        monkeypatch.setenv("WANGP_SINGULARITY_SNAPSHOT", flag)
    expected = "WanGPSingularityWorker" if flag == "1" else "WanGPVideoWorker"
    assert control.generation_worker_name("video", SINGULARITY_MODEL) == expected
    for model in ("minimax_h3_ref2va_pruned", "minimax_h3_vdn", SINGULARITY_MODEL + "_other"):
        assert control.generation_worker_name("video", model) == "WanGPVideoWorker"
    for kind in ("image", "audio"):
        assert control.generation_worker_name(kind, SINGULARITY_MODEL) == "WanGPImageWorker"


@pytest.mark.parametrize("dispatch_error", [False, True])
def test_submission_records_snapshot_route_and_dispatch_failure(monkeypatch, dispatch_error):
    records = {}
    spawned = []
    monkeypatch.setenv("WANGP_SINGULARITY_SNAPSHOT", "1")
    monkeypatch.setattr(control, "job_store", SimpleNamespace(
        put=lambda key, value: records.update({key: copy.deepcopy(value)})
    ))
    monkeypatch.setattr(control, "load_deployed_catalog", lambda: {
        "models": [{"model_type": SINGULARITY_MODEL, "main_output": ["video"]}]
    })

    def lookup(app_name, worker_name):
        assert worker_name == "WanGPSingularityWorker"
        if dispatch_error:
            raise RuntimeError("worker is not deployed")

        def spawn(*args):
            spawned.append(args)
            return SimpleNamespace(object_id="fc-snapshot")

        return lambda: SimpleNamespace(run=SimpleNamespace(spawn=spawn))

    monkeypatch.setattr(control.modal.Cls, "from_name", lookup)
    if dispatch_error:
        with pytest.raises(RuntimeError, match="not deployed"):
            control.submit_generation(SINGULARITY_MODEL, {"prompt": "ball"})
    else:
        result = control.submit_generation(SINGULARITY_MODEL, {"prompt": "ball"})
        assert spawned == [(result["id"], SINGULARITY_MODEL, {"prompt": "ball"})]
    record = next(iter(records.values()))
    assert record["worker"] == "WanGPSingularityWorker"
    assert record["status"] == ("failed" if dispatch_error else "queued")
    assert record["call_id"] == (None if dispatch_error else "fc-snapshot")


def test_worker_rejects_other_models_before_runtime_or_job_store_access():
    raw_run = app.WanGPSingularityWorker._get_user_cls().run._get_raw_f()
    with pytest.raises(ValueError, match="only accepts"):
        raw_run(SimpleNamespace(), "job", "minimax_h3_vdn", {})


@pytest.fixture
def warmup_env(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "create_reference", lambda path: path.write_bytes(b"reference"))
    transformer = SimpleNamespace(
        _loras_active_adapters=["0"],
        _loras_adapters={"0": "/data/loras/minimax_h3/" + snapshot.ACCELERATOR},
    )
    context = SimpleNamespace(
        model_type=SINGULARITY_MODEL,
        offloadobj=SimpleNamespace(models={"transformer": transformer}),
        model_def={"loras": ["https://example.invalid/" + snapshot.ACCELERATOR]},
        profile=4,
        config_id="",
    )
    wgp = SimpleNamespace(get_loaded_model_context=lambda: context, clear_gen_cache=lambda: None)
    session = SimpleNamespace(
        _output_dir=tmp_path,
        active_job=None,
        _state={"gen": {"file_list": ["warmup.mp4"], "queue": ["warmup"]}},
        _job_lock=threading.Lock(),
        _create_headless_state=lambda: {"gen": {"queue": [], "file_list": []}},
        get_default_settings=lambda model: {"model_type": model},
        _ensure_runtime=lambda: SimpleNamespace(module=wgp),
        _configure_runtime=lambda runtime: None,
    )
    env = SimpleNamespace(
        session=session, context=context, wgp=wgp, root=tmp_path, events=[],
        success=True, timeout=False, output=True, released=[],
    )

    def submit(task, callbacks):
        env.task = copy.deepcopy(task)
        output = session._output_dir / "warmup.mp4"
        output.parent.mkdir(parents=True)
        if env.output:
            output.write_bytes(b"video")
        result = SimpleNamespace(success=env.success, generated_files=[str(output)],
                                 errors=[] if env.success else [SimpleNamespace(message="out of memory")])
        done = threading.Event()
        job = SimpleNamespace(
            cancel=lambda: env.released.append("cancel"),
            release_input_payload=lambda: env.released.append("input"),
            release_output_payload=lambda: env.released.append("output"),
        )

        def work():
            callbacks.on_progress(SimpleNamespace(phase="inference", current_step=1))
            # Native WanGP unloads adapters before completing the task.
            context.offloadobj.models["transformer"]._loras_active_adapters = []
            session.active_job = None
            done.set()

        def get_result(timeout):
            done.wait(timeout=timeout)
            if env.timeout:
                raise TimeoutError("warmup timeout")
            return result

        job.result = get_result
        job._thread = threading.Thread(target=work)
        session.active_job = job
        env.job = job
        job._thread.start()
        return job

    session.submit_task = submit
    return env


def warm(env):
    return snapshot.warm_session(
        env.session, env.wgp, env.root,
        lambda stage, **details: env.events.append((stage, details)), timeout=1,
    )


def assert_clean(env):
    assert list(env.root.iterdir()) == []
    assert env.session._output_dir == env.root
    assert env.session._state == {"gen": {"queue": [], "file_list": []}}
    assert not env.job._thread.is_alive()
    assert env.released[-2:] == ["input", "output"]


def test_warmup_uses_native_preset_without_publication_and_retains_model(warmup_env, monkeypatch):
    env = warmup_env

    def forbidden(*args, **kwargs):
        pytest.fail("private warmup must not publish jobs or artifacts")

    monkeypatch.setattr(app.WanGPRuntime, "run", forbidden)
    monkeypatch.setattr(app, "job_store", SimpleNamespace(put=forbidden, get=forbidden))
    monkeypatch.setattr(app, "upload_output_artifacts", forbidden)
    state = warm(env)
    assert state["model"] == SINGULARITY_MODEL
    assert state["active_loras"] == []
    assert state["accelerator_verified_during_warmup"] is True
    assert env.wgp.get_loaded_model_context() is env.context
    params = env.task["params"]
    assert params["num_inference_steps"] == 4
    assert params["video_prompt_type"] == "I"
    assert params["seed"] == 42
    assert params["activated_loras"] == []  # preset LoRA is loaded natively
    assert params["_api"]["return_media"] is False
    assert not Path(params["image_refs"][0]).exists()
    assert_clean(env)


@pytest.mark.parametrize("failure", ["failed", "timeout", "no_output", "wrong_model", "missing_lora"])
def test_bad_warmup_aborts_capture_and_cleans_state(warmup_env, failure):
    env = warmup_env
    if failure == "failed":
        env.success = False
    elif failure == "timeout":
        env.timeout = True
    elif failure == "no_output":
        env.output = False
    elif failure == "wrong_model":
        env.context.model_type = "minimax_h3_vdn"
    else:
        env.context.offloadobj.models["transformer"]._loras_adapters = {"0": "other.safetensors"}
    with pytest.raises((RuntimeError, TimeoutError)):
        warm(env)
    assert "warmup_complete" not in [stage for stage, _ in env.events]
    assert_clean(env)


def test_active_session_cannot_be_snapshot(warmup_env):
    warmup_env.session.active_job = object()
    with pytest.raises(RuntimeError, match="active job"):
        warm(warmup_env)


def test_restore_refreshes_boot_and_rng_without_replacing_capture(monkeypatch):
    seeds = []
    monkeypatch.setitem(sys.modules, "wgp", SimpleNamespace())
    monkeypatch.setattr(common, "reseed_after_restore", lambda: seeds.append("reseed"))
    monkeypatch.setattr(snapshot, "loaded_state", lambda wgp: {"model": SINGULARITY_MODEL})
    monkeypatch.setattr(common, "emit_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(common, "memory_report", lambda wgp: resident_report())
    worker = SimpleNamespace(capture_id="captured", snapshot_revision="v1")
    restore = app.WanGPSingularityWorker._get_user_cls().after_restore._get_raw_f()
    restore(worker)
    first_boot = worker.boot_id
    worker.requests_served = 10
    worker.last_request = {"job_id": "previous"}
    restore(worker)
    assert seeds == ["reseed", "reseed"]
    assert worker.capture_id == "captured"
    assert worker.boot_id != first_boot
    assert worker.requests_served == 0
    assert worker.last_request is None


@pytest.mark.parametrize("reload_model", [False, True])
def test_snapshot_request_records_first_inference_and_actual_model_reuse(monkeypatch, reload_model):
    events = []
    wgp = SimpleNamespace(wan_model=object(), load_models=lambda: None)
    monkeypatch.setitem(sys.modules, "wgp", wgp)
    monkeypatch.setattr(common, "emit_snapshot", lambda stage, **values: events.append((stage, values)))
    params = {"prompt": "user prompt", "seed": 123, "num_inference_steps": 4}

    def run(job_id, model, settings, *, progress_observer):
        assert (job_id, model, settings) == ("job", SINGULARITY_MODEL, params)
        progress_observer(SimpleNamespace(phase="loading_model", current_step=1))
        progress_observer(SimpleNamespace(phase="inference", current_step=50, total_steps=50))
        assert not any(stage == "first_denoising_progress" for stage, _ in events)
        progress_observer(SimpleNamespace(phase="inference", current_step=0, total_steps=4))
        progress_observer(SimpleNamespace(phase="inference", current_step=1, total_steps=4))
        progress_observer(SimpleNamespace(phase="inference", current_step=2, total_steps=4))
        if reload_model:
            wgp.load_models()
            wgp.wan_model = object()
        return {"success": True}

    worker = SimpleNamespace(runtime=SimpleNamespace(run=run), boot_id="boot", requests_served=0)
    raw_run = app.WanGPSingularityWorker._get_user_cls().run._get_raw_f()
    assert raw_run(worker, "job", SINGULARITY_MODEL, params) == {"success": True}
    assert worker.last_request["model_reused"] is not reload_model
    assert worker.last_request["load_models_calls"] == int(reload_model)
    assert worker.last_request["first_denoising_seconds"] is not None
    assert [stage for stage, _ in events].count("first_denoising_progress") == 1
    assert worker.requests_served == 1


def resident_report():
    return {"cuda_allocated_bytes": int(19.6 * 1024**3),
            "mmgp_active_models": ["transformer"],
            "component_logical_elements_by_device": {
                "transformer": {"cuda:0": 20_111_439_344},
                "text_encoder": {"cpu": 25_157_829_884},
                "vae": {"cpu": 2_423_533_440}}}


def test_stage_transformer_uses_mmgp_only_after_idle_cleanup(monkeypatch):
    calls = []
    manager = SimpleNamespace(active_models_ids=[])
    manager.ensure_model_loaded = lambda name: calls.append(name)
    wgp = SimpleNamespace(offloadobj=manager)
    monkeypatch.setattr(snapshot, "loaded_state", lambda wgp: {"profile": 1})
    monkeypatch.setattr(snapshot, "memory_report", lambda wgp: resident_report() if calls else {
        "cuda_allocated_bytes": 33 * 1024**2})
    session = SimpleNamespace(active_job=None)
    result = snapshot.make_transformer_resident(session, wgp, lambda *a, **kw: None)
    assert calls == ["transformer"]
    assert result["cuda_delta_bytes"] > 19 * 1024**3
    calls.clear()
    session.active_job = object()
    with pytest.raises(RuntimeError, match="job is active"):
        snapshot.make_transformer_resident(session, wgp, lambda *a, **kw: None)
    assert not calls
    session.active_job = None
    manager.active_models_ids = ["vae"]
    with pytest.raises(RuntimeError, match="cleanup must finish"):
        snapshot.make_transformer_resident(session, wgp, lambda *a, **kw: None)
    assert not calls


@pytest.mark.parametrize("fault", ["low_allocation", "partial_transformer", "extra_component", "inactive"])
def test_residency_guard_rejects_partial_or_wrong_component_capture(fault):
    report = resident_report()
    if fault == "low_allocation":
        report["cuda_allocated_bytes"] = 33 * 1024**2
    elif fault == "partial_transformer":
        report["component_logical_elements_by_device"]["transformer"]["cpu"] = 100
    elif fault == "extra_component":
        report["component_logical_elements_by_device"]["vae"] = {"cuda:0": 100}
    else:
        report["mmgp_active_models"] = []
    with pytest.raises(RuntimeError):
        snapshot.validate_transformer_residency(report)


def test_restore_rejects_lost_gpu_residency_without_reloading(monkeypatch):
    wgp = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "wgp", wgp)
    report = resident_report()
    report["component_logical_elements_by_device"]["transformer"] = {"cpu": 20_111_439_344}
    monkeypatch.setattr(common, "memory_report", lambda wgp: report)
    monkeypatch.setattr(common, "reseed_after_restore", lambda: pytest.fail("must validate first"))
    restore = app.WanGPSingularityWorker._get_user_cls().after_restore._get_raw_f()
    with pytest.raises(RuntimeError, match="entirely on CUDA"):
        restore(SimpleNamespace())
