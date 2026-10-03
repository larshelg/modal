from copy import deepcopy
from types import SimpleNamespace

import pytest

import control
from fizgig_common import training_intent, validate_training_request


class Store:
    def __init__(self):
        self.values = {}

    def get(self, key, default=None):
        return deepcopy(self.values.get(key, default))

    def put(self, key, value):
        self.values[key] = deepcopy(value)


@pytest.fixture
def store(monkeypatch):
    store = Store()
    monkeypatch.setattr(control, "job_store", store)
    return store


def request(**overrides):
    return {"family": "krea2", "dataset": "linda", "output_name": "linda_v1",
            "preset": "krea2_defaults", "trigger_word": "linda", **overrides}


def worker(monkeypatch, spawn):
    def lookup(app, name):
        assert (app, name) == ("fizgig-modal-app", "run_training")
        return SimpleNamespace(spawn=spawn)
    monkeypatch.setattr(control.modal.Function, "from_name", lookup)


def seed(store, status="running", call_id="fc-test", **extra):
    store.put("job-1", {"id": "job-1", "status": status, "request": request(), **extra})
    store.put("submission:job-1", {"request": request(), "call_id": call_id})


def call(monkeypatch, **methods):
    def lookup(call_id):
        assert call_id == "fc-test"
        return SimpleNamespace(**methods)
    monkeypatch.setattr(control.modal.FunctionCall, "from_id", lookup)


def raises(exc):
    def invoke(*args, **kwargs):
        raise exc
    return invoke


@pytest.mark.parametrize("state", ["running", "succeeded"])
def test_dispatch_preserves_fast_worker_progress(store, monkeypatch, state):
    def spawn(job_id, intent):
        assert intent == training_intent(request())
        assert "network_dim" not in intent
        assert store.get(job_id)["status"] == "queued"
        store.put(job_id, {"id": job_id, "status": state, "progress": {"phase": "training"}})
        return SimpleNamespace(object_id="fc-test")
    worker(monkeypatch, spawn)
    submitted = control.submit_training(request())
    assert submitted["call_id"] == "fc-test"
    assert submitted["status"] == "queued"
    assert store.get(submitted["id"])["status"] == state
    assert store.get(f"submission:{submitted['id']}")["call_id"] == "fc-test"


@pytest.mark.parametrize("invalid", [
    {"dataset": "../escape"}, {"family": "unknown"}, {"family": []},
    {"preset": "h3_character_fast"}, {"preset": {}}, {"epochs": True},
    {"epochs": 0}, {"epochs": 501}, {"cli_args": ["--foo"]}, {"resume_from": "latest"},
])
def test_invalid_request_never_creates_job(store, monkeypatch, invalid):
    worker(monkeypatch, raises(AssertionError("must not dispatch")))
    with pytest.raises(ValueError):
        control.submit_training(request(**invalid))
    assert not store.values


def test_dispatch_failure_is_recorded(store, monkeypatch):
    worker(monkeypatch, raises(RuntimeError("deployment missing")))
    with pytest.raises(RuntimeError, match="deployment missing"):
        control.submit_training(request())
    records = [value for key, value in store.values.items() if not key.startswith("submission:")]
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["error"]["stage"] == "dispatch"


def test_poll_running_job_filters_log_tail(store, monkeypatch):
    seed(store, progress={"phase": "training", "epoch": 3, "log_tail": ["diagnostics"]})
    def get(timeout):
        assert timeout == 0
        raise TimeoutError
    call(monkeypatch, get=get)
    result = control.get_training_job("job-1")
    assert result["status"] == "running"
    assert result["call_id"] == "fc-test"
    assert result["progress"] == {"phase": "training", "epoch": 3}
    assert control.get_training_job("job-1", logs=True)["progress"]["log_tail"] == ["diagnostics"]


def test_startup_failure_becomes_failed(store, monkeypatch):
    seed(store, status="queued")
    call(monkeypatch, get=raises(RuntimeError("container failed to start")))
    result = control.get_training_job("job-1")
    assert result["status"] == "failed"
    assert result["error"]["stage"] == "modal"
    assert store.get("job-1")["status"] == "failed"


def test_poll_preserves_worker_terminal_result_on_exception(store, monkeypatch):
    seed(store)
    def get(timeout):
        seed(store, status="succeeded", result={"paused": False, "artifact_path": "/data/loras/test"})
        raise RuntimeError("late transport failure")
    call(monkeypatch, get=get)
    assert control.get_training_job("job-1")["status"] == "succeeded"


def test_poll_recovers_terminal_return_value(store, monkeypatch):
    seed(store)
    call(monkeypatch, get=lambda timeout: {"status": "succeeded", "result": {"paused": True}})
    result = control.get_training_job("job-1")
    assert result["result"]["paused"] is True
    assert store.get("job-1")["status"] == "succeeded"


def test_expired_output_does_not_invent_failure(store, monkeypatch):
    seed(store)
    call(monkeypatch, get=raises(control.modal.exception.OutputExpiredError("expired")))
    with pytest.raises(ValueError, match="result expired"):
        control.get_training_job("job-1")
    assert store.get("job-1")["status"] == "running"


def test_pause_calls_deployed_function_and_remains_running(store, monkeypatch):
    seed(store)
    call(monkeypatch, get=raises(TimeoutError()))
    def lookup(app, name):
        assert (app, name) == ("fizgig-modal-app", "request_pause")
        def remote(job_id):
            record = store.get(job_id)
            record["pause_requested"] = True
            store.put(job_id, record)
            return record
        return SimpleNamespace(remote=remote)
    monkeypatch.setattr(control.modal.Function, "from_name", lookup)
    result = control.pause_training_job("job-1")
    assert result["status"] == "running"
    assert result["pause_requested"] is True


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_preserves_intent_and_uses_new_job_id(store, monkeypatch, legacy):
    seed(store, status="succeeded", result={"paused": True, "resume_from": "linda_v1-000010-state"})
    if legacy:
        del store.values["submission:job-1"]
        record = store.get("job-1")
        record["request"] = validate_training_request(request())
        store.put("job-1", record)
    def spawn(job_id, intent):
        assert job_id != "job-1"
        assert intent == {**training_intent(request()), "resume_from": "linda_v1-000010-state"}
        return SimpleNamespace(object_id="fc-resumed")
    worker(monkeypatch, spawn)
    result = control.resume_training_job("job-1")
    assert result["resumed_from"] == "job-1"
    assert result["call_id"] == "fc-resumed"
    assert store.get("job-1")["result"]["paused"] is True


@pytest.mark.parametrize("state,paused", [("running", False), ("failed", True), ("succeeded", False)])
def test_resume_rejects_jobs_without_completed_pause(store, monkeypatch, state, paused):
    seed(store, status=state, result={"paused": paused})
    call(monkeypatch, get=raises(TimeoutError()))
    with pytest.raises(ValueError, match="successfully paused"):
        control.resume_training_job("job-1")


def test_cancel_terminates_call_then_persists_status(store, monkeypatch):
    seed(store)
    def cancel(terminate_containers):
        assert terminate_containers is True
    call(monkeypatch, cancel=cancel)
    assert control.cancel_training_job("job-1")["status"] == "cancelled"
    assert store.get("job-1")["status"] == "cancelled"
    assert control.cancel_training_job("job-1")["status"] == "cancelled"


def test_cancel_preserves_racing_completion(store, monkeypatch):
    seed(store)
    def cancel(terminate_containers):
        seed(store, status="succeeded", result={"paused": False})
    call(monkeypatch, cancel=cancel)
    assert control.cancel_training_job("job-1")["status"] == "succeeded"


def test_cancel_failure_keeps_job_running(store, monkeypatch):
    seed(store)
    call(monkeypatch, cancel=raises(RuntimeError("unavailable")))
    with pytest.raises(RuntimeError, match="unavailable"):
        control.cancel_training_job("job-1")
    assert store.get("job-1")["status"] == "running"


def test_cancel_requires_a_call_id(store):
    seed(store, call_id=None)
    with pytest.raises(RuntimeError, match="no FunctionCall ID"):
        control.cancel_training_job("job-1")
    assert store.get("job-1")["status"] == "running"


def test_missing_job_is_reported(store):
    with pytest.raises(ValueError, match="not found"):
        control.get_training_job("missing")
