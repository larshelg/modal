from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import app
from fizgig_s3 import upload_artifact
from test_s3 import FakeS3
from test_refmod import intent, local_paths


@pytest.mark.parametrize("kind", ["lora", "refmod"])
def test_verified_artifact_uri_and_idempotent_retry(tmp_path, kind):
    output = tmp_path / "test.safetensors"
    output.write_bytes(b"model")
    client = FakeS3({})
    original_upload = client.upload_file
    client.upload_file = Mock(side_effect=original_upload)
    result = upload_artifact(client, "bucket", "job-1", output, kind)
    assert result == {
        "storage": "s3", "bucket": "bucket", "key": "runninghub/fizgig/job-1/000-test.safetensors",
        "uri": "s3://bucket/runninghub/fizgig/job-1/000-test.safetensors", "filename": "test.safetensors",
        "size_bytes": 5, "media_type": "application/octet-stream", "sha256": sha256(b"model").hexdigest(),
        "artifact_type": kind,
    }
    assert upload_artifact(client, "bucket", "job-1", output, kind) == result
    assert client.upload_file.call_count == 1
    assert output.read_bytes() == b"model"
    output.write_bytes(b"different model")
    with pytest.raises(FileExistsError):
        upload_artifact(client, "bucket", "job-1", output, kind)
    assert client.upload_file.call_count == 1


@pytest.mark.parametrize("head", [
    {"ContentLength": 0, "Metadata": {}},
    {"ContentLength": 5, "Metadata": {"sha256": "wrong"}},
])
def test_failed_verification_never_returns_link(tmp_path, head):
    path = tmp_path / "test.safetensors"
    path.write_bytes(b"model")
    client = FakeS3({})
    original_head = client.head_object
    def corrupt_head(**kw):
        value = original_head(**kw)  # first call must still report missing
        return head
    client.head_object = corrupt_head
    with pytest.raises(RuntimeError, match="verification failed"):
        upload_artifact(client, "bucket", "job-1", path, "refmod")
    assert path.read_bytes() == b"model"


def test_permission_error_does_not_trigger_upload(tmp_path):
    path = tmp_path / "test.safetensors"
    path.write_bytes(b"model")
    client = FakeS3({})
    error = RuntimeError("access denied")
    error.response = {"Error": {"Code": "403"}}
    client.head_object = Mock(side_effect=error)
    client.upload_file = Mock()
    with pytest.raises(RuntimeError, match="access denied"):
        upload_artifact(client, "bucket", "job-1", path, "lora")
    client.upload_file.assert_not_called()


@pytest.fixture
def completed_job(tmp_path, monkeypatch):
    paths = local_paths(tmp_path)
    paths["run_dir"].mkdir()
    (paths["run_dir"] / "person_ref.safetensors").write_bytes(b"model")
    request = intent()
    result = {"artifact_path": str(paths["promoted_refmod"]), "paused": False, "size_bytes": 5}
    records = {"job-1": {"id": "job-1", "status": "succeeded", "request": request,
                         "result": result, "completed_at": "original-time", "progress": {"phase": "completed"}}}
    monkeypatch.setattr(app, "job_store", SimpleNamespace(
        get=lambda key, default=None: deepcopy(records.get(key, default)),
        put=lambda key, value: records.update({key: deepcopy(value)})))
    monkeypatch.setattr(app, "paths_for_request", lambda request: paths)
    monkeypatch.setattr(app, "data_volume", SimpleNamespace(reload=Mock(), commit=Mock()))
    client = FakeS3({})
    client.close = Mock()
    monkeypatch.setattr(app, "create_s3_client", lambda: (client, "bucket"))
    return records, client


def test_cpu_publish_backfills_completed_job(completed_job):
    records, client = completed_job
    record = app.publish_artifact.local("job-1")
    assert record["status"] == "succeeded"
    assert record["completed_at"] == "original-time"
    assert record["result"]["artifact_uri"].startswith("s3://bucket/runninghub/fizgig/job-1/")
    assert record["result"]["outputs"][0]["artifact_type"] == "refmod"
    assert records["job-1"] == record
    client.close.assert_called_once()


def test_cpu_publish_recovers_failed_upload(completed_job):
    records, client = completed_job
    records["job-1"].update(status="failed", progress={"phase": "uploading_artifact"}, error={"message": "timeout"})
    record = app.publish_artifact.local("job-1")
    assert record["status"] == "succeeded" and record["error"] is None
    assert record["progress"]["phase"] == "completed"
    assert record["result"]["artifact_uri"]


@pytest.mark.parametrize("status,phase,paused", [
    ("running", "uploading_artifact", False), ("queued", "queued", False),
    ("failed", "training", False), ("cancelled", "uploading_artifact", False),
    ("succeeded", "paused", True),
])
def test_cpu_publish_rejects_ineligible_jobs(completed_job, status, phase, paused):
    records, client = completed_job
    records["job-1"].update(status=status, progress={"phase": phase})
    records["job-1"]["result"]["paused"] = paused
    with pytest.raises(ValueError, match="only completed"):
        app.publish_artifact.local("job-1")
    assert not hasattr(client, "artifacts")


def test_publish_rejects_unrelated_artifact_path(completed_job):
    records, client = completed_job
    records["job-1"]["result"]["artifact_path"] = "/data/something-else.safetensors"
    with pytest.raises(ValueError, match="does not match"):
        app.publish_artifact.local("job-1")
    assert not hasattr(client, "artifacts")


def test_paused_results_never_create_s3_client(monkeypatch):
    create = Mock(side_effect=AssertionError("should not contact S3"))
    monkeypatch.setattr(app, "create_s3_client", create)
    result = {"paused": True, "resume_from": "person_ref-000001-state"}
    assert app._publish_result("job-1", intent(), result) == result
    create.assert_not_called()


@pytest.mark.parametrize("fail_upload", [False, True])
def test_worker_waits_for_verified_upload_before_success(completed_job, monkeypatch, fail_upload):
    records, client = completed_job
    local_result = deepcopy(records["job-1"]["result"])
    records["job-1"]["status"] = "queued"
    monkeypatch.setattr(app, "_verify_models", lambda request: None)
    monkeypatch.setattr(app, "_prepare_run", lambda *args, **kw: None)
    monkeypatch.setattr(app, "build_pipeline_commands", lambda *args: [])
    monkeypatch.setattr(app, "_finalize_run", lambda *args: local_result)
    original_upload = client.upload_file
    def upload(*args, **kwargs):
        assert records["job-1"]["status"] == "running"
        assert records["job-1"]["progress"]["phase"] == "uploading_artifact"
        assert records["job-1"]["result"] == local_result
        if fail_upload:
            raise RuntimeError("S3 upload failed")
        return original_upload(*args, **kwargs)
    client.upload_file = upload
    if fail_upload:
        with pytest.raises(RuntimeError, match="S3 upload failed"):
            app.run_training.local("job-1", intent())
        assert records["job-1"]["status"] == "failed"
        assert records["job-1"]["result"] == local_result
        assert "artifact_uri" not in local_result
        client.upload_file = original_upload
        retried = app.publish_artifact.local("job-1")
        assert retried["status"] == "succeeded" and retried["result"]["artifact_uri"]
    else:
        record = app.run_training.local("job-1", intent())
        assert record["status"] == "succeeded"
        assert record["result"]["artifact_uri"] == record["result"]["outputs"][0]["uri"]
