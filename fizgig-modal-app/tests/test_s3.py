from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import app
import control
from fizgig_common import normalize_dataset_s3, parse_dataset_s3, training_intent, validate_training_request
from fizgig_s3 import DatasetLimits, load_dataset_snapshot, materialize_dataset, plan_dataset


URI = "s3://studio-bucket/datasets/person/"
PREFIX = "datasets/person/"


class FakeS3:
    def __init__(self, objects):
        self.objects = {PREFIX + key: value for key, value in objects.items()}
        self.downloads = []
        self.list_calls = []
        self.bodies = []
        self.change_response = lambda value: value

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self

    def paginate(self, **kwargs):
        self.list_calls.append(kwargs)
        assert kwargs == {"Bucket": "studio-bucket", "Prefix": PREFIX}
        # One object per page exercises pagination and pairing across pages.
        for key, value in self.objects.items():
            yield {"Contents": [{"Key": key, "Size": len(value), "ETag": '"' + sha256(value).hexdigest() + '"'}]}

    def get_object(self, Bucket, Key, IfMatch):
        assert Bucket == "studio-bucket"
        value = self.objects[Key]
        etag = '"' + sha256(value).hexdigest() + '"'
        assert IfMatch == etag
        self.downloads.append(Key)
        body = BytesIO(value)
        self.bodies.append(body)
        return self.change_response({
            "Body": body, "ContentLength": len(value), "ETag": etag,
            "Metadata": {"sha256": sha256(value).hexdigest()},
        })

    def close(self):
        pass


def intent(**overrides):
    return {"family": "krea2", "dataset_s3": URI, "output_name": "s3-test", "preset": "krea2_defaults", **overrides}


@pytest.mark.parametrize("uri", [
    None, "", "https://studio-bucket/dataset", "s3://studio-bucket/",
    "s3://user:secret@studio-bucket/folder", "s3://studio-bucket:443/folder",
    "s3://studio-bucket/a?token=x", "s3://studio-bucket/a#fragment",
    "s3://studio-bucket/../a", "s3://studio-bucket/%2e%2e/a",
    "s3://studio-bucket/a\\b", "s3://studio-bucket/a%00b",
    "s3://studio-bucket/a//b", "s3://studio-bucket/a//", "s3://studio-bucket/\na",
])
def test_reject_unsafe_s3_source(uri):
    with pytest.raises(ValueError):
        normalize_dataset_s3(uri)


def test_s3_normalization_and_bucket_scope():
    assert normalize_dataset_s3("s3://studio-bucket/datasets/a b") == "s3://studio-bucket/datasets/a%20b/"
    assert parse_dataset_s3("s3://studio-bucket/datasets/a%20b/") == ("studio-bucket", "datasets/a b/")
    with pytest.raises(ValueError, match="S3_BUCKET"):
        parse_dataset_s3(URI, "other-bucket")


def test_s3_request_needs_no_volume_dataset_name():
    result = training_intent(intent())
    assert result["dataset_s3"] == URI
    assert "dataset" not in result
    assert result["epochs"] == 30
    with pytest.raises(ValueError, match="exactly one"):
        training_intent(intent(dataset="local"))
    with pytest.raises(ValueError, match="exactly one"):
        training_intent(intent(dataset_s3=None))


def test_s3_paths_are_isolated_per_run():
    first = app.paths_for_request(validate_training_request(intent()))
    second = app.paths_for_request(validate_training_request(intent(output_name="second")))
    assert first["image_dir"] == Path("/data/fizgig/runs/s3-test/dataset/images")
    assert first["cache_dir"].parent == first["dataset_dir"]
    assert first["dataset_dir"] != second["dataset_dir"]


def test_recursive_listing_preserves_pairing_and_avoids_basename_collisions(tmp_path):
    client = FakeS3({"a/photo.jpg": b"image A", "b/photo.jpg": b"image B",
                     "a/photo.txt": b"caption A", "b/photo.TXT": b"caption B",
                     "notes.json": b"ignored", "folder/": b""})
    destination = tmp_path / "dataset"
    manifest = materialize_dataset(client, URI, "studio-bucket", destination)
    assert manifest["image_count"] == manifest["caption_count"] == 2
    assert manifest["images_without_caption"] == 0
    images = [entry for entry in manifest["files"] if entry["kind"] == "image"]
    assert images[0]["name"] != images[1]["name"]
    for entry in images:
        path = destination / "images" / entry["name"]
        assert path.read_bytes() == client.objects[entry["key"]]
        expected = b"caption A" if "/a/" in entry["key"] else b"caption B"
        assert path.with_suffix(".txt").read_bytes() == expected
    assert all(body.closed for body in client.bodies)
    assert not any("notes.json" in key for key in client.downloads)
    assert load_dataset_snapshot(destination, URI) == manifest


def test_missing_captions_are_reported_and_same_stem_extensions_remain_distinct(tmp_path):
    client = FakeS3({"photo.jpg": b"jpg", "photo.png": b"png", "photo.txt": b"paired", "other.webp": b"webp"})
    plan = materialize_dataset(client, URI, "studio-bucket", tmp_path / "dataset")
    assert plan["image_count"] == 3
    assert plan["caption_count"] == 2
    assert plan["images_without_caption"] == 1
    assert len({entry["name"] for entry in plan["files"]}) == 5


@pytest.mark.parametrize("objects", [{}, {"file.json": b"{}"}, {"empty.jpg": b""},
                                     {"../escape.jpg": b"x"}, {"a\\escape.jpg": b"x"},
                                     {"a.jpg": b"x", "a.txt": b"one", "a.TXT": b"two"}])
def test_invalid_listing_never_downloads_or_publishes(tmp_path, objects):
    client = FakeS3(objects)
    with pytest.raises(ValueError):
        materialize_dataset(client, URI, "studio-bucket", tmp_path / "dataset")
    assert not client.downloads
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize("limits", [
    replace(DatasetLimits(), max_images=1), replace(DatasetLimits(), max_objects=1),
    replace(DatasetLimits(), max_image_bytes=1), replace(DatasetLimits(), max_caption_bytes=1),
    replace(DatasetLimits(), max_total_bytes=3),
])
def test_limits_are_enforced_before_downloading(tmp_path, limits):
    client = FakeS3({"a.png": b"aa", "b.jpg": b"bb", "a.txt": b"caption"})
    with pytest.raises(ValueError):
        materialize_dataset(client, URI, "studio-bucket", tmp_path / "dataset", limits=limits)
    assert not client.downloads


@pytest.mark.parametrize("change", [
    lambda r: {**r, "ContentLength": 100},
    lambda r: {**r, "ETag": '"changed"'},
    lambda r: {**r, "Metadata": {"sha256": "incorrect"}},
    lambda r: {**r, "Body": BytesIO(b"too many bytes")},
    lambda r: {**r, "Body": BytesIO(b"")},
])
def test_changed_or_corrupt_download_cleans_partial_dataset(tmp_path, change):
    client = FakeS3({"a.png": b"data"})
    client.change_response = change
    with pytest.raises(ValueError):
        materialize_dataset(client, URI, "studio-bucket", tmp_path / "dataset")
    assert list(tmp_path.iterdir()) == []


def test_transfer_failure_after_first_image_cannot_publish_partial_dataset(tmp_path):
    client = FakeS3({"a.png": b"a", "b.png": b"b"})
    original = client.get_object
    def download(**kwargs):
        if kwargs["Key"].endswith("b.png"):
            raise RuntimeError("S3 access denied")
        return original(**kwargs)
    client.get_object = download
    with pytest.raises(RuntimeError, match="access denied"):
        materialize_dataset(client, URI, "studio-bucket", tmp_path / "dataset")
    assert list(tmp_path.iterdir()) == []


def test_resume_preserves_recaptioned_text_without_contacting_s3(tmp_path):
    client = FakeS3({"a.png": b"a", "a.txt": b"original"})
    destination = tmp_path / "dataset"
    manifest = materialize_dataset(client, URI, "studio-bucket", destination)
    image = next(entry for entry in manifest["files"] if entry["kind"] == "image")
    caption = (destination / "images" / image["name"]).with_suffix(".txt")
    caption.write_text("improved caption")
    client.objects.clear()
    assert load_dataset_snapshot(destination, URI) == manifest
    assert caption.read_text() == "improved caption"
    with pytest.raises(ValueError, match="source"):
        load_dataset_snapshot(destination, "s3://studio-bucket/other/")
    (destination / "images" / image["name"]).write_bytes(b"b")
    with pytest.raises(ValueError, match="SHA-256"):
        load_dataset_snapshot(destination, URI)


def test_existing_destination_is_never_overwritten(tmp_path):
    destination = tmp_path / "dataset"
    destination.mkdir()
    client = FakeS3({"a.png": b"a"})
    with pytest.raises(FileExistsError):
        materialize_dataset(client, URI, "studio-bucket", destination)
    assert not client.list_calls


def test_s3_resume_keeps_source_and_creates_new_job(monkeypatch):
    record = {"id": "original", "status": "succeeded", "request": intent(),
              "result": {"paused": True, "resume_from": "s3-test-000001-state"}}
    monkeypatch.setattr(control, "get_training_job", lambda job_id: record)
    def spawn(request, resumed_from):
        assert request["dataset_s3"] == URI
        assert "dataset" not in request
        assert request["resume_from"] == "s3-test-000001-state"
        assert resumed_from == "original"
        return {"id": "resumed", "status": "queued"}
    monkeypatch.setattr(control, "_spawn_training", spawn)
    assert control.resume_training_job("original")["id"] == "resumed"


def local_paths(tmp_path):
    run = tmp_path / "run"
    dataset = run / "dataset"
    return {"run_dir": run, "dataset_dir": dataset, "image_dir": dataset / "images",
            "cache_dir": dataset / "cache", "config_path": dataset / "dataset.toml",
            "pause_path": run / ".pause_requested", "promoted_lora": tmp_path / "loras/final.safetensors"}


def prepare_mocks(monkeypatch, client):
    monkeypatch.setattr(app, "_ensure_layout", lambda: None)
    monkeypatch.setattr(app, "dataset_config_text", lambda image, cache: "test dataset config")
    monkeypatch.setattr(app, "create_s3_client", lambda: (client, "studio-bucket"))


def test_h3_rejects_missing_captions_before_training(tmp_path, monkeypatch):
    client = FakeS3({"a.jpg": b"image"})
    prepare_mocks(monkeypatch, client)
    request = validate_training_request(intent(family="minimax_h3", preset="h3_character_fast"))
    with pytest.raises(ValueError, match="requires non-empty"):
        app._prepare_run(request, local_paths(tmp_path))


def test_training_stages_and_commits_s3_snapshot_before_running_pipeline(tmp_path, monkeypatch):
    client = FakeS3({"a.jpg": b"image", "a.txt": b"caption"})
    prepare_mocks(monkeypatch, client)
    paths = local_paths(tmp_path)
    monkeypatch.setattr(app, "paths_for_request", lambda request: paths)
    monkeypatch.setattr(app, "_verify_models", lambda request: None)
    records = {}
    class Store:
        def get(self, key, default=None):
            return deepcopy(records.get(key, default))
        def put(self, key, value):
            records[key] = deepcopy(value)
    monkeypatch.setattr(app, "job_store", Store())
    events = []
    monkeypatch.setattr(app, "data_volume", SimpleNamespace(
        reload=lambda: events.append("reload"), commit=lambda: events.append("commit")))
    def stage(job_id, phase, command, pause_path):
        assert events[0:2] == ["reload", "commit"]
        assert (paths["dataset_dir"] / "manifest.json").is_file()
        assert len(list(paths["image_dir"].glob("*.jpg"))) == 1
        events.append(phase)
        if phase == "training":
            (paths["run_dir"] / "s3-test.safetensors").write_bytes(b"lora")
    monkeypatch.setattr(app, "_run_stage", stage)
    result = app.run_training.local("job-s3", intent())
    assert result["status"] == "succeeded"
    assert result["dataset"]["source"] == URI
    assert result["dataset"]["image_count"] == 1
    assert paths["promoted_lora"].read_bytes() == b"lora"


def test_prepare_resume_reuses_snapshot_without_initializing_s3(tmp_path, monkeypatch):
    client = FakeS3({"a.jpg": b"image", "a.txt": b"caption"})
    prepare_mocks(monkeypatch, client)
    paths = local_paths(tmp_path)
    app._prepare_run(validate_training_request(intent()), paths)
    caption = next(paths["image_dir"].glob("*.txt"))
    caption.write_text("recaptioned")
    def forbidden():
        raise AssertionError("resume should not contact S3")
    monkeypatch.setattr(app, "create_s3_client", forbidden)
    result = app._prepare_run(validate_training_request(intent(resume_from="latest")), paths)
    assert result["source"] == URI
    assert caption.read_text() == "recaptioned"
