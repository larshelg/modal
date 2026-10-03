from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import app
from fizgig_common import training_intent, validate_training_request
from fizgig_s3 import DatasetLimits, dataset_summary, load_dataset_snapshot, materialize_dataset, plan_dataset
from test_refmod import intent, local_paths, prepare_paths
from test_s3 import FakeS3, URI


@pytest.mark.parametrize("mode", ["still", "motion"])
def test_video_controls_roundtrip(mode):
    normalized = validate_training_request(training_intent(intent(clips=mode, token_cap=5120)))
    assert normalized["clips"] == mode
    assert normalized["token_cap"] == 5120


@pytest.mark.parametrize("invalid", [
    {"clips": "video"}, {"clips": None}, {"clips": []},
    {"token_cap": -1}, {"token_cap": True}, {"token_cap": 1.5}, {"token_cap": 65537},
    {"token_cap": "5120"}, {"preset": "h3_character_fast", "clips": "motion"},
])
def test_invalid_video_controls_rejected(invalid):
    with pytest.raises(ValueError):
        training_intent(intent(**invalid))


def test_refmod_video_defaults_preserve_image_requests():
    normalized = validate_training_request(intent())
    assert normalized["clips"] == "still" and normalized["token_cap"] == 0


def test_s3_video_is_opt_in_for_image_workflows():
    client = FakeS3({"a.jpg": b"image", "b.mp4": b"video", "b.txt": b"caption"})
    plan = plan_dataset(client, URI, "studio-bucket")
    assert plan["schema_version"] == 1
    assert plan["image_count"] == 1
    assert all(entry["kind"] == "image" for entry in plan["files"])
    assert "video_count" not in dataset_summary(plan)


def test_video_only_s3_snapshot_pairs_captions_and_preserves_mute_marker(tmp_path):
    client = FakeS3({"front/clip_mute.MP4": b"video A", "front/clip_mute.txt": b"caption A",
                     "back/clip.mp4": b"video B", "back/clip.txt": b"caption B"})
    destination = tmp_path / "snapshot"
    plan = materialize_dataset(client, URI, "studio-bucket", destination, include_video=True)
    assert plan["schema_version"] == 2
    assert plan["image_count"] == 0
    assert plan["video_count"] == plan["media_count"] == plan["caption_count"] == 2
    assert plan["media_without_caption"] == 0
    for entry in [e for e in plan["files"] if e["kind"] == "video"]:
        path = destination / "images" / entry["name"]
        assert path.read_bytes() == client.objects[entry["key"]]
        assert path.with_suffix(".txt").read_bytes() == (b"caption A" if "/front/" in entry["key"] else b"caption B")
        assert path.suffix == ".mp4"
        assert path.stem.endswith("_mute") == ("clip_mute" in entry["key"])
    assert load_dataset_snapshot(destination, URI, include_video=True) == plan
    with pytest.raises(ValueError, match="source"):
        load_dataset_snapshot(destination, URI)


def test_mixed_s3_counts_caption_gaps_by_media_type(tmp_path):
    client = FakeS3({"a.jpg": b"image", "a.txt": b"image caption", "a.mp4": b"video",
                     "b.mp4": b"video", "c.png": b"image", "unused.mov": b"ignored"})
    plan = materialize_dataset(client, URI, "studio-bucket", tmp_path / "dataset", include_video=True)
    summary = dataset_summary(plan)
    assert summary["image_count"] == summary["video_count"] == summary["caption_count"] == 2
    assert summary["media_count"] == 4
    assert summary["images_without_caption"] == summary["videos_without_caption"] == 1
    assert summary["media_without_caption"] == 2
    assert len({entry["name"] for entry in plan["files"]}) == 6
    assert not any(key.endswith(".mov") for key in client.downloads)


@pytest.mark.parametrize("limits,objects", [
    (replace(DatasetLimits(), max_videos=1), {"a.mp4": b"a", "b.mp4": b"b"}),
    (replace(DatasetLimits(), max_video_bytes=1), {"a.mp4": b"too big"}),
    (replace(DatasetLimits(), max_total_bytes=3), {"a.mp4": b"aa", "b.jpg": b"bb"}),
    (DatasetLimits(), {"a.mp4": b""}),
    (DatasetLimits(), {"../a.mp4": b"bad path"}),
])
def test_video_import_limits_and_paths_fail_before_download(tmp_path, limits, objects):
    client = FakeS3(objects)
    with pytest.raises(ValueError):
        materialize_dataset(client, URI, "studio-bucket", tmp_path / "dataset", limits=limits, include_video=True)
    assert not client.downloads
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize("change", [
    lambda r: {**r, "ContentLength": 900},
    lambda r: {**r, "ETag": '"different"'},
    lambda r: {**r, "Metadata": {"sha256": "bad"}},
    lambda r: {**r, "Body": BytesIO(b"cut")},
])
def test_bad_video_transfer_cleans_snapshot(tmp_path, change):
    client = FakeS3({"a.mp4": b"video"})
    client.change_response = change
    with pytest.raises(ValueError):
        materialize_dataset(client, URI, "studio-bucket", tmp_path / "dataset", include_video=True)
    assert list(tmp_path.iterdir()) == []


def test_video_snapshot_hash_is_checked(tmp_path):
    destination = tmp_path / "dataset"
    plan = materialize_dataset(FakeS3({"a.mp4": b"video"}), URI, "studio-bucket", destination, include_video=True)
    video = next(e for e in plan["files"] if e["kind"] == "video")
    (destination / "images" / video["name"]).write_bytes(b"other")
    with pytest.raises(ValueError, match="SHA-256"):
        load_dataset_snapshot(destination, URI, include_video=True)


@pytest.mark.parametrize("steps", [0, 200])
@pytest.mark.parametrize("mode", ["still", "motion"])
def test_refmod_caches_sharp_stills_in_both_resolutions(steps, mode):
    request = validate_training_request(intent(steps=steps, clips=mode, token_cap=5120))
    commands = dict(app.build_pipeline_commands(request, app.paths_for_request(request)))
    for phase in ("caching_latents", "caching_references"):
        if phase in commands:
            assert "--clip_still" in commands[phase]
    command = commands["making_refmod"]
    assert command[command.index("--clips") + 1] == mode
    assert command[command.index("--token_cap") + 1] == "5120"
    assert "--audio" not in command and "--audio_vae" not in command


@pytest.mark.parametrize("steps,caption,accepted", [(0, None, True), (200, None, False), (200, " ", False), (200, "walking", True)])
def test_video_only_volume_caption_rules(tmp_path, monkeypatch, steps, caption, accepted):
    paths = prepare_paths(tmp_path, monkeypatch)
    (paths["image_dir"] / "a.png").unlink()
    (paths["image_dir"] / "clip.mp4").write_bytes(b"video")
    if caption is not None:
        (paths["image_dir"] / "clip.txt").write_text(caption)
    request = validate_training_request(intent(steps=steps))
    if accepted:
        app._prepare_run(request, paths)
        assert paths["config_path"].exists()
    else:
        with pytest.raises(ValueError, match="requires non-empty"):
            app._prepare_run(request, paths)


@pytest.mark.parametrize("verify", [False, True])
def test_cpu_inspection_includes_video_without_gpu(tmp_path, monkeypatch, verify):
    client = FakeS3({"clip.mp4": b"video"})
    monkeypatch.setattr(app, "create_s3_client", lambda: (client, "studio-bucket"))
    result = app.inspect_dataset.local(URI, verify_download=verify, include_video=True)
    assert result["video_count"] == result["media_count"] == 1
    assert result["image_count"] == 0
    assert result["download_verified"] == verify
    assert bool(client.downloads) == verify


@pytest.mark.parametrize("steps", [0, 200])
@pytest.mark.parametrize("source", ["s3", "volume"])
def test_video_only_worker_lifecycle(tmp_path, monkeypatch, steps, source):
    paths = local_paths(tmp_path)
    if source == "s3":
        paths["dataset_dir"] = paths["run_dir"] / "dataset"
        paths["image_dir"] = paths["dataset_dir"] / "images"
    else:
        paths["image_dir"].mkdir(parents=True)
        (paths["image_dir"] / "clip.mp4").write_bytes(b"video")
        if steps:
            (paths["image_dir"] / "clip.txt").write_text("person walking")
    monkeypatch.setattr(app, "paths_for_request", lambda request: paths)
    monkeypatch.setattr(app, "require_data_path", lambda path, **kw: path)
    monkeypatch.setattr(app, "_ensure_layout", lambda: None)
    monkeypatch.setattr(app, "_verify_models", lambda request: None)
    objects = {"clip.mp4": b"video", **({"clip.txt": b"person walking"} if steps else {})}
    monkeypatch.setattr(app, "create_s3_client", lambda: (FakeS3(objects), "studio-bucket"))
    records, stages = {}, []
    monkeypatch.setattr(app, "job_store", SimpleNamespace(
        get=lambda key, default=None: deepcopy(records.get(key, default)),
        put=lambda key, value: records.update({key: deepcopy(value)})))
    monkeypatch.setattr(app, "data_volume", SimpleNamespace(reload=lambda: None, commit=lambda: None))
    def stage(job, phase, command, pause_path):
        assert len(list(paths["image_dir"].glob("*.mp4"))) == 1
        stages.append(phase)
        if phase == "making_refmod":
            assert command[command.index("--clips") + 1] == "motion"
            (paths["run_dir"] / "person_ref.safetensors").write_bytes(b"refmod")
    monkeypatch.setattr(app, "_run_stage", stage)
    request = intent(steps=steps, clips="motion", token_cap=5120)
    if source == "s3":
        del request["dataset"]
        request["dataset_s3"] = URI
    result = app.run_training.local("video-refmod-job", request)
    assert result["status"] == "succeeded"
    assert result["result"]["clips"] == "motion"
    assert result["result"]["token_cap"] == 5120
    if source == "s3":
        assert result["dataset"]["video_count"] == 1 and result["dataset"]["image_count"] == 0
    assert ("caching_text" in stages) == bool(steps)
    assert ("caching_references" in stages) == bool(steps)
    assert paths["promoted_refmod"].exists()
    assert not paths["promoted_lora"].exists()
