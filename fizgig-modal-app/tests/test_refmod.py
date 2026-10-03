from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tomllib
from unittest.mock import Mock

import pytest

import app
import control
from fizgig_common import training_intent, validate_training_request


def intent(**extra):
    return {"family": "minimax_h3", "dataset": "person", "output_name": "person_ref",
            "preset": "h3_refmod_community", **extra}


@pytest.mark.parametrize("preset,steps,refs,mp", [
    ("h3_refmod_community", 0, 8, 1.0),
    ("h3_refmod_lite", 200, 16, 0.5),
    ("h3_refmod_quality", 200, 16, 1.0),
])
def test_refmod_intent_roundtrips_to_worker(preset, steps, refs, mp):
    request = training_intent(intent(preset=preset))
    resolved = validate_training_request(request)
    assert resolved["artifact_type"] == "refmod"
    assert (resolved["steps"], resolved["max_refs"], resolved["target_mp"]) == (steps, refs, mp)
    assert "epochs" not in request and "artifact_type" not in request
    assert resolved["base_model"] == "ref2va"


@pytest.mark.parametrize("invalid", [
    {"steps": -1}, {"steps": 2001}, {"steps": True}, {"steps": 0.5},
    {"max_refs": 0}, {"max_refs": 65}, {"max_refs": False},
    {"target_mp": float("nan")}, {"target_mp": float("inf")}, {"target_mp": True},
    {"target_mp": "0.5"}, {"target_mp": 0.1}, {"target_mp": 2.0},
    {"grid": 16}, {"grid": "4"}, {"base_model": "../../model"},
    {"concept_type": "wrong"}, {"description": "a\nb"}, {"description": "x" * 1001},
    {"epochs": 2}, {"trigger_word": "person"}, {"resume_from": "latest"},
    {"family": "krea2"}, {"audio": "bundle"}, {"dit": "/data/arbitrary"},
])
def test_refmod_rejects_invalid_or_unsupported_settings(invalid):
    with pytest.raises(ValueError):
        validate_training_request(intent(**invalid))


def test_lora_rejects_refmod_options():
    with pytest.raises(ValueError, match="RefMod settings require"):
        training_intent(intent(preset="h3_character_fast", steps=0))


def test_refmod_cache_and_output_paths_are_isolated():
    request = validate_training_request(intent())
    paths = app.paths_for_request(request)
    assert paths["image_dir"] == Path("/data/fizgig/datasets/person/images")
    assert paths["cache_dir"] == paths["run_dir"] / "cache"
    assert paths["config_path"] == paths["run_dir"] / "dataset.toml"
    assert paths["promoted_refmod"] == Path("/data/refmods/person_ref.safetensors")


@pytest.mark.parametrize("steps", [0, 200])
@pytest.mark.parametrize("base", ["ref2va", "fl2va"])
def test_refmod_commands_match_work_and_base(steps, base):
    request = validate_training_request(intent(steps=steps, base_model=base))
    commands = dict(app.build_pipeline_commands(request, app.paths_for_request(request)))
    assert "training" not in commands and "captioning" not in commands
    assert ("caching_text" in commands) == bool(steps)
    assert ("caching_references" in commands) == bool(steps)
    assert ("--captions_optional" in commands["caching_latents"]) == (steps == 0)
    command = commands["making_refmod"]
    assert base in command
    assert str(app.REFMOD_REF_DIT if base == "ref2va" else app.MODEL_PATHS["minimax_h3"]["dit"]) in command
    assert "--save_state" not in command


def test_plain_encode_only_requires_vae(monkeypatch):
    vae = app.MODEL_PATHS["minimax_h3"]["vae"]
    monkeypatch.setattr(Path, "is_file", lambda path: path == vae)
    app._verify_models(validate_training_request(intent()))
    with pytest.raises(FileNotFoundError, match="ref2va"):
        app._verify_models(validate_training_request(intent(steps=1)))


@pytest.mark.parametrize("base", ["ref2va", "fl2va"])
def test_optimized_verifies_selected_dit_only(monkeypatch, base):
    models = app.MODEL_PATHS["minimax_h3"]
    available = {models["vae"], models["text_encoder"], app.REFMOD_REF_DIT if base == "ref2va" else models["dit"]}
    monkeypatch.setattr(Path, "is_file", lambda path: path in available)
    app._verify_models(validate_training_request(intent(steps=1, base_model=base)))


def local_paths(tmp_path):
    dataset = tmp_path / "dataset"
    run = tmp_path / "run"
    return {"dataset_dir": dataset, "image_dir": dataset / "images", "run_dir": run,
            "cache_dir": run / "cache", "config_path": run / "dataset.toml",
            "pause_path": run / ".pause_requested", "promoted_refmod": tmp_path / "refmods/person_ref.safetensors",
            "promoted_lora": tmp_path / "loras/person_ref.safetensors"}


def prepare_paths(tmp_path, monkeypatch):
    paths = local_paths(tmp_path)
    paths["image_dir"].mkdir(parents=True)
    (paths["image_dir"] / "a.png").write_bytes(b"image")
    monkeypatch.setattr(app, "_ensure_layout", lambda: None)
    monkeypatch.setattr(app, "require_data_path", lambda path, **kw: path)
    return paths


@pytest.mark.parametrize("steps,mp,expected", [(0, 1.0, 992), (0, 0.5, 704), (200, 1.0, 496)])
def test_config_resolution_matches_cache_resolution(tmp_path, monkeypatch, steps, mp, expected):
    paths = prepare_paths(tmp_path, monkeypatch)
    (paths["image_dir"] / "a.txt").write_text("a person")
    app._prepare_run(validate_training_request(intent(steps=steps, target_mp=mp)), paths)
    config = tomllib.loads(paths["config_path"].read_text())
    assert config["general"]["resolution"] == [expected, expected]
    assert config["datasets"][0]["cache_directory"] == str(paths["cache_dir"])


@pytest.mark.parametrize("preset", ["h3_refmod_lite", "h3_character_fast"])
@pytest.mark.parametrize("caption", [None, "   "])
def test_captioned_h3_modes_reject_incomplete_volume_dataset(tmp_path, monkeypatch, preset, caption):
    paths = prepare_paths(tmp_path, monkeypatch)
    if caption is not None:
        (paths["image_dir"] / "a.txt").write_text(caption)
    with pytest.raises(ValueError, match="requires non-empty"):
        app._prepare_run(validate_training_request(intent(preset=preset)), paths)


def test_plain_encode_accepts_uncaptioned_volume_dataset(tmp_path, monkeypatch):
    paths = prepare_paths(tmp_path, monkeypatch)
    app._prepare_run(validate_training_request(intent()), paths)
    assert paths["config_path"].is_file()
    assert not (paths["image_dir"] / "a.txt").exists()


def test_refmod_finalization_does_not_promote_lora_or_overwrite(tmp_path):
    paths = local_paths(tmp_path)
    paths["run_dir"].mkdir()
    output = paths["run_dir"] / "person_ref.safetensors"
    output.write_bytes(b"refmod")
    request = validate_training_request(intent())
    result = app._finalize_run(request, paths)
    assert result["artifact_type"] == "refmod" and result["paused"] is False
    assert result["base_model"] is None
    assert paths["promoted_refmod"].read_bytes() == b"refmod"
    assert not paths["promoted_lora"].exists()
    output.write_bytes(b"replacement")
    with pytest.raises(FileExistsError):
        app._finalize_run(request, paths)
    assert paths["promoted_refmod"].read_bytes() == b"refmod"


def test_missing_refmod_output_never_counts_as_pause(tmp_path):
    paths = local_paths(tmp_path)
    (paths["run_dir"] / "person_ref-000001-state").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="RefMod"):
        app._finalize_run(validate_training_request(intent()), paths)


@pytest.mark.parametrize("line,expected", [
    ("[refmod] step 50/200  loss 0.4312  drift 0.012", (50, 200)),
    ("[refmod] saved example.safetensors", None),
])
def test_step_progress(line, expected):
    assert app._parse_refmod_step(line) == expected


def test_client_pause_resume_rejected_without_remote_calls(monkeypatch):
    monkeypatch.setattr(control, "get_training_job", lambda job: {"request": intent(), "status": "running"})
    remote = Mock(side_effect=AssertionError("must not call Modal"))
    monkeypatch.setattr(control.modal.Function, "from_name", remote)
    for function in (control.pause_training_job, control.resume_training_job):
        with pytest.raises(ValueError, match="do not support pause/resume"):
            function("job-1")


def test_worker_pause_rejected(monkeypatch):
    store = SimpleNamespace(get=lambda key: {"request": intent(), "status": "running"}, put=Mock())
    monkeypatch.setattr(app, "job_store", store)
    with pytest.raises(ValueError, match="do not support pause/resume"):
        app.request_pause.local("job-1")
    store.put.assert_not_called()


def test_optional_model_fetch_includes_ref2va(monkeypatch):
    monkeypatch.setattr(app, "_ensure_layout", lambda: None)
    run = Mock()
    monkeypatch.setattr(app.subprocess, "run", run)
    result = app.fetch_models.local("minimax_h3", dry_run=True, include_optional=True)
    assert "--include-optional" in run.call_args.args[0]
    assert result["expected_models"]["ref_dit"] == str(app.REFMOD_REF_DIT)


def test_refmod_worker_lifecycle_with_uncaptioned_s3(tmp_path, monkeypatch):
    from test_s3 import FakeS3, URI
    paths = local_paths(tmp_path)
    paths["dataset_dir"] = paths["run_dir"] / "dataset"
    paths["image_dir"] = paths["dataset_dir"] / "images"
    monkeypatch.setattr(app, "paths_for_request", lambda request: paths)
    monkeypatch.setattr(app, "require_data_path", lambda path, **kw: path)
    monkeypatch.setattr(app, "_ensure_layout", lambda: None)
    monkeypatch.setattr(app, "_verify_models", lambda request: None)
    client = FakeS3({"a.jpg": b"image"})
    monkeypatch.setattr(app, "create_s3_client", lambda: (client, "studio-bucket"))
    records = {}
    monkeypatch.setattr(app, "job_store", SimpleNamespace(
        get=lambda key, default=None: deepcopy(records.get(key, default)),
        put=lambda key, value: records.update({key: deepcopy(value)})))
    events = []
    monkeypatch.setattr(app, "data_volume", SimpleNamespace(
        reload=lambda: events.append("reload"), commit=lambda: events.append("commit")))
    def stage(job, phase, command, pause_path):
        assert (paths["dataset_dir"] / "manifest.json").is_file()
        events.append(phase)
        if phase == "making_refmod":
            (paths["run_dir"] / "person_ref.safetensors").write_bytes(b"refmod")
    monkeypatch.setattr(app, "_run_stage", stage)
    request = intent(dataset_s3=URI)
    del request["dataset"]
    result = app.run_training.local("refmod-job", request)
    assert result["status"] == "succeeded"
    assert result["progress"]["phase"] == "completed"
    assert result["result"]["artifact_type"] == "refmod"
    assert result["dataset"]["image_count"] == 1
    assert "caching_text" not in events and "training" not in events
    assert events[:2] == ["reload", "commit"]
    assert paths["promoted_refmod"].exists() and not paths["promoted_lora"].exists()


def test_failed_refmod_copy_cleans_partial_destination(tmp_path, monkeypatch):
    paths = local_paths(tmp_path)
    paths["run_dir"].mkdir()
    (paths["run_dir"] / "person_ref.safetensors").write_bytes(b"refmod")
    def failed_copy(source, target):
        target.write(b"partial")
        raise OSError("storage failure")
    monkeypatch.setattr(app.shutil, "copyfileobj", failed_copy)
    with pytest.raises(OSError, match="storage failure"):
        app._finalize_run(validate_training_request(intent()), paths)
    assert not paths["promoted_refmod"].exists()


def test_refmod_preflight_preserves_existing_artifact(tmp_path, monkeypatch):
    paths = prepare_paths(tmp_path, monkeypatch)
    paths["promoted_refmod"].parent.mkdir()
    paths["promoted_refmod"].write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="already exists"):
        app._prepare_run(validate_training_request(intent()), paths)
    assert paths["promoted_refmod"].read_bytes() == b"existing"
    assert not paths["run_dir"].exists()


def test_optimized_s3_rejects_missing_caption(tmp_path, monkeypatch):
    from test_s3 import FakeS3, URI
    paths = local_paths(tmp_path)
    client = FakeS3({"a.jpg": b"image"})
    monkeypatch.setattr(app, "_ensure_layout", lambda: None)
    monkeypatch.setattr(app, "create_s3_client", lambda: (client, "studio-bucket"))
    request = intent(dataset_s3=URI, preset="h3_refmod_lite")
    del request["dataset"]
    with pytest.raises(ValueError, match="requires non-empty"):
        app._prepare_run(validate_training_request(request), paths)
