from pathlib import Path
from types import SimpleNamespace

import pytest

from app import (
    FIZGIG_JOB_DICT_NAME,
    JOB_DICT_NAME,
    TRAINING_JOB_DICT_NAME,
    canonical_output_path,
    filter_models,
    merge_training_record,
    normalized_absolute_path,
    public_training_record,
    resolve_generation_kind,
    serialize_result,
    validate_data_paths,
    validate_job_request,
    validate_training_request,
)


def test_request_overlays_model_type():
    assert validate_job_request(" qwen_image ", {"prompt": "fox"}) == {
        "model_type": "qwen_image",
        "prompt": "fox",
    }


def test_request_rejects_reserved_api_metadata():
    with pytest.raises(ValueError, match="reserved"):
        validate_job_request("qwen_image", {"_api": {"return_media": True}})


def test_data_paths_accept_data_volume_and_virtual_suffix():
    validate_data_paths({"video_guide": "/data/inputs/a.mp4|start_frame=1,end_frame=2"})


def test_data_paths_reject_outside_volume():
    with pytest.raises(ValueError, match="under /data"):
        validate_data_paths({"image_start": "/tmp/a.png"})


def test_data_paths_reject_parent_traversal():
    with pytest.raises(ValueError, match="under /data"):
        validate_data_paths({"image_start": "/data/../etc/passwd"})


def test_normalized_path_does_not_resolve_volume_mount_symlinks(tmp_path: Path):
    mounted = tmp_path / "data"
    target = tmp_path / "internal-volume"
    target.mkdir()
    mounted.symlink_to(target, target_is_directory=True)
    path = normalized_absolute_path(mounted / "outputs" / "image.png")
    assert str(path).startswith(str(mounted))


def test_canonical_output_path_maps_modal_internal_mount():
    assert canonical_output_path(
        "/__modal/volumes/vo-example/outputs/nested/image.png"
    ) == Path("/data/outputs/nested/image.png")


def test_result_is_metadata_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    output = output_dir / "out.png"
    output.write_bytes(b"png")
    monkeypatch.setattr("app.DATA_ROOT", tmp_path)
    result = SimpleNamespace(
        success=True,
        generated_files=[str(output)],
        total_tasks=1,
        successful_tasks=1,
        failed_tasks=0,
        errors=[],
    )
    metadata = {
        "id": "output-id",
        "filename": "out.png",
        "size_bytes": 3,
        "media_type": "image/png",
        "url": "/outputs/output-id",
    }
    assert serialize_result(result, [metadata]) == {
        "success": True,
        "outputs": [metadata],
        "total_tasks": 1,
        "successful_tasks": 1,
        "failed_tasks": 0,
        "errors": [],
    }


def test_serialize_output_hides_internal_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    output = output_dir / "clip.mp4"
    output.write_bytes(b"video")
    monkeypatch.setattr("app.DATA_ROOT", tmp_path)

    from app import serialize_output

    assert serialize_output(str(output)) == {
        "filename": "clip.mp4",
        "size_bytes": 5,
        "media_type": "video/mp4",
    }


def test_filter_models_uses_cached_metadata():
    models = [
        {"model_type": "one", "family": "qwen"},
        {"model_type": "two", "family": "flux"},
    ]
    assert filter_models(models, family="qwen") == [models[0]]
    assert filter_models(models, model_type="two") == [models[1]]


@pytest.fixture
def generation_catalog():
    return {
        "models": [
            {"model_type": "krea2_turbo", "main_output": ["image"]},
            {"model_type": "minimax_h3", "main_output": ["video"]},
            {"model_type": "switchable", "main_output": ["image", "video"]},
            {"model_type": "tts", "main_output": ["audio"]},
        ],
        "defaults": {
            "switchable": {"image_mode": 0},
        },
    }


def test_generation_kind_is_inferred_from_model_metadata(generation_catalog):
    assert resolve_generation_kind(generation_catalog, "krea2_turbo") == "image"
    assert resolve_generation_kind(generation_catalog, "minimax_h3") == "video"
    assert resolve_generation_kind(generation_catalog, "tts") == "audio"


def test_generation_kind_accepts_matching_explicit_route(generation_catalog):
    assert (
        resolve_generation_kind(generation_catalog, "minimax_h3", "video")
        == "video"
    )


def test_generation_kind_rejects_model_route_mismatch(generation_catalog):
    with pytest.raises(ValueError, match="not image"):
        resolve_generation_kind(generation_catalog, "minimax_h3", "image")


def test_generation_kind_rejects_unknown_model(generation_catalog):
    with pytest.raises(ValueError, match="unknown model"):
        resolve_generation_kind(generation_catalog, "missing")


def test_generation_kind_uses_native_image_mode_for_dual_output_model(
    generation_catalog,
):
    assert resolve_generation_kind(generation_catalog, "switchable") == "video"
    assert (
        resolve_generation_kind(
            generation_catalog,
            "switchable",
            params={"image_mode": 2},
        )
        == "image"
    )


def test_training_request_contains_only_public_intent():
    assert validate_training_request(
        {
            "family": "minimax_h3",
            "dataset": "anna",
            "output_name": "anna_h3",
            "preset": "h3_character_fast",
        }
    ) == {
        "family": "minimax_h3",
        "dataset": "anna",
        "output_name": "anna_h3",
        "preset": "h3_character_fast",
    }


def test_krea2_request_leaves_execution_defaults_to_worker_preset():
    assert validate_training_request(
        {
            "family": "krea2",
            "dataset": "linda",
            "output_name": "linda_krea2_v1",
            "preset": "krea2_defaults",
            "trigger_word": "linda",
            "epochs": 24,
        }
    ) == {
        "family": "krea2",
        "dataset": "linda",
        "output_name": "linda_krea2_v1",
        "preset": "krea2_defaults",
        "trigger_word": "linda",
        "epochs": 24,
    }


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            {
                "family": "unknown",
                "dataset": "anna",
                "output_name": "anna",
                "preset": "h3_character_fast",
            },
            "family",
        ),
        (
            {
                "family": "minimax_h3",
                "dataset": "../anna",
                "output_name": "anna",
                "preset": "h3_character_fast",
            },
            "must not contain a path",
        ),
        (
            {
                "family": "minimax_h3",
                "dataset": "anna",
                "output_name": "anna",
                "preset": "h3_character_fast",
                "epochs": True,
            },
            "integer",
        ),
        (
            {
                "family": "minimax_h3",
                "dataset": "anna",
                "output_name": "anna",
                "preset": "h3_character_fast",
                "unknown": 1,
            },
            "unsupported",
        ),
        (
            {
                "family": "krea2",
                "dataset": "linda",
                "output_name": "linda",
                "preset": "krea2_defaults",
                "auto_caption": True,
            },
            "unsupported",
        ),
        ({"dataset": "anna", "output_name": "anna"}, "missing required"),
    ],
)
def test_training_request_rejects_unsupported_or_unsafe_values(body, message):
    with pytest.raises(ValueError, match=message):
        validate_training_request(body)


def test_training_resume_state_is_endpoint_controlled():
    request = {
        "family": "minimax_h3",
        "dataset": "anna",
        "output_name": "anna_h3",
        "preset": "h3_character_fast",
        "resume_from": "anna_h3-epoch-10-state",
    }
    with pytest.raises(ValueError, match="resume endpoint"):
        validate_training_request(request)
    assert validate_training_request(request, allow_resume=True)["resume_from"] == (
        "anna_h3-epoch-10-state"
    )


def test_training_record_merge_preserves_public_identity_and_request():
    public = {
        "id": "public-id",
        "call_id": "call-id",
        "status": "queued",
        "request": {"dataset": "anna"},
        "created_at": "created",
    }
    internal = {
        "id": "internal-id",
        "status": "running",
        "request": {"dataset": "other"},
        "progress": {"phase": "training"},
        "started_at": "started",
        "updated_at": "updated",
    }
    assert merge_training_record(public, internal) == {
        **public,
        "status": "running",
        "progress": {"phase": "training"},
        "started_at": "started",
        "updated_at": "updated",
    }


def test_public_training_record_omits_internal_log_tail():
    record = {
        "id": "job-id",
        "call_id": "internal-call-id",
        "status": "running",
        "pause_requested": False,
        "progress": {
            "phase": "training",
            "epoch": 2,
            "epochs_total": 30,
            "log_tail": ["internal output"],
        },
    }
    assert public_training_record(record) == {
        "id": "job-id",
        "status": "running",
        "progress": {"phase": "training", "epoch": 2, "epochs_total": 30},
    }
    assert "log_tail" in record["progress"]


def test_training_and_generation_use_separate_job_domains():
    assert len({JOB_DICT_NAME, TRAINING_JOB_DICT_NAME, FIZGIG_JOB_DICT_NAME}) == 3
