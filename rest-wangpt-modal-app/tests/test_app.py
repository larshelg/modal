from pathlib import Path
from types import SimpleNamespace

import pytest

from app import (
    canonical_output_path,
    filter_models,
    normalized_absolute_path,
    serialize_result,
    validate_data_paths,
    validate_job_request,
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
