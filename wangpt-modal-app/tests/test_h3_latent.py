import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from h3_latent import PACKAGE, PLUGIN_KEY, TASK_KEY, include_latent_outputs, install_headless_hooks, native_task, validate_frame_alignment
from wangpt_common import validate_job_request
from app import materialize_s3_image_inputs, remove_local_outputs, upload_output_artifacts
from test_app import FakeInputS3


def test_native_task_detaches_plugin_data_and_captures_options():
    settings = {
        "model_type": "minimax_h3_ref2va_singularity_pruned",
        "prompt": "fox",
        "plugin_data": {PLUGIN_KEY: {"save": True}},
    }
    original = copy.deepcopy(settings)
    task = native_task(settings)
    assert "plugin_data" not in task["params"]
    assert task["params"][TASK_KEY]["options"] == {"save": True, "continue": False, "latent_path": ""}
    assert task["params"][TASK_KEY]["model_type"] == settings["model_type"]
    task["plugin_data"][PLUGIN_KEY]["save"] = False
    assert task["params"][TASK_KEY]["options"]["save"] is True
    assert settings == original


def test_native_task_without_plugin_has_no_latent_snapshot():
    task = native_task({"model_type": "qwen_image", "prompt": "fox"})
    assert task["plugin_data"] == {}
    assert TASK_KEY not in task["params"]
    assert task["params"]["_api"] == {"return_media": False}


def test_source_video_frames_must_match_exported_checkpoint_endpoint():
    validate_frame_alignment({"output_frames": 124}, 124)
    with pytest.raises(ValueError, match="243 video frames.*247"):
        validate_frame_alignment({"output_frames": 247}, 243)
    with pytest.raises(ValueError, match="lacks an exported frame count"):
        validate_frame_alignment({}, 124)


def test_headless_install_preserves_wrapper_chain_and_is_idempotent(monkeypatch):
    calls = []

    def wrapper(original, *args):
        def wrapped(*values):
            calls.append("wrapper")
            return original(*values)
        return wrapped

    integration = SimpleNamespace(
        JOB=SimpleNamespace(get=lambda: None),
        make_generation_wrapper=wrapper,
        make_save_wrapper=wrapper,
        make_record_wrapper=wrapper,
    )
    monkeypatch.setitem(sys.modules, PACKAGE, SimpleNamespace())
    monkeypatch.setitem(sys.modules, f"{PACKAGE}.integration", integration)
    monkeypatch.setitem(sys.modules, f"{PACKAGE}.native_bridge", SimpleNamespace(install_loader=lambda: calls.append("loader")))
    wgp = SimpleNamespace(
        generate_media=lambda value: value,
        save_video=lambda value: value,
        record_file_metadata=lambda value: value,
        get_base_model_type=lambda model: model,
        combine_and_concatenate_video_with_audio_tracks=lambda value: value,
    )
    install_headless_hooks(wgp)
    first_wrapper = wgp.generate_media
    install_headless_hooks(wgp)
    assert wgp.generate_media is first_wrapper
    assert wgp.generate_media("result") == "result"
    assert calls == ["loader", "wrapper"]


@pytest.mark.parametrize("params", [
    {"plugin_data": {"api": {"return_media": True}}},
    {"plugin_data": []},
    {"plugin_data": {PLUGIN_KEY: {"save": "true"}}},
    {"plugin_data": {PLUGIN_KEY: {"latent_path": []}}},
    {TASK_KEY: {}},
])
def test_public_request_rejects_invalid_or_reserved_plugin_data(params):
    with pytest.raises(ValueError):
        validate_job_request("minimax_h3_ref2va_pruned", params)


def test_continuation_inputs_download_without_mutating_request(tmp_path):
    client = FakeInputS3({
        ("bucket", "clip.mp4"): {"body": b"video"},
        ("bucket", "clip.safetensors"): {"body": b"latent"},
    })
    params = {
        "video_source": "s3://bucket/clip.mp4",
        "plugin_data": {PLUGIN_KEY: {"continue": True, "latent_path": "s3://bucket/clip.safetensors"}},
    }
    original = copy.deepcopy(params)
    settings, directory, count = materialize_s3_image_inputs(params, "job", client, "bucket", root=tmp_path)
    assert count == 2
    assert Path(settings["video_source"]).read_bytes() == b"video"
    assert Path(settings["plugin_data"][PLUGIN_KEY]["latent_path"]).read_bytes() == b"latent"
    assert params == original
    assert directory.is_dir()


def test_continuation_input_failure_cleans_downloaded_pair(tmp_path):
    client = FakeInputS3({
        ("bucket", "clip.mp4"): {"body": b"video"},
        ("bucket", "clip.safetensors"): {"body": b"latent", "metadata": {"sha256": "wrong"}},
    })
    with pytest.raises(ValueError, match="SHA-256"):
        materialize_s3_image_inputs({
            "video_source": "s3://bucket/clip.mp4",
            "plugin_data": {PLUGIN_KEY: {"latent_path": "s3://bucket/clip.safetensors"}},
        }, "job", client, "bucket", root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_sidecar_is_required_and_uploaded_with_video(tmp_path, monkeypatch):
    monkeypatch.setattr("app.GENERATED_OUTPUT_ROOT", tmp_path)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    result = SimpleNamespace(success=True, generated_files=[str(video)])
    settings = {"plugin_data": {PLUGIN_KEY: {"save": True}}}
    with pytest.raises(RuntimeError, match="was not saved"):
        include_latent_outputs(result, settings)
    latent = video.with_suffix(".safetensors")
    latent.write_bytes(b"latent")
    include_latent_outputs(result, settings)
    include_latent_outputs(result, settings)
    assert result.generated_files == [str(video), str(latent)]

    class S3:
        objects = {}

        def upload_file(self, path, bucket, key, ExtraArgs):
            self.objects[key] = {"ContentLength": Path(path).stat().st_size, "Metadata": ExtraArgs["Metadata"]}

        def head_object(self, Bucket, Key):
            return self.objects[Key]

    outputs = upload_output_artifacts(result, "job", S3(), "bucket")
    assert [item["filename"] for item in outputs] == ["clip.mp4", "clip.safetensors"]
    assert all(item["sha256"] for item in outputs)
    remove_local_outputs(result)
    assert not video.exists() and not latent.exists()
