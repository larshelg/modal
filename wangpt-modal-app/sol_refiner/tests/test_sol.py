import io
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from sol_refiner import control as sol_control
from sol_refiner import jobs as sol_jobs
from sol_refiner import storage as sol_storage
from sol_refiner.common import validate_frame_count, validate_request
from sol_refiner.runtime import encode_rgb_frames, probe_video, restore_audio, verify_output


def request(**overrides):
    return {"input_url": "s3://studio/source.mp4", "prompt": "A presenter speaking.",
            "output_width": 1344, "output_height": 768, **overrides}


@pytest.mark.parametrize("changes", [
    {"prompt": " "}, {"input_url": "file:///etc/passwd"},
    {"input_url": "http://example.com/a.mp4"}, {"input_url": "https://user:pass@example.com/a"},
    {"input_url": "s3://studio/a?token=x"}, {"output_width": True},
    {"output_height": 767}, {"output_width": 1920, "output_height": 1920},
    {"seed": -1}, {"decoder_seed": 2**63}, {"sampler": "other"},
])
def test_request_rejects_unsupported_inputs(changes):
    with pytest.raises(ValueError):
        validate_request(request(**changes))


def test_signed_https_and_portrait_requests_are_supported():
    result = validate_request(request(input_url="https://example.com/a?signature=abc",
                                      output_width=1080, output_height=1920))
    assert result["seed"] == result["decoder_seed"] == 0


@pytest.mark.parametrize("frames", [0, 1, 8, 120, 124, 249])
def test_frame_policy_never_silently_truncates(frames):
    with pytest.raises(ValueError):
        validate_frame_count(frames)


def test_frame_policy_accepts_reference_clip():
    validate_frame_count(121)


def test_explicit_padding_preserves_source_count_and_reaches_valid_length():
    assert validate_frame_count(124, "pad") == 129
    assert validate_frame_count(121, "pad") == 121
    assert validate_request(request(frame_policy="pad"))["frame_policy"] == "pad"
    assert validate_request(request())["frame_policy"] == "strict"
    with pytest.raises(ValueError, match="frame_policy"):
        validate_request(request(frame_policy="truncate"))


class Store:
    def __init__(self):
        self.data = {}

    def get(self, key, default=None):
        import copy
        return copy.deepcopy(self.data.get(key, default))

    def put(self, key, value):
        import copy
        self.data[key] = copy.deepcopy(value)


def test_dispatch_does_not_overwrite_fast_completed_worker(monkeypatch):
    store = Store()
    monkeypatch.setattr(sol_control, "jobs", store)

    def spawn(job_id, params):
        store.put(job_id, {"id": job_id, "status": "succeeded", "result": {"success": True}})
        return SimpleNamespace(object_id="fc-1")

    worker = SimpleNamespace(refine=SimpleNamespace(spawn=spawn))
    monkeypatch.setattr(sol_control.modal.Cls, "from_name", lambda *args: lambda: worker)
    submitted = sol_control.submit_refinement(request())
    assert sol_control.get_job(submitted["id"])["status"] == "succeeded"
    assert store.get(f"call:{submitted['id']}") == "fc-1"


def test_startup_failure_is_reconciled(monkeypatch):
    store = Store()
    store.put("job", {"id": "job", "status": "queued"})
    store.put("call:job", "fc-1")
    monkeypatch.setattr(sol_control, "jobs", store)

    def fail(**kwargs):
        raise RuntimeError("model not prepared")

    monkeypatch.setattr(sol_control.modal.FunctionCall, "from_id", lambda _: SimpleNamespace(get=fail))
    result = sol_control.get_job("job")
    assert result["status"] == "failed"
    assert result["error"]["phase"] == "modal"


def test_upload_failure_marks_job_failed_and_cleans_scratch(monkeypatch):
    store = Store()
    store.put("job", {"id": "job", "status": "queued"})
    seen = []
    monkeypatch.setattr(sol_jobs, "s3_client", lambda: (object(), "studio"))

    def download(url, path, *args):
        path.write_bytes(b"source")
        seen.append(path.parent)
        return path

    def refine(original, directory, params):
        output = directory / "refined.mp4"
        output.write_bytes(b"video")
        return output, {}

    def fail(*args):
        raise RuntimeError("upload verification failed")

    monkeypatch.setattr(sol_jobs, "download_input", download)
    monkeypatch.setattr(sol_jobs, "upload_output", fail)
    with pytest.raises(RuntimeError, match="verification"):
        sol_jobs.run_job(store, SimpleNamespace(refine=refine), "job", request())
    assert store.get("job")["status"] == "failed"
    assert store.get("job")["error"]["phase"] == "upload"
    assert not seen[0].exists()


def test_successful_job_publishes_verified_artifact_and_metrics(monkeypatch):
    store = Store()
    store.put("job", {"id": "job", "status": "queued"})
    monkeypatch.setattr(sol_jobs, "s3_client", lambda: (object(), "studio"))
    monkeypatch.setattr(sol_jobs, "download_input", lambda url, path, *args: path)
    artifact = {"uri": "s3://studio/runninghub/sol/job/refined.mp4"}
    monkeypatch.setattr(sol_jobs, "upload_output", lambda *args: artifact)
    runtime = SimpleNamespace(refine=lambda original, directory, params:
                              (directory / "refined.mp4", {"refine_ms": 10, "output": {"frames": 124}}))
    result = sol_jobs.run_job(store, runtime, "job", request(asset_id="presenter"))
    assert result["output"] == artifact
    assert result["output_media"] == {"frames": 124}
    assert result["asset_id"] == "presenter"
    assert result["refine_ms"] == 10
    assert store.get("job")["status"] == "succeeded"


def test_variable_frame_timestamps_rejected(monkeypatch):
    from sol_refiner import runtime as sol_runtime

    metadata = {"streams": [{"codec_type": "video", "codec_name": "h264", "width": 128,
                              "height": 128, "avg_frame_rate": "24/1", "duration": str(17 / 24)}]}
    times = [{"best_effort_timestamp_time": str(i / 24 + (0.01 if i == 5 else 0))}
             for i in range(17)]
    responses = iter([metadata, {"frames": times}])
    monkeypatch.setattr(sol_runtime, "_probe", lambda *args: next(responses))
    with pytest.raises(ValueError, match="variable-frame-rate"):
        probe_video(Path("source.mp4"))


def test_s3_download_verifies_digest_and_removes_partial(tmp_path):
    body = io.BytesIO(b"video")
    client = SimpleNamespace(get_object=lambda **kw: {
        "Body": body, "ContentLength": 5, "Metadata": {"sha256": "bad"},
    })
    with pytest.raises(ValueError, match="SHA-256"):
        sol_storage.download_input("s3://studio/a.mp4", tmp_path / "a.mp4", client, "studio")
    assert body.closed
    assert not list(tmp_path.iterdir())


def test_s3_wrong_bucket_rejected_before_download(tmp_path):
    with pytest.raises(ValueError, match="bucket"):
        sol_storage.download_input("s3://other/a.mp4", tmp_path / "a.mp4", object(), "studio")


def test_download_limit_is_enforced_on_actual_bytes(tmp_path):
    with pytest.raises(ValueError, match="limit"):
        sol_storage._copy_bounded(io.BytesIO(b"12345"), tmp_path / "a", None, 4)


def test_upload_requires_matching_remote_metadata(tmp_path):
    output = tmp_path / "refined.mp4"
    output.write_bytes(b"video")
    client = SimpleNamespace(upload_file=lambda *a, **kw: None,
                             head_object=lambda **kw: {"ContentLength": 5, "Metadata": {}})
    with pytest.raises(RuntimeError, match="verification"):
        sol_storage.upload_output(output, "job", client, "studio")


@pytest.fixture
def clip(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg tools are required for media integration tests")
    path = tmp_path / "source.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=128x128:rate=24",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", str(17 / 24),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    ], check=True, capture_output=True)
    return path


def test_real_media_probe_and_audio_remux(clip, tmp_path):
    source = probe_video(clip)
    assert source["frames"] == 17
    raw, final = tmp_path / "raw.mp4", tmp_path / "final.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-an", "-c:v", "copy", str(raw)],
                   check=True, capture_output=True)
    assert restore_audio(raw, clip, final, source) == "copied_aac"
    output = probe_video(final)
    verify_output(source, output, 128, 128)
    assert output["audio"]["duration"] == source["audio"]["duration"]


def test_real_incompatible_frame_count_rejected(clip, tmp_path):
    truncated = tmp_path / "16.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-frames:v", "16", "-an",
                    "-c:v", "libx264", str(truncated)], check=True, capture_output=True)
    with pytest.raises(ValueError, match="8k\\+1"):
        probe_video(truncated)
    assert probe_video(truncated, "pad")["frames"] == 16


def test_encoder_preserves_fractional_frame_rate(clip, tmp_path):
    output = tmp_path / "fractional.mp4"
    encode_rgb_frames((bytes(128 * 128 * 3) for _ in range(17)), output, 128, 128, "24000/1001")
    metadata = probe_video(output)
    assert metadata["fps_rational"] == "24000/1001"
    assert metadata["frames"] == 17


def test_sol_entrypoint_never_imports_wangp():
    import sys
    result = subprocess.run([sys.executable, "-c", "import sol_refiner.app, sol_refiner.control, sys; "
        "assert not {'app', 'control', 'wangpt_common', 'h3_latent'} & sys.modules.keys()"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
