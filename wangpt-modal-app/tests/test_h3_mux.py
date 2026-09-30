"""CPU regressions; set WANGP_TEST_ROOT to a checkout of the pinned Wan2GP."""

import ast
import json
import os
import shutil
import subprocess
from contextvars import ContextVar
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

import h3_mux
from h3_latent import validate_frame_alignment


def test_disabled_job_uses_native_mux_without_probing(monkeypatch):
    job = ContextVar("disabled", default=None)
    monkeypatch.setattr(h3_mux, "probe_video", lambda path: pytest.fail("unexpected probe"))
    seen = []

    def original(*args, **kwargs):
        seen.append((args, kwargs))
        return "native"

    assert h3_mux.make_preserving_mux(original, job)("out", "in") == "native"
    assert seen == [(("out", "in"), {})]


def test_active_mux_rejects_frame_loss(monkeypatch):
    job = ContextVar("active", default={"enabled": True})
    base = {"frames": 247, "fps": Fraction(24), "width": 832, "height": 480}
    monkeypatch.setattr(h3_mux, "probe_video", lambda path: {**base, "frames": 243 if path == "out" else 247})

    def native(save_path_tmp, video_path, *, video_duration=None):
        assert video_duration == pytest.approx(247 / 24)

    with pytest.raises(RuntimeError, match="changed the video timeline"):
        h3_mux.make_preserving_mux(native, job)("out", "in")


def test_final_metadata_rewrite_is_verified_before_checkpoint_can_be_saved(monkeypatch):
    info = {"output_frames": 247, "fps": 24, "width": 832, "height": 480}
    job = ContextVar("pending", default={"pending": {"info": info}})
    seen = []

    def metadata(video_path):
        seen.append("metadata written")

    def probe(path):
        assert seen == ["metadata written"]
        return {"frames": 243, "fps": Fraction(24), "width": 832, "height": 480}

    monkeypatch.setattr(h3_mux, "probe_video", probe)
    with pytest.raises(ValueError, match="243 video frames.*247"):
        h3_mux.make_verified_record(metadata, job, validate_frame_alignment)("output.mp4")


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=60)


@pytest.fixture(scope="module")
def real_mux(tmp_path_factory):
    root = os.environ.get("WANGP_TEST_ROOT")
    if not root or not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Set WANGP_TEST_ROOT and install ffmpeg/ffprobe for native mux regression")
    directory = tmp_path_factory.mktemp("native-h3-mux")
    source = Path(root) / "shared/utils/audio_video.py"
    copy = directory / "shared/utils/audio_video.py"
    copy.parent.mkdir(parents=True)
    shutil.copyfile(source, copy)
    patch = Path(__file__).resolve().parents[1] / "patches/h3-latent-mux.patch"
    run("patch", "-p1", "-d", str(directory), "-i", str(patch))

    # Execute only the actual upstream mux/codec functions; no GPU imports needed.
    names = {"combine_and_concatenate_video_with_audio_tracks", "get_mp4_audio_codec_settings", "get_audio_file_channels"}
    tree = ast.parse(copy.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]

    def probe(path, cmd="ffprobe"):
        return json.loads(run(cmd, "-v", "error", "-show_streams", "-of", "json", str(path)).stdout)

    env = {
        "subprocess": subprocess, "os": os,
        "_ffmpeg_binary": lambda: "ffmpeg", "_ffprobe_binary": lambda: "ffprobe",
        "ffmpeg": SimpleNamespace(probe=probe),
    }
    exec(compile(tree, str(copy), "exec"), env)
    mux = env["combine_and_concatenate_video_with_audio_tracks"]
    video = directory / "video.mp4"
    run("ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=24",
        "-frames:v", "247", "-c:v", "libx264", "-bf", "3", str(video))
    audios = {}
    for name, duration in (("source", 124 / 24), ("exact", 123 / 24), ("short", 4.9), ("long", 7)):
        path = directory / f"{name}.wav"
        run("ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=32000:duration={duration}", str(path))
        audios[name] = str(path)
    return mux, str(video), audios


def packet_hashes(path):
    packets = json.loads(run(
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_data_hash", "sha256", "-show_entries", "packet=data_hash,pts_time,duration_time",
        "-of", "json", str(path),
    ).stdout)["packets"]
    return packets


def test_native_shortest_regression_is_reproduced(real_mux, tmp_path):
    mux, video, audio = real_mux
    output = str(tmp_path / "old.mp4")
    mux(output, video, [audio["source"]], [audio["exact"]], 124 / 24, 32000, new_audio_from_start=True)
    actual = h3_mux.probe_video(output)["frames"]
    assert actual < 247
    print(f"Native -shortest regression: 247 input frames -> {actual} output frames")


@pytest.mark.parametrize("case", ["exact", "short", "long", "multiple", "initial", "silence"])
def test_bounded_audio_preserves_all_video_packets(real_mux, tmp_path, case):
    native, video, audio = real_mux
    output = str(tmp_path / f"fixed-{case}.mp4")
    source = [audio["source"]]
    new = [audio[case if case in ("exact", "short", "long") else "long"]]
    duration = 124 / 24
    if case == "multiple":
        new = [audio["short"], audio["long"]]
    elif case == "initial":
        source, duration = [], 0
    elif case == "silence":
        source, new = [], []
    job = ContextVar("real-active", default={"enabled": True})
    h3_mux.make_preserving_mux(native, job)(output, video, source, new, duration, 32000, new_audio_from_start=True)
    assert h3_mux.probe_video(output)["frames"] == 247
    assert packet_hashes(output) == packet_hashes(video)
    streams = json.loads(run("ffprobe", "-v", "error", "-show_streams", "-of", "json", output).stdout)["streams"]
    audio_streams = [stream for stream in streams if stream["codec_type"] == "audio"]
    assert len(audio_streams) == (2 if case == "multiple" else 1)
    assert all(abs(float(stream["duration"]) - 247 / 24) < 0.002 for stream in audio_streams)
