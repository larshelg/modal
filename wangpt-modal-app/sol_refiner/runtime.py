"""Media checks and an in-process adapter for NVIDIA's pinned H3 refiner."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from sol_refiner.common import MAX_FRAMES, MAX_PIXELS, validate_frame_count
from sol_refiner.versions import MODEL_REVISION, SANA_COMMIT


def _probe(path: Path, *options: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", *options, "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    return json.loads(result.stdout)


def probe_video(path: Path, frame_policy: str = "strict") -> dict[str, Any]:
    data = _probe(path, "-show_streams", "-show_format")
    videos = [s for s in data["streams"] if s["codec_type"] == "video"]
    audios = [s for s in data["streams"] if s["codec_type"] == "audio"]
    if len(videos) != 1 or len(audios) > 1:
        raise ValueError("input must have exactly one video and at most one audio stream")
    video = videos[0]
    fps = Fraction(video["avg_frame_rate"])
    if not 1 <= fps <= 60:
        raise ValueError("frame rate must be between 1 and 60 fps")
    if video["width"] * video["height"] > MAX_PIXELS:
        raise ValueError("video exceeds the initial 1920x1088 pixel limit")
    if any(item.get("rotation", 0) for item in video.get("side_data_list", [])):
        raise ValueError("normalize video rotation before refinement")
    duration = float(video.get("duration", data.get("format", {}).get("duration", 0)))
    if not math.isfinite(duration) or duration <= 0 or duration > (MAX_FRAMES + 1) / float(fps):
        raise ValueError("video duration is missing or exceeds the initial frame limit")
    frame_data = _probe(path, "-select_streams", "v:0", "-show_frames",
                        "-show_entries", "frame=best_effort_timestamp_time")
    frames = frame_data.get("frames", [])
    validate_frame_count(len(frames), frame_policy)
    times = [float(f["best_effort_timestamp_time"]) for f in frames]
    tolerance = max(0.0001, 0.01 / float(fps))
    if abs(times[0]) > tolerance:
        raise ValueError("video must start at timestamp zero; normalize it before refinement")
    if any(not math.isfinite(t) or abs(t - i / float(fps)) > tolerance for i, t in enumerate(times)):
        raise ValueError("variable-frame-rate input is unsupported; normalize to constant fps first")
    expected_duration = len(frames) / float(fps)
    if abs(duration - expected_duration) > 1 / float(fps):
        raise ValueError("video duration does not match its frame timestamps")
    audio = None
    if audios:
        stream = audios[0]
        start = float(stream.get("start_time", 0))
        length = float(stream.get("duration", 0))
        if not math.isfinite(start) or abs(start) > 0.001 or not math.isfinite(length) or length <= 0:
            raise ValueError("audio must have a known duration and start at timestamp zero")
        if abs(length - expected_duration) > max(0.1, 1 / float(fps)):
            raise ValueError("audio/video durations differ; normalize the clip before refinement")
        audio = {"codec": stream["codec_name"], "duration": length, "start_time": start}
    return {"width": video["width"], "height": video["height"], "frames": len(frames),
            "fps": float(fps), "fps_rational": str(fps), "duration": expected_duration,
            "codec": video["codec_name"], "audio": audio}


def verify_output(source: dict, output: dict, width: int, height: int) -> None:
    if (output["width"], output["height"]) != (width, height):
        raise RuntimeError("refiner output dimensions do not match the request")
    if output["frames"] != source["frames"]:
        raise RuntimeError("refiner changed frame count")
    if abs(output["fps"] - source["fps"]) > 0.001:
        raise RuntimeError("refiner changed frame rate")
    if abs(output["duration"] - source["duration"]) > 1 / source["fps"]:
        raise RuntimeError("refiner changed video duration")


def restore_audio(refined: Path, original: Path, destination: Path, source: dict) -> str:
    if not source["audio"]:
        refined.replace(destination)
        return "none"
    copy = source["audio"]["codec"] == "aac"
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-n", "-i", str(refined), "-i", str(original),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy" if copy else "aac",
        "-map_metadata", "-1", "-movflags", "+faststart", str(destination),
    ], check=True, capture_output=True, timeout=120)
    return "copied_aac" if copy else "encoded_aac"


def encode_rgb_frames(frames, destination: Path, width: int, height: int, fps: str) -> None:
    """Encode at the exact source rational rate rather than rounding a float."""
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([
            "ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", f"{width}x{height}", "-r", fps, "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(destination),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            for frame in frames:
                if len(frame) != width * height * 3:
                    raise RuntimeError("refiner returned an invalid RGB frame")
                process.stdin.write(frame)
            process.stdin.close()
            if process.wait(timeout=120):
                errors.seek(0)
                raise RuntimeError("video encoding failed: " + errors.read(2048).decode(errors="replace"))
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            if not process.stdin.closed:
                process.stdin.close()


def snapshot_path(root: Path = Path("/models")) -> Path:
    manifest = root / "manifests" / f"{MODEL_REVISION}.json"
    if not manifest.is_file():
        raise RuntimeError("Sol model is not prepared. Run: modal run -m sol_refiner.app::prepare")
    data = json.loads(manifest.read_text())
    if data.get("revision") != MODEL_REVISION:
        raise RuntimeError("Sol model manifest has the wrong revision")
    path = root / "huggingface/hub" / "models--Efficient-Large-Model--SoL-Refiner-LTX-2.5-for-MiniMax-H3" / "snapshots" / MODEL_REVISION
    for name, size in data["files"].items():
        file = path / name
        if not file.is_file() or file.stat().st_size != size:
            raise RuntimeError(f"Sol snapshot is incomplete: {name}; rerun model preparation")
    return path


class SolRuntime:
    def __init__(self, model_path: Path):
        started = time.monotonic()
        import torch
        from sol_refiner_h3 import SoLRefinerH3Pipeline
        from sol_refiner_h3.attention import LocalNattenProcessor

        self.pipe = SoLRefinerH3Pipeline.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, local_files_only=True,
        )
        self.pipe.diffusion_decoder.set_attn_processor(LocalNattenProcessor())
        self.pipe.diffusion_decoder.enable_tiling(
            tile_sample_min_height=768, tile_sample_min_width=768, tile_sample_min_num_frames=128,
            tile_sample_stride_height=512, tile_sample_stride_width=512, tile_sample_stride_num_frames=80,
        )
        self.pipe.enable_model_cpu_offload()
        torch.cuda.synchronize()
        self.model_load_ms = round((time.monotonic() - started) * 1000)
        self.calls = 0

    def refine(self, original: Path, directory: Path, request: dict) -> tuple[Path, dict]:
        import torch
        from diffusers.utils import load_video

        started = time.monotonic()
        frame_policy = request.get("frame_policy", "strict")
        source = probe_video(original, frame_policy)
        width, height = request["output_width"], request["output_height"]
        # Avoid silently stretching a composition into another aspect ratio.
        if abs((width / height) / (source["width"] / source["height"]) - 1) > 0.02:
            raise ValueError("output aspect ratio must match the source within 2%")
        frames = load_video(str(original))
        if len(frames) != source["frames"]:
            raise RuntimeError("video decoder and ffprobe disagree on frame count")
        inference_frames = validate_frame_count(len(frames), frame_policy)
        padding = inference_frames - len(frames)
        frames.extend(frames[-1].copy() for _ in range(padding))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        inference_start = time.monotonic()
        result = self.pipe(
            frames, request["prompt"], width=width, height=height, frame_rate=source["fps"],
            generator=torch.Generator("cuda").manual_seed(request["seed"]),
            decoder_generator=torch.Generator("cuda").manual_seed(request["decoder_seed"]),
        ).frames[0]
        torch.cuda.synchronize()
        refine_ms = round((time.monotonic() - inference_start) * 1000)
        if len(result) != inference_frames:
            raise RuntimeError("Sol changed the frame count; output was not published")
        result = result[:source["frames"]]
        raw, final = directory / "video-only.mp4", directory / "refined.mp4"
        pixels = ((frame.clip(0, 1) * 255).astype("uint8").tobytes() for frame in result)
        encode_rgb_frames(pixels, raw, width, height, source["fps_rational"])
        audio_handling = restore_audio(raw, original, final, source)
        output = probe_video(final, frame_policy)
        verify_output(source, output, width, height)
        if bool(output["audio"]) != bool(source["audio"]):
            raise RuntimeError("output lost or unexpectedly added audio")
        if source["audio"] and abs(output["audio"]["duration"] - source["audio"]["duration"]) > 0.1:
            raise RuntimeError("audio duration changed during remux")
        metrics = {
            "input": source, "output": output, "audio_handling": audio_handling,
            "model_revision": MODEL_REVISION, "sana_commit": SANA_COMMIT,
            "seed": request["seed"], "decoder_seed": request["decoder_seed"],
            "frame_policy": frame_policy, "inference_frames": inference_frames, "padded_frames": padding,
            "gpu": torch.cuda.get_device_name(), "first_request": self.calls == 0,
            "model_load_ms": self.model_load_ms, "refine_ms": refine_ms,
            "media_ms": round((time.monotonic() - started) * 1000),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        self.calls += 1
        return final, metrics
