"""Preserve the video timeline when muxing audio for active H3 latent jobs."""

from __future__ import annotations

import functools
import inspect
import json
import subprocess
from fractions import Fraction
from typing import Any, Callable


def probe_video(path: str) -> dict[str, Any]:
    """Decode-count frames so container duration cannot hide a truncated tail."""
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames,avg_frame_rate,width,height",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True, timeout=120,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("H3 latent export requires one readable video stream")
    stream = streams[0]
    frames = int(stream["nb_read_frames"])
    fps = Fraction(stream["avg_frame_rate"])
    if frames <= 0 or fps <= 0:
        raise ValueError("H3 latent export has an empty or invalid video timeline")
    return {
        "frames": frames, "fps": fps,
        "width": int(stream["width"]), "height": int(stream["height"]),
    }


def make_preserving_mux(original: Callable, job_state: Any) -> Callable:
    signature = inspect.signature(original)

    @functools.wraps(original)
    def mux(*args: Any, **kwargs: Any) -> Any:
        job = job_state.get()
        if not job or not job.get("enabled"):
            return original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        before = probe_video(bound.arguments["video_path"])
        bound.arguments["video_duration"] = float(before["frames"] / before["fps"])
        result = original(*bound.args, **bound.kwargs)
        after = probe_video(bound.arguments["save_path_tmp"])
        if after != before:
            raise RuntimeError(f"H3 latent mux changed the video timeline: {before} -> {after}")
        print(f'[H3 Latent] Mux preserved {after["frames"]} frames at {after["fps"]} FPS.', flush=True)
        return result

    return mux


def make_verified_record(
    original: Callable, job_state: Any, validate_frames: Callable,
) -> Callable:
    """Run inside the plugin's record wrapper, before it writes the checkpoint."""
    signature = inspect.signature(original)

    @functools.wraps(original)
    def record(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        job = job_state.get()
        if not job or not job.get("pending"):
            return result
        bound = signature.bind(*args, **kwargs)
        info = job["pending"]["info"]
        actual = probe_video(bound.arguments["video_path"])
        validate_frames(info, actual["frames"])
        if (
            abs(float(actual["fps"]) - float(info["fps"])) > 1e-6
            or (actual["width"], actual["height"]) != (info["width"], info["height"])
        ):
            raise RuntimeError("H3 latent final video geometry/FPS differs from its checkpoint")
        print(f'[H3 Latent] Final export verified: {actual["frames"]} frames.', flush=True)
        return result

    return record
