import json
from types import SimpleNamespace

import pytest

from h3_refmod_numbered import (
    REPORTS_ATTR, numbered_options, prepare_numbered_job, reference_mapping,
    validate_numbered_request, video_sample_indices,
)
from wangpt_common import validate_job_request


def request(**options):
    return {"custom_settings": {"h3_refmod_state": json.dumps({
        "text_encode": True, "rows": [{"mod": "alice"}], **options,
    })}}


@pytest.mark.parametrize("options, error", [
    ({"text_encode": "true"}, "boolean"),
    ({"rows": []}, "non-empty"),
    ({"rows": [{"mod": "../alice"}]}, "relative library"),
    ({"rows": [{"mod": "alice", "strength": float("nan")}]}, "strength"),
    ({"rows": [{"mod": "alice", "copies": 1.5}]}, "copies"),
    ({"rows": [{"mod": "alice", "reference_fps": 0}]}, "reference_fps"),
    ({"retention": -1}, "retention"),
    ({"scramble_seed": 42}, "stable reference order"),
])
def test_invalid_numbered_settings_fail_locally(options, error):
    with pytest.raises(ValueError, match=error):
        validate_job_request("minimax_h3_ref2va", request(**options))


@pytest.mark.parametrize("settings", [
    {"video_length": 200, "sliding_window_size": 124},
    {"multi_prompts_gen_type": "PW"},
    {"video_source": "/data/previous.mp4", "video_length": 124, "sliding_window_size": 124, "sliding_window_overlap": 18},
])
def test_sliding_windows_rejected_including_continuation_overlap(settings):
    with pytest.raises(ValueError):
        validate_numbered_request({**request(), **settings})


@pytest.mark.parametrize("length, overlap", [(124, 1), (107, 18)])
def test_single_window_latent_continuation_allowed(length, overlap):
    settings = {**request(), "video_source": "/data/previous.mp4", "video_length": length,
                "sliding_window_size": 124, "sliding_window_overlap": overlap,
                "plugin_data": {"h3_latent_prototype": {"continue": True, "video_context_frames": 52}}}
    validate_job_request("minimax_h3_ref2va_singularity_pruned", settings)


def test_single_window_and_existing_basic_requests():
    validate_numbered_request({**request(), "video_length": 107, "sliding_window_size": 124})
    validate_numbered_request({**request(text_encode=False), "video_length": 500, "sliding_window_size": 124})
    assert numbered_options({}) is None


def test_warm_worker_clears_previous_reports_and_checks_model_family():
    wgp = SimpleNamespace(get_base_model_type=lambda _: "qwen", **{REPORTS_ATTR: ["old"]})
    with pytest.raises(ValueError, match="H3 Ref2VA"):
        prepare_numbered_job(wgp, {"model_type": "qwen", **request()})
    assert getattr(wgp, REPORTS_ATTR) == []


def test_mapping_includes_normal_images_and_independent_video_numbers():
    mapping = reference_mapping([
        {"type": "image"}, {"type": "image"},
        {"type": "image", "_refmod": {"source": "refmod", "mod": "alice"}},
        {"type": "video", "_refmod": {"source": "refmod", "mod": "walk"}},
        {"type": "audio"},
    ])
    assert [entry["label"] for entry in mapping] == ["<Picture 1>", "<Picture 2>", "<Picture 3>", "<Video 1>", "<Audio 1>"]
    assert mapping[2]["mod"] == "alice"


def test_noninteger_fps_sampling_uses_timestamps_without_drift():
    indices, times = video_sample_indices(61, 29.97)
    assert times == [0, 0.5, 1.0, 1.5, 2.0]
    assert indices == [0, 15, 30, 45, 60]
    assert video_sample_indices(1, 24) == ([0], [0.0])


def test_frame_scheduler_preflight_receives_prompt_list():
    def has_slash(prompts):
        assert isinstance(prompts, list)
        return "/frames" in prompts[0]
    wgp = SimpleNamespace(get_base_model_type=lambda _: "minimax_h3_ref2va", has_slash_commands=has_slash)
    with pytest.raises(ValueError, match="frame-scheduler"):
        prepare_numbered_job(wgp, {**request(), "model_type": "h3", "prompt": "/frames 107"})
