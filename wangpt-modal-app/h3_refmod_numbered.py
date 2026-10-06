"""Opt-in image/video RefMod presentation for WanGP's H3 text encoder.

No ComfyUI dependency. The upstream plugin still owns basic latent injection;
numbered requests use a strict path so failed references cannot be ignored.
"""
from __future__ import annotations

import functools
import json
import math
from collections import Counter
from typing import Any

SETTING = "h3_refmod_state"
STATE_ATTR = "_modal_numbered_refmods"
REPORTS_ATTR = "_modal_refmod_reference_maps"


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return float(value)


def numbered_options(settings: dict[str, Any]) -> dict[str, Any] | None:
    custom = settings.get("custom_settings") or {}
    raw = custom.get(SETTING) if isinstance(custom, dict) else None
    if not raw:
        return None
    try:
        options = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError) as exc:
        raise ValueError("h3_refmod_state must contain valid JSON") from exc
    if not isinstance(options, dict):
        raise ValueError("h3_refmod_state must encode an object")
    enabled = options.get("text_encode", False)
    if type(enabled) is not bool:
        raise ValueError("RefMod text_encode must be a boolean")
    if not enabled:
        return None
    if options.get("scramble_seed", -1) != -1:
        raise ValueError("Numbered RefMods require scramble_seed=-1 for stable reference order")
    _number(options.get("retention", 1.0), "retention", 0, 2)
    _number(options.get("reference_fps", 24.0), "reference_fps", 1, 120)
    rows = options.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Numbered RefMods require a non-empty rows list")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each RefMod row must be an object")
        name = row.get("mod")
        if not isinstance(name, str) or not name or any(
            not part or any(not (c.isalnum() or c in "_-") for c in part)
            for part in name.split("/")
        ):
            raise ValueError("RefMod mod must be a relative library name without the file extension")
        _number(row.get("strength", 1.0), "strength", 0, 2)
        copies = row.get("copies", 1)
        if type(copies) is not int or not 1 <= copies <= 10:
            raise ValueError("RefMod copies must be an integer from 1 to 10")
        _number(row.get("reference_fps", options.get("reference_fps", 24)), "reference_fps", 1, 120)
    return options


def validate_numbered_request(settings: dict[str, Any]) -> None:
    if numbered_options(settings) is None:
        return
    if "W" in str(settings.get("multi_prompts_gen_type", "")):
        raise ValueError("Numbered RefMods do not support sliding-window prompt modes")
    length, window = settings.get("video_length"), settings.get("sliding_window_size")
    if isinstance(length, (int, float)) and isinstance(window, (int, float)) and window > 0 and length > window:
        raise ValueError("Numbered RefMods require video_length <= sliding_window_size (one window)")
    continuing = settings.get("video_source") or settings.get("plugin_data", {}).get("h3_latent_prototype", {}).get("continue")
    if continuing and isinstance(length, (int, float)) and isinstance(window, (int, float)) and window > 0:
        # WanGP shares one source frame, but additional assembly overlap consumes
        # window capacity. Use the requested overlap as a conservative bound;
        # saved latent context (18/35/52 frames) is independent of this setting.
        overlap = max(1, int(settings.get("sliding_window_overlap", 1)))
        if length + overlap - 1 > window:
            raise ValueError("Numbered RefMod continuation requires video_length + sliding_window_overlap - 1 <= sliding_window_size (one window)")


def prepare_numbered_job(wgp: Any, settings: dict[str, Any]) -> None:
    setattr(wgp, REPORTS_ATTR, [])
    validate_numbered_request(settings)
    if numbered_options(settings) is not None:
        base = wgp.get_base_model_type(settings["model_type"]) or ""
        if not base.startswith("minimax_h3_ref2va"):
            raise ValueError("Numbered RefMods require an H3 Ref2VA model")
        if wgp.has_slash_commands([str(settings.get("prompt", ""))]):
            raise ValueError("Numbered RefMods do not support frame-scheduler slash commands")


def video_sample_indices(frame_count: int, fps: float) -> tuple[list[int], list[float]]:
    """Native Qwen video presentation: two samples per second, no accumulated drift."""
    count = max(1, math.ceil(frame_count * 2 / fps))
    times = [i / 2 for i in range(count)]
    return [min(round(t * fps), frame_count - 1) for t in times], times


def reference_mapping(presentation: list[dict]) -> list[dict]:
    counts = {"image": 0, "video": 0, "audio": 0}
    labels = {"image": "Picture", "video": "Video", "audio": "Audio"}
    mapping = []
    for item in presentation:
        kind = item["type"]
        counts[kind] += 1
        mapping.append({"label": f"<{labels[kind]} {counts[kind]}>",
                        "kind": kind, **item.get("_refmod", {"source": "native"})})
    return mapping


def _decode_presentation(pipeline, sentinel, kind, h3):
    import torch

    pipeline._check_abort()
    pipeline._use_shared_components()
    with torch.inference_mode():
        pixels = pipeline.vae.decode(sentinel.latent.to(
            device=pipeline.device, dtype=pipeline.vae._model_dtype,
        )).clamp_(-1, 1)[0].cpu()
    if pixels.ndim != 4 or pixels.shape[0] != 3 or pixels.shape[1] < 1:
        raise ValueError("H3 VAE returned invalid RefMod pixels")
    item = {"type": kind, "_refmod": sentinel.info}
    if kind == "image":
        item["frames"] = h3._qwen_frames(pixels[:, :1])
    else:
        indices, times = video_sample_indices(pixels.shape[1], sentinel.info["reference_fps"])
        item.update(frames=h3._qwen_frames(pixels[:, indices]), timestamps=times)
    pipeline._check_abort()
    return item


def install_numbered_hooks(wgp: Any, patches: Any, storage: Any, h3: Any) -> None:
    Pipeline = h3.MiniMaxH3Pipeline
    if getattr(Pipeline, "_modal_numbered_installed", False):
        return
    basic_generate = Pipeline.generate
    native_generate = getattr(basic_generate, "__wrapped__", None)
    if native_generate is None:
        raise RuntimeError("Pinned RefMod generate wrapper changed; cannot install numbered encoding")
    basic_image = Pipeline._add_image_reference
    basic_video = Pipeline._add_video_reference
    native_encode = Pipeline._encode_prompt

    class ImageRef(patches._RefModImageSentinel):
        def __init__(self, latent, info):
            super().__init__(latent)
            self.info = info

    class VideoRef(patches._RefModVideoSentinel):
        def __init__(self, latent, info):
            super().__init__(latent)
            self.info = info

        def __getitem__(self, key):
            trimmed = super().__getitem__(key)
            if trimmed is self:
                return self
            return VideoRef(trimmed.latent, self.info)

    def inject(pipeline, kwargs, options):
        images = list(kwargs.get("input_ref_images") or [])
        video_keys = ("input_frames", "input_frames2", "input_frames3")
        video_flags = str(kwargs.get("video_prompt_type") or "")
        videos = []
        active = 0
        for row_index, row in enumerate(options["rows"]):
            strength = min(2.0, row.get("strength", 1.0) * options.get("retention", 1.0))
            if strength == 0:
                continue
            mod = storage.load_refmod(row["mod"])
            if mod.kind not in {"image", "video"}:
                raise ValueError("Numbered mode currently supports image/video RefMods only")
            latent = mod.weighted_latent(strength, curve=options.get("curve"))
            if latent is None or latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != 24 or any(n <= 0 for n in latent.shape):
                raise ValueError(f"Invalid visual RefMod latent: {row['mod']}")
            info = {"source": "refmod", "mod": row["mod"], "row": row_index,
                    "strength": strength, "copies": row.get("copies", 1)}
            if mod.kind == "image":
                for copy in range(info["copies"]):
                    for frame in range(latent.shape[2]):
                        images.append(ImageRef(latent[:, :, frame:frame + 1],
                                               {**info, "copy": copy, "frame": frame}))
            else:
                fps = row.get("reference_fps", options.get("reference_fps", 24.0))
                videos.append(VideoRef(latent.repeat(1, 1, info["copies"], 1, 1),
                                       {**info, "reference_fps": fps}))
            active += 1
        if not active:
            raise ValueError("Numbered RefMods require at least one active reference")
        if len(images) > 9:
            raise ValueError("Numbered RefMods support at most 9 image references including normal refs and copies")
        kwargs["input_ref_images"] = images
        if videos:
            if any(flag in video_flags for flag in ("G", "1")) or ("U" in video_flags and "-" not in video_flags):
                raise ValueError("Numbered video RefMods cannot be combined with control videos or video excerpts")
            if "K" in str(kwargs.get("audio_prompt_type") or ""):
                raise ValueError("Numbered video RefMods do not carry reference-video soundtracks; use separate audio references")
            for video in videos:
                slot = next((key for key in video_keys if kwargs.get(key) is None), None)
                if slot is None:
                    raise ValueError("Numbered RefMods support at most 3 reference videos including normal refs")
                kwargs[slot] = video
            for key, flag in zip(video_keys, ("V", "+", "*")):
                if kwargs.get(key) is not None and flag not in video_flags:
                    video_flags += flag
            if "V" not in video_flags:
                video_flags += "V"
            if "-" not in video_flags:
                video_flags += "-"
            kwargs["video_prompt_type"] = video_flags
        expected = [ref.info for ref in images if isinstance(ref, ImageRef)] + [ref.info for ref in videos]
        getattr(pipeline, STATE_ATTR)["expected"] = Counter(
            (info["row"], info.get("copy"), info.get("frame")) for info in expected
        )
        return kwargs

    def present(pipeline, ref, kind, presentation):
        state = getattr(pipeline, STATE_ATTR, None)
        if state is None:
            return
        key = (ref.info["row"], ref.info.get("copy"), ref.info.get("frame"), tuple(ref.latent.shape))
        if key not in state["decoded"]:
            state["decoded"][key] = _decode_presentation(pipeline, ref, kind, h3)
        presentation.append(state["decoded"][key])

    @functools.wraps(basic_image)
    def image_reference(self, image, target_width, target_height, image_refs_relative_size, presentation, visual_latents, refs):
        if isinstance(image, ImageRef):
            present(self, image, "image", presentation)
        return basic_image(self, image, target_width, target_height, image_refs_relative_size, presentation, visual_latents, refs)

    @functools.wraps(basic_video)
    def video_reference(self, video, soundtrack, fps, presentation, visual_latents, audio_latents, refs):
        if isinstance(video, VideoRef):
            present(self, video, "video", presentation)
        return basic_video(self, video, soundtrack, fps, presentation, visual_latents, audio_latents, refs)

    @functools.wraps(native_encode)
    def encode_prompt(self, prompt, presentation):
        state = getattr(self, STATE_ATTR, None)
        if state is not None and any("_refmod" in item for item in presentation):
            mapping = reference_mapping(presentation)
            actual = Counter((r["row"], r.get("copy"), r.get("frame")) for r in mapping if r["source"] == "refmod")
            if actual != state["expected"]:
                raise RuntimeError("Not all selected RefMods reached the H3 reference presentation")
            if state["mapping"] is None:
                state["mapping"] = mapping
                getattr(wgp, REPORTS_ATTR).append({"references": mapping})
                print("[H3 RefMods] Numbered references: " + json.dumps(mapping), flush=True)
            elif state["mapping"] != mapping:
                raise RuntimeError("RefMod numbering changed between H3 encoding phases")
        return native_encode(self, prompt, presentation)

    @functools.wraps(basic_generate)
    def generate(self, *args, **kwargs):
        options = numbered_options(kwargs)
        if options is None:
            return basic_generate(self, *args, **kwargs)
        if not self.reference_mode or self.fixed_prompt is not None or self.audio_only:
            raise ValueError("Numbered RefMods require a visual H3 Ref2VA pipeline with a text encoder")
        if kwargs.get("window_no", 1) not in (0, 1):
            raise ValueError("Numbered RefMods do not support sliding windows; continue the saved result in a new request")
        if (kwargs.get("custom_settings") or {}).get(patches.SETTING_EXTRACT):
            raise ValueError("Numbered RefMods cannot be combined with extraction")
        patches._reset_gen_state(self)
        state = {"decoded": {}, "mapping": None}
        setattr(self, STATE_ATTR, state)
        try:
            injected = inject(self, dict(kwargs), options)
            result = native_generate(self, *args, **injected)
            if state["mapping"] is None:
                raise RuntimeError("Numbered RefMods were not presented to H3's text encoder")
            return result
        finally:
            delattr(self, STATE_ATTR)
            patches._reset_gen_state(self)

    Pipeline.generate = generate
    Pipeline._add_image_reference = image_reference
    Pipeline._add_video_reference = video_reference
    Pipeline._encode_prompt = encode_prompt
    Pipeline._modal_numbered_installed = True
    setattr(wgp, REPORTS_ATTR, [])
