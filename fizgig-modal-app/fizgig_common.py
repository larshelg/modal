"""Shared training contract; importing this module never constructs a GPU image."""

from __future__ import annotations

import re
import math
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, unquote, urlsplit

APP_NAME = "fizgig-modal-app"
DATA_VOLUME_NAME = "wangp-data"
JOB_DICT_NAME = "fizgig-modal-jobs"
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}

SUPPORTED_FAMILY = "minimax_h3"
SUPPORTED_FAMILIES = (SUPPORTED_FAMILY, "krea2")
SUPPORTED_PRESETS: dict[str, dict[str, Any]] = {
    "h3_character_fast": {
        "family": "minimax_h3",
        "network_dim": 8,
        "network_alpha": 8,
        "epochs": 40,
        "learning_rate": 2e-4,
        "optimizer_type": "adamw",
        "adapter_ramp": 0.0,
    },
    "h3_character_quality": {
        "family": "minimax_h3",
        "network_dim": 16,
        "network_alpha": 16,
        "epochs": 60,
        "learning_rate": 2e-4,
        "optimizer_type": "adamw",
        "adapter_ramp": 0.003,
    },
    "krea2_defaults": {
        "family": "krea2",
        "network_dim": 32,
        "network_alpha": 32,
        "epochs": 30,
        "learning_rate": 1e-4,
        "optimizer_type": "adamw8bit",
        "adaptive_lr": False,
        "adaptive_lr_min": 1e-4,
        "adaptive_lr_max": 4e-4,
        "auto_caption": True,
        "auto_recaption": True,
    },
    "krea2_ultra_fast": {
        "family": "krea2",
        "network_dim": 8,
        "network_alpha": 8,
        "epochs": 20,
        "learning_rate": 1e-4,
        "optimizer_type": "adamw8bit",
        "adaptive_lr": True,
        "adaptive_lr_min": 2e-4,
        "adaptive_lr_max": 4e-4,
        "auto_caption": True,
        "auto_recaption": True,
    },
}

# RefMods use the H3 family but produce reference latents, not LoRA weights.
REFMOD_PRESETS = {
    "h3_refmod_community": {"steps": 0, "max_refs": 8, "target_mp": 1.0},
    "h3_refmod_lite": {"steps": 200, "max_refs": 16, "target_mp": 0.5},
    "h3_refmod_quality": {"steps": 200, "max_refs": 16, "target_mp": 1.0},
}
for _name, _settings in REFMOD_PRESETS.items():
    SUPPORTED_PRESETS[_name] = {
        "family": "minimax_h3", "artifact_type": "refmod", "grid": "full",
        "base_model": "ref2va", "concept_type": "identity", "description": "",
        "clips": "still", "token_cap": 0,
        **_settings,
    }

REFMOD_REQUEST_KEYS = {"steps", "max_refs", "target_mp", "grid", "base_model", "concept_type", "description", "clips", "token_cap"}


def is_refmod(request: dict[str, Any]) -> bool:
    return request.get("preset") in REFMOD_PRESETS


def _refmod_settings(request: dict[str, Any], preset: dict[str, Any]) -> dict[str, Any]:
    for key in ("epochs", "trigger_word", "resume_from"):
        if key in request:
            raise ValueError(f"{key} is not supported for RefMod jobs")
    settings = {**preset, **{key: request[key] for key in REFMOD_REQUEST_KEYS if key in request}}
    settings["steps"] = _bounded_int(settings["steps"], "steps", 0, 2000)
    settings["max_refs"] = _bounded_int(settings["max_refs"], "max_refs", 1, 64)
    settings["token_cap"] = _bounded_int(settings["token_cap"], "token_cap", 0, 65536)
    for key, choices in (
        ("clips", ("still", "motion")),
        ("grid", ("full", "8", "16", "32")),
        ("base_model", ("ref2va", "fl2va")),
        ("concept_type", ("identity", "style", "pose_motion", "clothing", "background", "generic")),
    ):
        if not isinstance(settings[key], str) or settings[key] not in choices:
            raise ValueError(f"{key} must be one of: {', '.join(choices)}")
    mp = settings["target_mp"]
    if isinstance(mp, bool) or not isinstance(mp, (int, float)) or not math.isfinite(mp) or not 0.25 <= mp <= 1.0:
        raise ValueError("target_mp must be a finite number between 0.25 and 1.0")
    description = settings["description"]
    if not isinstance(description, str) or len(description) > 1000 or any(ord(c) < 32 for c in description):
        raise ValueError("description must be a single-line string of at most 1000 characters")
    settings["description"] = description.strip()
    return settings


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ALLOWED_REQUEST_KEYS = {
    "family",
    "dataset",
    "dataset_s3",
    "output_name",
    "preset",
    "trigger_word",
    "epochs",
    "resume_from",
} | REFMOD_REQUEST_KEYS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_component(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    value = value.strip()
    if not _SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            f"{field} must contain only letters, numbers, '.', '_' or '-' and "
            "must not contain a path"
        )
    return value


def _bounded_int(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    parsed = value
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return parsed


def validate_training_request(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    unknown = sorted(set(request) - _ALLOWED_REQUEST_KEYS)
    if unknown:
        raise ValueError(f"unsupported request fields: {', '.join(unknown)}")

    missing = sorted(
        field
        for field in ("family", "output_name", "preset")
        if field not in request
    )
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")

    family = request["family"]
    if not isinstance(family, str) or family not in SUPPORTED_FAMILIES:
        choices = ", ".join(SUPPORTED_FAMILIES)
        raise ValueError(f"family must be one of: {choices}")

    has_dataset = request.get("dataset") is not None
    has_s3 = request.get("dataset_s3") is not None
    if has_dataset == has_s3:
        raise ValueError("provide exactly one of dataset or dataset_s3")
    dataset = validate_component(request["dataset"], "dataset") if has_dataset else None
    dataset_s3 = normalize_dataset_s3(request["dataset_s3"]) if has_s3 else None
    output_name = validate_component(request["output_name"], "output_name")
    preset_name = request["preset"]
    if not isinstance(preset_name, str) or preset_name not in SUPPORTED_PRESETS:
        choices = ", ".join(sorted(SUPPORTED_PRESETS))
        raise ValueError(f"preset must be one of: {choices}")

    preset = dict(SUPPORTED_PRESETS[preset_name])
    if preset["family"] != family:
        raise ValueError(f"preset {preset_name!r} does not support family {family!r}")
    if is_refmod(request):
        return {
            **_refmod_settings(request, preset), "family": family, "dataset": dataset,
            "dataset_s3": dataset_s3, "output_name": output_name, "preset": preset_name,
            "seed": 42, "resume_from": None,
        }
    if REFMOD_REQUEST_KEYS & request.keys():
        raise ValueError("RefMod settings require an h3_refmod preset")
    epochs = _bounded_int(request.get("epochs", preset["epochs"]), "epochs", 1, 500)

    trigger_word = request.get("trigger_word")
    if trigger_word is not None:
        trigger_word = validate_component(trigger_word, "trigger_word")

    resume_from = request.get("resume_from")
    if resume_from is not None and resume_from != "latest":
        resume_from = validate_component(resume_from, "resume_from")
        if not resume_from.endswith("-state"):
            raise ValueError("resume_from must be 'latest' or a state-directory basename")

    return {
        **preset,
        "family": family,
        "dataset": dataset,
        "dataset_s3": dataset_s3,
        "output_name": output_name,
        "preset": preset_name,
        "trigger_word": trigger_word,
        "epochs": epochs,
        "seed": 42,
        "save_every_n_epochs": 1,
        "resume_from": resume_from,
    }


def training_intent(request: dict[str, Any], *, allow_resume: bool = False) -> dict[str, Any]:
    """Validate locally and send only the worker's allowlisted request fields."""
    if isinstance(request, dict) and "resume_from" in request and not allow_resume:
        raise ValueError("resume_from is controlled by the resume command")
    normalized = validate_training_request(request)
    return {
        key: normalized[key]
        for key in _ALLOWED_REQUEST_KEYS
        if normalized.get(key) is not None
    }


def parse_dataset_s3(uri: str, configured_bucket: str | None = None) -> tuple[str, str]:
    """Accept a bucket-scoped S3 folder, never credentials or a presigned URL."""
    if not isinstance(uri, str) or any(ord(char) < 32 for char in uri):
        raise ValueError("dataset_s3 must use s3://BUCKET/PREFIX/")
    parsed = urlsplit(uri)
    if (parsed.scheme != "s3" or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", parsed.netloc)
            or parsed.query or parsed.fragment):
        raise ValueError("dataset_s3 must use s3://BUCKET/PREFIX/ without credentials, query or fragment")
    if configured_bucket is not None and parsed.netloc != configured_bucket:
        raise ValueError("dataset_s3 bucket does not match configured S3_BUCKET")
    prefix = unquote(parsed.path.removeprefix("/"))
    if (not prefix.strip("/") or "\\" in prefix or any(ord(char) < 32 for char in prefix)
            or any(part in {".", "..", ""} for part in prefix.removesuffix("/").split("/"))):
        raise ValueError("dataset_s3 must name a non-empty folder prefix without traversal")
    return parsed.netloc, prefix.rstrip("/") + "/"


def normalize_dataset_s3(uri: str) -> str:
    bucket, prefix = parse_dataset_s3(uri)
    return f"s3://{bucket}/{quote(prefix, safe='/')}"
