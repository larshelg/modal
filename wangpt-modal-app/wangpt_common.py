"""Shared constants and pure request-routing helpers for WanGP."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


WORKER_APP_NAME = "wangpt-modal-app"
JOB_DICT_NAME = "wangpt-modal-jobs"
CATALOG_DICT_NAME = "wangpt-model-catalogs"
WAN_COMMIT = "92f56e5ee7227d490f6d85281c019e4c4e2dc393"
DATA_ROOT = Path("/data")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_absolute_path(path: str | os.PathLike[str]) -> Path:
    """Normalize traversal without resolving Modal Volume mount internals."""
    return Path(os.path.abspath(os.path.normpath(path)))


def validate_job_request(model: str, params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError("params must be a JSON object")
    model = model.strip()
    if not model:
        raise ValueError("model must not be empty")
    if "_api" in params:
        raise ValueError("params._api is reserved by the runtime")
    settings = dict(params)
    settings["model_type"] = model
    return settings


def validate_data_paths(value: Any, key: str = "params") -> None:
    """Reject absolute filesystem paths outside the shared data Volume."""
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            validate_data_paths(child_value, f"{key}.{child_key}")
    elif isinstance(value, list):
        for index, child_value in enumerate(value):
            validate_data_paths(child_value, f"{key}[{index}]")
    elif isinstance(value, str) and value.startswith("/"):
        raw_path = value.split("|", 1)[0]
        try:
            normalized_absolute_path(raw_path).relative_to(DATA_ROOT)
        except ValueError as exc:
            raise ValueError(f"{key} must reference a path under /data") from exc


def filter_models(
    models: list[dict[str, Any]],
    family: str | None = None,
    model_type: str | None = None,
) -> list[dict[str, Any]]:
    result = models
    if family:
        result = [item for item in result if item.get("family") == family]
    if model_type:
        result = [item for item in result if item.get("model_type") == model_type]
    return result


def resolve_generation_kind(
    catalog: dict[str, Any],
    model: str,
    requested_kind: str | None = None,
    params: dict[str, Any] | None = None,
) -> Literal["image", "video", "audio"]:
    """Resolve and validate the worker route from WanGP model metadata."""
    model = model.strip()
    metadata = next(
        (
            item
            for item in catalog.get("models", [])
            if item.get("model_type") == model
        ),
        None,
    )
    if metadata is None:
        raise ValueError(f"unknown model: {model}")

    raw_outputs = metadata.get("main_output", [])
    if isinstance(raw_outputs, str):
        raw_outputs = [raw_outputs]
    outputs = [
        output
        for output in raw_outputs
        if output in {"image", "video", "audio"}
    ]
    if not outputs:
        raise ValueError(f"model {model!r} does not declare a routable output type")

    if requested_kind is not None:
        if requested_kind not in {"image", "video", "audio"}:
            raise ValueError("kind must be one of: image, video, audio")
        if requested_kind not in outputs:
            choices = ", ".join(outputs)
            raise ValueError(
                f"model {model!r} supports {choices}, not {requested_kind}"
            )
        return requested_kind

    if len(outputs) == 1:
        return outputs[0]

    settings = dict(catalog.get("defaults", {}).get(model, {}))
    settings.update(params or {})
    if "image" in outputs and "video" in outputs:
        return "image" if settings.get("image_mode", 0) else "video"
    return outputs[0]
