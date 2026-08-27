"""S3 request validation and geometry helpers for frame-zero display detection."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from sam_stage import RUN_ID_PATTERN, SHA256_PATTERN, validate_key


DISPLAY_STAGE = "display-detection"
DEFAULT_TEXT_PROMPTS = (
    "display screen",
    "television screen",
    "monitor screen",
    "screen",
)
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def validate_display_request(
    request: Any, configured_bucket: str
) -> dict[str, Any]:
    request = _object(request, "request")
    allowed = {
        "schemaVersion",
        "runId",
        "stage",
        "inputHash",
        "inputs",
        "parameters",
        "output",
    }
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise ValueError(f"unsupported request fields: {', '.join(unknown)}")
    if request.get("schemaVersion") != 1:
        raise ValueError("schemaVersion must be 1")
    run_id = request.get("runId")
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("runId is invalid")
    if request.get("stage") != DISPLAY_STAGE:
        raise ValueError(f"stage must be {DISPLAY_STAGE}")
    input_hash = request.get("inputHash")
    if not isinstance(input_hash, str) or not SHA256_PATTERN.fullmatch(input_hash):
        raise ValueError("inputHash must be a lowercase SHA-256")

    experiment_prefix = f"studio-experiments/{run_id}/"
    inputs = _object(request.get("inputs"), "inputs")
    if set(inputs) != {"image"}:
        raise ValueError("inputs must contain only image")
    image = _object(inputs["image"], "inputs.image")
    required_ref = {
        "storage",
        "bucket",
        "key",
        "sha256",
        "sizeBytes",
        "contentType",
    }
    if set(image) != required_ref:
        raise ValueError("image ArtifactRef fields are invalid")
    if image.get("storage") != "s3":
        raise ValueError("image.storage must be s3")
    if image.get("bucket") != configured_bucket:
        raise ValueError("image.bucket does not match configured S3_BUCKET")
    validate_key(image.get("key"), experiment_prefix, "image.key")
    if not isinstance(image.get("sha256"), str) or not SHA256_PATTERN.fullmatch(
        image["sha256"]
    ):
        raise ValueError("image.sha256 must be a lowercase SHA-256")
    size = image.get("sizeBytes")
    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= MAX_IMAGE_BYTES:
        raise ValueError(
            f"image.sizeBytes must be between 1 and {MAX_IMAGE_BYTES}"
        )
    if image.get("contentType") not in ALLOWED_IMAGE_TYPES:
        raise ValueError(
            "image.contentType must be image/jpeg, image/png, or image/webp"
        )

    parameters = _object(request.get("parameters", {}), "parameters")
    allowed_parameters = {
        "textPrompts",
        "scoreThreshold",
        "minAreaRatio",
        "maxAreaRatio",
    }
    unknown_parameters = sorted(set(parameters) - allowed_parameters)
    if unknown_parameters:
        raise ValueError(
            f"unsupported parameters: {', '.join(unknown_parameters)}"
        )
    raw_prompts = parameters.get("textPrompts", list(DEFAULT_TEXT_PROMPTS))
    if not isinstance(raw_prompts, list) or not 1 <= len(raw_prompts) <= 8:
        raise ValueError("textPrompts must contain between 1 and 8 prompts")
    prompts: list[str] = []
    for index, value in enumerate(raw_prompts):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"textPrompts[{index}] must be a non-empty string")
        prompt = value.strip()
        if len(prompt) > 100:
            raise ValueError(f"textPrompts[{index}] must not exceed 100 characters")
        if prompt not in prompts:
            prompts.append(prompt)

    score_threshold = _finite_number(
        parameters.get("scoreThreshold", 0.35), "scoreThreshold"
    )
    if not 0 < score_threshold < 1:
        raise ValueError("scoreThreshold must be between 0 and 1")
    min_area_ratio = _finite_number(
        parameters.get("minAreaRatio", 0.01), "minAreaRatio"
    )
    max_area_ratio = _finite_number(
        parameters.get("maxAreaRatio", 0.95), "maxAreaRatio"
    )
    if not 0 < min_area_ratio < max_area_ratio < 1:
        raise ValueError(
            "minAreaRatio and maxAreaRatio must satisfy 0 < min < max < 1"
        )
    request["parameters"] = {
        "textPrompts": prompts,
        "scoreThreshold": score_threshold,
        "minAreaRatio": min_area_ratio,
        "maxAreaRatio": max_area_ratio,
    }

    output = _object(request.get("output"), "output")
    if set(output) != {"bucket", "prefix"}:
        raise ValueError("output fields are invalid")
    if output.get("bucket") != configured_bucket:
        raise ValueError("output.bucket does not match configured S3_BUCKET")
    expected_prefix = (
        f"{experiment_prefix}{DISPLAY_STAGE}/{input_hash}/"
    )
    if output.get("prefix") != expected_prefix:
        raise ValueError(
            "output.prefix does not match runId, stage, and inputHash"
        )
    return request


def decode_image(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as source:
            source.load()
            if source.width <= 0 or source.height <= 0:
                raise ValueError("image dimensions must be positive")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"image exceeds the {MAX_IMAGE_PIXELS}-pixel limit"
                )
            return np.asarray(source.convert("RGB"), dtype=np.uint8)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"input is not a decodable image: {error}") from None


def _order_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(4, 2)
    sums = points.sum(axis=1)
    differences = points[:, 1] - points[:, 0]
    indices = [
        int(np.argmin(sums)),
        int(np.argmin(differences)),
        int(np.argmax(sums)),
        int(np.argmax(differences)),
    ]
    if len(set(indices)) == 4:
        ordered = points[indices]
    else:
        centroid = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        ordered = points[np.argsort(angles)]
        start = int(np.argmin(ordered.sum(axis=1)))
        ordered = np.roll(ordered, -start, axis=0)
        if ordered[1, 0] < ordered[-1, 0]:
            ordered = ordered[[0, 3, 2, 1]]
    if not cv2.isContourConvex(ordered.astype(np.float32)):
        raise ValueError("derived display corners are not convex")
    return ordered


def _mask_quadrilateral(mask: np.ndarray) -> tuple[np.ndarray, str, float]:
    checked = np.asarray(mask, dtype=np.uint8)
    if checked.ndim != 2 or not checked.any():
        raise ValueError("candidate mask must be a non-empty 2D array")
    contours, _ = cv2.findContours(
        checked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("candidate mask has no contour")
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 0:
        raise ValueError("candidate mask contour is degenerate")

    quadrilateral: np.ndarray | None = None
    method = "convex-approximation"
    for epsilon_ratio in np.linspace(0.005, 0.08, 31):
        approximation = cv2.approxPolyDP(
            hull, float(epsilon_ratio) * perimeter, True
        ).reshape(-1, 2)
        if len(approximation) == 4 and cv2.isContourConvex(
            approximation.astype(np.float32)
        ):
            quadrilateral = approximation.astype(np.float64)
            break
    if quadrilateral is None:
        quadrilateral = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float64)
        method = "minimum-area-rectangle"

    ordered = _order_corners(quadrilateral)
    quad_area = abs(float(cv2.contourArea(ordered.astype(np.float32))))
    contour_area = abs(float(cv2.contourArea(contour)))
    if quad_area <= 0:
        raise ValueError("derived display quadrilateral has zero area")
    rectangle = cv2.minAreaRect(hull)
    enclosing_area = float(rectangle[1][0] * rectangle[1][1])
    if enclosing_area <= 0:
        raise ValueError("candidate mask has a degenerate enclosing rectangle")
    rectangularity = min(1.0, contour_area / enclosing_area)
    return ordered, method, rectangularity


def select_display_candidate(
    prompt_outputs: list[tuple[str, Any, Any]],
    *,
    score_threshold: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for prompt, raw_masks, raw_scores in prompt_outputs:
        masks = np.asarray(raw_masks, dtype=bool)
        scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3:
            raise ValueError("SAM masks must have shape [objects, height, width]")
        if len(masks) != len(scores):
            raise ValueError("SAM masks and scores must have matching lengths")
        for object_index, (mask, raw_score) in enumerate(
            zip(masks, scores, strict=True)
        ):
            sam_score = float(raw_score)
            if not math.isfinite(sam_score) or sam_score < score_threshold:
                continue
            area_ratio = float(mask.mean())
            if not min_area_ratio <= area_ratio <= max_area_ratio:
                continue
            try:
                corners, method, rectangularity = _mask_quadrilateral(mask)
            except ValueError:
                continue
            confidence = 0.75 * sam_score + 0.25 * rectangularity
            candidates.append(
                {
                    "prompt": prompt,
                    "objectIndex": object_index,
                    "samScore": sam_score,
                    "rectangularity": rectangularity,
                    "confidence": confidence,
                    "areaRatio": area_ratio,
                    "cornerMethod": method,
                    "corners": corners,
                    "mask": mask,
                }
            )
    if not candidates:
        raise RuntimeError(
            "SAM 3.1 found no display-like mask above the configured thresholds"
        )
    candidates.sort(
        key=lambda item: (
            -item["confidence"],
            -item["samScore"],
            -item["rectangularity"],
            item["prompt"],
            item["objectIndex"],
        )
    )
    selected = candidates[0]
    selected["candidateCount"] = len(candidates)
    return selected


def render_display_overlay(
    image_rgb: np.ndarray, candidate: dict[str, Any], output: Path
) -> None:
    image = np.asarray(image_rgb, dtype=np.uint8)
    mask = np.asarray(candidate["mask"], dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2]:
        raise ValueError("overlay image and mask dimensions do not match")
    overlay = image.astype(np.float32)
    tint = np.asarray([255.0, 36.0, 190.0], dtype=np.float32)
    overlay[mask] = overlay[mask] * 0.58 + tint * 0.42
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    corners = np.rint(candidate["corners"]).astype(np.int32)
    cv2.polylines(overlay, [corners], True, (0, 255, 80), 4, cv2.LINE_AA)
    labels = ("TL", "TR", "BR", "BL")
    for label, point in zip(labels, corners, strict=True):
        x, y = int(point[0]), int(point[1])
        cv2.circle(overlay, (x, y), 7, (255, 220, 0), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            label,
            (x + 9, y - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    summary = (
        f"{candidate['prompt']}  confidence={candidate['confidence']:.3f}"
    )
    cv2.putText(
        overlay,
        summary,
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay, mode="RGB").save(
        output, format="PNG", optimize=True
    )


def public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key not in {"mask", "corners"}
    }
