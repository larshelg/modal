"""Pure OpenCV helpers for experimental chroma-assisted geometry arbitration."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any

import cv2
import numpy as np


DEFAULT_LOWER_HSV = (35, 50, 40)
DEFAULT_UPPER_HSV = (95, 255, 255)
CANONICAL_QUAD = np.asarray(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
)


def green_mask(
    frame_bgr: np.ndarray,
    lower_hsv: tuple[int, int, int] = DEFAULT_LOWER_HSV,
    upper_hsv: tuple[int, int, int] = DEFAULT_UPPER_HSV,
) -> np.ndarray:
    """Return the configured green-screen classification as an 8-bit mask."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame must be an HxWx3 BGR image")
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(
        hsv,
        np.asarray(lower_hsv, dtype=np.uint8),
        np.asarray(upper_hsv, dtype=np.uint8),
    )


def polygon_mask(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    """Rasterize a convex candidate quad, clipping naturally at frame bounds."""
    points = np.asarray(corners, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("corners must be a finite 4x2 array")
    output = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(output, [np.rint(points).astype(np.int32)], 255)
    return output


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
        angles = np.arctan2(
            points[:, 1] - centroid[1], points[:, 0] - centroid[0]
        )
        ordered = points[np.argsort(angles)]
        start = int(np.argmin(ordered.sum(axis=1)))
        ordered = np.roll(ordered, -start, axis=0)
        if ordered[1, 0] < ordered[-1, 0]:
            ordered = ordered[[0, 3, 2, 1]]
    if not cv2.isContourConvex(ordered.astype(np.float32)):
        raise ValueError("derived corners are not convex")
    return ordered


def largest_green_component(mask: np.ndarray, minimum_pixels: int = 1_000) -> np.ndarray:
    """Return the largest connected green component as a binary mask."""
    checked = np.asarray(mask, dtype=np.uint8)
    if checked.ndim != 2:
        raise ValueError("green mask must be two-dimensional")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(checked, 8)
    if count <= 1:
        raise ValueError("no green component was found")
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pixels = int(stats[index, cv2.CC_STAT_AREA])
    if pixels < minimum_pixels:
        raise ValueError("largest green component is too small")
    return np.where(labels == index, 255, 0).astype(np.uint8)


def mask_quadrilateral(mask: np.ndarray) -> tuple[np.ndarray, str, float]:
    """Fit an ordered convex quad to a connected chroma component."""
    checked = np.asarray(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(
        checked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("component has no contour")
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 0:
        raise ValueError("component contour is degenerate")

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
    contour_area = abs(float(cv2.contourArea(contour)))
    quad_area = abs(float(cv2.contourArea(ordered.astype(np.float32))))
    if quad_area <= 0:
        raise ValueError("derived quadrilateral has zero area")
    return ordered, method, min(1.0, contour_area / quad_area)


def project_reference_quad(
    reference_measurement: np.ndarray,
    current_measurement: np.ndarray,
    reference_outer_corners: np.ndarray,
) -> np.ndarray:
    """Transfer the frame-zero outer display quad through a chroma measurement."""
    reference = np.asarray(reference_measurement, dtype=np.float32)
    current = np.asarray(current_measurement, dtype=np.float32)
    outer = np.asarray(reference_outer_corners, dtype=np.float32)
    if reference.shape != (4, 2) or current.shape != (4, 2) or outer.shape != (4, 2):
        raise ValueError("all quadrilaterals must have shape 4x2")
    matrix = cv2.getPerspectiveTransform(reference, current)
    projected = cv2.perspectiveTransform(outer.reshape(1, 4, 2), matrix)[0]
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
        raise ValueError("chroma projection produced invalid geometry")
    return projected.astype(np.float64)


def transform_quad(source: np.ndarray, destination: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Project four points through the homography from source to destination."""
    source_checked = np.asarray(source, dtype=np.float32)
    destination_checked = np.asarray(destination, dtype=np.float32)
    points_checked = np.asarray(points, dtype=np.float32)
    if (
        source_checked.shape != (4, 2)
        or destination_checked.shape != (4, 2)
        or points_checked.shape != (4, 2)
    ):
        raise ValueError("all quadrilaterals must have shape 4x2")
    matrix = cv2.getPerspectiveTransform(source_checked, destination_checked)
    projected = cv2.perspectiveTransform(points_checked.reshape(1, 4, 2), matrix)[0]
    if not np.isfinite(projected).all():
        raise ValueError("homography projection is not finite")
    return projected.astype(np.float64)


def normalize_inner_measurement(
    outer_corners: np.ndarray, inner_measurement: np.ndarray
) -> np.ndarray:
    """Express a measured green quadrilateral on the outer screen's unit square."""
    normalized = transform_quad(
        np.asarray(outer_corners), CANONICAL_QUAD, np.asarray(inner_measurement)
    )
    if not cv2.isContourConvex(normalized.astype(np.float32)):
        raise ValueError("normalized inner measurement is not convex")
    return normalized


def calibrate_inner_quad(
    samples: list[tuple[np.ndarray, np.ndarray]],
    *,
    maximum_corner_residual: float = 0.10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Calibrate the bezel-to-green inset from trusted outer/inner quad pairs."""
    normalized: list[np.ndarray] = []
    for outer, inner in samples:
        candidate = normalize_inner_measurement(outer, inner)
        area = abs(float(cv2.contourArea(candidate.astype(np.float32))))
        if (
            area >= 0.35
            and area <= 1.10
            and candidate.min() >= -0.20
            and candidate.max() <= 1.20
        ):
            normalized.append(candidate)
    if len(normalized) < 3:
        raise ValueError("at least three plausible calibration samples are required")

    values = np.stack(normalized)
    preliminary = np.median(values, axis=0)
    residuals = np.max(np.linalg.norm(values - preliminary, axis=2), axis=1)
    median_residual = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median_residual)))
    threshold = min(
        maximum_corner_residual,
        max(0.025, median_residual + 3.0 * max(mad, 1e-6)),
    )
    keep = residuals <= threshold
    if int(keep.sum()) < 3:
        keep = np.argsort(residuals)[:3]
        retained = values[keep]
        retained_indices = [int(index) for index in keep]
    else:
        retained = values[keep]
        retained_indices = np.flatnonzero(keep).astype(int).tolist()
    model = np.median(retained, axis=0)
    if not cv2.isContourConvex(model.astype(np.float32)):
        raise ValueError("calibrated inner quadrilateral is not convex")
    return model, {
        "inputSamples": len(samples),
        "plausibleSamples": len(normalized),
        "retainedSamples": len(retained),
        "retainedIndices": retained_indices,
        "residualThreshold": round(float(threshold), 6),
        "maximumRetainedResidual": round(
            float(np.max(np.max(np.linalg.norm(retained - model, axis=2), axis=1))),
            6,
        ),
    }


def outer_from_inner_measurement(
    calibrated_inner: np.ndarray, measured_inner: np.ndarray
) -> np.ndarray:
    """Recover an outer display quad from its calibrated visible green inset."""
    outer = transform_quad(calibrated_inner, measured_inner, CANONICAL_QUAD)
    if not cv2.isContourConvex(outer.astype(np.float32)):
        raise ValueError("derived outer quadrilateral is not convex")
    return outer


def predicted_inner_quad(
    outer_corners: np.ndarray, calibrated_inner: np.ndarray
) -> np.ndarray:
    """Project the calibrated green inset through a candidate outer display quad."""
    return transform_quad(CANONICAL_QUAD, outer_corners, calibrated_inner)


def _sample_mask(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    x = np.rint(points[:, 0]).astype(int)
    y = np.rint(points[:, 1]).astype(int)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    values = np.zeros(len(points), dtype=bool)
    values[inside] = mask[y[inside], x[inside]] > 0
    return values


def edge_evidence(
    component: np.ndarray,
    inner_corners: np.ndarray,
    *,
    sample_count: int = 96,
    offset_pixels: float = 5.0,
) -> list[dict[str, Any]]:
    """Measure directional green transitions independently on all four edges."""
    checked = np.asarray(component, dtype=np.uint8)
    corners = np.asarray(inner_corners, dtype=np.float64)
    names = ("top", "right", "bottom", "left")
    output: list[dict[str, Any]] = []
    positions = np.linspace(0.08, 0.92, sample_count)
    for index, name in enumerate(names):
        start = corners[index]
        end = corners[(index + 1) % 4]
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length <= 1e-6:
            raise ValueError("candidate has a degenerate edge")
        inward = np.asarray([-edge[1], edge[0]], dtype=np.float64) / length
        centers = start + positions[:, None] * edge
        inside_votes = []
        outside_votes = []
        for multiplier in (0.7, 1.0, 1.3):
            distance = offset_pixels * multiplier
            inside_votes.append(_sample_mask(checked, centers + distance * inward))
            outside_votes.append(_sample_mask(checked, centers - distance * inward))
        inside = np.mean(np.stack(inside_votes), axis=0) >= 0.5
        outside = np.mean(np.stack(outside_votes), axis=0) >= 0.5
        positive = int(np.count_nonzero(inside & ~outside))
        negative = int(np.count_nonzero(~inside & outside))
        ambiguous = sample_count - positive - negative
        support = (positive + negative) / sample_count
        score = (positive - negative) / sample_count
        output.append(
            {
                "edge": name,
                "positive": positive,
                "negative": negative,
                "ambiguous": ambiguous,
                "support": round(float(support), 6),
                "score": round(float(score), 6),
                "supported": support >= 0.08 and positive >= 4,
            }
        )
    return output


def score_calibrated_candidate(
    component: np.ndarray,
    outer_corners: np.ndarray,
    calibrated_inner: np.ndarray,
    *,
    minimum_supported_edges: int = 2,
    minimum_component_fill: float = 0.45,
) -> dict[str, Any]:
    """Score a candidate using calibrated inset overlap and per-edge evidence."""
    checked = np.asarray(component, dtype=np.uint8)
    height, width = checked.shape
    inner = predicted_inner_quad(outer_corners, calibrated_inner)
    predicted = polygon_mask(inner, width, height)
    intersection = int(cv2.countNonZero(cv2.bitwise_and(checked, predicted)))
    measured_pixels = int(cv2.countNonZero(checked))
    predicted_pixels = int(cv2.countNonZero(predicted))
    precision = intersection / max(1, predicted_pixels)
    recall = intersection / max(1, measured_pixels)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    union = measured_pixels + predicted_pixels - intersection
    iou = intersection / max(1, union)
    edges = edge_evidence(checked, inner)
    supported = [edge for edge in edges if edge["supported"]]
    supported_count = len(supported)
    edge_score = (
        float(np.mean([edge["score"] for edge in supported]))
        if supported
        else 0.0
    )
    score = 0.45 * f1 + 0.25 * iou + 0.30 * max(0.0, edge_score)
    sufficient = (
        supported_count >= minimum_supported_edges
        and precision >= minimum_component_fill
        and measured_pixels >= 1_000
    )
    return {
        "score": round(float(score), 6) if sufficient else None,
        "evidenceSufficient": sufficient,
        "reason": None if sufficient else "destructive-occlusion-or-insufficient-edges",
        "predictedInnerCorners": [
            [round(float(value), 3) for value in point] for point in inner
        ],
        "measuredPixels": measured_pixels,
        "predictedPixels": predicted_pixels,
        "intersectionPixels": intersection,
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "intersectionOverUnion": round(float(iou), 6),
        "supportedEdgeCount": supported_count,
        "edgeScore": round(float(edge_score), 6),
        "edges": edges,
    }


def coherent_transform_gate(
    previous: np.ndarray,
    candidate: np.ndarray,
    *,
    maximum_residual: float = 24.0,
    minimum_scale: float = 0.80,
    maximum_scale: float = 1.25,
    maximum_rotation_degrees: float = 12.0,
    maximum_translation: float = 140.0,
) -> dict[str, Any]:
    """Require a candidate transition to resemble one coupled similarity transform."""
    source = np.asarray(previous, dtype=np.float32)
    destination = np.asarray(candidate, dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(source, destination, method=cv2.LMEDS)
    if matrix is None or not np.isfinite(matrix).all():
        return {"accepted": False, "reason": "similarity-fit-failed"}
    projected = cv2.transform(source.reshape(1, 4, 2), matrix)[0]
    residual = float(np.max(np.linalg.norm(projected - destination, axis=1)))
    a = float(matrix[0, 0])
    b = float(matrix[1, 0])
    scale = float(np.hypot(a, b))
    rotation = float(np.degrees(np.arctan2(b, a)))
    translation = float(np.hypot(matrix[0, 2], matrix[1, 2]))
    reasons: list[str] = []
    if residual > maximum_residual:
        reasons.append("non-coherent-corners")
    if not minimum_scale <= scale <= maximum_scale:
        reasons.append("implausible-scale")
    if abs(rotation) > maximum_rotation_degrees:
        reasons.append("implausible-rotation")
    if translation > maximum_translation:
        reasons.append("implausible-translation")
    return {
        "accepted": not reasons,
        "reason": ",".join(reasons) if reasons else None,
        "maximumResidual": round(residual, 6),
        "scale": round(scale, 6),
        "rotationDegrees": round(rotation, 6),
        "translation": round(translation, 6),
    }


def temporal_prediction_gate(
    previous_previous: np.ndarray,
    previous: np.ndarray,
    candidate: np.ndarray,
    *,
    minimum_tolerance: float = 28.0,
    displacement_multiplier: float = 4.0,
) -> dict[str, Any]:
    """Reject a candidate that departs sharply from constant-velocity motion."""
    before = np.asarray(previous_previous, dtype=np.float64)
    current = np.asarray(previous, dtype=np.float64)
    proposed = np.asarray(candidate, dtype=np.float64)
    prediction = current + (current - before)
    error = float(np.max(np.linalg.norm(proposed - prediction, axis=1)))
    prior_displacement = float(np.max(np.linalg.norm(current - before, axis=1)))
    tolerance = max(minimum_tolerance, prior_displacement * displacement_multiplier)
    return {
        "accepted": error <= tolerance,
        "reason": None if error <= tolerance else "temporal-prediction-error",
        "maximumPredictionError": round(error, 6),
        "tolerance": round(tolerance, 6),
    }


def stabilize_arbitration_sequence(
    baseline: np.ndarray,
    selected: np.ndarray,
    sources: list[str],
    abstained: list[bool],
    *,
    assisted_source: str = "chromaProjection",
    transition_frames: int = 6,
) -> tuple[np.ndarray, list[str]]:
    """Make offline arbitration handoffs continuous without trusting abstained frames.

    A bounded run of abstentions is filled by interpolating the correction from
    the baseline at the trusted assisted anchors on either side.  An initial
    baseline-to-assisted switch is eased over a short interval.  The function
    never consumes a candidate from an abstained frame.
    """
    baseline_checked = np.asarray(baseline, dtype=np.float64)
    selected_checked = np.asarray(selected, dtype=np.float64)
    if (
        baseline_checked.ndim != 3
        or baseline_checked.shape[1:] != (4, 2)
        or selected_checked.shape != baseline_checked.shape
        or len(sources) != len(baseline_checked)
        or len(abstained) != len(baseline_checked)
    ):
        raise ValueError("arbitration sequence inputs have incompatible shapes")
    if transition_frames < 1:
        raise ValueError("transition_frames must be positive")

    output = selected_checked.copy()
    labels = list(sources)
    frame_count = len(labels)

    index = 0
    while index < frame_count:
        if not abstained[index]:
            index += 1
            continue
        start = index
        while index < frame_count and abstained[index]:
            index += 1
        end = index
        bounded = (
            start > 0
            and end < frame_count
            and sources[start - 1] == assisted_source
            and sources[end] == assisted_source
        )
        if not bounded:
            continue
        left_correction = output[start - 1] - baseline_checked[start - 1]
        right_correction = output[end] - baseline_checked[end]
        span = end - (start - 1)
        for frame_index in range(start, end):
            weight = (frame_index - (start - 1)) / span
            correction = (
                (1.0 - weight) * left_correction + weight * right_correction
            )
            output[frame_index] = baseline_checked[frame_index] + correction
            labels[frame_index] = "occlusionBridge"

    for start in range(1, frame_count):
        if (
            sources[start] != assisted_source
            or sources[start - 1] == assisted_source
            or labels[start - 1] == "occlusionBridge"
        ):
            continue
        for offset in range(transition_frames):
            frame_index = start + offset
            if frame_index >= frame_count or sources[frame_index] != assisted_source:
                break
            progress = (offset + 1) / transition_frames
            weight = progress * progress * (3.0 - 2.0 * progress)
            output[frame_index] = baseline_checked[frame_index] + weight * (
                selected_checked[frame_index] - baseline_checked[frame_index]
            )
            labels[frame_index] = "transitionBlend"

    return output, labels


def _smoothstep(value: float) -> float:
    """Return a zero-slope blend weight for a visually quiet hand-off."""
    checked = max(0.0, min(1.0, float(value)))
    return checked * checked * (3.0 - 2.0 * checked)


def _terminal_recovery_start(coordinate_frames: list[dict[str, Any]]) -> int | None:
    """Locate a non-direct run that reaches the final frame, if one exists."""
    start = len(coordinate_frames)
    while (
        start > 0
        and coordinate_frames[start - 1].get("resolutionSource") != "direct"
    ):
        start -= 1
    return start if start < len(coordinate_frames) else None


def latch_terminal_chroma_tail(
    baseline: np.ndarray,
    chroma_projection: np.ndarray,
    supported: list[bool],
    acquisition: list[bool],
    coordinate_frames: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, Any]]:
    """Apply the conservative, offline terminal-tail policy.

    This is intentionally separate from chroma measurement so the decision is
    deterministic and easy to unit test. The short pre-roll is retrospective:
    once N consecutive tail frames confirm chroma, earlier *supported* frames
    are blended to hide the hand-off. No rejected or raw tracker candidate is
    ever imported by this policy.
    """
    baseline_checked = np.asarray(baseline, dtype=np.float64)
    chroma_checked = np.asarray(chroma_projection, dtype=np.float64)
    frame_count = len(coordinate_frames)
    if (
        baseline_checked.shape != (frame_count, 4, 2)
        or chroma_checked.shape != baseline_checked.shape
        or len(supported) != frame_count
        or len(acquisition) != frame_count
    ):
        raise ValueError("terminal chroma policy inputs have incompatible shapes")

    minimum_tail = int(settings["minimumTailFrames"])
    acquisition_frames = int(settings["acquisitionFrames"])
    transition_frames = int(settings["transitionFrames"])
    applied = baseline_checked.copy()
    weights = np.zeros(frame_count, dtype=np.float64)
    sources = ["baseline"] * frame_count
    tail_start = _terminal_recovery_start(coordinate_frames)
    evidence_start: int | None = None
    confirmation_frame: int | None = None
    blend_start: int | None = None
    reason = "no-terminal-recovery-tail"

    if tail_start is not None and frame_count - tail_start >= minimum_tail:
        run = 0
        for index in range(tail_start, frame_count):
            run = run + 1 if acquisition[index] else 0
            if run >= acquisition_frames:
                evidence_start = index - acquisition_frames + 1
                confirmation_frame = index
                break
        if confirmation_frame is None:
            reason = "no-sustained-acquisition-evidence"
        elif not all(supported[confirmation_frame:]):
            # A presenter can destroy multiple chroma edges. In that case we
            # abstain instead of bridging across an unobserved terminal tail.
            reason = "chroma-support-not-continuous-through-tail"
        else:
            blend_start = max(0, confirmation_frame - transition_frames + 1)
            if not all(supported[blend_start : confirmation_frame + 1]):
                blend_start = evidence_start
            for index in range(blend_start, frame_count):
                if index <= confirmation_frame:
                    progress = (index - blend_start + 1) / (
                        confirmation_frame - blend_start + 1
                    )
                    weight = _smoothstep(progress)
                else:
                    weight = 1.0
                applied[index] = (
                    baseline_checked[index] * (1.0 - weight)
                    + chroma_checked[index] * weight
                )
                weights[index] = weight
                sources[index] = (
                    "chromaTail" if weight >= 1.0 - 1e-9 else "tailBlend"
                )
            reason = "latched-terminal-chroma-tail"
    elif tail_start is not None:
        reason = "terminal-recovery-tail-too-short"

    if not all(cv2.isContourConvex(quad.astype(np.float32)) for quad in applied):
        raise ValueError("terminal chroma recovery produced invalid geometry")
    summary = {
        "mode": "latched-terminal-chroma-recovery",
        "applied": reason == "latched-terminal-chroma-tail",
        "reason": reason,
        "settings": {
            **settings,
            "requiredSupportedEdges": 4,
            "rawCandidates": "ignored",
            "postLatchCandidate": "calibratedChromaProjection-only",
        },
        "terminalRecoveryStart": tail_start,
        "evidenceStart": evidence_start,
        "confirmationFrame": confirmation_frame,
        "blendStart": blend_start,
        "sourceCounts": dict(sorted(Counter(sources).items())),
    }
    return applied, sources, weights, summary


def _candidate_gate(
    candidate: np.ndarray,
    history: list[tuple[int, np.ndarray]],
    frame_index: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Require one coherent chroma trajectory, including after evidence gaps."""
    checked = np.asarray(candidate, dtype=np.float64)
    valid = (
        checked.shape == (4, 2)
        and np.isfinite(checked).all()
        and cv2.isContourConvex(checked.astype(np.float32))
        and abs(float(cv2.contourArea(checked.astype(np.float32)))) > 1.0
    )
    if not valid:
        return {"accepted": False, "reason": "invalid-geometry"}
    if not history:
        return {"accepted": True, "reason": None}
    previous_frame, previous = history[-1]
    gap = frame_index - previous_frame
    if gap > 1:
        reacquired = bool(
            metrics.get("evidenceSufficient")
            and metrics.get("supportedEdgeCount") == 4
            and float(metrics.get("precision", 0.0)) >= 0.80
            and float(metrics.get("score") or 0.0) >= 0.75
        )
        return {
            "accepted": reacquired,
            "reason": None if reacquired else "reacquisition-evidence-insufficient",
            "reacquiredAfterGap": gap if reacquired else None,
        }
    coherent = coherent_transform_gate(previous, checked)
    temporal = (
        temporal_prediction_gate(history[-2][1], previous, checked)
        if len(history) >= 2 and history[-2][0] == previous_frame - 1
        else None
    )
    accepted = bool(coherent["accepted"]) and (
        temporal is None or bool(temporal["accepted"])
    )
    reasons = [
        item["reason"]
        for item in (coherent, temporal)
        if item is not None and item.get("reason")
    ]
    return {
        "accepted": accepted,
        "reason": ",".join(reasons) if reasons else None,
        "coherentTransform": coherent,
        "temporalPrediction": temporal,
    }


def _replace_motion_metrics(metrics: dict[str, Any], stable: np.ndarray) -> None:
    """Refresh only metric series affected by replacement corner geometry."""
    displacements = [0.0]
    areas = [abs(float(cv2.contourArea(stable[0].astype(np.float32))))]
    area_changes = [0.0]
    centroid_moves = [0.0]
    for index in range(1, len(stable)):
        displacements.append(
            float(np.max(np.linalg.norm(stable[index] - stable[index - 1], axis=1)))
        )
        areas.append(abs(float(cv2.contourArea(stable[index].astype(np.float32)))))
        area_changes.append(abs(areas[-1] - areas[-2]) / max(areas[-2], 1e-9))
        centroid_moves.append(
            float(
                np.linalg.norm(
                    np.mean(stable[index], axis=0)
                    - np.mean(stable[index - 1], axis=0)
                )
            )
        )
    replacements = {
        "cornerDisplacement": [round(value, 4) for value in displacements],
        "maximumCornerDisplacement": round(max(displacements), 4),
        "maximumCornerDisplacementFrame": int(np.argmax(displacements)),
        "homographyJump": [round(value, 4) for value in displacements],
        "maximumHomographyJump": round(max(displacements), 4),
        "maximumHomographyJumpFrame": int(np.argmax(displacements)),
        "quadArea": [round(value, 4) for value in areas],
        "quadAreaChange": [round(value, 6) for value in area_changes],
        "maximumQuadAreaChange": round(max(area_changes), 6),
        "maximumQuadAreaChangeFrame": int(np.argmax(area_changes)),
        "quadCentroidMovement": [round(value, 4) for value in centroid_moves],
        "maximumCentroidMovement": round(max(centroid_moves), 4),
        "maximumCentroidMovementFrame": int(np.argmax(centroid_moves)),
    }
    metrics.update(replacements)


def apply_terminal_chroma_tail_recovery(
    frames_rgb: np.ndarray,
    coordinates: dict[str, Any],
    metrics: dict[str, Any],
    stable_analysis: np.ndarray,
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    """Measure and optionally replace a failing terminal baseline trajectory.

    The bezel-to-green inset is calibrated only on trustworthy direct frames.
    Every later proposal must pass per-edge scoring plus coherent-transform and
    temporal gates. A destructive occlusion yields insufficient evidence and
    therefore an explicit abstention, leaving baseline geometry untouched.
    """
    if not settings.get("enabled"):
        return coordinates, metrics, stable_analysis

    output_coordinates = copy.deepcopy(coordinates)
    output_metrics = copy.deepcopy(metrics)
    coordinate_frames = output_coordinates["frames"]
    frames_checked = np.asarray(frames_rgb)
    stable_checked = np.asarray(stable_analysis, dtype=np.float64)
    if (
        frames_checked.ndim != 4
        or frames_checked.shape[0] != len(coordinate_frames)
        or stable_checked.shape != (len(coordinate_frames), 4, 2)
    ):
        raise ValueError("video, coordinates, and stable geometry frame counts differ")

    source = output_coordinates["source"]
    analysis = output_coordinates["analysis"]
    analysis_to_source = np.asarray(
        [source["width"] / analysis["width"], source["height"] / analysis["height"]],
        dtype=np.float64,
    )
    source_to_analysis = 1.0 / analysis_to_source
    baseline = stable_checked * analysis_to_source

    measurements: list[tuple[np.ndarray, str, float] | None] = []
    components: list[np.ndarray | None] = []
    measurement_errors: list[str | None] = []
    for frame in frames_checked:
        try:
            # imageio supplies RGB; the chroma helper intentionally accepts BGR
            # to match OpenCV callers used by the standalone experiments.
            bgr = np.ascontiguousarray(frame[:, :, :3][:, :, ::-1])
            component = largest_green_component(green_mask(bgr))
            measurement = mask_quadrilateral(component)
            components.append(component)
            measurements.append(measurement)
            measurement_errors.append(None)
        except ValueError as error:
            components.append(None)
            measurements.append(None)
            measurement_errors.append(str(error))

    calibration_samples: list[tuple[np.ndarray, np.ndarray]] = []
    calibration_frames: list[int] = []
    diagonal = float(np.hypot(source["width"], source["height"]))
    for index, detail in enumerate(coordinate_frames):
        measured = measurements[index]
        if (
            measured is None
            or detail.get("state") != "good"
            or detail.get("resolutionSource") != "direct"
        ):
            continue
        measurement = measured[0]
        if float(np.max(np.linalg.norm(measurement - baseline[index], axis=1))) / diagonal <= 0.09:
            calibration_samples.append((baseline[index], measurement))
            calibration_frames.append(index)
    try:
        calibrated_inner, calibration = calibrate_inner_quad(calibration_samples)
    except ValueError as error:
        summary = {
            "mode": "latched-terminal-chroma-recovery",
            "applied": False,
            "reason": "calibration-unavailable",
            "detail": str(error),
            "settings": {**settings, "requiredSupportedEdges": 4},
            "calibrationSampleFrames": calibration_frames,
        }
        output_coordinates["summary"]["chromaTailRecovery"] = summary
        output_metrics["chromaTailApplied"] = False
        output_metrics["chromaTailReason"] = summary["reason"]
        return output_coordinates, output_metrics, stable_checked

    calibration["sampleFrames"] = calibration_frames
    calibration["retainedFrames"] = [
        calibration_frames[index] for index in calibration["retainedIndices"]
    ]
    calibration["model"] = [
        [round(float(value), 6) for value in point] for point in calibrated_inner
    ]

    chroma_projection = baseline.copy()
    supported = [False] * len(coordinate_frames)
    acquisition = [False] * len(coordinate_frames)
    history: list[tuple[int, np.ndarray]] = []
    evidence: list[dict[str, Any]] = []
    for index, detail in enumerate(coordinate_frames):
        measured = measurements[index]
        component = components[index]
        if measured is None or component is None:
            evidence.append(
                {
                    "available": False,
                    "supported": False,
                    "acquisition": False,
                    "reason": measurement_errors[index],
                }
            )
            continue
        measurement, method, rectangularity = measured
        try:
            candidate = outer_from_inner_measurement(calibrated_inner, measurement)
            candidate_metrics = score_calibrated_candidate(
                component, candidate, calibrated_inner
            )
            baseline_metrics = score_calibrated_candidate(
                component, baseline[index], calibrated_inner
            )
            gate = _candidate_gate(candidate, history, index, candidate_metrics)
            if candidate_metrics["evidenceSufficient"] and gate["accepted"]:
                history.append((index, candidate))
            chroma_projection[index] = candidate
            candidate_score = candidate_metrics.get("score")
            baseline_score = baseline_metrics.get("score")
            supported[index] = bool(
                gate["accepted"]
                and candidate_metrics.get("evidenceSufficient")
                and candidate_metrics.get("supportedEdgeCount") == 4
                and float(candidate_score or 0.0) >= float(settings["minimumScore"])
                and float(candidate_metrics.get("precision") or 0.0)
                >= float(settings["minimumPrecision"])
            )
            acquisition[index] = bool(
                supported[index]
                and detail.get("resolutionSource") != "direct"
                and candidate_score is not None
                and (
                    baseline_score is None
                    or float(candidate_score)
                    >= float(baseline_score) + float(settings["scoreMargin"])
                )
            )
            evidence.append(
                {
                    "available": True,
                    "measurementMethod": method,
                    "rectangularity": round(float(rectangularity), 6),
                    "score": candidate_score,
                    "baselineScore": baseline_score,
                    "scoreAdvantage": (
                        round(float(candidate_score) - float(baseline_score), 6)
                        if candidate_score is not None and baseline_score is not None
                        else None
                    ),
                    "precision": candidate_metrics.get("precision"),
                    "supportedEdgeCount": candidate_metrics.get("supportedEdgeCount"),
                    "gateAccepted": bool(gate["accepted"]),
                    "supported": supported[index],
                    "acquisition": acquisition[index],
                    "reason": gate.get("reason") or candidate_metrics.get("reason"),
                }
            )
        except ValueError as error:
            evidence.append(
                {
                    "available": False,
                    "supported": False,
                    "acquisition": False,
                    "reason": str(error),
                }
            )

    applied_source, sources, weights, summary = latch_terminal_chroma_tail(
        baseline,
        chroma_projection,
        supported,
        acquisition,
        coordinate_frames,
        settings,
    )
    summary["calibration"] = calibration
    output_coordinates["summary"]["chromaTailRecovery"] = summary
    output_coordinates["settings"]["chromaTailRecovery"] = settings

    applied_analysis = applied_source * source_to_analysis
    if summary["applied"]:
        from tracking_geometry import homography_between

        canonical = np.asarray(
            output_coordinates["reference"]["canonicalCorners"], dtype=np.float64
        )
        for index, frame in enumerate(coordinate_frames):
            frame["baselineCorners"] = frame["corners"]
            frame["corners"] = [
                [round(float(value), 3) for value in point]
                for point in applied_analysis[index]
            ]
            frame["homography"] = [
                [round(float(value), 6) for value in row]
                for row in homography_between(canonical, applied_analysis[index])
            ]
            frame["chromaTailSource"] = sources[index]
            frame["chromaTailWeight"] = round(float(weights[index]), 6)
        _replace_motion_metrics(output_metrics, applied_analysis)
    for frame, frame_evidence in zip(coordinate_frames, evidence, strict=True):
        frame["chromaTailEvidence"] = frame_evidence

    output_metrics.update(
        {
            "chromaTailApplied": bool(summary["applied"]),
            "chromaTailReason": summary["reason"],
            "chromaTailRecoveryStart": summary["terminalRecoveryStart"],
            "chromaTailConfirmationFrame": summary["confirmationFrame"],
            "chromaTailBlendStart": summary["blendStart"],
            "chromaTailSourceCounts": summary["sourceCounts"],
        }
    )
    return output_coordinates, output_metrics, applied_analysis


def score_candidate(
    chroma: np.ndarray,
    corners: np.ndarray,
    *,
    band_pixels: int = 10,
    search_pixels: int = 32,
    minimum_visible_pixels: int = 500,
    minimum_green_pixels: int = 250,
) -> dict[str, Any]:
    """Score chroma support for a candidate display quad.

    Coverage alone favors undersized polygons. The score therefore combines
    interior coverage, capture of nearby green evidence, and the contrast
    between narrow bands immediately inside and outside the candidate edge.
    """
    checked = np.asarray(chroma, dtype=np.uint8)
    if checked.ndim != 2:
        raise ValueError("chroma must be a two-dimensional mask")
    height, width = checked.shape
    candidate = polygon_mask(np.asarray(corners), width, height)
    visible_pixels = int(cv2.countNonZero(candidate))
    if visible_pixels < minimum_visible_pixels:
        return {
            "score": None,
            "evidenceSufficient": False,
            "reason": "insufficient-visible-area",
            "visiblePixels": visible_pixels,
        }

    band_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (band_pixels * 2 + 1, band_pixels * 2 + 1)
    )
    search_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (search_pixels * 2 + 1, search_pixels * 2 + 1)
    )
    eroded = cv2.erode(candidate, band_kernel, iterations=1)
    dilated = cv2.dilate(candidate, band_kernel, iterations=1)
    search = cv2.dilate(candidate, search_kernel, iterations=1)
    inner_band = cv2.subtract(candidate, eroded)
    outer_band = cv2.subtract(dilated, candidate)

    green_inside = int(cv2.countNonZero(cv2.bitwise_and(checked, candidate)))
    total_green = int(cv2.countNonZero(checked))
    nearby_green = int(cv2.countNonZero(cv2.bitwise_and(checked, search)))
    inner_pixels = max(1, int(cv2.countNonZero(inner_band)))
    outer_pixels = max(1, int(cv2.countNonZero(outer_band)))
    inner_green = int(cv2.countNonZero(cv2.bitwise_and(checked, inner_band)))
    outer_green = int(cv2.countNonZero(cv2.bitwise_and(checked, outer_band)))

    coverage = green_inside / visible_pixels
    recall = green_inside / max(1, total_green)
    f1 = 2.0 * coverage * recall / max(1e-9, coverage + recall)
    intersection_over_union = green_inside / max(
        1, visible_pixels + total_green - green_inside
    )
    capture = green_inside / max(1, nearby_green)
    inner_rate = inner_green / inner_pixels
    outer_rate = outer_green / outer_pixels
    edge_contrast = inner_rate - outer_rate
    normalized_contrast = (edge_contrast + 1.0) / 2.0
    score = (
        0.60 * f1
        + 0.25 * intersection_over_union
        + 0.15 * normalized_contrast
    )
    sufficient = green_inside >= minimum_green_pixels
    return {
        "score": round(float(score), 6) if sufficient else None,
        "evidenceSufficient": sufficient,
        "reason": None if sufficient else "insufficient-green-evidence",
        "visiblePixels": visible_pixels,
        "greenInside": green_inside,
        "totalGreen": total_green,
        "nearbyGreen": nearby_green,
        "coverage": round(float(coverage), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "intersectionOverUnion": round(float(intersection_over_union), 6),
        "capture": round(float(capture), 6),
        "innerEdgeGreen": round(float(inner_rate), 6),
        "outerEdgeGreen": round(float(outer_rate), 6),
        "edgeContrast": round(float(edge_contrast), 6),
    }
