"""Deterministic TAPNext++ homography, QA, and review rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PLANE_QUERY_COUNT = 32
DEFAULT_QUERY_LAYOUT = "perimeter-32"
HYBRID_QUERY_LAYOUT = "hybrid-24-edge-8-interior"
ADDITIVE_QUERY_LAYOUT = "perimeter-32-plus-interior-8"
QUERY_LAYOUT_PLANE_COUNTS = {
    DEFAULT_QUERY_LAYOUT: 32,
    HYBRID_QUERY_LAYOUT: 32,
    ADDITIVE_QUERY_LAYOUT: 40,
}
CORNER_NAMES = ("tl", "tr", "br", "bl")
MAX_TAIL_PREDICTION_FRAMES = 24
OVERVIEW_SHEET_FRAME_COUNT = 30
OVERVIEW_SHEET_COLUMNS = 6
OVERVIEW_TILE_SIZE = (256, 224)


@dataclass
class RawFrame:
    frame: int
    homography: np.ndarray | None
    corners: np.ndarray | None
    visible_indices: list[int]
    inlier_indices: list[int]
    rejected_indices: list[int]
    reprojection_error: float | None
    state: str


def polygon_area(corners: np.ndarray) -> float:
    x = corners[:, 0]
    y = corners[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2


def is_convex(corners: np.ndarray) -> bool:
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return False
    signs = []
    for index in range(4):
        first = corners[(index + 1) % 4] - corners[index]
        second = corners[(index + 2) % 4] - corners[(index + 1) % 4]
        signs.append(float(first[0] * second[1] - first[1] * second[0]))
    return all(value > 0 for value in signs) or all(value < 0 for value in signs)


def homography_between(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    if not is_convex(source) or not is_convex(destination):
        raise ValueError("homography corners must be convex and consistently ordered")
    result = cv2.getPerspectiveTransform(
        source.astype(np.float32), destination.astype(np.float32)
    ).astype(np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError("homography must be a finite 3x3 matrix")
    return result


def plane_query_count(query_layout: str) -> int:
    """Return the number of RANSAC plane queries before QA V2 corner probes."""
    try:
        return QUERY_LAYOUT_PLANE_COUNTS[query_layout]
    except KeyError:
        raise ValueError(f"unsupported query layout: {query_layout}") from None


def seed_queries(
    corners: list[list[float]], source_width: int, source_height: int,
    analysis_width: int, analysis_height: int, inset: float = 6.0,
    include_corner_probes: bool = False,
    query_layout: str = DEFAULT_QUERY_LAYOUT,
) -> tuple[list[list[float]], np.ndarray]:
    """Seed the selected plane-query layout and optional QA V2 probes.

    The historical perimeter layout remains the default. The hybrid layout is
    deliberately opt-in: it trades one quarter of the correlated edge points
    for an interior 4x2 lattice. The additive layout preserves all 32 original
    perimeter points and appends the same eight interior points.
    """
    source = np.asarray(corners, dtype=np.float64)
    analysis = source * np.array(
        [analysis_width / source_width, analysis_height / source_height],
        dtype=np.float64,
    )
    if not is_convex(analysis):
        raise ValueError("screen corners must be convex and consistently ordered")
    centroid = np.mean(analysis, axis=0)
    points: list[np.ndarray] = []
    expected_plane_points = plane_query_count(query_layout)
    points_per_edge = 6 if query_layout == HYBRID_QUERY_LAYOUT else 8
    for edge in range(4):
        start, end = analysis[edge], analysis[(edge + 1) % 4]
        for step in range(points_per_edge):
            point = start + (end - start) * (step / points_per_edge)
            inward = centroid - point
            distance = float(np.linalg.norm(inward))
            if distance <= inset:
                raise ValueError("screen quadrilateral is too small for query inset")
            points.append(point + inward * (inset / distance))
    if query_layout in {HYBRID_QUERY_LAYOUT, ADDITIVE_QUERY_LAYOUT}:
        # Project a normalized 4x2 lattice through the actual screen quad.
        # Projective placement keeps the interior distribution meaningful on
        # a perspective-skewed TV instead of treating its bounding box as flat.
        unit_corners = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        normalized_interior = np.asarray(
            [
                [u, v]
                for v in (1.0 / 3.0, 2.0 / 3.0)
                for u in (0.2, 0.4, 0.6, 0.8)
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(
            unit_corners, analysis.astype(np.float32)
        )
        interior = cv2.perspectiveTransform(
            normalized_interior.reshape(1, -1, 2), matrix
        )[0].astype(np.float64)
        points.extend(interior)
    if len(points) != expected_plane_points:
        raise RuntimeError("query layout produced the wrong plane-point count")
    queries = [[0, float(p[0]), float(p[1])] for p in points]
    if include_corner_probes:
        queries.extend([[0, float(p[0]), float(p[1])] for p in analysis])
    return queries, analysis


def estimate_raw_frames(
    reference_points: np.ndarray, tracks: np.ndarray, visibility: np.ndarray,
    reference_corners: np.ndarray, analysis_width: int, analysis_height: int,
    maximum_reference_area_ratio: float,
) -> list[RawFrame]:
    if tracks.ndim != 3 or tracks.shape[2] != 2:
        raise ValueError("tracks must have shape [frames, points, 2]")
    if visibility.shape != tracks.shape[:2] or not np.isfinite(tracks).all():
        raise ValueError("visibility and finite tracks must have expected dimensions")
    frames: list[RawFrame] = []
    displacements: list[float] = []
    previous_corners: np.ndarray | None = None
    previous_area: float | None = None
    reference_area = polygon_area(reference_corners)
    for frame_index, current in enumerate(tracks):
        valid = (
            visibility[frame_index].astype(bool)
            & np.isfinite(current).all(axis=1)
            & (current[:, 0] >= 0) & (current[:, 0] < analysis_width)
            & (current[:, 1] >= 0) & (current[:, 1] < analysis_height)
        )
        visible_indices = np.flatnonzero(valid).astype(int).tolist()
        matrix = corners = None
        inliers: list[int] = []
        error = None
        state = "insufficient_points"
        if len(visible_indices) >= 4:
            source = reference_points[valid].astype(np.float32)
            destination = current[valid].astype(np.float32)
            matrix, mask = cv2.findHomography(
                source, destination, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            if matrix is not None and mask is not None and np.isfinite(matrix).all():
                keep = mask.reshape(-1).astype(bool)
                inliers = [i for i, selected in zip(visible_indices, keep, strict=True) if selected]
                projected = cv2.perspectiveTransform(source.reshape(1, -1, 2), matrix)[0]
                errors = np.linalg.norm(projected - destination, axis=1)
                error = float(np.median(errors[keep])) if keep.any() else None
                corners = cv2.perspectiveTransform(
                    reference_corners.reshape(1, 4, 2).astype(np.float32), matrix
                )[0].astype(np.float64)
                state = "good"
                area = polygon_area(corners)
                if len(inliers) < 8 or len(inliers) / len(visible_indices) < 0.60:
                    state = "low_confidence"
                elif not is_convex(corners) or not (
                    reference_area * 0.10 <= area <= reference_area * maximum_reference_area_ratio
                ):
                    state = "invalid_geometry"
                elif previous_corners is not None and previous_area is not None:
                    displacement = float(np.max(np.linalg.norm(corners - previous_corners, axis=1)))
                    median = float(np.median(displacements[-5:])) if displacements else 0.0
                    if displacement > max(12.0, median * 4.0) or not 0.70 <= area / previous_area <= 1.30:
                        state = "temporal_reject"
                    else:
                        displacements.append(displacement)
                if state == "good":
                    previous_corners, previous_area = corners, area
        frames.append(RawFrame(
            frame_index, matrix, corners, visible_indices, inliers,
            sorted(set(visible_indices) - set(inliers)), error, state,
        ))
    return frames


def _interpolate(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not valid.any():
        raise ValueError("no valid homography frames were produced")
    indices = np.arange(len(values), dtype=np.float64)
    result = values.copy()
    for corner in range(4):
        for coordinate in range(2):
            result[:, corner, coordinate] = np.interp(
                indices, indices[valid], values[valid, corner, coordinate]
            )
    return result


def _smooth_filled(filled: np.ndarray) -> np.ndarray:
    median = filled.copy()
    for index in range(len(filled)):
        median[index] = np.median(filled[max(0, index - 2):min(len(filled), index + 3)], axis=0)
    median[0], median[-1] = filled[0], filled[-1]
    forward, backward = median.copy(), median.copy()
    for index in range(1, len(median)):
        forward[index] = 0.25 * median[index] + 0.75 * forward[index - 1]
    for index in range(len(median) - 2, -1, -1):
        backward[index] = 0.25 * median[index] + 0.75 * backward[index + 1]
    stable = (forward + backward) / 2
    stable[0], stable[-1] = median[0], median[-1]
    if any(not is_convex(corners) for corners in stable):
        raise ValueError("stabilization produced invalid geometry")
    return stable


def stabilize(raw_frames: list[RawFrame]) -> np.ndarray:
    values = np.full((len(raw_frames), 4, 2), np.nan, dtype=np.float64)
    valid = np.zeros(len(raw_frames), dtype=bool)
    for index, frame in enumerate(raw_frames):
        if frame.state == "good" and frame.corners is not None:
            values[index], valid[index] = frame.corners, True
    return _smooth_filled(_interpolate(values, valid))


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack(
        [points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)]
    )
    return (matrix.astype(np.float64) @ homogeneous.T).T


def _valid_quad(
    corners: np.ndarray,
    reference_area: float,
    maximum_reference_area_ratio: float,
) -> bool:
    if not is_convex(corners):
        return False
    area = polygon_area(corners)
    return reference_area * 0.10 <= area <= reference_area * maximum_reference_area_ratio


def _predict_quad(previous_previous: np.ndarray, previous: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.estimateAffinePartial2D(
        previous_previous.astype(np.float32),
        previous.astype(np.float32),
        method=cv2.LMEDS,
    )
    if matrix is not None and matrix.shape == (2, 3) and np.isfinite(matrix).all():
        predicted = _transform_points(previous, matrix)
        if is_convex(predicted):
            return predicted
    translation = np.median(previous - previous_previous, axis=0)
    return previous + translation


def _trusted_corner_probes(
    predicted: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    analysis_width: int,
    analysis_height: int,
    recent_motion: float,
    allow_reacquisition: bool = False,
) -> np.ndarray:
    inside = (
        visibility.astype(bool)
        & np.isfinite(tracks).all(axis=1)
        & (tracks[:, 0] >= 0)
        & (tracks[:, 0] < analysis_width)
        & (tracks[:, 1] >= 0)
        & (tracks[:, 1] < analysis_height)
    )
    innovation = np.linalg.norm(tracks - predicted, axis=1)
    if allow_reacquisition:
        return inside
    return inside & (innovation <= max(12.0, recent_motion * 4.0))


def _correct_prediction(
    predicted: np.ndarray,
    observed: np.ndarray,
    trusted: np.ndarray,
) -> np.ndarray | None:
    indices = np.flatnonzero(trusted)
    if len(indices) < 2:
        return None
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        predicted[indices].astype(np.float32),
        observed[indices].astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if matrix is None or not np.isfinite(matrix).all():
        return None
    if inlier_mask is not None and int(np.count_nonzero(inlier_mask)) < 2:
        return None
    scale = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    rotation = abs(math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))))
    if not 0.70 <= scale <= 1.30 or rotation > 20.0:
        return None
    return _transform_points(predicted, matrix)


def _bridge_hidden_gaps(
    resolved: np.ndarray,
    details: list[dict[str, Any]],
) -> None:
    index = 0
    count = len(resolved)
    hidden_sources = {"predicted", "held"}
    while index < count:
        if details[index]["resolutionSource"] not in hidden_sources:
            index += 1
            continue
        start = index
        while index < count and details[index]["resolutionSource"] in hidden_sources:
            index += 1
        end = index
        if start == 0 or end >= count:
            continue
        previous_previous = resolved[start - 2] if start >= 2 else resolved[start - 1]
        previous = resolved[start - 1]
        forward: list[np.ndarray] = []
        for _ in range(start, end + 1):
            predicted = _predict_quad(previous_previous, previous)
            forward.append(predicted)
            previous_previous, previous = previous, predicted
        correction, _ = cv2.estimateAffinePartial2D(
            forward[-1].astype(np.float32),
            resolved[end].astype(np.float32),
            method=cv2.LMEDS,
        )
        if correction is None or not np.isfinite(correction).all():
            continue
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        for offset, frame_index in enumerate(range(start, end)):
            weight = (offset + 1) / (end - start + 1)
            blended = identity * (1.0 - weight) + correction * weight
            candidate = _transform_points(forward[offset], blended)
            if is_convex(candidate):
                resolved[frame_index] = candidate
                details[frame_index]["resolutionSource"] = "bridged"
                details[frame_index]["predictionAge"] = min(
                    frame_index - (start - 1), end - frame_index
                )


def _offscreen_details(
    corners: np.ndarray,
    analysis_width: int,
    analysis_height: int,
) -> tuple[int, bool]:
    inside = (
        (corners[:, 0] >= 0)
        & (corners[:, 0] < analysis_width)
        & (corners[:, 1] >= 0)
        & (corners[:, 1] < analysis_height)
    )
    frame_polygon = np.asarray(
        [[0, 0], [analysis_width - 1, 0], [analysis_width - 1, analysis_height - 1], [0, analysis_height - 1]],
        dtype=np.float32,
    )
    intersection, _ = cv2.intersectConvexConvex(
        corners.astype(np.float32), frame_polygon
    )
    return int(4 - np.count_nonzero(inside)), float(intersection) <= 1e-6


def resolve_offscreen_v2(
    raw_frames: list[RawFrame],
    corner_tracks: np.ndarray,
    corner_visibility: np.ndarray,
    reference_corners: np.ndarray,
    analysis_width: int,
    analysis_height: int,
    maximum_reference_area_ratio: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if corner_tracks.shape != (len(raw_frames), 4, 2):
        raise ValueError("V2 corner tracks must have shape [frames, 4, 2]")
    if corner_visibility.shape != (len(raw_frames), 4):
        raise ValueError("V2 corner visibility must have shape [frames, 4]")

    reference_area = polygon_area(reference_corners)
    resolved = np.empty((len(raw_frames), 4, 2), dtype=np.float64)
    details: list[dict[str, Any]] = []
    last_direct = -1
    unsupported_run = 0

    for index, raw in enumerate(raw_frames):
        if raw.state == "good" and raw.corners is not None:
            current = raw.corners.copy()
            source = "direct"
            last_direct = index
            unsupported_run = 0
            probe_inside = (
                corner_visibility[index].astype(bool)
                & np.isfinite(corner_tracks[index]).all(axis=1)
                & (corner_tracks[index, :, 0] >= 0)
                & (corner_tracks[index, :, 0] < analysis_width)
                & (corner_tracks[index, :, 1] >= 0)
                & (corner_tracks[index, :, 1] < analysis_height)
            )
            corner_sources = ["tracked" if value else "predicted" for value in probe_inside]
        else:
            if index == 0:
                predicted = reference_corners.copy()
            elif index == 1:
                predicted = resolved[0].copy()
            else:
                predicted = _predict_quad(resolved[index - 2], resolved[index - 1])
            recent_motion = (
                float(np.max(np.linalg.norm(resolved[index - 1] - resolved[index - 2], axis=1)))
                if index >= 2
                else 0.0
            )
            trusted = _trusted_corner_probes(
                predicted,
                corner_tracks[index],
                corner_visibility[index],
                analysis_width,
                analysis_height,
                recent_motion,
                allow_reacquisition=unsupported_run > 0,
            )
            corrected = _correct_prediction(predicted, corner_tracks[index], trusted)
            if corrected is not None and _valid_quad(
                corrected, reference_area, maximum_reference_area_ratio
            ):
                current = corrected
                source = "partial-affine"
                unsupported_run = 0
                corner_sources = ["tracked" if value else "predicted" for value in trusted]
            else:
                unsupported_run += 1
                if unsupported_run > MAX_TAIL_PREDICTION_FRAMES and index > 0:
                    current = resolved[index - 1].copy()
                    source = "held"
                else:
                    current = predicted
                    source = "predicted"
                corner_sources = ["predicted"] * 4
            if not _valid_quad(current, reference_area, maximum_reference_area_ratio):
                if index == 0:
                    current = reference_corners.copy()
                else:
                    current = resolved[index - 1].copy()
                source = "held"
                corner_sources = ["predicted"] * 4

        resolved[index] = current
        details.append(
            {
                "resolutionSource": source,
                "predictionAge": 0 if source == "direct" else max(1, index - last_direct),
                "cornerSources": dict(zip(CORNER_NAMES, corner_sources, strict=True)),
            }
        )

    _bridge_hidden_gaps(resolved, details)
    stable = _smooth_filled(resolved)
    for index, current in enumerate(stable):
        count, fully = _offscreen_details(current, analysis_width, analysis_height)
        details[index]["offscreenCornerCount"] = count
        details[index]["fullyOffscreen"] = fully
    return stable, details


def _rounded(value: np.ndarray, digits: int) -> list[list[float]]:
    return [[round(float(item), digits) for item in row] for row in value]


def build_geometry(
    result: dict[str, Any], parameters: dict[str, Any], expected: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], np.ndarray]:
    tracks = np.asarray(result["tracks"], dtype=np.float64)
    visibility = np.asarray(result["visibility"], dtype=bool)
    analysis = parameters["analysis"]
    frame_count = int(expected["frames"])
    qa_version = int(parameters.get("qaVersion", 1))
    query_layout = parameters.get("queryLayout", DEFAULT_QUERY_LAYOUT)
    plane_count = plane_query_count(query_layout)
    queries, reference_corners = seed_queries(
        parameters["frameZeroCorners"], expected["width"], expected["height"],
        analysis["width"], analysis["height"],
        include_corner_probes=qa_version == 2,
        query_layout=query_layout,
    )
    if (
        tracks.shape != (frame_count, len(queries), 2)
        or visibility.shape != (frame_count, len(queries))
        or not np.isfinite(tracks).all()
    ):
        raise ValueError("tracker output dimensions do not match expectedMedia.frames")
    raw = estimate_raw_frames(
        np.asarray([query[1:] for query in queries[:plane_count]]),
        tracks[:, :plane_count], visibility[:, :plane_count],
        reference_corners, analysis["width"], analysis["height"],
        parameters["maxReferenceAreaRatio"],
    )
    resolution_details: list[dict[str, Any]] | None = None
    if qa_version == 2:
        stable, resolution_details = resolve_offscreen_v2(
            raw,
            tracks[:, plane_count:],
            visibility[:, plane_count:],
            reference_corners,
            analysis["width"],
            analysis["height"],
            parameters["maxReferenceAreaRatio"],
        )
    else:
        stable = stabilize(raw)
    surface = parameters["surface"]
    canonical = np.asarray([
        [0, 0], [surface["width"], 0],
        [surface["width"], surface["height"]], [0, surface["height"]],
    ], dtype=np.float64)
    matrices = [homography_between(canonical, corners) for corners in stable]
    frames: list[dict[str, Any]] = []
    displacements = [0.0]
    area_changes = [0.0]
    centroid_moves = [0.0]
    areas = [polygon_area(stable[0])]
    for index in range(1, len(stable)):
        displacements.append(float(np.max(np.linalg.norm(stable[index] - stable[index - 1], axis=1))))
        areas.append(polygon_area(stable[index]))
        area_changes.append(abs(areas[-1] - areas[-2]) / max(areas[-2], 1e-9))
        centroid_moves.append(float(np.linalg.norm(np.mean(stable[index], axis=0) - np.mean(stable[index - 1], axis=0))))
    for index, (item, corners, matrix) in enumerate(zip(raw, stable, matrices, strict=True)):
        candidate_count, inlier_count = len(item.visible_indices), len(item.inlier_indices)
        frame = {
            "frame": item.frame, "state": item.state, "recovered": item.state != "good",
            "recoverySource": "interpolate-valid-frames" if item.state != "good" else None,
            "candidateCount": candidate_count, "inlierCount": inlier_count,
            "inlierRatio": round(inlier_count / candidate_count, 4) if candidate_count else 0,
            "reprojectionError": round(item.reprojection_error, 4) if item.reprojection_error is not None else None,
            "visibleIndices": item.visible_indices, "inlierIndices": item.inlier_indices,
            "rejectedIndices": item.rejected_indices,
            "rawCorners": _rounded(item.corners, 3) if item.corners is not None else None,
            "corners": _rounded(corners, 3), "homography": _rounded(matrix, 6),
        }
        if resolution_details is not None:
            detail = resolution_details[index]
            frame.update(detail)
            frame["recoverySource"] = (
                detail["resolutionSource"] if detail["resolutionSource"] != "direct" else None
            )
        frames.append(frame)
    recovered = [item.frame for item in raw if item.state != "good"]
    coordinates = {
        "version": qa_version, "tracker": "tapnextpp", "model": result.get("model"),
        "source": {"width": expected["width"], "height": expected["height"], "fps": expected["fps"]},
        "analysis": analysis, "surface": surface,
        "reference": {"sourceCorners": parameters["frameZeroCorners"], "analysisCorners": reference_corners.tolist(), "canonicalCorners": canonical.tolist()},
        "settings": {"points": len(queries), "planePoints": plane_count, "queryLayout": query_layout, "ransacThreshold": 3.0, "minimumInliers": 8, "minimumInlierRatio": 0.60, "maximumReferenceAreaRatio": parameters["maxReferenceAreaRatio"], "medianWindow": 5, "emaAlpha": 0.25, "fallback": "coherent-offscreen-v2" if qa_version == 2 else "interpolate-valid-frames"},
        "summary": {"frameCount": frame_count, "goodFrames": frame_count - len(recovered), "recoveredFrames": recovered},
        "frames": frames,
    }
    if resolution_details is not None:
        resolution_sources = [detail["resolutionSource"] for detail in resolution_details]
        coordinates["summary"].update(
            {
                "partialRecovered": resolution_sources.count("partial-affine"),
                "predictionOnly": resolution_sources.count("predicted"),
                "bridged": resolution_sources.count("bridged"),
                "held": resolution_sources.count("held"),
                "fullyOffscreenFrames": [
                    index
                    for index, detail in enumerate(resolution_details)
                    if detail["fullyOffscreen"]
                ],
                "maximumPredictionAge": max(
                    int(detail["predictionAge"]) for detail in resolution_details
                ),
            }
        )
    reprojections = [float(frame["reprojectionError"]) for frame in frames if frame["reprojectionError"] is not None]
    metrics = {
        "frameCount": frame_count, "directGood": frame_count - len(recovered), "recovered": len(recovered),
        "invalidGeometry": 0, "recoveredFrames": recovered,
        "cornerDisplacement": [round(v, 4) for v in displacements],
        "maximumCornerDisplacement": round(max(displacements), 4), "maximumCornerDisplacementFrame": int(np.argmax(displacements)),
        "homographyJump": [round(v, 4) for v in displacements],
        "maximumHomographyJump": round(max(displacements), 4), "maximumHomographyJumpFrame": int(np.argmax(displacements)),
        "quadArea": [round(v, 4) for v in areas], "quadAreaChange": [round(v, 6) for v in area_changes],
        "maximumQuadAreaChange": round(max(area_changes), 6), "maximumQuadAreaChangeFrame": int(np.argmax(area_changes)),
        "quadCentroidMovement": [round(v, 4) for v in centroid_moves],
        "maximumCentroidMovement": round(max(centroid_moves), 4), "maximumCentroidMovementFrame": int(np.argmax(centroid_moves)),
        "reprojectionError": [frame["reprojectionError"] for frame in frames],
        "p95ReprojectionError": round(float(np.percentile(reprojections, 95)), 4) if reprojections else None,
        "visibilityCount": [frame["candidateCount"] for frame in frames],
        "inlierCount": [frame["inlierCount"] for frame in frames],
        "inlierRatio": [frame["inlierRatio"] for frame in frames],
    }
    if resolution_details is not None:
        sources = [detail["resolutionSource"] for detail in resolution_details]
        metrics.update(
            {
                "partialRecovered": sources.count("partial-affine"),
                "predictionOnly": sources.count("predicted"),
                "bridged": sources.count("bridged"),
                "held": sources.count("held"),
                "fullyOffscreen": sum(bool(detail["fullyOffscreen"]) for detail in resolution_details),
                "maximumPredictionAge": max(int(detail["predictionAge"]) for detail in resolution_details),
                "resolutionSource": sources,
                "offscreenCornerCount": [int(detail["offscreenCornerCount"]) for detail in resolution_details],
            }
        )
    suspects = select_suspects(frames, metrics)
    return coordinates, metrics, suspects, stable


def select_suspects(frames: list[dict[str, Any]], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: dict[int, dict[str, Any]] = {}
    series = [
        ("corner_displacement_spike", "cornerDisplacement"),
        ("homography_jump", "homographyJump"),
        ("large_quad_area_change", "quadAreaChange"),
        ("centroid_jump", "quadCentroidMovement"),
    ]
    for reason, name in series:
        values = np.asarray(metrics[name], dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scores = np.abs(values - median) / mad if mad > 0 else np.abs(values)
        for frame in np.argsort(-scores, kind="stable")[:2]:
            index = int(frame)
            entry = candidates.setdefault(index, {"frame": index, "reasons": [], "values": {}, "score": 0.0})
            entry["reasons"].append(reason)
            entry["values"][name] = round(float(values[index]), 4)
            entry["score"] = max(entry["score"], round(float(scores[index]), 4))
    for frame in frames:
        if frame["recovered"]:
            index = frame["frame"]
            entry = candidates.setdefault(index, {"frame": index, "reasons": [], "values": {}, "score": 0.0})
            entry["reasons"].append("recovered_tracking")
            if frame["state"] == "low_confidence":
                entry["reasons"].append("low_confidence")
            entry["score"] = max(entry["score"], 100.0)
        source = frame.get("resolutionSource")
        source_reason = {
            "partial-affine": "partial_tracking",
            "bridged": "offscreen_prediction",
            "predicted": "offscreen_prediction",
            "held": "held_prediction",
        }.get(source)
        if source_reason is not None:
            index = frame["frame"]
            entry = candidates.setdefault(
                index,
                {"frame": index, "reasons": [], "values": {}, "score": 0.0},
            )
            entry["reasons"].append(source_reason)
            entry["score"] = max(entry["score"], 100.0)
        if frame.get("resolutionSource") == "direct" and frame["frame"] > 0:
            previous_source = frames[frame["frame"] - 1].get("resolutionSource")
            if previous_source in {"bridged", "predicted", "held"}:
                index = frame["frame"]
                entry = candidates.setdefault(
                    index,
                    {"frame": index, "reasons": [], "values": {}, "score": 0.0},
                )
                entry["reasons"].append("reentry_tracking")
                entry["score"] = max(entry["score"], 100.0)
    ordered = sorted(candidates.values(), key=lambda item: (-item["score"], item["frame"]))[:10]
    for item in ordered:
        item["reasons"] = sorted(set(item["reasons"]))
    return ordered


def annotate(frame: np.ndarray, corners: np.ndarray, frame_number: int, label: str, extra: str = "") -> np.ndarray:
    result = frame.copy()
    # Frames are RGB here, so these channel values render as cyan.
    cv2.polylines(result, [np.round(corners).astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA)
    text = f"frame {frame_number:03d} | {label}" + (f" | {extra}" if extra else "")
    cv2.rectangle(result, (0, 0), (result.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(result, text, (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def _draw_offscreen_locator(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    if frame.shape[1] < 240 or frame.shape[0] < 180:
        return frame
    result = frame.copy()
    inset_width, inset_height = 160, 110
    left, top = frame.shape[1] - inset_width - 12, 60
    frame_quad = np.asarray(
        [[0, 0], [frame.shape[1] - 1, 0], [frame.shape[1] - 1, frame.shape[0] - 1], [0, frame.shape[0] - 1]],
        dtype=np.float64,
    )
    combined = np.vstack([frame_quad, corners])
    minimum = np.min(combined, axis=0)
    maximum = np.max(combined, axis=0)
    span = np.maximum(maximum - minimum, 1.0)

    def locate(points: np.ndarray) -> np.ndarray:
        normalized = (points - minimum) / span
        mapped = normalized * np.array([inset_width - 16, inset_height - 16])
        return np.round(mapped + np.array([left + 8, top + 8])).astype(np.int32)

    cv2.rectangle(result, (left, top), (left + inset_width, top + inset_height), (0, 0, 0), -1)
    cv2.rectangle(result, (left, top), (left + inset_width, top + inset_height), (255, 255, 255), 1)
    cv2.polylines(result, [locate(frame_quad)], True, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.polylines(result, [locate(corners)], True, (0, 255, 255), 2, cv2.LINE_AA)
    return result


def render_review_artifacts(
    frames: np.ndarray, stable_analysis: np.ndarray, analysis: dict[str, int],
    suspects: list[dict[str, Any]], output_dir: Path,
    frame_details: list[dict[str, Any]] | None = None,
) -> tuple[Path, list[Path], dict[int, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = np.array([frames.shape[2] / analysis["width"], frames.shape[1] / analysis["height"]])
    stable = stable_analysis * scale
    annotated = np.empty_like(frames[:, :, :, :3])
    for index, frame in enumerate(frames):
        detail = frame_details[index] if frame_details is not None else None
        resolution_label = (
            detail.get("resolutionSource", "direct") if detail else "direct"
        )
        chroma_label = detail.get("chromaTailSource") if detail else None
        label = (
            chroma_label
            if chroma_label in {"tailBlend", "chromaTail"}
            else resolution_label
        )
        image = annotate(frame[:, :, :3], stable[index], index, label)
        if detail and (
            int(detail.get("offscreenCornerCount", 0)) > 0
            or resolution_label != "direct"
        ):
            image = _draw_offscreen_locator(image, stable[index])
        annotated[index] = image

    import imageio.v3 as iio
    review_video = output_dir / "tracking-review.mp4"
    iio.imwrite(review_video, annotated, fps=24, codec="libx264", pixelformat="yuv420p")
    sheets: list[Path] = []
    tile_width, tile_height = OVERVIEW_TILE_SIZE
    blank_tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
    for start in range(0, len(annotated), OVERVIEW_SHEET_FRAME_COUNT):
        end = min(start + OVERVIEW_SHEET_FRAME_COUNT, len(annotated))
        tiles = [
            cv2.resize(frame, OVERVIEW_TILE_SIZE, interpolation=cv2.INTER_AREA)
            for frame in annotated[start:end]
        ]
        tiles.extend(
            blank_tile.copy()
            for _ in range(OVERVIEW_SHEET_FRAME_COUNT - len(tiles))
        )
        rows = [
            np.hstack(tiles[row:row + OVERVIEW_SHEET_COLUMNS])
            for row in range(0, OVERVIEW_SHEET_FRAME_COUNT, OVERVIEW_SHEET_COLUMNS)
        ]
        sheet = np.vstack(rows)
        path = output_dir / f"overview-{start:03d}-{end - 1:03d}.jpg"
        if not cv2.imwrite(
            str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        ):
            raise RuntimeError(f"failed to write {path.name}")
        sheets.append(path)
    suspect_paths: dict[int, Path] = {}
    suspect_dir = output_dir / "suspects"
    suspect_dir.mkdir(parents=True, exist_ok=True)
    for suspect in suspects:
        index = suspect["frame"]
        extra = ",".join(suspect["reasons"])
        image = annotate(frames[index, :, :, :3], stable[index], index, "suspect", extra)
        path = suspect_dir / f"frame-{index:03d}.jpg"
        if not cv2.imwrite(
            str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        ):
            raise RuntimeError(f"failed to write {path.name}")
        suspect_paths[index] = path
    return review_video, sheets, suspect_paths
