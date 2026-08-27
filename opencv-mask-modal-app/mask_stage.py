"""Validation and pure OpenCV logic for the guided chroma mask stage."""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import urlsplit


DEFAULT_LOWER_HSV = [35, 50, 40]
DEFAULT_UPPER_HSV = [95, 255, 255]
DEFAULT_SOFT_CHROMA = {
    "mode": "disabled",
    "radius_px": 3,
    "hsv_ramp": [8, 40, 40],
}
MAX_DEBUG_FRAMES = 20
MAX_TRACKING_FRAMES = 3_600
MAX_SOFT_RADIUS = 10
MAX_COORDINATE_MAGNITUDE = 1_000_000_000


def parse_s3_url(value: Any, field: str, *, prefix: bool = False) -> tuple[str, str]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty s3:// URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.startswith("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{field} must be an s3://bucket/key URL")
    key = parsed.path[1:]
    if not key or "\\" in key or any(ord(character) < 32 for character in key):
        raise ValueError(f"{field} contains an unsafe S3 key")
    check_key = key[:-1] if prefix and key.endswith("/") else key
    if not check_key or any(part in ("", ".", "..") for part in check_key.split("/")):
        raise ValueError(f"{field} contains an unsafe S3 key")
    if prefix:
        key = f"{check_key}/"
    elif key.endswith("/"):
        raise ValueError(f"{field} must identify an S3 object")
    return parsed.netloc, key


def s3_url(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def _hsv_triplet(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must contain [hue, saturation, value]")
    limits = (179, 255, 255)
    result: list[int] = []
    for index, (component, maximum) in enumerate(zip(value, limits, strict=True)):
        if isinstance(component, bool) or not isinstance(component, int):
            raise ValueError(f"{field}[{index}] must be an integer")
        if not 0 <= component <= maximum:
            raise ValueError(f"{field}[{index}] must be between 0 and {maximum}")
        result.append(component)
    return result


def _soft_chroma(value: Any) -> dict[str, Any]:
    if value is None:
        return dict(DEFAULT_SOFT_CHROMA)
    if not isinstance(value, dict):
        raise ValueError("soft_chroma must be an object")
    unknown = sorted(set(value) - {"mode", "radius_px", "hsv_ramp"})
    if unknown:
        raise ValueError(f"unsupported soft_chroma fields: {', '.join(unknown)}")
    mode = value.get("mode", "boundary")
    if mode not in {"disabled", "boundary"}:
        raise ValueError("soft_chroma.mode must be disabled or boundary")
    radius = value.get("radius_px", DEFAULT_SOFT_CHROMA["radius_px"])
    if (
        isinstance(radius, bool)
        or not isinstance(radius, int)
        or not 1 <= radius <= MAX_SOFT_RADIUS
    ):
        raise ValueError(
            f"soft_chroma.radius_px must be an integer between 1 and {MAX_SOFT_RADIUS}"
        )
    ramp = _hsv_triplet(
        value.get("hsv_ramp", DEFAULT_SOFT_CHROMA["hsv_ramp"]),
        "soft_chroma.hsv_ramp",
    )
    if any(component <= 0 for component in ramp):
        raise ValueError("soft_chroma.hsv_ramp values must be positive")
    return {"mode": mode, "radius_px": radius, "hsv_ramp": ramp}


def validate_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    allowed = {
        "video_url",
        "tracking_url",
        "output_prefix",
        "green_hsv",
        "debug_frames",
        "soft_chroma",
    }
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise ValueError(f"unsupported request fields: {', '.join(unknown)}")
    for field in ("video_url", "tracking_url"):
        parse_s3_url(request.get(field), field)
    parse_s3_url(request.get("output_prefix"), "output_prefix", prefix=True)

    raw_hsv = request.get("green_hsv", {})
    if not isinstance(raw_hsv, dict):
        raise ValueError("green_hsv must be an object")
    unknown_hsv = sorted(set(raw_hsv) - {"lower", "upper"})
    if unknown_hsv:
        raise ValueError(f"unsupported green_hsv fields: {', '.join(unknown_hsv)}")
    lower = _hsv_triplet(raw_hsv.get("lower", DEFAULT_LOWER_HSV), "green_hsv.lower")
    upper = _hsv_triplet(raw_hsv.get("upper", DEFAULT_UPPER_HSV), "green_hsv.upper")
    if any(low > high for low, high in zip(lower, upper, strict=True)):
        raise ValueError("green_hsv.lower must not exceed green_hsv.upper")

    raw_debug_frames = request.get("debug_frames", [])
    if not isinstance(raw_debug_frames, list):
        raise ValueError("debug_frames must be an array")
    debug_frames: list[int] = []
    for index, frame in enumerate(raw_debug_frames):
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError(f"debug_frames[{index}] must be a non-negative integer")
        debug_frames.append(frame)
    debug_frames = sorted(set(debug_frames))
    if len(debug_frames) > MAX_DEBUG_FRAMES:
        raise ValueError(f"debug_frames must contain at most {MAX_DEBUG_FRAMES} frames")

    return {
        "video_url": request["video_url"],
        "tracking_url": request["tracking_url"],
        "output_prefix": s3_url(*parse_s3_url(request["output_prefix"], "output_prefix", prefix=True)),
        "green_hsv": {"lower": lower, "upper": upper},
        "debug_frames": debug_frames,
        "soft_chroma": _soft_chroma(request.get("soft_chroma")),
    }


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if abs(result) > MAX_COORDINATE_MAGNITUDE:
        raise ValueError(
            f"{field} magnitude must not exceed {MAX_COORDINATE_MAGNITUDE}"
        )
    return result


def _is_convex(corners: list[list[float]]) -> bool:
    signs: list[float] = []
    for index in range(4):
        origin = corners[index]
        first = corners[(index + 1) % 4]
        second = corners[(index + 2) % 4]
        edge_a = (first[0] - origin[0], first[1] - origin[1])
        edge_b = (second[0] - first[0], second[1] - first[1])
        signs.append(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])
    return all(value > 0 for value in signs) or all(value < 0 for value in signs)


def validate_tracking(payload: Any) -> list[list[list[float]]]:
    if not isinstance(payload, dict):
        raise ValueError("tracking JSON must be an object")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("tracking.frames must be a non-empty array")
    if len(frames) > MAX_TRACKING_FRAMES:
        raise ValueError(f"tracking.frames exceeds the {MAX_TRACKING_FRAMES}-frame limit")

    result: list[list[list[float]]] = []
    for expected_index, item in enumerate(frames):
        field = f"tracking.frames[{expected_index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{field} must be an object")
        frame = item.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame != expected_index:
            raise ValueError(f"{field}.frame must be {expected_index}")
        raw_corners = item.get("corners")
        if not isinstance(raw_corners, list) or len(raw_corners) != 4:
            raise ValueError(f"{field}.corners must contain four points")
        corners: list[list[float]] = []
        for corner_index, corner in enumerate(raw_corners):
            if not isinstance(corner, list) or len(corner) != 2:
                raise ValueError(f"{field}.corners[{corner_index}] must be [x, y]")
            corners.append(
                [
                    _finite_number(corner[0], f"{field}.corners[{corner_index}][0]"),
                    _finite_number(corner[1], f"{field}.corners[{corner_index}][1]"),
                ]
            )
        if not _is_convex(corners):
            raise ValueError(f"{field}.corners must be convex and consistently ordered")
        result.append(corners)
    return result


def tracking_corners_to_source(
    payload: dict[str, Any],
    tracking: list[list[list[float]]],
    video_width: int,
    video_height: int,
) -> list[list[list[float]]]:
    """Convert TAPNext analysis-canvas corners to source-video coordinates.

    A minimal tracking document without ``analysis`` is already in source
    coordinates. The production TAPNext++ coordinates artifact includes both
    ``source`` and ``analysis`` metadata and stores every frame's corners in
    the analysis coordinate space.
    """
    source = payload.get("source")
    if source is not None:
        if not isinstance(source, dict):
            raise ValueError("tracking.source must be an object")
        if source.get("width") != video_width or source.get("height") != video_height:
            raise ValueError("tracking source dimensions do not match video_url")

    analysis = payload.get("analysis")
    if analysis is None:
        return tracking
    if not isinstance(analysis, dict):
        raise ValueError("tracking.analysis must be an object")
    analysis_width = analysis.get("width")
    analysis_height = analysis.get("height")
    for value, field in (
        (analysis_width, "tracking.analysis.width"),
        (analysis_height, "tracking.analysis.height"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if source is None:
        raise ValueError("tracking.source is required with tracking.analysis")

    scale_x = video_width / analysis_width
    scale_y = video_height / analysis_height
    return [
        [[x * scale_x, y * scale_y] for x, y in corners]
        for corners in tracking
    ]


def clip_polygon_to_frame(
    corners: list[list[float]], width: int, height: int
) -> list[list[float]]:
    """Clip an ordered convex polygon to the visible source-frame rectangle.

    TAPNext++ V2 deliberately preserves negative and beyond-edge coordinates.
    Clipping the polygon keeps the projective shape correct; clamping each
    corner independently would distort it and can turn a fully off-screen quad
    into a false edge strip.
    """
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")

    points = [(float(x), float(y)) for x, y in corners]
    if not points:
        return []

    def clip_boundary(
        polygon: list[tuple[float, float]],
        inside: Any,
        intersection: Any,
    ) -> list[tuple[float, float]]:
        if not polygon:
            return []
        output: list[tuple[float, float]] = []
        previous = polygon[-1]
        previous_inside = inside(previous)
        for current in polygon:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current))
            previous = current
            previous_inside = current_inside
        return output

    def vertical(
        first: tuple[float, float],
        second: tuple[float, float],
        boundary: float,
    ) -> tuple[float, float]:
        delta = second[0] - first[0]
        if abs(delta) < 1e-12:
            return boundary, first[1]
        ratio = (boundary - first[0]) / delta
        return boundary, first[1] + ratio * (second[1] - first[1])

    def horizontal(
        first: tuple[float, float],
        second: tuple[float, float],
        boundary: float,
    ) -> tuple[float, float]:
        delta = second[1] - first[1]
        if abs(delta) < 1e-12:
            return first[0], boundary
        ratio = (boundary - first[1]) / delta
        return first[0] + ratio * (second[0] - first[0]), boundary

    right = float(width - 1)
    bottom = float(height - 1)
    points = clip_boundary(
        points, lambda point: point[0] >= 0.0, lambda a, b: vertical(a, b, 0.0)
    )
    points = clip_boundary(
        points, lambda point: point[0] <= right, lambda a, b: vertical(a, b, right)
    )
    points = clip_boundary(
        points, lambda point: point[1] >= 0.0, lambda a, b: horizontal(a, b, 0.0)
    )
    points = clip_boundary(
        points,
        lambda point: point[1] <= bottom,
        lambda a, b: horizontal(a, b, bottom),
    )
    return [[float(x), float(y)] for x, y in points]


def _channel_membership(
    channel: Any,
    lower: int,
    upper: int,
    ramp: int,
    maximum: int,
) -> Any:
    import numpy as np

    values = np.asarray(channel, dtype=np.float32)
    membership = np.ones(values.shape, dtype=np.float32)
    if lower > 0:
        outer = max(0, lower - ramp)
        inner = min(maximum, lower + ramp)
        membership = np.minimum(
            membership,
            np.clip((values - outer) / max(1, inner - outer), 0.0, 1.0),
        )
    if upper < maximum:
        inner = max(0, upper - ramp)
        outer = min(maximum, upper + ramp)
        membership = np.minimum(
            membership,
            np.clip((outer - values) / max(1, outer - inner), 0.0, 1.0),
        )
    return membership


def soft_green_score(
    hsv: Any,
    lower_hsv: list[int],
    upper_hsv: list[int],
    hsv_ramp: list[int],
) -> Any:
    """Return continuous green membership in the inclusive 0..255 range."""
    import numpy as np

    memberships = [
        _channel_membership(
            hsv[:, :, index],
            lower_hsv[index],
            upper_hsv[index],
            hsv_ramp[index],
            maximum,
        )
        for index, maximum in enumerate((179, 255, 255))
    ]
    score = np.minimum.reduce(memberships)
    return np.rint(score * 255.0).astype(np.uint8)


def create_masks(
    frame_bgr: Any,
    corners: list[list[float]],
    lower_hsv: list[int],
    upper_hsv: list[int],
    soft_chroma: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any, Any, float | None]:
    """Return visible polygon, green, hard/final masks, and visible coverage."""
    import cv2
    import numpy as np

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame must be an HxWx3 BGR image")
    height, width = frame_bgr.shape[:2]
    polygon_mask = np.zeros((height, width), dtype=np.uint8)
    clipped = clip_polygon_to_frame(corners, width, height)
    if len(clipped) >= 3:
        points = np.rint(np.asarray(clipped, dtype=np.float64)).astype(np.int32)
        cv2.fillPoly(polygon_mask, [points], 255)
    polygon_pixels = int(cv2.countNonZero(polygon_mask))
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(
        hsv,
        np.asarray(lower_hsv, dtype=np.uint8),
        np.asarray(upper_hsv, dtype=np.uint8),
    )
    hard_replace_mask = cv2.bitwise_and(green_mask, polygon_mask)
    replace_mask = hard_replace_mask.copy()
    settings = _soft_chroma(soft_chroma)
    if polygon_pixels > 0 and settings["mode"] == "boundary":
        foreground = (polygon_mask > 0) & (hard_replace_mask == 0)
        radius = settings["radius_px"]
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        foreground_u8 = foreground.astype(np.uint8)
        dilated = cv2.dilate(foreground_u8, kernel, iterations=1)
        eroded = cv2.erode(foreground_u8, kernel, iterations=1)
        boundary = (dilated != eroded) & (polygon_mask > 0)
        score = soft_green_score(
            hsv, lower_hsv, upper_hsv, settings["hsv_ramp"]
        )
        replace_mask[boundary] = score[boundary]
    coverage = (
        cv2.countNonZero(hard_replace_mask) / polygon_pixels
        if polygon_pixels > 0
        else None
    )
    return (
        polygon_mask,
        green_mask,
        hard_replace_mask,
        replace_mask,
        float(coverage) if coverage is not None else None,
    )
