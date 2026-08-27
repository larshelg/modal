"""Structural mask QA, metrics, suspects, and review evidence for SAM 3.1."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def validate_masks(
    masks: list[Any], width: int, height: int, frame_count: int
) -> np.ndarray:
    result = np.asarray(masks, dtype=bool)
    if result.shape != (frame_count, height, width):
        raise ValueError(
            f"masks must have shape [{frame_count}, {height}, {width}], got {result.shape}"
        )
    areas = result.sum(axis=(1, 2))
    missing = np.flatnonzero(areas <= 0).astype(int).tolist()
    if missing:
        raise ValueError(f"SAM stage produced empty masks for frames {missing}")
    return result


def _bbox(mask: np.ndarray) -> list[int]:
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("mask must be non-empty")
    x0, y0, x1, y1 = int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1
    return [x0, y0, x1 - x0, y1 - y0]


def _centroid(mask: np.ndarray) -> list[float]:
    y, x = np.nonzero(mask)
    return [round(float(x.mean()), 4), round(float(y.mean()), 4)]


def build_metrics(masks: list[Any], metadata: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = metadata["source"]
    frame_count = int(source["frameCount"])
    checked = validate_masks(masks, source["width"], source["height"], frame_count)
    areas = checked.sum(axis=(1, 2)).astype(np.int64)
    total = source["width"] * source["height"]
    boxes = [_bbox(mask) for mask in checked]
    centroids = [_centroid(mask) for mask in checked]
    adjacent: list[float | None] = [None]
    area_changes: list[float] = [0.0]
    centroid_moves: list[float] = [0.0]
    bbox_changes: list[float] = [0.0]
    for index in range(1, frame_count):
        intersection = int(np.logical_and(checked[index - 1], checked[index]).sum())
        union = int(np.logical_or(checked[index - 1], checked[index]).sum())
        adjacent.append(round(intersection / max(union, 1), 6))
        area_changes.append(round(abs(int(areas[index]) - int(areas[index - 1])) / max(int(areas[index - 1]), 1), 6))
        centroid_moves.append(round(math.dist(centroids[index - 1], centroids[index]), 4))
        bbox_changes.append(round(max(abs(a - b) for a, b in zip(boxes[index - 1], boxes[index], strict=True)), 4))

    selected_ids = metadata["tracking"]["selectedObjectIds"]
    compatibility = metadata.get("compatibility", {})
    per_frame = []
    for index in range(frame_count):
        per_frame.append({
            "frame": index,
            "foregroundPixels": int(areas[index]),
            "areaRatio": round(int(areas[index]) / total, 8),
            "adjacentIou": adjacent[index],
            "centroid": centroids[index],
            "centroidMovement": centroid_moves[index],
            "bboxXywh": boxes[index],
            "bboxChange": bbox_changes[index],
            "areaChangeRatio": area_changes[index],
            "selectedObjectId": selected_ids[index],
        })
    iou_values = [float(value) for value in adjacent[1:] if value is not None]
    min_iou_frame = 1 + int(np.argmin(iou_values))
    min_area_frame = int(np.argmin(areas))
    max_area_frame = int(np.argmax(areas))
    metrics = {
        "frameCount": frame_count,
        "nonemptyFrames": frame_count,
        "missingFrames": [],
        "foregroundPixels": [int(value) for value in areas],
        "areaRatio": [frame["areaRatio"] for frame in per_frame],
        "adjacentIou": adjacent,
        "minimumAdjacentIou": min(iou_values),
        "minimumAdjacentIouFrame": min_iou_frame,
        "centroid": centroids,
        "centroidMovement": centroid_moves,
        "maximumCentroidMovement": max(centroid_moves),
        "maximumCentroidMovementFrame": int(np.argmax(centroid_moves)),
        "bboxXywh": boxes,
        "bboxChange": bbox_changes,
        "maximumBboxChange": max(bbox_changes),
        "maximumBboxChangeFrame": int(np.argmax(bbox_changes)),
        "areaChangeRatio": area_changes,
        "maximumAreaChangeRatio": max(area_changes),
        "maximumAreaChangeFrame": int(np.argmax(area_changes)),
        "minimumArea": int(areas[min_area_frame]),
        "minimumAreaFrame": min_area_frame,
        "maximumArea": int(areas[max_area_frame]),
        "maximumAreaFrame": max_area_frame,
        "selectedObjectId": selected_ids,
        "gpuAttempts": metadata.get("dispatch", {}).get("gpuAttempts", []),
        "compatibilityAdapter": bool(compatibility.get("discardFalseOffloadStateToCpu", False)),
        "perFrame": per_frame,
    }
    return metrics, select_suspects(metrics)


def _scores(values: list[float]) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    return np.abs(data - median) / mad if mad > 0 else np.abs(data)


def select_suspects(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    events: dict[int, dict[str, Any]] = {}

    def add(frame: int, reason: str, name: str, value: float, score: float, related: list[int] | None = None):
        event = events.setdefault(frame, {
            "frame": frame, "reasons": [], "values": {}, "score": 0.0,
        })
        event["reasons"].append(reason)
        event["values"][name] = round(float(value), 6)
        event["score"] = max(float(event["score"]), float(score))
        if related:
            event["relatedFrames"] = sorted(set(event.get("relatedFrames", []) + related))

    series = [
        ("large_mask_area_change", "areaChangeRatio"),
        ("centroid_jump", "centroidMovement"),
        ("large_bbox_change", "bboxChange"),
    ]
    for reason, name in series:
        values = [float(value) for value in metrics[name]]
        scores = _scores(values)
        selected = [int(index) for index in np.flatnonzero(scores > 3)]
        if not selected:
            selected = [int(np.argmax(scores))]
        for frame in sorted(selected, key=lambda index: (-scores[index], index))[:2]:
            add(frame, reason, name, values[frame], scores[frame], [max(0, frame - 1)])

    iou_frame = int(metrics["minimumAdjacentIouFrame"])
    iou = float(metrics["minimumAdjacentIou"])
    add(iou_frame, "low_adjacent_iou", "adjacentIou", iou, max(0.0, 1.0 - iou), [iou_frame - 1])
    add(int(metrics["minimumAreaFrame"]), "minimum_mask_size", "foregroundPixels", metrics["minimumArea"], 1.0)
    add(int(metrics["maximumAreaFrame"]), "maximum_mask_size", "foregroundPixels", metrics["maximumArea"], 1.0)

    ordered = sorted(events.values(), key=lambda event: (-float(event["score"]), int(event["frame"])))[:10]
    for event in ordered:
        event["reasons"] = sorted(set(event["reasons"]))
        event["score"] = round(float(event["score"]), 4)
    return ordered


def verify_mask_video(path: Path, source: dict[str, Any]) -> dict[str, Any]:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("mask must contain exactly one video stream")
    stream = streams[0]
    expected = {
        "codec_name": "ffv1", "pix_fmt": "gray",
        "width": source["width"], "height": source["height"],
        "r_frame_rate": source["fpsRatio"],
        "nb_read_frames": str(source["frameCount"]),
    }
    for key, value in expected.items():
        if stream.get(key) != value:
            raise ValueError(f"mask video {key} is {stream.get(key)!r}; expected {value!r}")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        check=True, capture_output=True,
    )
    return stream


def _overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = frame[:, :, :3].astype(np.float32)
    alpha = mask.astype(np.float32)[:, :, None] * 0.58
    tint = np.asarray([255.0, 36.0, 190.0], dtype=np.float32)
    return np.clip(rgb * (1 - alpha) + tint * alpha, 0, 255).astype(np.uint8)


def _label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(result, text, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def render_review_artifacts(
    frames: np.ndarray, masks: list[Any], metrics: dict[str, Any],
    suspects: list[dict[str, Any]], output_dir: Path,
) -> tuple[list[Path], dict[int, Path]]:
    frame_count = int(frames.shape[0])
    checked = validate_masks(masks, frames.shape[2], frames.shape[1], frame_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets: list[Path] = []
    for start in range(0, frame_count, 30):
        tiles = []
        end = min(start + 30, frame_count)
        for index in range(start, end):
            overlay = _overlay(frames[index], checked[index])
            iou = metrics["adjacentIou"][index]
            suffix = "n/a" if iou is None else f"{float(iou):.3f}"
            label = f"f{index:03d} area {metrics['areaRatio'][index] * 100:.1f}% iou {suffix}"
            tile = _label(overlay, label)
            tiles.append(cv2.resize(tile, (256, 224), interpolation=cv2.INTER_AREA))
        blank = np.zeros_like(tiles[0])
        while len(tiles) % 6:
            tiles.append(blank)
        sheet = np.vstack(
            [np.hstack(tiles[row:row + 6]) for row in range(0, len(tiles), 6)]
        )
        path = output_dir / f"overview-{start:03d}-{end - 1:03d}.jpg"
        if not cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise RuntimeError(f"failed to write {path.name}")
        sheets.append(path)

    suspect_paths: dict[int, Path] = {}
    suspect_dir = output_dir / "suspects"
    suspect_dir.mkdir(parents=True, exist_ok=True)
    for suspect in suspects:
        index = int(suspect["frame"])
        original = frames[index, :, :, :3]
        overlay = _overlay(original, checked[index])
        combined = np.hstack([original, overlay])
        reasons = ",".join(suspect["reasons"])
        combined = _label(combined, f"frame {index:03d} | {reasons} | {suspect['values']}")
        path = suspect_dir / f"frame-{index:03d}.jpg"
        if not cv2.imwrite(str(path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write {path.name}")
        suspect_paths[index] = path
    return sheets, suspect_paths
