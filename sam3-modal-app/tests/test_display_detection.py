import hashlib

import cv2
import numpy as np
import pytest
from PIL import Image

from display_detection import (
    DEFAULT_TEXT_PROMPTS,
    render_display_overlay,
    select_display_candidate,
    validate_display_request,
)


HASH = "a" * 64
IMAGE_BYTES = b"image"
IMAGE_HASH = hashlib.sha256(IMAGE_BYTES).hexdigest()


def valid_request():
    run_id = "display-test-20260825"
    return {
        "schemaVersion": 1,
        "runId": run_id,
        "stage": "display-detection",
        "inputHash": HASH,
        "inputs": {
            "image": {
                "storage": "s3",
                "bucket": "bucket",
                "key": (
                    f"studio-experiments/{run_id}/source/hash/"
                    "review/keyframes/frame-000.png"
                ),
                "sha256": IMAGE_HASH,
                "sizeBytes": len(IMAGE_BYTES),
                "contentType": "image/png",
            }
        },
        "parameters": {},
        "output": {
            "bucket": "bucket",
            "prefix": (
                f"studio-experiments/{run_id}/display-detection/{HASH}/"
            ),
        },
    }


def test_display_request_accepts_s3_image_and_applies_defaults():
    checked = validate_display_request(valid_request(), "bucket")
    assert checked["parameters"] == {
        "textPrompts": list(DEFAULT_TEXT_PROMPTS),
        "scoreThreshold": 0.35,
        "minAreaRatio": 0.01,
        "maxAreaRatio": 0.95,
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.update(stage="sam"), "stage"),
        (lambda r: r["inputs"]["image"].update(storage="https"), "storage"),
        (lambda r: r["inputs"]["image"].update(key="../image.png"), "key"),
        (
            lambda r: r["inputs"]["image"].update(contentType="application/json"),
            "contentType",
        ),
        (lambda r: r["parameters"].update(textPrompts=[]), "textPrompts"),
        (lambda r: r["parameters"].update(scoreThreshold=1), "scoreThreshold"),
        (
            lambda r: r["parameters"].update(
                minAreaRatio=0.5, maxAreaRatio=0.4
            ),
            "minAreaRatio",
        ),
        (
            lambda r: r["output"].update(prefix="studio-experiments/other/"),
            "output.prefix",
        ),
    ],
)
def test_display_request_rejects_invalid_or_unsafe_inputs(change, message):
    request = valid_request()
    change(request)
    with pytest.raises(ValueError, match=message):
        validate_display_request(request, "bucket")


def synthetic_display_mask():
    mask = np.zeros((90, 120), dtype=np.uint8)
    corners = np.asarray(
        [[18, 22], [101, 14], [106, 70], [13, 77]], dtype=np.int32
    )
    cv2.fillConvexPoly(mask, corners, 1)
    return mask.astype(bool), corners


def test_display_mask_becomes_ordered_perspective_corners():
    mask, expected = synthetic_display_mask()
    selected = select_display_candidate(
        [("display screen", mask[None, None], np.asarray([0.91]))],
        score_threshold=0.35,
        min_area_ratio=0.01,
        max_area_ratio=0.95,
    )
    assert selected["prompt"] == "display screen"
    assert selected["candidateCount"] == 1
    assert selected["cornerMethod"] == "convex-approximation"
    assert np.allclose(selected["corners"], expected, atol=2)
    assert selected["rectangularity"] > 0.85


def test_geometry_score_prefers_rectangular_display_mask():
    display, _ = synthetic_display_mask()
    distractor = np.zeros_like(display, dtype=np.uint8)
    cv2.circle(distractor, (60, 45), 31, 1, -1)
    selected = select_display_candidate(
        [
            (
                "screen",
                np.stack([distractor.astype(bool), display])[:, None],
                np.asarray([0.80, 0.80]),
            )
        ],
        score_threshold=0.35,
        min_area_ratio=0.01,
        max_area_ratio=0.95,
    )
    assert selected["objectIndex"] == 1


def test_overlay_contains_source_mask_and_corner_labels(tmp_path):
    mask, _ = synthetic_display_mask()
    selected = select_display_candidate(
        [("monitor screen", mask[None], np.asarray([0.88]))],
        score_threshold=0.35,
        min_area_ratio=0.01,
        max_area_ratio=0.95,
    )
    image = np.full((90, 120, 3), 24, dtype=np.uint8)
    output = tmp_path / "overlay.png"
    render_display_overlay(image, selected, output)
    rendered = np.asarray(Image.open(output).convert("RGB"))
    assert rendered.shape == image.shape
    assert output.stat().st_size > 0
    assert not np.array_equal(rendered, image)


def test_detection_fails_without_a_candidate_above_threshold():
    mask, _ = synthetic_display_mask()
    with pytest.raises(RuntimeError, match="no display-like mask"):
        select_display_candidate(
            [("screen", mask[None], np.asarray([0.2]))],
            score_threshold=0.35,
            min_area_ratio=0.01,
            max_area_ratio=0.95,
        )
