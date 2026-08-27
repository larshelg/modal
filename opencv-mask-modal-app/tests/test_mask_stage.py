import numpy as np
import pytest

from mask_stage import (
    DEFAULT_LOWER_HSV,
    DEFAULT_SOFT_CHROMA,
    DEFAULT_UPPER_HSV,
    clip_polygon_to_frame,
    create_masks,
    parse_s3_url,
    tracking_corners_to_source,
    validate_request,
    validate_tracking,
)


def valid_request(**overrides):
    request = {
        "video_url": "s3://studio/input.mp4",
        "tracking_url": "s3://studio/tracking.json",
        "output_prefix": "s3://studio/job-123/",
    }
    request.update(overrides)
    return request


def test_request_defaults_are_small_and_explicit():
    assert validate_request(valid_request()) == {
        **valid_request(),
        "green_hsv": {"lower": DEFAULT_LOWER_HSV, "upper": DEFAULT_UPPER_HSV},
        "debug_frames": [],
        "soft_chroma": DEFAULT_SOFT_CHROMA,
    }


def test_request_normalizes_prefix_and_debug_frames():
    normalized = validate_request(
        valid_request(output_prefix="s3://studio/job-123", debug_frames=[42, 0, 42])
    )
    assert normalized["output_prefix"] == "s3://studio/job-123/"
    assert normalized["debug_frames"] == [0, 42]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"video_url": "https://example.test/input.mp4"}, "s3://"),
        ({"tracking_url": "s3://studio/../tracking.json"}, "unsafe"),
        ({"output_prefix": "s3://studio/"}, "unsafe"),
        ({"green_hsv": {"lower": [180, 0, 0]}}, "between 0 and 179"),
        (
            {"green_hsv": {"lower": [95, 50, 40], "upper": [35, 255, 255]}},
            "must not exceed",
        ),
        ({"debug_frames": [False]}, "non-negative integer"),
        ({"soft_chroma": {"mode": "global"}}, "disabled or boundary"),
        ({"soft_chroma": {"radius_px": 0}}, "between 1 and 10"),
        ({"soft_chroma": {"hsv_ramp": [8, 0, 40]}}, "must be positive"),
        ({"surprise": True}, "unsupported"),
    ],
)
def test_request_rejects_invalid_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_request(valid_request(**overrides))


def test_s3_parser_preserves_bucket_and_key():
    assert parse_s3_url("s3://studio/path/input.mp4", "video_url") == (
        "studio",
        "path/input.mp4",
    )


def test_tracking_accepts_richer_tapnext_coordinates():
    payload = {
        "version": 1,
        "source": {"width": 20, "height": 10, "fps": "24/1"},
        "frames": [
            {
                "frame": 0,
                "state": "good",
                "corners": [[1, 1], [18, 1], [18, 8], [1, 8]],
                "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
        ],
    }
    assert validate_tracking(payload) == [
        [[1.0, 1.0], [18.0, 1.0], [18.0, 8.0], [1.0, 8.0]]
    ]


def test_tapnext_analysis_corners_are_scaled_to_source_video():
    payload = {
        "source": {"width": 20, "height": 10, "fps": 24},
        "analysis": {"width": 10, "height": 5},
    }
    tracking = [[[1.0, 1.0], [8.0, 1.0], [8.0, 4.0], [1.0, 4.0]]]
    assert tracking_corners_to_source(payload, tracking, 20, 10) == [
        [[2.0, 2.0], [16.0, 2.0], [16.0, 8.0], [2.0, 8.0]]
    ]


def test_minimal_tracking_without_analysis_is_already_source_space():
    tracking = [[[1.0, 1.0], [8.0, 1.0], [8.0, 4.0], [1.0, 4.0]]]
    assert tracking_corners_to_source({}, tracking, 20, 10) is tracking


def test_tracking_source_dimensions_must_match_video():
    with pytest.raises(ValueError, match="source dimensions"):
        tracking_corners_to_source(
            {"source": {"width": 21, "height": 10}},
            [[[1.0, 1.0], [8.0, 1.0], [8.0, 4.0], [1.0, 4.0]]],
            20,
            10,
        )


def test_tracking_requires_one_ordered_quad_per_frame():
    with pytest.raises(ValueError, match="must be 0"):
        validate_tracking(
            {
                "frames": [
                    {
                        "frame": 1,
                        "corners": [[1, 1], [8, 1], [8, 8], [1, 8]],
                    }
                ]
            }
        )
    with pytest.raises(ValueError, match="convex"):
        validate_tracking(
            {
                "frames": [
                    {
                        "frame": 0,
                        "corners": [[1, 1], [8, 8], [8, 1], [1, 8]],
                    }
                ]
            }
        )


def test_mask_is_green_intersection_with_polygon():
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    frame[:, :] = [0, 255, 0]
    frame[5:7, 5:7] = [0, 0, 255]
    corners = [[1, 1], [10, 1], [10, 10], [1, 10]]
    polygon, green, hard_replace, replace, coverage = create_masks(
        frame, corners, DEFAULT_LOWER_HSV, DEFAULT_UPPER_HSV
    )
    assert polygon[1, 1] == 255
    assert replace[0, 0] == 0
    assert green[0, 0] == 255
    assert np.array_equal(hard_replace, replace)
    assert replace[3, 3] == 255
    assert replace[5, 5] == 0
    assert coverage == pytest.approx(0.96)


def test_soft_chroma_changes_only_mixed_pixels_near_foreground_boundary():
    frame = np.zeros((15, 15, 3), dtype=np.uint8)
    frame[:, :] = [0, 255, 0]
    frame[5:10, 5:10] = [0, 0, 255]
    frame[7, 10] = [80, 100, 80]
    corners = [[1, 1], [13, 1], [13, 13], [1, 13]]

    polygon, _, hard, soft, _ = create_masks(
        frame,
        corners,
        DEFAULT_LOWER_HSV,
        DEFAULT_UPPER_HSV,
        {"mode": "boundary", "radius_px": 2, "hsv_ramp": [8, 40, 40]},
    )

    assert hard[7, 10] == 255
    assert 0 < soft[7, 10] < 255
    assert soft[7, 7] == 0
    assert soft[2, 2] == 255
    assert np.all(soft[polygon == 0] == 0)


def test_disabled_soft_chroma_is_byte_identical_to_hard_mask():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :] = [0, 255, 0]
    frame[3:5, 3:5] = [0, 0, 255]

    _, _, hard, final, _ = create_masks(
        frame,
        [[1, 1], [6, 1], [6, 6], [1, 6]],
        DEFAULT_LOWER_HSV,
        DEFAULT_UPPER_HSV,
        DEFAULT_SOFT_CHROMA,
    )

    assert np.array_equal(final, hard)
    assert not np.any((final > 0) & (final < 255))


def test_partially_offscreen_polygon_is_clipped_without_corner_clamping():
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    frame[:, :] = [0, 255, 0]
    corners = [[-4, 1], [16, 1], [16, 10], [-4, 10]]

    clipped = clip_polygon_to_frame(corners, 12, 12)
    polygon, _, _, replace, coverage = create_masks(
        frame, corners, DEFAULT_LOWER_HSV, DEFAULT_UPPER_HSV
    )

    assert clipped == [[0.0, 1.0], [11.0, 1.0], [11.0, 10.0], [0.0, 10.0]]
    assert np.all(replace[1:11] == 255)
    assert np.all(replace[0] == 0)
    assert np.array_equal(polygon, replace)
    assert coverage == 1.0


def test_four_offscreen_corners_can_still_enclose_the_visible_frame():
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    frame[:, :] = [0, 255, 0]
    corners = [[-2, -2], [13, -2], [13, 13], [-2, 13]]

    polygon, _, hard, replace, coverage = create_masks(
        frame, corners, DEFAULT_LOWER_HSV, DEFAULT_UPPER_HSV
    )

    assert np.all(polygon == 255)
    assert np.array_equal(hard, replace)
    assert coverage == 1.0


def test_fully_offscreen_polygon_emits_black_mask_and_no_coverage():
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    frame[:, :] = [0, 255, 0]
    corners = [[20, 1], [30, 1], [30, 10], [20, 10]]

    polygon, green, hard, replace, coverage = create_masks(
        frame, corners, DEFAULT_LOWER_HSV, DEFAULT_UPPER_HSV
    )

    assert np.all(polygon == 0)
    assert np.all(green == 255)
    assert np.all(hard == 0)
    assert np.all(replace == 0)
    assert coverage is None
