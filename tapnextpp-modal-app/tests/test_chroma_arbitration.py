import cv2
import numpy as np

from chroma_arbitration import (
    CANONICAL_QUAD,
    apply_terminal_chroma_tail_recovery,
    calibrate_inner_quad,
    coherent_transform_gate,
    edge_evidence,
    green_mask,
    latch_terminal_chroma_tail,
    largest_green_component,
    mask_quadrilateral,
    project_reference_quad,
    score_calibrated_candidate,
    score_candidate,
    stabilize_arbitration_sequence,
    temporal_prediction_gate,
    transform_quad,
)


def synthetic_frame() -> np.ndarray:
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    frame[25:95, 35:150] = [0, 255, 0]
    frame[50:90, 110:130] = [30, 30, 30]
    return frame


def test_correct_quad_outscores_shifted_quad_with_occlusion():
    chroma = green_mask(synthetic_frame())
    correct = np.asarray([[35, 25], [149, 25], [149, 94], [35, 94]])
    shifted = correct + np.asarray([18, 0])

    correct_score = score_candidate(chroma, correct)
    shifted_score = score_candidate(chroma, shifted)

    assert correct_score["evidenceSufficient"] is True
    assert shifted_score["evidenceSufficient"] is True
    assert correct_score["score"] > shifted_score["score"]


def test_no_green_evidence_abstains():
    chroma = np.zeros((80, 100), dtype=np.uint8)
    result = score_candidate(
        chroma,
        np.asarray([[10, 10], [90, 10], [90, 70], [10, 70]]),
    )
    assert result["evidenceSufficient"] is False
    assert result["score"] is None


def test_largest_component_quad_and_projection_follow_motion():
    first = np.zeros((120, 180), dtype=np.uint8)
    second = np.zeros_like(first)
    cv2.fillPoly(
        first,
        [np.asarray([[35, 25], [145, 30], [135, 95], [40, 90]], np.int32)],
        255,
    )
    cv2.fillPoly(
        second,
        [np.asarray([[20, 15], [160, 20], [150, 105], [25, 100]], np.int32)],
        255,
    )
    first_quad, _, _ = mask_quadrilateral(largest_green_component(first))
    second_quad, _, _ = mask_quadrilateral(largest_green_component(second))
    outer = first_quad + np.asarray([[-3, -3], [3, -3], [3, 3], [-3, 3]])

    projected = project_reference_quad(first_quad, second_quad, outer)

    assert projected.shape == (4, 2)
    assert cv2.isContourConvex(projected.astype(np.float32))
    assert cv2.contourArea(projected.astype(np.float32)) > cv2.contourArea(
        outer.astype(np.float32)
    )


def test_calibrated_inset_and_per_edge_score_prefer_correct_outer_quad():
    outer = np.asarray([[20, 15], [160, 20], [150, 105], [25, 100]], dtype=float)
    inner_model = np.asarray(
        [[0.06, 0.07], [0.94, 0.07], [0.94, 0.93], [0.06, 0.93]], dtype=float
    )
    inner = transform_quad(CANONICAL_QUAD, outer, inner_model)
    component = np.zeros((130, 190), dtype=np.uint8)
    cv2.fillPoly(component, [np.rint(inner).astype(np.int32)], 255)
    samples = [(outer, inner), (outer + 1, inner + 1), (outer - 1, inner - 1)]
    calibrated, details = calibrate_inner_quad(samples)

    correct = score_calibrated_candidate(component, outer, calibrated)
    shifted = score_calibrated_candidate(
        component, outer + np.asarray([14.0, 0.0]), calibrated
    )

    assert details["retainedSamples"] == 3
    assert correct["evidenceSufficient"] is True
    assert correct["supportedEdgeCount"] == 4
    assert correct["score"] > shifted["score"]


def test_coherent_and_temporal_gates_reject_corner_spike():
    previous_previous = np.asarray(
        [[10, 10], [90, 10], [90, 90], [10, 90]], dtype=float
    )
    previous = previous_previous + np.asarray([2.0, 0.0])
    coherent = previous + np.asarray([2.0, 0.0])
    spike = coherent.copy()
    spike[2] += np.asarray([80.0, 0.0])

    assert coherent_transform_gate(previous, coherent)["accepted"] is True
    assert coherent_transform_gate(previous, spike)["accepted"] is False
    assert temporal_prediction_gate(
        previous_previous, previous, coherent
    )["accepted"] is True
    assert temporal_prediction_gate(
        previous_previous, previous, spike
    )["accepted"] is False


def test_calibrated_score_abstains_when_occlusion_erases_most_of_screen():
    inner = np.asarray([[20, 20], [160, 20], [160, 100], [20, 100]], dtype=float)
    component = np.zeros((120, 180), dtype=np.uint8)
    component[20:45, 20:160] = 255
    result = score_calibrated_candidate(component, inner, CANONICAL_QUAD)
    assert result["evidenceSufficient"] is False
    assert result["score"] is None


def test_arbitration_sequence_bridges_abstention_without_using_gap_candidates():
    baseline = np.stack(
        [CANONICAL_QUAD * 100 + np.asarray([index * 2.0, 0.0]) for index in range(9)]
    )
    assisted = baseline + np.asarray([20.0, 4.0])
    selected = baseline.copy()
    selected[2:5] = assisted[2:5]
    selected[7:] = assisted[7:]
    selected[5:7] = 10_000.0  # Destructive proposals must never enter the bridge.
    sources = ["baseline"] * 2 + ["chromaProjection"] * 3 + ["baseline"] * 2 + [
        "chromaProjection"
    ] * 2
    abstained = [False] * 5 + [True, True] + [False] * 2

    stabilized, labels = stabilize_arbitration_sequence(
        baseline, selected, sources, abstained, transition_frames=2
    )

    assert labels[2:4] == ["transitionBlend", "transitionBlend"]
    assert labels[5:7] == ["occlusionBridge", "occlusionBridge"]
    assert float(stabilized[5:7].max()) < 200.0
    assert np.allclose(stabilized[5] - baseline[5], [20.0, 4.0])
    assert np.allclose(stabilized[6] - baseline[6], [20.0, 4.0])


def test_terminal_tail_policy_requires_confirmation_and_blends_supported_preroll():
    baseline = np.stack(
        [CANONICAL_QUAD * 100 + np.asarray([index, 0.0]) for index in range(20)]
    )
    chroma = baseline + np.asarray([5.0, 2.0])
    frames = [
        {
            "frame": index,
            "resolutionSource": "direct" if index < 12 else "predicted",
        }
        for index in range(20)
    ]
    supported = [False] * 9 + [True] * 11
    acquisition = [False] * 12 + [True] * 8
    settings = {
        "enabled": True,
        "minimumTailFrames": 8,
        "acquisitionFrames": 3,
        "scoreMargin": 0.04,
        "minimumScore": 0.90,
        "minimumPrecision": 0.95,
        "transitionFrames": 6,
    }

    applied, sources, weights, summary = latch_terminal_chroma_tail(
        baseline, chroma, supported, acquisition, frames, settings
    )

    assert summary["applied"] is True
    assert summary["terminalRecoveryStart"] == 12
    assert summary["evidenceStart"] == 12
    assert summary["confirmationFrame"] == 14
    assert summary["blendStart"] == 9
    assert sources[:9] == ["baseline"] * 9
    assert sources[9:14] == ["tailBlend"] * 5
    assert sources[14:] == ["chromaTail"] * 6
    assert np.allclose(applied[:9], baseline[:9])
    assert 0.0 < weights[9] < weights[13] < 1.0
    assert np.allclose(applied[14:], chroma[14:])


def test_terminal_tail_policy_abstains_if_support_is_lost_after_confirmation():
    baseline = np.repeat((CANONICAL_QUAD * 100)[None, :, :], 20, axis=0)
    chroma = baseline + 8.0
    frames = [
        {"resolutionSource": "direct" if index < 12 else "held"}
        for index in range(20)
    ]
    settings = {
        "enabled": True,
        "minimumTailFrames": 8,
        "acquisitionFrames": 3,
        "scoreMargin": 0.04,
        "minimumScore": 0.90,
        "minimumPrecision": 0.95,
        "transitionFrames": 6,
    }
    supported = [False] * 12 + [True] * 5 + [False] + [True] * 2
    acquisition = [False] * 12 + [True] * 8

    applied, sources, weights, summary = latch_terminal_chroma_tail(
        baseline, chroma, supported, acquisition, frames, settings
    )

    assert summary["applied"] is False
    assert summary["reason"] == "chroma-support-not-continuous-through-tail"
    assert sources == ["baseline"] * 20
    assert np.count_nonzero(weights) == 0
    assert np.array_equal(applied, baseline)


def test_full_terminal_recovery_calibrates_rgb_frames_and_publishes_homographies():
    frame_count = 20
    outer = np.asarray(
        [[20.0, 15.0], [160.0, 20.0], [150.0, 105.0], [25.0, 100.0]]
    )
    inner_model = np.asarray(
        [[0.06, 0.07], [0.94, 0.07], [0.94, 0.93], [0.06, 0.93]]
    )
    inner = transform_quad(CANONICAL_QUAD, outer, inner_model)
    frames_rgb = np.zeros((frame_count, 120, 180, 3), dtype=np.uint8)
    for frame in frames_rgb:
        cv2.fillPoly(frame, [np.rint(inner).astype(np.int32)], (0, 255, 0))

    stable = np.repeat(outer[None, :, :], frame_count, axis=0)
    # Simulate the terminal prediction drift which the experiment corrected.
    stable[12:] += np.asarray([10.0, 0.0])
    canonical = CANONICAL_QUAD * np.asarray([900.0, 1280.0])
    coordinate_frames = []
    for index, corners in enumerate(stable):
        direct = index < 12
        coordinate_frames.append(
            {
                "frame": index,
                "state": "good" if direct else "low_confidence",
                "resolutionSource": "direct" if direct else "predicted",
                "corners": corners.tolist(),
                "homography": np.eye(3).tolist(),
            }
        )
    coordinates = {
        "source": {"width": 180, "height": 120, "fps": "24/1"},
        "analysis": {"width": 180, "height": 120},
        "reference": {"canonicalCorners": canonical.tolist()},
        "settings": {},
        "summary": {},
        "frames": coordinate_frames,
    }
    metrics = {"frameCount": frame_count}
    settings = {
        "enabled": True,
        "minimumTailFrames": 8,
        "acquisitionFrames": 3,
        "scoreMargin": 0.04,
        "minimumScore": 0.85,
        "minimumPrecision": 0.90,
        "transitionFrames": 3,
    }

    recovered_coordinates, recovered_metrics, recovered = (
        apply_terminal_chroma_tail_recovery(
            frames_rgb, coordinates, metrics, stable, settings
        )
    )

    summary = recovered_coordinates["summary"]["chromaTailRecovery"]
    assert summary["applied"] is True
    assert summary["terminalRecoveryStart"] == 12
    assert recovered_metrics["chromaTailApplied"] is True
    assert recovered_coordinates["frames"][-1]["chromaTailSource"] == "chromaTail"
    assert "baselineCorners" in recovered_coordinates["frames"][-1]
    assert np.isfinite(
        np.asarray(recovered_coordinates["frames"][-1]["homography"])
    ).all()
    assert np.max(np.abs(recovered[-1] - outer)) < 2.0
