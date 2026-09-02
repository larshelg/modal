import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from tracking_geometry import (
    ADDITIVE_QUERY_LAYOUT,
    HYBRID_QUERY_LAYOUT,
    MAX_TAIL_PREDICTION_FRAMES,
    PLANE_QUERY_COUNT,
    build_geometry,
    render_review_artifacts,
    seed_queries,
)
from tracking_stage import (
    artifact_ref,
    download_verified,
    s3_settings_from_env,
    validate_stage_request,
)


HASH = "a" * 64
VIDEO_HASH = hashlib.sha256(b"video").hexdigest()


def valid_stage_request(**changes):
    run_id = "run_20260830_greenscreen"
    request = {
        "schemaVersion": 1,
        "runId": run_id,
        "stage": "tracking",
        "inputHash": HASH,
        "inputs": {
            "normalizedVideo": {
                "storage": "s3",
                "bucket": "bucket",
                "key": f"runs/{run_id}/video/{VIDEO_HASH}/normalized.mp4",
                "sha256": VIDEO_HASH,
                "sizeBytes": 5,
                "contentType": "video/mp4",
            }
        },
        "parameters": {
            "frameZeroCorners": [[10, 10], [90, 10], [90, 90], [10, 90]],
            "cornerOrder": "top-left, top-right, bottom-right, bottom-left",
            "surface": {"width": 900, "height": 1280},
            "analysis": {"width": 100, "height": 100},
            "maxReferenceAreaRatio": 4,
            "qaVersion": 1,
        },
        "expectedMedia": {"frames": 120, "fps": "24/1", "width": 100, "height": 100},
        "output": {
            "bucket": "bucket",
            "prefix": f"runs/{run_id}/tracking/{HASH}/",
        },
    }
    request.update(changes)
    return request


def test_stage_request_accepts_custom_tracking_subdirectory():
    request = valid_stage_request()
    request["output"]["prefix"] = "runs/run_20260830_greenscreen/tracking/custom-layout/"
    checked = validate_stage_request(request, "bucket")
    assert checked["output"]["prefix"].endswith("custom-layout/")


def test_stage_request_accepts_exact_contract():
    request = validate_stage_request(valid_stage_request(), "bucket")
    assert request["expectedMedia"] == {"frames": 120, "fps": "24/1", "width": 100, "height": 100}
    assert request["parameters"]["evidenceMode"] == "full"
    assert request["parameters"]["chromaTailRecovery"]["enabled"] is False
    assert request["parameters"]["queryLayout"] == "perimeter-32"


def test_stage_request_accepts_evidence_free_mode():
    request = valid_stage_request()
    request["parameters"]["evidenceMode"] = "none"
    checked = validate_stage_request(request, "bucket")
    assert checked["parameters"]["evidenceMode"] == "none"


def test_stage_request_accepts_offscreen_geometry_v2():
    request = valid_stage_request()
    request["parameters"]["qaVersion"] = 2
    checked = validate_stage_request(request, "bucket")
    assert checked["parameters"]["qaVersion"] == 2


def test_stage_request_accepts_hybrid_query_layout():
    request = valid_stage_request()
    request["parameters"]["queryLayout"] = HYBRID_QUERY_LAYOUT
    checked = validate_stage_request(request, "bucket")
    assert checked["parameters"]["queryLayout"] == HYBRID_QUERY_LAYOUT


def test_stage_request_accepts_additive_40_point_query_layout():
    request = valid_stage_request()
    request["parameters"]["queryLayout"] = ADDITIVE_QUERY_LAYOUT
    checked = validate_stage_request(request, "bucket")
    assert checked["parameters"]["queryLayout"] == ADDITIVE_QUERY_LAYOUT


def test_stage_request_accepts_explicit_chroma_tail_recovery():
    request = valid_stage_request()
    request["parameters"]["qaVersion"] = 2
    request["parameters"]["chromaTailRecovery"] = {
        "enabled": True,
        "minimumTailFrames": 16,
        "acquisitionFrames": 4,
    }
    checked = validate_stage_request(request, "bucket")
    assert checked["parameters"]["chromaTailRecovery"] == {
        "enabled": True,
        "minimumTailFrames": 16,
        "acquisitionFrames": 4,
        "scoreMargin": 0.04,
        "minimumScore": 0.90,
        "minimumPrecision": 0.95,
        "transitionFrames": 6,
    }


def test_stage_request_rejects_chroma_tail_recovery_on_qa_v1():
    request = valid_stage_request()
    request["parameters"]["chromaTailRecovery"] = {"enabled": True}
    with pytest.raises(ValueError, match="requires qaVersion 2"):
        validate_stage_request(request, "bucket")


def test_stage_request_rejects_unknown_chroma_tail_setting():
    request = valid_stage_request()
    request["parameters"]["qaVersion"] = 2
    request["parameters"]["chromaTailRecovery"] = {
        "enabled": True,
        "useRawCandidates": True,
    }
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_stage_request(request, "bucket")


def test_stage_request_accepts_eight_second_media():
    request = valid_stage_request()
    request["expectedMedia"]["frames"] = 192
    checked = validate_stage_request(request, "bucket")
    assert checked["expectedMedia"]["frames"] == 192


def test_stage_request_accepts_thirty_second_media():
    request = valid_stage_request()
    request["expectedMedia"]["frames"] = 720
    checked = validate_stage_request(request, "bucket")
    assert checked["expectedMedia"]["frames"] == 720


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.update(stage="sam"), "stage"),
        (lambda r: r["expectedMedia"].update(frames=0), "positive integer"),
        (lambda r: r["expectedMedia"].update(frames=721), "must not exceed 720"),
        (lambda r: r["output"].update(prefix="runs/other-run/tracking/hash/"), "output.prefix"),
        (lambda r: r["inputs"]["normalizedVideo"].update(key="../video.mp4"), "key"),
        (lambda r: r["parameters"].update(frameZeroCorners=[[10, 10], [90, 90], [90, 10], [10, 90]]), "convex"),
        (lambda r: r["parameters"].update(evidenceMode="brief"), "evidenceMode"),
        (lambda r: r["parameters"].update(qaVersion=3), "qaVersion"),
        (lambda r: r["parameters"].update(queryLayout="grid-32"), "queryLayout"),
    ],
)
def test_stage_request_rejects_mismatched_or_unsafe_inputs(mutate, message):
    request = valid_stage_request()
    mutate(request)
    with pytest.raises(ValueError, match=message):
        validate_stage_request(request, "bucket")


def test_secret_uses_studio_s3_variable_names_without_logging_values():
    values = {
        "S3_ACCESS_KEY_ID": "id",
        "S3_SECRET_ACCESS_KEY": "secret",
        "S3_ENDPOINT": "https://example.test",
        "S3_BUCKET": "bucket",
        "S3_REGION": "auto",
    }
    settings = s3_settings_from_env(values)
    assert settings.bucket == "bucket"
    with pytest.raises(RuntimeError, match="S3_REGION") as caught:
        s3_settings_from_env({key: value for key, value in values.items() if key != "S3_REGION"})
    assert values["S3_SECRET_ACCESS_KEY"] not in str(caught.value).split(": ")[-1]


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.upload_order = []

    def download_fileobj(self, bucket, key, stream):
        stream.write(self.objects[(bucket, key)]["body"])

    def upload_file(self, path, bucket, key, ExtraArgs):
        body = Path(path).read_bytes()
        self.objects[(bucket, key)] = {
            "body": body,
            "contentType": ExtraArgs["ContentType"],
            "metadata": ExtraArgs["Metadata"],
        }
        self.upload_order.append(key)

    def head_object(self, Bucket, Key):
        item = self.objects[(Bucket, Key)]
        return {"ContentLength": len(item["body"]), "Metadata": item["metadata"]}


def test_verified_download_and_upload_metadata(tmp_path):
    client = FakeS3()
    client.objects[("bucket", "input.mp4")] = {"body": b"video"}
    reference = {"bucket": "bucket", "key": "input.mp4", "sha256": VIDEO_HASH, "sizeBytes": 5}
    destination = download_verified(client, reference, tmp_path / "input.mp4")
    assert destination.read_bytes() == b"video"
    uploaded = artifact_ref(client, "bucket", "output.json", destination, "application/json")
    assert uploaded["sha256"] == VIDEO_HASH
    assert client.objects[("bucket", "output.json")]["metadata"]["sha256"] == VIDEO_HASH


def synthetic_tracking():
    parameters = valid_stage_request()["parameters"]
    expected = valid_stage_request()["expectedMedia"]
    queries, _ = seed_queries(parameters["frameZeroCorners"], 100, 100, 100, 100)
    reference = np.asarray([query[1:] for query in queries], dtype=np.float64)
    tracks = np.stack([reference + np.array([frame * 0.05, 0.0]) for frame in range(120)])
    result = {
        "tracks": tracks.tolist(),
        "visibility": np.ones(tracks.shape[:2], dtype=bool).tolist(),
        "model": {"name": "fixture"},
    }
    return result, parameters, expected


def synthetic_tracking_v2(offsets=None):
    parameters = valid_stage_request()["parameters"]
    parameters["qaVersion"] = 2
    expected = valid_stage_request()["expectedMedia"]
    queries, _ = seed_queries(
        parameters["frameZeroCorners"],
        100,
        100,
        100,
        100,
        include_corner_probes=True,
    )
    reference = np.asarray([query[1:] for query in queries], dtype=np.float64)
    if offsets is None:
        offsets = np.arange(120, dtype=np.float64) * 0.05
    tracks = np.stack(
        [reference + np.array([float(offset), 0.0]) for offset in offsets]
    )
    visibility = (
        (tracks[:, :, 0] >= 0)
        & (tracks[:, :, 0] < 100)
        & (tracks[:, :, 1] >= 0)
        & (tracks[:, :, 1] < 100)
    )
    result = {
        "tracks": tracks.tolist(),
        "visibility": visibility.tolist(),
        "model": {"name": "fixture"},
    }
    return result, parameters, expected


def test_hybrid_layout_has_24_edge_and_8_projected_interior_queries():
    queries, corners = seed_queries(
        [[10, 10], [90, 10], [90, 90], [10, 90]],
        100,
        100,
        100,
        100,
        include_corner_probes=True,
        query_layout=HYBRID_QUERY_LAYOUT,
    )

    assert len(queries) == 36
    assert np.allclose(np.asarray(queries[-4:])[:, 1:], corners)
    interior = np.asarray(queries[24:32])[:, 1:]
    assert np.allclose(
        interior,
        [
            [26, 36.666667], [42, 36.666667],
            [58, 36.666667], [74, 36.666667],
            [26, 63.333333], [42, 63.333333],
            [58, 63.333333], [74, 63.333333],
        ],
        atol=1e-5,
    )


def test_additive_layout_preserves_all_32_perimeter_queries_then_adds_interior():
    corners = [[10, 10], [90, 10], [90, 90], [10, 90]]
    perimeter, _ = seed_queries(corners, 100, 100, 100, 100)
    additive, reference = seed_queries(
        corners,
        100,
        100,
        100,
        100,
        include_corner_probes=True,
        query_layout=ADDITIVE_QUERY_LAYOUT,
    )
    hybrid, _ = seed_queries(
        corners,
        100,
        100,
        100,
        100,
        query_layout=HYBRID_QUERY_LAYOUT,
    )

    assert len(additive) == 44
    assert additive[:32] == perimeter
    assert np.allclose(additive[32:40], hybrid[24:32])
    assert np.allclose(np.asarray(additive[40:])[:, 1:], reference)


def test_geometry_slices_40_plane_queries_before_four_v2_corner_probes():
    parameters = valid_stage_request()["parameters"]
    parameters["qaVersion"] = 2
    parameters["queryLayout"] = ADDITIVE_QUERY_LAYOUT
    expected = valid_stage_request()["expectedMedia"]
    queries, _ = seed_queries(
        parameters["frameZeroCorners"],
        100,
        100,
        100,
        100,
        include_corner_probes=True,
        query_layout=ADDITIVE_QUERY_LAYOUT,
    )
    reference = np.asarray([query[1:] for query in queries], dtype=np.float64)
    tracks = np.repeat(reference[None, :, :], expected["frames"], axis=0)
    result = {
        "tracks": tracks.tolist(),
        "visibility": np.ones(tracks.shape[:2], dtype=bool).tolist(),
        "model": {"name": "fixture"},
    }

    coordinates, metrics, _, stable = build_geometry(result, parameters, expected)

    assert coordinates["settings"]["queryLayout"] == ADDITIVE_QUERY_LAYOUT
    assert coordinates["settings"]["planePoints"] == 40
    assert coordinates["settings"]["points"] == 44
    assert metrics["directGood"] == 120
    assert stable.shape == (120, 4, 2)


def test_geometry_has_exact_frame_contract_and_finite_homographies():
    result, parameters, expected = synthetic_tracking()
    coordinates, metrics, suspects, stable = build_geometry(result, parameters, expected)
    assert len(coordinates["frames"]) == 120
    assert metrics["frameCount"] == 120
    assert stable.shape == (120, 4, 2)
    for index, frame in enumerate(coordinates["frames"]):
        assert frame["frame"] == index
        assert np.isfinite(np.asarray(frame["homography"])).all()
    assert 1 <= len(suspects) <= 10


def test_geometry_uses_dynamic_expected_frame_count():
    result, parameters, expected = synthetic_tracking()
    expected["frames"] = 192
    reference = np.asarray(result["tracks"][0], dtype=np.float64)
    tracks = np.stack(
        [reference + np.array([frame * 0.02, 0.0]) for frame in range(192)]
    )
    result["tracks"] = tracks.tolist()
    result["visibility"] = np.ones(tracks.shape[:2], dtype=bool).tolist()
    coordinates, metrics, _, stable = build_geometry(result, parameters, expected)
    assert len(coordinates["frames"]) == 192
    assert metrics["frameCount"] == 192
    assert stable.shape == (192, 4, 2)


def test_geometry_matches_retained_local_opencv_golden_values():
    """Golden values were captured from apps/weather homography_tracking.py."""
    result, parameters, expected = synthetic_tracking()
    coordinates, _, _, _ = build_geometry(result, parameters, expected)
    assert coordinates["summary"] == {
        "frameCount": 120, "goodFrames": 120, "recoveredFrames": []
    }
    assert coordinates["frames"][0]["homography"] == [
        [0.088889, 0.0, 10.0], [0.0, 0.0625, 10.0], [0.0, 0.0, 1.0]
    ]
    assert coordinates["frames"][60]["corners"] == [
        [13.0, 10.0], [93.0, 10.0], [93.0, 90.0], [13.0, 90.0]
    ]
    assert coordinates["frames"][119]["homography"] == [
        [0.088889, 0.0, 15.95], [0.0, 0.0625, 10.0], [0.0, 0.0, 1.0]
    ]


def test_v2_direct_geometry_matches_v1_numeric_output():
    v1_result, v1_parameters, expected = synthetic_tracking()
    v1_coordinates, _, _, _ = build_geometry(v1_result, v1_parameters, expected)
    v2_result, v2_parameters, _ = synthetic_tracking_v2()
    v2_coordinates, _, _, _ = build_geometry(v2_result, v2_parameters, expected)

    assert v2_coordinates["version"] == 2
    assert v2_coordinates["settings"]["points"] == PLANE_QUERY_COUNT + 4
    assert [frame["corners"] for frame in v2_coordinates["frames"]] == [
        frame["corners"] for frame in v1_coordinates["frames"]
    ]
    assert all(
        frame["resolutionSource"] == "direct"
        for frame in v2_coordinates["frames"]
    )


def test_v2_uses_two_corner_probes_for_coherent_partial_recovery():
    result, parameters, expected = synthetic_tracking_v2()
    visibility = np.asarray(result["visibility"], dtype=bool)
    visibility[60, :PLANE_QUERY_COUNT] = False
    visibility[60, PLANE_QUERY_COUNT:] = [False, True, True, False]
    result["visibility"] = visibility.tolist()

    coordinates, metrics, _, stable = build_geometry(result, parameters, expected)
    frame = coordinates["frames"][60]
    assert frame["resolutionSource"] == "partial-affine"
    assert frame["cornerSources"] == {
        "tl": "predicted", "tr": "tracked", "br": "tracked", "bl": "predicted"
    }
    assert metrics["partialRecovered"] == 1
    assert np.allclose(stable[60], np.asarray([[13, 10], [93, 10], [93, 90], [13, 90]]), atol=0.1)


def test_v2_rejects_a_single_or_implausible_corner_probe():
    result, parameters, expected = synthetic_tracking_v2()
    visibility = np.asarray(result["visibility"], dtype=bool)
    tracks = np.asarray(result["tracks"], dtype=np.float64)
    visibility[60, :PLANE_QUERY_COUNT] = False
    visibility[60, PLANE_QUERY_COUNT:] = [True, True, False, False]
    tracks[60, PLANE_QUERY_COUNT] += np.array([60.0, 0.0])
    result["visibility"] = visibility.tolist()
    result["tracks"] = tracks.tolist()

    coordinates, metrics, _, _ = build_geometry(result, parameters, expected)
    frame = coordinates["frames"][60]
    assert frame["resolutionSource"] == "bridged"
    assert all(source == "predicted" for source in frame["cornerSources"].values())
    assert metrics["partialRecovered"] == 0


def test_v2_bridges_fully_hidden_gap_to_reentry_anchor():
    result, parameters, expected = synthetic_tracking_v2()
    visibility = np.asarray(result["visibility"], dtype=bool)
    visibility[50:60] = False
    result["visibility"] = visibility.tolist()

    coordinates, metrics, _, _ = build_geometry(result, parameters, expected)
    assert all(
        coordinates["frames"][index]["resolutionSource"] == "bridged"
        for index in range(50, 60)
    )
    assert coordinates["frames"][60]["resolutionSource"] == "direct"
    assert metrics["bridged"] == 10


def test_v2_reacquires_after_hidden_gap_with_large_coherent_translation():
    result, parameters, expected = synthetic_tracking_v2()
    visibility = np.asarray(result["visibility"], dtype=bool)
    tracks = np.asarray(result["tracks"], dtype=np.float64)
    visibility[50:60] = False
    visibility[60, :PLANE_QUERY_COUNT] = False
    visibility[60, PLANE_QUERY_COUNT:] = [False, True, True, False]
    tracks[60, PLANE_QUERY_COUNT + 1] += np.array([-35.0, 0.0])
    tracks[60, PLANE_QUERY_COUNT + 2] += np.array([-35.0, 0.0])
    result["visibility"] = visibility.tolist()
    result["tracks"] = tracks.tolist()

    coordinates, metrics, _, _ = build_geometry(result, parameters, expected)
    assert coordinates["frames"][60]["resolutionSource"] == "partial-affine"
    assert metrics["partialRecovered"] == 1


def test_v2_predicts_then_holds_unanchored_tail():
    result, parameters, expected = synthetic_tracking_v2()
    visibility = np.asarray(result["visibility"], dtype=bool)
    visibility[90:] = False
    result["visibility"] = visibility.tolist()

    coordinates, metrics, _, _ = build_geometry(result, parameters, expected)
    assert coordinates["frames"][90]["resolutionSource"] == "predicted"
    assert coordinates["frames"][90 + MAX_TAIL_PREDICTION_FRAMES - 1]["resolutionSource"] == "predicted"
    assert coordinates["frames"][90 + MAX_TAIL_PREDICTION_FRAMES]["resolutionSource"] == "held"
    assert metrics["predictionOnly"] == MAX_TAIL_PREDICTION_FRAMES
    assert metrics["held"] == 120 - 90 - MAX_TAIL_PREDICTION_FRAMES


def test_v2_emits_unclamped_offscreen_corners():
    offsets = np.concatenate(
        [np.arange(55, dtype=np.float64) * 2.0, np.full(65, 108.0)]
    )
    result, parameters, expected = synthetic_tracking_v2(offsets)
    coordinates, metrics, _, _ = build_geometry(result, parameters, expected)

    assert max(point[0] for point in coordinates["frames"][50]["corners"]) > 100
    assert coordinates["frames"][50]["offscreenCornerCount"] > 0
    assert metrics["fullyOffscreen"] > 0


def test_review_sheets_cover_all_frames_and_suspects_are_annotated(tmp_path):
    result, parameters, expected = synthetic_tracking()
    _, _, suspects, stable = build_geometry(result, parameters, expected)
    frames = np.zeros((120, 100, 100, 3), dtype=np.uint8)
    video, sheets, suspect_paths = render_review_artifacts(
        frames, stable, parameters["analysis"], suspects, tmp_path
    )
    assert video.stat().st_size > 0
    assert [path.name for path in sheets] == [
        "overview-000-029.jpg", "overview-030-059.jpg",
        "overview-060-089.jpg", "overview-090-119.jpg",
    ]
    assert all(path.stat().st_size > 0 for path in sheets)
    assert set(suspect_paths) == {suspect["frame"] for suspect in suspects}


def test_review_sheets_cover_more_than_120_frames_and_pad_final_sheet(tmp_path):
    frame_count = 243
    frames = np.zeros((frame_count, 100, 100, 3), dtype=np.uint8)
    corners = np.asarray(
        [[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float64
    )
    stable = np.repeat(corners[None, :, :], frame_count, axis=0)

    video, sheets, suspect_paths = render_review_artifacts(
        frames,
        stable,
        {"width": 100, "height": 100},
        [],
        tmp_path,
    )

    assert video.stat().st_size > 0
    assert [path.name for path in sheets] == [
        "overview-000-029.jpg",
        "overview-030-059.jpg",
        "overview-060-089.jpg",
        "overview-090-119.jpg",
        "overview-120-149.jpg",
        "overview-150-179.jpg",
        "overview-180-209.jpg",
        "overview-210-239.jpg",
        "overview-240-242.jpg",
    ]
    assert all(path.stat().st_size > 0 for path in sheets)
    assert suspect_paths == {}
