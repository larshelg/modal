import hashlib
from pathlib import Path

import numpy as np
import pytest

from app import _encode_mask_video
from sam_review import (
    build_metrics,
    render_review_artifacts,
    validate_masks,
    verify_mask_video,
)
from sam_stage import (
    artifact_ref,
    download_verified,
    s3_settings_from_env,
    validate_stage_request,
)


HASH = "a" * 64
VIDEO_HASH = hashlib.sha256(b"video").hexdigest()


def valid_stage_request():
    run_id = "natural-20260823-120000"
    return {
        "schemaVersion": 1,
        "runId": run_id,
        "stage": "sam",
        "inputHash": HASH,
        "inputs": {
            "normalizedVideo": {
                "storage": "s3", "bucket": "bucket",
                "key": f"studio-experiments/{run_id}/h3/hash/normalized.mp4",
                "sha256": VIDEO_HASH, "sizeBytes": 5, "contentType": "video/mp4",
            }
        },
        "parameters": {
            "anchorFrame": 61,
            "boxXywh": [0.31, 0.18, 0.41, 0.77],
            "textPrompt": "person",
            "outputProbabilityThreshold": 0.5,
            "qaVersion": 1,
        },
        "expectedMedia": {"frames": 120, "fps": "24/1", "width": 96, "height": 64},
        "output": {
            "bucket": "bucket",
            "prefix": f"studio-experiments/{run_id}/sam/{HASH}/",
        },
    }


def test_stage_request_accepts_exact_contract():
    request = validate_stage_request(valid_stage_request(), "bucket")
    assert request["parameters"]["anchorFrame"] == 61
    assert request["parameters"]["evidenceMode"] == "full"


def test_stage_request_accepts_evidence_free_mode():
    request = valid_stage_request()
    request["parameters"]["evidenceMode"] = "none"
    checked = validate_stage_request(request, "bucket")
    assert checked["parameters"]["evidenceMode"] == "none"


def test_stage_request_accepts_eight_second_media():
    request = valid_stage_request()
    request["expectedMedia"]["frames"] = 192
    checked = validate_stage_request(request, "bucket")
    assert checked["expectedMedia"]["frames"] == 192


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.update(stage="tracking"), "stage"),
        (lambda r: r["expectedMedia"].update(frames=0), "positive integer"),
        (lambda r: r["parameters"].update(anchorFrame=120), "anchorFrame"),
        (lambda r: r["parameters"].update(boxXywh=[0.9, 0.1, 0.2, 0.2]), "boxXywh"),
        (lambda r: r["output"].update(prefix="studio-experiments/other/"), "output.prefix"),
        (lambda r: r["inputs"]["normalizedVideo"].update(key="../video.mp4"), "key"),
        (lambda r: r["parameters"].update(evidenceMode="brief"), "evidenceMode"),
    ],
)
def test_stage_request_rejects_invalid_or_unsafe_inputs(change, message):
    request = valid_stage_request()
    change(request)
    with pytest.raises(ValueError, match=message):
        validate_stage_request(request, "bucket")


def test_secret_uses_studio_s3_names_without_exposing_values():
    values = {
        "S3_ACCESS_KEY_ID": "id", "S3_SECRET_ACCESS_KEY": "private-value",
        "S3_ENDPOINT": "https://example.test", "S3_BUCKET": "bucket",
        "S3_REGION": "auto",
    }
    assert s3_settings_from_env(values).bucket == "bucket"
    with pytest.raises(RuntimeError) as caught:
        s3_settings_from_env({key: value for key, value in values.items() if key != "S3_REGION"})
    assert values["S3_SECRET_ACCESS_KEY"] not in str(caught.value)


class FakeS3:
    def __init__(self):
        self.objects = {}

    def download_fileobj(self, bucket, key, stream):
        stream.write(self.objects[(bucket, key)]["body"])

    def upload_file(self, path, bucket, key, ExtraArgs):
        body = Path(path).read_bytes()
        self.objects[(bucket, key)] = {"body": body, "metadata": ExtraArgs["Metadata"]}

    def head_object(self, Bucket, Key):
        item = self.objects[(Bucket, Key)]
        return {"ContentLength": len(item["body"]), "Metadata": item["metadata"]}


def test_s3_transfer_verifies_size_hash_and_metadata(tmp_path):
    client = FakeS3()
    client.objects[("bucket", "input.mp4")] = {"body": b"video"}
    reference = {"bucket": "bucket", "key": "input.mp4", "sha256": VIDEO_HASH, "sizeBytes": 5}
    path = download_verified(client, reference, tmp_path / "input.mp4")
    uploaded = artifact_ref(client, "bucket", "mask.mkv", path, "video/x-matroska")
    assert uploaded["sha256"] == VIDEO_HASH
    assert client.objects[("bucket", "mask.mkv")]["metadata"]["sha256"] == VIDEO_HASH


def synthetic_masks():
    masks = []
    for frame in range(120):
        mask = np.zeros((64, 96), dtype=bool)
        x = 20 + frame // 30
        mask[10:58, x:x + 24] = True
        masks.append(mask)
    metadata = {
        "source": {"width": 96, "height": 64, "fpsRatio": "24/1", "frameCount": 120},
        "tracking": {"selectedObjectIds": [7] * 120},
        "compatibility": {"discardFalseOffloadStateToCpu": True},
        "dispatch": {"gpuAttempts": ["L40S"]},
    }
    return masks, metadata


def test_mask_metrics_are_complete_deterministic_and_nonempty():
    masks, metadata = synthetic_masks()
    metrics, first = build_metrics(masks, metadata)
    _, second = build_metrics(masks, metadata)
    assert metrics["frameCount"] == 120
    assert metrics["nonemptyFrames"] == 120
    assert metrics["missingFrames"] == []
    assert len(metrics["perFrame"]) == 120
    assert first == second
    assert 1 <= len(first) <= 10


def test_empty_mask_is_a_structural_failure():
    masks, _ = synthetic_masks()
    masks[31][:] = False
    with pytest.raises(ValueError, match="31"):
        validate_masks(masks, 96, 64, 120)


def test_mask_metrics_use_dynamic_source_frame_count():
    masks, metadata = synthetic_masks()
    masks.extend(masks[:72])
    metadata["source"]["frameCount"] = 192
    metadata["tracking"]["selectedObjectIds"] = [7] * 192
    metrics, _ = build_metrics(masks, metadata)
    assert metrics["frameCount"] == 192
    assert metrics["nonemptyFrames"] == 192


def test_lossless_mask_and_review_contract(tmp_path):
    masks, metadata = synthetic_masks()
    mask_path = tmp_path / "mask.mkv"
    _encode_mask_video(masks, metadata["source"], mask_path)
    stream = verify_mask_video(mask_path, metadata["source"])
    assert stream["codec_name"] == "ffv1"
    assert stream["pix_fmt"] == "gray"

    metrics, suspects = build_metrics(masks, metadata)
    frames = np.zeros((120, 64, 96, 3), dtype=np.uint8)
    sheets, suspect_paths = render_review_artifacts(frames, masks, metrics, suspects, tmp_path / "review")
    assert [path.name for path in sheets] == [
        "overview-000-029.jpg", "overview-030-059.jpg",
        "overview-060-089.jpg", "overview-090-119.jpg",
    ]
    assert all(path.stat().st_size > 0 for path in sheets)
    assert set(suspect_paths) == {item["frame"] for item in suspects}
