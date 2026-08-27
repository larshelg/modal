import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

import app as modal_app
from app import (
    APP_NAME,
    MASK_CODEC,
    MASK_FILENAME,
    MASK_PIXEL_FORMAT,
    _finish_mask_encoder,
    _start_mask_encoder,
    _verify_mask_video,
)


def test_modal_image_contract():
    assert APP_NAME == "opencv-mask-modal-app"
    assert MASK_FILENAME == "occlusion-mask.mkv"
    assert MASK_CODEC == "ffv1"
    assert MASK_PIXEL_FORMAT == "gray"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is not installed on the test host",
)
def test_lossless_mask_encoder_preserves_media_contract(tmp_path):
    output = tmp_path / MASK_FILENAME
    encoder = _start_mask_encoder(output, width=16, height=8, fps=24.0)
    assert encoder.stdin is not None
    for frame_index in range(3):
        mask = np.zeros((8, 16), dtype=np.uint8)
        mask[:, frame_index : frame_index + 4] = 255
        encoder.stdin.write(mask.tobytes())
    _finish_mask_encoder(encoder)
    _verify_mask_video(output, width=16, height=8, fps=24.0, frame_count=3)
    assert output.stat().st_size > 0


class FakeS3:
    def __init__(self, root: Path):
        self.root = root
        self.metadata = {}
        self.uploads = []

    def _path(self, key):
        return self.root / key

    def head_object(self, *, Bucket, Key):
        path = self._path(Key)
        return {
            "ContentLength": path.stat().st_size,
            "Metadata": self.metadata.get(Key, {}),
        }

    def download_fileobj(self, bucket, key, stream):
        stream.write(self._path(key).read_bytes())

    def upload_file(self, source, bucket, key, ExtraArgs):
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        self.metadata[key] = ExtraArgs["Metadata"]
        self.uploads.append(key)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is not installed on the test host",
)
def test_complete_job_writes_mask_debug_bundle_and_result_last(tmp_path, monkeypatch):
    bucket_root = tmp_path / "bucket"
    bucket_root.mkdir()
    video_path = bucket_root / "inputs/source.avi"
    video_path.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (64, 48)
    )
    if not writer.isOpened():
        pytest.skip("test host cannot encode an MJPG source")
    for _ in range(3):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :] = [0, 255, 0]
        frame[16:32, 28:36] = [0, 0, 255]
        writer.write(frame)
    writer.release()

    tracking_path = bucket_root / "inputs/tracking.json"
    tracking_path.write_text(
        json.dumps(
            {
                "source": {"width": 64, "height": 48, "fps": 24},
                "frames": [
                    {
                        "frame": 0,
                        "corners": [[4, 4], [59, 4], [59, 43], [4, 43]],
                    },
                    {
                        "frame": 1,
                        "corners": [[-10, 4], [70, 4], [70, 43], [-10, 43]],
                    },
                    {
                        "frame": 2,
                        "corners": [[80, 4], [120, 4], [120, 43], [80, 43]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    fake_s3 = FakeS3(bucket_root)
    settings = {
        "S3_ACCESS_KEY_ID": "test",
        "S3_SECRET_ACCESS_KEY": "test",
        "S3_ENDPOINT": "https://s3.invalid",
        "S3_BUCKET": "studio",
        "S3_REGION": "auto",
    }
    monkeypatch.setattr(modal_app, "_s3_settings", lambda: settings)
    monkeypatch.setattr(modal_app, "_s3_client", lambda ignored: fake_s3)

    result = modal_app._process(
        {
            "video_url": "s3://studio/inputs/source.avi",
            "tracking_url": "s3://studio/inputs/tracking.json",
            "output_prefix": "s3://studio/jobs/test/",
            "debug_frames": [0, 1, 2],
        }
    )

    assert result["status"] == "completed"
    assert result["frames"] == 3
    assert result["mask_url"] == "s3://studio/jobs/test/occlusion-mask.mkv"
    assert result["mask_mode"] == "hard"
    assert result["partial_replacement_pixels"] == 0
    assert result["partial_offscreen_frames"] == [1]
    assert result["fully_offscreen_frames"] == [2]
    assert result["maximum_offscreen_corner_count"] == 4
    assert result["visible_polygon_pixels_min"] == 0
    assert 0 < result["green_coverage_min"] < 1
    assert (bucket_root / "jobs/test/occlusion-mask.mkv").is_file()
    assert (bucket_root / "jobs/test/debug/frame_000000_overlay.jpg").is_file()
    assert (bucket_root / "jobs/test/debug/frame_000002_polygon.jpg").is_file()
    assert fake_s3.uploads[-1] == "jobs/test/result.json"
