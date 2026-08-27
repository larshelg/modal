import pytest
import numpy as np

from app import (
    APP_NAME,
    CHECKPOINT_NAME,
    FALLBACK_GPU_TYPE,
    MASK_CODEC,
    MASK_PIXEL_FORMAT,
    MODEL_REVISION,
    PRIMARY_GPU_TYPE,
    SAM_COMMIT,
    _as_numpy,
    apply_init_state_compat,
    is_cuda_oom,
    model_metadata,
    select_continuous_mask,
    validate_request,
    validate_video_bytes,
)


def valid_request(**overrides):
    request = {
        "anchor_frame": 90,
        "box_xywh": [0.59, 0.18, 0.39, 0.80],
        "text_prompt": "person",
        "output_prob_threshold": 0.5,
        "include_debug_png_zip": False,
        "include_preview_video": True,
    }
    request.update(overrides)
    return request


def test_deployment_model_and_mask_contract_are_pinned():
    assert APP_NAME == "sam3-modal-app"
    assert len(SAM_COMMIT) == 40
    assert len(MODEL_REVISION) == 40
    assert CHECKPOINT_NAME == "sam3.1_multiplex.pt"
    assert PRIMARY_GPU_TYPE == "L40S"
    assert FALLBACK_GPU_TYPE == "A100-80GB"
    assert MASK_CODEC == "ffv1"
    assert MASK_PIXEL_FORMAT == "gray"
    assert model_metadata()["commit"] == SAM_COMMIT
    assert model_metadata()["checkpointRevision"] == MODEL_REVISION


def test_request_preserves_the_normalized_box():
    normalized = validate_request(valid_request())
    assert normalized == valid_request()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"anchor_frame": False}, "integer"),
        ({"anchor_frame": -1}, "between"),
        ({"box_xywh": [0.1, 0.1, 0.2]}, "must be"),
        ({"box_xywh": [0.9, 0.1, 0.2, 0.2]}, "inside"),
        ({"box_xywh": [0.1, 0.1, 0, 0.2]}, "positive"),
        ({"text_prompt": ""}, "non-empty"),
        ({"text_prompt": 42}, "non-empty"),
        ({"output_prob_threshold": 0}, "between"),
        ({"output_prob_threshold": True}, "numeric"),
        ({"include_debug_png_zip": "yes"}, "boolean"),
        ({"include_preview_video": "yes"}, "boolean"),
        ({"unexpected": True}, "unsupported"),
    ],
)
def test_request_rejects_invalid_inputs(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_request(valid_request(**overrides))


def test_video_payload_validation():
    validate_video_bytes(b"video")
    with pytest.raises(ValueError, match="must not be empty"):
        validate_video_bytes(b"")
    with pytest.raises(ValueError, match="must be bytes"):
        validate_video_bytes(bytearray(b"video"))


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory. Tried to allocate 2.00 GiB",
        "torch.OutOfMemoryError: allocation failed",
        "cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
    ],
)
def test_only_recognized_cuda_ooms_trigger_fallback(message):
    assert is_cuda_oom(RuntimeError(message))


def test_non_oom_errors_do_not_trigger_fallback():
    assert not is_cuda_oom(RuntimeError("checkpoint is missing"))


def test_bfloat16_outputs_are_converted_to_numpy_float32():
    class FakeBfloat16Tensor:
        dtype = "torch.bfloat16"

        def detach(self):
            return self

        def float(self):
            self.dtype = "torch.float32"
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray([0.75], dtype=np.float32)

    converted = _as_numpy(FakeBfloat16Tensor())
    assert converted.dtype == np.float32
    assert converted.tolist() == [0.75]


def test_init_state_compat_only_discards_false_offload_flag():
    calls = []

    class Model:
        def init_state(self, resource_path, offload_video_to_cpu=False):
            calls.append((resource_path, offload_video_to_cpu))
            return {"ready": True}

    class Predictor:
        model = Model()

    predictor = Predictor()
    assert apply_init_state_compat(predictor)
    assert predictor.model.init_state(
        "video.mp4", offload_video_to_cpu=True, offload_state_to_cpu=False
    ) == {"ready": True}
    assert calls == [("video.mp4", True)]
    with pytest.raises(ValueError, match="does not support"):
        predictor.model.init_state("video.mp4", offload_state_to_cpu=True)


def test_continuity_selection_follows_mask_when_object_id_changes():
    reference = np.zeros((6, 6), dtype=bool)
    reference[1:5, 2:5] = True
    distractor = np.zeros((6, 6), dtype=bool)
    distractor[0:2, 0:2] = True
    presenter = np.zeros((6, 6), dtype=bool)
    presenter[1:5, 3:6] = True
    object_id, mask, iou = select_continuous_mask(
        {
            "out_obj_ids": np.asarray([4, 19]),
            "out_binary_masks": np.stack([distractor, presenter]),
        },
        reference,
    )
    assert object_id == 19
    assert np.array_equal(mask, presenter)
    assert iou > 0.4


def test_continuity_selection_rejects_unrelated_masks():
    reference = np.zeros((6, 6), dtype=bool)
    reference[0:2, 0:2] = True
    unrelated = np.zeros((6, 6), dtype=bool)
    unrelated[4:6, 4:6] = True
    object_id, mask, iou = select_continuous_mask(
        {
            "out_obj_ids": np.asarray([8]),
            "out_binary_masks": np.stack([unrelated]),
        },
        reference,
    )
    assert object_id is None
    assert mask is None
    assert iou == 0
