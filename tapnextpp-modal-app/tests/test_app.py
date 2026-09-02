import pytest

from app import (
    APP_NAME,
    CHECKPOINT_GENERATION,
    CHECKPOINT_MD5,
    CHECKPOINT_SIZE,
    MAX_FRAMES,
    MODEL_INPUT_RESOLUTION,
    TAPNET_COMMIT,
    _parameter_fingerprint,
    model_metadata,
    validate_request,
)


def valid_request(**overrides):
    request = {
        "analysis_width": 384,
        "analysis_height": 688,
        "queries": [
            [0, 13.0, 64.0],
            [0, 220.0, 64.0],
            [0, 220.0, 490.0],
            [0, 13.0, 490.0],
        ],
        "backward_tracking": False,
    }
    request.update(overrides)
    return request


def test_deployment_and_model_are_pinned():
    assert APP_NAME == "tapnextpp-modal-app"
    assert len(TAPNET_COMMIT) == 40
    assert len(CHECKPOINT_GENERATION) > 10
    assert len(CHECKPOINT_MD5) == 32
    assert CHECKPOINT_SIZE > 2_000_000_000
    assert MODEL_INPUT_RESOLUTION == 512
    assert MAX_FRAMES == 720
    assert model_metadata()["commit"] == TAPNET_COMMIT
    assert model_metadata()["license"] == "Apache-2.0"


def test_request_preserves_query_order_and_normalizes_numbers():
    normalized = validate_request(valid_request())
    assert normalized["analysis_width"] == 384
    assert normalized["analysis_height"] == 688
    assert normalized["queries"][0] == [0.0, 13.0, 64.0]
    assert normalized["queries"][-1] == [0.0, 13.0, 490.0]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"analysis_width": 0}, "analysis_width"),
        ({"queries": []}, "between"),
        ({"queries": [[0, -1, 20]] * 4}, "outside"),
        ({"queries": [[False, 1, 2]] * 4}, "must be 0"),
        ({"queries": [[1, 1, 2]] * 4}, "must be 0"),
        ({"backward_tracking": "yes"}, "boolean"),
        ({"backward_tracking": True}, "forward tracking only"),
        ({"unexpected": True}, "unsupported"),
    ],
)
def test_request_rejects_invalid_inputs(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_request(valid_request(**overrides))


def test_stage_cache_fingerprint_changes_with_chroma_policy():
    baseline = {
        "parameters": {
            "qaVersion": 2,
            "chromaTailRecovery": {"enabled": False},
        }
    }
    assisted = {
        "parameters": {
            "qaVersion": 2,
            "chromaTailRecovery": {"enabled": True},
        }
    }
    assert _parameter_fingerprint(baseline) != _parameter_fingerprint(assisted)
