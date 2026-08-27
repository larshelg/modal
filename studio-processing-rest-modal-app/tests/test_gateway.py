from __future__ import annotations

from copy import deepcopy

import pytest

from studio_gateway import (
    IdempotencyConflict,
    RequestError,
    StageExecutionError,
    execute_pipeline,
    existing_job_for_request,
    idempotency_slot,
    new_job,
    request_digest,
    validate_pipeline_request,
)


def stage_request(stage: str, *, evidence_mode: str = "full") -> dict:
    digest = "a" * 64 if stage == "tracking" else "b" * 64
    return {
        "schemaVersion": 1,
        "stage": stage,
        "runId": "rest-test",
        "inputHash": digest,
        "inputs": {
            "normalizedVideo": {
                "storage": "s3",
                "bucket": "studio",
                "key": "studio-experiments/rest-test/h3/source.mp4",
                "sha256": "c" * 64,
                "sizeBytes": 1234,
                "contentType": "video/mp4",
            }
        },
        "parameters": {"evidenceMode": evidence_mode},
        "expectedMedia": {
            "frames": 120,
            "fps": "24/1",
            "width": 1080,
            "height": 1920,
        },
        "output": {
            "bucket": "studio",
            "prefix": f"studio-experiments/rest-test/{stage}/{digest}/",
        },
    }


def pipeline_request(*, evidence_mode: str = "full") -> dict:
    return {
        "schemaVersion": 1,
        "tracking": stage_request("tracking", evidence_mode=evidence_mode),
        "sam": stage_request("sam", evidence_mode=evidence_mode),
    }


def test_validates_shared_stage_relationships():
    request = pipeline_request(evidence_mode="none")
    assert validate_pipeline_request(request) == request

    mismatched = deepcopy(request)
    mismatched["sam"]["expectedMedia"]["frames"] = 121
    with pytest.raises(RequestError, match="expectedMedia"):
        validate_pipeline_request(mismatched)

    mismatched = deepcopy(request)
    mismatched["sam"]["parameters"]["evidenceMode"] = "full"
    with pytest.raises(RequestError, match="evidenceMode"):
        validate_pipeline_request(mismatched)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda body: body.update(extra=True), "unsupported request fields"),
        (lambda body: body["tracking"].update(stage="sam"), "tracking.stage"),
        (
            lambda body: body["sam"].update(inputHash="not-a-digest"),
            "sam.inputHash",
        ),
        (
            lambda body: body["sam"]["output"].update(
                prefix=body["tracking"]["output"]["prefix"]
            ),
            "output prefixes",
        ),
    ],
)
def test_rejects_invalid_requests(mutation, message):
    body = pipeline_request()
    mutation(body)
    with pytest.raises(RequestError, match=message):
        validate_pipeline_request(body)


def test_execution_is_strictly_tracking_then_sam():
    calls = []
    updates = []

    def invoke(stage, request):
        calls.append(stage)
        return {"stage": request["stage"], "ok": True}

    def update(stage, status, payload):
        updates.append((stage, status, payload))

    results = execute_pipeline(pipeline_request(), invoke, update)
    assert calls == ["tracking", "sam"]
    assert [(stage, status) for stage, status, _ in updates] == [
        ("tracking", "running"),
        ("tracking", "succeeded"),
        ("sam", "running"),
        ("sam", "succeeded"),
    ]
    assert results["tracking"]["stage"] == "tracking"
    assert results["sam"]["stage"] == "sam"


def test_tracking_failure_prevents_sam_dispatch():
    calls = []
    updates = []

    def invoke(stage, _request):
        calls.append(stage)
        raise RuntimeError("private worker details")

    with pytest.raises(StageExecutionError) as raised:
        execute_pipeline(
            pipeline_request(),
            invoke,
            lambda stage, status, payload: updates.append((stage, status, payload)),
        )

    assert raised.value.stage == "tracking"
    assert calls == ["tracking"]
    assert updates[-1][2]["message"].endswith("inspect the Modal worker logs.")
    assert "private worker details" not in str(updates[-1][2])


def test_idempotency_uses_hashed_slots_and_request_digest():
    request = pipeline_request()
    digest = request_digest(request)
    record = new_job("job-1", request, digest)
    assert idempotency_slot("secret-client-key").startswith("idempotency:")
    assert "secret-client-key" not in idempotency_slot("secret-client-key")
    assert existing_job_for_request(record, digest) is record
    with pytest.raises(IdempotencyConflict):
        existing_job_for_request(record, "f" * 64)
