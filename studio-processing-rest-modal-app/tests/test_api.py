from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from fastapi.testclient import TestClient

from app import create_web_app
from studio_gateway import idempotency_slot, utc_now
from test_gateway import pipeline_request


class MemoryStore:
    def __init__(self):
        self.values = {}

    def get(self, key, default=None):
        return deepcopy(self.values.get(key, default))

    def put(self, key, value):
        self.values[key] = deepcopy(value)


@dataclass
class SpawnedCall:
    object_id: str


class PendingCall:
    def get(self, timeout):
        assert timeout == 0
        raise TimeoutError


def client_fixture():
    store = MemoryStore()
    submissions = []

    def spawn(job_id, request):
        submissions.append((job_id, request))
        return SpawnedCall("fc-1")

    client = TestClient(create_web_app(store, spawn, lambda _call_id: PendingCall()))
    return client, store, submissions


def test_health_describes_sequential_workers():
    client, _, _ = client_fixture()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["execution"] == "sequential"
    assert set(response.json()["workers"]) == {"tracking", "sam"}


def test_submission_requires_idempotency_key_and_valid_request():
    client, _, submissions = client_fixture()
    response = client.post("/v1/jobs", json=pipeline_request())
    assert response.status_code == 400
    assert submissions == []

    response = client.post(
        "/v1/jobs",
        headers={"Idempotency-Key": "test-1"},
        json={"schemaVersion": 1},
    )
    assert response.status_code == 422
    assert submissions == []


def test_submit_poll_and_idempotent_replay():
    client, store, submissions = client_fixture()
    headers = {"Idempotency-Key": "test-1"}
    response = client.post("/v1/jobs", headers=headers, json=pipeline_request())
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["currentStage"] == "tracking"
    assert len(submissions) == 1

    poll = client.get(f"/v1/jobs/{body['id']}")
    assert poll.status_code == 200
    assert poll.json()["status"] == "queued"
    assert "callId" not in poll.json()
    assert "request" not in poll.json()

    replay = client.post("/v1/jobs", headers=headers, json=pipeline_request())
    assert replay.status_code == 202
    assert replay.json()["id"] == body["id"]
    assert len(submissions) == 1

    different = pipeline_request()
    different["sam"]["inputHash"] = "d" * 64
    conflict = client.post("/v1/jobs", headers=headers, json=different)
    assert conflict.status_code == 409
    assert len(submissions) == 1

    assert store.get(idempotency_slot("test-1")) == body["id"]


def test_poll_returns_worker_results_without_internal_fields():
    client, store, _ = client_fixture()
    submitted = client.post(
        "/v1/jobs",
        headers={"Idempotency-Key": "test-results"},
        json=pipeline_request(evidence_mode="none"),
    ).json()
    record = store.get(submitted["id"])
    timestamp = utc_now()
    record.update(
        status="succeeded",
        currentStage="completed",
        result={
            "tracking": {"stage": "tracking", "artifacts": {}},
            "sam": {"stage": "sam", "artifacts": {}},
        },
        completedAt=timestamp,
        updatedAt=timestamp,
    )
    record["stages"] = {
        "tracking": {"status": "succeeded", "result": record["result"]["tracking"]},
        "sam": {"status": "succeeded", "result": record["result"]["sam"]},
    }
    store.put(submitted["id"], record)

    response = client.get(f"/v1/jobs/{submitted['id']}")
    assert response.status_code == 200
    assert response.json()["result"] == record["result"]
    assert "requestDigest" not in response.json()


def test_unknown_job_is_404():
    client, _, _ = client_fixture()
    assert client.get("/v1/jobs/missing").status_code == 404
