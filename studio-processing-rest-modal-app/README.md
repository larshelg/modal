# Studio processing REST API on Modal

This app is an authenticated asynchronous REST control plane for the existing
natural-studio Modal workers. One job invokes them in strict order:

1. `tapnextpp-modal-app/run_tracking_stage`
2. `sam3-modal-app/run_sam_stage`

The gateway is deliberately lightweight. It sends only the two existing JSON
StageRequests through Modal, stores compact job state in a Modal Dict, and
returns the workers' unchanged StageResults. Videos, masks, coordinates,
review evidence, and commit markers continue to move directly through S3.

The worker apps remain separate deployments with different runtimes. The REST
app does not contain TAPNext++, SAM, model weights, GPU images, S3 credentials,
or media-processing code.

## Current integration status

This is an additive API. `scripts/studio-experiment` still uses the Modal Python
SDK to submit and poll tracking and SAM independently. Deploying this app does
not change existing debug or fast experiment behavior.

## Requirements

- Deploy `tapnextpp-modal-app` and `sam3-modal-app` first.
- Authenticate the Modal CLI in the same workspace as both workers.
- Create Modal proxy credentials for REST clients. Do not put them in request
  bodies, committed files, job records, or command output.
- The worker apps retain responsibility for the `studio-s3` and
  `huggingface-secret` secrets.

## Local checks

```bash
cd /Users/larshelg/comfyui/modal2/studio-processing-rest-modal-app
uv sync --dev
uv run pytest
uv run python -m py_compile app.py studio_gateway.py
```

## Deploy

```bash
cd /Users/larshelg/comfyui/modal2/tapnextpp-modal-app
uv run modal deploy app.py

cd /Users/larshelg/comfyui/modal2/sam3-modal-app
uv run modal deploy app.py

cd /Users/larshelg/comfyui/modal2/studio-processing-rest-modal-app
uv run modal deploy app.py
```

The worker names can be changed at deployment time without changing the API:

```bash
export STUDIO_TRACKING_MODAL_APP=tapnextpp-modal-app
export STUDIO_TRACKING_MODAL_FUNCTION=run_tracking_stage
export STUDIO_SAM_MODAL_APP=sam3-modal-app
export STUDIO_SAM_MODAL_FUNCTION=run_sam_stage
uv run modal deploy app.py
```

The deployment prints the HTTPS endpoint. Clients must send Modal proxy-auth
headers on every request:

```bash
export STUDIO_PROCESSING_URL=https://YOUR-ENDPOINT

curl --fail-with-body \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  "$STUDIO_PROCESSING_URL/health"
```

The health response names both configured workers and reports
`"execution": "sequential"`; it does not cold-start either worker.

## Submit a job

Create a JSON document containing the unchanged tracking and SAM StageRequests:

```json
{
  "schemaVersion": 1,
  "tracking": {
    "schemaVersion": 1,
    "stage": "tracking",
    "runId": "example-run",
    "inputHash": "<64 lowercase hex characters>",
    "inputs": {"normalizedVideo": {"storage": "s3", "bucket": "...", "key": "...", "sha256": "...", "sizeBytes": 123, "contentType": "video/mp4"}},
    "parameters": {"evidenceMode": "full"},
    "expectedMedia": {"frames": 120, "fps": "24/1", "width": 1080, "height": 1920},
    "output": {"bucket": "...", "prefix": "studio-experiments/example-run/tracking/<input-hash>/"}
  },
  "sam": {
    "schemaVersion": 1,
    "stage": "sam",
    "runId": "example-run",
    "inputHash": "<64 lowercase hex characters>",
    "inputs": {"normalizedVideo": {"storage": "s3", "bucket": "...", "key": "...", "sha256": "...", "sizeBytes": 123, "contentType": "video/mp4"}},
    "parameters": {"evidenceMode": "full"},
    "expectedMedia": {"frames": 120, "fps": "24/1", "width": 1080, "height": 1920},
    "output": {"bucket": "...", "prefix": "studio-experiments/example-run/sam/<input-hash>/"}
  }
}
```

The full stage-specific parameters are defined by the existing worker
contracts; the abbreviated example only shows the shared gateway fields. Both
requests must use the same run ID, normalized-video ArtifactRef, expected media,
output bucket, and evidence mode. Their output prefixes must differ.

Submit the file with a client-generated idempotency key:

```bash
export IDEMPOTENCY_KEY=example-run-tracking-sam-v1

curl --fail-with-body -X POST \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: $IDEMPOTENCY_KEY" \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  "$STUDIO_PROCESSING_URL/v1/jobs" \
  --data-binary @pipeline-request.json
```

The endpoint returns HTTP `202 Accepted` immediately:

```json
{"id":"<job-id>","status":"queued","currentStage":"tracking"}
```

Retrying the identical request with the same `Idempotency-Key` returns the
same job and does not intentionally dispatch another coordinator. Reusing the
key with a different request returns HTTP 409.

## Poll a job

```bash
export JOB_ID=<job-id>

curl --fail-with-body \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  "$STUDIO_PROCESSING_URL/v1/jobs/$JOB_ID"
```

Statuses are `queued`, `running`, `succeeded`, and `failed`.
`currentStage` is `tracking`, `sam`, or `completed`. A successful response
contains the unchanged tracking and SAM StageResults under both `stages` and
the compact top-level `result`. Internal Function Call IDs, original requests,
and request digests are never returned.

If tracking fails, SAM is not submitted. Public failures contain only a stable
code, stage, exception type, retryability flag, and a message directing the
operator to Modal logs; raw exception text and tracebacks are not persisted in
the public job record.

## Evidence modes and retry behavior

Use `parameters.evidenceMode: "full"` in both requests for debug evidence or
`"none"` in both for fast-mode core artifacts. The gateway does not add,
remove, or inspect worker artifacts.

There is no cancellation or automatic retry endpoint in v1. Resubmit a failed
pipeline with a new idempotency key. The workers reuse valid hash-qualified S3
`result.json` commits, so an already completed stage does not allocate its GPU
again.
