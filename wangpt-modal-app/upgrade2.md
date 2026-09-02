# Upgrade plan: GPU worker startup and job-stage observability

## Summary

Extend job records so REST clients can tell when a submitted job has entered a
GPU worker and which broad WanGP phase is currently running.

This is an application-code change shared between the CPU API and GPU worker.
It does not require new Modal resources, new dependencies, or changes to the
CUDA/PyTorch/WanGP image definition. Both functions live in `app.py` and are
released together with a normal `modal deploy`.

The existing public job statuses remain unchanged:

```text
queued, running, succeeded, failed, cancelled
```

A new `stage` field and additional timestamps provide finer detail without
breaking existing clients.

## What the signal means

The GPU worker's `run()` method executes only after Modal has assigned the call
to a container and completed that container's `@modal.enter()` startup hook.
When `run()` records `worker_started_at`, the service can therefore confirm
that the worker is ready and has begun handling that job.

The service cannot reliably distinguish these states while the call remains
queued:

```text
waiting for Modal scheduling
GPU container provisioning
GPU container executing its startup hook
```

User code does not run during those platform-controlled phases. Consequently,
`worker_started_at` means worker startup has finished; it is not a live signal
that container booting has begun.

This distinction also applies to warm workers. The timestamp records when the
job entered a worker, whether that worker was newly started or reused.

## Proposed job lifecycle

```text
status=queued
  stage=waiting_for_worker
          |
          | GPU method begins
          v
status=running
  stage=initializing_wangp
          |
          | task submitted to WanGP
          v
status=running
  stage=submitted_to_wangp
          |
          | WanGP progress callbacks
          v
status=running
  stage=loading_model | encoding_text | inference | decoding |
        downloading_output
          |
          +--------------------+--------------------+
          v                    v                    v
status=succeeded          status=failed        status=cancelled
  stage=completed           stage=failed         stage=cancelled
```

Modal failures that happen before or outside normal WanGP execution use
`status="failed"` and `stage="modal_failed"`.

## Job record additions

The initial queued record will include:

```json
{
  "status": "queued",
  "stage": "waiting_for_worker",
  "created_at": "2026-08-15T12:00:00+00:00",
  "updated_at": "2026-08-15T12:00:00+00:00"
}
```

After the worker begins handling the job:

```json
{
  "status": "running",
  "stage": "initializing_wangp",
  "worker_started_at": "2026-08-15T12:00:03+00:00",
  "updated_at": "2026-08-15T12:00:03+00:00"
}
```

After submission to WanGP:

```json
{
  "status": "running",
  "stage": "submitted_to_wangp",
  "wangp_submitted_at": "2026-08-15T12:00:05+00:00"
}
```

Existing structured progress remains available and drives later stages:

```json
{
  "status": "running",
  "stage": "inference",
  "progress": {
    "phase": "inference",
    "status": "Prompt 1/1 | Denoising",
    "progress": 50,
    "current_step": 2,
    "total_steps": 4
  }
}
```

Do not add `model_loaded_at` unless WanGP provides an explicit and reliable
model-loaded event. Inferring it from the first inference callback would make
the field misleading for models with different loading paths.

## Implementation plan

### 1. Centralize job-record updates

- Introduce `TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}`.
- Add a small helper that reads the latest record, applies a change, refreshes
  `updated_at`, and writes it back to `job_store`.
- By default, ignore non-terminal progress changes after a job becomes
  terminal.
- Allow explicit terminal transitions from the worker, cancellation endpoint,
  and Modal failure handler.
- Preserve fields written by other paths by always starting with the latest
  stored record.

Modal Dict updates are read-modify-write operations rather than transactional
field patches. The helper reduces accidental field loss, while each terminal
path must still re-read and respect the latest status before writing.

### 2. Update the CPU API path

When `POST /jobs` creates a record:

- Keep `status="queued"`.
- Add `stage="waiting_for_worker"`.

When the polling endpoint detects a Modal infrastructure failure:

- Keep `status="failed"`.
- Add `stage="modal_failed"`.

When cancellation succeeds:

- Keep `status="cancelled"`.
- Add `stage="cancelled"`.

No new endpoint or Pydantic response model is required because `GET /jobs/{id}`
already returns the stored record.

### 3. Update the GPU worker path

At the start of `WanGPWorker.run()`:

- Re-read the record and retain the existing early-cancellation check.
- Change the status to `running`.
- Set `stage="initializing_wangp"`.
- Set `worker_started_at` and `started_at`.

After `session.submit_task(...)` returns:

- Set `stage="submitted_to_wangp"`.
- Set `wangp_submitted_at`.

When execution terminates:

- Successful result: `status="succeeded"`, `stage="completed"`.
- WanGP unsuccessful result: `status="failed"`, `stage="failed"`.
- Runtime exception: `status="failed"`, `stage="failed"`, unless the latest
  record is already cancelled.

The image definition and worker resource configuration remain unchanged.

### 4. Connect WanGP progress to stages

Extend `JobCallbacks.on_progress()` to copy a recognized progress phase into
the top-level `stage` while retaining the full `progress` object.

Initially recognized phases are:

```text
loading_model
encoding_text
inference
inference_stage_1
inference_stage_2
inference_stage_3
decoding
downloading_output
cancelled
```

Unknown or empty phases must not erase the last meaningful stage. Human-readable
WanGP status text remains inside `progress.status` and must not be used as a
stable machine-readable stage.

### 5. Support rolling deployments and old records

The new fields are additive. Old CPU and GPU containers may briefly coexist
during Modal's rolling deployment, and records created before the upgrade will
not contain `stage`.

Before returning a job, derive a fallback only when `stage` is missing:

```python
if "stage" not in record:
    record["stage"] = (
        "waiting_for_worker"
        if record.get("status") == "queued"
        else record.get("status", "unknown")
    )
```

This fallback is for response compatibility and does not need to rewrite old
records. New GPU workers can update records created by old API containers, and
new API containers can return records written by old GPU workers.

### 6. Update documentation

- Add `stage`, `worker_started_at`, and `wangp_submitted_at` to the polling
  examples in `README.md`.
- Document `stage` as an evolving, coarse-grained phase rather than a second
  terminal-status field.
- State that `worker_started_at` confirms method entry after container startup.
- State that `queued` cannot distinguish Modal scheduling from active container
  provisioning.
- Preserve the existing documented status enum.

## Testing plan

Add unit tests for:

- New job records begin at `stage="waiting_for_worker"`.
- Worker entry sets `status="running"`, `stage="initializing_wangp"`, and
  `worker_started_at`.
- WanGP submission records `wangp_submitted_at`.
- Recognized progress phases update the top-level stage.
- Empty and unknown progress phases retain the previous meaningful stage.
- A progress callback cannot move a succeeded, failed, or cancelled job back
  into a running stage.
- Success, WanGP failure, runtime failure, Modal failure, and cancellation each
  receive the correct terminal stage.
- Old records without `stage` receive the correct response fallback.
- Existing status, result, error, and progress fields retain their current
  formats.

Modal smoke testing should verify:

- A scale-from-zero job remains `waiting_for_worker` until the worker method
  begins.
- `worker_started_at` appears before WanGP progress.
- A warm-worker job also records a fresh job-specific `worker_started_at`.
- Progress stages appear during a real generation.
- Cancellation during startup and generation remains terminal.
- Existing clients that only inspect `status` continue to work.

## Optional future observability

These additions are not required for this upgrade:

- Record container identity and whether the worker was warm or newly created.
- Store container initialization timestamps from `@modal.enter()` and copy them
  into the job record after method entry.
- Expose aggregate worker-pool statistics such as runner count and backlog.
- Record per-stage durations or publish a bounded event history.

Aggregate Modal function statistics are pool-level and cannot replace the
job-specific `worker_started_at` signal.

## Failure handling

- If the GPU call fails before entering `run()`, polling marks the job as
  `failed/modal_failed` through the existing FunctionCall error path.
- If cancellation wins a race with a progress callback, cancellation remains
  terminal and later progress is ignored.
- If WanGP reports an unknown phase, retain the prior stage and continue storing
  the full progress payload.
- If a record is missing new fields, return a compatible fallback rather than
  treating it as corrupt.

## Rollback

Rollback only requires deploying the previous application code. Old code
ignores the additive `stage`, `worker_started_at`, and `wangp_submitted_at`
fields already present in Modal Dict records. No data migration or cleanup is
required.

## Acceptance criteria

- A client can distinguish `waiting_for_worker` from a job that entered a GPU
  worker.
- `worker_started_at` is written as soon as the GPU method begins.
- WanGP progress produces stable, machine-readable stages when available.
- Terminal jobs cannot regress into running stages.
- The existing public status enum and response fields remain compatible.
- Old job records and rolling deployments remain supported.
- No GPU image dependency or resource change is required.
- Unit tests and an authenticated Modal generation smoke test pass.
