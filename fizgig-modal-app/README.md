# Fizgig training on Modal

This app runs Fizgig's pinned headless LoRA training pipeline on Modal. The
local `control.py` client calls the deployed Modal functions directly, using
the same local-client/deployed-worker pattern as `wangpt-modal-app`.
No REST app, endpoint URL, or HTTP proxy credentials are required.

## Client and worker

- `app.py` deploys `fizgig-modal-app`, with `run_training`, `request_pause`,
  `health`, and `fetch_models` functions.
- `control.py` has only local entrypoints. It validates requests, spawns the
  stable deployed worker, and polls/cancels Modal FunctionCalls.
- `fizgig_common.py` shares request validation and presets between client and
  worker without importing the CUDA image.
- Jobs live in the existing `fizgig-modal-jobs` Dict. Client request/call metadata
  uses separate `submission:<job_id>` keys, so submission cannot overwrite
  progress from a worker that starts before `spawn()` returns.

The worker defaults to one L40S GPU container, 64 GiB host RAM, and a 24-hour
training timeout. It scales to zero when idle. Fizgig is pinned to
[`5d5ced3739f303065ca6d13b51eb95c5ec85c00d`](https://github.com/shootthesound/Fizgig/commit/5d5ced3739f303065ca6d13b51eb95c5ec85c00d)
(v6.8.3), the upstream `master` head checked on 2026-10-03.

The supported presets are:

- Krea2: `krea2_defaults` (rank 32, 30 epochs) and `krea2_ultra_fast`
  (rank 8, 20 epochs, adaptive learning rate).
- MiniMax H3: `h3_character_fast` (rank 8, 40 epochs) and
  `h3_character_quality` (rank 16, 60 epochs).

## Compatibility with Fizgig 6.8.3

Krea2 now runs `src/fizgig/families/cache.py` (with `--stage latents|text`)
and `src/fizgig/families/train.py`, both with `--family krea2`. The old
`krea2_cache_*` and `krea2_train.py` scripts were removed upstream. Training
uses the new automatic precision/block-swap planner and passes `--captioner`
to keep auto-recaptioning enabled. Rank, learning rate, epochs, optimizer,
trigger, and checkpoint settings remain those of our existing presets.

The new driver uses `krea2drv` cache filenames, rebuilding latents/text alongside
the original Krea2 caches. Existing model weights are still compatible. Upstream
can resume an old Krea2 state from its LoRA weights and epoch, but initializes a
fresh optimizer/EMA because parameter ordering changed. A cross-version resume
is not a bit-for-bit continuation.

MiniMax H3 retains its existing cache/training entrypoints. The image build
checks that both families' command-line entrypoints import successfully before
deployment. GPU training and cross-version resume still require a real dataset
run to validate end to end.

## Setup and deployment

Run these commands from `fizgig-modal-app`, using a Python environment with
its dependencies installed and an authenticated Modal SDK profile:

```bash
uv sync --dev
uv run modal deploy app.py
uv run modal run control.py::health
```

The existing `huggingface-secret` Modal secret must contain `HF_TOKEN`.
Optional deployment settings are `FIZGIG_GPU` and `FIZGIG_MAX_CONTAINERS`.
Do not deploy `control.py`; it runs locally and constructs no worker image.
Health calls the small deployed CPU function, without starting a GPU.

The direct client also works with an existing deployment exposing the four
functions above; adopting the client does not itself require redeployment.
The legacy REST service is not consulted or shut down by this change.
Use normal Modal SDK credentials, not `MODAL_KEY`/`MODAL_SECRET` proxy headers.

## Models and datasets

Download weights through the deployed CPU function (or inspect its plan):

```bash
uv run modal run control.py::fetch_models --family krea2 --dry-run
uv run modal run control.py::fetch_models --family krea2
```

Use `--family minimax_h3` for H3. Both training and WanGP share the existing
`wangp-data` Volume mounted at `/data`:

```text
/data/fizgig/models/                        downloaded Fizgig models
/data/fizgig/datasets/<dataset>/images/     input images and caption sidecars
/data/fizgig/datasets/<dataset>/cache/      latent and text caches
/data/fizgig/runs/<output_name>/            checkpoints and resumable state
/data/loras/<output_name>.safetensors       promoted final LoRA
```

Upload a dataset and verify its resulting layout:

```bash
uv run modal volume put wangp-data ./linda /fizgig/datasets/linda/images
uv run modal volume ls wangp-data /fizgig/datasets/linda/images
```

Krea2 generates missing/empty `.txt` captions with Qwen3-VL and preserves
existing non-empty captions. H3 requires prepared caption sidecars.

## Submit and inspect training

```bash
uv run modal run control.py::submit \
  --family krea2 \
  --dataset linda \
  --output-name linda_krea2_v1 \
  --preset krea2_defaults \
  --trigger-word linda
```

`--trigger-word` and `--epochs` are optional. An omitted/zero CLI epoch value
uses the preset default; explicit JSON epoch overrides must be 1–500.
The equivalent JSON request is in `request.example.json`:

```bash
uv run modal run control.py::submit_file --request-file request.example.json
uv run modal run control.py::status --job-id JOB_ID
uv run modal run control.py::status --job-id JOB_ID --logs
```

Submission returns a job ID, FunctionCall ID, and queued acknowledgment.
It spawns the stable deployment, so jobs continue after the local client exits
without `--detach`. Status polls once and reports worker phase/epoch progress;
`--logs` includes the diagnostic log tail. Container startup failures are
reconciled into the persistent job record.

Requests accept only `family`, `dataset`, `output_name`, `preset`, optional
`trigger_word`, and optional `epochs`. Resume is controlled by the resume
command. Raw CLI arguments and arbitrary model/filesystem paths are rejected.

For Krea2, the worker captions images, caches latents and text, then trains
with per-image loss/LR tracking and recaptioning of stuck images. The dataset
configuration is 512×512 with aspect-ratio buckets, batch size 1, and one repeat.
Checkpoints are saved each epoch; two resumable state directories are retained.
On completion, only the unnumbered final LoRA is copied to `/data/loras`.

## Pause, resume, and cancel

```bash
uv run modal run control.py::pause --job-id JOB_ID
uv run modal run control.py::status --job-id JOB_ID
uv run modal run control.py::resume --job-id JOB_ID
uv run modal run control.py::cancel --job-id JOB_ID
```

Pause is cooperative: the job stays `running` until an epoch boundary saves
state. A completed pause is `status: succeeded`, `progress.phase: paused`, and
`result.paused: true`. Resume preserves the request and returns a **new job ID**.
Cancellation terminates a queued/running FunctionCall; use pause to preserve
resumable state. Completed jobs cannot be cancelled.

Existing worker records remain readable by ID, and paused records with saved
requests can be resumed. Older submissions without a stored FunctionCall ID
cannot be reconciled or cancelled by this client. Use their original client
for cancellation; no historical REST records are migrated automatically.

## Python integration

With `fizgig-modal-app` on your Python import path:

```python
from control import submit_training, get_training_job

job = submit_training({
    "family": "krea2",
    "dataset": "linda",
    "output_name": "linda_krea2_v1",
    "preset": "krea2_defaults",
})
print(get_training_job(job["id"]))
```

`pause_training_job`, `resume_training_job`, and `cancel_training_job` expose
that same lifecycle to Python callers. The worker contract remains
`modal.Function.from_name("fizgig-modal-app", "run_training").spawn(job_id, request)`;
prefer the client helpers when you need persistent tracking and reconciliation.

## Verify locally

```bash
uv run pytest
python3 -m py_compile app.py control.py fizgig_common.py caption_dataset.py
# Also check all preset commands against a checkout of the pinned upstream:
FIZGIG_TEST_ROOT=/path/to/Fizgig uv run pytest
```

The tests exercise validation, generated training commands, and mocked Modal
job lifecycle operations without downloading models or starting GPUs.
