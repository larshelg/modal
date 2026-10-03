# Fizgig direct Modal reference

## Client contract

Training uses `fizgig-modal-app/control.py`, whose local entrypoints invoke
`fizgig-modal-app` functions through the Modal SDK. It has no remote functions
or container image of its own. Use normal Modal SDK authentication; no REST URL
or Modal proxy authentication headers are needed.

The source of truth for presets and validation is
`fizgig-modal-app/fizgig_common.py`; pipeline construction is in
`fizgig-modal-app/app.py`. Do not invoke upstream training scripts outside the
worker.

The request contains required `family`, `dataset`, `output_name`, and `preset`;
optional `trigger_word` and `epochs` (integer 1–500) are the only public
training overrides. `resume_from` is controlled by the resume operation.
Seeds, caption behavior, checkpoint cadence, and preview behavior come from
the worker preset.

```json
{
  "family": "krea2",
  "dataset": "linda",
  "output_name": "linda_krea2_v1",
  "preset": "krea2_defaults",
  "trigger_word": "linda"
}
```

Submit a JSON file with:

```bash
python3 -m modal run fizgig-modal-app/control.py::submit_file \
  --request-file fizgig-modal-app/request.example.json
```

Krea2 presets are `krea2_defaults` (rank/alpha 32, 30 epochs, LR `1e-4`) and
`krea2_ultra_fast` (rank/alpha 8, 20 epochs, adaptive LR `2e-4` to `4e-4`).
H3 presets are `h3_character_fast` (rank 8, 40 epochs) and
`h3_character_quality` (rank 16, 60 epochs).

## Pinned upstream compatibility

The worker pins Fizgig `5d5ced3739f303065ca6d13b51eb95c5ec85c00d` (v6.8.3,
master checked 2026-10-03). Krea2 uses `src/fizgig/families/cache.py` with
`--family krea2 --stage latents|text --model ...`, then
`src/fizgig/families/train.py --family krea2`. The original `krea2_cache_*`
and `krea2_train.py` scripts no longer exist. The new driver uses automatic
precision/block-swap planning; `--captioner` supplies the encoder needed by
`--auto_recaption`.

Krea2 caches are rebuilt with the `krea2drv` architecture suffix alongside old
caches. Existing model weights remain usable. Upstream resumes old Krea2 state
weights and epoch with a fresh optimizer/EMA, because parameter ordering changed;
explain this when resuming a pre-upgrade run. MiniMax H3 keeps its existing
headless entrypoints. The wrapper still exposes only Krea2 and H3 training;
new upstream families and GUI tools are not automatically added to its API.

## Lifecycle and compatibility

`submit_training`, `get_training_job`, `pause_training_job`,
`resume_training_job`, and `cancel_training_job` are also importable Python
helpers in `control.py`. Submission calls the stable deployed `run_training`
function with `.spawn(job_id, request)`, so it does not require `--detach`.
Pause calls the small deployed `request_pause` function. Status and cancellation
use the Modal Dict and FunctionCall SDK directly.

Top-level states are `queued`, `running`, `succeeded`, `failed`, and `cancelled`.
Worker details are in `progress.phase`. A completed pause is a successful
function execution with `result.paused: true`; it is not completed training.
Only a successfully paused job may resume, producing a new job ID.

The existing `fizgig-modal-jobs` Dict holds worker records. Separate
`submission:<job_id>` entries store the original request, call ID, and resume
parent without racing worker progress updates. Status reconciles failures that
occur before the worker starts and strips log tails unless `--logs` is passed.
An expired FunctionCall output does not prove training failed.

Older worker records remain readable. Paused records containing their resolved
request can be resumed through the new client. Historical submissions without
stored FunctionCall IDs cannot be cancelled or reconciled by the new client;
use their original client if cancellation is needed. No REST deployment or
historical job records are deleted or migrated. `scripts/fizgig_api.py` remains
only for explicit legacy REST compatibility.

## Setup and storage

Deploy `fizgig-modal-app/app.py`, not `control.py`. An existing deployment with
`health`, `fetch_models`, `run_training`, and `request_pause` already supports
the direct client. The worker requires `huggingface-secret` with `HF_TOKEN`.

```bash
python3 -m modal run fizgig-modal-app/control.py::fetch_models --family krea2 --dry-run
python3 -m modal run fizgig-modal-app/control.py::fetch_models --family krea2
```

Model fetching uses the deployed CPU function and persistent `wangp-data` Volume.
Use the requested family (`krea2` or `minimax_h3`); a dry run only prints the plan.

```text
/data/fizgig/models/
/data/fizgig/datasets/<dataset>/images
/data/fizgig/datasets/<dataset>/cache
/data/fizgig/runs/<output_name>
/data/loras/<output_name>.safetensors
```

The worker generates a bucketed 512×512 dataset config with batch size 1 and
one repeat. Krea2 captions missing sidecars, caches image latents and text,
then trains with per-image loss/LR tracking and Qwen3-VL recaptioning. Existing
non-empty captions are preserved. Checkpoints are saved each epoch and the
last two state directories are retained. Only the unnumbered final LoRA is
automatically copied to `/data/loras`; numbered checkpoints remain in the run.

## Artifact operations

For a requested dataset upload, use Modal Volume commands and verify layout:

```bash
python3 -m modal volume put wangp-data ./linda /fizgig/datasets/linda/images
python3 -m modal volume ls wangp-data fizgig/datasets/linda/images --json
```

For diagnostics, read a bounded log tail:

```bash
python3 -m modal app logs fizgig-modal-app --tail 300 --timestamps
```

Summarize relevant phase, epochs, checkpoint saves, or errors. The job record
is authoritative for terminal success, not the log tail.

List runs and promoted LoRAs with Volume-relative paths (omit `/data`):

```bash
python3 -m modal volume ls wangp-data fizgig/runs/<output_name> --json
python3 -m modal volume ls wangp-data loras --json
```

For an explicitly requested epoch promotion, first verify the source exists
and the destination does not, then copy with an epoch postfix:

```bash
python3 -m modal volume cp wangp-data \
  fizgig/runs/<output_name>/<output_name>-000003.safetensors \
  loras/<output_name>_epoch_003.safetensors
```

Preserve source checkpoints and the unnumbered final LoRA. Require confirmation
before overwriting an existing destination or deleting artifacts.

## WanGP generation

Generation uses `wangpt-modal-app/control.py`, independently of training.
Read that app's README for current routing and model-specific behavior. Native
params go in a JSON file passed with `--params-file`; `_api` is reserved.
Absolute asset and LoRA paths stay under `/data`. `--kind image|video|audio`
is optional and validated against the model catalog.

```bash
python3 -m modal run wangpt-modal-app/control.py::models
python3 -m modal run wangpt-modal-app/control.py::defaults --model krea2_turbo
python3 -m modal run wangpt-modal-app/control.py::schema --model krea2_turbo
python3 -m modal run wangpt-modal-app/control.py::submit \
  --model krea2_turbo --kind image --params-file request-params.json
python3 -m modal run wangpt-modal-app/control.py::status --job-id JOB_ID
```
