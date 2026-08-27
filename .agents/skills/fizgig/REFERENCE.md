# Fizgig REST contract

The protected endpoint is deployed by `rest-wangpt-modal-app`. There is
no `/v1` prefix in the current implementation.

## Training

Krea2 identity request:

```json
{
  "family": "krea2",
  "dataset": "linda",
  "output_name": "linda_krea2_v1",
  "preset": "krea2_defaults",
  "trigger_word": "linda"
}
```

Krea2 presets are `krea2_defaults` (rank 32, 30 epochs) and
`krea2_ultra_fast` (rank 8, adaptive LR, 20 epochs). The preset enables
initial Qwen3-VL captioning for missing or empty `.txt` sidecars and repairs
stuck images between epochs. These are execution details, not public switches.

MiniMax H3 request:

```json
{
  "family": "minimax_h3",
  "dataset": "anna",
  "output_name": "anna_h3_v1",
  "preset": "h3_character_quality",
  "trigger_word": "anna",
  "epochs": 60
}
```

`family`, `dataset`, `output_name`, and `preset` are required. `trigger_word`
and `epochs` are the only optional public training fields. Seeds, checkpoint
cadence, caption behavior, and preview settings are resolved by the worker
preset. `resume_from` is controlled by the resume endpoint and must not be
submitted by clients.

Route mapping:

```text
POST /training/jobs                  training-submit
GET  /training/jobs/{id}             training-status
POST /training/jobs/{id}/pause       training-pause
POST /training/jobs/{id}/resume      training-resume
POST /training/jobs/{id}/cancel      training-cancel
```

Top-level states are `queued`, `running`, `succeeded`, `failed`, and
`cancelled`. Worker detail is carried in `progress.phase`.

The first version requires images to exist in the shared Modal Volume:

```text
/data/fizgig/datasets/<dataset>/images
```

Runs and promoted LoRAs are stored at:

```text
/data/fizgig/runs/<output_name>
/data/loras/<output_name>.safetensors
```

There is no dataset upload, checkpoint-listing, preview-listing,
comparison, evaluation, or per-epoch promotion endpoint yet.

## Headless worker execution

The REST request contains training intent, not raw CLI arguments. The
`fizgig-modal-app` worker validates that request, creates a 512x512 bucketed
dataset config with batch size 1 and one repeat, and constructs the pinned
Fizgig commands inside the GPU container.

For `krea2_defaults`, the stages are:

```text
caption missing sidecars with Qwen3-VL
krea2_cache_latents.py --skip_existing
krea2_cache_text.py --skip_existing
krea2_train.py
```

The training command resolves to rank/alpha 32, learning rate `1e-4`, 30 epochs,
seed 42, one numbered checkpoint per epoch, resumable state, two retained state
directories, `adamw8bit`, `compile_blocks=auto`, per-image loss logging and LR,
and Qwen3-VL auto-recaptioning. No NF4, INT8, or block-swap flag is supplied, so
Krea2 uses its validated dynamic-FP8 default with zero block swapping. A trigger
word, when present, is passed both as the training trigger and metadata phrase.

`krea2_ultra_fast` uses rank/alpha 8 and 20 epochs, and adds adaptive LR with a
`2e-4` to `4e-4` range. The source of truth for command construction is
`fizgig-modal-app/app.py`; do not reconstruct or execute these upstream commands
outside the worker.

After a completed run, Fizgig writes the unnumbered final LoRA in the run
directory. The worker automatically copies only that file to:

```text
/data/loras/<output_name>.safetensors
```

For a full 30-epoch run, this unnumbered artifact corresponds to epoch 30.
Numbered epoch checkpoints remain under the run directory until explicitly
promoted.

## Raw Modal CLI fallback

Use the following commands only for direct development or operations not yet
available through REST. Run them from the repository root. Load the existing
local environment without printing credential values:

```bash
set -a
source .env
set +a
```

### Direct development submission

Edit `fizgig-modal-app/request.example.json`, keeping the same allowlisted REST
shape, then detach the worker call:

```bash
python3 -m modal run --detach fizgig-modal-app/app.py \
  --request-json fizgig-modal-app/request.example.json
```

This is the pre-REST/manual submission flow. It prints a job ID and Modal
FunctionCall ID, but does not expose arbitrary Fizgig CLI arguments. Prefer
`training-submit` for normal agent operation.

### Diagnostic logs

Read a bounded log tail rather than following indefinitely:

```bash
python3 -m modal app logs fizgig-modal-app --tail 300 --timestamps
```

Summarize stage, epoch/step progress, checkpoint saves, plateau or exclusion
warnings, final artifact lines, and any fatal error. Do not claim terminal job
success from logs alone.

### Volume inspection

List a run or the promoted LoRA directory as JSON:

```bash
python3 -m modal volume ls \
  wangp-data fizgig/runs/<output_name> --json

python3 -m modal volume ls wangp-data loras --json
```

The paths passed to `modal volume` are relative to the `/data` mount; omit the
leading `/data`.

### Manual epoch promotion

First verify that the source exists and the destination does not. Then copy
within the same Volume using a zero-padded epoch postfix:

```bash
python3 -m modal volume cp wangp-data \
  fizgig/runs/<output_name>/<output_name>-000003.safetensors \
  loras/<output_name>_epoch_003.safetensors
```

Repeat only for epochs explicitly requested by the user. Preserve the source
checkpoint and the unnumbered final LoRA. If a destination already exists,
require confirmation before overwriting it. Deletion remains outside this
fallback unless the user explicitly requests it and confirms the exact target.

## WanGP generation

Submission shape:

```json
{
  "kind": "image",
  "model": "krea2_turbo",
  "params": {
    "prompt": "A portrait in natural window light",
    "resolution": "1024x1024",
    "num_inference_steps": 8,
    "seed": 42
  }
}
```

`kind` is optional and accepts `image`, `video`, or `audio`. The server infers
it from the selected model's `main_output` metadata when omitted. An explicit
kind is validated against the model and a mismatch returns `400`. Video jobs
run on the dedicated H100 worker; image and audio jobs use the standard L40S
worker. Models that can switch between image and video are inferred from the
effective native `image_mode` setting when `kind` is omitted.

Route mapping:

```text
GET  /models                         models
GET  /models/{model}/defaults        model-defaults
GET  /models/{model}/schema          model-schema
POST /jobs                           generation-submit
GET  /jobs/{id}                      generation-status
POST /jobs/{id}/cancel               generation-cancel
GET  /outputs/{id}                   protected output download
```

Generation accepts native WanGP settings. `_api` is reserved, and
absolute filesystem paths must remain under `/data`.

## HTTP behavior

- `400`: invalid field, value, or path.
- `401` or `403`: invalid Modal proxy authentication.
- `404`: unknown or expired resource.
- `409`: invalid lifecycle transition.
- `422`: malformed request shape.
- `503`: Fizgig deployment or lifecycle operation unavailable.

The client exits nonzero and writes a JSON error object to stderr for
HTTP and transport failures.
