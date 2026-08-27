# Fizgig training on Modal

This standalone Modal app runs Fizgig's documented headless CLI pipeline on a
GPU. It intentionally exposes no second REST or MCP server.

`rest-wangpt-modal-app` is the public, proxy-authenticated endpoint and
dispatches training work to the stable functions deployed by this app.

## Current scope

- Modal app name: `fizgig-modal-app`
- Fizgig revision: `6912b8aabb64600dd9da8702c5a04c8f867f7bc2`
- Model families: Krea2 and MiniMax H3
- Krea2 presets: `krea2_defaults` and `krea2_ultra_fast`
- MiniMax H3 presets: `h3_character_fast` and `h3_character_quality`
- Default GPU: `L40S`
- Maximum GPU containers: 1
- Persistent storage: the existing `wangp-data` Modal Volume only
- Object storage: no S3 integration

The Krea2 worker executes the pinned Fizgig headless pipeline:

```text
Qwen3-VL auto-caption missing sidecars
        ↓
krea2_cache_latents.py
        ↓
krea2_cache_text.py
        ↓
krea2_train.py
        ├── per-image loss and LR control
        ├── Qwen3-VL recaption of confirmed-stuck images
        └── checkpoints and resumable state
```

MiniMax H3 continues to use `minimax_cache_latents.py`,
`minimax_cache_text.py`, and `minimax_train.py`.

## Storage layout

Both this app and `rest-wangpt-modal-app` use the same `wangp-data` Volume
mounted at `/data`:

```text
/data/fizgig/models/                         downloaded Fizgig models
/data/fizgig/datasets/<dataset>/images/     input images and caption sidecars
/data/fizgig/datasets/<dataset>/cache/      VAE and text-encoder cache
/data/fizgig/runs/<output_name>/             checkpoints and resumable state
/data/loras/<output_name>.safetensors        promoted final LoRA
```

Krea2 jobs auto-caption missing or empty `.txt` sidecars before caching by
default. Existing non-empty captions are preserved. H3 datasets still require
prepared Fizgig-compatible caption sidecars.

## Install local development dependencies

```bash
cd fizgig-modal-app
uv sync --dev
```

## Deploy

The existing Modal secret `huggingface-secret` must contain `HF_TOKEN`.

```bash
cd fizgig-modal-app
python3 -m modal deploy app.py
```

Optional deployment configuration:

```bash
export FIZGIG_GPU=L40S
export FIZGIG_MAX_CONTAINERS=1
python3 -m modal deploy app.py
```

The deployment exposes named Modal functions, not a public HTTP endpoint:

- `health`
- `fetch_models`
- `run_training`
- `request_pause`

## Download models

The download is idempotent and writes directly to `wangp-data`:

```bash
cd fizgig-modal-app
python3 -m modal run app.py --fetch --family krea2
```

Use `--family minimax_h3` for H3. The pinned upstream Krea2 download includes
the raw DiT, Qwen3-VL text encoder/captioner, VAE, Turbo LoRA, and Turbo DiT.

Inspect the download plan without downloading weights:

```bash
python3 -m modal run app.py --fetch --family krea2 --dry-run
```

## Upload a prepared dataset

For a dataset named `linda`, upload the images. Manual captions are not
required for the default Krea2 flow:

```bash
python3 -m modal volume put \
  wangp-data \
  ./linda \
  /fizgig/datasets/linda/images
```

Confirm the resulting layout before training:

```bash
python3 -m modal volume ls wangp-data /fizgig/datasets/linda/images
```

## Submit a training job directly

Copy and edit `request.example.json`, then run:

```bash
python3 -m modal run --detach app.py --request-json request.example.json
```

The command prints a job ID and Modal FunctionCall ID. `--detach` allows the
remote training function to continue after the local command exits.

The request schema is allowlisted. The public REST API sends only training
intent: family, dataset, output name, preset, optional trigger word, and an
optional epoch override. The worker resolves seeds, checkpoint cadence,
caption behavior, and preview behavior from the preset. It never accepts raw
CLI arguments, arbitrary paths, shell commands, or Modal function names.

For Krea2, initial captioning and mid-run recaptioning are preset behavior.
Missing captions are generated before caching and existing captions are
preserved.

## Query or pause a deployed job

Agent clients query and pause through the protected REST API in
`rest-wangpt-modal-app`. For direct worker diagnostics, the pause function can
also be called by name:

```python
import modal

request_pause = modal.Function.from_name("fizgig-modal-app", "request_pause")

# print(request_pause.remote("REPLACE_WITH_JOB_ID"))
```

A pause request is stored in `fizgig-modal-jobs`. The running worker observes
it and creates Fizgig's `.pause_requested` sentinel inside the same container.
Fizgig exits cleanly at the next epoch boundary after saving resumable state.

To resume, submit the same request again with either:

```json
{"resume_from": "latest"}
```

or an explicit state-directory basename such as:

```json
{"resume_from": "anna_h3_v1-000010-state"}
```

Use a new job ID for the resumed function call. The REST integration exposes
the lifecycle through `/training/jobs/{id}/pause` and
`/training/jobs/{id}/resume`.

## Verify locally

```bash
cd fizgig-modal-app
uv run pytest
python3 -m py_compile app.py caption_dataset.py
```

These checks validate the request boundary and generated CLI commands without
building the multi-gigabyte Modal image or starting a GPU.
