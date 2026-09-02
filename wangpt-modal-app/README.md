# WanGP Modal App

This project runs asynchronous WanGP image, video, and audio generation on
Modal without an HTTP or REST layer. It consists of a deployed GPU worker app
and a local Modal CLI:

- Video models run on `WanGPVideoWorker`, which uses A100 by default.
- Image and audio models run on `WanGPImageWorker`, which uses L40S by default.
- Models supporting both image and video follow their native `image_mode`
  setting unless `--kind` is supplied explicitly.

The CLI in `control.py` validates requests locally and sends them directly to
the correct deployed class in `wangpt-modal-app`. Jobs continue after the local
command exits, and their records are stored in the `wangpt-modal-jobs` Modal
Dict.

## Architecture

The worker app in `app.py` contains:

- `WanGPImageWorker`: L40S image/audio execution pool.
- `WanGPVideoWorker`: A100 video execution pool.
- `publish_catalog`: a CPU-only, heavyweight function used only to publish the
  catalog generated from the pinned WanGP build.

The local CLI in `control.py` contains:

- `submit_generation`: validates a request, resolves its output kind from the
  cached catalog, records a job, and spawns the correct deployed worker class.
- `get_generation_job`: polls and reconciles a spawned FunctionCall.
- `cancel_generation_job`: cancels one queued or running FunctionCall.
- `inspect_catalog`: exposes model metadata, defaults, and schemas to the local
  CLI without starting the WanGP image.

Catalogs are cached in the `wangpt-model-catalogs` Modal Dict by WanGP commit.
A missing catalog is published automatically; `refresh_catalog` can publish it
explicitly after a worker deployment. Once cached, model discovery reads the
Dict directly from the local process. Submission, status, cancellation, and
LoRA listing also use Modal's client APIs directly.

`control.py` has no remote Modal functions and no Modal image. Running one of
its `local_entrypoint`s does not start a control container. The only deployed
app is `wangpt-modal-app`. On a catalog cache miss or explicit refresh, the CLI
starts the CPU-only `publish_catalog` function in that worker app once.

WanGP and Wan2AI are pinned in the Modal image. Models, LoRAs, settings, input
assets, and caches use the existing `wangp-data` Volume. Generated media is
written to container-local scratch space, uploaded and verified in S3, then
removed locally.

## Deploy

The existing `huggingface-secret` Modal secret must contain `HF_TOKEN`. The
existing `studio-s3` secret must contain:

- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`
- `S3_ENDPOINT`
- `S3_BUCKET`
- `S3_REGION`

Deploy the worker app before using the local CLI entrypoints:

```bash
cd wangpt-modal-app
python3 -m modal deploy app.py
python3 -m modal run control.py::refresh_catalog
```

The deployed app name is `wangpt-modal-app`. Do not deploy `control.py`; Modal
runs its entrypoints on your local machine.

Optional worker configuration:

```bash
export WANGP_IMAGE_GPU=L40S
export WANGP_IMAGE_MAX_CONTAINERS=3
export WANGP_IMAGE_MEMORY_MB=65536
export WANGP_IMAGE_PROFILE=4

export WANGP_VIDEO_GPU=A100
export WANGP_VIDEO_MAX_CONTAINERS=1
export WANGP_VIDEO_MEMORY_MB=131072
export WANGP_VIDEO_PROFILE=4

python3 -m modal deploy app.py
python3 -m modal run control.py::refresh_catalog
```

`WANGP_GPU` and `WANGP_MAX_CONTAINERS` remain aliases for the image worker.
Set `WANGP_MODEL_LOAD_TRACE_INTERVAL_SECONDS=0` to disable periodic stack dumps
during slow model loading.

## Discover models

List every model in the catalog:

```bash
python3 -m modal run control.py::models
```

Filter the list:

```bash
python3 -m modal run control.py::models --family qwen
python3 -m modal run control.py::models --model-type krea2_turbo
```

Inspect one model's defaults or schema:

```bash
python3 -m modal run control.py::defaults --model krea2_turbo
python3 -m modal run control.py::schema --model krea2_turbo
```

## Submit generation

### Krea2 Turbo

Krea2 Turbo uses a full native WanGP JSON request. The local help command prints
the required fields, optional fields and defaults, a complete request, and a
LoRA request:

```bash
python3 -m modal run control.py::krea_help
```

Submit the complete JSON object inline through the Krea-specific entrypoint:

```bash
python3 -m modal run control.py::krea \
  --params-json '{"prompt":"A red fox walking through fresh snow","seed":-1}'
```

The JSON string can contain every native Krea2 parameter:

```json
{
  "prompt": "A red fox walking through fresh snow at golden hour",
  "negative_prompt": "blurry, low quality",
  "resolution": "1024x1024",
  "num_inference_steps": 6,
  "seed": -1,
  "batch_size": 1,
  "guidance_scale": 0,
  "flow_shift": 5.0
}
```

Krea2 Turbo also has a dedicated [parameter reference](docs/krea2-turbo.md) and
a checked-in [JSON example](examples/krea2_turbo.json). File-based submission
remains available through the generic entrypoint when useful:

```bash
python3 -m modal run control.py::submit \
  --model krea2_turbo \
  --kind image \
  --params-file examples/krea2_turbo.json
```

The reference documents the minimal request, the recommended 8-step baseline,
LoRA parameters, inpainting controls, and the commands for querying the exact
defaults and schema baked into the deployed image.

### Generic request

Put native WanGP parameters in a local JSON file:

```json
{
  "prompt": "A fox crossing a snowy field at dawn",
  "seed": -1
}
```

Submit it:

```bash
python3 -m modal run control.py::submit \
  --model krea2_turbo \
  --params-file request-params.json
```

The dispatcher normally infers the output kind. An explicit kind can be used
when selecting a modality supported by the model:

```bash
python3 -m modal run control.py::submit \
  --model MODEL_NAME \
  --kind video \
  --params-file request-params.json
```

`--kind` accepts `image`, `video`, or `audio`. A kind incompatible with the
selected model is rejected before GPU work is spawned. Absolute asset and LoRA
paths in the parameters must remain under `/data`; `_api` is reserved.

Submission prints a record containing the job ID, queued status, and resolved
kind. The command invokes the stable deployment, so `modal run --detach` is not
required.

## List LoRAs

The `loras` entrypoint calls the Modal Volume API from the local process. It
does not start either app or any remote container:

```bash
python3 -m modal run control.py::loras
python3 -m modal run control.py::loras --recursive
```

The equivalent built-in Modal command is:

```bash
python3 -m modal volume ls wangp-data loras --json
```

## Status and cancellation

Poll through the CLI entrypoint:

```bash
python3 -m modal run control.py::status --job-id JOB_ID
```

Or read the persistent record directly:

```bash
python3 -m modal dict get wangpt-modal-jobs JOB_ID
```

Statuses are `queued`, `running`, `succeeded`, `failed`, or `cancelled`.
Progress reported by WanGP is stored under `progress`. Successful terminal
records contain verified S3 output metadata under `result.outputs`.

Cancel one queued or running job:

```bash
python3 -m modal run control.py::cancel --job-id JOB_ID
```

Cancellation terminates that FunctionCall and its worker container. Completed
or failed jobs cannot be cancelled; cancelling an already-cancelled job is
idempotent.

Application logs remain available through Modal:

```bash
python3 -m modal app logs wangpt-modal-app
```

## Development

Run the local unit tests without building the Modal GPU image:

```bash
python3 -m pytest
```

The tests cover request validation, path restrictions, catalog routing,
image/video worker selection, S3 verification, and CLI parameter loading.
