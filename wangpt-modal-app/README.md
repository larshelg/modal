# WanGP Modal App

This project runs asynchronous WanGP image, video, and audio generation on
Modal without an HTTP or REST layer. A small CPU dispatcher reads the WanGP
catalog, validates each request, and sends it to one of two independently
scaled GPU worker pools:

- Video models run on `WanGPVideoWorker`, which uses H100 by default.
- Image and audio models run on `WanGPImageWorker`, which uses L40S by default.
- Models supporting both image and video follow their native `image_mode`
  setting unless `--kind` is supplied explicitly.

The CLI talks to the deployed Modal functions by name. Jobs continue after the
local command exits, and their records are stored in the `wangpt-modal-jobs`
Modal Dict.

## Architecture

The deployed app contains:

- `submit_generation`: validates a request, resolves its output kind from the
  baked catalog, records a job, and spawns the correct worker.
- `WanGPImageWorker`: L40S image/audio execution pool.
- `WanGPVideoWorker`: H100 video execution pool.
- `get_generation_job`: polls and reconciles a spawned FunctionCall.
- `cancel_generation_job`: cancels one queued or running FunctionCall.
- `inspect_catalog`: exposes model metadata, defaults, and schemas to the local
  CLI without running a web service.

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

Deploy once before using the CLI entrypoints:

```bash
cd wangpt-modal-app
python3 -m modal deploy app.py
```

The deployed Modal app name is `wangpt-modal-app`.

Optional worker configuration:

```bash
export WANGP_IMAGE_GPU=L40S
export WANGP_IMAGE_MAX_CONTAINERS=3
export WANGP_IMAGE_MEMORY_MB=65536
export WANGP_IMAGE_PROFILE=4

export WANGP_VIDEO_GPU=H100
export WANGP_VIDEO_MAX_CONTAINERS=1
export WANGP_VIDEO_MEMORY_MB=131072
export WANGP_VIDEO_PROFILE=4

python3 -m modal deploy app.py
```

`WANGP_GPU` and `WANGP_MAX_CONTAINERS` remain aliases for the image worker.
Set `WANGP_MODEL_LOAD_TRACE_INTERVAL_SECONDS=0` to disable periodic stack dumps
during slow model loading.

## Discover models

List every model in the catalog:

```bash
python3 -m modal run app.py::models
```

Filter the list:

```bash
python3 -m modal run app.py::models --family qwen
python3 -m modal run app.py::models --model-type krea2_turbo
```

Inspect one model's defaults or schema:

```bash
python3 -m modal run app.py::defaults --model krea2_turbo
python3 -m modal run app.py::schema --model krea2_turbo
```

## Submit generation

Put native WanGP parameters in a local JSON file:

```json
{
  "prompt": "A fox crossing a snowy field at dawn",
  "seed": -1
}
```

Submit it:

```bash
python3 -m modal run app.py::submit \
  --model krea2_turbo \
  --params-file request-params.json
```

The dispatcher normally infers the output kind. An explicit kind can be used
when selecting a modality supported by the model:

```bash
python3 -m modal run app.py::submit \
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

## Status and cancellation

Poll through the CLI entrypoint:

```bash
python3 -m modal run app.py::status --job-id JOB_ID
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
python3 -m modal run app.py::cancel --job-id JOB_ID
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
