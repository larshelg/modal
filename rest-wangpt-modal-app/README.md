# WanGP and Fizgig REST API on Modal

This app is the protected asynchronous REST control plane for two independently
deployed Modal runtimes:

- WanGP generation runs through separate image and video workers using WanGP's
  documented `shared.api`. Image and audio jobs use L40S by default; video jobs
  use H100.
- Fizgig training is dispatched by name to the separately deployed
  `fizgig-modal-app`; this REST app does not contain or duplicate Fizgig's CLI
  integration.

The CPU control plane accepts requests immediately and uses Modal FunctionCalls
as the job queue. The image and video worker pools scale independently. Models,
LoRAs, caches, settings, and outputs share the existing `wangp-data` Volume.
Generation and training use separate public Modal Dicts so their records cannot
collide.

## Deploy

The existing `huggingface-secret` Modal secret must contain `HF_TOKEN`. Deploy
the Fizgig execution app first, then deploy this shared REST app in the same
Modal environment:

```bash
cd fizgig-modal-app
python3 -m modal deploy app.py

cd ../rest-wangpt-modal-app
python3 -m modal deploy app.py
```

Optional deployment configuration:

```bash
export WANGP_IMAGE_GPU=L40S
export WANGP_IMAGE_MAX_CONTAINERS=3
export WANGP_VIDEO_GPU=H100
export WANGP_VIDEO_MAX_CONTAINERS=1
export FIZGIG_MODAL_APP_NAME=fizgig-modal-app
python3 -m modal deploy app.py
```

`WANGP_GPU` and `WANGP_MAX_CONTAINERS` remain backward-compatible aliases for
the image worker. The video worker defaults to 128 GiB host RAM and WanGP memory
profile 3; override these with `WANGP_VIDEO_MEMORY_MB` and
`WANGP_VIDEO_PROFILE` when benchmarking a different H100 configuration.

`FIZGIG_MODAL_APP_NAME` defaults to `fizgig-modal-app`. The function names are
the fixed internal contract `run_training` and `request_pause`.

The resulting endpoint requires Modal proxy-auth headers:

```bash
curl -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  https://YOUR-ENDPOINT/health
```

## Using the REST API

Set the endpoint URL printed by `modal deploy` and the Modal proxy credentials:

```bash
export WANGP_URL="https://YOUR-ENDPOINT"
export MODAL_KEY="YOUR-MODAL-KEY"
export MODAL_SECRET="YOUR-MODAL-SECRET"
```

Every request must include both authentication headers:

```text
Modal-Key: <key>
Modal-Secret: <secret>
```

### Check service health

```bash
curl --fail-with-body \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  "$WANGP_URL/health"
```

Example response:

```json
{
  "service": "wangp-rest",
  "ready": true,
  "gpu": "L40S",
  "max_gpu_containers": 3,
  "generation_workers": {
    "image": {
      "gpu": "L40S",
      "max_containers": 3,
      "memory_mb": 65536,
      "wangp_profile": "4"
    },
    "video": {
      "gpu": "H100",
      "max_containers": 1,
      "memory_mb": 131072,
      "wangp_profile": "3"
    }
  },
  "wangp_commit": "92f56e5ee7227d490f6d85281c019e4c4e2dc393",
  "wan2ai_commit": "2539c3a87b64fa0f619695f02410fc92c63cba7d",
  "fizgig_app": "fizgig-modal-app",
  "training_families": ["minimax_h3", "krea2"]
}
```

The health route reports the configured Fizgig app name without cold-starting
it. A training submission returns `503 Service Unavailable` if that deployment
cannot be resolved or called.

## Fizgig training API

Training supports Krea2 with `krea2_defaults` and `krea2_ultra_fast`, and
MiniMax H3 with `h3_character_fast` and `h3_character_quality`. Place images in
the shared Modal Volume at `/data/fizgig/datasets/<dataset>/images` before
submission.

For Krea2, the execution app uses the pinned Krea2 Qwen3-VL encoder to generate
missing or empty caption sidecars before caching. It preserves existing
non-empty captions, runs the official Krea2 latent and text cache scripts, and
trains with per-image loss/LR support plus Fizgig's between-epoch
auto-recaptioning. H3 still requires prepared caption sidecars.

### Submit a training job

```bash
curl --fail-with-body -X POST \
  -H "Content-Type: application/json" \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  "$WANGP_URL/training/jobs" \
  --data-binary @- <<'JSON'
{
  "family": "krea2",
  "dataset": "linda",
  "output_name": "linda_krea2_v1",
  "preset": "krea2_defaults",
  "trigger_word": "linda"
}
JSON
```

The response is HTTP `202 Accepted`:

```json
{
  "id": "5ed81512-87c4-4888-a438-d23693507c21",
  "status": "queued"
}
```

The required fields are `family`, `dataset`, `output_name`, and `preset`.
`trigger_word` and `epochs` are optional. Captioning, recaptioning, seed,
checkpoint cadence, and preview behavior are resolved by the selected worker
preset. Clients cannot set `resume_from`; the resume route selects it from a
successfully paused job.

### Poll training status

```bash
export TRAINING_JOB_ID="5ed81512-87c4-4888-a438-d23693507c21"

curl --fail-with-body \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  "$WANGP_URL/training/jobs/$TRAINING_JOB_ID"
```

Training uses the same top-level statuses as generation: `queued`, `running`,
`succeeded`, `failed`, and `cancelled`. Detailed work appears under
`progress.phase`, including dataset preparation, captioning, caching, training,
finalizing, paused, and completed phases.

A completed run promotes the final LoRA into the shared Volume and reports it
in the terminal record:

```json
{
  "id": "5ed81512-87c4-4888-a438-d23693507c21",
  "status": "succeeded",
  "progress": {"phase": "completed"},
  "result": {
    "paused": false,
    "artifact_path": "/data/loras/linda_krea2_v1.safetensors",
    "run_path": "/data/fizgig/runs/linda_krea2_v1",
    "size_bytes": 12345678
  }
}
```

### Pause, resume, or cancel training

Pause requests are cooperative. The current Fizgig stage writes a persisted
state checkpoint and then finishes with `status: "succeeded"`,
`progress.phase: "paused"`, and `result.paused: true`:

```bash
curl --fail-with-body -X POST \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  "$WANGP_URL/training/jobs/$TRAINING_JOB_ID/pause"
```

After polling reaches the paused terminal record, resume it. Resume creates a
new job ID and links it to the previous job:

```bash
curl --fail-with-body -X POST \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  "$WANGP_URL/training/jobs/$TRAINING_JOB_ID/resume"
```

```json
{
  "id": "144649c2-37e7-423e-976a-274e3fd37ba8",
  "status": "queued",
  "resumed_from": "5ed81512-87c4-4888-a438-d23693507c21"
}
```

Cancel immediately terminates the training FunctionCall and its single-purpose
GPU container:

```bash
curl --fail-with-body -X POST \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  "$WANGP_URL/training/jobs/$TRAINING_JOB_ID/cancel"
```

Pause is valid only while a job is running. Resume is valid only for a
successfully paused job. Cancel is idempotent for an already cancelled job;
invalid lifecycle transitions return `409 Conflict`.

## WanGP generation API

### Discover models and settings

List model metadata:

```bash
curl --fail-with-body \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  "$WANGP_URL/models"
```

The model list supports optional filters:

```bash
# Filter by family.
curl --get --fail-with-body \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  --data-urlencode "family=qwen" \
  "$WANGP_URL/models"

```

Get the native defaults or schema for one model:

```bash
MODEL="qwen_image_2512_20B"

curl --fail-with-body \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  "$WANGP_URL/models/$MODEL/defaults"

curl --fail-with-body \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  "$WANGP_URL/models/$MODEL/schema"
```

Model metadata, defaults, and schemas are generated from the pinned WanGP
runtime while the Modal image is built. These endpoints are served from the
baked JSON catalog by the CPU API container and never cold-start a GPU.
`available=true` is not supported by the static catalog because checkpoint
availability can change independently of the deployed image.

### Submit a generation job

`POST /jobs` accepts a model name and an opaque dictionary of native WanGP
settings:

```json
{
  "kind": "image",
  "model": "qwen_image_2512_20B",
  "params": {
    "prompt": "A red fox in snow, cinematic natural light",
    "resolution": "1024x1024",
    "num_inference_steps": 4,
    "seed": 12345,
    "spatial_upsampling": "off"
  }
}
```

`kind` accepts `image`, `video`, or `audio`. It is optional: the API normally
infers it from the model catalog. Supplying it makes the intended route
explicit, and the API returns `400 Bad Request` if it conflicts with the
selected model. Video jobs route to the H100 pool; image and audio jobs route to
the L40S pool. For a model that can produce both image and video, inference uses
the effective native `image_mode` setting.

Submit it with `curl`:

```bash
curl --fail-with-body -X POST \
  -H "Content-Type: application/json" \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  "$WANGP_URL/jobs" \
  --data-binary @- <<'JSON'
{
  "kind": "image",
  "model": "qwen_image_2512_20B",
  "params": {
    "prompt": "A red fox in snow, cinematic natural light",
    "resolution": "1024x1024",
    "num_inference_steps": 4,
    "seed": 12345,
    "spatial_upsampling": "off"
  }
}
JSON
```

The endpoint returns HTTP `202 Accepted` immediately:

```json
{
  "id": "ba2972c6-65f3-44ec-8368-38708e99c28d",
  "status": "queued",
  "kind": "image"
}
```

WanGP defaults are loaded first and then overlaid with `params`. The service
always sets `model_type` from the top-level `model`; clients should not put it
in `params`. The `_api` key is reserved and rejected.

### Important flags

The object returned by `GET /models/{model}/defaults` is a starting payload,
not an exhaustive allowlist. WanGP may accept optional native settings that are
not present in the defaults. Always consult `GET /models/{model}/schema` for
capabilities, supported value choices, model definitions, and additional
settings. Because `params` is intentionally opaque, supported native WanGP
settings pass through without requiring a REST API change.

Important general flags include:

- `seed`: reproducible generation seed; commonly accepted even when omitted
  from a model's defaults. Use `-1` when the model supports random seeds.
- `resolution`: output dimensions in WanGP's `WIDTHxHEIGHT` format.
- `num_inference_steps`: generation step count; distilled/turbo models usually
  require their documented low step count.
- `batch_size`: number of outputs generated by the task.
- `negative_prompt`, `guidance_scale`, `flow_shift`, and
  `denoising_strength`: model-dependent generation controls.
- `activated_loras`: list of LoRA paths or identifiers understood by WanGP.
- `loras_multipliers`: matching LoRA weights. Keep its ordering aligned with
  `activated_loras`.
- `spatial_upsampling`: WanGP upscaling mode; use `"off"` when upscaling should
  be performed as a separate operation.
- `_api`: reserved for the REST service and rejected in client payloads.
- `model_type`: controlled by the top-level `model` field and overwritten by
  the service.

For the currently installed `krea2_turbo` checkpoint, the normal text-to-image
starting point is:

```json
{
  "model": "krea2_turbo",
  "params": {
    "prompt": "A red fox walking through fresh snow at golden hour",
    "negative_prompt": "blurry, low quality",
    "resolution": "1024x1024",
    "num_inference_steps": 8,
    "seed": 12345,
    "batch_size": 1,
    "guidance_scale": 0,
    "flow_shift": 5.0
  }
}
```

Krea2 Turbo supports text-to-image, LoRAs, and inpainting. Its schema currently
reports no image-to-image, reference-image, video, or audio capability. For
inpainting, use `masking_strength`, `denoising_strength`, and `model_mode`:

- `2`: LanPaint 2 steps, easy task.
- `3`: LanPaint 5 steps, medium task.
- `4`: LanPaint 10 steps, hard task.
- `5`: LanPaint 15 steps, very hard task.

### Poll job status and progress

```bash
export JOB_ID="ba2972c6-65f3-44ec-8368-38708e99c28d"

curl --fail-with-body \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  "$WANGP_URL/jobs/$JOB_ID"
```

Possible status values are `queued`, `running`, `succeeded`, `failed`, and
`cancelled`. While a job is running, the response includes its latest structured
WanGP progress when available:

```json
{
  "id": "ba2972c6-65f3-44ec-8368-38708e99c28d",
  "status": "running",
  "model": "qwen_image_2512_20B",
  "progress": {
    "phase": "inference",
    "status": "Prompt 1/1 | Denoising",
    "progress": 50,
    "current_step": 2,
    "total_steps": 4
  },
  "created_at": "2026-08-13T08:00:00+00:00",
  "started_at": "2026-08-13T08:00:02+00:00",
  "updated_at": "2026-08-13T08:00:20+00:00"
}
```

A successful terminal response contains storage-agnostic metadata and protected
download URLs for the files stored in the shared Volume:

```json
{
  "id": "ba2972c6-65f3-44ec-8368-38708e99c28d",
  "status": "succeeded",
  "model": "qwen_image_2512_20B",
  "result": {
    "success": true,
    "outputs": [
      {
        "id": "2d457dea-9cc8-436f-ad85-d72fefec2343",
        "filename": "2026-08-13-result.png",
        "size_bytes": 1842201,
        "media_type": "image/png",
        "url": "/outputs/2d457dea-9cc8-436f-ad85-d72fefec2343"
      }
    ],
    "total_tasks": 1,
    "successful_tasks": 1,
    "failed_tasks": 0,
    "errors": []
  }
}
```

Download an output using its `url`. The endpoint uses the same Modal proxy
authentication as every other route:

```bash
OUTPUT_URL="/outputs/2d457dea-9cc8-436f-ad85-d72fefec2343"

curl --fail-with-body --location \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  --output "result.png" \
  "$WANGP_URL$OUTPUT_URL"
```

The public result does not expose the internal `/data` filesystem path. Download
URLs remain valid while their output and job records remain in the Modal Dict.
The API mounts `wangp-data` read-only and reloads it before serving a file.

### Cancel a job

```bash
curl --fail-with-body -X POST \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  "$WANGP_URL/jobs/$JOB_ID/cancel"
```

Cancellation terminates the job's Modal FunctionCall and its single-purpose GPU
container. Repeating cancellation for a cancelled job is safe. Cancelling a job
that has already succeeded or failed returns HTTP `409 Conflict`.

### Reference images, videos, audio, and LoRAs

There is no upload endpoint in v1. Place input assets in the shared `wangp-data`
Volume first, then use their absolute `/data/...` paths in the native WanGP
parameters:

```json
{
  "model": "i2v_2_2",
  "params": {
    "prompt": "The subject turns toward the camera",
    "image_start": "/data/inputs/portrait.png",
    "resolution": "832x480",
    "video_length": 81
  }
}
```

Virtual-media suffixes are supported because WanGP receives the path unchanged:

```json
{
  "video_guide": "/data/inputs/source.mp4|start_frame=120,end_frame=240"
}
```

Absolute paths outside `/data` and paths that traverse outside `/data` are
rejected. URLs are not downloaded by the REST layer.

### HTTP errors and retention

- `400 Bad Request`: reserved `_api` settings or an invalid filesystem path.
- `401/403`: missing or invalid Modal proxy credentials.
- `404 Not Found`: unknown model, job, or expired job result.
- `409 Conflict`: cancellation requested after successful or failed completion.
- `422 Unprocessable Entity`: malformed JSON or an invalid request shape.
- `503 Service Unavailable`: the Fizgig deployment or a training lifecycle
  operation could not be reached.
- `500 Internal Server Error`: unexpected control-plane failure.

Generation failures normally produce a terminal `failed` job record with
structured errors rather than turning the polling request into an HTTP 500.
Generation records use `wangp-rest-jobs`; public training records use
`fizgig-rest-jobs`; worker-owned training state uses `fizgig-modal-jobs`.

## Test

```bash
uv run pytest
python3 -m modal deploy --help
```

WanGP and Wan2AI revisions are pinned in `app.py` for reproducible builds.
