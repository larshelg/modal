# WanGP Modal App

For the **Sol-only video refiner**, use [the Sol worker guide](docs/sol-refiner.md)
and deploy `sol_app.py`. That entrypoint has its own image, model Volume, and
client (`sol_control.py`); it does not load WanGP. The documentation below covers
the existing WanGP entrypoint.

This project runs asynchronous WanGP image, video, and audio generation on
Modal without an HTTP or REST layer. It consists of a deployed GPU worker app
and a local Modal CLI:

- Video models run on `WanGPVideoWorker`, which uses H100 by default.
- Image and audio models run on `WanGPImageWorker`, which uses H100 by default.
- Models supporting both image and video follow their native `image_mode`
  setting unless `--kind` is supplied explicitly.

The CLI in `control.py` validates requests locally and sends them directly to
the correct deployed class in `wangpt-modal-app`. Jobs continue after the local
command exits, and their records are stored in the `wangpt-modal-jobs` Modal
Dict.

## Architecture

The worker app in `app.py` contains:

- `WanGPImageWorker`: H100 image/audio execution pool.
- `WanGPVideoWorker`: H100 video execution pool.
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
export WANGP_IMAGE_GPU=H100
export WANGP_IMAGE_MAX_CONTAINERS=3
export WANGP_IMAGE_MEMORY_MB=65536
export WANGP_IMAGE_PROFILE=4

export WANGP_VIDEO_GPU=H100
export WANGP_VIDEO_MAX_CONTAINERS=1
export WANGP_VIDEO_MEMORY_MB=131072
export WANGP_VIDEO_PROFILE=4

python3 -m modal deploy app.py
python3 -m modal run control.py::refresh_catalog
```

`WANGP_GPU` and `WANGP_MAX_CONTAINERS` remain aliases for the image worker.
Set `WANGP_MODEL_LOAD_TRACE_INTERVAL_SECONDS=0` to disable periodic stack dumps
during slow model loading.

## Krea2 Turbo L40S snapshot worker

`krea_app.py` deploys `WanGPKreaWorker` in the independent Modal app
`wangpt-krea-modal-app`. It accepts only `krea2_turbo` and retains the existing
`run(job_id, model, params)` contract, shared job store, and verified S3 outputs.
It shares the pinned WanGP image build/runtime with the other workers. Deploying
it does not redeploy the main app or invalidate its Singularity snapshots.

The worker uses L40S, profile 1, 64 GiB host RAM, zero minimum containers, one
maximum container, and a 300-second idle timeout. RAM, maximum containers, and
idle timeout can be set before deployment with `WANGP_KREA_MEMORY_MB`,
`WANGP_KREA_MAX_CONTAINERS`, and `WANGP_KREA_SCALEDOWN_WINDOW`.

Native 1024x1024, eight-step warmup runs privately before capture. After joined
task cleanup, MMGP reloads only the transformer to CUDA; text encoder and VAE
remain loaded in CPU memory. Capture and immediate restore validate full tensor
placement and the L40S device. The restore hook never reloads weights to hide a
failed restoration. Per-request diagnostics report pipeline reuse and native
model-loader calls. User LoRAs load normally during requests.

```bash
.venv/bin/python -m modal deploy krea_app.py
python3 -m modal run control.py::krea_snapshot_probe
WANGP_KREA_SNAPSHOT=1 python3 -m modal run control.py::submit \
  --model krea2_turbo --kind image --params-file examples/krea2_turbo.json
```

The CLI flag is opt-in and affects only the exact image model `krea2_turbo`;
other models, including `krea2_turbo_edit`, retain their normal routes. A direct
TypeScript client can look up `WanGPKreaWorker` in `wangpt-krea-modal-app` and call
`run` with its existing job ID, model, and native parameters. No CLI flag is
needed for direct class calls. See [Krea validation](docs/krea-snapshot.md).

## Singularity GPU snapshot experiment

`WanGPSingularityWorker` warms the exact
`minimax_h3_ref2va_singularity_pruned` preset, including its built-in 4-step LoRA,
in `@modal.enter(snap=True)`. It uses a synthetic reference image and removes
warmup artifacts without publishing a job or uploading to S3. After native
cleanup, MMGP's `ensure_model_loaded("transformer")` stages only the transformer
on CUDA. Capture and immediate restore checks require all transformer tensors
on CUDA, other components on CPU, and at least 19 GiB allocated (the INT8
transformer is about 19.6 GiB). A failed restore check never reloads the model to
hide the failure. See [snapshot.md](snapshot.md) for deployed validation.
WanGP also removes active LoRA adapters at task completion. Warmup verifies the
accelerator during inference; it is loaded again through the native path on
subsequent requests. An empty idle `active_loras` list is therefore expected.

The worker has independent H100 settings: `WANGP_SINGULARITY_PROFILE=1`,
`WANGP_SINGULARITY_MEMORY_MB=131072`, `WANGP_SINGULARITY_MAX_CONTAINERS=1`, and
`WANGP_SINGULARITY_SCALEDOWN_WINDOW=300`. Set these before deployment if needed.
Generic image and video workers keep their existing settings.
This transformer-resident revision requires profile 1. Native inference and
cleanup remain in control of later CPU/GPU transfers; a request may offload the
captured transformer while encoding text. No extra components or identity LoRAs
are staged. Rolling back to the CPU/offload baseline requires the previous
worker revision, not just a profile override.

Deploy the updated `app.py`, then explicitly submit Singularity to the snapshot
pool from the local client:

```bash
WANGP_SINGULARITY_SNAPSHOT=1 python3 -m modal run control.py::submit \
  --model minimax_h3_ref2va_singularity_pruned \
  --params-file job-singularity.json
```

Only the exact value `1` enables this route. Without it, Singularity still uses
the ordinary video worker. All other models keep their usual routes regardless
of this switch. Job records include the selected `worker`.

To initialize or inspect the dedicated pool without publishing a generation:

```bash
python3 -m modal run control.py::snapshot_probe
```

This command starts a billable GPU container and runs warmup when a snapshot is
not available. Its result reports capture/boot IDs, warmup timing, loaded
configuration and accelerator, CUDA memory, logical tensor placement, and
separate capture/immediate-restore reports. The stored restore report precedes
all inference and any reloading; current memory can differ after generation.
A shared capture ID across distinct boot IDs is evidence of reused captured
state; confirm actual restoration in Modal's container view/logs. The first
call can create a snapshot and is not a restore benchmark.

Snapshot logs also report request duration, first denoising progress, native
`load_models` calls, and whether the model object was reused. First-denoising
progress is recorded only when the total matches the explicitly requested step
count, excluding the text encoder's 50-step transition. Timing starts at method entry and
excludes queueing/container startup; combine it with client/job timestamps for
end-to-end latency. A changed request configuration may legitimately reload
the model. Bump `SNAPSHOT_REVISION` in `h3_snapshot.py` when changing external
assets: Volume changes alone do not invalidate snapshots.

Rollback for new requests is simply omitting/unsetting
`WANGP_SINGULARITY_SNAPSHOT`; in-flight jobs are not retried or cancelled.

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

### H3 Singularity

`minimax_h3_ref2va_singularity_pruned` is a custom Singularity v1.3 Ref2VA
20B preset. Its definition in `finetunes/` pins the author's pruned INT8
checkpoint and the LightX2V Ref2V 4-step LoRA to immutable Hugging Face
revisions. The accelerator loads automatically at strength 1.0. Defaults are
4 Euler steps, guidance 1, flow shift 12, and one guidance phase.

```bash
python3 -m modal run control.py::submit \
  --model minimax_h3_ref2va_singularity_pruned \
  --kind video --params-file job-singularity.json
```

The example is a text-only smoke test. For reference-image conditioning, add
`"video_prompt_type": "I"` and `"image_refs": ["s3://YOUR_S3_BUCKET/input.jpg"]`,
then refer to `<Picture 1>` in the prompt. The first run downloads the roughly
21 GB checkpoint plus any missing shared H3 components to `wangp-data`.

Custom definitions are copied into the image before catalog generation. After
changing `finetunes/`, deploy `app.py` and run `control.py::refresh_catalog`;
the catalog cache is keyed by the upstream WanGP commit, so custom preset
changes require an explicit refresh even when that commit stays the same.

### H3 latent checkpoints and continuation

The worker includes [H3 Latent Continue](https://github.com/g3n3rativ3/wan2gp-h3-latent-continue)
0.3.4, pinned to `6b1334823e2acff297d8ab05b0e17eb1ba6a4a3b`. `h3_latent.py`
installs its existing generation/export hooks without starting Gradio. The
headless bridge is initialized by the worker; GUI `enabled_plugins` configuration
is not required. Latent operations are disabled unless requested in a job.

Generate an initial Singularity clip and save its original inference latents:

```bash
python3 -m modal run control.py::submit \
  --model minimax_h3_ref2va_singularity_pruned \
  --kind video --params-file job-singularity-latent.json
```

The flat params file includes this extra field, which the worker forwards to
WanGP's native task envelope:

```json
{
  "plugin_data": {
    "h3_latent_prototype": {
      "save": true,
      "continue": false,
      "join_mode": "baseline"
    }
  }
}
```

A successful job with `save: true` must return both the MP4 and its matching
`.safetensors` checkpoint in `result.outputs`. Both are uploaded to S3, verified,
and removed from worker scratch storage. The checkpoint is an output artifact,
not a model weight or LoRA.

To continue, keep the same model, resolution and FPS, set `image_prompt_type`
to `"V"`, and set `video_source` to the original MP4's S3 URI. In the options
above, set `continue` to `true` and `latent_path` to the matching checkpoint's
S3 URI. Keep `save: true` to obtain a fresh checkpoint. Use the exact original
MP4; the plugin verifies its bytes against the checkpoint. Before continuation,
the worker also checks the source video's frame count against the checkpoint's
exported frame count. A matching SHA-256 alone does not establish matching
latent/video endpoints. Checkpoints without an exported frame count are rejected.

`job-singularity-latent-continue.json` contains the actual S3 pair from the
initial smoke test and can be submitted directly:

```bash
python3 -m modal run control.py::submit \
  --model minimax_h3_ref2va_singularity_pruned \
  --kind video --params-file job-singularity-latent-continue.json
```

Replace both URIs with the matching outputs from your own initial job when
continuing another clip. The example uses `join_mode: "baseline"` and an
assembly overlap of one frame; plugin 0.3.4 still supplies at least 18 frames
of saved latent history when available.

Supported base architectures are H3 FL2VA/Ref2VA, full or pruned, including the
Singularity preset. Use one guidance phase, batch size one, and one generated
window per job. Two-phase upscaling, output postprocessing, frame trimming,
color correction, extra audio refinement, PDD and VDN are unsupported by this
plugin. Continuation is experimental; retaining original latents does not
guarantee a seamless visual or audio join.

Before the mux fix, an H100 test with Singularity on 2026-09-21 showed: initial job
`27db2109-da3d-4dbb-a5a6-9722009a27f0` saved 124 frames and a 10.4 MB checkpoint;
continuation job `b2d86b94-adff-4c24-a10f-6e521893f5c3` saved a new video and
checkpoint. Both jobs succeeded at 832×480 / 24 FPS with four Euler steps.
The continuation log confirmed direct latent context loading without VAE
re-encoding, and its checkpoint recorded `continuation_mode: "latent"`.
Both S3 pairs were downloaded and their SHA-256 checksums and video/checkpoint
bindings verified. However, the continued MP4 contains only 243 video frames
(10.125 seconds), while its checkpoint records 247 frames and the audio lasts
10.292 seconds. This reproduces the upstream plugin's documented
[post-mux frame-count limitation](https://github.com/g3n3rativ3/wan2gp-h3-latent-continue/blob/6b1334823e2acff297d8ab05b0e17eb1ba6a4a3b/DIAGNOSIS-0.3.2.md).
Do not chain that continuation pair: the worker's frame-count guard rejects it.
The original 124-frame pair remains valid for retrying continuation.

The worker now applies `patches/h3-latent-mux.patch` at image build time. For
active latent jobs, `h3_mux.py` supplies the exact encoded video duration to
WanGP's audio muxer. Each audio track is padded/trimmed to that duration, and
all encoded video packets are copied without `-shortest`. Ordinary jobs keep
the existing mux behavior. Frame count, FPS and dimensions are checked before
and after muxing; final output is checked again after metadata writing, before
the plugin writes its checkpoint. A mismatch fails the job instead of publishing
a successful latent checkpoint.

GPU verification of the deployed fix on 2026-09-21 used two consecutive
Singularity continuations from the intact 124-frame source:

- `7ccc0f9b-34ef-4137-99ea-12d742d9c2cf`: 247 decoded video frames and 247
  checkpoint output frames (10.291667 seconds).
- `ab596019-20fb-4559-a85a-748b9cf6f1c9`: continued that corrected pair to 370
  decoded frames and 370 checkpoint output frames (15.416667 seconds).

Both jobs succeeded at 832×480 / 24 FPS, four Euler steps, baseline join mode.
Both output pairs were downloaded and their SHA-256/video bindings verified;
both checkpoints recorded `continuation_mode: "latent"`. Final decoded frames
were also compared to the checkpoints' saved final-frame tensors. This verifies
the frame-alignment fix across two continuations; extended-context and frozen
audio-prefix modes were not tested, and visual/audio joins remain experimental.

`job-singularity-latent-continue-2.json` is the tested second continuation request,
using the corrected first continuation's S3 pair. Submit it with the same model
and `--kind video` as the other examples.

All 75 tests passed, including the CPU regression against the real pinned
WanGP mux function with short,
exact-duration, long, multiple-track and silent audio. It reproduces frame loss
without the fix and verifies identical video packet hashes/timestamps with the
fix. To run it along with the unit tests, point to a checkout of our `WAN_COMMIT`:

```bash
WANGP_TEST_ROOT=/path/to/Wan2GP python3 -m pytest -q
```

### S3 inputs

Native WanGP image fields can reference an object in the configured
`studio-s3` bucket with an `s3://` URI. For example, an H3 image-to-video
request can use:

```json
{
  "image_start": "s3://YOUR_S3_BUCKET/inputs/start.png",
  "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\nintegrated_multimodal_description: [Shot 1] Continue directly from <Picture 1> as the subject slowly turns toward the camera. One continuous shot.\noverall_soundscape: Gentle wind and quiet natural ambience.\nnon_diegetic_music: N/A",
  "resolution": "832x480",
  "video_length": 124,
  "num_inference_steps": 20,
  "guidance_phases": 1,
  "seed": -1
}
```

Submit it normally with `--model minimax_h3_fl2va_pruned --kind video`.
`image_start`, `image_end`, `image_guide`, `image_mask`, and entries in
`image_refs` support S3 URIs, as do `video_source` and the H3 plugin's
`latent_path`. The URI bucket must equal `S3_BUCKET`; credentials,
query strings, fragments, and cross-bucket reads are rejected. The worker
downloads each object to job-local scratch storage, verifies its size and any
available `sha256` object metadata, and removes it when the job ends. Inputs are
limited to 128 MiB each by default; set `WANGP_S3_INPUT_MAX_BYTES` on the worker
deployment to change that limit.

## List LoRAs

The `loras` entrypoint calls the Modal Volume API from the local process. It
does not start either app or any remote container:

```bash
python3 -m modal run control.py::loras
python3 -m modal run control.py::loras --family krea2
python3 -m modal run control.py::loras --family h3
python3 -m modal run control.py::loras --recursive
```

The `h3` alias selects the canonical `loras/minimax_h3` directory.

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


.venv/bin/python -m modal run control.py::submit \
    --model minimax_h3_ref2va_singularity_pruned \
    --kind video \
    --params-file job-singularity.json
