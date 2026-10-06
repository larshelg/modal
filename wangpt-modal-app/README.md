# WanGP Modal App

For the **Sol-only video refiner**, use [the Sol worker guide](sol_refiner/README.md)
and deploy `sol_refiner/app.py`. That entrypoint has its own image, model Volume, and
client (`sol_refiner/control.py`); it does not load WanGP. The documentation below covers
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

The independent model snapshot apps live in folders named after their Modal
deployments:

- [wangpt-krea-modal-app](wangpt-krea-modal-app/README.md)
- [wangpt-krea-raw-modal-app](wangpt-krea-raw-modal-app/README.md)
- [wangpt-krea-raw-edit-modal-app](wangpt-krea-raw-edit-modal-app/README.md)
- [wangpt-krea-edit-modal-app](wangpt-krea-edit-modal-app/README.md)
- [wangpt-qwen-image-21-modal-app](wangpt-qwen-image-21-modal-app/README.md)
- [wangpt-qwen-noctq-modal-app](wangpt-qwen-noctq-modal-app/README.md)

Each folder contains `app.py`, `snapshot.py`, `README.md`, `examples/` and
`tests/`. Shared runtime code, model definitions and `control.py` remain at the
workspace root. Run commands from that root, using Modal module mode for these
packages; for example:

```bash
.venv/bin/python -m modal deploy -m wangpt-qwen-noctq-modal-app.app
```

Hyphenated folder names are loaded through Python's module importer; use `-m`
when deploying these apps so their relative imports resolve correctly.

`snapshot_common.py` holds shared memory reporting, snapshot logging, random
reseeding, model-load tracking, active LoRA inspection and synthetic reference
creation. All snapshot workers include it in their images. Model-specific
warmup and residency checks live in each app's `snapshot.py`; Singularity uses
`h3_snapshot.py` for its H3-specific logic.

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

WanGP is pinned in `wangpt_common.py` to
[`b8b18f8114e432eea8f3d7e853a51dd91fa99571`](https://github.com/deepbeepmeep/Wan2GP/commit/b8b18f8114e432eea8f3d7e853a51dd91fa99571),
the upstream `main` head checked on 2026-10-01. The H3 audio mux patch is rebased
for this revision's audio encoding helper. All 146 local tests pass with
`WANGP_TEST_ROOT` pointing to this revision, including the native FFmpeg regressions.
The `wangpt-modal-app` deployment and its 237-model catalog were updated to this
revision on 2026-10-01. GPU generation has not yet been smoke-tested after this upgrade.

## MiniMax H3 RefMods

[MiniMaxH3Mod-for-WanGP](https://github.com/g3n3rativ3/MiniMaxH3Mod-for-WanGP)
0.31.0 is installed in the worker image at pinned commit
`bff0a8554ae4f6f916e38eaa0ef9a1af03fd437b`. `h3_refmod.py` activates its
pipeline hooks and refreshes model definitions for headless API requests.
The build checks that RefMod custom settings are present in the model catalog.
No GUI plugin toggle is needed. Existing WanGP dependencies cover the plugin;
its optional OpenCV backend is not installed separately.

The plugin library uses `/data/refmods` on `wangp-data`, shared with Fizgig's
completed RefMods. Select files by name without `.safetensors`, for example
`linda_refmod_v1` or `characters/linda`. For a manually supplied file:

```bash
python3 -m modal volume put wangp-data ./linda_refmod_v1.safetensors /refmods/linda_refmod_v1.safetensors
```

Use an H3 **Ref2VA** model (including the Singularity Ref2VA finetune), and pass
the plugin's native JSON string in `custom_settings.h3_refmod_state`:

```bash
python3 -m modal run control.py::submit \
  --model minimax_h3_ref2va_singularity_pruned --kind video \
  --params-file examples/h3-refmod.json
```

Edit the example's mod name to match an existing file. Its `rows` contain
`mod`, `strength`, and optional `copies`; `retention` scales all strengths.
These settings can coexist with the existing latent-continuation `plugin_data`.
The plugin's Gradio Extract/Library/Generate tabs are not served by this app;
use `fizgig-modal-app` to create RefMods with its tracked artifact workflow.
After adding new Volume files, use a fresh worker container so it sees the
latest Volume snapshot; warm workers do not reload open model files.

Redeploy `app.py` and run `control.py::refresh_catalog` to activate this change.
The independent `wangpt-krea-modal-app/app.py` deployment is unaffected until separately deployed.

### Experimental numbered image and video RefMods

Set `text_encode: true` inside the JSON string in
`custom_settings.h3_refmod_state`. This app extension decodes each active visual
RefMod through H3's loaded video VAE and presents the reconstructed media to
the native text/vision encoder. The same strength/curve-adjusted latent is used
for generation, without re-encoding the reconstructed pixels or applying the
RefMod twice. Basic requests without this option retain upstream behavior.

Start with `examples/h3-refmod-numbered.json` (image) or
`examples/h3-refmod-numbered-video.json` (image + video); replace the example
library names with actual files. The selection JSON, before string encoding, is:

```json
{
  "text_encode": true,
  "rows": [
    {"mod": "identity_refmod", "strength": 1.0, "copies": 1},
    {"mod": "motion_refmod", "strength": 1.0, "reference_fps": 24}
  ],
  "retention": 1.0,
  "scramble_seed": -1
}
```

With no other images or videos, the above image/video pair becomes
`<Picture 1>` and `<Video 1>`. Define subjects explicitly in the prompt, e.g.
`<Subject 1> is the person from <Picture 1>`. Normal references and start/end
frames also consume labels. Image RefMods follow normal image references;
video RefMods fill free native video slots. Images, videos and audio are
numbered independently. Multi-image RefMods contribute one Picture per latent
frame, and image `copies` repeat those entries. Video `copies` repeat time
within the same Video entry. Zero-strength rows are omitted. Scrambling is
rejected in numbered mode so the requested ordering stays predictable.

#### Reference ordering in prompts

With `text_encode: true`, **image RefMods are appended after the normal
`image_refs`**, preserving the order of both `image_refs` and the image RefMod
rows. A RefMod is therefore not always the second reference. For single-image
RefMods with `copies: 1`, and no start/end frames or continuation:

- One image RefMod alone: `<Picture 1>` = RefMod.
- One normal image + one image RefMod: `<Picture 1>` = `image_refs[0]`,
  `<Picture 2>` = RefMod.
- Two normal images + one image RefMod: `<Picture 1>` = `image_refs[0]`,
  `<Picture 2>` = `image_refs[1]`, `<Picture 3>` = RefMod.
- One normal image + two image RefMods: `<Picture 1>` = `image_refs[0]`,
  `<Picture 2>` = first image RefMod, `<Picture 3>` = second image RefMod.

For example, when the normal image supplies a location and the RefMod supplies
a person's identity, bind them explicitly:

```text
subject_definitions: <Subject 1> is the person from <Picture 2>.
summary: [reference generation] <Subject 1> stands in the courtyard from <Picture 1>.
```

`<Subject 1>` is a prompt-defined subject, not reference slot 1. The basic
latent-only RefMod path (`text_encode` absent or `false`) does not assign the
RefMod one of these numbered media labels.

Start/end frames and other native image conditions can shift Picture numbers.
In single-window latent continuation, the preceding clip's saved final frame
occupies `<Picture 1>`; with no other image inputs, the first image RefMod is
`<Picture 2>`. A video RefMod uses the separate `<Video N>` sequence and does
not consume a Picture number. With no normal reference videos, the first video
RefMod is `<Video 1>`, regardless of how many Picture references there are.

Use `result.refmod_reference_maps` as the authoritative mapping for each job.
The tested normal-image + RefMod request is
[job-refmod-normal-image-order.json](runs/job-refmod-normal-image-order.json);
its [verification record](runs/refmod-normal-image-order-verification.json)
links the native `<Picture 1>` to the Krea image URI and identifies the RefMod
as `<Picture 2>`.

Results include `refmod_reference_maps`, with the actual labels, library names,
row indices and image frame/copy indices for each generation. The worker logs
the mapping before text encoding. Native media entries are marked `source:
native`. RefMod indices are zero-based; prompt labels start at 1.

Video frames are sampled at two per second for Qwen using `reference_fps`
(per row, or at selection level; default 24). This describes reconstructed
playback for text encoding, not output FPS, and cannot recover timing lost by
compression. Native video trimming is reflected in the decoded presentation.
Both H3 denoising phases reuse decoded references and verify consistent labels.
Warm workers discard decoded references after each generation; WanGP's prompt
cache includes the reconstructed pixels and timestamps.

Numbered mode supports a **single window**, including image-only or video-only
RefMod selections and Singularity continuation. Set
`video_length <= sliding_window_size`. For continuation, the assembly overlap
also consumes capacity: `video_length + sliding_window_overlap - 1` must fit
within `sliding_window_size`. Using `sliding_window_overlap: 1` leaves saved
latent history controls independent of that capacity. Sliding windows,
window prompt modes and frame-scheduler slash commands are rejected. Numbered
video RefMods cannot share a request with video control/excerpt modes or
reference-video soundtrack extraction. Audio RefMods are not included in this
extension; ordinary audio reference inputs remain available. Native limits of
9 image / 3 video / 3 audio references and 12 references total apply. Missing
files, failed decoding, unsupported modes or missing presentation entries fail
the job instead of falling back to unconditioned generation.

For Singularity latent continuation, combine `text_encode: true` with the usual
`plugin_data.h3_latent_prototype` continuation settings and the matching original
MP4/checkpoint pair. See `examples/h3-refmod-numbered-continue.json`; replace its
paths and RefMod names. The continuation's saved final frame is `<Picture 1>`,
so the first image RefMod is `<Picture 2>` when no other image inputs precede
it. Video references have separate numbering, starting at `<Video 1>`. Check
`result.refmod_reference_maps` for the actual labels. Each request generates
one window; with `save: true`, its output pair can be continued in a new request.
Existing latent continuation restrictions (including single-phase generation)
still apply. `patches/h3-latent-native.patch` updates the pinned continuation
plugin's capture insertion point for the current WanGP source; image builds
check every native generation insertion point before deployment.
The headless integration also registers its private task-options snapshot as
transport metadata with WanGP's setting-name validator; callers continue to
provide only the public `plugin_data.h3_latent_prototype` options.

VAE decoding and vision encoding add runtime and memory cost. Numbered binding
is experimental; labels do not guarantee identity separation or motion transfer.
The CPU tensor regressions run during the Modal image build, even when local
PyTorch is unavailable.

Verified on `WanGPVideoWorker` with Singularity Ref2VA on 2026-10-03:
`61c0e893-e2e4-4b91-b970-780495f2f28a` completed with the image RefMod mapped to
`<Picture 1>`; `da618fd0-b412-4c84-a630-752422801cdb` completed with image and
video RefMods mapped to `<Picture 1>` and `<Video 1>`. The matching image
baseline (`1cde92d7-b155-4e22-a4e6-4de4817bce91`) also completed. Requests,
job results and downloaded MP4s are under `runs/`. These tests verify execution
and mapping, not general identity or motion-transfer quality. The retained
`wangp_numbered_video_test_20261003` library fixture was encoded from our
previously generated garden video using Fizgig's community motion preset.

Singularity numbered RefMod + latent continuation was verified on 2026-10-03
through `WanGPVideoWorker`: source job `0483fb7b-ed55-43fa-a948-df30384e671f`
saved 107 frames and continuation job `7f215331-0139-458d-86f5-460754438949`
saved 213 frames, at 832×480 / 24 FPS with four Euler steps and baseline join
mode. Both used image and video RefMods. The continuation map contained the
native `<Picture 1>`, identity RefMod `<Picture 2>`, and motion RefMod `<Video 1>`.
Both downloaded video/checkpoint pairs passed artifact hashes, video binding,
and decoded frame-count checks; the continuation checkpoint recorded
`continuation_mode: "latent"`. The exact requests are
`runs/job-refmod-numbered-latent-source.json` and
`runs/job-refmod-numbered-latent-continue.json`. The original
`runs/job-singularity-latent.json` remains the basic fox example without RefMods.
Extended-context and audio-prefix join modes were not part of this GPU test.

Normal image + numbered RefMod ordering was also verified on 2026-10-03.
`WanGPKreaWorker` job `29e5cdb4-ca02-41d0-8a33-474aaa43d45a` created a courtyard
reference. Fresh Singularity job `7e3560c0-45e1-4983-bd72-6a671f702a75` supplied
that S3 image through `image_refs[0]` alongside the identity RefMod. Its encoder
map confirmed `<Picture 1>` = normal image and `<Picture 2>` = identity RefMod;
the output contained 124 frames at 24 FPS with no continuation inputs. The exact
request is `runs/job-refmod-normal-image-order.json`, and the asserted mapping,
input URI/hash and output checks are in
`runs/refmod-normal-image-order-verification.json`. This verifies reference
ordering and successful generation, not a quantitative identity-fidelity score.

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

`wangpt-krea-modal-app/app.py` deploys `WanGPKreaWorker` in the independent Modal app
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
.venv/bin/python -m modal deploy -m wangpt-krea-modal-app.app
python3 -m modal run control.py::krea_snapshot_probe
WANGP_KREA_SNAPSHOT=1 python3 -m modal run control.py::submit \
  --model krea2_turbo --kind image --params-file wangpt-krea-modal-app/examples/krea2_turbo.json
```

The CLI flag is opt-in and affects only the exact image model `krea2_turbo`;
other models, including `krea2_turbo_edit`, retain their normal routes. A direct
TypeScript client can look up `WanGPKreaWorker` in `wangpt-krea-modal-app` and call
`run` with its existing job ID, model, and native parameters. No CLI flag is
needed for direct class calls. See [Krea validation](wangpt-krea-modal-app/README.md).

## Krea2 RAW L40S snapshot worker

`wangpt-krea-raw-modal-app/app.py` deploys `WanGPKreaRawWorker` independently in
`wangpt-krea-raw-modal-app`, accepting only the base model `krea2_raw`. It uses
L40S, WanGP profile 1, 64 GiB host RAM, zero minimum containers, one maximum
container and a 300-second idle timeout. The runtime, job store, Volume and S3
handling are shared with the other workers.

Private warmup uses 1024x1024, 52 steps, guidance 3.5 and flow shift 5. Capture
keeps the transformer on CUDA and text encoder/VAE on CPU, without active
LoRAs. Shared lifecycle and diagnostic helpers come from `snapshot_common.py`.

```bash
.venv/bin/python -m modal deploy -m wangpt-krea-raw-modal-app.app
.venv/bin/python -m modal run control.py::krea_raw_snapshot_probe
WANGP_KREA_RAW_SNAPSHOT=1 .venv/bin/python -m modal run control.py::submit \
  --model krea2_raw --kind image \
  --params-file wangpt-krea-raw-modal-app/examples/krea2_raw.json
```

The flag affects only this exact image model. Deployment settings use
`WANGP_KREA_RAW_MEMORY_MB`, `WANGP_KREA_RAW_MAX_CONTAINERS` and
`WANGP_KREA_RAW_SCALEDOWN_WINDOW`. See [RAW snapshot details](wangpt-krea-raw-modal-app/README.md).

## Krea2 RAW Edit L40S snapshot worker

`wangpt-krea-raw-edit-modal-app/app.py` deploys `WanGPKreaRawEditWorker` in
`wangpt-krea-raw-edit-modal-app`, accepting only `krea2_raw_edit`. It uses L40S,
profile 1, 64 GiB host RAM, zero minimum containers, one maximum container and
a 300-second idle timeout, with shared runtime, Volume, jobs and S3 handling.

Private warmup edits a synthetic reference at 1024x1024, 20 steps, guidance 2
and flow shift 5. It verifies the preset Identity Edit v1.2 LoRA during inference.
Capture retains the transformer on CUDA and text/vision encoders and VAE on CPU.
Native cleanup unloads task adapters; each request activates them normally.

```bash
.venv/bin/python -m modal deploy -m wangpt-krea-raw-edit-modal-app.app
.venv/bin/python -m modal run control.py::krea_raw_edit_snapshot_probe
WANGP_KREA_RAW_EDIT_SNAPSHOT=1 .venv/bin/python -m modal run control.py::submit \
  --model krea2_raw_edit --kind image \
  --params-file wangpt-krea-raw-edit-modal-app/examples/krea2_raw_edit.json
```

Replace the example's reference with an existing `/data` path or S3 URI in the
configured bucket. Deployment settings use `WANGP_KREA_RAW_EDIT_MEMORY_MB`,
`WANGP_KREA_RAW_EDIT_MAX_CONTAINERS` and `WANGP_KREA_RAW_EDIT_SCALEDOWN_WINDOW`.
See [RAW Edit snapshot details](wangpt-krea-raw-edit-modal-app/README.md).

## Krea2 Turbo Edit L40S snapshot worker

`wangpt-krea-edit-modal-app/app.py` deploys `WanGPKreaEditWorker` in the independent Modal app
`wangpt-krea-edit-modal-app`. It accepts only `krea2_turbo_edit`, uses L40S with
profile 1 and 64 GiB host RAM, and shares the runtime, job store, Volume and S3
handling. It scales to zero after 300 idle seconds, with at most one container.

Its private warmup edits a synthetic reference and verifies that the Identity
Edit v1.2 LoRA is active during inference. The snapshot keeps the transformer
on CUDA and the text encoder, vision encoder and VAE on CPU. Native cleanup
unloads task LoRAs; each request loads the preset adapter normally.

```bash
.venv/bin/python -m modal deploy -m wangpt-krea-edit-modal-app.app
.venv/bin/python -m modal run control.py::krea_edit_snapshot_probe
WANGP_KREA_EDIT_SNAPSHOT=1 .venv/bin/python -m modal run control.py::submit \
  --model krea2_turbo_edit --kind image --params-file wangpt-krea-edit-modal-app/examples/krea2_turbo_edit.json
```

Replace the example's reference path with an existing `/data` image or an S3
URI in the configured bucket. The edit route has its own opt-in flag; plain
Turbo routing still uses `WANGP_KREA_SNAPSHOT`. Memory, container limit and idle
timeout use `WANGP_KREA_EDIT_MEMORY_MB`, `WANGP_KREA_EDIT_MAX_CONTAINERS` and
`WANGP_KREA_EDIT_SCALEDOWN_WINDOW`. See [edit snapshot details](wangpt-krea-edit-modal-app/README.md).

## Qwen Image 2.1 7B L40S snapshot worker

`wangpt-qwen-image-21-modal-app/app.py` deploys `WanGPQwenImage21Worker` in the independent Modal
app `wangpt-qwen-image-21-modal-app`. It accepts only `qwen_image_21_7B` and uses
the shared runtime, job store, Volume and S3 handling. The worker uses L40S,
profile 1, 64 GiB host RAM, zero minimum containers, one maximum container and
a 300-second idle timeout.

Its private 40-step reference-image warmup exercises editing and the vision
encoder. Capture keeps the full transformer on CUDA and the text encoder,
vision encoder and VAE on CPU, without active task LoRAs. Both text-to-image
and image editing use the same `run(job_id, model, params)` method.

```bash
.venv/bin/python -m modal deploy -m wangpt-qwen-image-21-modal-app.app
.venv/bin/python -m modal run control.py::qwen_image_21_snapshot_probe
WANGP_QWEN_IMAGE_21_SNAPSHOT=1 .venv/bin/python -m modal run control.py::submit \
  --model qwen_image_21_7B --kind image --params-file wangpt-qwen-image-21-modal-app/examples/qwen_image_21_7B.json
```

The client flag affects only this exact image model. Memory, container limit
and idle timeout use `WANGP_QWEN_IMAGE_21_MEMORY_MB`,
`WANGP_QWEN_IMAGE_21_MAX_CONTAINERS` and `WANGP_QWEN_IMAGE_21_SCALEDOWN_WINDOW`.
See [Qwen snapshot details](wangpt-qwen-image-21-modal-app/README.md).

## Qwen Noct Q V4 L40S snapshot worker

`wangpt-qwen-noctq-modal-app/app.py` deploys `WanGPQwenNoctQWorker` in the independent app
`wangpt-qwen-noctq-modal-app`, accepting only `qwen_image_21_noctq_v4`. It uses
L40S, profile 1, 64 GiB host RAM, zero minimum containers, one maximum container
and a 300-second idle timeout, with the shared runtime, Volume, jobs and S3.

Private warmup edits a synthetic reference at 1024x1024, 25 steps and guidance
3. Capture checks the pinned Noct Q V4 checkpoint, the full transformer on CUDA
and the text encoder, vision encoder and VAE on CPU, with no active task LoRAs.

```bash
.venv/bin/python -m modal deploy -m wangpt-qwen-noctq-modal-app.app
.venv/bin/python -m modal run control.py::qwen_noctq_snapshot_probe
WANGP_QWEN_NOCTQ_SNAPSHOT=1 .venv/bin/python -m modal run control.py::submit \
  --model qwen_image_21_noctq_v4 --kind image \
  --params-file wangpt-qwen-noctq-modal-app/examples/qwen_image_21_noctq_v4.json
```

The flag affects only this exact image model. Deployment settings use
`WANGP_QWEN_NOCTQ_MEMORY_MB`, `WANGP_QWEN_NOCTQ_MAX_CONTAINERS` and
`WANGP_QWEN_NOCTQ_SCALEDOWN_WINDOW`. See [Noct Q snapshot details](wangpt-qwen-noctq-modal-app/README.md).

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
a checked-in [JSON example](wangpt-krea-modal-app/examples/krea2_turbo.json). File-based submission
remains available through the generic entrypoint when useful:

```bash
python3 -m modal run control.py::submit \
  --model krea2_turbo \
  --kind image \
  --params-file wangpt-krea-modal-app/examples/krea2_turbo.json
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

### Qwen Image 2.1 Noct Q V4

`qwen_image_21_noctq_v4` selects Noctaluna's Noct Q V4 7B checkpoint,
pinned to Hugging Face revision `a81b9af51120a78e285e57906f2250a2a02080e9`.
The checkpoint uses INT8 ConvRot. The image applies
`patches/qwen21-fused-checkpoint.patch` to pass the fused `gate_up` split map
into MMGP's checkpoint loader; upstream only supplied that map for LoRAs.
MMGP splits the gate/projection weights and their quantization scales without
dequantizing. Build-time regression tests load fused INT8, already-split INT8,
and fused BF16 checkpoints, and reproduce the missing-key error without the map.
The model inherits the standard Qwen 2.1
text encoder and VAE. The first generation downloads the roughly 7.3 GB
checkpoint and any missing shared components into the persistent Volume.

The preset uses 25 steps and guidance 3, with WanGP's FlowMatch Euler
solver (`sample_solver: "default"`). This is WanGP's native schedule, not
an exact reproduction of the author's ComfyUI `simple` schedule.
The model is distributed under the Qwen Research License for non-commercial
use; see the [model repository](https://huggingface.co/Noctaluna/Noct-Q-Uncensored-Qwen-Image-2.1).

After deploying `app.py` and running `control.py::refresh_catalog`:

```bash
python3 -m modal run control.py::submit \
  --model qwen_image_21_noctq_v4 --kind image \
  --params-file wangpt-qwen-noctq-modal-app/examples/qwen_image_21_noctq_v4.json
```

GPU verification on 2026-10-04 (Europe/Oslo): job
`1b4458a6-96dd-458e-bc65-5ff9d77a400a` succeeded on the H100 image worker
with the example's 1024×1024 teapot prompt, 25 steps, guidance 3, and seed 42.
The saved JPEG was downloaded, verified against its S3 size/SHA-256, and
visually inspected. Local artifacts: `runs/noctq-loader-test.jpg` and
`runs/noctq-loader-test-result.json`.

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
