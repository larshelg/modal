# Fizgig training and H3 RefMods on Modal

This app runs Fizgig's pinned headless LoRA training and H3 image/video RefMod pipelines on Modal. The
local `control.py` client calls the deployed Modal functions directly, using
the same local-client/deployed-worker pattern as `wangpt-modal-app`.
No REST app, endpoint URL, or HTTP proxy credentials are required.

## Client and worker

- `app.py` deploys `fizgig-modal-app`, with `run_training`, `request_pause`,
  `health`, `fetch_models`, `inspect_dataset`, and `publish_artifact` functions.
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

## Create an H3 RefMod

RefMods store reference latents in a `.safetensors` file for H3's reference path.
The worker calls the pinned upstream `minimax_refmod.py`; it does not create a
LoRA. This extension supports **images, prepared MP4 clips, or a mixture**, from
either the Volume or S3. Video-only datasets work in both plain and optimized
modes. Audio bundles, previews, and continuation from an existing RefMod are
not exposed by this client. See the upstream
[RefMod guide](https://github.com/shootthesound/Fizgig/blob/5d5ced3739f303065ca6d13b51eb95c5ec85c00d/docs/REFMOD_HOWDOI.md)
for how the files work in ComfyUI.

Deploy this revision before submitting RefMod jobs. The family stays
`minimax_h3`; the preset selects the output type:

- `h3_refmod_community`: plain encode, 0 optimization steps, up to 8 references
  at 1 MP. Captions, the text encoder and the H3 DiT are not needed; only the
  video VAE is required.
- `h3_refmod_lite`: 200 optimization steps, up to 16 references at 0.5 MP.
- `h3_refmod_quality`: 200 optimization steps, up to 16 references at 1 MP.

Optimized jobs require a non-empty UTF-8 `.txt` caption for every input image or clip,
plus the H3 text encoder and selected DiT. References have their own cache;
optimization targets are always cached at 0.25 MP. Both cache resolutions and
the dataset configuration live under the run directory, so existing LoRA
caches are not reused or changed.

The optimization base defaults to `ref2va`, matching upstream's RefMod workflow.
Use `--base-model fl2va` to make a mod for the first/last-frame model. There is
no fallback between the two models: missing selected weights fail explicitly.
Fetch the optional reference DiT through the existing CPU model downloader:

```bash
uv run modal run control.py::fetch_models --family minimax_h3 --include-optional --dry-run
uv run modal run control.py::fetch_models --family minimax_h3 --include-optional
```

The optional reference DiT is approximately 21 GB. These commands also fetch
any missing standard H3 model files. Plain encoding can use an already-present
H3 video VAE without fetching the reference DiT.

```bash
# Captionless image dataset: plain reference encoding.
uv run modal run control.py::refmod \
  --dataset linda --output-name linda_refmod_v1 \
  --preset h3_refmod_community --description "A woman with blonde hair"

# Captioned S3 image dataset: optimize against the reference base.
uv run modal run control.py::refmod \
  --dataset-s3 s3://YOUR_BUCKET/datasets/linda/ \
  --output-name linda_refmod_lite_v1 --preset h3_refmod_lite \
  --base-model ref2va

uv run modal run control.py::status --job-id JOB_ID
```

`refmod` accepts optional `--steps` (0–2000), `--max-refs` (1–64),
`--target-mp` (0.25–1.0), `--clips` (`still`, `motion`), `--token-cap`
(0–65536; 0 means no thinning), `--grid` (`full`, `8`, `16`, `32`), `--base-model`
(`ref2va`, `fl2va`), `--description` (single line, at most 1000 characters), and
`--concept-type` (`identity`, `style`, `pose_motion`, `clothing`, `background`,
`generic`). For a lightweight style mod, use the community preset with
`--grid 16 --concept-type style --target-mp 0.25`. More/larger references also
cost more tokens during generation; `max_refs` is a ceiling, not a required
media-file count. Photos are selected before clips, so a dataset with at least
`max_refs` photos can exclude all its clips; use a video-only dataset when the
reference must come from video. CLI defaults `steps=-1`, `max_refs=0`, and `target_mp=0` mean
“use the preset”; explicit JSON values must satisfy the stated ranges.

Use `submit_file --request-file request.refmod.example.json` or the existing
Python `submit_training({...})` helper for the same typed contract. JSON uses
snake_case keys such as `base_model` and `max_refs`. RefMod requests reject
`epochs`, `trigger_word`, and `resume_from`; put the identity wording in your
captions and the descriptive hint in `description`.

RefMods follow the existing job lifecycle for submission, status and
cancellation. Optimized progress reports `phase: making_refmod`, `step`, and
`steps_total`. **Pause/resume is unavailable** because this upstream pipeline
has no resumable checkpoint protocol. Both client and worker reject pause;
use `cancel` to stop a job and a new output name to start again.

Successful results contain `artifact_uri`, `outputs`, `artifact_type: refmod`,
`steps`, `clips`, `token_cap`, and `base_model`
(null for a plain encode). Files are retained at:

```text
/data/fizgig/runs/<output_name>/<output_name>.safetensors
/data/refmods/<output_name>.safetensors
```

No RefMod is placed in `/data/loras`. Existing RefMod destinations are rejected.
Use unique output names; the app defaults to one GPU worker and does not provide
a distributed reservation for simultaneous submissions using the same name.
Download a completed file into your ComfyUI `models/refmods` folder, for example:

```bash
uv run modal volume get wangp-data /refmods/linda_refmod_v1.safetensors ./linda_refmod_v1.safetensors
```


### Video references

`--clips still` (the default) contributes each clip's sharpest face frame.
`--clips motion` contributes its latent frame sequence. Every RefMod latent
cache pass also saves the selected still, so optimized jobs can use those
stills as loss targets even when the saved reference uses motion. Optimization
runs on photos and extracted stills, not the full video sequence.

`--token-cap 5120` asks upstream to thin motion frames to fit the token budget:
it removes near-duplicates first, then resamples the remaining frames. It does
not trim input files, cap photo references, or guarantee a hard total when
photos and the minimum one-frame-per-clip already exceed the budget. Leave it
at `0` to retain every latent motion frame.

Use a folder containing prepared `.mp4` files and optional same-stem `.txt`
sidecars. On the Volume the folder remains
`/data/fizgig/datasets/<dataset>/images/` for compatibility. In S3, nested clips
are flattened to unique names with matching captions; the `_mute` suffix is
preserved. Video is enabled automatically for RefMod submissions. Existing
S3 LoRA imports continue to select images only.

```bash
# Count both images and videos without starting a GPU.
uv run modal run control.py::dataset_info \
  --dataset-s3 s3://YOUR_BUCKET/datasets/linda-clips/ --include-video

# Make a motion reference from a video-only S3 folder, without captions.
uv run modal run control.py::refmod \
  --dataset-s3 s3://YOUR_BUCKET/datasets/linda-clips/ \
  --output-name linda_motion_v1 --preset h3_refmod_community \
  --clips motion --token-cap 5120

# Optimize a captioned Volume dataset; use each clip's sharpest still.
uv run modal run control.py::refmod \
  --dataset linda-clips --output-name linda_clip_stills_v1 \
  --preset h3_refmod_lite --clips still
```

The equivalent motion request is `request.refmod.video.example.json`. JSON
uses `clips` and `token_cap`. `dataset_info --include-video` reports separate
`image_count` and `video_count`, combined `media_count`/`caption_count`, and
missing-caption counts per media type. Without `--include-video`, inspection
keeps the existing image-only behavior. `--verify-download` verifies transfers
and checksums; it does not decode clips or validate their encoding.

Prepare clips with upstream Gizmo or an equivalent process before submission.
The pinned upstream cache validates these requirements when decoding:

- MP4 at 24 fps; width and height must be multiples of 32.
- Frame counts: 5, 22, 39, 56, 73, 90, 107, or 124.
- No soundtrack, or a 32 kHz stereo soundtrack. Sound is not included in this
  visual RefMod workflow and no audio VAE is required.

The worker does not automatically transcode, trim, or resample off-spec input.
Both still and motion modes use the upstream clip decoder and these checks.
S3 video imports allow up to 256 clips, 512 MiB per clip, within the existing
10 GiB total and 20,000 listed-object limits. Image limits remain 5,000 files
and 128 MiB per image. These are S3 transfer limits, not GPU-memory guarantees.
Unsupported video containers such as MOV are not imported; convert them to
prepared MP4 clips first. Video-enabled S3 manifests use schema 2 and retain
file hashes; existing schema-1 image snapshots remain readable.


## S3 artifact results

Completed LoRAs and RefMods are uploaded to the same `studio-s3` bucket used by
WanGP, under a separate prefix:

```text
s3://<S3_BUCKET>/runninghub/fizgig/<job_id>/000-<output_name>.safetensors
```

The worker reports `progress.phase: uploading_artifact` while publishing. It
marks the job successful only after S3's object size and SHA-256 metadata match
the local artifact. The result includes:

- `artifact_uri`: the persistent `s3://` link to the finished file.
- `outputs`: WanGP-style metadata containing `storage`, `bucket`, `key`, `uri`,
  `filename`, `media_type`, `size_bytes`, `sha256`, and `artifact_type`.
- `artifact_path` and `run_path`: retained Volume locations for local use and
  recovery. Source datasets, caches, checkpoints, and resumable state are not
  uploaded or deleted by artifact publishing.

The URI uses authenticated S3 access; it is not a public or expiring HTTPS URL.
The final artifact is uploaded from its run directory. A paused LoRA job has no
final artifact to publish and continues to return its saved resume information.

If publishing fails, the job is `failed` with phase `uploading_artifact` and its
local result remains recorded. Retry just the upload on CPU:

```bash
uv run modal run control.py::publish --job-id JOB_ID
```

This command also publishes artifacts from jobs completed before S3 output
support was deployed. It updates the existing job result, so subsequent
`status` calls return the S3 link. Repeated publishing accepts an existing
object with matching size/hash without uploading again; an existing different
object is rejected. Active jobs, cancelled jobs, failed training stages, and
paused jobs cannot be published with this command.


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
The same `studio-s3` secret used by WanGP supplies `S3_ACCESS_KEY_ID`,
`S3_SECRET_ACCESS_KEY`, `S3_ENDPOINT`, `S3_BUCKET`, and `S3_REGION` for S3 inputs.
The local client needs only Modal authentication; it never needs S3 credentials.
Optional deployment settings are `FIZGIG_GPU` and `FIZGIG_MAX_CONTAINERS`.
Do not deploy `control.py`; it runs locally and constructs no worker image.
Health calls the small deployed CPU function, without starting a GPU.

Deploy this revision before using S3 datasets. The older Volume-dataset client
continues to work with the existing training functions.
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

## Use an S3 dataset folder

Pass `--dataset-s3` instead of `--dataset`; no prior Modal Volume upload is needed:

```bash
uv run modal run control.py::dataset_info \
  --dataset-s3 s3://YOUR_BUCKET/datasets/linda/

uv run modal run control.py::submit \
  --family krea2 \
  --dataset-s3 s3://YOUR_BUCKET/datasets/linda/ \
  --output-name linda_s3_v1 \
  --preset krea2_defaults \
  --trigger-word linda
```

The CPU-only `dataset_info` command reports image count, matching caption count,
missing sidecars, and total download size. Add `--verify-download` to download
and check the whole dataset in temporary CPU-container storage, then discard it.
This does not start training or create a run. It verifies transfers; it does not
decode images or test model training.

Fizgig needs writable local files for captions and caches. The training worker
lists the S3 prefix, downloads images and matching `.txt` sidecars, verifies
sizes and available SHA-256 metadata, and publishes a complete snapshot at:

```text
/data/fizgig/runs/<output_name>/dataset/images/
/data/fizgig/runs/<output_name>/dataset/manifest.json
/data/fizgig/runs/<output_name>/dataset/cache/
/data/fizgig/runs/<output_name>/dataset/dataset.toml
```

Downloads use a temporary staging directory on the same Volume; failed transfers
are cleaned up and cannot become a training dataset. The completed snapshot,
captions and caches are retained with the run so pause/resume survives container
restarts. Resume checks image hashes and reuses this snapshot without contacting
S3 or overwriting generated captions. To use an updated S3 folder, start a new
run with a new output name. Nothing is written back to S3.

S3 folders are prefixes and include subfolders. The image importer accepts `.jpg`,
`.jpeg`, `.png`, `.webp`, and `.bmp`, pairing each image with an adjacent `.txt`
file of the same stem. It flattens paths to deterministic hash filenames to avoid
collisions such as `front/photo.jpg` and `side/photo.jpg`; the manifest records
the original object keys. Other file types and unpaired captions are not imported.
Krea2 creates missing/empty captions; H3 requires non-empty captions for every image.

The bucket must match `S3_BUCKET` in `studio-s3`. Supply an `s3://bucket/prefix/`
URI, not an HTTPS or presigned URL. The importer uses the configured S3-compatible
endpoint and path-style addressing, as WanGP does. An ETag precondition rejects
objects changed between listing and download.

Limits per import: 5,000 images, 20,000 listed objects, 128 MiB per image,
1 MiB per caption, and 10 GiB total staged data. Progress appears as
`downloading_dataset`, including file and byte counts. Job status includes a
`dataset` summary and snapshot path. Model weights, checkpoints and the final
LoRA continue to use their existing Modal Volume locations.

The equivalent JSON request is in `request.s3.example.json`:

```bash
uv run modal run control.py::submit_file --request-file request.s3.example.json
```

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

Requests require `family`, `output_name`, `preset`, and exactly one of `dataset`
(a prepared Volume dataset name) or `dataset_s3` (an S3 folder URI).
`trigger_word` and `epochs` are optional for LoRA requests; RefMod requests use
the fields described above. Resume is controlled by the resume
command. Raw CLI arguments and arbitrary model/filesystem paths are rejected.

For Krea2, the worker captions images, caches latents and text, then trains
with per-image loss/LR tracking and recaptioning of stuck images. The LoRA dataset
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

The tests exercise validation, LoRA and RefMod pipeline commands, model requirements,
RefMod artifact routing, caption checks, and mocked Modal
job lifecycle operations without downloading models or starting GPUs.
