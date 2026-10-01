# SoL-only Modal service in this repository

## Scope

Reuse `wangpt-modal-app` as the repository and reuse its Modal/S3 service patterns.
The new deployment runs **only SoL-Refiner**. It accepts an already-generated
video; it does not generate an H3 draft or load WanGP, Wan2AI, or H3 generator
weights. The H3-specific SoL checkpoint is the refiner, not the H3 generator.

```text
Existing video + prompt -> SoL worker on Modal -> refined MP4
```

Recommended deployment name: `sol-refiner`, with `sol_app.py` as its entrypoint.
The repository name does not determine which runtime Modal deploys. Keep existing
WanGP files during development, but the Sol entrypoint must not import `app.py`,
`control.py`, or `wangpt_common.py`: those modules construct WanGP resources or
pull in H3-specific code.

Deploying a new entrypoint does not stop an already-deployed WanGP app. If the
account must have only Sol deployed, retiring the old deployments is a separate
cutover action after the Sol smoke test succeeds. This document does not perform
that action.

## Verified upstream details

Reviewed 2026-10-01. These links track a moving branch; pin the Sana commit and
the model revision before building the image.

- [H3 refiner installation and inference](https://github.com/NVlabs/Sana/blob/sol-engine/models/sol-refiner/MiniMax-H3/README.md):
  upstream reports testing on H100 80GB with Python 3.12, PyTorch 2.9.1+cu126,
  and NATTEN 0.21.5 using its torch290cu126 wheel. H100 is our initial target.
- [Dependencies](https://github.com/NVlabs/Sana/blob/sol-engine/models/sol-refiner/MiniMax-H3/requirements.txt):
  Diffusers is pinned to `e0abab83b5df05de9e7abd788643c1a7c1e42e28`.
  Other requirements include Transformers, Accelerate, Hugging Face Hub,
  safetensors, PyAV, and imageio-ffmpeg. Resolve and pin those packages for our
  image; the upstream requirements are not a complete lockfile.
- [Official inference entrypoint](https://github.com/NVlabs/Sana/blob/sol-engine/models/sol-refiner/MiniMax-H3/infer.py):
  uses `SoLRefinerH3Pipeline`, BF16 weights, `LocalNattenProcessor`, decoder
  tiling, and model CPU offload. It requires a prompt and has independent
  refiner and decoder seeds. Reproduce these settings before optimizing.
- [Pipeline implementation](https://github.com/NVlabs/Sana/blob/sol-engine/models/sol-refiner/MiniMax-H3/sol_refiner_h3/pipeline.py):
  performs one refiner transformer evaluation and Euler update, then a diffusion
  decoder. “One step” does not mean the entire request is one GPU operation.
  It rounds the internal spatial canvas up to multiples of 64, crops to the
  requested output size, and truncates temporal length to `8k + 1` frames.
- [Official checkpoint](https://huggingface.co/Efficient-Large-Model/SoL-Refiner-LTX-2.5-for-MiniMax-H3):
  use the H3-specific LTX-2.5 refiner for the H3 videos described in this project.
  Validate other source generators separately before offering general support.

The [project page](https://nvlabs.github.io/Sana/Sol-Refiner/) reports the headline
H3 deployment timing on GB200 hardware. It is not a latency promise for this
H100 service or for the released Diffusers implementation. Measure our own
cold start, warm inference, memory usage, and end-to-end request time.

## What fits from the current app

Reuse the existing patterns, with Sol-specific names and request validation:

- `control.py`: local client, detached Modal method calls, call IDs, status
  polling, and dispatch failure reporting. Sol needs a direct `refine` route;
  it does not need catalog discovery or image/video/audio model routing.
- `app.py`: S3 client configuration, bucket-scoped input validation, download
  verification, SHA-256 metadata, and verified output uploads. Move reusable
  helpers to a module with no Modal app or WanGP imports when adding the service.
  Give video downloads their own measured size limit rather than inheriting the
  current image-oriented 128 MiB default.
- `h3_mux.py`: use its ffprobe approach as a reference. Implement Sol's audio
  handling independently of H3 continuation and latent-output hooks.
- Existing `studio-s3` and `huggingface-secret` secrets can supply the same
  credentials, subject to the checkpoint's access requirements.

Build a separate Sol image. The current image installs CUDA 13.0, PyTorch 2.10,
WanGP, Wan2AI, and H3 plugins; it is not the upstream-tested Sol environment.
Do not inherit it just to reuse S3 functions.

Implementation files (usage: [Sol worker guide](docs/sol-refiner.md)):

```text
sol_app.py           Modal image, SolRefiner class, one-time model preparation
sol_runtime.py       Upstream pipeline adapter, probing, refinement, encoding
sol_control.py       Local refine/status client without WanGP imports
sol_common.py        Sol names, request validation, result metadata
sol_storage.py       Independent S3/HTTPS transfers and verified uploads
sol_jobs.py          Job lifecycle and scratch cleanup
sol_versions.py      Immutable code and model revisions
tests/test_sol.py    Request, frame policy, job lifecycle, and output checks
```

Use a separate `sol-refiner-jobs` Modal Dict so Sol results cannot be mistaken
for jobs routed through the WanGP catalog.

## Persistent model storage

Use a dedicated Modal Volume named `sol-refiner-models`, mounted at `/models`.
Keep the existing `wangp-data` Volume intact; Sol does not need the WanGP
checkpoints or LoRAs stored there.

Set both the downloader and worker to the same cache layout:

```text
HF_HOME=/models/huggingface
HF_HUB_CACHE=/models/huggingface/hub
```

Run `snapshot_download()` once with a pinned model revision and
`cache_dir="/models/huggingface/hub"`. Download the complete pipeline, including
its conditioning, VAE, upsampler, and decoder components. Verify the snapshot
can load without network access, then commit the Volume.

Pass the resolved local snapshot path into `from_pretrained()` with
`local_files_only=True`. Use offline Hugging Face settings in inference
containers. Missing assets should cause an actionable startup error, not a
large download during a request. Restart warm workers after changing the model
revision so they load the new snapshot.

Generated media belongs in per-job scratch directories and, in the service
phase, S3. It does not belong in the model cache. Retain the downloaded smoke
test output locally for inspection.

## Worker lifecycle

Start with one H100 container at a time, one request per container, zero minimum
containers, and a 300-second scaledown window. Set startup and request timeouts
independently and adjust them from measurements.

Load the pipeline once in `@modal.enter()`. Keep the initialized pipeline across
requests, including upstream CPU offload and decoder tiling settings. This does
not mean keeping every component resident in GPU memory. Provision enough host
RAM for loading and offload, then measure the actual peak.

Call the pipeline in process from the worker method. Spawning `infer.py` for
every request would reload the weights and lose warm-container reuse. The
upstream CLI is useful for the first correctness smoke test.

## Input contract and media behavior

The first test takes a local MP4 and its prompt. The service phase accepts an
HTTPS URL or bucket-scoped S3 URI and returns a new output artifact.

Required request fields:

- `input_url`
- `prompt` — use the prompt describing the existing clip, ideally its original
  generation prompt.
- `output_width` and `output_height` — positive even numbers, within limits
  established by the smoke tests.

Optional fields: `seed`, `decoder_seed`, `generation_id`, and `asset_id`.
Keep sampler internals fixed to the upstream baseline.

Probe input resolution, frame count, frame rate, duration, and audio streams.
Initially accept constant-frame-rate clips and reject variable-frame-rate input
until a deliberate timestamp-preserving conversion is implemented.

The upstream frame rule is:

```text
output_frames = 1 + 8 * floor((input_frames - 1) / 8)
120 input frames -> 113 output frames
121 input frames -> 121 output frames
124 input frames -> 121 output frames
```

The default service contract rejects incompatible counts with a clear
explanation. Explicit `frame_policy: "pad"` repeats the final frame to the next
compatible length, refines, and removes only the added frames. It preserves the
source duration and audio; inspect the tail for temporal artifacts. This mode
supports the selected 124-frame H3 smoke clip without dropping source frames.

The upstream export contains no audio. Add the original soundtrack after
refinement, preserving timestamps and copying the audio stream when the output
container supports it; otherwise encode explicitly. Verify final video duration,
audio duration, frame rate, and frame count. `ffmpeg -shortest` alone is not a
synchronization check and can hide accidental truncation.

Write a new S3 key under `runninghub/sol/<job-id>/`; never overwrite the source.
Return the verified S3 URI and a presigned download URL if needed, following the
current storage conventions. Include actual dimensions, input/output frame
counts, frame rate, duration, audio handling, model/code revisions, seeds,
timings, peak GPU memory, and output bytes in the result metadata.

## Implementation sequence

1. **Pin the baseline.** Resolve Sana and checkpoint revisions, build the tested
   Python/PyTorch/CUDA/NATTEN combination, lock dependencies, and download the
   complete model snapshot to the Volume.
2. **Run one local-file smoke test on Modal.** Use an existing H3 clip with a
   compatible frame count and its prompt. Start at 1344×768, preserving aspect
   ratio; test 1920×1080 separately. Download and inspect the resulting MP4.
   No H3 generation, S3 service, or studio workflow is needed for this milestone.
3. **Add the warm worker.** Move the verified initialization into the container
   lifecycle and run the same clip twice. Confirm repeated invocation works and
   record model load time separately from refinement time.
4. **Add the service.** Implement URL/S3 input, detached jobs, status reporting,
   audio restoration, verified uploads, and cleanup. Check invalid requests and
   failures as well as a successful clip.
5. **Cut over if replacing the old deployment.** Switch clients to Sol, verify a
   full request, then retire old worker deployments as a separate explicit
   operation. Existing model volumes remain reusable.

Success means an existing clip becomes a valid refined MP4 through a deployment
that loads only Sol. Review faces, hands, straight lines, temporal consistency,
and audio synchronization before adopting the output. Benchmark the same source
clip against SeedVR2 if useful; source generation is outside this service.

Snapshots, quantization, alternate GPUs, long-clip chunking, Remotion, and studio
integration follow after the baseline is measured. The worker, preparation,
client, storage, media handling, and CPU tests are implemented. GPU inference
and visual quality still require a representative H3 clip and prepared weights.
