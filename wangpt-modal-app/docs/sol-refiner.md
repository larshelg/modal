# Sol-only Modal worker

`sol_app.py` runs NVIDIA's H3-specific SoL refiner on an existing video and prompt.
It has no WanGP imports, generation stage, model catalog, or H3 generator weights.
The deployment is named `sol-refiner`, independently of this repository's name.

## Build and prepare

Use the repository's Python environment (`uv sync` or `.venv/bin/python`). The
examples below assume its Python is active and Modal authentication is configured.

```bash
# Build the image and check upstream imports/tests on CPU; no GPU or model download.
python -m modal run sol_app.py::check

# Download the pinned complete checkpoint into the persistent Volume, once.
python -m modal run sol_app.py::prepare

# Deploy only the Sol entrypoint.
python -m modal deploy sol_app.py
```

Required Modal secrets:

- `huggingface-secret`: `HF_TOKEN`, for the one-time downloader.
- `studio-s3`: `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_ENDPOINT`,
  `S3_BUCKET`, and `S3_REGION`, for service input/output.

Weights live on `sol-refiner-models` at `/models/huggingface/hub`. Preparation
writes a revision manifest only after the complete download succeeds and commits
the Volume. Inference loads that local snapshot with offline settings. Missing
files fail startup with preparation instructions. No weights download per request.

The worker starts at zero containers and scales to at most one H100, with one
request at a time and a five-minute idle window. It reserves 128 GiB of host RAM
for loading and CPU offload. Startup and each request have a 30-minute timeout.
Repeated requests reuse one initialized pipeline. Memory and latency still need
measurement on a representative H3 clip.

## First local-file test

Use an existing short H3 MP4 with its original prompt. The local client requires
`ffmpeg` and `ffprobe`. Files transferred by this smoke command are capped at
64 MiB each; the normal service has a 1 GiB input cap.

```bash
python -m modal run sol_control.py::smoke \
  --input-file approved-h3.mp4 \
  --prompt 'A presenter explaining the weather in a television studio.' \
  --output-file sol-refined.mp4 \
  --width 1344 --height 768
```

This calls the deployed worker and saves both the MP4 and `sol-refined.mp4.json`
with timings and media metadata. It refuses to overwrite existing output files.
Run again with a different output filename to measure warm-container reuse.
Model initialization time is reported separately and is the container's original
load measurement; `first_request` identifies the first successful refinement.

## Submit an asynchronous job

Create a request JSON (see `examples/sol-refine.json`) with the real source URI,
prompt, and matching output aspect ratio:

```bash
python -m modal run sol_control.py::refine --params-file request.json
python -m modal run sol_control.py::status --job-id JOB_ID
```

Inputs can be HTTPS URLs, including presigned URLs, or `s3://BUCKET/KEY` in the
configured bucket. HTTPS redirects must remain HTTPS and resolve to public
addresses. This is an authenticated Modal client, not a public HTTP endpoint.

The client returns a job ID and Modal call ID. Status reports queued/running/
succeeded/failed and the current phase. Container startup failures and timeouts
are reconciled from the Modal call when status is queried. Calls and job state
use separate keys to avoid overwriting results when a worker finishes quickly.

Successful results contain a verified S3 artifact under
`runninghub/sol/<job-id>/refined.mp4`. Source objects are never overwritten. Local
scratch files are removed on success and on ordinary request failures. Results
include media metadata, revision pins, seeds, GPU memory peaks, and timings.
`worker_total_ms` includes transfer and processing after model initialization;
it does not include queue time or container/model startup.

Optionally generate a fresh download URL:

```bash
python -m modal run sol_control.py::output_url --job-id JOB_ID --expires 3600
```

This optional command requires the same S3 environment variables **locally**.
Submit/status/smoke only need local Modal authentication; the service uses Modal
secrets. Expiring output URLs are not persisted in job records.

## Media contract

- A non-empty prompt and even output dimensions are required. Seeds are optional
  nonnegative integers. Defaults are zero for service requests; smoke uses fixed
  reference seeds. Unknown fields are rejected.
- Clips must have 9–241 frames, with `(frames - 1) % 8 == 0`; 121 frames works.
  By default a 120-frame clip is rejected because upstream would shorten it to
  113 frames. Set `frame_policy: "pad"` explicitly to repeat the final frame to
  the next valid length, refine, then remove only the padding. Source frame count,
  duration, and audio are preserved. Results record the number of padded frames;
  inspect the tail for temporal artifacts when evaluating this mode.
- Only constant frame rates from 1–60 fps, starting at timestamp zero, are
  accepted. The original frame count/rate are checked after processing.
- The initial internal output canvas is capped at 1920×1088 pixels, including
  rounding each dimension to 64. Dimensions are at most 1920 each; portrait
  1080×1920 is supported by the contract. These are conservative service limits,
  not a claim that every permitted combination has been benchmarked.
- Source pixels are similarly capped. Output aspect ratio must match within 2%.
  Rotated metadata and multiple video/audio streams are rejected.
- Audio must start at zero and agree with the video duration. AAC is copied;
  other supported source codecs are encoded to AAC. The final audio presence and
  duration are verified. No silent trimming with `-shortest` is used.

Review faces, hands, background geometry, temporal consistency, and lip sync on
real H3 material before treating this as production-ready.

Inspect an S3/HTTPS candidate on CPU before starting the GPU:

```bash
python -m modal run sol_app.py::inspect --input-url s3://BUCKET/clip.mp4 \
  --output-dir /tmp/sol-inspection
```

This saves probe metadata (including any embedded generation prompt) and three
preview frames. Add `--save-video` to also download the inspected video, up to
64 MiB. It uses the remote S3 secret rather than requiring local S3 credentials.

## Reproducibility and validation

Code and checkpoint pins are in `sol_versions.py`. GPU dependencies are resolved
for Linux x86_64/Python 3.12 in `sol-requirements.lock`; NATTEN's specific CUDA 12.6
binary wheel is pinned in the image. Rebuild/redeploy and restart warm containers
after changing versions; prepare the model again when its revision changes.

```bash
uv pip compile sol-requirements.in --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match --output-file sol-requirements.lock

python -m pytest tests/test_sol.py -q
```

The image build runs NVIDIA's CPU tests and verifies pipeline/NATTEN imports.
Local tests cover media validation and audio muxing with real FFmpeg, storage
failure paths, job failures, and Sol/WanGP import isolation. They do not replace
a full H100 inference test with prepared weights.

Validation on 2026-10-01: 139 repository tests passed (33 Sol tests), with seven
existing skips. The [CPU-only Modal build check](https://modal.com/apps/larshelg/main/ap-ToUswAgRAg20HMoA58U3kh)
succeeded, including dependency consistency, Sol/NATTEN imports, and all five
upstream contract tests. The pinned 32-file checkpoint (70,746,026,408 bytes) was
prepared on `sol-refiner-models`, and the Sol-only service was deployed. The
selected real-video test uses `examples/sol-h3-smoke.json`, which retains the
source video's embedded generation prompt and explicitly enables frame padding.
Full H100 inference validation is in progress.

Deploying Sol does not stop existing WanGP deployments or delete `wangp-data`.
Retire old deployments separately after a successful Sol smoke test if the
account should have only Sol deployed.
