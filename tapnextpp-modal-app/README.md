# TAPNext++ tracking stage on Modal

This standalone Modal app exposes one public workflow entrypoint.
`run_tracking_stage` downloads a normalized H3 video, runs TAPNext++,
constructs and stabilizes screen homographies, creates review evidence, uploads
durable artifacts, and writes `result.json` last.

## Runtime

- App: `tapnextpp-modal-app`
- Function: `run_tracking_stage` (S3-native CPU coordinator)
- Private GPU boundary: `track_stage_points_s3`
- Default GPU: L4
- Model: official Google DeepMind TAPNext++ 512×512 checkpoint
- Source: official `google-deepmind/tapnet` repository at a pinned commit
- License: Apache-2.0 for source and checkpoint
- Secret: `studio-s3`
- Persistence: none; the immutable checkpoint is baked into the image
- Scratch storage: a unique `/tmp/tapnextpp-stage-*` directory per call
- Public HTTP endpoint: none

The app deliberately does not mount a Modal Volume. Downloaded videos, decoded
frames, raw tracks, JSON, review images, and review video are temporary. Only
verified S3 uploads are durable.

## S3 secret

The existing `studio-s3` Modal Secret must provide all five keys:

```text
S3_ACCESS_KEY_ID
S3_SECRET_ACCESS_KEY
S3_ENDPOINT
S3_BUCKET
S3_REGION
```

The request contains only bucket/key/hash metadata. It never contains
credentials or signed URLs. The configured `S3_BUCKET` must match both the
input ArtifactRef and output bucket.

The checkpoint is approximately 2.53 GB, so the first image build takes longer
than the CoTracker app. Later deploys reuse Modal image layers when unchanged.

## Local checks

```bash
cd /Users/larshelg/comfyui/modal2/tapnextpp-modal-app
uv sync --dev
uv run pytest
uv run python -m py_compile app.py
```

## Deploy

```bash
python3 -m modal deploy app.py
```

Check deployed metadata without starting a GPU:

```bash
python3 -m modal run app.py
```

Run an S3 stage request:

```bash
python3 -m modal run app.py \
  --stage-request-json /absolute/path/to/tracking-stage-request.json \
  --output-json /absolute/path/to/tracking-stage-result.json
```

## S3 stage contract

The JSON request follows
`docs/natural-studio-s3/architecture-and-contracts.md`: schema version 1,
`stage: tracking`, one S3 `normalizedVideo` ArtifactRef, a positive dynamic
frame count at exact 24-fps
media metadata, source-space frame-zero corners, analysis/surface dimensions,
an optional `evidenceMode` (`full`, the default, or `none`), and a
hash-qualified output prefix.

Before allocating the GPU worker, the coordinator validates all request relationships, downloads to
a `.part` file under `/tmp`, verifies byte size and SHA-256, atomically renames
the input, decodes it, and checks the exact media contract. It uploads
coordinates, full metrics, and per-attempt debug records. With
`evidenceMode: full` it also uploads the review video, four overview sheets,
and annotated suspects. `evidenceMode: none` skips all review rendering and
returns an empty suspects list while preserving identical tracking geometry
and metrics. Every upload is HEAD-verified against size and SHA-256 metadata.
`result.json` is the last upload and commit marker.

The compact Modal return value is the same StageResult stored in `result.json`;
video bytes, images, and raw point tracks are never returned from the S3-native
entrypoint.

## Off-screen geometry V2

Set `parameters.qaVersion` to `2` to enable partial and off-screen display
recovery. New natural-studio requests use V2; `qaVersion: 1` remains accepted
for historical replay and retains the original 32-point interpolation path.

V2 keeps the proven 32 inset plane queries for normal RANSAC estimation and
adds four exact TL/TR/BR/BL corner probes used only for recovery and
provenance. When direct plane tracking is weak, the geometry layer predicts one
coherent similarity transform from the two previous resolved quads. Two or
more trustworthy corner probes may correct that prediction as one coupled
translation/rotation/scale transform. It never overwrites corners
independently and never clamps a coordinate to the video boundary.

An invisible interval with a later direct observation is bridged to that
re-entry anchor. An invisible terminal interval is predicted for 24 frames,
then held and explicitly labeled low-confidence. Per-frame output includes
`resolutionSource`, `predictionAge`, `cornerSources`,
`offscreenCornerCount`, and `fullyOffscreen`. V2 metrics summarize partial,
predicted, bridged, held, and fully off-screen frames. Review evidence labels
the recovery mode and draws a locator inset when the quad extends beyond the
camera view.

The private GPU function independently repeats request and input verification
in its own `/tmp/tapnextpp-gpu-*` directory before model inference. It returns
raw tracks only to the coordinator inside Modal; those tracks never cross the
external stage boundary or become durable S3 artifacts.

## Private inference boundary

`track_stage_points_s3` accepts the coordinator's generated frame-zero point
queries only inside Modal. It downloads and verifies the same S3 video, runs
the official 512×512 recurrent model forward, and returns raw tracks to
`run_tracking_stage`. Raw point tracks are not a public API and never cross
the external stage boundary or become durable artifacts.
