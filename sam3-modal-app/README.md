# SAM 3.1 presenter-mask and display-detection stages on Modal

This independent Modal app segments the presenter in an H3 studio video. Its
S3-native `run_sam_stage` entrypoint validates and downloads the normalized
source, dispatches strict SAM inference, creates mask QA/review evidence, and
commits verified artifacts to S3. It does **not** intersect the presenter mask
with the tracked TV and does **not** render the final composite.

The independent S3-native `detect_display` function takes one still image,
locates a TV, monitor, or other planar display from text-concept masks, derives
four source-pixel corners, uploads an annotated PNG overlay, and returns both
the corners and overlay ArtifactRef. It does not track video; pass its
`frameZeroCorners` result to TAPNext++ `run_tracking_stage`.

It deliberately does not modify or share a container with
`tapnextpp-modal-app`. SAM 3.1 requires Python 3.12+, while the tested TAPNext++
worker remains on Python 3.11.

## Requirements

- Modal CLI authenticated for this workspace.
- A Modal secret named `huggingface-secret` with an `HF_TOKEN` key.
- A Modal secret named `studio-s3` with `S3_ACCESS_KEY_ID`,
  `S3_SECRET_ACCESS_KEY`, `S3_ENDPOINT`, `S3_BUCKET`, and `S3_REGION`.
- That Hugging Face account must have accepted access to the gated
  `facebook/sam3.1` repository.

The app pins both the official SAM source commit and the Hugging Face model
revision. The checkpoint is downloaded once into the independent
`sam3-model-cache` Modal Volume; it is never baked into the image or written to
this repository.

Only the model checkpoint uses a Volume. Source videos, decoded frames, masks,
review media, and upload staging use unique `/tmp/sam3-*` directories and are
ephemeral. S3 is the durable experiment-artifact store.

## Download, test, and deploy

```bash
cd /Users/larshelg/comfyui/modal2/sam3-modal-app
uv sync --group dev
uv run pytest
uv run modal run app.py --download-only
uv run modal deploy app.py
```

Running without a video prints health and checkpoint-cache state:

```bash
uv run modal run app.py
```

Run an S3-native stage request:

```bash
uv run modal run app.py \
  --stage-request-json /absolute/path/to/sam-stage-request.json \
  --output-json /absolute/path/to/sam-stage-result.json
```

Run an S3-native frame-zero display request:

```bash
uv run modal run app.py \
  --display-request-json /absolute/path/to/display-request.json \
  --output-json /absolute/path/to/display-result.json
```

The display request contract is:

```json
{
  "schemaVersion": 1,
  "runId": "display-test-20260825",
  "stage": "display-detection",
  "inputHash": "<lowercase sha256 of the canonical request inputs>",
  "inputs": {
    "image": {
      "storage": "s3",
      "bucket": "<configured S3_BUCKET>",
      "key": "studio-experiments/display-test-20260825/source/hash/review/keyframes/frame-000.png",
      "sha256": "<image sha256>",
      "sizeBytes": 123456,
      "contentType": "image/png"
    }
  },
  "parameters": {
    "textPrompts": [
      "display screen",
      "television screen",
      "monitor screen",
      "screen"
    ],
    "scoreThreshold": 0.35,
    "minAreaRatio": 0.01,
    "maxAreaRatio": 0.95
  },
  "output": {
    "bucket": "<configured S3_BUCKET>",
    "prefix": "studio-experiments/display-test-20260825/display-detection/<inputHash>/"
  }
}
```

`parameters` may be empty to use those defaults. SAM 3.1 evaluates each
short singular prompt, and the app ranks surviving masks by model confidence
and geometric rectangularity. It fits the winning mask to a perspective
quadrilateral, orders it TL/TR/BR/BL, and writes `overlay.png` and
`result.json` below the requested prefix. Repeating the same committed
request reuses `result.json` without allocating a GPU.

The compact result includes:

```json
{
  "success": true,
  "frameZeroCorners": [[240, 447], [691, 359], [691, 782], [240, 758]],
  "cornerOrder": "top-left, top-right, bottom-right, bottom-left",
  "artifacts": {
    "overlay": {
      "storage": "s3",
      "bucket": "<configured S3_BUCKET>",
      "key": "<output prefix>/overlay.png"
    }
  },
  "overlayS3Uri": "s3://<bucket>/<output prefix>/overlay.png"
}
```

The overlay tints the selected SAM mask and labels the derived corners. Treat
it as required review evidence before sending the coordinates to TAPNext++.

`run_sam_stage` is a CPU preflight coordinator. It validates the request,
downloads and SHA-256 verifies the source in `/tmp`, and checks the requested
positive frame count at exact 24 fps before allocating a GPU. The private
`run_sam_stage_gpu` function independently downloads and verifies the source,
runs the pinned model on L40S, and retries on A100-80GB only for recognized CUDA
OOM failures.

The S3 StageRequest accepts optional `parameters.evidenceMode`: `full`
(default) or `none`. The latter is the fast-orchestrator contract: inference,
the lossless mask, strict validation, and metrics are unchanged, while review
overlay encoding, sheets, suspect images, and their uploads are skipped.

## Reproduce the first presenter-mask test

The checked-in request uses SAM 3.1's text-concept prompt `person`. On frame
90, the normalized `[x, y, width, height]` box
`[0.59, 0.18, 0.39, 0.80]` selects the intended person from the detections and
avoids the presenter reflection in the TV. The box is not sent as a visual
concept prompt: that mode matched this presenter in only a few nearby frames.
Propagation is bidirectional from the selected anchor mask. Because multiplex
object IDs can change, each next mask is selected by adjacent-frame overlap;
the job fails if fewer than 90% of source frames have a non-empty mask.

```bash
cd /Users/larshelg/comfyui/modal2/sam3-modal-app

uv run modal run app.py \
  --video-path /path/to/ai-platform/h3-ref-to-video-2090809738306854913.mp4 \
  --request-json request-presenter-2090809738306854913.json \
  --mask-output /path/to/ai-platform/apps/weather/out/no-green-tapnextpp-v1/sam3-presenter-mask.mkv \
  --preview-output /path/to/ai-platform/apps/weather/out/no-green-tapnextpp-v1/sam3-presenter-mask-preview.mp4 \
  --metadata-output /path/to/ai-platform/apps/weather/out/no-green-tapnextpp-v1/sam3-presenter-mask-metadata.json
```

The dispatcher uses an L40S first. It retries once on an A100-80GB only when
the returned failure is a recognized CUDA out-of-memory error.

The pinned upstream wrapper passes an unsupported false
`offload_state_to_cpu` argument into the current multiplex model. The app's
compatibility adapter discards only that false value and records its use in
the metadata; requesting actual state offload remains an error.

## Output contract

The S3-native entrypoint returns only a compact StageResult. It always uploads
and HEAD-verifies `mask.mkv`, full `metrics.json`, and per-attempt debug records.
In full evidence mode it also uploads `review/sam-review.mp4`, four 30-frame
overview sheets, and deterministic suspect images. No-evidence mode omits those
artifacts and returns an empty suspects list. `result.json` is uploaded last as
the commit marker. A repeated request with the same input hash reuses the
committed result without GPU work.

The strict S3 stage requires every requested mask to be non-empty. The older
`segment_presenter` byte-returning entrypoint remains available for filesystem
compatibility and retains its original behavior.

- `sam3-presenter-mask.mkv`: canonical lossless FFV1 video, 8-bit `gray`, with
  exactly the input dimensions, frame rate, and frame count. The official SAM
  3.1 public predictor exposes binary masks, so background is `0` and
  foreground is `255`. No feathering or dilation is applied.
- `sam3-presenter-mask-preview.mp4`: H.264 diagnostic preview with a magenta
  presenter overlay.
- `sam3-presenter-mask-metadata.json`: source geometry, pinned model identity,
  prompt, GPU runtime, missing frames, and per-frame mask bounds and areas.
- A per-frame PNG ZIP is optional. Set `include_debug_png_zip` in the request
  and pass `--debug-png-zip-output`; it is not the canonical artifact.

The next pipeline phase will calculate `presenter_mask AND tv_mask`, feather
only that small overlap, and restore original H3 presenter pixels over the
warped weather. None of that compositing is part of this app.
