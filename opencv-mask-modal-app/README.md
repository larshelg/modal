# TAPNext-guided OpenCV mask stage on Modal

This standalone, CPU-only Modal app implements the guided chroma-occlusion idea:
TAPNext supplies the tracked display polygon, and fixed HSV green detection
inside that polygon supplies the replacement mask. Hard binary masking remains
the default; an additive boundary-soft mode can emit continuous replacement
values near hard foreground boundaries.

The app does one job. It does not warp or composite the weather map, alter the
TAPNext homographies, run SAM, smooth masks over time, apply morphology, feather
edges, or despill green.

## Contract

Deploy function: `create_occlusion_mask`

Required request fields:

```json
{
  "video_url": "s3://studio-bucket/path/to/input.mp4",
  "tracking_url": "s3://studio-bucket/path/to/coordinates.json",
  "output_prefix": "s3://studio-bucket/path/to/job-123/"
}
```

Optional fields and defaults:

```json
{
  "green_hsv": {
    "lower": [35, 50, 40],
    "upper": [95, 255, 255]
  },
  "soft_chroma": {
    "mode": "disabled",
    "radius_px": 3,
    "hsv_ramp": [8, 40, 40]
  },
  "debug_frames": []
}
```

Set `soft_chroma.mode` to `boundary` to retain the hard mask away from
foreground edges and calculate continuous green membership only inside the
configured boundary radius. This conservative mode does not recover a
translucent object that never creates a hard foreground seed, and it does not
perform RGB despill.

All URLs must use the bucket named by the existing `studio-s3` Modal Secret.
The secret provides `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_ENDPOINT`,
`S3_BUCKET`, and `S3_REGION`. Credentials never enter the request or result.

The tracking object must have one zero-based, ordered quadrilateral per video
frame. The app accepts the complete `coordinates.json` emitted by
`tapnextpp-modal-app`: its frame corners are declared on the `analysis` canvas
and are scaled back to the declared `source` video dimensions before masking.
Unrelated frame fields such as `homography` are ignored. A minimal document
without `analysis` metadata is interpreted as source-video coordinates.

```json
{
  "frames": [
    {
      "frame": 0,
      "corners": [
        [210.4, 120.2],
        [590.7, 131.8],
        [575.2, 402.1],
        [224.1, 395.6]
      ]
    }
  ]
}
```

Coordinates use source-video pixels and consistent top-left, top-right,
bottom-right, bottom-left order. TAPNext++ V2 corners may be negative or extend
beyond the right or bottom source edge. The service geometrically clips the
quad to the visible frame without clamping individual corners. A partially
visible display is masked only where its polygon intersects the source frame;
a fully invisible display produces an all-black replacement frame. Four
off-screen corners do not by themselves mean the display is invisible because
the quad may still enclose or cross the frame.

The service still rejects gaps, extra or missing tracking frames, non-convex
quads, non-finite coordinates, and unreasonably large coordinate magnitudes.

## Outputs

The app uploads these objects below `output_prefix`:

```text
occlusion-mask.mkv
result.json
debug/
  frame_000042_original.jpg
  frame_000042_polygon.jpg
  frame_000042_green-mask.png
  frame_000042_hard-replace-mask.png  # soft mode only
  frame_000042_replace-mask.png
  frame_000042_overlay.jpg
```

`occlusion-mask.mkv` is an 8-bit grayscale FFV1 stream with the same width,
height, frame count, and frame rate as the source. White (`255`) means replace
with the weather map; black (`0`) means preserve the H3 frame; intermediate
values are continuous replacement alpha in boundary-soft mode. MKV is used
instead of MOV because it is the well-supported FFV1 container. Encoding is
verified with `ffprobe` before upload.

Artifacts are uploaded and HEAD-verified. `result.json` is uploaded last as
the commit marker:

```json
{
  "status": "completed",
  "frames": 120,
  "width": 864,
  "height": 480,
  "fps": 24.0,
  "mask_url": "s3://studio-bucket/path/to/job-123/occlusion-mask.mkv",
  "debug_prefix": "s3://studio-bucket/path/to/job-123/debug/",
  "mask_mode": "soft-boundary",
  "partial_replacement_pixels": 1234,
  "green_coverage_mean": 0.82,
  "green_coverage_min": 0.61,
  "partial_offscreen_frame_count": 12,
  "partial_offscreen_frames": [48, 49],
  "fully_offscreen_frame_count": 2,
  "fully_offscreen_frames": [62, 63],
  "maximum_offscreen_corner_count": 4,
  "visible_polygon_pixels_min": 0
}
```

Coverage is the fraction of visible pixels inside the clipped tracked polygon
that pass the HSV threshold. Fully invisible frames are excluded from coverage
aggregates; if every frame is fully invisible, the coverage values are
`null`. Coverage remains diagnostic only and does not reject low-coverage
frames. Off-screen debug frames include a locator inset showing the source
frame and the full tracked quad.

## Test and deploy

```bash
cd /Users/larshelg/comfyui/modal2/opencv-mask-modal-app
uv sync --group dev
uv run pytest
uv run python -m py_compile app.py mask_stage.py
uv run modal deploy app.py
```

Check the deployed image without reading S3:

```bash
uv run modal run app.py
```

Run a request:

```bash
uv run modal run app.py \
  --request-json /absolute/path/to/request.json \
  --output-json /absolute/path/to/result.json
```
