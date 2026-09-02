# H3 Two-Phase Latent Upscaling

WanGP v12.643 adds two-phase latent upscaling for MiniMax H3. The REST app
already exposes the native WanGP settings, so no application code changes are
required.

## Minimal request change

Set the desired final resolution and enable two phases:

```diff
- "resolution": "832x480"
+ "resolution": "1664x960"
+ "guidance_phases": 2
```

For this example, WanGP generates phase one internally at `832x480`, upscales
the latent to `1664x960`, and performs a fixed three-step refinement at the
target resolution. WanGP automatically manages the required LightX2V Turbo
LoRA.

## Complete five-second request

```bash
curl --fail-with-body -X POST \
  -H "Content-Type: application/json" \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  "https://larshelg--wangp-rest-api.modal.run/jobs" \
  --data-binary @- <<'JSON'
{
  "kind": "video",
  "model": "minimax_h3_fl2va_pruned",
  "params": {
    "prompt": "integrated_multimodal_description: [Shot 1] A cinematic continuous shot of a brass automaton tending a glowing rooftop garden at blue hour. The camera slowly circles as it waters crystalline flowers and warm city lights appear below.\noverall_soundscape: Soft mechanical movements, water droplets, evening wind, and distant city ambience.\nnon_diegetic_music: A restrained, dreamlike string motif.",
    "resolution": "1664x960",
    "video_length": 124,
    "num_inference_steps": 20,
    "guidance_phases": 2,
    "seed": -1
  }
}
JSON
```

At H3's native 24 FPS, `124` frames produces approximately `5.17` seconds.

## Two-phase 8-step generation with stacked LoRAs

The following parameters combine WanGP's predefined LightX2V FL2VA 8-step
Turbo accelerator with `HMNSFW_AIO_V2.safetensors`, while also enabling H3's
two-phase latent upscaling:

```json
{
  "resolution": "1664x960",
  "activated_loras": [
    "https://huggingface.co/DeepBeepMeep/MiniMax-H3/resolve/main/loras/minimax_h3_lightx2v_fl2v_turbo_8step_alpha8_v1.0_bf16.safetensors",
    "HMNSFW_AIO_V2.safetensors"
  ],
  "loras_multipliers": "0.5;0 0.5;0.5",
  "num_inference_steps": 8,
  "guidance_scale": 1,
  "flow_shift": 12,
  "guidance_phases": 2,
  "switch_threshold": 0.9035
}
```

The semicolon separates the multiplier used in phase one from the multiplier
used in phase two:

- `0.5;0` applies the FL2VA 8-step Turbo LoRA at `0.5` during phase one and
  disables it during phase two. WanGP disables other selected Turbo LoRAs
  during phase two and automatically supplies the separate Turbo LoRA required
  by its refinement pipeline.
- `0.5;0.5` applies `HMNSFW_AIO_V2.safetensors` at `0.5` during both phases.

No LoRA trigger word is required by this example. The `resolution` value is the
final canvas: WanGP generates phase one internally at `832x480`, enlarges its
latent by 2x, and performs the fixed phase-two refinement at `1664x960`.

### `spatial_upsampling` is a separate operation

`guidance_phases` and `spatial_upsampling` control different pipelines:

- `"guidance_phases": 1` generates directly at the requested resolution and
  performs no H3 latent upscale.
- `"guidance_phases": 2` enables H3's integrated two-phase latent upscale and
  refinement. This is the setting used by the example above.
- `spatial_upsampling` selects a decoded-video pixel or reconstruction
  postprocessor such as FlashVSR. It does not enable or disable H3's latent
  phase-two process.

For ordinary H3 generation on this deployed WanGP revision, omit
`spatial_upsampling` when no decoded-video upscaler is wanted. Passing
`"spatial_upsampling": "off"` was rejected by the installed H3 generation path
as an unsupported method; omission leaves postprocessing disabled through the
model defaults. To use FlashVSR, first finish the H3 generation and then submit
a separate `mode: "edit_postprocessing"` task with a supported value such as
`"spatial_upsampling": "flashvsr2.5"`, as documented later in this file.

## Phase-two noise level

The optional `switch_threshold` setting controls how much noise begins the
target-resolution refinement:

```json
"switch_threshold": 0.9035
```

`0.9035` is the current H3 default, so it can normally be omitted. The accepted
range is `0.7` through `1.0`:

- Lower values preserve more of phase one and favor smoother blending.
- Higher values allow stronger new detail, with more risk of visible changes or
  seams.

## Lower-VRAM tiled refinement

To process phase two as four overlapping tiles, add the `~` WanGP video flag:

```json
"video_prompt_type": "~"
```

Tiling lowers phase-two peak VRAM usage but is slower and may create seams or
local inconsistencies. Normal two-phase mode does not reduce the peak VRAM used
by the final high-resolution refinement.

If the request already uses other `video_prompt_type` flags, preserve them and
append `~` instead of replacing the complete value.

## Longer videos with sliding windows

WanGP does not use a separate `sliding_windows` boolean. Sliding-window
generation starts automatically when the requested total frame count is larger
than the configured window size:

```text
video_length > sliding_window_size
```

The primary settings are:

```json
{
  "video_length": 706,
  "sliding_window_size": 362,
  "sliding_window_overlap": 18
}
```

At 24 FPS, this example produces approximately `29.4` seconds in two windows:

```text
362 + (362 - 18) = 706 final frames
```

Current H3 window constraints are:

- `sliding_window_size`: `124` through `481`, in steps of `17`; the default is
  `362`.
- `sliding_window_overlap`: `1`, `18`, `35`, `52`, `69`, `86`, `103`, or
  `120`; the default is `18`.

The overlap frames from the previous window condition the next window and are
removed when WanGP joins the final video. A larger overlap may help continuity,
but it increases the amount of repeated generation.

### One prompt across every window

Use `FG` when the complete prompt should remain one structured prompt. WanGP
will reuse it while extending the video:

```json
{
  "kind": "video",
  "model": "minimax_h3_fl2va_pruned",
  "params": {
    "prompt": "integrated_multimodal_description: [Shot 1] A continuous cinematic sequence...\noverall_soundscape: ...\nnon_diegetic_music: N/A",
    "resolution": "832x480",
    "video_length": 706,
    "sliding_window_size": 362,
    "sliding_window_overlap": 18,
    "multi_prompts_gen_type": "FG"
  }
}
```

### A different prompt for each window

Set `multi_prompts_gen_type` to `PW` to treat each paragraph separated by an
empty line as the prompt for another window. Keep the three H3 prompt fields for
one window together without empty lines:

```json
{
  "params": {
    "prompt": "[/duration=15s] integrated_multimodal_description: [Shot 1] The first part of the continuous scene...\noverall_soundscape: ...\nnon_diegetic_music: N/A\n\n[/duration=15s,/overlap=18] integrated_multimodal_description: [Shot 1] Continue the same subject and action...\noverall_soundscape: ...\nnon_diegetic_music: N/A",
    "video_length": 706,
    "sliding_window_size": 362,
    "sliding_window_overlap": 18,
    "multi_prompts_gen_type": "PW"
  }
}
```

Useful per-window commands are:

- `[/duration=5s]`: request approximately five seconds of committed output
  from that window.
- `[/overlap=18]`: override the overlap for that transition.
- `[/new_shot]`: create a hard cut without continuity frames; equivalent to
  `[/overlap=0]`.
- `[/duration=5s,/overlap=18]`: combine multiple commands.

WanGP removes these commands before sending the creative prompt to H3. Overlap
frames are generated in addition to a window's requested duration and are not
counted twice in the final output.

Two-phase latent upscaling can be combined with sliding windows by adding
`"guidance_phases": 2`. Omit the `~` video flag to use normal whole-frame
phase-two refinement.

### How overlap continuation works

The overlap is a short memory bridge, not extra final footage. For four
eight-second windows at 24 FPS, the committed timeline is:

```text
0s          8s          16s         24s         32s
[ W1: 192 ][ W2: 192 ][ W3: 192 ][ W4: 192 ]
```

After window one, WanGP keeps its last 18 frames, representing `0.75` seconds,
as an in-memory tensor. It passes that tail and the matching audio tail to H3
FL2VA as chronological continuation context for window two. This is not a
general Ref2VA reference video and does not require encoding and reading the
intermediate MP4.

For the eight-second schedule used below, a continuation window has this
internal geometry:

```text
[18 previous-context frames][192 new committed frames][16 alignment frames]
             discarded              kept                 discarded
```

H3 accepts frame counts following its `17k + 5` geometry. The desired 18
overlap frames plus 192 new frames would total 210, which is not valid, so
WanGP generates 226 frames and trims the final 16 alignment frames. It also
removes the repeated 18-frame video and audio overlap before appending the new
content. The overlap therefore improves continuity without increasing the
final duration.

WanGP saves cumulative checkpoints after each completed window:

```text
Checkpoint 1: W1                         = 192 frames / 8s
Checkpoint 2: W1 + W2                    = 384 frames / 16s
Checkpoint 3: W1 + W2 + W3               = 576 frames / 24s
Checkpoint 4: W1 + W2 + W3 + W4          = 768 frames / 32s
```

The server configuration defaults `keep_intermediate_sliding_windows` to `1`,
so all cumulative checkpoints are retained. An eight-second checkpoint seen
while the job is still running is only window one's cumulative output, not a
failed 32-second generation.

### Continuation, reference, and control video

- A sliding-window continuation prefix is the exact, time-aligned tail of the
  preceding generated window. It tells H3 what happened immediately before the
  new footage.
- A Ref2VA reference video provides general identity, appearance, scene, or
  motion guidance. It is not necessarily chronological and is not appended to
  the output timeline.
- A control video guides motion or layout across the target timeline and is a
  separate mechanism from both continuation and identity references.

### Character identity across windows

Sliding-window overlap provides only short-term memory. If a character appears
near the beginning of window one but is absent from its final 18 frames, window
two receives no direct visual evidence of that character. Repeating a detailed
description may recreate it, but exact identity, clothing, proportions, or
surface details can drift.

Use one or more of these approaches when identity must survive across windows:

- Keep the character visible in the overlap whenever it continues into the
  next window.
- Repeat the same precise character description in every window prompt.
- Increase overlap to 35 frames when the character is visible near the
  boundary and additional context is worth the generation cost.
- Use `minimax_h3_ref2va_pruned` with persistent character reference images
  when a character may leave the frame and return later.
- Prepare camera-matched keyframes and inject them at planned timeline
  positions.

### FL2VA positioned-frame injection

H3 FL2VA can insert reference images at explicit positions using the native
`KFI` mode. For a four-window, 768-frame sequence, one possible boundary-anchor
configuration is:

```json
{
  "video_prompt_type": "KFI",
  "image_refs": [
    "/data/inputs/crab-window-1.png",
    "/data/inputs/crab-window-2.png",
    "/data/inputs/crab-window-3.png",
    "/data/inputs/crab-window-4.png"
  ],
  "frames_positions": "192 384 576 768"
}
```

WanGP requires exactly one position per reference image. Anchoring the character
at each window endpoint places it inside the context used for the following
window. Create all keyframes from one strong master image, but adapt the pose,
camera angle, action, environment, and composition to their intended moments.
Repeating one identical image at every boundary can cause composition snapping
or unnatural camera changes.

Positioned frames are best for planned poses and compositions. For general
identity persistence, switch to `minimax_h3_ref2va_pruned` and provide one or
more character references; Ref2VA keeps those references available across all
sliding windows even when the character is absent from the overlap.

Automatically promoting a character generated inside window one into later
references is not available as a single request parameter. That workflow would
require either generating window one separately, extracting a clean frame, and
submitting the continuation with that frame as a Ref2VA reference, or extending
the REST/WanGP integration to capture a generated frame and inject it into later
windows dynamically.

## Example run: four eight-second upscaled windows

This run combines paragraph-driven sliding windows with normal, whole-frame
two-phase latent upscaling.

- Submitted: `2026-08-26 20:52:53 CEST`
- Completed: `2026-08-26 21:16:35 CEST`
- Job ID: `754b799a-8813-4f92-b643-ed178aa137ec`
- Status: `succeeded`
- Output: `1664x960`, 24 FPS, `768` committed frames, approximately
  `32` seconds, with generated audio
- Final output ID: `c6d9b473-1158-446d-a80f-7c1ed8a6709b`

Normal phase-two refinement is used: the request deliberately omits the `~`
tiling flag.

### Submitted parameters

```json
{
  "kind": "video",
  "model": "minimax_h3_fl2va_pruned",
  "params": {
    "resolution": "1664x960",
    "video_length": 768,
    "sliding_window_size": 362,
    "sliding_window_overlap": 18,
    "multi_prompts_gen_type": "PW",
    "num_inference_steps": 20,
    "guidance_phases": 2,
    "seed": -1
  }
}
```

Each prompt paragraph requests `8s`, or 192 committed frames at 24 FPS. H3
preserves that exact committed duration. The first window generates 192 frames;
each continuation window internally generates 226 frames consisting of 18
overlap frames, 192 committed frames, and 16 automatically trimmed alignment
frames.

### Submitted prompt

```text
[/duration=8s] integrated_multimodal_description: [Shot 1] At desert dawn, a tiny cobalt-blue clockwork crab with weathered enamel, intricate brass joints, glowing amber lens-eyes, and a distinctive chipped left claw pushes itself out of fine wind-rippled sand. Preserve this exact design throughout the video. Begin with an extreme macro shot at sand level as grains slide from its shell. The camera slowly pushes forward, then makes a smooth clockwise semicircle around the crab as it tests its legs and turns toward a distant glint. The crab begins walking left-to-right up a small dune ridge. End with the camera tracking close beside it as the mysterious glint becomes visible beyond the crest. One continuous shot, no cuts, physically coherent motion, crisp sunrise rim light.
overall_soundscape: Fine desert wind, sand grains sliding over metal, delicate clockwork ticks, tiny brass footfalls, and one distant crystalline tone.
non_diegetic_music: A sparse glass-harmonica motif over a very soft sustained low string.

[/duration=8s,/overlap=18] integrated_multimodal_description: [Shot 1] Continue seamlessly with the same cobalt-blue clockwork crab, chipped left claw, brass joints, and amber lens-eyes already walking left-to-right over the dune crest. Preserve its scale, direction, gait, lighting, and exact appearance. The camera maintains a low lateral tracking movement as the crab descends into a sheltered hollow, creating strong parallax between foreground sand ridges. A half-buried orchestra of transparent glass instruments is gradually revealed ahead. The camera rises smoothly from ground level into an over-the-shell perspective while remaining in motion. The crab approaches a tall crystal key and slowly extends its intact right claw. End with the claw hovering millimeters from the glass. One continuous shot, no cuts.
overall_soundscape: Continuous desert wind and clockwork footfalls, joined by faint resonant glass vibrations as the crab approaches the buried instruments.
non_diegetic_music: The glass-harmonica motif gains a quiet plucked-string pulse while retaining the same tempo.

[/duration=8s,/overlap=18] integrated_multimodal_description: [Shot 1] Continue seamlessly from the same moment as the cobalt-blue clockwork crab’s intact right claw touches the crystal key. Preserve the crab, glass orchestra, sunrise direction, and spatial layout exactly. The crystal rings and translucent musical notes bloom from it, transforming into small luminous birds made from refracted light. Begin in macro close-up on the claw and vibrating crystal, then smoothly dolly backward between the glass instruments as more glowing birds emerge. Without cutting, crane upward with the flock while keeping the crab visible in the lower center of the composition. The crab raises its distinctive chipped left claw triumphantly. End with the birds beginning a wide clockwise spiral overhead and the camera still rising. Physically coherent glass, reflections, and motion.
overall_soundscape: The continuing wind and mechanical ticks blend with layered crystalline chimes, soft wingbeats, and grains of sand disturbed by the rising flock.
non_diegetic_music: Glass harmonica and warm strings expand gently, synchronized with the birds’ emergence without overpowering the natural sounds.

[/duration=8s,/overlap=18] integrated_multimodal_description: [Shot 1] Continue seamlessly with the camera already rising above the same cobalt-blue clockwork crab and the luminous birds already spiraling clockwise over the half-buried glass orchestra. Preserve every identity, direction, object, and lighting detail. The camera transitions from the upward crane into a broad, slow counterclockwise orbit, revealing the crab looking upward with its chipped left claw still raised. Then execute a smooth accelerating aerial pullback across the amber dunes. The luminous flock stretches into a graceful arc that points toward the tiny crab and the sparkling glass instruments below. End in a majestic very wide aerial tableau with the crab still clearly identifiable at the center of the glowing desert landmark. One continuous shot, no cuts, no subject duplication or morphing.
overall_soundscape: Desert wind broadens across the wide landscape while crystalline echoes, clockwork ticks, and soft wingbeats gradually recede into the distance.
non_diegetic_music: The glass and string theme reaches one warm sustained resolution, then fades naturally with the final aerial pullback.
```

### Produced checkpoints

WanGP retained four cumulative MP4 files:

- Window 1: 8 seconds, `30,837,539` bytes.
- Windows 1-2: 16 seconds, `68,993,022` bytes.
- Windows 1-3: 24 seconds, `109,651,059` bytes.
- Windows 1-4: 32 seconds, `142,967,848` bytes; this is the final artifact.

### Review checklist

Verify the reported resolution, frame count, FPS, duration, and audio stream.
Visually inspect approximately one second on both sides of the joins at 8, 16,
and 24 seconds for crab identity, direction of travel, camera momentum,
lighting, and audio continuity.

## Final decoded-video upscale with FlashVSR

FlashVSR is a second, independent upscale stage. Unlike H3 two-phase generation,
its input is an already completed, decoded image or video rather than H3's
in-progress latent tensors, and it does not rerun H3 denoising. However,
FlashVSR is not a conventional pixel resize filter: it encodes the decoded
frames into its own learned representation, runs generative reconstruction and
denoising, and decodes the reconstructed frames into a new upscaled file.
Therefore, “decoded-video postprocessing” describes the boundary between H3
and FlashVSR, not FlashVSR's internal implementation. Use Lanczos or another
ordinary scaler when a purely non-generative pixel resize is desired.

Use the stages in this order for a long sliding-window video:

1. Generate every H3 window with `guidance_phases: 2` if two-phase latent
   upscaling is desired.
2. Let WanGP remove overlaps and assemble the complete cumulative video.
3. Submit the final MP4 as a separate WanGP late-postprocessing task.
4. Crop and resize the FlashVSR output only if an exact delivery resolution is
   required.

Do not put FlashVSR in the H3 sliding-window generation request when the goal is
one final upscale. Generation-time postprocessing may run as each cumulative
window is produced. Late postprocessing guarantees that the FlashVSR
reconstruction runs once on the finished video.

### FlashVSR method values

WanGP serializes the method and multiplier in `spatial_upsampling`:

```text
flashvsr2         one-pass 2x
flashvsr2pass2    two-pass 2x
flashvsr2.5       one-pass 2.5x
flashvsr2pass2.5  two-pass 2.5x
flashvsr4         one-pass 4x
flashvsr2pass4    two-pass 4x
```

The current FlashVSR integration accepts multipliers from `1x` through `4x` in
`0.5x` increments. Two-pass mode performs two reconstruction passes and may
reduce horizontal banding, but costs approximately twice as much processing.

### Headless FlashVSR configuration

FlashVSR is enabled globally in `wgp_config.json`, not by an H3 generation
parameter. No Gradio interface is required. The relevant configuration is:

```json
{
  "spatial_upsamplers": {
    "persistence": 1,
    "flashvsr": {
      "mode": 1,
      "backend": "auto",
      "topk_ratio": 0.0
    }
  }
}
```

The values mean:

- `mode: 0`: disabled.
- `mode: 1`: Tiny, with faster decoding and lower RAM usage; recommended for
  long videos.
- `mode: 2`: Full, with the best reconstruction quality but slower decoding
  and greater RAM usage.
- `persistence: 1`: unload the spatial upsampler after the task.
- `backend: "auto"`: let WanGP select the available sparse Triton backend.
- `topk_ratio: 0.0`: use WanGP's automatic sparse-attention quality setting.

FlashVSR requires Triton. SpargeAttention is optional. Configuration validation
and a small postprocessing test should be performed before committing to a
long, high-resolution run.

WanGP's Python API uses `<root>/wgp_config.json` by default and also accepts an
explicit `config_path` pointing to a file named `wgp_config.json`. The current
Modal image build calls `generate_catalog.py`, which initializes WanGP and
creates `/opt/Wan2GP/wgp_config.json`. The pinned WanGP fresh-install migration
currently promotes FlashVSR from disabled to Tiny mode. This migration side
effect should not be treated as the application contract: the image build
should explicitly set and validate the FlashVSR section so a future WanGP
upgrade cannot silently disable it.

For this stateless REST deployment, a configuration baked into the Modal image
is preferable to several cold-starting workers mutating one Volume-backed file.
If configuration must become runtime-editable later, pass an explicit
`config_path` under `/data/config/` and add concurrency-safe configuration
management.

### Late-postprocessing task

The native WanGP task shape for final pixel upscaling is:

```json
{
  "mode": "edit_postprocessing",
  "prompt": "Media postprocessing",
  "image_mode": 0,
  "video_source": "/data/outputs/final-video.mp4",
  "temporal_upsampling": "",
  "spatial_upsampling": "flashvsr2.5"
}
```

WanGP also exposes the equivalent Python helper:

```python
job = session.submit_media_postprocessing(
    "/data/outputs/final-video.mp4",
    spatial_upsampling="flashvsr2.5",
)
result = job.result()
```

The REST service currently requires a top-level model so it can select a worker,
even though native late-postprocessing tasks do not need a generation model.
A dedicated postprocessing route would remove that transport-level mismatch and
call `submit_media_postprocessing` directly. Until then, a video model can be
used only for worker routing while `mode: "edit_postprocessing"` selects the
native WanGP operation.

### Resolution and exact UHD delivery

FlashVSR takes a scale multiplier rather than a target width and height. For an
H3 output at `1664x960`:

```text
2x   -> 3328x1920
2.5x -> 4160x2400
4x   -> 6656x3840
```

None is exactly UHD `3840x2160`, and `1664x960` is slightly narrower than
16:9. A practical exact-UHD finishing path is:

```text
1664x960
  -> FlashVSR 2.5x: 4160x2400
  -> centered crop: 4160x2340
  -> high-quality downscale: 3840x2160
```

This avoids geometric stretching. FlashVSR `4x` followed by a crop and
downscale provides more oversampling but is substantially more expensive. Use
`2.5x` for the normal final-4K workflow and reserve `4x` for comparisons where
the additional detail justifies the cost.

### Completed FlashVSR 2.5x run: robot performance

On 2026-08-26, the final approximately 40-second robot singing-and-dancing MP4
was submitted as a separate late-postprocessing task:

```json
{
  "kind": "video",
  "model": "minimax_h3_fl2va_pruned",
  "params": {
    "mode": "edit_postprocessing",
    "prompt": "Media postprocessing",
    "image_mode": 0,
    "video_source": "/data/outputs/2026-08-26-20h05m33s_seed690112156_[duration=8s] integrated_multimodal_description [Shot 1] In a playful retro-futurist music room, a.mp4",
    "temporal_upsampling": "",
    "spatial_upsampling": "flashvsr2.5"
  }
}
```

- WanGP job: `0ee9bc80-1581-484e-abdb-0d0736766f58`
- Status: `succeeded` at 2026-08-26 23:53 CEST
- FlashVSR configuration: mode 1 (Tiny), automatic sparse-attention backend
- Input canvas: `1664x960`
- Output canvas: `4160x2400`; WanGP temporarily edge-padded to `4224x2432`
  and restored the final canvas with a crop
- Output ID: `4bf11425-d826-49aa-badb-449af9746ab0`
- Output filename suffix: `_post.mp4`
- Output size: `879,939,590` bytes (approximately 839 MiB)

The first run also downloaded the FlashVSR transformer, projection model,
TCDecoder, prompt embedding, and Wan VAE. Those downloads are cached on the
Modal Volume for later runs. The reconstruction used 119 temporal denoising
chunks followed by 119 TCDecoder chunks, then full-resolution color correction
and MP4 encoding. This explains why an H100 run is still lengthy: the operation
is a generative video reconstruction over roughly 9.7 billion output pixels,
not a simple resize.

## Upstream implementation

- [WanGP v12.643 update](https://github.com/deepbeepmeep/Wan2GP/blob/92f56e5ee7227d490f6d85281c019e4c4e2dc393/README.md)
- [MiniMax H3 two-phase pipeline](https://github.com/deepbeepmeep/Wan2GP/blob/92f56e5ee7227d490f6d85281c019e4c4e2dc393/models/minimax_h3/pipeline.py)
- [WanGP sliding-window processing guide](https://github.com/deepbeepmeep/Wan2GP/blob/92f56e5ee7227d490f6d85281c019e4c4e2dc393/docs/PROCESSING.md)
- [WanGP prompt commands](https://github.com/deepbeepmeep/Wan2GP/blob/92f56e5ee7227d490f6d85281c019e4c4e2dc393/docs/PROMPTS.md)
- [WanGP Python API and late postprocessing](https://github.com/deepbeepmeep/Wan2GP/blob/92f56e5ee7227d490f6d85281c019e4c4e2dc393/docs/API.md)
- [Spatial upsampler configuration](https://github.com/deepbeepmeep/Wan2GP/blob/92f56e5ee7227d490f6d85281c019e4c4e2dc393/docs/SPATIAL_UPSAMPLERS.md)
- [FlashVSR bridge and supported scales](https://github.com/deepbeepmeep/Wan2GP/blob/92f56e5ee7227d490f6d85281c019e4c4e2dc393/postprocessing/flashvsr/wgp_bridge.py)
- [WanGP extension configuration migration](https://github.com/deepbeepmeep/Wan2GP/blob/92f56e5ee7227d490f6d85281c019e4c4e2dc393/shared/utils/wgp_config_migration.py)


Example: 
