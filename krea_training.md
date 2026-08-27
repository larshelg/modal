# Krea2 LoRA Training --- Fizgig + Modal + Codex-First REST Workflow

## Goal

Build a headless Krea2 LoRA training workflow where Fizgig performs
training, Modal supplies GPU compute, and the existing protected REST
interface incrementally exposes jobs, progress, checkpoints, previews,
comparisons, evaluation, and checkpoint promotion.

The immediate client is **Codex**, using a local project skill file.
Eve integration comes later, after the complete workflow has been used
and validated through Codex. Eve is the eventual conversational agent
framework, not a prerequisite for local development or for proving the
training-to-evaluation flow.

The key principle is:

> A checkpoint file is not the primary user-facing object. The visual
> evaluation of that checkpoint is.

The user should rarely need to inspect `.safetensors` manually.

## Current milestone

The first runnable Krea2 training slice is now implemented in
`fizgig-modal-app`, exposed through `rest-wangpt-modal-app`, and operated by
the project-local Codex skill. It supports:

- Volume-backed datasets at `/data/fizgig/datasets/<dataset>/images`;
- Qwen3-VL auto-captioning of missing or empty sidecars before caching;
- the pinned Fizgig Krea2 latent-cache, text-cache, and training scripts;
- `krea2_defaults` and `krea2_ultra_fast` presets;
- per-image loss/LR behavior, adaptive LR in the ultra-fast preset, and
  Qwen3-VL auto-recaptioning of confirmed-stuck images;
- checkpoints, cooperative pause/resume, and final LoRA promotion to
  `/data/loras/<output_name>.safetensors`.

The public request intentionally contains only family, dataset, output name,
preset, optional trigger word, and optional epochs. Preview prompts and related
controls return in the later preview-resource phase.

The next live steps are deploying both updated Modal apps, fetching the Krea2
model family, and submitting the staged `linda` dataset through the Codex
skill. Checkpoint listing, comparison grids, evaluation, and per-epoch
promotion remain later phases. Eve integration remains deferred until this
Codex-driven flow has been validated.

## Delivery order

Implement and validate the clients in this order:

1.  Create `.agents/skills/fizgig/SKILL.md` for Codex.
2.  Use that skill to exercise the authenticated
    `rest-wangpt-modal-app` training and generation routes.
3.  Extend the skill as Krea2 checkpoint, preview, comparison,
    evaluation, and promotion routes become available.
4.  Validate the end-to-end Krea2 workflow locally through Codex.
5.  Integrate Eve later by mapping Eve tools onto the already proven
    REST operations and semantics.

The skill must call the shared REST interface. It must not call the
internal `fizgig-modal-app` directly, invoke raw training shell
commands, or duplicate Fizgig request-to-CLI translation.

All Eve dialogue and tool examples later in this document describe the
future user experience. Until the Eve milestone begins, use Codex plus
the local skill as the agent-facing interface for the same flows.

## Current development architecture

``` text
USER
  |
  v
CODEX
  |
  v
.agents/skills/fizgig/SKILL.md
  |
  v
rest-wangpt-modal-app
  |
  +--------------------------+
  |                          |
  v                          v
fizgig-modal-app       WanGP generation worker
  |
  v
Fizgig CLI
  |
  +-- checkpoints
  +-- previews
  +-- metrics
  |
  v
Modal Volume
```

## Later Eve architecture

After the Codex workflow is proven, Eve replaces Codex only at the
agent-framework layer:

``` text
USER
  |
  v
EVE
  |
  v
Eve Fizgig tools
  |
  v
rest-wangpt-modal-app
  |
  +-- training routes --> fizgig-modal-app --> Fizgig CLI
  |
  +-- generation routes --> WanGP worker
```

The REST contracts, Modal apps, job lifecycle, and storage layout should
not be redesigned for Eve. Eve should reuse the behavior first proven by
the Codex skill.

For final checkpoint evaluation:

``` text
Fizgig checkpoints
        |
        v
candidate epochs
        |
        v
Wan2GP Krea2 inference
        |
        v
same prompts + seeds + settings
        |
        v
comparison grids
        |
        v
Codex (Eve later) -> user selects winner -> promote checkpoint
```

## Future Eve user experience

``` text
User:
Train a Krea2 character LoRA for Anna from dataset anna-v3.
Use Krea2 Defaults.
Generate samples every 2 epochs using my standard character prompts.

Eve:
Starting Anna Krea2 training.

Dataset: anna-v3
Preset: krea2_defaults
Images: 38
Epochs: 30
Preview interval: 2 epochs
```

Later:

``` text
User:
How's Anna doing?

Eve:
Epoch 18/30.

Loss is still improving.
No confirmed plateau yet.

Two images were flagged for caption mismatch and were recaptioned.
The latest previews are available.
```

After completion:

``` text
Eve:
Anna training completed.

Plateau analysis suggests the strongest region is epochs 22–26.
I've prepared previews for epochs 22, 24 and 26.
```

The user can then request:

``` text
Test 22, 24 and 26 using Wan2GP.
```

Eve runs all candidates through the real Krea2 inference environment
with identical prompts, seeds and settings.

``` text
User:
24 is clearly best.

Eve:
Epoch 24 promoted.

anna_krea2_v1.safetensors
/data/loras/anna_krea2_v1.safetensors
```

## Training Job Creation

Training is asynchronous.

``` http
POST /training/jobs
```

Example request:

``` json
{
  "family": "krea2",
  "dataset": "anna-v3",
  "output_name": "anna_krea2_v1",
  "preset": "krea2_defaults",
  "trigger_word": "annaxs"
}
```

Response:

``` json
{
  "id": "job_123",
  "status": "queued"
}
```

Do not keep the HTTP request open for the duration of training.

## Presets

Eve should normally select tested presets rather than inventing
hyperparameters.

Implemented presets from the pinned Fizgig revision:

``` text
krea2_defaults
krea2_ultra_fast
```

`krea2_defaults` uses rank/alpha 32 for 30 epochs. `krea2_ultra_fast`
uses rank/alpha 8 for 20 epochs and enables adaptive LR. Both default to
initial auto-captioning and mid-run auto-recaptioning.

Optional overrides can be supported:

``` json
{
  "preset": "krea2_defaults",
  "epochs": 40
}
```

Validate all overrides.

## Job Status

``` http
GET /training/jobs/{job_id}
```

Example:

``` json
{
  "id": "job_123",
  "status": "running",
  "progress": {
    "phase": "training",
    "epoch": 12,
    "epochs_total": 30
  }
}
```

Useful states:

``` text
queued
preparing_dataset
captioning
caching
starting
training
paused
evaluating
completed
failed
cancelled
```

## Structured Events

Prefer structured events to parsing logs.

``` json
{
  "event": "epoch_completed",
  "epoch": 15,
  "loss": 0.082,
  "checkpoint": "epoch_015.safetensors"
}
```

Useful events include:

``` text
dataset_prepared
captioning_started
captioning_completed
image_recaptioned
cache_complete
training_started
epoch_started
epoch_completed
checkpoint_saved
preview_generated
problem_image_detected
plateau_detected
training_completed
training_failed
```

Keep raw logs for debugging, but do not make Eve depend on their text
format.

## Checkpoints as REST Resources

Do not make Eve inspect the filesystem directly.

``` http
GET /training/jobs/{job_id}/checkpoints
```

Example:

``` json
{
  "checkpoints": [
    {
      "id": "checkpoint_006",
      "epoch": 6,
      "artifact_url": "...",
      "loss": 0.094,
      "preview_count": 4
    },
    {
      "id": "checkpoint_008",
      "epoch": 8,
      "artifact_url": "...",
      "loss": 0.083,
      "preview_count": 4
    },
    {
      "id": "checkpoint_010",
      "epoch": 10,
      "artifact_url": "...",
      "loss": 0.076,
      "preview_count": 4
    }
  ]
}
```

Useful checkpoint metadata:

``` text
id
epoch
step
loss
learning_rate
created_at
artifact_path
artifact_size
preview_count
evaluation_score
likeness_score
plateau_region
promoted
```

Do not assume the lowest training loss is automatically the visually
best LoRA.

## Preview Inspection

The normal way to inspect a checkpoint is through generated images.

``` http
GET /training/jobs/{job_id}/checkpoints/{epoch}/previews
```

Example:

``` json
{
  "epoch": 10,
  "previews": [
    {"prompt_id": "portrait", "image_url": "https://.../epoch_010/portrait.webp"},
    {"prompt_id": "smile", "image_url": "https://.../epoch_010/smile.webp"},
    {"prompt_id": "full_body", "image_url": "https://.../epoch_010/full_body.webp"},
    {"prompt_id": "cafe", "image_url": "https://.../epoch_010/cafe.webp"}
  ]
}
```

Store the prompt, seed, resolution, inference settings, LoRA strength
and base model/version so comparisons are reproducible.

## Fixed Seeds

Checkpoint comparisons must use fixed seeds.

If epoch 18 and epoch 24 use different random seeds, visual comparison
becomes much less useful.

Evaluation suites should therefore contain fixed prompt/seed pairs.

## Test Multiple Dimensions

Do not judge a character LoRA from one portrait.

A useful standard suite can test:

``` text
portrait
smile
profile
full body
outdoors
indoors
different clothing
different lighting
```

This helps identify likeness, flexibility, overfitting, pose/expression
generalization, clothing leakage and background leakage.

## Contact Sheet / Comparison Endpoint

Add an endpoint specifically for visual checkpoint comparison.

``` http
POST /training/jobs/{job_id}/comparisons
```

Example request:

``` json
{
  "epochs": [8, 10, 12, 14, 16, 18],
  "prompt_ids": ["portrait"]
}
```

Result:

``` json
{
  "comparison_id": "cmp_123",
  "image_url": "https://.../comparison.webp"
}
```

Conceptually:

``` text
          E08      E10      E12      E14      E16      E18
       +--------+--------+--------+--------+--------+--------+
Anna   | image  | image  | image  | image  | image  | image  |
       +--------+--------+--------+--------+--------+--------+
```

Eve can then handle:

``` text
Show me the portrait progression from epochs 8 through 20.
```

For several prompts:

``` text
              E18        E20        E22        E24

portrait      image      image      image      image
smile         image      image      image      image
full_body     image      image      image      image
cafe          image      image      image      image
```

This should be a primary checkpoint inspection interface.

## LoRA Royale / Epoch Morph

Fizgig has functionality for inspecting LoRA behavior across epochs
("LoRA Royale"). Inspect the current upstream implementation and reuse
it headlessly if practical.

Conceptual endpoint:

``` http
POST /training/jobs/{job_id}/epoch-morph
```

Request:

``` json
{
  "prompt": "annaxs woman, close-up portrait",
  "seed": 123456,
  "start_epoch": 4,
  "end_epoch": 30
}
```

Response:

``` json
{
  "video_url": "https://.../anna-v3/epoch_morph.mp4"
}
```

Then Eve can respond to:

``` text
Show me how likeness develops across the run.
```

with:

``` text
epoch 4 -> 6 -> 8 -> 10 -> 12 -> ... -> 30
```

This makes the likeness sweet spot and possible overtraining easy to
see.

## Plateau Detection

Where Fizgig provides plateau detection, surface it through REST.

``` json
{
  "plateau": {
    "detected": true,
    "suggested_start_epoch": 18,
    "suggested_end_epoch": 24
  }
}
```

Eve can say:

``` text
Fizgig detected a plateau.

Suggested checkpoint window: epochs 18–24.
I recommend comparing 18, 20, 22 and 24.
```

Use plateau detection to narrow candidates, not to automatically choose
the production LoRA.

## Auto Captioning and Recaptioning

Expose Fizgig's captioning capabilities in the workflow.

``` text
dataset
  |
  v
auto caption
  |
  v
cache
  |
  v
training
  |
  v
problem image detected
  |
  v
recaption if supported/configured
  |
  v
continue training
```

Example event:

``` json
{
  "event": "image_recaptioned",
  "image_id": "img_017",
  "reason": "persistent_high_loss"
}
```

Eve might summarize:

``` text
Two images were consistently producing unusually high loss.
Fizgig recaptioned them and training continued.
```

Require user approval for destructive changes to the canonical dataset.

## Wan2GP Production Evaluation

Fizgig previews are excellent for monitoring training, but final
candidates should also be tested in the actual production inference
environment.

Use the existing Wan2GP Krea2 pipeline as a second evaluation stage.

``` text
                    Fizgig
                       |
          +------------+------------+
          v            v            v
       epoch 22      epoch 24      epoch 26
          |            |            |
          +------------+------------+
                       |
                       v
                    Wan2GP
                       |
                same Krea2 model
                same prompt suite
                same seeds
                same resolution
                same settings
                same LoRA strength
                       |
                       v
                comparison sheets
```

This answers the important question:

> Which checkpoint performs best in the production generation stack?

## Evaluation Endpoint

``` http
POST /training/jobs/{job_id}/evaluate
```

Example:

``` json
{
  "backend": "wangp",
  "epochs": [22, 24, 26],
  "suite": "character_standard_v1"
}
```

Response:

``` json
{
  "evaluation_id": "eval_456",
  "status": "running"
}
```

Later:

``` http
GET /evaluations/eval_456
```

Example:

``` json
{
  "status": "completed",
  "epochs": [22, 24, 26],
  "comparison_urls": [
    "https://.../portrait_grid.webp",
    "https://.../smile_grid.webp",
    "https://.../full_body_grid.webp",
    "https://.../cafe_grid.webp"
  ]
}
```

## Promotion

Once a checkpoint is selected, explicitly promote it.

``` http
POST /training/jobs/{job_id}/checkpoints/{epoch}/promote
```

Example:

``` http
POST /training/jobs/job_123/checkpoints/24/promote
```

Response:

``` json
{
  "status": "promoted",
  "lora": "anna_krea2_v1.safetensors",
  "source_epoch": 24,
  "artifact_path": "/data/loras/anna_krea2_v1.safetensors"
}
```

Promotion should:

1.  identify the source checkpoint;
2.  copy the selected `.safetensors` to canonical Modal Volume storage;
3.  register the LoRA/version in Postgres;
4.  record the training job and selected epoch;
5.  preserve relevant training/evaluation metadata.

## LoRA Registry Metadata

Example:

``` json
{
  "name": "anna",
  "version": "v1",
  "family": "krea2",
  "trigger_word": "annaxs",
  "dataset": "anna-v3",
  "training_job_id": "job_123",
  "checkpoint_epoch": 24,
  "preset": "krea2_defaults",
  "evaluation_suite": "character_standard_v1",
  "artifact": "/data/loras/anna_krea2_v1.safetensors"
}
```

Potential tables:

``` text
datasets
training_jobs
training_checkpoints
training_previews
training_events
evaluations
evaluation_images
loras
lora_versions
```

## Checkpoint Retention

Initially retain all useful checkpoints.

``` text
training completed
       |
       v
retain checkpoints
       |
       v
evaluate candidates
       |
       v
promote winner
       |
       v
optional retention policy later
```

A later policy could keep the promoted checkpoint, nearby candidates,
final checkpoint and best evaluation candidates while archiving/deleting
uninteresting intermediates.

Do not optimize checkpoint retention before the workflow is proven.

## Modal Volume Layout

``` text
/models
    /krea2

/datasets
    /anna-v3

/cache
    /anna-v3

/runs
    /job_123
        config.json
        training_state.json
        events.jsonl

        /checkpoints
            epoch_002.safetensors
            epoch_004.safetensors
            epoch_006.safetensors

        /previews
            /epoch_002
                portrait.webp
                smile.webp
                full_body.webp
                cafe.webp

        /comparisons
            portrait_progression.webp
            candidate_grid.webp

        /royale
            portrait_epoch_morph.mp4
```

## Later S3 layout

The first Codex-driven version uses the shared Modal Volume, not S3.
Introduce S3 only in a later persistence phase after the active workflow
is reliable. The intended later layout is:

``` text
s3://datasets/
    anna/
        v3/

s3://training-runs/
    job_123/
        config.json
        metrics.json
        previews/
        comparisons/

s3://loras/
    anna/
        krea2/
            v1/
                anna_krea2_v1.safetensors
                metadata.json
```

Modal Volume holds all artifacts in the first version. Later, it can
continue to hold active run artifacts while S3 holds canonical and
long-term artifacts.

## Codex skill surface (Eve tools later)

Create `.agents/skills/fizgig/SKILL.md` before Eve integration. Keep its
operations high level and map them to the authenticated REST API:

``` text
start_krea_training()
get_training_status()
list_checkpoints()
get_checkpoint_previews()
compare_checkpoints()
generate_epoch_morph()
evaluate_checkpoints()
promote_checkpoint()
pause_training()
resume_training()
cancel_training()
```

The Codex skill should provide the first usable version of these flows.
When Eve integration begins later, expose equivalent typed tools with
the same payloads and lifecycle behavior. Neither client should need raw
filesystem paths.

Example:

``` text
compare_checkpoints(
    job_id="job_123",
    epochs=[18,20,22,24],
    prompts=["portrait","smile","full_body","cafe"]
)
```

## Conversational Examples

Start:

``` text
User:
Train Anna using Krea2.

Eve:
I'll use dataset anna-v3 and the standard Krea2 character preset.
Starting 30 epochs with previews every 2 epochs.
```

Status:

``` text
User:
How is Anna doing?

Eve:
Epoch 18/30.
Training is still improving and no plateau has been confirmed.
The latest four previews are ready.
```

Inspect progression:

``` text
User:
Show me the portrait from 10 to 20.

Eve:
I've generated a fixed-seed comparison for epochs
10, 12, 14, 16, 18 and 20.
```

Narrow candidates:

``` text
User:
Which epochs should we look at?

Eve:
Plateau detection suggests epochs 20–26.
I'd compare 20, 22, 24 and 26 visually.
```

Production test:

``` text
User:
Test 22, 24 and 26 for real.

Eve:
I'm running those checkpoints through Wan2GP using the
standard Krea2 character test suite with identical seeds and settings.
```

Promote:

``` text
User:
24 wins.

Eve:
Epoch 24 promoted to anna_krea2_v1.safetensors.
```

## Fizgig GUI Debugging

The REST interface should not remove access to Fizgig's original GUI.

Use one Fizgig installation:

``` text
                   Fizgig installation
                          |
               +----------+----------+
               v                     v
         Headless / CLI          Original GUI
               |                     |
               +----------+----------+
                          |
                          v
                    same storage
```

Use the GUI as an optional debugging/admin interface when REST behavior
is incorrect, training telemetry needs deeper inspection, datasets need
manual investigation, or upstream functionality is not yet exposed via
REST.

Do not create separate model caches/run directories for GUI and REST.

Investigate the current GUI implementation before choosing remote
exposure. If web-native, expose it through a protected temporary Modal
endpoint. If desktop/display based, consider an on-demand virtual
display/noVNC container.

Do not reimplement the Fizgig GUI.

## Security

Protect all training endpoints.

At minimum:

-   authentication
-   job authorization
-   validated dataset IDs
-   validated checkpoint IDs
-   model-family allowlist
-   preset allowlist
-   numeric parameter validation
-   maximum training duration
-   GPU allocation limits
-   output path restrictions
-   no arbitrary shell commands
-   no arbitrary filesystem access

Codex must use the skill's typed REST operations, never raw training
shell commands. Eve must follow the same rule when it is integrated
later.

## Codex implementation instructions

First create `.agents/skills/fizgig/SKILL.md` around the routes already
implemented by `rest-wangpt-modal-app`. The initial skill should cover
training submission, status polling, pause, resume, cancellation, and
WanGP generation without calling `fizgig-modal-app` directly.

Before extending the REST layer for Krea2, inspect current Fizgig
source/docs for:

1.  headless/CLI training entry points;
2.  Krea2 preset/configuration handling;
3.  dataset preparation;
4.  auto-captioning;
5.  recaptioning;
6.  training callbacks/events;
7.  checkpoint generation;
8.  pause/resume;
9.  preview generation;
10. plateau detection;
11. LoRA Royale / epoch comparison;
12. output directory conventions.

Reuse Fizgig functionality wherever possible.

Do not reimplement features already available upstream.

The REST layer should mainly:

``` text
validate request
      |
      v
translate high-level request
      |
      v
invoke Fizgig
      |
      v
collect structured state
      |
      v
expose through REST
```

## Implementation Phases

### Phase 0 --- Codex Skill and Existing Flow Validation (in progress)

Create and validate `.agents/skills/fizgig/SKILL.md` against the shared
authenticated REST API.

Implement skill guidance for:

``` text
submit training
poll training status
pause training
resume paused training
cancel training
submit WanGP generation
poll generation status
```

Use the existing API job IDs and return control to the user between
polls. Keep credentials out of skill documentation and output. Do not
call named functions in `fizgig-modal-app` from the skill.

Goal: Codex can reliably operate the current flow before Krea2-specific
API capabilities or Eve integration are added.

Current status: the project skill, REST reference, and typed client are
implemented. Local client tests are complete. Live validation against
the deployed authenticated endpoint is still required.

### Phase 1 --- Basic Krea2 Training

Implement:

``` text
POST training job
GET job status
GET checkpoints
GET previews
POST promote checkpoint
```

Use a prepared dataset, one tested Krea2 preset, fixed sample prompts,
Modal Volume and final `.safetensors`.

Goal: reliable headless Krea2 training.

### Phase 2 --- Better Inspection

Add:

``` text
checkpoint comparison grids
fixed-seed evaluation suites
plateau information
epoch morph / LoRA Royale
pause/resume
structured events
```

Goal: make REST inspection better than browsing checkpoint files
manually.

### Phase 3 --- Wan2GP Evaluation

Connect candidate checkpoints to the existing Wan2GP Krea2 inference
service.

Add `evaluate_checkpoints()` using the same prompts, seeds, inference
settings and LoRA strength.

Goal: select checkpoints based on the real production inference
environment.

### Phase 4 --- Eve Integration and Agentic Training Assistant

Only after Phases 0–3 have been validated through Codex, map the proven
skill operations and REST semantics into Eve tools. Then let Eve
interpret training intelligence.

Examples:

``` text
Training has plateaued. Compare epochs 20–26?

Image 17 remains an outlier after recaptioning.
Would you like to inspect it?

Epoch 24 appears stronger than 26 for likeness,
while 26 shows slightly better pose flexibility.

The final epoch appears mildly overtrained compared with 24.
```

Keep destructive decisions user-approved.

## Desired End State

``` text
User:
Train Anna Krea2 from anna-v3.

Eve:
Started.

       |
       v

Fizgig:
caption -> cache -> train -> checkpoint -> preview

       |
       v

Eve:
Epoch 18/30. Still improving.

       |
       v

Fizgig:
plateau detected around 22–26

       |
       v

Eve:
I recommend comparing 22, 24 and 26.

       |
       v

User:
Test them in Wan2GP.

       |
       v

Wan2GP:
same prompts + same seeds + same settings

       |
       v

Eve:
Here are the comparison grids.

       |
       v

User:
24 wins.

       |
       v

Eve:
Promoted epoch 24.

anna_krea2_v1.safetensors
```

During development, the user interacts through Codex and
`.agents/skills/fizgig/SKILL.md`. Later, Eve becomes the conversational
agent framework by reusing the same REST flows. Fizgig remains the
training engine. Wan2GP remains the production inference/testing engine.
Modal supplies GPU compute. Modal Volumes preserve active training state.
S3 and Postgres remain later persistence layers rather than requirements
for the first Codex-driven workflow. The Fizgig GUI remains an optional
debugging/admin console.
