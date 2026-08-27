# Fizgig on Modal — Agent-Controlled Headless LoRA Training

## Goal

Run Fizgig as a headless LoRA training engine on Modal. Expose a small,
stable API over Fizgig's documented command-line workflow instead of
automating or deploying the Fizgig UI.

Eve is the eventual conversational control plane. Eve is an **agent
framework**, not the training runtime and not the WanGP service. It
should orchestrate Fizgig training and use the existing
`rest-wangpt-modal-app` for WanGP inference, preview generation, and
checkpoint testing. The same in-project Modal app will also be extended
to provide the Fizgig REST interface, while a new `fizgig-modal-app`
will own the Fizgig runtime and GPU jobs behind that interface.

Eve integration is deliberately deferred. During local development,
Codex will act as the agent client. A project-local `SKILL.md` will teach
Codex the same operations, request shapes, safety rules, and workflow
that Eve will use later.

## Scope and delivery order

There are four distinct responsibility areas:

1.  **Public REST application:** reuse and extend the existing
    `rest-wangpt-modal-app` already in this project. It remains the
    authenticated asynchronous REST control plane for both job domains.
2.  **Fizgig execution application:** create a new project folder named
    `fizgig-modal-app`, modeled on the existing `my-wangpt-modal-app`.
    It owns the Fizgig image, pinned source, CLI, GPU configuration, and
    long-running training functions.
3.  **WanGP execution:** preserve the existing generation contract while
    routing image/audio jobs to an L40S worker and video jobs to a separate
    H100 worker.
4.  **Agent layer:** Codex plus a local skill during development, then
    Eve plus the equivalent skill/integration later.

The implementation order is:

``` text
1. Create fizgig-modal-app from the relevant my-wangpt-modal-app patterns
2. Extend rest-wangpt-modal-app with Fizgig training routes
3. Connect those routes to fizgig-modal-app training functions
4. Add a local Codex SKILL.md using the extended REST API
5. Validate end-to-end training and generation workflows
6. Integrate the proven skill/tool contract into Eve
```

Implementation is split into two delivery stages:

### Stage 1 — `fizgig-modal-app` only (completed)

Create and validate the standalone Fizgig execution app. It must be
deployable and directly testable through named Modal functions, but it
must not modify `rest-wangpt-modal-app` or expose a new public REST/MCP
server. Stage 1 includes the pinned Fizgig image, shared Volume mount,
model fetch, H3 cache/train pipeline, job metadata, pause requests,
resume support, tests, and deployment documentation.

### Stage 2 — REST integration (implemented)

The existing `rest-wangpt-modal-app` now exposes namespaced submit,
status, pause, resume, and cancel routes under `/training/jobs` and
dispatches them to the deployed `fizgig-modal-app`. The WanGP routes are
preserved. Public training records, internal Fizgig records, and WanGP
records use separate Modal Dicts. The Codex skill and later Eve
integration should use this REST surface rather than call the execution
app directly.

Eve-specific framework work is not required to complete the local
development phases.

### Generation GPU routing milestone (implemented locally)

`POST /jobs` now accepts an optional `kind` of `image`, `video`, or `audio`.
The REST API validates explicit values against WanGP's baked `main_output`
model metadata and infers the kind when it is omitted, preserving existing
clients. Image and audio jobs use the L40S worker pool. Video jobs, including
MiniMax H3, use a separate H100 worker pool with independently configurable
container count, host RAM, and WanGP memory profile. Status, cancellation,
output download, job storage, and the shared `wangp-data` Volume remain common.

### Krea2 pipeline milestone (implemented locally)

The execution app and REST request boundary now support Krea2 in addition to
H3. The Krea2 path uses the pinned upstream Krea2 scripts and models:

``` text
Qwen3-VL caption missing/empty sidecars
  -> krea2_cache_latents.py
  -> krea2_cache_text.py
  -> krea2_train.py
  -> /data/loras/<output_name>.safetensors
```

The supported presets are `krea2_defaults` and `krea2_ultra_fast`. Initial
auto-captioning and Fizgig's mid-run Qwen3-VL auto-recaptioning default to
enabled. Existing non-empty captions are preserved, so the first Krea2 run does
not require manual captions. The public request contains only training intent;
execution settings such as seed, checkpoint cadence, and caption behavior are
resolved by the preset. Deployment, Krea2 model availability, and an end-to-end
30-epoch `linda` training run have been validated on Modal.

## Development architecture

The first working version should look like:

``` text
YOU
 │
 ▼
CODEX
 │
 ▼
Project-local Fizgig/WanGP SKILL.md
 │
 ▼
rest-wangpt-modal-app (shared authenticated REST API)
 │
 ├── training routes ────▶ fizgig-modal-app ──▶ Fizgig CLI ──▶ training GPU
 │
 └── generation routes
       ├── image/audio ──▶ WanGP image worker ─▶ L40S
       └── video ────────▶ WanGP video worker ─▶ H100
```

The skill is the local orchestration specification. It should describe
which endpoint to call, how to build and validate requests, how to poll
asynchronous jobs, how to handle artifacts, and when user approval is
required. It must call stable APIs; it must not contain Fizgig training
logic or duplicate WanGP behavior.

## Target Eve architecture

After the local workflow is reliable, replace Codex as the agent host
with Eve while keeping the service contracts and agent capabilities the
same:

``` text
YOU
 │
 ▼
EVE AGENT FRAMEWORK
 │
 ▼
Fizgig/WanGP agent capability
 │
 ▼
rest-wangpt-modal-app
 │
 ├── fizgig-modal-app ──▶ Fizgig CLI training
 └── WanGP generation and LoRA testing
```

The local Codex skill is therefore a development harness and the
reference behavior for the later Eve integration. It should not depend
on Codex-only behavior that cannot be expressed in Eve.

## Core architectural decisions

Do **not** automate the Fizgig GUI.

Treat Fizgig as a headless CLI training engine. Fizgig explicitly
documents that the GUI is a frontend over scripts in
`src/fizgig/scripts/` and that the CLI is intended to be
feature-complete. Build a thin Modal job layer over that supported CLI
surface and let an agent orchestrate it through structured API calls.

The integration preference order is:

1.  use documented CLI commands and dataset configuration;
2.  call documented scripts from `src/fizgig/scripts/` when the CLI
    guide requires it;
3.  import public Python APIs only when there is no suitable CLI path;
4.  touch private or GUI-coupled internals only as a last resort.

This is simpler and less fragile than wrapping the state assembled by
the GUI. Fizgig has already implemented much of the headless,
agent-friendly plumbing that the original plan expected us to build.

Do **not** build a second REST application for Fizgig. Reuse and extend
the existing `rest-wangpt-modal-app` directory and `modal.App`, which
already provide a protected FastAPI endpoint, asynchronous Modal
FunctionCall jobs, polling and cancellation, job metadata, path
validation, tests, and a shared Modal Volume.

Preserve the existing WanGP endpoints and behavior. Add namespaced
training routes that dispatch work to the separately deployed
`fizgig-modal-app`. This keeps Fizgig's dependencies, source pin, GPU
selection, scaling, timeouts, and long-running training lifecycle out of
the WanGP image and worker runtime.

Both job domains should follow the same general boundary:

``` text
Codex now / Eve later
  │
  ▼
Agent skill
  │
  ▼
Small stable API
  │
  ▼
Existing upstream implementation
  │
  ▼
GPU
```

Do not port or rewrite Fizgig's training implementation unless
absolutely necessary. Prefer invoking the pinned upstream CLI and
exposing a stable API interface around it.

## New `fizgig-modal-app`

Create a new top-level project folder named `fizgig-modal-app`, using
`my-wangpt-modal-app` as the local reference for how this repository
packages and deploys a standalone Modal GPU application.

Initial structure:

``` text
fizgig-modal-app/
    app.py
    pyproject.toml
    uv.lock
    README.md
    tests/
```

Use `APP_NAME = "fizgig-modal-app"`. The app should contain:

-   a pinned Fizgig source revision installed into a purpose-built
    Modal image;
-   a named Modal Volume mounted at `/data`;
-   the Hugging Face secret and cache environment needed by Fizgig;
-   a dedicated training GPU type, concurrency limit, timeout, and
    scale-down policy;
-   callable Modal functions or a worker class for preparing data,
    starting training, reporting progress, pausing/resuming, and
    finalizing artifacts;
-   deployment instructions using `python3 -m modal deploy app.py`;
-   unit tests for configuration, path validation, request translation,
    and artifact metadata.

Reuse the useful infrastructure patterns from `my-wangpt-modal-app`,
including image construction, pinned upstream source, named Volumes,
named secrets, cache directories, environment-based GPU configuration,
and deployment documentation. Do not copy its WanGP dependencies, MCP
server, `sitecustomize.py` transport patch, or public web-server process;
those are specific to that app.

`fizgig-modal-app` is the internal execution plane, not a second public
REST interface. `rest-wangpt-modal-app` remains the authenticated public
API and dispatches training calls to the deployed Fizgig app by its
stable Modal app/function name. Phase 0 should verify the exact Modal
cross-app lookup and spawn mechanism before fixing that internal
contract.

## Preferred headless pipeline

The production target is:

``` text
Eve agent framework
 │
 ▼
Fizgig skill
 │
 ▼
rest-wangpt-modal-app
 │
 ▼
fizgig-modal-app job
 │
 ├── prepare dataset
 ├── AI auto-caption with the family-supported captioner
 ├── cache dataset
 ├── Fizgig CLI training
 │     ├── adaptive LR and per-image loss monitoring
 │     ├── Qwen3-VL auto-recaption of stuck images where supported
 │     ├── Context LoRA where supported
 │     └── pause/resume with persisted training state
 ├── periodic checkpoints and previews
 └── final LoRA
```

During local development, Codex replaces Eve at the top of this diagram
and follows the same steps through the project-local skill.

### Upstream capabilities to reuse

The upstream Fizgig documentation currently describes:

-   headless operation on Linux and display-less machines;
-   a CLI pipeline and dataset configuration format;
-   feature parity between the trainer and CLI, including adaptive LR,
    per-image loss monitoring, auto-recaptioning, Context LoRA, and
    pause/resume;
-   dataset preparation and automatic captioning;
-   full-state resume at epoch boundaries;
-   Qwen3-VL recaptioning of persistently problematic images during
    Krea 2 training.

These are capabilities to expose and orchestrate, not reimplement.

Capability support is model-family-specific. In particular, the
Qwen3-VL problem-image recaptioning workflow is currently documented as
experimental for Krea 2, and Context LoRA is not currently wired for H3.
Phase 0 must verify the exact CLI support for the selected family at the
pinned Fizgig commit before defining a preset or promising a feature.

Sources:

-   [Fizgig README](https://github.com/shootthesound/Fizgig#headless--cli-training)
-   [Fizgig CLI guide](https://github.com/shootthesound/Fizgig/blob/main/docs/CLI.md)

## Why Modal fits

Modal can provide ephemeral GPU workers while persistent data lives
outside the worker lifecycle.

Useful persistent data includes:

-   model checkpoints
-   downloaded base models
-   dataset caches
-   training state
-   optimizer state
-   intermediate LoRA checkpoints
-   previews
-   metrics
-   final `.safetensors`

The GPU worker can therefore scale to zero after training without losing
important state.

Large models should be downloaded once into persistent storage rather
than on every cold start.

## Suggested persistent filesystem layout

``` text
/data
    /fizgig
        /models
            /h3
            /krea2
            /klein

        /datasets
            /anna
            /weather_girl_01
            /weather_girl_02

        /cache
            /anna_h3

        /runs
            /anna_h3_v1
                epoch_001.safetensors
                epoch_002.safetensors
                ...
                training_state.json
                loss.json
                previews/

    /loras
        /characters/anna/h3/v1/anna_h3_v1.safetensors
```

## First-version storage decision

The first version will use **Modal Volumes only** for persistent storage.
Base models, datasets, caches, run state, checkpoints, previews, and
final LoRAs must all remain on Modal Volumes. S3/object-storage
integration is explicitly out of scope for the first version.

For the first version, reuse the existing `wangp-data` Modal Volume from
`rest-wangpt-modal-app` and its `/data` mount. The new
`fizgig-modal-app` must mount that same named Volume read/write, as does
the WanGP worker. Training-specific state should live under
`/data/fizgig`, while promoted final LoRAs should be placed under the
existing `/data/loras` tree so WanGP can load them without a copy or an
S3 handoff. Preserve the existing path validation and Modal Volume
commit/reload behavior across both apps.

S3 can be evaluated later for canonical datasets, distribution, or
long-term archival after the Volume-based workflow is proven.

## Model acquisition

Fizgig provides model-fetching functionality such as:

``` bash
python -m fizgig.scripts.fetch_models --family krea2
python -m fizgig.scripts.fetch_models --family klein
python -m fizgig.scripts.fetch_models --family tools
```

Use this where possible rather than creating a second independent
model-download implementation.

Models should be cached persistently.

## Modal GPU strategy

Fizgig is designed to operate under consumer-GPU constraints as well as
larger GPUs.

On Modal, prefer a straightforward configuration appropriate to the
selected GPU rather than automatically applying aggressive low-VRAM
optimizations.

For H3, a larger Modal GPU such as A100/H100 gives room to avoid
unnecessary block swapping and similar memory-saving measures where
possible.

The objective is:

1.  maximize training quality/reliability;
2.  keep implementation simple;
3.  optimize cost only where it does not materially hurt training.

GPU selection should eventually be configurable per job.

Example:

``` json
{
  "gpu": "A100",
  "family": "minimax_h3"
}
```

## CLI-backed Fizgig adapter

Fizgig already exposes the important training behavior through its CLI;
it does not need to provide its own REST server for this architecture.

Create a small REST/job adapter that validates a high-level request,
writes the expected Fizgig dataset/training configuration, invokes a
fixed Fizgig CLI entry point, and translates CLI outputs into structured
job status and artifacts.

Split this adapter across the two apps at a narrow boundary:

-   `rest-wangpt-modal-app` owns HTTP validation, authentication,
    public job records, REST responses, and the training routes;
-   `fizgig-modal-app` owns Fizgig request-to-CLI translation, the
    Fizgig image and worker, GPU execution, progress production, and
    artifacts.

Reuse the REST app's asynchronous submit/poll/cancel semantics and test
approach. The REST app should call a stable, allowlisted function in the
deployed `fizgig-modal-app`; it must not accept or forward an arbitrary
Modal function name.

The adapter must not accept an arbitrary command or arbitrary CLI
arguments from the agent. It should map validated presets and explicit
overrides onto a pinned, known Fizgig command.

Initial operations should be roughly:

``` text
prepare_dataset()
caption_dataset()
start_training()
get_training_status()
pause_training()
resume_training()
get_preview()
promote_checkpoint()
```

Potential later operations:

``` text
cancel_training()
list_checkpoints()
evaluate_checkpoint()
delete_run()
download_model()
validate_dataset()
recaption_image()
exclude_image()
```

The public API should remain stable even if Fizgig's CLI arguments or
internal implementation change.

## Example training request

The agent client—Codex during development and Eve later—should send
structured requests rather than trying to manipulate UI controls.

Conceptual request:

``` json
{
  "family": "minimax_h3",
  "dataset": "anna-v3",
  "trigger_word": "annaxs",
  "preset": "minimax_h3_fast",
  "output_name": "anna_h3_v1",
  "samples": [
    "annaxs woman, close-up portrait",
    "annaxs woman walking outdoors"
  ]
}
```

The adapter translates this high-level configuration into Fizgig's
documented dataset configuration and CLI arguments.

Do not expose every Fizgig parameter to the agent initially.

Prefer named presets plus optional overrides.

Example:

``` json
{
  "preset": "h3_character_quality",
  "overrides": {
    "epochs": 40
  }
}
```

## Conversation as the UI

The main user interface should be conversation.

The examples below use Eve to show the intended end state. During local
development, Codex should be able to perform the same workflow through
the project skill before any Eve integration is added.

Example:

``` text
User:
Train a fast H3 character LoRA for Emma from dataset emma-v2.
Use the normal likeness preset and give me a preview every 5 epochs.

Agent (Codex now / Eve later):
Starting H3 training.

Dataset: emma-v2
Images: 27
Preset: h3_character_fast
Rank: 8
Epochs: 40
Preview interval: 5 epochs
```

During training the agent should be able to query status:

``` text
Epoch 15 completed.

Training loss is still improving.
Latest likeness score: 0.73
Previous likeness score: 0.69

Two new previews are available.
```

When training plateaus:

``` text
Training appears to have plateaued around epochs 31–34.

Epoch 32 currently has the strongest evaluation result.

Would you like to keep epoch 32 as the final LoRA?
```

Then:

``` text
User:
Yes.

Agent (Codex now / Eve later):
Promoted checkpoint:

anna_h3_v1.safetensors

Saved on Modal Volume:
/loras/characters/anna/h3/v1/anna_h3_v1.safetensors
```

## Training intelligence

One advantage of an agent-controlled workflow is that the agent can
interpret training telemetry instead of requiring the user to watch
graphs.

Where Fizgig exposes or supports relevant information, surface things
such as:

-   per-image loss
-   global training loss
-   learning-rate changes
-   problem-image detection
-   plateau detection
-   likeness/evaluation scores
-   checkpoint comparison
-   generated previews
-   caption quality
-   potentially bad training images

The skill should tell Codex—and later Eve—how to summarize these signals
in human language.

Do not let either agent autonomously make destructive dataset changes
initially.

For example:

``` text
Agent (Codex now / Eve later):
Image 17 is consistently producing unusually high loss and may be hurting the dataset.

Exclude it from training?
```

The user can approve.

## Dataset workflow

Eventually support a high-level workflow such as:

``` text
User:
These 35 images are our new weather girl.
Clean the dataset, caption it, train a Krea2 LoRA,
pick the strongest checkpoint and test it on five prompts.
```

Desired orchestration:

``` text
input images
    │
    ▼
dataset validation
    │
    ├── duplicates
    ├── bad images
    ├── resolution checks
    └── possible outliers
    │
    ▼
captioning
    │
    ▼
user review if needed
    │
    ▼
Fizgig training
    │
    ▼
periodic checkpoints
    │
    ▼
preview generation
    │
    ▼
checkpoint evaluation
    │
    ▼
best checkpoint
    │
    ▼
final .safetensors
    │
    ▼
Modal Volume + model registry metadata
```

## Storage responsibilities

### Modal Volumes — first version

Modal Volumes are the source of truth for all first-version files:

-   downloaded base models
-   prepared and source datasets
-   training caches
-   current run state
-   checkpoints while training
-   resumable state
-   previews
-   final LoRAs

Store paths in job metadata so Codex—and later Eve—can refer to
artifacts without guessing filesystem locations.

### S3/object storage — later, not version 1

Do not implement or require S3 in the first version. A later production
phase may use it for:

-   canonical datasets
-   final LoRAs
-   important checkpoints
-   preview images
-   long-term artifacts

### Postgres

Good for metadata/state, not large binaries.

Potential schema concepts:

``` text
training_jobs
training_runs
datasets
dataset_images
loras
lora_versions
checkpoints
evaluations
```

Example LoRA metadata:

``` json
{
  "name": "anna",
  "version": "v1",
  "family": "minimax_h3",
  "trigger_word": "annaxs",
  "dataset_id": "anna-v3",
  "training_job_id": "...",
  "checkpoint_epoch": 32,
  "volume": "wangp-data",
  "artifact_path": "/loras/characters/anna/h3/v1/anna_h3_v1.safetensors"
}
```

## Relationship to `rest-wangpt-modal-app`

`rest-wangpt-modal-app` already exists in this project and will be the
REST interface for the combined workflow. Extend it rather than creating
a second public endpoint. The new `fizgig-modal-app` sits behind this
REST interface as a separately deployed execution app.

The agent skill should use one authenticated REST application with two
namespaced job domains:

``` text
                    CODEX NOW / EVE LATER
                              │
                        Agent skill
                              │
                              ▼
                  rest-wangpt-modal-app
                  shared authenticated API
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
        /training/jobs                  /jobs
                 │                         │
                 ▼              ┌──────────┴──────────┐
        fizgig-modal-app         │                     │
          Fizgig worker          ▼                     ▼
          training GPU      image/audio worker    video worker
                 │               L40S                 H100
                 │                 │                     │
                 └─────────────────┴──────────┬──────────┘
                              ▼
                  wangp-data Modal Volume
                    S3/Postgres only later
```

The responsibilities are intentionally separate:

-   Fizgig handles training, checkpoints, resume state, and training
    telemetry.
-   WanGP handles generation and tests trained LoRAs through separate L40S
    image/audio and H100 video worker pools.
-   `rest-wangpt-modal-app` owns the shared REST transport,
    authentication, routing, job submission/polling/cancellation, and
    public job records.
-   `fizgig-modal-app` owns the Fizgig runtime, training GPU work,
    progress production, and training artifacts.
-   Both Modal apps mount the same `wangp-data` Volume for first-version
    artifact handoff.
-   The skill handles orchestration and conversational interpretation.
-   Codex hosts the skill during local development.
-   Eve hosts the equivalent capability after the API and workflow have
    been proven locally.

This enables workflows such as:

``` text
Train a Krea2 LoRA for Anna.
When training finishes, test the best three checkpoints
with the same five prompts through rest-wangpt-modal-app.
Show me the comparison.
```

This cross-worker workflow belongs in the agent skill. The shared REST
app coordinates transport and job state, while the shared Volume
coordinates files. The Fizgig and WanGP execution apps should not
directly own each other's runtime responsibilities.

## Local Codex skill

Create a project-local skill, expected to live at
`.agents/skills/fizgig/SKILL.md`, after the initial API contract is
defined. Keep the main `SKILL.md` concise and place large API examples or
schemas in directly linked reference files if needed.

The skill should contain:

-   when to use the Fizgig training routes versus the WanGP generation
    routes in `rest-wangpt-modal-app`;
-   required environment variables, endpoint URLs, and authentication
    headers without embedding credentials;
-   request templates for training and generation;
-   supported presets and safe override rules;
-   job submission, polling, timeout, failure, and resume behavior;
-   artifact handoff from a Fizgig checkpoint to a WanGP LoRA test;
-   operations that require explicit user confirmation;
-   concise status and result formats for conversational use.

The skill is not the Eve integration itself. It is the executable local
specification used to validate the capabilities Eve will receive later.
When Eve integration begins, preserve the API payloads, state model,
safety rules, and user-facing semantics rather than redesigning them
inside Eve.

## API/job design

Training is asynchronous.

Do not keep a Codex or Eve request open for the entire GPU training run.

Extend the FastAPI application already returned by
`rest-wangpt-modal-app/api()` with namespaced training routes. Preserve
the existing generation routes such as `POST /jobs`, `GET /jobs/{id}`,
and `POST /jobs/{id}/cancel`.

The route handlers should resolve the deployed `fizgig-modal-app` by a
fixed Modal app/function name, spawn the appropriate training operation,
and retain its call ID in the REST job record. Polling, cancellation,
and infrastructure-error handling should remain consistent with the
existing WanGP job API even though execution occurs in a different
Modal app.

Use a job model:

``` text
POST /training/jobs
        │
        ▼
returns job_id
        │
        ▼
Modal GPU training
```

Then:

``` text
GET /training/jobs/{job_id}
```

Possible response:

``` json
{
  "id": "abc123",
  "status": "running",
  "progress": {
    "phase": "training",
    "epoch": 15,
    "epochs_total": 40,
    "loss": 0.082,
    "latest_checkpoint": "epoch_015.safetensors",
    "latest_preview": "..."
  }
}
```

Reuse the app's existing top-level job states:

``` text
queued
running
succeeded
failed
cancelled
```

Represent training detail inside `progress.phase`, for example:

``` text
preparing_dataset
captioning
caching
starting
training
paused
evaluating
finalizing
```

Also add training-specific pause, resume, and cancel routes using the
same authentication, validation, job lookup, and error conventions as
the existing WanGP routes. A separate job-store namespace or explicit
job type must prevent generation and training records from colliding.

This maps cleanly to both the local Codex skill and the later Eve
integration.

## Progress/events

Prefer structured events instead of parsing logs.

Example:

``` json
{
  "event": "epoch_completed",
  "epoch": 15,
  "loss": 0.082,
  "checkpoint": "epoch_015.safetensors"
}
```

Other useful events:

``` text
dataset_prepared
cache_complete
training_started
epoch_started
epoch_completed
checkpoint_saved
preview_generated
plateau_detected
training_completed
upload_completed
training_failed
```

Logs should still be retained for debugging.

## Resume behavior

Training should survive agent disconnects and ideally worker restarts
where Fizgig supports restoration.

Persist:

-   training configuration
-   current epoch/step
-   optimizer state
-   scheduler state
-   relevant Fizgig state
-   checkpoint
-   dataset/cache identity

The user should be able to say to Codex locally, and later to Eve:

``` text
Resume Anna H3 training.
```

The API should resolve the latest resumable run and restart the Modal
worker using the persisted state.

## Presets

The agent should operate mostly through presets.

Examples:

``` text
h3_character_fast
h3_character_quality
krea2_defaults
krea2_ultra_fast
klein_character
style_lora
```

Each preset maps to tested Fizgig settings.

This prevents the agent from inventing arbitrary hyperparameters.

Advanced overrides can still be supported.

## Security

Protect the Modal API.

Do not expose an unauthenticated training endpoint.

At minimum:

-   API authentication
-   validate dataset paths
-   restrict output paths
-   restrict model families
-   validate numeric training parameters
-   cap GPU/job duration
-   prevent arbitrary shell execution
-   prevent arbitrary filesystem paths

Codex and Eve should invoke defined operations, not send executable
commands.

## Implementation principles

Before writing the adapter, pin a Fizgig commit and inspect its README,
`docs/CLI.md`, and relevant scripts to identify:

1.  the supported CLI entry point for each selected model family;
2.  dataset preparation, captioning, caching, and training commands;
3.  the dataset and training configuration formats;
4.  checkpoint save, pause, and resume behavior;
5.  progress output, loss logs, exit codes, and generated artifacts;
6.  preview generation and model-family limitations;
7.  model downloading/loading and persistent cache requirements;
8.  cancellation, failure, and cleanup behavior.

Then build the thinnest possible REST/job adapter around those commands.

Do **not** duplicate Fizgig's training code.

Prefer a fixed executable plus validated argument array, conceptually:

``` python
command = [python, "-m", pinned_fizgig_command, "--config", config_path]
run_without_shell(command)
```

The exact command must come from the pinned Fizgig CLI guide. Do not use
`shell=True`, accept a raw command string, or let the agent supply
unvalidated flags. Parse stable files or structured output where
available; retain raw logs for debugging.

Only introduce a Python compatibility shim if a required capability is
missing from the documented CLI. Keep such a shim narrow and avoid GUI
state or private internals wherever possible.

Keep the agent boundary framework-agnostic: Codex and Eve should receive
the same conceptual operations and payloads. Any Codex-specific command
details belong in the local skill packaging, not in the shared REST
API's Fizgig routes.

## Phase 0 — define the contracts

Before implementation:

-   pin the Fizgig commit and verify its CLI entry points and examples;
-   build a capability matrix for H3, Krea 2, and Klein covering
    captioning, adaptive LR, per-image loss monitoring, auto-recaption,
    Context LoRA, previews, and pause/resume;
-   inspect the existing `rest-wangpt-modal-app` API, job lifecycle,
    tests, authentication, path validation, and worker structure;
-   inspect `my-wangpt-modal-app` for reusable Modal image, Volume,
    secret, GPU configuration, deployment, and documentation patterns;
-   define the minimum Fizgig job contract as namespaced additions to
    that existing REST API;
-   define the fixed cross-app call contract between
    `rest-wangpt-modal-app` and `fizgig-modal-app`;
-   scaffold `fizgig-modal-app` with its own image, worker/functions,
    tests, dependencies, and deployment lifecycle without changing the
    WanGP worker runtime;
-   confirm the first-version layout on the existing `wangp-data` Volume
    and define the `/data/loras` artifact path passed to WanGP;
-   outline the local `SKILL.md` around those contracts.

Output: the `fizgig-modal-app` scaffold, stable request/response
examples, a verified CLI command map, the family capability matrix, the
cross-app call contract, and a clear ownership boundary.

## Phase 1 — local training through Codex (implemented; live validation next)

Keep the first implementation deliberately small.

Target:

``` text
User
 │
 ▼
Codex + local SKILL.md
 │
 ▼
start_training()
 │
 ▼
rest-wangpt-modal-app /training/jobs
 │
 ▼
fizgig-modal-app
 │
 ▼
Fizgig CLI
 │
 ▼
LoRA
```

Phase 1 requirements:

-   H3 and Krea2 model-family pipelines
-   dataset images already staged in the shared Volume
-   predefined training preset
-   documented Fizgig CLI entry point from the pinned commit
-   new top-level `fizgig-modal-app` project modeled on relevant patterns
    from `my-wangpt-modal-app`
-   `APP_NAME = "fizgig-modal-app"` and independent Modal deployment
-   training routes implemented in the existing
    `rest-wangpt-modal-app`
-   fixed cross-app dispatch from the REST routes to the Fizgig app
-   existing Modal proxy authentication and REST/job patterns reused
-   separate Fizgig image, worker/function, dependencies, and GPU
    configuration
-   start job
-   query status
-   checkpoint periodically
-   pause/resume when supported by the selected family and CLI command
-   final `.safetensors`
-   existing `wangp-data` Modal Volume for models, datasets, caches, run
    state, checkpoints, previews, and final artifacts
-   no S3 dependency or S3 integration
-   project-local skill that lets Codex submit and monitor the job

Avoid building automated dataset cleaning, automatic checkpoint
selection, and complex evaluation before basic training works reliably.

Acceptance criterion: Codex can start a training run, report status, and
return the final artifact without Eve being present.

## Phase 2 — local training-to-generation workflow

Add:

-   dataset upload/import into a Modal Volume
-   Krea2 Qwen3-VL initial captioning and mid-run auto-recaptioning
-   validation
-   preview generation
-   pause/resume
-   richer telemetry
-   Postgres job registry
-   checkpoint/LoRA handoff between the Fizgig and WanGP worker domains
    through `/data/loras`
-   preview or test generation through its existing REST API
-   supported adaptive LR and per-image loss monitoring surfaced from
    Fizgig instead of reimplemented in the adapter

Acceptance criterion: Codex can train a LoRA, submit one or more WanGP
test jobs through `rest-wangpt-modal-app`, poll both job domains through
the same REST endpoint, and present the results as one workflow using
the shared Modal Volume for all file storage.

## Phase 3 — agentic evaluation in Codex

Add agentic evaluation:

-   compare checkpoints
-   likeness scoring
-   detect plateau
-   identify problematic images
-   surface Fizgig's problem-image detection and recaptioning decisions
-   use Qwen3-VL auto-recaptioning for Krea 2 when enabled, while
    requiring review before exclusions or other destructive changes
-   suggest exclusions/manual recaptioning for families without the
    upstream automatic workflow
-   generate test images through `rest-wangpt-modal-app`
-   promote best checkpoint
-   register LoRA automatically

Acceptance criterion: the local skill and service contracts cover the
complete intended workflow and have been exercised without relying on
Eve-specific code.

## Phase 4 — Eve integration

Only after the local Codex workflow is stable:

-   expose the proven operations to the Eve agent framework;
-   port or adapt the local skill instructions to Eve's skill/tool
    format;
-   configure Eve to call the training and generation routes exposed by
    the extended `rest-wangpt-modal-app`;
-   preserve authentication, validation, confirmation, polling, and
    error-handling behavior from the local skill;
-   run parity tests showing that Codex and Eve produce equivalent API
    requests for representative user prompts.

Do not block Phases 0–3 on Eve framework setup.

## End-state experience

The user should eventually be able to say:

``` text
Train a new H3 LoRA for Anna using anna-v4.

Use the quality character preset.
Generate previews every five epochs.
Stop if the evaluation clearly plateaus.
Then test the best three checkpoints through rest-wangpt-modal-app and
show me the results.
```

Codex handles orchestration during development; Eve handles it after the
deferred integration phase.

`fizgig-modal-app` and its Fizgig CLI worker handle training execution.

WanGP handles generation/testing.

The extended `rest-wangpt-modal-app` exposes the shared authenticated
REST interface for both job domains and dispatches training work to
`fizgig-modal-app`.

Modal supplies ephemeral GPU compute.

Modal Volumes hold every first-version file and artifact.

S3 is an optional later archival/distribution layer, not a first-version
dependency.

Postgres holds state and metadata.

The Fizgig UI remains optional throughout.
