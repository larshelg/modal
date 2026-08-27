---
name: fizgig
description: Operates Fizgig LoRA training and WanGP generation through the project's authenticated Modal REST API, with bounded Modal CLI fallbacks for direct development, logs, Volume inspection, and checkpoint promotion. Use when starting, inspecting, pausing, resuming, or cancelling training; generating with WanGP; managing Fizgig artifacts; or discussing the Fizgig training workflow.
---

# Fizgig workflows

Use the shared `rest-wangpt-modal-app` API for normal job lifecycle operations.
The `fizgig-modal-app` worker internally runs Fizgig's pinned headless CLI;
never invoke those upstream training scripts locally or accept raw CLI arguments
from a user request.

## Client

Run the bundled client from the repository root:

```text
python3 .agents/skills/fizgig/scripts/fizgig_api.py <operation> [arguments]
```

It reads:

- `FIZGIG_REST_URL`, falling back to `WANGP_URL`;
- `MODAL_KEY`;
- `MODAL_SECRET`.

Never print, persist, or place those credential values in command
arguments. If configuration is missing, tell the user which environment
variable to set without asking them to paste its value into chat.

Run `health` before the first operation in a session:

```text
python3 .agents/skills/fizgig/scripts/fizgig_api.py health
```

## Training workflow

Training supports **Krea2** and **MiniMax H3**. Always submit the family
the user selected; never translate between them.

Before submission, confirm:

- the dataset name under `/data/fizgig/datasets/<dataset>/images`;
- a unique output name;
- a family-compatible preset;
- an optional trigger word and epoch override.

For Krea2 identity training, use `krea2_defaults` unless the user asks
for the faster `krea2_ultra_fast` preset. Missing captions are generated
before caching with Fizgig's Qwen3-VL training-caption task. Existing
non-empty captions are preserved. Mid-run `auto_recaption` is enabled by
default to repair confirmed-stuck images.

Submit with typed flags:

```text
python3 .agents/skills/fizgig/scripts/fizgig_api.py training-submit \
  --family krea2 \
  --dataset linda \
  --output-name linda_krea2_v1 \
  --preset krea2_defaults \
  --trigger-word linda
```

Report the returned job ID and queued status. Do not keep the current
request open for the duration of GPU training.

Poll once when the user asks for status:

```text
python3 .agents/skills/fizgig/scripts/fizgig_api.py training-status JOB_ID
```

Summarize status, `progress.phase`, epoch progress, and the terminal
result or error. Log tails are diagnostic; do not dump them unless they
explain a failure or the user requests them.

Pause is cooperative:

```text
python3 .agents/skills/fizgig/scripts/fizgig_api.py training-pause JOB_ID
```

After requesting pause, status remains `running` until Fizgig saves
state. A completed pause is represented by `status: succeeded`,
`progress.phase: paused`, and `result.paused: true`.

Resume creates a new job ID:

```text
python3 .agents/skills/fizgig/scripts/fizgig_api.py training-resume JOB_ID
```

Use the new ID for all later operations. Cancel only queued or running
jobs:

```text
python3 .agents/skills/fizgig/scripts/fizgig_api.py training-cancel JOB_ID
```

## Generation workflow

Discover models when the requested WanGP model is uncertain:

```text
python3 .agents/skills/fizgig/scripts/fizgig_api.py models
```

Write native WanGP parameters to a JSON file, then submit them with a
model selected from discovery:

```text
python3 .agents/skills/fizgig/scripts/fizgig_api.py generation-submit \
  --model krea2_turbo \
  --kind image \
  --params request-params.json
```

`--kind` accepts `image`, `video`, or `audio`. It is optional because the API
infers the output kind from WanGP's model catalog, but include it when the user
has explicitly selected a modality. The API rejects a kind that the selected
model cannot produce. Video routes to the H100 worker; image and audio route to
the standard L40S worker.

Poll or cancel with:

```text
python3 .agents/skills/fizgig/scripts/fizgig_api.py generation-status JOB_ID
python3 .agents/skills/fizgig/scripts/fizgig_api.py generation-cancel JOB_ID
```

Treat `params` as native WanGP settings. Do not add `_api`; it is
reserved. Absolute asset and LoRA paths must remain under `/data`.

## Raw Modal CLI fallback

Read [REFERENCE.md](REFERENCE.md#raw-modal-cli-fallback) before using raw Modal
commands. Use this fallback only when the user explicitly requests the direct
development flow or the REST API lacks the required read/artifact operation.

Allowed fallback operations are:

- submit an allowlisted request JSON through the worker's detached local
  entrypoint during direct development;
- read recent `fizgig-modal-app` logs for diagnostics;
- list run or LoRA files on the `wangp-data` Volume;
- copy a selected numbered checkpoint into `/data/loras` with an epoch postfix.

These commands are not a second training API. Do not call named worker functions
directly, run `krea2_train.py` or `minimax_train.py` yourself, or mutate/delete
artifacts outside an explicit user request. Treat logs as diagnostic evidence;
the REST job record remains authoritative for terminal status.

## Safety and lifecycle rules

- Use the named REST client operations by default; use only the bounded Modal CLI
  fallbacks documented above when their preconditions apply.
- Do not silently change model families, presets, datasets, or output
  names.
- Require confirmation before cancellation or any later destructive
  dataset/checkpoint operation.
- Preserve job IDs in status reports.
- Do not claim success until the API reports a terminal successful
  record.
- A failed HTTP operation is not a queued job; report its structured
  error.
- Keep Eve integration out of this workflow until the Codex flow is
  proven.

For exact request fields, states, and route mappings, read
[REFERENCE.md](REFERENCE.md).
