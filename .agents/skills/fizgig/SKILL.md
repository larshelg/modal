---
name: fizgig
description: Operates Fizgig LoRA training and WanGP generation through their local Modal clients and deployed workers. Use when starting, inspecting, pausing, resuming, or cancelling training; generating with WanGP; managing Fizgig artifacts; or discussing the Fizgig training workflow.
---

# Fizgig workflows

Use the local Modal client in `fizgig-modal-app/control.py` for training and
`wangpt-modal-app/control.py` for generation. Both call stable deployed workers
directly with Modal SDK authentication. The REST app and the legacy
`scripts/fizgig_api.py` client are not required for these workflows.

Commands below run from the repository root (`modal2`) with `modal` installed
in the chosen Python environment. Use the app's `.venv/bin/python` when needed.
Never print or persist credential values. If authentication is missing, report
that the Modal SDK profile needs configuring; do not ask for secrets in chat.

## Training

Read deployment health before the first remote training operation:

```bash
python3 -m modal run fizgig-modal-app/control.py::health
```

Training supports **Krea2** and **MiniMax H3**. Preserve the user's selected
family. Establish the dataset, unique output name, family-compatible preset,
and any trigger/epoch override before submitting; infer established choices
from the conversation rather than requesting redundant confirmation.
Datasets must exist at `/data/fizgig/datasets/<dataset>/images`.

For Krea2 identity training, use `krea2_defaults` unless the user asks for
`krea2_ultra_fast`. Missing captions are generated with Qwen3-VL; existing
non-empty captions are preserved. Mid-run auto-recaptioning repairs stuck
images. H3 needs prepared captions and uses `h3_character_fast` or
`h3_character_quality`.

```bash
python3 -m modal run fizgig-modal-app/control.py::submit \
  --family krea2 --dataset linda --output-name linda_krea2_v1 \
  --preset krea2_defaults --trigger-word linda
python3 -m modal run fizgig-modal-app/control.py::status --job-id JOB_ID
python3 -m modal run fizgig-modal-app/control.py::pause --job-id JOB_ID
python3 -m modal run fizgig-modal-app/control.py::resume --job-id JOB_ID
python3 -m modal run fizgig-modal-app/control.py::cancel --job-id JOB_ID
```

Optional `--epochs` overrides the preset. Submission returns immediately and
needs no `--detach`. Report the job ID and queued acknowledgment; do not hold
the request open for training. On a status request, poll once and summarize
status, phase, epochs, and result/error. Add `--logs` only for diagnostics.

Pause remains `running` until the next epoch boundary saves state. A completed
pause is `status: succeeded`, `progress.phase: paused`, `result.paused: true`.
Resume creates a new job ID; use that ID thereafter. Cancel only queued/running
jobs when cancellation has been explicitly authorized. Require explicit
confirmation before destructive dataset/checkpoint operations or overwrites.
Never report success from logs alone or treat a failed dispatch as a queued job.

Use only the typed client request. Do not run upstream training scripts locally
or pass raw Fizgig CLI arguments. The worker constructs its pinned headless
pipeline. Keep Eve integration out of this workflow until the Codex flow is proven.

## WanGP generation

Use `wangpt-modal-app/control.py`; discover models when the selection is unclear:

```bash
python3 -m modal run wangpt-modal-app/control.py::models
python3 -m modal run wangpt-modal-app/control.py::defaults --model krea2_turbo
python3 -m modal run wangpt-modal-app/control.py::submit \
  --model krea2_turbo --kind image --params-file request-params.json
python3 -m modal run wangpt-modal-app/control.py::status --job-id JOB_ID
python3 -m modal run wangpt-modal-app/control.py::cancel --job-id JOB_ID
```

Use native WanGP parameters; `_api` is reserved, and absolute asset/LoRA paths
must stay under `/data`. `--kind` is optional (`image`, `video`, `audio`); include
it when the user selected a modality. Let the client route to its configured
worker; do not assume GPU types from old REST documentation.

For request details, model setup, artifact inspection/promotion, and historical
job limitations, read [REFERENCE.md](REFERENCE.md).
