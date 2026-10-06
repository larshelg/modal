# Qwen Noct Q V4 snapshot worker

Run all commands below from the `wangpt-modal-app` workspace root. Use Modal
module mode (`-m wangpt-qwen-noctq-modal-app.app`) for this package.

Dedicated app: `wangpt-qwen-noctq-modal-app`; class: `WanGPQwenNoctQWorker`.
Model: `qwen_image_21_noctq_v4`; GPU: L40S; profile: 1; host RAM: 65536 MiB.
Revision: `qwen-noctq-v4-l40s-transformer-v1`.

The independent `wangpt-qwen-noctq-modal-app/app.py` deployment shares the WanGP runtime, Volume,
job store and verified S3 output handling. It uses the existing Noct Q V4
INT8 ConvRot checkpoint pinned to revision
`a81b9af51120a78e285e57906f2250a2a02080e9` in
`finetunes/qwen_image_21_noctq_v4.json`. The shared image retains the tested
fused-checkpoint loader patch. Warmup and restored model diagnostics require
that pinned checkpoint URL, rejecting a base-model or unpinned definition.

Private warmup edits a synthetic reference at 1024x1024 with 25 steps, guidance
3, flow shift 5, `sample_solver: "default"`, `video_prompt_type: "KI"`, and
KV cache/RGBA disabled. It joins native task cleanup, releases input/output
payloads and clears session state/caches. Temporary warmup images are removed;
they never become jobs or S3 artifacts. Capture requires the full transformer
on CUDA and the text encoder, vision encoder and VAE on CPU, with no active
task LoRAs. Restore validates placement before reseeding and does not reload
weights to conceal a failed restore.

Both text generation and reference-image editing use
`run(job_id, model, params)`. CLI routing is opt-in for this exact image model:

```bash
.venv/bin/python -m modal deploy -m wangpt-qwen-noctq-modal-app.app
.venv/bin/python -m modal run control.py::qwen_noctq_snapshot_probe
WANGP_QWEN_NOCTQ_SNAPSHOT=1 .venv/bin/python -m modal run control.py::submit \
  --model qwen_image_21_noctq_v4 --kind image \
  --params-file wangpt-qwen-noctq-modal-app/examples/qwen_image_21_noctq_v4.json
.venv/bin/python -m modal run control.py::status --job-id JOB_ID
```

For an edit, add ordered `image_refs` using existing `/data` paths or S3 URIs
in the configured bucket, plus `video_prompt_type: "KI"` (main scene first) or
`"I"` (people/objects). Native WanGP performs reference handling.

Direct TypeScript clients can select the dedicated class without a CLI flag:

```ts
const cls = await modal.cls.fromName("wangpt-qwen-noctq-modal-app", "WanGPQwenNoctQWorker");
const worker = await cls.instance();
const call = await worker.method("run").spawn([jobId, "qwen_image_21_noctq_v4", params]);
```

Keep job-record creation and polling in the caller. Other models are rejected.
Different configs, profiles or upscalers can trigger normal reconfiguration;
snapshot reuse is scoped to the default finetune config.

Shared lifecycle and diagnostic helpers are in `snapshot_common.py` at the
workspace root and included in this worker image. Model-specific warmup and
residency checks are in this folder's `snapshot.py`.

## Validation

Local suite: 208 passed, 28 skipped; `git diff --check` passed. Tests cover exact
opt-in dispatch, component placement, restore validation, warmup cleanup,
profile/LoRA restrictions and the pinned V4 checkpoint definition.

Deployment, initial GPU snapshot restoration and generation succeeded on
2026-10-05:

- [Deployment](https://modal.com/apps/larshelg/main/deployed/wangpt-qwen-noctq-modal-app)
  completed in 19.918 seconds; image `im-u4U8g54Fgo3wzs1XdwefMp`.
- Actual GPU: NVIDIA L40S. Capture:
  `45ac3ed9-905e-4418-8b02-1ca65a4350c4`; restored boot:
  `22d47b5b-a779-4c07-9a60-e20b937a52e5`.
- Reference-image warmup: 118.388 seconds; initialization through capture-ready:
  205.755 seconds. Diagnostics confirm `NoctQ_V4_int8_convrot.safetensors` at
  the pinned revision, model `qwen_image_21_noctq_v4`, profile 1.
- Capture/restore CUDA allocation: 7,263,601,664 bytes (~6.76 GiB). All
  7,115,125,248 reported logical transformer elements were on CUDA; text
  encoder (7,568,406,136 elements), vision encoder (576,388,588) and VAE
  (337,740,404) were on CPU. Active MMGP models: exactly `["transformer"]`.
- [Test submission](https://modal.com/apps/larshelg/main/ap-GTlZmkzfrkjCjwzpaaK7kf):
  job `8bcccb08-b9e7-42a7-83b6-3978efaab1b0` used the existing Noct Q teapot
  example: 1024x1024, 25 steps, guidance 3, default solver and seed 42.
  Terminal status `succeeded`, one image, no errors; routed to
  `WanGPQwenNoctQWorker` in the dedicated app.
- Worker elapsed: 30.700 seconds, `load_models_calls=0`, `model_reused=true`,
  on the same restored boot. The runtime verified the S3 upload.
- Saved output metadata: JPEG, 384,504 bytes; SHA-256
  `5df527e41a62aaaf6d73d2098d2a7cd0b59d7707916b525063745cd28bb10037`.
  The output was not downloaded or visually inspected.

Saved evidence: [snapshot diagnostics](../runs/qwen-noctq-snapshot-probe.json)
and [job result](../runs/qwen-noctq-dedicated-smoke-result.json).

Initial capture/immediate restore is verified. Reuse after scale-to-zero has
not been measured; compare capture and boot IDs across containers for that
check. Masks, outpainting, RGBA and user LoRAs have not been exercised in this
snapshot validation.
