# Krea2 RAW Edit snapshot worker

Run commands from the `wangpt-modal-app` workspace root. Use Modal module mode
(`-m wangpt-krea-raw-edit-modal-app.app`) for this package.

Dedicated app: `wangpt-krea-raw-edit-modal-app`; class: `WanGPKreaRawEditWorker`.
Model: `krea2_raw_edit`; GPU: L40S; WanGP profile: 1; host RAM: 65536 MiB.
Revision: `krea2-raw-edit-l40s-transformer-v1`.

The worker follows the Turbo Edit snapshot lifecycle using the RAW Identity
Edit preset: 1024x1024, 20 steps, guidance 2 and flow shift 5. Private warmup
edits a synthetic reference with `video_prompt_type: "KI"` and verifies that
`krea2_identity_edit_v1_2.safetensors` is active during inference. The edit
defaults override the ordinary RAW model's 52-step/guidance-3.5 warmup values.

Warmup joins native task cleanup, releases input/output payloads, resets session
state/caches and removes temporary reference/output images before capture.
It never creates a job or uploads warmup media to S3. Native cleanup unloads
task LoRAs; each request activates the preset edit adapter and user LoRAs through
normal WanGP loading. Capture requires the full transformer on CUDA and the
text encoder, vision encoder and VAE on CPU, without active task adapters.
Restore validates placement before reseeding; it never reloads weights to hide
a failed restoration.

The worker shares `snapshot_common.py`, RAW base residency validation, the
WanGP runtime, Volume, job store and verified S3 output handling.

```bash
.venv/bin/python -m modal deploy -m wangpt-krea-raw-edit-modal-app.app
.venv/bin/python -m modal run control.py::krea_raw_edit_snapshot_probe
WANGP_KREA_RAW_EDIT_SNAPSHOT=1 .venv/bin/python -m modal run control.py::submit \
  --model krea2_raw_edit --kind image \
  --params-file wangpt-krea-raw-edit-modal-app/examples/krea2_raw_edit.json
.venv/bin/python -m modal run control.py::status --job-id JOB_ID
```

Replace the example's reference path with an existing `/data` image or S3 URI
in the configured bucket. The routing flag affects only `krea2_raw_edit` image
jobs. The worker retains `run(job_id, model, params)` and rejects other models
before accessing job records. Direct TypeScript clients need no CLI flag:

```ts
const cls = await modal.cls.fromName("wangpt-krea-raw-edit-modal-app", "WanGPKreaRawEditWorker");
const worker = await cls.instance();
const call = await worker.method("run").spawn([jobId, "krea2_raw_edit", params]);
```

Keep job-record creation and polling in the caller. Native reference ordering,
inpainting and outpainting settings remain available. Different configs,
profiles or upscalers can cause normal reconfiguration; reuse is scoped to the
default RAW Edit preset.

The worker scales to zero after 300 idle seconds, allowing at most one
container. Deployment overrides: `WANGP_KREA_RAW_EDIT_MEMORY_MB`,
`WANGP_KREA_RAW_EDIT_MAX_CONTAINERS` and `WANGP_KREA_RAW_EDIT_SCALEDOWN_WINDOW`.

## Validation

Local suite: 233 passed, 28 skipped; `git diff --check` passed. Tests cover exact
opt-in dispatch, RAW/Turbo Edit routing isolation, vision encoder placement,
restore validation, reference warmup, preset adapter observation and payload
cleanup on success/failure. Module imports and dependency source mounts resolve.

Deployment, initial GPU snapshot restoration and reference-image editing
succeeded on 2026-10-06:

- [Deployment](https://modal.com/apps/larshelg/main/deployed/wangpt-krea-raw-edit-modal-app)
  completed in 21.665 seconds; image `im-kAqd41X3LNK6RnJvSv4Nm0`.
- [Initial probe](https://modal.com/apps/larshelg/main/ap-PsGKN7SyEbLAGZ3XP2eSLM)
  returned capture `aed85612-52c6-4932-bc15-b66b1a99f391` and restored boot
  `f045867a-005c-407d-87b1-5efdc4d21fd9`, with `krea2_raw_edit` under profile 1.
- Reference warmup: 121.886 seconds; initialization through capture-ready:
  193.502 seconds. The Identity Edit v1.2 adapter was verified during inference.
- Capture/restore CUDA allocation: 13,501,398,016 bytes (~12.57 GiB). All
  12,820,073,484 logical transformer elements were on CUDA. Text encoder
  (4,022,468,664 elements), vision encoder (415,347,728) and VAE (126,892,531)
  were on CPU. Active MMGP models were exactly `["transformer"]`.
- [Test submission](https://modal.com/apps/larshelg/main/ap-BCZCSIylM3duQhqmnRMX9u):
  job `3d0f271e-d2fe-49c3-9478-683d6e34ff7b`, a courtyard reference-image edit
  from S3 at 1024x576, 20 steps, guidance 2, flow shift 5 and seed 12345.
  Terminal status `succeeded`, one image, no errors, on `WanGPKreaRawEditWorker`.
- Worker elapsed: 53.366 seconds; `load_models_calls=0`, `model_reused=true`, on
  the same restored boot. The shared runtime verified the S3 upload.
- Saved output metadata: JPEG, 143,491 bytes; SHA-256
  `cb473a1e1da23b17534825e4604479267c3b562f55dbb6e804888e2ae1d696e6`.
  The output was not downloaded or visually inspected.

Saved evidence: [request](../runs/job-krea-raw-edit-smoke.json),
[snapshot diagnostics](../runs/krea-raw-edit-snapshot-probe.json), and
[job result](../runs/krea-raw-edit-dedicated-smoke-result.json).

Initial capture/immediate restore is verified. Reuse after scale-to-zero has
not been measured; compare capture and boot IDs across containers for that
check. Two-reference composition, masks, outpainting and additional user LoRAs
were not exercised in this validation.
