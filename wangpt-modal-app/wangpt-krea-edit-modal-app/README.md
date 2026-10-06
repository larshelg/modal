# Krea2 Turbo Edit snapshot worker

Run all commands below from the `wangpt-modal-app` workspace root. Use Modal
module mode (`-m wangpt-krea-edit-modal-app.app`) for this package.

Dedicated app: `wangpt-krea-edit-modal-app`; class: `WanGPKreaEditWorker`.
Model: `krea2_turbo_edit`; GPU: L40S; profile: 1; host RAM: 65536 MiB.
Revision: `krea2-turbo-edit-l40s-transformer-v1`.

Deploy `wangpt-krea-edit-modal-app/app.py` independently. Warmup runs a private eight-step
1024x1024 edit of a synthetic reference with `video_prompt_type: "KI"`.
It verifies that `krea2_identity_edit_v1_2.safetensors` is active during
inference and joins the native generation thread before clearing input/output
payloads, session state and generation caches. Temporary warmup images are
removed and never published as jobs or uploaded to S3.

Native task cleanup unloads LoRAs. Capture requires no active task adapters,
the entire transformer on CUDA, and the text encoder, vision encoder and VAE
on CPU. Restore validates the same placement before reseeding or inference;
it never loads weights to conceal a failed restore. The model's preset edit
adapter and any user LoRAs load through the normal per-request path.

The worker keeps `run(job_id, model, params)`, shared job records and verified
S3 output handling. CLI submissions use the separate deployment only when
`WANGP_KREA_EDIT_SNAPSHOT=1` and the exact image model is `krea2_turbo_edit`.
The checked-in example requires an existing reference path; `/data` paths and
S3 URIs in the configured bucket are supported.

```bash
.venv/bin/python -m modal deploy -m wangpt-krea-edit-modal-app.app
.venv/bin/python -m modal run control.py::krea_edit_snapshot_probe
WANGP_KREA_EDIT_SNAPSHOT=1 .venv/bin/python -m modal run control.py::submit \
  --model krea2_turbo_edit --kind image --params-file wangpt-krea-edit-modal-app/examples/krea2_turbo_edit.json
.venv/bin/python -m modal run control.py::status --job-id JOB_ID
```

Direct TypeScript callers can look up the dedicated class without a CLI flag:

```ts
const cls = await modal.cls.fromName("wangpt-krea-edit-modal-app", "WanGPKreaEditWorker");
const worker = await cls.instance();
const call = await worker.method("run").spawn([jobId, "krea2_turbo_edit", params]);
```

Keep job-record creation and polling in the caller. The worker rejects other
models. Reference order, inpainting, outpainting and user LoRA behavior remain
native WanGP behavior. Different configs, profiles or upscalers may cause normal
model reconfiguration; the snapshot baseline is the default edit preset.

Shared lifecycle and diagnostic helpers are in `snapshot_common.py` at the
workspace root and included in this worker image. Model-specific warmup and
residency checks are in this folder's `snapshot.py`.

## Validation

Local suite: 174 passed, 28 skipped. Focused tests include edit routing, required
vision encoder placement, adapter observation during warmup, payload cleanup
and restore checks. `git diff --check` passed.

Deployment, initial GPU snapshot restoration and real editing verified on
2026-10-05:

- [Deployment](https://modal.com/apps/larshelg/main/deployed/wangpt-krea-edit-modal-app)
  completed in 21.202 seconds; image `im-ASqa1ECV5ibUgBbqoFAKeA`.
- [Initial probe](https://modal.com/apps/larshelg/main/ap-pAuhyTMPjGdqbwnBes5ELg)
  returned capture `13a440e1-1139-4d2d-aab5-425cca11aced` and restored boot
  `b5a3bad1-2998-4411-b846-60c419d120a8`.
- Warmup: 56.838 seconds; initialization through capture-ready: 150.728 seconds.
  The Identity Edit v1.2 adapter was observed during inference.
- Capture and restore CUDA allocation: 13,501,398,016 bytes (~12.57 GiB).
  All 12,820,073,484 logical transformer elements were on CUDA. Text encoder
  (4,022,468,664 elements), VAE (126,892,531) and vision encoder (415,347,728)
  were entirely on CPU. MMGP active models were exactly `["transformer"]`.
- Real job: `d80e79c2-119c-4bab-97b2-1dc598c07afc`, eight steps, 1024x576,
  guidance 0, flow shift 5, seed 12345; terminal status `succeeded`, one image,
  no errors. The input was the previously generated courtyard image in S3.
- Worker elapsed: 20.693 seconds, `load_models_calls=0`, `model_reused=true`.
  It ran on the same restored boot and loaded the preset adapter normally.
- Verified output downloaded from S3: JPEG, 1024x576, 169,279 bytes; SHA-256
  `0bfc1d5828cfdf625fd602c8479c66d995ce61db077b0d61374d6ac4bea2dce4`.
  Visual inspection confirms the blue door became deep red, preserving the
  courtyard, plants, pots, bench, framing and lighting.

Saved evidence:
[request](../runs/job-krea-edit-smoke.json),
[job result](../runs/krea-edit-smoke-result.json),
[snapshot diagnostics](../runs/krea-edit-snapshot-probe.json),
[edited image](../runs/krea-edit-smoke.jpg).

Initial capture/immediate restore is verified. Reuse of an existing snapshot
after scale-to-zero has not been measured; compare capture and boot IDs across
separate containers for that check. Two-reference composition, masks, outpainting
and additional user LoRAs have not been exercised in this validation.
