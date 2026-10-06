# Krea2 RAW snapshot worker

Run commands from the `wangpt-modal-app` workspace root. Use Modal module mode
(`-m wangpt-krea-raw-modal-app.app`) for this package.

Dedicated app: `wangpt-krea-raw-modal-app`; class: `WanGPKreaRawWorker`.
Model: `krea2_raw`; GPU: L40S; WanGP profile: 1; host RAM: 65536 MiB.
Revision: `krea2-raw-l40s-transformer-v1`.

The worker follows the Krea Turbo snapshot lifecycle, using the RAW model and
its own defaults: 1024x1024, 52 steps, guidance 3.5 and flow shift 5. Private
warmup generates an image, joins the native task thread, releases task payloads,
clears session state/caches and removes temporary output before capture.
Warmup is never published as a job or uploaded to S3.

Capture keeps the full transformer on CUDA, with text encoder and VAE on CPU,
and requires profile 1 without active task LoRAs. Restore validates placement
before reseeding or inference; it never reloads weights to conceal a failed
restoration. The worker shares `snapshot_common.py`, the WanGP runtime, Volume,
job store and verified S3 output handling.

```bash
.venv/bin/python -m modal deploy -m wangpt-krea-raw-modal-app.app
.venv/bin/python -m modal run control.py::krea_raw_snapshot_probe
WANGP_KREA_RAW_SNAPSHOT=1 .venv/bin/python -m modal run control.py::submit \
  --model krea2_raw --kind image \
  --params-file wangpt-krea-raw-modal-app/examples/krea2_raw.json
.venv/bin/python -m modal run control.py::status --job-id JOB_ID
```

The client flag is opt-in and affects only `krea2_raw` image jobs. RAW Edit,
Turbo and other models keep their respective routes. Direct TypeScript clients
can select the class without a CLI flag:

```ts
const cls = await modal.cls.fromName("wangpt-krea-raw-modal-app", "WanGPKreaRawWorker");
const worker = await cls.instance();
const call = await worker.method("run").spawn([jobId, "krea2_raw", params]);
```

Keep job-record creation and polling in the caller. The worker retains
`run(job_id, model, params)` and rejects other models before accessing jobs.
LoRAs and native inpainting settings load through normal WanGP generation.
Changes to config, profile or upscalers may trigger normal reconfiguration;
snapshot reuse is scoped to the default RAW model config.

The worker scales to zero after 300 idle seconds and allows at most one
container. Deployment overrides: `WANGP_KREA_RAW_MEMORY_MB`,
`WANGP_KREA_RAW_MAX_CONTAINERS` and `WANGP_KREA_RAW_SCALEDOWN_WINDOW`.

## Validation

Local suite: 220 passed, 28 skipped; `git diff --check` passed. Tests cover exact
opt-in routing and dispatch, isolation from Turbo/Edit routes, required tensor
placement, restore validation before reseeding and warmup payload cleanup.
Modal module imports and source mounts resolved locally.

Deployment, initial GPU snapshot restoration and generation succeeded on
2026-10-06:

- [Deployment](https://modal.com/apps/larshelg/main/deployed/wangpt-krea-raw-modal-app)
  completed in 35.069 seconds; image `im-D9K669OYm59CIxIBFIIrGv`.
- [Initial probe](https://modal.com/apps/larshelg/main/ap-uKXBsVIQLnfWDp29TUIFBb)
  was followed by the queued generation. Snapshot diagnostics returned capture
  `dcb8a61d-a3ab-48c1-b2f9-69c5f391b5eb` and restored boot
  `df7d6910-40b5-47ab-ba92-b2ecfa8c2e9c`, with `krea2_raw` loaded under profile 1.
- Warmup: 120.802 seconds; initialization through capture-ready: 227.573 seconds.
- Restored CUDA allocation: 13,501,396,992 bytes (~12.57 GiB). All
  12,820,073,484 logical transformer elements were on CUDA. Text encoder
  (4,022,468,664 elements) and VAE (126,892,531) were on CPU. Active MMGP models
  were exactly `["transformer"]`; no active task LoRAs remained at capture.
- [Test submission](https://modal.com/apps/larshelg/main/ap-tFWCL2veVMgMnmztcKohMz):
  job `182c7b2c-6fdb-43b1-bd6a-6427da0e3cac`, using the fox example at 1024x1024,
  52 steps, guidance 3.5, flow shift 5 and seed 12345. Terminal status:
  `succeeded`, one image, no errors; routed to `WanGPKreaRawWorker` in this app.
- Worker elapsed: 95.966 seconds; `load_models_calls=0`, `model_reused=true` on
  the same restored boot. The shared runtime verified the S3 upload.
- Saved output metadata: JPEG, 268,679 bytes; SHA-256
  `1cf03ad9044687745c71e7d5ab9acf4e46285e28a1c16468605d690f59210b80`.
  The output was not downloaded or visually inspected.

Saved evidence: [snapshot diagnostics](../runs/krea-raw-snapshot-probe.json)
and [job result](../runs/krea-raw-dedicated-smoke-result.json).

Initial capture/immediate restore is verified. Reuse after scale-to-zero has
not been measured; compare capture and boot IDs across containers for that
check. Inpainting and user LoRAs were not exercised in this validation.
