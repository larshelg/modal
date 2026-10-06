# Krea2 Turbo snapshot worker

Run all commands below from the `wangpt-modal-app` workspace root. Use Modal
module mode (`-m wangpt-krea-modal-app.app`) for this package.

Dedicated app: `wangpt-krea-modal-app`; class: `WanGPKreaWorker`.
Model: `krea2_turbo`; GPU: L40S; profile: 1; host RAM: 65536 MiB.
Revision: `krea2-turbo-l40s-transformer-v1`.

The base transformer is resident on CUDA at capture. The text encoder and VAE
are included in CPU memory. Native 1024x1024/eight-step image warmup completes
and its thread joins before MMGP stages the transformer. Warmup artifacts are
temporary and are neither published as jobs nor uploaded to S3. No edit LoRA,
vision encoder, or user identity LoRA is included in this base-model snapshot.

Restoration checks tensor placement before inference or weight loading, reseeds
RNGs, and creates a new boot ID while retaining the capture ID. Generation
diagnostics count native `wgp.load_models` calls and compare pipeline identity.
Cold cache reuse requires an existing capture ID in a new boot after scale-down;
Modal's immediate restore of a newly created snapshot does not establish reuse.

Deploy only `wangpt-krea-modal-app/app.py` for this worker. The original app and H3 snapshot
sources are unchanged. CLI routing requires `WANGP_KREA_SNAPSHOT=1`; direct
TypeScript clients select the separate app/class explicitly:

```ts
const cls = await modal.cls.fromName("wangpt-krea-modal-app", "WanGPKreaWorker");
const worker = await cls.instance();
const call = await worker.method("run").spawn([jobId, "krea2_turbo", params]);
```

Keep existing job-record creation and polling in the caller. The edit preset
and all non-Krea models are rejected by this worker. Different model configs,
profile overrides, or upscalers may require normal native reconfiguration;
snapshot reuse is validated for the default preset, not every possible setting.

Shared lifecycle and diagnostic helpers are in `snapshot_common.py` at the
workspace root and included in this worker image. Model-specific warmup and
residency checks are in this folder's `snapshot.py`.

## Validation

Local tests: `106 passed, 7 skipped`; `git diff --check` passed.

Deployment and first restored generation succeeded on 2026-10-01:

- [Deployment](https://modal.com/apps/larshelg/main/deployed/wangpt-krea-modal-app)
  completed in `18.005` seconds; image `im-EUouTD3Im1OEEkFPNdoOUF`.
- [Initial probe](https://modal.com/apps/larshelg/main/ap-nrRaMEMKh01hgbiSaARU1F).
- Container `ta-01M3T6E4TV72A2P0GKWS8BR0HR`; actual GPU `NVIDIA L40S`.
- Capture `fab3e00d-19b2-45a9-8df1-44ba636d7faf`;
  restored boot `74e8648f-9089-410f-964f-cee7a8e79515`.
- Loaded checkpoint `Krea2Turbo_quanto_bf16_int8.safetensors`; text encoder
  `Qwen3-VL-4B-Instruct_quanto_bf16_int8.safetensors`.
- Warmup `44.697` seconds; initialization through capture-ready `136.678`
  seconds; post-cleanup transformer staging `1.458` seconds.
- CUDA allocation before staging `9,568,256` bytes; capture and immediate
  restore allocation exactly `13,501,396,992` bytes (~12.57 GiB).
- All `12,820,073,484` logical transformer elements on CUDA; text encoder and
  VAE tensors on CPU; MMGP active IDs exactly `["transformer"]`.
- Peak CUDA allocation during warmup `13,882,143,232` bytes (~12.93 GiB).
  At capture, free GPU memory `33,430,568,960` bytes (~31.13 GiB).
- Capture RSS `29,284,155,392` bytes; immediate restored RSS `25,285,140,480`.
- Capture-ready 00:22:15 Europe/Oslo; snapshot created/restoring 00:24:17–18;
  restored residency validated/ready 00:24:32. This is initial snapshot
  restoration, not an independently cached cold-start benchmark.

Real generation on that same restored boot:

- Job `6705fcc0-3e21-4d4d-995c-1133d873a990`.
- [Submission](https://modal.com/apps/larshelg/main/ap-JTkNMOXQZcarhmE4g3t7tB).
- Standard `wangpt-krea-modal-app/examples/krea2_turbo.json`: fox walking through fresh snow at golden
  hour; 1024x1024, eight steps, guidance 0, flow shift 5, seed 12345.
- Terminal status `succeeded`, one successful task, no errors.
- Worker elapsed `14.740` seconds; `load_models_calls=0`, `model_reused=true`.
- Created `2026-09-30T22:29:25.655680+00:00`; started
  `22:29:26.395297+00:00`; completed `22:29:40.996803+00:00`.
- Verified S3 output downloaded locally: JPEG, 1024x1024, `229,639` bytes.
  Image inspection shows the prompted fox in snow. SHA-256:
  `efdb0fec01296640ebf1c54af2e381adb09a6a7173c7b4abd4d522c51125d95f`.
- S3 prefix:
  `s3://bucket-rlnfehwax8alyao51e/runninghub/wangp/6705fcc0-3e21-4d4d-995c-1133d873a990/`.

Snapshot capture, immediate GPU-resident restoration, and real generation on
L40S are verified. Reuse in a new container after scale-to-zero is not yet
measured. The original H3 deployment and its snapshot sources were not changed.
