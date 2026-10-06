# Qwen Image 2.1 7B snapshot worker

Dedicated app: `wangpt-qwen-image-21-modal-app`; class: `WanGPQwenImage21Worker`.
Model: `qwen_image_21_7B`; GPU: L40S; profile: 1; host RAM: 65536 MiB.
Revision: `qwen-image-21-7b-l40s-transformer-v1`.

Deploy `qwen_image_21_app.py` independently. Warmup runs a private 40-step
832x480 edit of a synthetic reference with `video_prompt_type: "KI"`, guidance
4, flow shift 5 and the default FlowMatch Euler solver. The base model uses no
preset acceleration adapter. KV cache and RGBA are disabled for the baseline.

After joining native cleanup, warmup releases input/output payloads, restores
the output directory, resets the headless session state and clears generation
caches. Temporary reference and output images are removed before capture and
never become jobs or S3 artifacts. Capture requires the full transformer on
CUDA, text/vision encoders and VAE on CPU, and no active task LoRAs. Restore
checks the same placement before reseeding; it never reloads weights to hide
a failed restore.

Both text generation and native reference-image editing use
`run(job_id, model, params)`, shared job records and verified S3 output handling.
CLI submissions use this deployment when `WANGP_QWEN_IMAGE_21_SNAPSHOT=1` and
the exact image model is `qwen_image_21_7B`:

```bash
.venv/bin/python -m modal deploy qwen_image_21_app.py
.venv/bin/python -m modal run control.py::qwen_image_21_snapshot_probe
WANGP_QWEN_IMAGE_21_SNAPSHOT=1 .venv/bin/python -m modal run control.py::submit \
  --model qwen_image_21_7B --kind image --params-file examples/qwen_image_21_7B.json
.venv/bin/python -m modal run control.py::status --job-id JOB_ID
```

For an edit, add ordered `image_refs` using existing `/data` images or S3 URIs
in the configured bucket, and set `video_prompt_type` to `KI` (main scene first)
or `I` (people/objects). Reference handling remains native WanGP behavior.

Direct TypeScript clients need no CLI flag:

```ts
const cls = await modal.cls.fromName("wangpt-qwen-image-21-modal-app", "WanGPQwenImage21Worker");
const worker = await cls.instance();
const call = await worker.method("run").spawn([jobId, "qwen_image_21_7B", params]);
```

Keep job-record creation and polling in the caller. The worker rejects other
models and finetunes. Changing config, profile or upscalers can trigger normal
model reconfiguration; the snapshot baseline uses the default model config.

## Validation

Local suite: 189 passed, 28 skipped; `git diff --check` passed. Tests cover exact
opt-in dispatch, required component placement, restore validation, native
warmup cleanup and rejection of wrong profiles, active LoRAs and other models.

Deployment, initial GPU snapshot restoration and generation succeeded on
2026-10-05:

- [Deployment](https://modal.com/apps/larshelg/main/deployed/wangpt-qwen-image-21-modal-app)
  completed in 21.723 seconds; image `im-K9pkhRxiq51PARKjJPw4Dr`.
- Actual GPU: NVIDIA L40S. Capture:
  `b42946b2-5cff-4ead-9646-ff7f65981416`; restored boot:
  `90aee8b8-977a-4966-9fd2-4a6e4d994bf6`.
- Reference-image warmup: 101.499 seconds; initialization through capture-ready:
  192.841 seconds. The loaded model remains `qwen_image_21_7B`, profile 1.
- Restored CUDA allocation: 7,263,634,432 bytes (~6.76 GiB). All 7,115,125,312
  logical transformer elements were on CUDA. Text encoder (7,568,406,136
  elements), vision encoder (576,388,588) and VAE (337,740,404) were on CPU;
  active MMGP models were exactly `["transformer"]`.
- [Test submission](https://modal.com/apps/larshelg/main/ap-9lYhhI19qBXHDF2znVk3QA):
  job `50219ced-3185-42b3-bd35-c4685bfe76ef`, using the checked-in teapot example:
  40 steps, guidance 4, 832x480, seed 42. Terminal status `succeeded`, one image,
  no errors; routed to `WanGPQwenImage21Worker` in the dedicated app.
- Worker elapsed: 26.529 seconds; `load_models_calls=0`, `model_reused=true` on
  the same restored boot. The shared runtime verified the S3 upload.
- Saved job metadata reports JPEG, 76,194 bytes; SHA-256
  `c39d140bb0aef130303ef7649d6dfe1e3e8f316b5d4d7f98b99013b5168b58ae`.
  The output was not downloaded or visually inspected.

Saved evidence: [snapshot diagnostics](../runs/qwen-image-21-snapshot-probe.json)
and [job result](../runs/qwen-image-21-dedicated-smoke-result.json).

Initial capture/immediate restore is verified. Reuse after scale-to-zero has
not been measured; compare capture and boot IDs across containers for that
check. Masks, outpainting, RGBA and user LoRAs have not been exercised in this
snapshot validation.
