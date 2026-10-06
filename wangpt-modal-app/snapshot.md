# Singularity GPU snapshot plan

## Transformer-only residency experiment (v3)

Implemented and deployed revision `singularity-transformer-resident-v3`.
Three independent captures and immediate GPU-residency restore checks passed.
A real post-restore generation succeeded with zero native model-loader calls.
The fourth cold probe confirmed reuse of the second capture in a new container,
with 19.72 GiB still allocated immediately after restore. Logs show about 36
seconds from restore start to ready, without initialization or warmup rerunning.
This is one cached-restore sample, not an end-to-end generation benchmark.
After native
profile-1 warmup, joined task cleanup, and
Volume commit, call MMGP `ensure_model_loaded("transformer")`. Check synchronized
CUDA allocation, MMGP active-model IDs, and every component's tensor placement.
Fail capture unless all transformer tensors are on CUDA, all other components
remain on CPU, only the transformer is active, and allocation is at least 19 GiB.
The lower bound accommodates this checkpoint's actual ~19.6 GiB INT8 weight size
without adding unrelated tensors to reach an arbitrary 20 GiB threshold.

Immediately after restore, record and validate residency before reseeding or
inference. Do not reload in the restore hook. Keep this report separately from
later current-memory reports. Record native `wgp.load_models` calls for each
generation and retain the model-object reuse check. LoRA loads and native
stage-to-stage CPU/GPU transfers remain normal behavior.

The first-denoising guard now requires the requested denoising step count,
excluding the previously misleading 50-step text-encoder transition. Local
validation: `94 passed, 7 skipped`. All three captures retained 19.72 GiB allocated
CUDA memory through restore, with only transformer weights on CUDA.
Generic worker defaults and exact-model opt-in routing remain unchanged.

The measurements below describe earlier revisions unless marked v3.

### v3 fourth cold probe: confirmed cached reuse — 2026-09-30

The app container list was empty before invocation. No deployment or worker
code change occurred between the three captures and this probe.

- [Probe run](https://modal.com/apps/larshelg/main/ap-t9fpiib99FlrAkKOjXbK4F).
- New container: `ta-01M3SYBZ3A8KMRZ0NSYEV483AR`.
- Reused capture: `f555beb8-c813-467f-8728-16e6faa9d245`, from the second probe.
- New boot: `74340fde-f51a-4829-9c67-1f172532385d` (original boot of this capture
  was `038777b9-8c82-4773-bc7b-3fac05943072`).
- GPU: `NVIDIA H100 80GB HBM3`; profile `1.0`.
- Restore started 21:58:52 Europe/Oslo; residency validated/container ready
  21:59:28. Approximately 36 seconds for the logged restore-to-ready interval.
  This excludes client submission/scheduling time and generation.
- No new capture, initialization, or warmup events in this container's logs.
  The returned warmup timings are saved values from capture two, not work
  repeated in this container.
- Immediate restored and current CUDA allocation: `21,171,887,616` bytes
  (~19.72 GiB). All transformer tensors on CUDA, all other components on CPU,
  and MMGP active IDs exactly `["transformer"]`.
- Immediate restored RSS `68,506,099,712` bytes.
- Probe succeeded with `requests_served=0`, `last_request=null`.
  No generation submitted during this probe.

This confirms cached GPU-resident snapshot reuse after scale-to-zero. Three
captures preceded the first observed hit, consistent with Modal's documented
host-variant coverage behavior; the underlying host differences were not
independently identified. Keep the deployment unchanged for further timings.
Generation after this independently cached restore is still untested; the
earlier successful generation followed the initial restore of the same capture.

### v3 initial GPU-resident capture and restore

- Deployment completed in `16.594` seconds; image `im-mQBDX6jazScHfaZcq6gy5G`.
- [Probe run](https://modal.com/apps/larshelg/main/ap-4moDmw7liOAeHrWPUbL7u9).
- Container: `ta-01M3SVCYAGMKV12Q8QSRW692ZR`.
- Capture ID: `ed4a07bd-fef7-438a-a4ea-72596efb6863`.
- Restored boot: `9de4c28d-d3bd-4a55-a012-733d7dad6652`.
- GPU: `NVIDIA H100 80GB HBM3`; profile `1.0`.
- Native warmup `364.824` seconds; initialization through capture-ready
  `470.265` seconds. Accelerator verification succeeded.
- MMGP transformer staging took `1.344` seconds. Allocated CUDA increased from
  `34,603,008` bytes to `21,171,887,616` bytes (~19.72 GiB), a delta of
  `21,137,284,608` bytes. This is an observed pinned-RAM-to-GPU transfer time,
  not a checkpoint-from-storage loading time.
- Before capture, immediately after restore, and in the probe method:
  transformer logical elements `20,111,439,144` on `cuda:0`, other components
  on CPU, and only `transformer` active in MMGP. CUDA allocation remained
  exactly `21,171,887,616` bytes in all three reports.
- Capture RSS: `72,533,266,432` bytes; immediate restored RSS:
  `68,497,723,392` bytes. Peak CUDA allocation: `26,229,072,896` bytes.
- Creation started 21:06:57 Europe/Oslo; capture-ready 21:14:53;
  snapshot created/restoring 21:17:19–20; restore validation/ready 21:17:31.
- Probe succeeded with `requests_served=0`. Restore validation performed no
  loading or warmup. This proves GPU residency survived Modal's initial
  snapshot/restore cycle, but is not yet a cached cold-start benchmark.

### v3 separate cold probe and real generation — 2026-09-30

The first container scaled down naturally. A read-only container listing
confirmed no active app containers before the next probe; no redeployment
occurred between probes.

- [Cold probe run](https://modal.com/apps/larshelg/main/ap-a2VzwpqewoAK8eoYmmO5NR).
- Container: `ta-01M3SWET9VNKQTA70EFX8BJ34R`.
- New capture ID: `f555beb8-c813-467f-8728-16e6faa9d245`.
- Restored boot: `038777b9-8c82-4773-bc7b-3fac05943072`.
- GPU: `NVIDIA H100 80GB HBM3`; profile `1.0`.
- Warmup `261.801` seconds; initialization through capture-ready `353.807`
  seconds. MMGP transformer staging took `1.611` seconds.
- Before capture, immediately after restore, and in the probe method:
  `21,171,887,616` bytes allocated CUDA (~19.72 GiB), all transformer tensors
  on CUDA, every other component on CPU, and only `transformer` active.
- Capture RSS `72,541,900,800` bytes; immediate restored RSS `68,506,099,712`.
- Container creation 21:25:28 Europe/Oslo; capture-ready 21:31:25;
  snapshot created 21:34:09; restoring 21:34:10; restore validation/ready
  21:34:43. Approximately 9m 15s creation-to-ready, not cached restore latency.
- Probe succeeded with `requests_served=0`, without restore-hook loading.

The following generation used the same restored boot:

- Job: `ad1d33ae-15c9-4630-a9fa-f99ba3f1807e`.
- Function call: `fc-01M3SX3Z845QZSXJYFQT14YC3M`.
- [Submission run](https://modal.com/apps/larshelg/main/ap-XZZC4wGgFN8iVlWP3ILMIg).
- Unchanged robot-and-teacup request in
  [`examples/singularity_snapshot_test.json`](examples/singularity_snapshot_test.json).
- Created `2026-09-30T19:36:59.821692+00:00`; running
  `19:37:00.861319+00:00`; completed `19:38:00.595567+00:00`.
- Terminal status `succeeded`, one successful task, no errors.
- Worker generation `60.2` seconds; submission-to-completion about `60.774`
  seconds. First actual denoising progress about `18.8` seconds.
- `load_models_calls=0`, `model_reused=true`. This verifies no native pipeline
  reconstruction; normal LoRA loading and CPU/GPU staging still occur.
- Output seed `300767515`; size `4,455,369` bytes; SHA-256 verified:
  `50816ba94ce4f0ef2571af08b22357396cbd33bf216877a771a5400d97fc4672`.
- Downloaded output verified: H.264 `832x480`, 24 fps, 124 frames,
  `5.166667` seconds; AAC 32000 Hz stereo. Sampled frames show the prompted
  brass robot and blue teacup. Audio content was not subjectively reviewed.
- S3 prefix:
  `s3://bucket-rlnfehwax8alyao51e/runninghub/wangp/ad1d33ae-15c9-4630-a9fa-f99ba3f1807e/`.

Conclusion: transformer-only GPU residency survives Modal capture and restore,
and real generation works afterward without another native model load. Cached
reuse remains the unresolved measurement: the cold probe built a new capture
even on the same reported GPU type. Capture IDs above are application-generated
diagnostic UUIDs, not Modal checkpoint IDs. This experiment establishes GPU
residency, not a cold-start latency improvement.

### v3 third cold probe — 2026-09-30

Another read-only container listing confirmed the app had scaled to zero before
running `python3 -m modal run control.py::snapshot_probe`. The deployment and
worker code remained unchanged.

- [Probe run](https://modal.com/apps/larshelg/main/ap-IBt1AHph5dflIZw6NMQKCx).
- Container: `ta-01M3SXHWM7D992CN7KK1J5V2DR`.
- New capture ID: `7e413ece-0a61-4d68-9045-69f92cbb92db`.
- Restored boot: `ec513659-f434-48a1-91f9-0f588d2637f1`.
- GPU: `NVIDIA H100 80GB HBM3`; profile `1.0`.
- Logs explicitly reported snapshot creation at 21:44:38 Europe/Oslo and
  native initialization with the new capture ID at 21:44:41.
- Capture-ready at 21:49:24; snapshot created/restoring at 21:51:26–27;
  restored residency validated at 21:51:37. Creation-to-ready was about
  6m 59s. The roughly 10-second restoration phase followed a freshly created
  snapshot and is not an independent cached cold-start latency measurement.
- Warmup `225.733` seconds; initialization through capture-ready `283.217`
  seconds. Transformer staging took `1.079` seconds.
- Capture, immediate restore, and probe reports all retained
  `21,171,887,616` allocated CUDA bytes (~19.72 GiB), all transformer tensors
  on CUDA, all other components on CPU, and only `transformer` active.
- Capture RSS `72,539,369,472` bytes; immediate restored RSS `68,503,085,056`.
- Probe completed successfully, `requests_served=0`, `last_request=null`.
  No additional real generation was submitted in this probe.

This was another new capture, not a cached hit. There are now three successful
H100 captures of the unchanged v3 deployment. Modal's
[snapshot FAQ](https://modal.com/docs/guide/memory-snapshots) says GPU functions
need 2–3 snapshots per GPU type because snapshots depend on underlying worker
types, including CPU flags. This makes the observed behavior plausible, but
does not guarantee reuse on any particular subsequent invocation. The next
useful cold probe should retain this deployment and compare its capture ID
against these three; earlier revisions do not count toward v3 coverage.

## Earlier profile-1 experiment (v2)

In v2, the dedicated Singularity worker default changed to profile 1,
revision `singularity-profile1-native-offload-v2`; deployment, initial snapshot
creation/restored probe, and one real generation succeeded on 2026-09-30.
Generic worker profiles and exact-model opt-in routing are unchanged.
GPU device names are now included in memory diagnostics.

The pinned WanGP `init_pipe()` only adds low-VRAM budgets for profiles 2/4/5;
MMGP profile 1 enables RAM pinning for all components with unrestricted default
budgets, subject to its memory safety checks. WanGP still calls
`offloadobj.unload_all()` after successful generation regardless of profile.
This experiment tests native profile 1 performance and capture residency; it
does not override cleanup or force all pipeline components into VRAM together.
Measured profile-1 idle diagnostics still show CPU weights: 24.4 GiB peak CUDA
allocation during warmup fell to 33 MiB at capture. The proposed fully resident
snapshot therefore remains a separate change. No permanent identity LoRA is
included. The real generation took 62.324 seconds on the restored warm worker;
this single sample does not establish a speedup over profile 4.

## Profile-1 experiment results — 2026-09-30

- Local verification: `87 passed, 7 skipped` with `.venv/bin/python -m pytest -q`.
- Deployment succeeded in `17.045` seconds; dedicated image
  `im-XANndo2aXFIIq5kf5e2wTB`. GPU request remains `H100` (upgrades allowed).
- [Probe run](https://modal.com/apps/larshelg/main/ap-euvCgV97fR90UIluSD97Va).
- Container: `ta-01M3ST9KEEBGBABGMBAQ5CZZ5R`.
- Capture: `27cec51d-eb18-47bb-a6fb-8b7a37b7d21a`.
- Boot: `2c1e088f-7a51-4677-a1f9-06cf2eec8bd1`.
- Confirmed GPU name: `NVIDIA H100 80GB HBM3`.
- Creation began at 20:47:40 Europe/Oslo; capture-ready at 20:53:24;
  snapshot created/restoring at 20:55:09; container ready at 20:55:22.
  Creation-start to ready: approximately 7m 42s, not a cached-start benchmark.
- Warmup: `272.055` seconds; initialization through capture-ready: `339.657`
  seconds. Model profile `1.0` and accelerator verified.
- Peak CUDA allocation: `26,229,072,896` bytes (~24.4 GiB), versus
  `3,381,774,336` bytes (~3.1 GiB) in earlier profile-4 probes.
- Capture CUDA allocation: `34,603,008` bytes (33 MiB). All component weights
  reported CPU placement; no active MMGP models or LoRA adapters at idle.
- Capture RSS: `72,672,612,352` bytes; restored RSS: `68,634,439,680` bytes.
- Probe returned successfully after restore with `requests_served=0`.

Post-restore real generation used the unchanged robot-and-teacup request:

- Job: `4590b56d-5ba4-4cc1-be70-9e4b49c95bd3`.
- Function call: `fc-01M3STRECA3SX511NY6QFJG5VH`.
- [Submission run](https://modal.com/apps/larshelg/main/ap-dqBxtQBG0VCmKrp9KjnLIb).
- Created `2026-09-30T18:55:45.007534+00:00`; running
  `18:55:46.045406+00:00`; completed `18:56:47.844089+00:00`.
- Terminal status `succeeded`, one successful task and no errors.
- Worker timing `62.324` seconds, `model_reused=true`; submission-to-completion
  `62.837` seconds. Same boot as probe, hence a warm-worker request.
- Exclude `first_denoising_seconds=4.885` from comparisons because the known
  step-50 text-encoding instrumentation issue remains.
- Output seed `412989168`; size `4,187,747` bytes; SHA-256 verified:
  `7b64f2b1f288c35fa899860a20acfb50b408fbb6fe253a6211bf39e9b19d2ea3`.
- Media verified locally: H.264 `832x480`, 24 fps, 124 frames, `5.166667`
  seconds; AAC 32000 Hz stereo. Audio content was not subjectively reviewed.
- S3 prefix:
  `s3://bucket-rlnfehwax8alyao51e/runninghub/wangp/4590b56d-5ba4-4cc1-be70-9e4b49c95bd3/`.

Conclusion: native profile 1 works through snapshot creation, restore, and
generation, but does not retain GPU weights after cleanup. Capturing resident
weights requires an additional, explicit MMGP-compatible residency step at the
idle boundary, with memory headroom and post-restore inference verification.
That step was not implemented in this experiment. No cached cold restore or
profile-1 performance improvement has been established.

Status: initial native-offload implementation deployed to Modal on 2026-09-30.
Initial warmup, snapshot creation, the first restored probe, and three real
generations succeeded.
Both later requests started cold but created new snapshots before restoring
and generating. Cached cold-restore benchmarks and
comparative performance benchmarks are pending. Routing remains disabled by
default.

Latest cold `snapshot_probe` also succeeded, but created a fourth capture.
Cached reuse remains unverified. Investigate repeated capture before spending
more runs on cold-start benchmarks; preserve the current deployment meanwhile.

Deployment: [wangpt-modal-app](https://modal.com/apps/larshelg/main/deployed/wangpt-modal-app).
Modal confirmed creation of `WanGPSingularityWorker.*` alongside the existing
image/video workers and successful app deployment. Deployment alone does not
verify that an H3 snapshot has been created or restored; the probe below supplied
that evidence.

## First deployed probe — 2026-09-30

Command: `python3 -m modal run control.py::snapshot_probe`.
The command exited successfully after snapshot creation and restoration.

- [Probe run](https://modal.com/apps/larshelg/main/ap-rHIcz7mxN8PSmv4mQaAgf6).
- Snapshot revision: `singularity-native-offload-v1`.
- Worker container: `ta-01M3SM6MZVH329B4MSQ9S8QHYR`.
- Capture ID: `5308fdfb-95fc-479c-ba3c-4f965e871992`.
- Restored boot ID: `559cebf8-9b87-41b3-8ade-c75b72490178`.
- Model: `minimax_h3_ref2va_singularity_pruned`, profile `4`, configuration `""`.
- Warmup completed in `410.102` seconds; initialization through capture-ready
  diagnostics took `501.3` seconds. These are initial preparation times, not
  restored cold-start latency.
- Warmup produced an MP4 and verified the LightX2V accelerator during inference.
- At 19:11:15 Europe/Oslo, Modal logged:
  `Snapshot created. Restoring Function from memory snapshot.`
- The restored `snapshot_info` method returned successfully, with the expected
  loaded model and `requests_served=0`.
- Capture process RSS: `68,376,420,352` bytes (about 63.7 GiB).
- Restored process RSS reported by the probe: `34,201,018,368` bytes (about
  31.9 GiB). This is one observation, not a steady-state memory estimate.
- CUDA allocated memory: `34,603,008` bytes (33 MiB), both at capture and probe.
- All reported model component parameters/buffers were on CPU; MMGP active
  models and active LoRA adapters were empty at the idle boundary, as expected
  with native cleanup.

This probe establishes that Modal can create and restore this initialized
CPU/CUDA baseline and execute a diagnostic method afterward. The real generation
below additionally verifies inference on that restored worker. Full GPU weight
residency and a cold-start speedup have not been established.

## First real generation after restore — 2026-09-30

- Job: `49d3e19d-5c74-46cf-b386-e3b48d9feda9`.
- Function call: `fc-01M3SMY4YDE7NAS8NXSF1MV08N`.
- [Submission run](https://modal.com/apps/larshelg/main/ap-GgrXcpOMyMJrv453AogqE8).
- Request: [`examples/singularity_snapshot_test.json`](examples/singularity_snapshot_test.json).
- Prompt: a tiny brass robot painting a golden stripe on a blue ceramic teacup
  in a sunlit workshop. Text-only input, four steps, random seed requested;
  output filename records seed `573770005`.
- Status: `succeeded`, one successful task, no reported errors.
- Worker: `WanGPSingularityWorker`, same restored boot ID as the first probe.
- Job created: `2026-09-30T17:14:00.313017+00:00`; running:
  `17:14:01.283163+00:00`; completed: `17:15:52.649417+00:00`.
- Submission-to-completion: approximately `112.34` seconds. Worker request-end
  diagnostics reported `111.613` seconds and `model_reused: true`.
- Output verified locally: H.264 video, `832x480`, `24` fps, `124` frames,
  `5.166667` seconds; AAC audio, `32000` Hz, stereo.
- File size: `4,420,784` bytes. Downloaded SHA-256 matched the job record:
  `6b1601e9b02f37bb373111d06fd1f31f3dd67993e953f4e8245e5a4d0842ddbb`.
- Sampled video frames show the brass robot, brush, blue cup, and golden stripe.
  Audio stream presence was verified; subjective audio quality was not reviewed.
- Output is stored under
  `s3://bucket-rlnfehwax8alyao51e/runninghub/wangp/49d3e19d-5c74-46cf-b386-e3b48d9feda9/`.

This request reused the already-running restored container. Its roughly
one-second dispatch delay is not a fresh cold-restore measurement. The next
validation is a new cold restore and an ordinary-worker comparison.

Instrumentation issue found: `first_denoising_seconds=24.419` was emitted with
`step=50`, corresponding to the end of the 50-step text-encoder progress rather
than the four denoising steps. The current guard accepts a generic `inference`
phase that WanGP can emit at that transition. Exclude this intermediate timing
from benchmarks and tighten the guard to identify actual denoising progress
before using it to compare time to first step. Total request timing and the
model-reuse result are unaffected.

## Second generation from a cold pool — 2026-09-30

- Job: `73bd6449-a252-40a1-8bcf-c7397bda42ce`.
- Function call: `fc-01M3SNE5S6SFYXTR7V346R2PZ3`.
- [Submission run](https://modal.com/apps/larshelg/main/ap-XDJiVpevK0DeJTgLQfaPyJ).
- Same request file and prompt as the first generation; new random output seed
  `997212529`. No worker code change or redeployment between these tests.
- Container: `ta-01M3SNE62DCTJQ258HASZMZRQR`.
- Capture ID: `b7919dc6-fa0b-49e0-a58b-5b0d804f4afa`.
- Restored boot ID: `4f216526-aee8-4303-816d-adb0b627b641`.
- At 19:22:47 Europe/Oslo, Modal logged creation of a new GPU memory snapshot.
  Warmup succeeded in `473.306` seconds and verified the accelerator.
- Capture-ready diagnostics at 19:32:30 reported CPU RSS `68,383,813,632`
  bytes, CUDA allocated `34,603,008` bytes, and all component tensors on CPU.
- At 19:36:01, Modal logged `Snapshot created. Restoring Function from memory
  snapshot.` Container-ready and request-start diagnostics followed at 19:36:08.
- Job created: `2026-09-30T17:22:45.572987+00:00`; running:
  `17:36:08.751465+00:00`; completed: `17:38:23.455948+00:00`.
- Submission-to-completion: `937.883` seconds (15m 38s). Time before running:
  `803.178` seconds (13m 23s). Running-to-completion: `134.704` seconds.
- Worker request-end diagnostics: `134.851` seconds, `model_reused: true`.
  The known step-50 instrumentation issue also occurred here; exclude its
  `first_denoising_seconds=18.559` value from benchmarks.
- Terminal status: `succeeded`, one successful task, no reported errors.
- Downloaded output: `4,617,268` bytes; SHA-256 verified against the job record:
  `6b11be6d9ead6dcc26ef8dc72bd0c664ee9e92db52725762e5e5e9ca836a03d0`.
- Media verified: H.264, `832x480`, `24` fps, `124` frames, `5.166667` seconds;
  AAC audio, `32000` Hz, stereo. Sampled frames show the brass robot painting
  a golden stripe on a blue cup. Subjective audio quality was not reviewed.
- Output is stored under
  `s3://bucket-rlnfehwax8alyao51e/runninghub/wangp/73bd6449-a252-40a1-8bcf-c7397bda42ce/`.

This validates another creation/restore/generation cycle, not a cached
cold-restore speedup. Modal documents that different underlying worker types
can require multiple snapshots; the logs do not establish why this invocation
created a new one. A request that reuses an existing capture on a new container
is still needed for the intended cold-start benchmark.

## Third generation / third capture — 2026-09-30

- Job: `ca003543-0a86-41c8-b5e7-f47e9531f8f5`.
- Function call: `fc-01M3SPTF2J1QPDR8SQJKFBSWVS`.
- [Submission run](https://modal.com/apps/larshelg/main/ap-fnoIqNJbSXBzJCmzChUE1H).
- Same request file and prompt; random output seed `610623951`.
  No worker change or redeployment between tests.
- New capture ID: `03ca21c2-27f0-4525-92f1-fbf20c4f5037`.
- Restored boot ID: `2836a0c9-471d-45ec-b085-299743ec3f7e`.
- Initialization logged at 19:47:03 Europe/Oslo; warmup completed at 19:51:41
  in `214.914` seconds. Accelerator verification succeeded.
- Capture-ready CPU RSS: `68,400,758,784` bytes; CUDA allocated: `34,603,008`
  bytes. Component weights were on CPU, as in the previous captures.
- Snapshot created/restoring logged at 19:52:26; container ready at 19:52:30.
- Job created: `2026-09-30T17:46:56.900086+00:00`; running:
  `17:52:31.509752+00:00`; completed: `17:53:31.840475+00:00`.
- Submission-to-completion: `394.940` seconds (6m 35s). Before running:
  `334.610` seconds (5m 35s). Running-to-completion: `60.331` seconds.
- Worker request-end timing: `61.209` seconds; `model_reused: true`.
  Exclude `first_denoising_seconds=17.122` because of the known step-50 bug.
- Terminal status: `succeeded`, one successful task, no errors.
- Download checksum verified:
  `a7743fb7fb2c96cb97dc23bc56b6e004399374759e2ab4cea820182c19e79a03`.
- Output: `4,776,974` bytes; H.264 `832x480`, 24 fps, 124 frames,
  `5.166667` seconds; AAC 32000 Hz stereo.
- Output is stored under
  `s3://bucket-rlnfehwax8alyao51e/runninghub/wangp/ca003543-0a86-41c8-b5e7-f47e9531f8f5/`.

This again created a new capture, so the shorter elapsed time is not evidence
of a cached cold-start improvement. Modal's documentation says GPU functions
typically need 2–3 snapshots per GPU type to cover underlying worker types.
Three captures are consistent with that guidance, but the precise cause is
unconfirmed. Keep this deployment unchanged. After scale-to-zero, use the
existing `snapshot_probe` for the next cache check: it returns capture and boot
IDs without a user generation, although a cache miss still incurs snapshot
warmup. An old capture ID with a new boot ID establishes cached restoration;
another new capture warrants investigating platform snapshot metadata before
repeating further cold generations.

## Fourth cold start: diagnostic probe — 2026-09-30

Command: `python3 -m modal run control.py::snapshot_probe`.

- [Probe run](https://modal.com/apps/larshelg/main/ap-LHzVvFcGQyNVnZqudFm25Z).
- Deployed app ID: `ap-EGO67OTG3FXQ7tpoaN9ucp`.
- Container: `ta-01M3SRBWARZ744JJPMS3KF2B4R`.
- New capture ID: `726c72f2-ff1e-4b0f-aaab-744ad6ae792a`.
- Restored boot ID: `68574fd1-1e80-427e-92ed-0217c360f159`.
- Snapshot creation began at 20:13:57 Europe/Oslo; initialization logged at
  20:13:59; capture ready at 20:23:13; snapshot created/restoring at 20:24:42;
  container ready at 20:25:12. Creation-start to ready was approximately
  11m 15s. This is not a measured client submission-to-response duration.
- Warmup: `470.01` seconds. Initialization through capture-ready diagnostics:
  `553.632` seconds. Accelerator verification succeeded.
- Probe returned successfully after restore, with the expected model/profile,
  `requests_served=0`, and `last_request=null`. No user generation was submitted.
- Capture RSS: `68,389,040,128` bytes; restored RSS: `34,213,330,944` bytes.
  CUDA allocation remained `34,603,008` bytes; component weights remained on CPU.
- GPU total memory reported `150,110,011,392` bytes, matching the first capture;
  the second and third captures reported `85,017,624,576` bytes. The worker
  requests `gpu="H100"`. Actual device names/worker fingerprints were not
  collected, so these differences do not establish the recapture cause.

Read-only deployment history confirms version `v10`, deployed at
`2026-09-30 19:00:08+02:00` using client `1.5.4`, remains the latest deployment.
There were no intervening deployments. The probe client is `1.5.0`; it looks up
the deployed class by name and does not deploy worker code. The current local
deployment environment reports `snapshot_debug=False`; this is not a readback
of the deployed function configuration. Public CLI metadata examined did not
expose a cache-miss reason, and no browser was available for dashboard inspection.

This is four distinct captures on the unchanged deployment, all of which
created and restored successfully. The expected cached-reuse behavior is still
unproven. Next investigation: inspect the deployed function's Containers and
snapshot metadata, or provide these app/container/capture IDs to Modal support
to identify worker-type coverage, invalidation, or restore fallback. No support
message has been sent. Do not redeploy merely to retry, since that invalidates
the existing captures.

## Platform metadata investigation — 2026-09-30

Read-only API calls used the installed Modal client: `AppGetLayout`,
`TaskGetInfo`, and historical `AppFetchLogs`, scoped to this deployment and
the four test containers. No worker was started and no deployment was changed.

- All four captures' historical logs belong to one function ID:
  `fu-UZUXRfTmbdLQCVbPDtzlCW` (`WanGPSingularityWorker.*`).
- Class ID: `cs-ffWugEPSUvxY2E6BuNP1iT`.
- The third container ID, previously missing from the notes, is
  `ta-01M3SPTF8P3ZBCC6GMMKEAVARR`.
- All four containers ultimately reported successful results, exit code `0`,
  and empty exceptions. Their logs show creation followed by restoration;
  no restore-error or fallback message was found in the retrieved logs.
- `TaskGetInfo` reports `gpu_type="H100"`, count `1`, on all four containers.
  Its `snapshot_behavior` field is `TASK_SNAPSHOT_BEHAVIOR_UNSPECIFIED` for all
  four, so it does not classify these known captures usefully.
- Platform enqueue-to-start times were `623.912`, `802.071`, `332.789`, and
  `675.492` seconds respectively. These include creation, not cached restores.
- The queried layout/task endpoints did not return checkpoint IDs, runtime
  fingerprints, or cache-miss reasons. The `capture_id` values in this document
  are application-generated tracing UUIDs, not Modal checkpoint IDs.

The hardware-capacity split now has a documented potential explanation:
[Modal automatically upgrades some H100 requests to H200](https://modal.com/docs/guide/gpu#automatic-upgrades-to-h200s).
Captures 1 and 4 reported roughly 140 GiB of GPU memory; captures 2 and 3
reported roughly 80 GiB. This is consistent with two H200 and two H100 captures,
but actual device names were not collected. A read-only `nvidia-smi` attempt
could not complete because the fourth container had scaled down; the client
was stopped and no replacement worker was launched.

[Modal's 2–3 snapshot guidance is per GPU type](https://modal.com/docs/guide/memory-snapshots).
Consequently, four distinct captures may still be normal initial coverage
across H100 and H200 plus host differences. This corrects the earlier inference
that a fourth capture necessarily exceeds the expected initial range. It does
not prove the cause or establish cached reuse.

For a controlled future benchmark, the documented `gpu="H100!"` setting
disables automatic H200 upgrades. It would narrow the hardware pool, although
multiple host-specific captures can still be required. Changing it requires a
redeployment and invalidates the existing snapshots, so it was not applied
during this read-only investigation. The alternative is to preserve the four
captures and ask Modal to map their task IDs to internal checkpoint/runtime
fingerprints and explain the cache decisions. No support message was sent.

## Fifth profile-4 cold start — 2026-09-30

- [Probe run](https://modal.com/apps/larshelg/main/ap-NReMZeiksrPkituNy26zoD).
- Confirmed no running containers before submission.
- Container: `ta-01M3SSJWSV8Q5KYBNCF1B31ZAR`.
- New capture: `2981b3e9-41af-4d80-aa22-bac5c77098d9`.
- Restored boot: `838bbf09-8198-4518-a5f7-6d60e52d14c7`.
- Creation started 20:35:16 Europe/Oslo; capture-ready at 20:41:11;
  snapshot created at 20:42:16; container ready at 20:42:37 (about 7m 21s).
- Warmup `256.258` seconds; initialization through capture-ready `347.71`
  seconds. Expected model and accelerator verified; probe succeeded.
- Capture RSS `68,390,014,976` bytes; restored RSS `34,215,428,096` bytes.
  CUDA allocation `34,603,008` bytes; all component weights on CPU.
- GPU capacity `85,017,624,576` bytes. This is the third capture in the ~80 GiB
  group; two others were in the ~140 GiB group. Hardware names remain unverified.
- The run did not reuse an old capture. No user generation was submitted.

## Implemented baseline

- `app.py`: dedicated H100 `WanGPSingularityWorker`, GPU snapshot lifecycle,
  capture/boot diagnostics, request timing, and model-reuse checks.
- `h3_snapshot.py`: private reference-conditioned 4-step warmup through the
  native session, accelerator validation, joined job thread, task/output cleanup,
  and H3 transformer residency checks.
- `snapshot_common.py`: shared memory diagnostics, snapshot logging, model-load
  tracking, LoRA inspection, reference creation, and fresh RNG state after restoration.
- `control.py`: exact-model opt-in using `WANGP_SINGULARITY_SNAPSHOT=1`, recorded
  worker selection, and `snapshot_probe` to initialize/inspect the dedicated pool.
- Focused local tests cover routing isolation, dispatch errors, model rejection,
  successful warmup cleanup, and failed/invalid warmup preventing capture.

The downloaded pinned H3 factory, handler, session API, and CLI job runner were
inspected during implementation. The session's native task path already calls
`wgp.load_models()` and activates the preset LoRA, so no separate manual preload
adapter is needed in this version. Warmup waits for the job thread to exit,
then resets headless task state while retaining the model and offload manager.
WanGP also calls `unload_loras_from_model()` at task completion. The baseline
verifies the accelerator during warmup inference and preserves native adapter
cleanup; its idle snapshot does not retain active acceleration LoRA tensors.
Subsequent requests activate the preset accelerator through the normal path.

Full GPU weight residency is a later experiment. No comparative GPU benchmark or
speedup claim is included in this initial implementation. Operational commands are in
the README's Singularity GPU snapshot section.

## Objective and scope

Reduce cold-start overhead for `minimax_h3_ref2va_singularity_pruned` using a
dedicated Modal worker with GPU memory snapshots. Preserve existing worker
behavior for every other model.

This plan targets `wangpt-modal-app`, dispatched through `control.py`. It does
not change the sibling REST app or its routing. Snapshot compatibility and a
performance improvement must be demonstrated before enabling the new route by
default.

## Fixed model configuration

Use the existing preset in
[`finetunes/minimax_h3_ref2va_singularity_pruned.json`](finetunes/minimax_h3_ref2va_singularity_pruned.json):

- Model: MiniMax H3 Singularity v1.3 Ref2VA Pruned 20B (4-Step).
- Model ID: `minimax_h3_ref2va_singularity_pruned`.
- Base architecture: `minimax_h3_ref2va_pruned`.
- Checkpoint: `Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors`,
  pinned to Hugging Face revision `af671d9214a6e41ab8c2f43e9f871ea56246115f`.
- Built-in accelerator: `minimax_h3_lightx2v_ref2v_turbo_4step_alpha8_v0.1_bf16.safetensors`,
  strength `1.0`, pinned to revision `7b61c8edb895aaf25b248f064e9726ffdcc7ec46`.
- Initial representative warmup: `832x480`, `124` frames, `4` steps, Euler,
  flow shift `12.0`, guidance scale `1.0`, one guidance phase, fixed seed.

Resolve the text encoder, VAEs, and other components through the pinned WanGP
configuration. Record the resolved choices; do not silently switch encoder
quantization or architecture to make the snapshot fit. Additional user LoRAs
remain request-specific initially.

The current WanGP revision is
`2345ae148f82740f66e82c41292dbbdd592e713d`. Implementation must inspect and use
that revision, including the existing H3 latent-continuation integration.

## Findings that determine the design

1. `WanGPRuntime.initialize()` in [`app.py`](app.py) prepares directories,
   symlinks, and the Volume. It does not construct H3.
2. `_session_for()` creates a reusable session using `shared.api.init()` during
   a request. This app embeds WanGP in-process; it does not launch a separate
   inference server that needs HTTP warmup.
3. The pinned `wgp.load_models()` resolves model files and configuration,
   invokes the family handler, and installs MMGP offloading. It returns the
   model and offload manager. WanGP also tracks the selected model, profile,
   configuration, and reload state.
4. The lower-level H3 `model_factory()` constructs pipeline components but
   does not, by itself, establish the complete session, preset LoRA, and
   generation state. The pasted FL2VA example is not the correct configuration
   for this Ref2VA Pruned model.
5. WanGP calls `offloadobj.unload_all()` during generation cleanup. MMGP
   unloads active GPU models. A completed warmup therefore does not prove that
   checkpoint tensors are still resident in VRAM at capture time.
6. Current workers default to memory profile `4` and SDPA. Start with those
   settings for a controlled comparison. A different residency policy is a
   separate, Singularity-only experiment.

The factory, family handler, higher-level loading and cleanup paths, and session
job lifecycle have now been inspected at the pinned WanGP revision. The image's
CUDA/MMGP behavior still requires execution and restore validation on Modal.

## Worker and routing design

Add `WanGPSingularityWorker` in `app.py`, sharing the existing image build and
runtime implementation where possible, with:

- One H100 GPU, initially 128 GiB system memory and one maximum container.
- `min_containers=0` and the existing 300-second scaledown window initially.
- Existing startup and execution timeouts, Volume, and required secrets.
- `enable_memory_snapshot=True`.
- `experimental_options={"enable_gpu_snapshot": True}`.
- A `@modal.enter(snap=True)` hook for initialization and warmup.
- A `@modal.enter(snap=False)` hook only for work that must happen on every
  start/restore, such as refreshing transient state or creating clients that
  should not survive capture.

Keep snapshot flags and lifecycle changes off `WanGPVideoWorker` and
`WanGPImageWorker`. Give the dedicated worker its own resource settings so
snapshot experiments cannot change generic worker behavior.

Extend routing in `control.py` to consider the exact model ID after existing
catalog validation and output-kind resolution:

- Singularity video requests with snapshot routing enabled:
  `WanGPSingularityWorker`.
- Other video requests, and Singularity when snapshot routing is disabled:
  `WanGPVideoWorker`.
- Image/audio requests: `WanGPImageWorker`.

Initially expose an explicit opt-in in the local client, for example
`WANGP_SINGULARITY_SNAPSHOT=1`, defaulting to disabled. This switch is evaluated
by `control.py` at submission time. Enable it only after deploying the new
worker. Preserve job IDs, status handling, validation, output delivery, and
cancellation behavior. The dedicated worker must reject any other model ID.

Do not automatically retry a possibly running generation on another worker.
Rollback routes subsequent submissions to the ordinary video worker; existing
jobs retain their original call IDs and status tracking.

## Initialization and warmup lifecycle

### Before snapshot capture

1. Initialize the filesystem/cache layout using the existing runtime.
2. Create the reusable WanGP session with the existing profile and attention
   settings. Install the current latent-continuation hooks before model loading.
3. Use WanGP's normal model-loading path for the exact Singularity preset.
   Prefer `wgp.load_models()` through a small version-specific adapter if an
   explicit preload is needed. Preserve the returned model/offload manager and
   WanGP's loaded-model bookkeeping, following its native preload behavior.
   A subsequent generation must not discard and reconstruct the preloaded model.
4. Run a representative task through the session API so the preset accelerator,
   text encoding, denoising, and audio/video decoding paths are exercised.
   Verify that the task succeeds. Choose valid inputs for the pinned Ref2VA
   implementation; a reference-image path should also be exercised before
   declaring reference-conditioned requests warm.
5. Wait for job completion and any associated background work to finish. Clear
   temporary task payloads, callbacks, outputs, and request-specific state
   without closing the session or releasing the loaded model.
6. Measure the actual post-warmup state. Record CPU memory, CUDA allocated and
   reserved memory, MMGP active model IDs, and component tensor residency.
   CUDA availability or allocator totals alone do not establish model residency.
7. For a GPU-resident capture experiment, establish the intended residency using
   MMGP-aware loading and explicit memory budgets. Validate headroom for the
   first real request. Do not globally disable `unload_all()`, blindly move the
   pipeline with `.to("cuda")`, or assume all components fit simultaneously.
8. Commit any newly downloaded cache assets as needed, finish file writes and
   CUDA work, and leave the session idle before returning from `snap=True`.

Create a separate warmup routine. Do not call `WanGPRuntime.run()` with a fake
job: that method changes the job store and publishes outputs to S3. Warmup must
not create user-visible jobs or artifacts.

### After restoration

- Reuse the captured session and model state.
- Reinitialize transient connections if any were created before capture. Keep
  the existing per-request S3 client creation outside the snapshot warmup.
- Ensure queues, locks, callbacks, and latent-plugin state are ready for a fresh
  request, with no unfinished warmup task.
- Preserve explicit seeds and verify that random-seed requests do not repeat
  merely because RNG state was captured.
- Run real requests through the existing runtime, including artifact upload,
  verification, cleanup, and terminal job recording.

## Capture strategies to compare

First establish a snapshot of the initialized and warmed session with native
offloading intact. It may retain mostly CPU model state plus initialized CUDA
and kernel state. This is a valid baseline, but must not be described as H3
already fully loaded in VRAM.

Then test deliberate GPU residency for the most useful components. Retaining
the transformer alone may not avoid initial transfers if text encoding causes
MMGP to evict it. Verify what the first real request actually does, including
component transitions and changes in request configuration.

Keep profile changes, residency adjustments, and compilation experiments scoped
to the dedicated worker. Avoid introducing `torch.compile` in the first version
solely for snapshotting. If later enabled, validate its snapshot compatibility
separately.

## Implementation order

1. Inspect the pinned H3 handler, factory, session lifecycle, native preload,
   LoRA activation, and MMGP cleanup paths in the built image. Identify a safe
   preload adapter and the correct idle boundary for capture.
2. Add a dedicated-worker configuration and a warmup routine with stage timing
   and memory/residency diagnostics. Preserve normal runtime semantics.
3. Add the snapshot-enabled class and an explicit model guard.
4. Add opt-in exact-model routing and record the selected worker in diagnostics.
5. Add focused tests for routing, disabled-switch fallback, worker rejection of
   other models, warmup failure handling, and absence of warmup job/S3 writes.
6. Deploy the worker and run the GPU validation below before changing the
   default route. A successful local test cannot establish CUDA restore safety.

## Validation and acceptance criteria

Compare the same request and resolved configuration in three cases:

1. Ordinary worker cold start with checkpoint assets already cached.
2. New container restored from a GPU snapshot.
3. Already-warm worker.

Measure submission-to-first-denoising-step and submission-to-completion, along
with session initialization, model construction, warmup, GPU residency, and peak
memory. Distinguish scheduling delay from worker initialization. Record snapshot
creation separately from restoration; the first invocation is not a restore
benchmark. Repeat restored starts sufficiently to detect failures and variable
startup times, reporting sample counts rather than unsupported latency claims.

Verify all of the following:

- Modal's container view/logs confirm actual snapshot creation and restoration.
- Restored requests reuse the intended model without reconstructing it.
- The built-in acceleration LoRA is active, with expected video/audio output.
- Reference-image generation, existing latent save/continuation, explicit seeds,
  random seeds, and multiple consecutive requests work after restore.
- Additional LoRA/configuration requests behave correctly; any required reload
  is observable and does not silently substitute the captured configuration.
- Warmup state and files do not leak into jobs or outputs.
- S3 delivery, job status, failure handling, and cancellation remain correct.
- Other H3 models, Krea, and image/audio routes continue to use existing workers.
- No snapshot/restore failures, memory regressions, or output regressions appear
  in the tested cases.
- Restored cold requests show a material measured benefit before enabling the
  route by default. Define the desired latency target from the baseline data.

Snapshot generation requires a deployed Modal app. The local `modal run
control.py::...` client can invoke that deployed worker, but testing a new worker
only in an ephemeral `modal run` app does not validate snapshot creation.

## Versioning, rollback, and later models

Treat the snapshot as tied to the checkpoint and accelerator revisions, resolved
encoder/configuration, WanGP/MMGP and plugin versions, attention mode, memory
profile, GPU type, and warmup implementation. Encode a snapshot revision in
worker code/configuration and advance it when these inputs change.

Modal Volume changes alone do not invalidate memory snapshots. Preserve files
referenced by captured state and redeploy a changed worker configuration when
cached model assets change. Avoid Volume reloads while checkpoint files remain
open; the current runtime already documents this constraint.

Rollback disables Singularity snapshot routing for new submissions. The existing
video worker remains available throughout the experiment.

A second H3 snapshot can later use another dedicated class or an allowlisted
`modal.parameter()` variant. Start with one fixed class to keep the experiment
bounded. Each new class/pool adds independent capacity; its container cap does
not share the generic video's cap. Modal may create multiple physical snapshots
for different underlying worker types, so "one snapshot per model" describes
the logical configuration rather than a guaranteed single snapshot artifact.

## Limits and open questions

- GPU memory snapshots are currently an alpha Modal feature.
- Compatibility of this exact H3/INT8/MMGP/plugin combination is untested.
- Construction, CUDA initialization, and kernel preparation may benefit, but
  checkpoint storage bandwidth may dominate. Modal cautions that snapshots can
  provide no benefit or add overhead for storage-bound loading.
- The best component residency policy and system/VRAM requirements need direct
  measurement. Keep enough capacity for activations and longer requests.
- Snapshot warmup covers specific execution paths; other resolutions, reference
  types, LoRAs, or configuration choices may still incur initialization work.
- No speedup or production readiness is claimed until restored generation has
  been measured and validated.

## Sources

- [Modal memory snapshots: lifecycle, GPU flags, limitations, deployment, and invalidation](https://modal.com/docs/guide/memory-snapshots)
- [Modal parameterized functions and independent container pools](https://modal.com/docs/guide/parametrized-functions)
- [Pinned WanGP loading, bookkeeping, and generation cleanup](https://github.com/deepbeepmeep/Wan2GP/blob/2345ae148f82740f66e82c41292dbbdd592e713d/wgp.py)
- [Pinned WanGP session API](https://github.com/deepbeepmeep/Wan2GP/blob/2345ae148f82740f66e82c41292dbbdd592e713d/shared/api.py)
- [Upstream H3 factory; verify against the deployed pin before implementation](https://github.com/deepbeepmeep/Wan2GP/blob/main/models/minimax_h3/minimax_h3_main.py)
- [MMGP offload implementation; verify against the installed version](https://github.com/deepbeepmeep/mmgp/blob/main/src/mmgp/offload.py)
