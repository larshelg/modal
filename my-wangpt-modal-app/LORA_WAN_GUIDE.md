# Adding a LoRA to Wan2GP

This app stores LoRAs persistently in the Modal Volume named `wangp-data`.
Wan2GP sees that Volume at `/data`, and `app.py` starts Wan2GP with
`--loras /data/loras`.

## 1. Check compatibility

Before uploading a LoRA, check the model card or download page for:

- the Wan generation it targets, such as Wan 2.1 or Wan 2.2;
- the model size, such as 1.3B, 5B, or 14B;
- the task, especially text-to-video (T2V) versus image-to-video (I2V);
- any trigger words, recommended multiplier, inference-step settings, or
  accelerator settings.

A LoRA trained for a different base model, size, or task may fail to load or
produce poor results. Wan2GP primarily uses `.safetensors` LoRA files.

## 2. Choose the Wan2GP directory

Upload the file below the `loras/` root in the `wangp-data` Volume:

| LoRA target | Path in the Volume | Path inside the app |
| --- | --- | --- |
| Wan T2V 14B/general | `loras/wan/` | `/data/loras/wan/` |
| Wan I2V | `loras/wan_i2v/` | `/data/loras/wan_i2v/` |
| Wan 5B | `loras/wan_5B/` | `/data/loras/wan_5B/` |
| Wan 1.3B | `loras/wan_1.3B/` | `/data/loras/wan_1.3B/` |

Do not put a Wan LoRA directly in `loras/`; use the matching subdirectory.

## 3. Upload the file

Run this from the project directory. Replace the local filename and destination
directory as appropriate:

```bash
python3 -m modal volume put \
  wangp-data \
  /path/to/my-lora.safetensors \
  loras/wan/
```

The trailing `/` makes the remote destination a directory and preserves the
local filename. Add `--force` before `wangp-data` if you intentionally want to
overwrite an existing file:

```bash
python3 -m modal volume put --force \
  wangp-data \
  /path/to/my-lora.safetensors \
  loras/wan/
```

Verify the upload:

```bash
python3 -m modal volume ls wangp-data loras/wan
```

## 4. Make the LoRA visible to Wan2GP

The simplest workflow is to upload LoRAs before starting or deploying the app.
A newly started container mounts the latest `wangp-data` snapshot.

If Wan2GP is already running, an external Volume upload is not automatically
visible in that container. Restart/recreate the running app container, then
reconnect to its MCP endpoint. Once the file is visible, use Wan2GP's **Refresh**
button in the LoRA section if the interface still shows a cached list.

Redeploying is also appropriate when a new container should be started:

```bash
python3 -m modal deploy app.py
```

## 5. Activate the LoRA

In the Wan2GP generation settings:

1. Select a compatible Wan model and generation task.
2. Open **Advanced**, then the **LoRAs** section.
3. Select the uploaded LoRA under **Activated LoRAs**.
4. Enter its strength under **LoRA Multipliers**. Start with the value from the
   LoRA's model card; otherwise, `1.0` is a reasonable initial test.
5. Add any required trigger word to the prompt.
6. Generate a short test before committing to a long render.

When multiple LoRAs are selected, enter their multipliers in the same order,
separated by spaces. For example, two selected LoRAs at strengths 0.8 and 1.1:

```text
0.8 1.1
```

For Wan 2.2's high-noise and low-noise phases, separate the two phase strengths
with a semicolon. For example, enable a LoRA only in the low-noise phase:

```text
0;1
```

Use phase-specific values only when the LoRA or workflow recommends them.

## Troubleshooting

- **The LoRA is not listed:** verify its Volume subdirectory, restart the
  container if the file was uploaded while it was running, and refresh the
  LoRA list.
- **Loading fails:** confirm the LoRA's Wan generation, size, and T2V/I2V task
  match the selected base model.
- **The result is too weak or too strong:** adjust the multiplier gradually and
  check that the prompt contains the documented trigger word.
- **Wan 2.2 behaves incorrectly:** check whether the LoRA targets the high-noise
  model, low-noise model, or both, and use phase-specific multipliers if needed.
- **An accelerator LoRA gives poor output:** copy all recommended inference
  steps, guidance, sampler, and multiplier settings; accelerator LoRAs usually
  depend on those settings as a group.

## References

- [Wan2GP LoRA guide](https://github.com/deepbeepmeep/Wan2GP/blob/main/docs/LORAS.md)
- [Modal Volume CLI](https://modal.com/docs/cli/latest/volume)
- [Modal Volume commits and reloads](https://modal.com/docs/guide/volumes#volume-commits-and-reloads)
