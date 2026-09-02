# Krea2 Turbo request parameters

`krea2_turbo` is an image model. The dispatcher therefore routes it to
`WanGPImageWorker`, which uses the configured image GPU (`L40S` by default).

## JSON CLI

The help entrypoint prints the complete JSON contract, defaults, and examples:

```bash
python3 -m modal run control.py::krea_help
```

Pass the complete native WanGP object as an inline JSON string:

```bash
python3 -m modal run control.py::krea \
  --params-json '{"prompt":"A red fox walking through fresh snow","seed":-1}'
```

Only `prompt` is required. Omitted settings use WanGP defaults. The Krea
entrypoint fixes the model to `krea2_turbo` and the route to `image`, so neither
belongs in the JSON object.

The `--params-json` string contains only native WanGP settings. The Krea model
name and output kind are fixed by the entrypoint, not fields in this JSON
object.

## Minimal request

WanGP loads the model defaults before applying the supplied parameters, so the
smallest useful JSON object is:

```json
{
  "prompt": "A red fox walking through fresh snow at golden hour"
}
```

Submit it with:

```bash
python3 -m modal run control.py::krea \
  --params-json '{"prompt":"A red fox walking through fresh snow at golden hour"}'
```

## Recommended starting request

The checked-in [example](../examples/krea2_turbo.json) expands to:

```json
{
  "prompt": "A red fox walking through fresh snow at golden hour",
  "negative_prompt": "blurry, low quality",
  "resolution": "1024x1024",
  "num_inference_steps": 8,
  "seed": 12345,
  "batch_size": 1,
  "guidance_scale": 0,
  "flow_shift": 5.0
}
```

These values are a baseline, not a required full payload. In particular, Turbo
is configured for 8 inference steps and guidance scale 0. Increasing those
values does not necessarily improve the distilled model.

## Common fields

- `prompt` (`string`): positive generation prompt.
- `negative_prompt` (`string`): content or qualities to avoid. It can be empty.
- `resolution` (`string`): output dimensions in `WIDTHxHEIGHT` form. The normal
  Krea2 Turbo default is `1024x1024`; consult the schema for supported choices.
- `num_inference_steps` (`integer`): denoising steps. Krea2 Turbo defaults to
  `8`.
- `seed` (`integer`): reproducible seed. Use `-1` for a random seed.
- `batch_size` (`integer`): number of images produced by the task. It defaults
  to `1`.
- `guidance_scale` (`number`): classifier-free guidance scale. Krea2 Turbo
  defaults to `0`.
- `flow_shift` (`number`): flow-matching shift. The current starting value is
  `5.0`.
- `spatial_upsampling`: optional WanGP upscaling mode. Use the schema-reported
  value; omit it to retain the model default.

## LoRAs

LoRAs use two aligned settings:

```json
{
  "prompt": "linda standing in a sunlit photography studio",
  "activated_loras": [
    "/data/loras/krea2/linda_krea2_v1.safetensors"
  ],
  "loras_multipliers": "0.8"
}
```

- `activated_loras` contains the LoRA paths understood by WanGP.
- `loras_multipliers` contains matching weights in the same order.
- Absolute paths must be under `/data`; other absolute paths are rejected by
  the dispatcher.
- A trained identity trigger word belongs in `prompt` when that LoRA expects
  one.

For multiple LoRAs, inspect the generated schema and an existing WanGP profile
before composing `loras_multipliers`; WanGP may encode model- or phase-specific
weights in this string.

## Inpainting controls

Krea2 exposes masked denoising and LanPaint modes through native WanGP fields:

- `masking_strength`
- `denoising_strength`
- `model_mode`

LanPaint `model_mode` values are:

- `2`: 2 steps, easy task.
- `3`: 5 steps, medium task.
- `4`: 10 steps, hard task.
- `5`: 15 steps, very hard task.

Image and mask input fields are catalog-version dependent. Use the deployed
schema described below before constructing an inpainting request.

## Authoritative defaults and schema

The checked-in example is intentionally stable, while the deployed catalog is
generated from the exact pinned WanGP runtime during the Modal image build.
Inspect it whenever you need every accepted field or current choice list:

```bash
python3 -m modal run control.py::defaults --model krea2_turbo
python3 -m modal run control.py::schema --model krea2_turbo
```

Defaults are a starting payload, not an exhaustive allowlist. The schema is the
authoritative source for capabilities and supported choices in the deployed
version.

Do not include these fields in the parameters file:

- `model` or `model_type`: select the model with `--model krea2_turbo`.
- `kind`: pass it as `--kind image`, or let the dispatcher infer it.
- `_api`: reserved for the WanGP runtime and rejected by this app.
