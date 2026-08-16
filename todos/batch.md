# WanGP MCP batch image generation

The api supports supports two forms of batch generation:

1. Use `batch_size` to create multiple variants from one prompt and one shared configuration.
2. Pass a list of task objects in `source` to use different prompts, LoRAs, or settings for each image.

## Multiple variants with the same settings

Set `batch_size` when every output should use the same prompt, model, LoRAs, and generation settings.

```json
{
  "source": {
    "model_type": "krea2_turbo",
    "prompt": "A red fox walking through fresh snow",
    "resolution": "1024x1024",
    "num_inference_steps": 2,
    "guidance_scale": 0,
    "batch_size": 4
  },
  "wait": true,
  "timeout_s": 600,
  "event_limit": 100
}
```

All four outputs share the same LoRA configuration. `batch_size` cannot assign a different LoRA to each image.

## Different prompts or LoRAs per image

Pass an array in `source`. Each array item is an independent task and can select its own prompt, LoRAs, multipliers, and other settings.

```json
{
  "source": [
    {
      "model_type": "krea2_turbo",
      "prompt": "A red fox walking through fresh snow",
      "resolution": "1024x1024",
      "num_inference_steps": 2,
      "guidance_scale": 0,
      "batch_size": 1,
      "activated_loras": [
        "first-lora.safetensors"
      ],
      "loras_multipliers": "1.0"
    },
    {
      "model_type": "krea2_turbo",
      "prompt": "A wolf beneath the northern lights",
      "resolution": "1024x1024",
      "num_inference_steps": 2,
      "guidance_scale": 0,
      "batch_size": 1,
      "activated_loras": [
        "second-lora.safetensors"
      ],
      "loras_multipliers": "0.8"
    }
  ],
  "wait": true,
  "timeout_s": 600,
  "event_limit": 100
}
```

## Multiple LoRAs on one task

List every LoRA in `activated_loras`. Supply its matching strength in `loras_multipliers`, in the same order.

```json
{
  "model_type": "krea2_turbo",
  "prompt": "A cinematic winter wildlife portrait",
  "num_inference_steps": 2,
  "activated_loras": [
    "style-lora.safetensors",
    "detail-lora.safetensors"
  ],
  "loras_multipliers": "0.8,1.0"
}
```

Confirm the multiplier format against the active WanGP model/profile when using several LoRAs; WanGP exposes this setting as a string.

## Choosing the batch form

| Requirement | Use |
|---|---|
| Same prompt and LoRAs, several variants | One task with `batch_size > 1` |
| Different prompts | A task list in `source` |
| Different LoRAs per image | A task list in `source` |
| Several LoRAs on one image | Multiple `activated_loras` in that task |

## Performance notes

- Tasks with different LoRAs may run sequentially and incur LoRA switching or loading overhead.
- A warm `krea2_turbo` container avoids the large Modal cold-start cost.
- Krea 2 Turbo is intended for few-step generation. Use `num_inference_steps: 2` for the initial speed benchmark, then increase it only if image quality requires it.
- Set `timeout_s` high enough for the complete batch, especially from a cold start.
- `wait: true` returns after all requested tasks finish; use the returned job ID with `wangp_get_job` when submitting asynchronously.
