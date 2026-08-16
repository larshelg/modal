 {
    "source": {
      "model_type": "krea2_turbo",
      "prompt": "Your image prompt",
      "resolution": "1024x1024",
      "num_inference_steps": 8,
      "guidance_scale": 0,
      "activated_loras": [
        "Krea2_TextFusion_Refusal_Reduction.safetensors"
      ],
      "loras_multipliers": "1.0"
    },
    "wait": true,
    "timeout_s": 300,
    "event_limit": 50
  }

  {
    "source": {
      "settings_version": 2.73,
      "model_type": "krea2_turbo_edit",
      "prompt": "Edit only the fox's face: replace the fox face with a realistic human face naturally integrated at the same position and scale. Keep the fox
      body, fur outside the face, pose, composition, camera angle, snowy pine forest, sunrise lighting, colors, depth of field, and every background detail
      unchanged.",
      "resolution": "1024x1024",
      "flow_shift": 5,
      "image_mode": 1,
      "batch_size": 1,
      "model_mode": 0,
      "denoising_strength": 1,
      "masking_strength": 1,
      "num_inference_steps": 8,
      "guidance_scale": 0,
      "video_prompt_type": "KI",
      "remove_background_images_ref": 0,
      "negative_prompt": "",
      "alt_prompt": "",
      "multi_prompts_gen_type": "PG",
      "postprocess_audio": "",
      "replace_voice_method": "",
      "replace_voice_sample": null,
      "replace_voice_sample2": null,
      "image_prompt_type": "",
      "image_refs": [
        "/data/outputs/2026-08-11-19h57m37s_seed258877949_A red fox sitting in a snowy pine forest at sunrise, natural light, detailed fur,
        photorealistic.jpg"
      ]
    },
    "wait": true,
    "timeout_s": 600,
    "event_limit": 30
  }