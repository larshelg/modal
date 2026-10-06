"""Model-independent snapshot diagnostics and lifecycle helpers.

GPU and image dependencies are imported only when a helper runs in a worker.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import json
import random
import secrets
from pathlib import Path
from typing import Any


def active_lora_files(context: Any) -> list[str]:
    if context is None:
        return []
    transformer = context.offloadobj.models["transformer"]
    active = list(getattr(transformer, "_loras_active_adapters", []))
    adapters = getattr(transformer, "_loras_adapters", {})
    return [Path(str(adapters.get(name, ""))).name for name in active]


def memory_report(wgp: Any) -> dict[str, Any]:
    import psutil
    import torch

    torch.cuda.synchronize()
    components = {}
    for name, model in wgp.offloadobj.models.items():
        devices: dict[str, int] = {}
        # Logical tensor elements, not physical storage bytes: quantized tensors
        # may wrap packed storage and multiple components can share parameters.
        for tensor in (*model.parameters(), *model.buffers()):
            device = str(tensor.device)
            devices[device] = devices.get(device, 0) + tensor.numel()
        components[name] = devices
    free, total = torch.cuda.mem_get_info()
    return {
        "cpu_rss_bytes": psutil.Process().memory_info().rss,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_free_bytes": free,
        "gpu_total_bytes": total,
        "gpu_name": torch.cuda.get_device_name(),
        "mmgp_active_models": list(wgp.offloadobj.active_models_ids),
        "component_logical_elements_by_device": components,
    }


def create_reference(path: Path) -> None:
    from PIL import Image, ImageDraw

    with Image.new("RGB", (832, 480), (220, 220, 220)) as image:
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 300, 831, 479), fill=(40, 95, 160))
        draw.ellipse((330, 140, 490, 300), fill=(190, 35, 35))
        image.save(path)


@contextmanager
def track_model_loads(wgp: Any):
    """Observe native model construction, not stage-specific CPU/GPU transfers."""
    original = wgp.load_models
    counts = {"load_models_calls": 0}

    @wraps(original)
    def tracked(*args: Any, **kwargs: Any):
        counts["load_models_calls"] += 1
        return original(*args, **kwargs)

    wgp.load_models = tracked
    try:
        yield counts
    finally:
        wgp.load_models = original


def reseed_after_restore() -> None:
    """Avoid replaying the captured random-seed sequence on every new container."""
    import numpy as np
    import torch

    random.seed(secrets.randbits(128))
    np.random.seed(secrets.randbits(32))
    torch.manual_seed(secrets.randbits(63))


def emit_snapshot(stage: str, **details: Any) -> None:
    print("[wangp-snapshot] " + json.dumps({"stage": stage, **details}, sort_keys=True), flush=True)
