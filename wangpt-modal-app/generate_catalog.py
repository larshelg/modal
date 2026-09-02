"""Generate a JSON-safe WanGP discovery catalog during a Modal image build."""

from __future__ import annotations

import json
import os
from pathlib import Path


WAN_ROOT = Path(os.environ.get("WAN2GP_ROOT", "/opt/Wan2GP"))
CATALOG_PATH = Path(os.environ.get("WANGP_CATALOG_PATH", "/opt/wangp-catalog.json"))


def main() -> None:
    # WanGP's attention registry probes compute capability at module-import
    # time, before it knows that this process only needs static model data.
    # Supply the deployed L40S capability for this build-only catalog process.
    import torch

    torch.cuda.get_device_capability = lambda device=None: (8, 9)
    torch.cuda.current_device = lambda: 0
    torch.cuda.device_count = lambda: 1
    torch.cuda.get_device_name = lambda device=None: "NVIDIA L40S"
    from shared.api import init

    session = init(
        root=WAN_ROOT,
        cli_args=["--profile", "4", "--attention", "sdpa"],
        console_output=False,
    )
    try:
        models = session.list_model_metadata(include_availability=False)
        defaults = {}
        schemas = {}
        for model in models:
            model_type = model["model_type"]
            defaults[model_type] = session.get_default_settings(model_type)
            schemas[model_type] = session.get_model_schema(model_type)
        payload = {"models": models, "defaults": defaults, "schemas": schemas}
        CATALOG_PATH.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        print(f"Wrote {len(models)} models to {CATALOG_PATH}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
