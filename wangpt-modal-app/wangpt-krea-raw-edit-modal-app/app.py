"""Dedicated Krea2 RAW Edit L40S snapshot worker, deployed independently."""

from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Any

import modal

import app as service
from snapshot_common import emit_snapshot, memory_report, reseed_after_restore, track_model_loads
from .snapshot import (
    KREA_RAW_EDIT_APP_NAME, KREA_RAW_EDIT_MODEL, SNAPSHOT_REVISION, loaded_state,
    make_transformer_resident, require_krea_raw_edit, validate_residency, warm_session,
)

krea_raw_edit_worker_image = (
    service.gpu_image
    .add_local_python_source("app", "snapshot_common", "wangpt-krea-raw-modal-app", "wangpt-krea-raw-edit-modal-app", copy=True)
    .env({"WANGP_WORKER_KIND": "krea-raw-edit-snapshot"})
)
app = modal.App(KREA_RAW_EDIT_APP_NAME)


@app.cls(
    image=krea_raw_edit_worker_image,
    gpu="L40S",
    memory=int(os.environ.get("WANGP_KREA_RAW_EDIT_MEMORY_MB", "65536")),
    min_containers=0,
    max_containers=int(os.environ.get("WANGP_KREA_RAW_EDIT_MAX_CONTAINERS", "1")),
    scaledown_window=int(os.environ.get("WANGP_KREA_RAW_EDIT_SCALEDOWN_WINDOW", "300")),
    startup_timeout=service.STARTUP_TIMEOUT,
    timeout=24 * 60 * 60,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    volumes={str(service.DATA_ROOT): service.data_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),
        service.studio_s3_secret,
    ],
)
class WanGPKreaRawEditWorker:
    @modal.enter(snap=True)
    def prepare_snapshot(self) -> None:
        started = time.monotonic()
        self.capture_id = str(uuid.uuid4())
        self.snapshot_revision = SNAPSHOT_REVISION
        emit_snapshot("krea_raw_edit_initialize", capture_id=self.capture_id, revision=SNAPSHOT_REVISION)
        self.runtime = service.WanGPRuntime("1")
        self.runtime.initialize()
        session = self.runtime._session_for(KREA_RAW_EDIT_MODEL)
        wgp = sys.modules["wgp"]
        self.warmup = warm_session(session, wgp, service.GENERATED_OUTPUT_ROOT)
        service.data_volume.commit()
        self.transformer_residency = make_transformer_resident(session, wgp)
        self.capture_memory = memory_report(wgp)
        validate_residency(self.capture_memory)
        self.warmup["initialize_seconds"] = round(time.monotonic() - started, 3)
        emit_snapshot("krea_raw_edit_capture_ready", capture_id=self.capture_id, **self.capture_memory)

    @modal.enter(snap=False)
    def after_restore(self) -> None:
        wgp = sys.modules["wgp"]
        self.restore_memory = memory_report(wgp)
        validate_residency(self.restore_memory)
        reseed_after_restore()
        self.boot_id = str(uuid.uuid4())
        self.requests_served = 0
        self.last_request = None
        emit_snapshot("krea_raw_edit_restore_residency", capture_id=self.capture_id,
                      boot_id=self.boot_id, **self.restore_memory)
        emit_snapshot("krea_raw_edit_container_ready", capture_id=self.capture_id,
                      boot_id=self.boot_id, revision=self.snapshot_revision, **loaded_state(wgp))

    @modal.method()
    def snapshot_info(self) -> dict[str, Any]:
        return {
            "revision": self.snapshot_revision, "capture_id": self.capture_id,
            "boot_id": self.boot_id, "requests_served": self.requests_served,
            "capture_memory": self.capture_memory, "restore_memory": self.restore_memory,
            "current_memory": memory_report(sys.modules["wgp"]),
            "transformer_residency": self.transformer_residency, "warmup": self.warmup,
            "loaded": loaded_state(sys.modules["wgp"]), "last_request": self.last_request,
        }

    @modal.method()
    def run(self, job_id: str, model: str, params: dict[str, Any]) -> dict[str, Any]:
        require_krea_raw_edit(model)
        started = time.monotonic()
        wgp = sys.modules["wgp"]
        previous_model = wgp.wan_model
        counts = {"load_models_calls": 0}
        emit_snapshot("krea_raw_edit_request_start", job_id=job_id, boot_id=self.boot_id)
        try:
            with track_model_loads(wgp) as counts:
                return self.runtime.run(job_id, model, params)
        finally:
            self.requests_served += 1
            self.last_request = {
                "job_id": job_id, "elapsed_seconds": round(time.monotonic() - started, 3),
                "model_reused": wgp.wan_model is previous_model, **counts,
            }
            emit_snapshot("krea_raw_edit_request_end", boot_id=self.boot_id, **self.last_request)
