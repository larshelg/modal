import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import control

from app import (
    JobCallbacks,
    canonical_output_path,
    download_s3_input,
    materialize_s3_image_inputs,
    parse_s3_uri,
    remove_local_outputs,
    s3_settings_from_env,
    serialize_result,
    upload_output_artifacts,
)
from control import generation_worker_name, load_params_file, load_params_json
from wangpt_common import (
    JOB_DICT_NAME,
    filter_models,
    normalized_absolute_path,
    resolve_generation_kind,
    validate_data_paths,
    validate_job_request,
)


def test_job_callbacks_cancel_load_trace_on_first_progress_only(monkeypatch):
    records = {"job": {"id": "job", "status": "running"}}

    class Store:
        def get(self, key, default=None):
            return records.get(key, default)

        def put(self, key, value):
            records[key] = value

    monkeypatch.setattr("app.job_store", Store())
    calls = []
    callbacks = JobCallbacks("job", on_first_progress=lambda: calls.append("cancel"))
    progress = SimpleNamespace(
        phase="inference",
        status="Denoising",
        progress=25,
        current_step=1,
        total_steps=20,
    )

    callbacks.on_progress(progress)
    callbacks.on_progress(progress)

    assert calls == ["cancel"]
    assert records["job"]["progress"]["phase"] == "inference"


def test_request_overlays_model_type():
    assert validate_job_request(" qwen_image ", {"prompt": "fox"}) == {
        "model_type": "qwen_image",
        "prompt": "fox",
    }


def test_request_rejects_reserved_api_metadata():
    with pytest.raises(ValueError, match="reserved"):
        validate_job_request("qwen_image", {"_api": {"return_media": True}})


def test_data_paths_accept_data_volume_and_virtual_suffix():
    validate_data_paths({"video_guide": "/data/inputs/a.mp4|start_frame=1,end_frame=2"})


def test_data_paths_accept_s3_uri_for_worker_materialization():
    validate_data_paths({"image_start": "s3://bucket/inputs/start.png"})


def test_data_paths_reject_outside_volume():
    with pytest.raises(ValueError, match="under /data"):
        validate_data_paths({"image_start": "/tmp/a.png"})


def test_data_paths_reject_parent_traversal():
    with pytest.raises(ValueError, match="under /data"):
        validate_data_paths({"image_start": "/data/../etc/passwd"})


def test_normalized_path_does_not_resolve_volume_mount_symlinks(tmp_path: Path):
    mounted = tmp_path / "data"
    target = tmp_path / "internal-volume"
    target.mkdir()
    mounted.symlink_to(target, target_is_directory=True)
    path = normalized_absolute_path(mounted / "outputs" / "image.png")
    assert str(path).startswith(str(mounted))


def test_canonical_output_path_rejects_modal_volume_output():
    with pytest.raises(ValueError, match="not under"):
        canonical_output_path("/__modal/volumes/vo-example/outputs/image.png")


def test_result_is_metadata_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_dir = tmp_path / "wangp-outputs"
    output_dir.mkdir()
    output = output_dir / "out.png"
    output.write_bytes(b"png")
    monkeypatch.setattr("app.GENERATED_OUTPUT_ROOT", output_dir)
    result = SimpleNamespace(
        success=True,
        generated_files=[str(output)],
        total_tasks=1,
        successful_tasks=1,
        failed_tasks=0,
        errors=[],
    )
    metadata = {
        "storage": "s3",
        "bucket": "bucket",
        "key": "runninghub/wangp/job/000-out.png",
        "uri": "s3://bucket/runninghub/wangp/job/000-out.png",
        "filename": "out.png",
        "size_bytes": 3,
        "media_type": "image/png",
        "sha256": "digest",
    }
    assert serialize_result(result, [metadata]) == {
        "success": True,
        "outputs": [metadata],
        "total_tasks": 1,
        "successful_tasks": 1,
        "failed_tasks": 0,
        "errors": [],
    }


def test_serialize_output_hides_internal_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output_dir = tmp_path / "wangp-outputs"
    output_dir.mkdir()
    output = output_dir / "clip.mp4"
    output.write_bytes(b"video")
    monkeypatch.setattr("app.GENERATED_OUTPUT_ROOT", output_dir)

    from app import serialize_output

    assert serialize_output(str(output)) == {
        "filename": "clip.mp4",
        "size_bytes": 5,
        "media_type": "video/mp4",
    }


def test_s3_settings_require_all_keys_without_exposing_values():
    values = {
        "S3_ACCESS_KEY_ID": "id",
        "S3_SECRET_ACCESS_KEY": "do-not-print",
        "S3_ENDPOINT": "https://s3.invalid",
        "S3_BUCKET": "bucket",
    }
    with pytest.raises(RuntimeError, match="S3_REGION") as caught:
        s3_settings_from_env(values)
    assert values["S3_SECRET_ACCESS_KEY"] not in str(caught.value)


def test_s3_input_uri_is_limited_to_configured_bucket():
    assert parse_s3_uri(
        "s3://studio-bucket/inputs/start%20frame.png",
        "studio-bucket",
    ) == ("studio-bucket", "inputs/start frame.png")

    with pytest.raises(ValueError, match="configured S3_BUCKET"):
        parse_s3_uri("s3://other-bucket/start.png", "studio-bucket")
    with pytest.raises(ValueError, match="query"):
        parse_s3_uri("s3://studio-bucket/start.png?versionId=1", "studio-bucket")


class FakeInputS3:
    def __init__(self, objects):
        self.objects = objects
        self.downloads = []

    def head_object(self, Bucket, Key):
        item = self.objects[(Bucket, Key)]
        return {
            "ContentLength": item.get("reported_size", len(item["body"])),
            "Metadata": item.get("metadata", {}),
        }

    def download_fileobj(self, bucket, key, stream):
        self.downloads.append((bucket, key))
        stream.write(self.objects[(bucket, key)]["body"])


def test_materialize_s3_image_inputs_downloads_supported_fields(tmp_path):
    start = b"start-image"
    reference = b"reference-image"
    client = FakeInputS3(
        {
            ("bucket", "inputs/start image.png"): {
                "body": start,
                "metadata": {"sha256": hashlib.sha256(start).hexdigest()},
            },
            ("bucket", "inputs/reference.jpg"): {"body": reference},
        }
    )
    params = {
        "image_start": "s3://bucket/inputs/start%20image.png",
        "image_refs": ["s3://bucket/inputs/reference.jpg"],
        "prompt": "A literal s3://bucket/example string is not an image parameter",
    }

    settings, input_dir, count = materialize_s3_image_inputs(
        params,
        "job-id",
        client,
        "bucket",
        root=tmp_path,
    )

    assert count == 2
    assert input_dir is not None
    assert Path(settings["image_start"]).read_bytes() == start
    assert Path(settings["image_start"]).suffix == ".png"
    assert Path(settings["image_refs"][0]).read_bytes() == reference
    assert settings["prompt"] == params["prompt"]
    assert params["image_start"] == "s3://bucket/inputs/start%20image.png"
    assert client.downloads == [
        ("bucket", "inputs/start image.png"),
        ("bucket", "inputs/reference.jpg"),
    ]


def test_download_s3_input_removes_partial_on_verification_failure(tmp_path):
    client = FakeInputS3(
        {("bucket", "start.png"): {"body": b"image", "reported_size": 6}}
    )
    destination = tmp_path / "start.png"

    with pytest.raises(ValueError, match="size verification"):
        download_s3_input(
            client,
            "s3://bucket/start.png",
            "bucket",
            destination,
        )

    assert not destination.exists()
    assert not (tmp_path / "start.png.part").exists()


def test_upload_outputs_returns_verified_s3_uri_then_local_file_can_be_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_dir = tmp_path / "wangp-outputs"
    output_dir.mkdir()
    output = output_dir / "clip.mp4"
    output.write_bytes(b"video")
    monkeypatch.setattr("app.GENERATED_OUTPUT_ROOT", output_dir)

    class FakeS3:
        def __init__(self):
            self.objects = {}

        def upload_file(self, path, bucket, key, ExtraArgs):
            self.objects[(bucket, key)] = {
                "body": Path(path).read_bytes(),
                "metadata": ExtraArgs["Metadata"],
                "content_type": ExtraArgs["ContentType"],
            }

        def head_object(self, Bucket, Key):
            item = self.objects[(Bucket, Key)]
            return {
                "ContentLength": len(item["body"]),
                "Metadata": item["metadata"],
            }

    result = SimpleNamespace(generated_files=[str(output)])
    client = FakeS3()
    artifacts = upload_output_artifacts(
        result,
        "job-id",
        client,
        "bucket",
        prefix="runninghub/wangp",
    )

    assert artifacts == [
        {
            "storage": "s3",
            "bucket": "bucket",
            "key": "runninghub/wangp/job-id/000-clip.mp4",
            "uri": "s3://bucket/runninghub/wangp/job-id/000-clip.mp4",
            "filename": "clip.mp4",
            "size_bytes": 5,
            "media_type": "video/mp4",
            "sha256": "0cab1c9617404faf2b24e221e189ca5945813e14d3f766345b09ca13bbe28ffc",
        }
    ]
    assert output.exists()
    remove_local_outputs(result)
    assert not output.exists()


def test_filter_models_uses_cached_metadata():
    models = [
        {"model_type": "one", "family": "qwen"},
        {"model_type": "two", "family": "flux"},
    ]
    assert filter_models(models, family="qwen") == [models[0]]
    assert filter_models(models, model_type="two") == [models[1]]


@pytest.fixture
def generation_catalog():
    return {
        "models": [
            {"model_type": "krea2_turbo", "main_output": ["image"]},
            {"model_type": "minimax_h3", "main_output": ["video"]},
            {"model_type": "switchable", "main_output": ["image", "video"]},
            {"model_type": "tts", "main_output": ["audio"]},
        ],
        "defaults": {
            "switchable": {"image_mode": 0},
        },
    }


def test_generation_kind_is_inferred_from_model_metadata(generation_catalog):
    assert resolve_generation_kind(generation_catalog, "krea2_turbo") == "image"
    assert resolve_generation_kind(generation_catalog, "minimax_h3") == "video"
    assert resolve_generation_kind(generation_catalog, "tts") == "audio"


def test_generation_kind_accepts_matching_explicit_route(generation_catalog):
    assert (
        resolve_generation_kind(generation_catalog, "minimax_h3", "video")
        == "video"
    )


def test_generation_kind_rejects_model_route_mismatch(generation_catalog):
    with pytest.raises(ValueError, match="not image"):
        resolve_generation_kind(generation_catalog, "minimax_h3", "image")


def test_generation_kind_rejects_unknown_model(generation_catalog):
    with pytest.raises(ValueError, match="unknown model"):
        resolve_generation_kind(generation_catalog, "missing")


def test_generation_kind_uses_native_image_mode_for_dual_output_model(
    generation_catalog,
):
    assert resolve_generation_kind(generation_catalog, "switchable") == "video"
    assert (
        resolve_generation_kind(
            generation_catalog,
            "switchable",
            params={"image_mode": 2},
        )
        == "image"
    )


def test_generation_worker_routes_video_to_video_pool_and_other_media_to_image_pool():
    assert generation_worker_name("video") == "WanGPVideoWorker"
    assert generation_worker_name("image") == "WanGPImageWorker"
    assert generation_worker_name("audio") == "WanGPImageWorker"


def test_generation_worker_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unsupported generation kind"):
        generation_worker_name("text")


def test_params_file_loads_json_object(tmp_path: Path):
    params_file = tmp_path / "params.json"
    params_file.write_text('{"prompt": "fox", "seed": 7}')
    assert load_params_file(str(params_file)) == {"prompt": "fox", "seed": 7}


def test_params_file_rejects_non_object(tmp_path: Path):
    params_file = tmp_path / "params.json"
    params_file.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        load_params_file(str(params_file))


def test_params_json_loads_inline_object():
    assert load_params_json('{"prompt":"fox","seed":-1}') == {
        "prompt": "fox",
        "seed": -1,
    }


def test_params_json_rejects_invalid_or_non_object_values():
    with pytest.raises(ValueError, match="not valid"):
        load_params_json("{")
    with pytest.raises(ValueError, match="must be an object"):
        load_params_json("[]")


def test_krea_submit_uses_fixed_model_and_image_route(monkeypatch):
    captured = {}

    def fake_submit(model, params, kind):
        captured.update(model=model, params=params, kind=kind)
        return {"id": "job", "status": "queued", "kind": kind}

    monkeypatch.setattr(control, "submit_generation", fake_submit)

    result = control.submit_krea_generation({"prompt": "fox", "seed": 42})

    assert result == {"id": "job", "status": "queued", "kind": "image"}
    assert captured["model"] == "krea2_turbo"
    assert captured["kind"] == "image"
    assert captured["params"]["seed"] == 42


def test_krea_submit_requires_prompt():
    with pytest.raises(ValueError, match="params.prompt"):
        control.submit_krea_generation({"seed": -1})


@pytest.mark.parametrize(
    ("variant", "expected_model"),
    [
        ("full", "minimax_h3_vdn"),
        ("pruned", "minimax_h3_vdn_pruned"),
    ],
)
def test_h3_vdn_submit_selects_variant_and_applies_defaults(
    monkeypatch,
    variant,
    expected_model,
):
    captured = {}

    def fake_submit(model, params, kind):
        captured.update(model=model, params=params, kind=kind)
        return {"id": "job", "status": "queued", "kind": kind}

    monkeypatch.setattr(control, "submit_generation", fake_submit)

    result = control.submit_h3_vdn_generation(
        {"prompt": "astronaut", "seed": 42},
        variant,
    )

    assert result == {"id": "job", "status": "queued", "kind": "video"}
    assert captured["model"] == expected_model
    assert captured["kind"] == "video"
    assert captured["params"] == {
        **control.H3_VDN_DEFAULTS,
        "prompt": "astronaut",
        "seed": 42,
    }
    assert "activated_loras" not in captured["params"]


def test_h3_vdn_submit_requires_prompt_and_known_variant():
    with pytest.raises(ValueError, match="params.prompt"):
        control.submit_h3_vdn_generation({"seed": -1})
    with pytest.raises(ValueError, match="full, pruned"):
        control.submit_h3_vdn_generation({"prompt": "astronaut"}, "other")


def test_h3_vdn_submit_enables_native_audio_refinement(monkeypatch):
    captured = {}

    def fake_submit(model, params, kind):
        captured.update(model=model, params=params, kind=kind)
        return {"id": "job", "status": "queued", "kind": kind}

    monkeypatch.setattr(control, "submit_generation", fake_submit)

    control.submit_h3_vdn_generation(
        {"prompt": "astronaut", "audio_refinement": " ENABLED "}
    )

    assert captured["params"]["audio_refinement"] == "enabled"


@pytest.mark.parametrize("value", [True, "six_steps", ""])
def test_h3_vdn_submit_rejects_invalid_audio_refinement(value):
    with pytest.raises(ValueError, match="audio_refinement"):
        control.submit_h3_vdn_generation(
            {"prompt": "astronaut", "audio_refinement": value}
        )


def test_audio_refinement_is_nested_for_supported_h3_model():
    catalog = {
        "defaults": {
            "minimax_h3_vdn": {
                "custom_settings": {
                    "audio_refinement": "none",
                    "h3_mask_mode": "grouped_rows",
                }
            }
        }
    }

    assert control.normalize_audio_refinement(
        catalog,
        "minimax_h3_vdn",
        {
            "prompt": "astronaut",
            "audio_refinement": " ENABLED ",
            "custom_settings": {"h3_mask_mode": "shared_timestep"},
        },
    ) == {
        "prompt": "astronaut",
        "custom_settings": {
            "audio_refinement": "enabled",
            "h3_mask_mode": "shared_timestep",
        },
    }


def test_audio_refinement_rejects_models_without_catalog_support():
    catalog = {"defaults": {"krea2_turbo": {"custom_settings": None}}}

    with pytest.raises(ValueError, match="does not support audio refinement"):
        control.normalize_audio_refinement(
            catalog,
            "krea2_turbo",
            {"prompt": "fox", "audio_refinement": "enabled"},
        )


def test_audio_refinement_rejects_conflicting_native_setting():
    catalog = {
        "defaults": {
            "minimax_h3_vdn": {
                "custom_settings": {"audio_refinement": "none"}
            }
        }
    }

    with pytest.raises(ValueError, match="conflicts"):
        control.normalize_audio_refinement(
            catalog,
            "minimax_h3_vdn",
            {
                "audio_refinement": "enabled",
                "custom_settings": {"audio_refinement": "none"},
            },
        )


def test_h3_vdn_help_describes_preset_and_automatic_lora():
    assert control.H3_VDN_HELP["models"] == {
        "full": "minimax_h3_vdn",
        "pruned": "minimax_h3_vdn_pruned",
    }
    assert control.H3_VDN_HELP["params_json"]["defaults"] == {
        **control.H3_VDN_DEFAULTS
    }
    assert "automatically loaded" in control.H3_VDN_HELP["vdn_acceleration_lora"]
    assert control.H3_VDN_HELP["params_json"]["audio_refinement"] == {
        "type": "string",
        "choices": ["none", "enabled"],
        "enabled_effect": "6 extra steps at denoising strength 0.5",
    }
    assert control.H3_VDN_HELP["h3_common"] is control.H3_HELP


def test_h3_help_documents_model_aware_sliding_windows():
    sliding = control.H3_HELP["sliding_windows"]

    assert control.H3_HELP["family"] == "minimax_h3"
    assert "--family h3" in control.H3_HELP["discover_models_command"]
    assert sliding["activation"] == (
        "video_length must be greater than sliding_window_size"
    )
    assert sliding["settings"]["multi_prompts_gen_type"] == {
        "FG": "reuse one complete prompt for every window",
        "PW": "use each blank-line-separated paragraph for a new window",
    }
    assert sliding["two_window_example"]["calculation"] == (
        "362 + (362 - 18) = 706 frames"
    )
    assert "image_start anchors only the first window" in (
        sliding["image_conditioning"]["fl2va"]
    )
    assert "video_prompt_type=KFI" in sliding["image_conditioning"]["vdn"]
    assert "remain available across windows" in (
        sliding["image_conditioning"]["ref2va"]
    )


def test_krea_help_describes_json_and_lora_contract():
    assert control.KREA_HELP["params_json"]["required"]["prompt"] == {
        "type": "string"
    }
    assert control.KREA_HELP["params_json"]["optional"]["seed"]["default"] == -1
    assert control.KREA_HELP["lora_example"] == {
        "prompt": "linda standing in a sunlit photography studio",
        "activated_loras": [
            "/data/loras/krea2/linda_krea2_v1.safetensors"
        ],
        "loras_multipliers": "0.8",
    }
    assert "--family krea2" in control.KREA_HELP["list_loras_command"]


def test_lora_listing_selects_safe_family_directory(monkeypatch):
    calls = []

    class Volume:
        def listdir(self, path, recursive=False):
            calls.append((path, recursive))
            return [
                SimpleNamespace(
                    path="loras/krea2/example.safetensors",
                    type=SimpleNamespace(value=1),
                    size=123,
                    mtime=456,
                )
            ]

    monkeypatch.setattr(control, "data_volume", Volume())

    assert control.list_loras(" krea2 ", recursive=True) == [
        {
            "path": "loras/krea2/example.safetensors",
            "type": 1,
            "size": 123,
            "mtime": 456,
        }
    ]
    assert calls == [("loras/krea2", True)]


def test_lora_listing_rejects_path_traversal():
    with pytest.raises(ValueError, match="family"):
        control.lora_directory("../krea2")


def test_h3_lora_alias_selects_minimax_h3_directory():
    assert control.lora_directory("h3") == "loras/minimax_h3"
    assert control.lora_directory("minimax_h3") == "loras/minimax_h3"


def test_krea2_turbo_example_has_documented_baseline():
    params = load_params_file("examples/krea2_turbo.json")
    assert params["resolution"] == "1024x1024"
    assert params["num_inference_steps"] == 8
    assert params["guidance_scale"] == 0
    assert params["flow_shift"] == 5.0
    assert "model" not in params
    assert "model_type" not in params
    assert "_api" not in params


def test_new_app_uses_an_independent_job_store():
    assert JOB_DICT_NAME == "wangpt-modal-jobs"


def test_control_cli_helpers_are_local_python_functions():
    assert inspect.isfunction(control.submit_generation)
    assert inspect.isfunction(control.get_generation_job)
    assert inspect.isfunction(control.cancel_generation_job)
    assert inspect.isfunction(control.inspect_catalog)
    assert not hasattr(control, "control_image")


def test_catalog_cache_hit_does_not_start_catalog_publisher(monkeypatch):
    catalog = {"models": []}

    class CatalogStore:
        def get(self, key):
            assert key
            return catalog

    def fail_if_called(*args, **kwargs):
        pytest.fail("cached catalog lookup must not start publish_catalog")

    monkeypatch.setattr(control, "catalog_store", CatalogStore())
    monkeypatch.setattr(control.modal.Function, "from_name", fail_if_called)

    assert control.load_deployed_catalog() is catalog


@pytest.mark.parametrize(
    ("operation", "expected"),
    [("defaults", {"seed": -1}), ("schema", {"prompt": {"type": "string"}})],
)
def test_catalog_inspection_uses_correct_collection(monkeypatch, operation, expected):
    catalog = {
        "defaults": {"model": {"seed": -1}},
        "schemas": {"model": {"prompt": {"type": "string"}}},
    }
    monkeypatch.setattr(control, "load_deployed_catalog", lambda: catalog)

    assert control.inspect_catalog(operation, "model") == expected


def test_catalog_model_listing_accepts_h3_family_alias(monkeypatch):
    vdn = {"model_type": "minimax_h3_vdn", "family": "minimax_h3"}
    catalog = {
        "models": [vdn, {"model_type": "krea2_turbo", "family": "krea2"}]
    }
    monkeypatch.setattr(control, "load_deployed_catalog", lambda: catalog)

    assert control.inspect_catalog("models", family=" h3 ") == [vdn]
