from pathlib import Path

import pytest

from app import (
    APP_NAME,
    DATA_ROOT,
    DATA_VOLUME_NAME,
    FIZGIG_COMMIT,
    FIZGIG_SCRIPTS,
    MODEL_PATHS,
    SUPPORTED_FAMILY,
    SUPPORTED_FAMILIES,
    _parse_epoch,
    _verify_models,
    build_pipeline_commands,
    dataset_config_text,
    paths_for_request,
    require_data_path,
    resolve_resume_path,
    validate_training_request,
)


def valid_request(**overrides):
    request = {
        "family": "minimax_h3",
        "dataset": "anna-v3",
        "output_name": "anna_h3_v1",
        "preset": "h3_character_fast",
        "trigger_word": "annaxs",
    }
    request.update(overrides)
    return request


def krea2_request(**overrides):
    request = {
        "family": "krea2",
        "dataset": "linda",
        "output_name": "linda_krea2_v1",
        "preset": "krea2_defaults",
        "trigger_word": "linda",
    }
    request.update(overrides)
    return request


def test_deployment_identity_is_stable_and_pinned():
    assert APP_NAME == "fizgig-modal-app"
    assert DATA_VOLUME_NAME == "wangp-data"
    assert len(FIZGIG_COMMIT) == 40
    assert FIZGIG_COMMIT != "master"


def test_supported_families_include_h3_and_krea2():
    normalized = validate_training_request(valid_request())
    assert normalized["family"] == SUPPORTED_FAMILY
    assert normalized["network_dim"] == 8
    assert normalized["epochs"] == 40

    krea2 = validate_training_request(krea2_request())
    assert set(SUPPORTED_FAMILIES) == {"minimax_h3", "krea2"}
    assert krea2["network_dim"] == 32
    assert krea2["epochs"] == 30
    assert krea2["auto_caption"] is True
    assert krea2["auto_recaption"] is True

    with pytest.raises(ValueError, match="does not support family"):
        validate_training_request(krea2_request(preset="h3_character_fast"))


def test_quality_preset_and_safe_overrides_are_applied():
    normalized = validate_training_request(
        valid_request(preset="h3_character_quality", epochs=12)
    )
    assert normalized["network_dim"] == 16
    assert normalized["adapter_ramp"] == 0.003
    assert normalized["epochs"] == 12
    assert normalized["seed"] == 42


def test_unknown_fields_and_path_components_are_rejected():
    with pytest.raises(ValueError, match="unsupported request fields"):
        validate_training_request(valid_request(cli_args=["--no_quantize"]))
    with pytest.raises(ValueError, match="must not contain a path"):
        validate_training_request(valid_request(dataset="../anna"))
    with pytest.raises(ValueError, match="must not contain a path"):
        validate_training_request(valid_request(output_name="anna/run"))
    with pytest.raises(ValueError, match="epochs must be an integer"):
        validate_training_request(valid_request(epochs=4.5))
    with pytest.raises(ValueError, match="unsupported request fields"):
        validate_training_request(valid_request(auto_caption=False))


def test_family_dataset_output_and_preset_are_required():
    with pytest.raises(ValueError, match="missing required fields"):
        validate_training_request({"dataset": "anna", "output_name": "anna_h3"})


def test_paths_stay_on_the_shared_volume():
    normalized = validate_training_request(valid_request())
    paths = paths_for_request(normalized)
    assert paths["image_dir"] == DATA_ROOT / "fizgig/datasets/anna-v3/images"
    assert paths["run_dir"] == DATA_ROOT / "fizgig/runs/anna_h3_v1"
    assert paths["promoted_lora"] == DATA_ROOT / "loras/anna_h3_v1.safetensors"

    with pytest.raises(ValueError, match="must remain under"):
        require_data_path("/data/../etc/passwd")


def test_dataset_config_uses_dedicated_image_and_cache_directories():
    config = dataset_config_text(
        DATA_ROOT / "fizgig/datasets/anna-v3/images",
        DATA_ROOT / "fizgig/datasets/anna-v3/cache",
    )
    assert "resolution = [512, 512]" in config
    assert f'image_directory = "{DATA_ROOT}/fizgig/datasets/anna-v3/images"' in config
    assert f'cache_directory = "{DATA_ROOT}/fizgig/datasets/anna-v3/cache"' in config


def test_pipeline_uses_only_pinned_fizgig_scripts_and_allowlisted_flags():
    normalized = validate_training_request(valid_request())
    paths = paths_for_request(normalized)
    commands = build_pipeline_commands(normalized, paths)

    assert [phase for phase, _ in commands] == [
        "caching_latents",
        "caching_text",
        "training",
    ]
    assert commands[0][1][1] == str(FIZGIG_SCRIPTS / "minimax_cache_latents.py")
    assert commands[1][1][1] == str(FIZGIG_SCRIPTS / "minimax_cache_text.py")
    train = commands[2][1]
    assert train[1] == str(FIZGIG_SCRIPTS / "minimax_train.py")
    assert "--no_train_adaln" in train
    assert "--sample_prompts" not in train
    assert str(MODEL_PATHS["minimax_h3"]["dit"]) in train
    assert all("shell" not in argument for argument in train)


def test_krea2_pipeline_auto_captions_caches_and_trains_with_upstream_scripts():
    normalized = validate_training_request(krea2_request())
    paths = paths_for_request(normalized)
    commands = build_pipeline_commands(normalized, paths)

    assert [phase for phase, _ in commands] == [
        "captioning",
        "caching_latents",
        "caching_text",
        "training",
    ]
    assert commands[1][1][1] == str(FIZGIG_SCRIPTS / "krea2_cache_latents.py")
    assert commands[2][1][1] == str(FIZGIG_SCRIPTS / "krea2_cache_text.py")
    caption = commands[0][1]
    assert "--trigger-word" in caption
    assert str(MODEL_PATHS["krea2"]["text_encoder"]) in caption

    train = commands[3][1]
    assert train[1] == str(FIZGIG_SCRIPTS / "krea2_train.py")
    assert str(MODEL_PATHS["krea2"]["dit"]) in train
    assert "--log_per_image_loss" in train
    assert "--per_image_lr" in train
    assert "--auto_recaption" in train
    assert "--turbo_lora" not in train
    assert "--sample_prompts" not in train


def test_krea2_caption_behavior_is_resolved_by_preset():
    normalized = validate_training_request(krea2_request())
    commands = build_pipeline_commands(normalized, paths_for_request(normalized))
    assert "captioning" in [phase for phase, _ in commands]
    assert "--auto_recaption" in commands[-1][1]
    with pytest.raises(ValueError, match="unsupported request fields"):
        validate_training_request(krea2_request(auto_caption=False))


def test_krea2_model_verification_requires_only_base_training_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    models = {
        name: tmp_path / f"{name}.safetensors"
        for name in ("dit", "text_encoder", "vae")
    }
    for name in ("dit", "text_encoder", "vae"):
        models[name].touch()
    monkeypatch.setitem(MODEL_PATHS, "krea2", models)

    _verify_models(validate_training_request(krea2_request()))
    models["vae"].unlink()
    with pytest.raises(FileNotFoundError, match="vae"):
        _verify_models(validate_training_request(krea2_request()))


def test_resume_path_must_be_an_existing_state_directory(tmp_path: Path):
    state = tmp_path / "anna_h3_v1-000010-state"
    state.mkdir()
    request = validate_training_request(valid_request(resume_from=state.name))
    assert resolve_resume_path(request, tmp_path) == state

    missing = validate_training_request(valid_request(resume_from="missing-state"))
    with pytest.raises(ValueError, match="resume state was not found"):
        resolve_resume_path(missing, tmp_path)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Epoch 4/40", (4, 40)),
        ("starting epoch 9 of 60", (9, 60)),
        ("cache complete", None),
    ],
)
def test_epoch_parser(line, expected):
    assert _parse_epoch(line) == expected
