"""Check emitted commands against a checkout of the exact pinned upstream.

Set FIZGIG_TEST_ROOT to enable. Only argparse declarations are evaluated;
upstream training code, GPU imports and model loading never run locally.
"""

import argparse
import ast
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from fizgig_common import is_refmod

from app import (
    CAPTION_SCRIPT,
    FIZGIG_COMMIT,
    FIZGIG_ROOT,
    SUPPORTED_PRESETS,
    build_pipeline_commands,
    paths_for_request,
    validate_training_request,
)


@pytest.fixture(scope="module")
def upstream():
    root = os.environ.get("FIZGIG_TEST_ROOT")
    if not root:
        pytest.skip("set FIZGIG_TEST_ROOT to the pinned Fizgig checkout")
    root = Path(root)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert commit == FIZGIG_COMMIT, "contract checks must use the pinned revision"
    return root


def _constant(root, relative, name):
    tree = ast.parse((root / relative).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return ast.literal_eval(node.value)


def _parser(root, path):
    tree = ast.parse(path.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "setup_parser" in functions:
        setup = functions["setup_parser"]
    else:
        # The generic cache CLI declares its parser inline in main. Stop
        # before parse_args, which is followed by the actual cache execution.
        setup = functions["main"]
        statements = []
        for node in setup.body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "parse_args"):
                statements.append(ast.Return(value=node.value.func.value))
                break
            statements.append(node)
        else:
            raise AssertionError("upstream cache parser boundary changed")
        setup.body = statements
        setup.name = "setup_parser"
    helpers = [functions[name] for name in ("_shift_arg",) if name in functions]
    namespace = {
        "argparse": argparse,
        "quant": SimpleNamespace(PRECISIONS=_constant(root, "src/fizgig/families/quant.py", "PRECISIONS")),
        # Package availability is checked in the built image; locally use the
        # real catalog's keys without importing optimizers or CUDA libraries.
        "available_optimizers": lambda: list(_constant(root, "src/fizgig/training/optimizers.py", "_CATALOG")),
    }
    module = ast.fix_missing_locations(ast.Module(body=[*helpers, setup], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["setup_parser"]()


@pytest.mark.parametrize("preset", SUPPORTED_PRESETS)
def test_every_preset_command_is_accepted_by_pinned_upstream(upstream, preset):
    intent = {
        "family": SUPPORTED_PRESETS[preset]["family"],
        "dataset": "contract-dataset",
        "output_name": "contract-run",
        "preset": preset,
        "trigger_word": "contractsubject",
        "epochs": 2,
    }
    if is_refmod(intent):
        intent.pop("epochs")
        intent.pop("trigger_word")
        intent["description"] = "--a hint with spaces"
    normalized = validate_training_request(intent)
    for phase, command in build_pipeline_commands(normalized, paths_for_request(normalized)):
        if command[1] == str(CAPTION_SCRIPT):
            continue
        script = upstream / Path(command[1]).relative_to(FIZGIG_ROOT)
        assert script.is_file(), f"{phase} points to a removed upstream script"
        parser = _parser(upstream, script)
        parsed = parser.parse_args(command[2:])
        assert parsed.dataset_config.endswith("dataset.toml")
        if phase == "making_refmod":
            assert parsed.steps == normalized["steps"]
            assert parsed.max_refs == normalized["max_refs"]
            assert parsed.base_model == "ref2va"
            assert parsed.description == intent["description"]
            assert parsed.audio == "off"
            assert not parsed.sample_prompts
            assert parsed.ref_cache_dir == ([str(paths_for_request(normalized)["cache_dir"]) + "-refs"]
                                            if normalized["steps"] else None)
        if phase == "caching_references":
            assert parsed.cache_suffix == "-refs"
            assert parsed.megapixels == normalized["target_mp"]
        if phase == "caching_latents" and is_refmod(intent):
            assert parsed.captions_optional == (normalized["steps"] == 0)
            assert parsed.megapixels == (0.25 if normalized["steps"] else normalized["target_mp"])
        if phase == "training":
            resumed = parser.parse_args([*command[2:], "--resume", "/data/fizgig/runs/contract-run/contract-run-000001-state"])
            assert resumed.resume.endswith("-state")
            assert parsed.max_train_epochs == 2
            assert parsed.save_state and parsed.save_state_on_train_end
            if intent["family"] == "krea2":
                assert parsed.family == "krea2"
                assert parsed.auto_recaption and parsed.captioner == parsed.text_encoder


def test_caption_helpers_keep_the_called_interface(upstream):
    cases = [
        ("src/fizgig/krea2/utils.py", "load_krea2_text_encoder", {"dtype", "device"}),
        ("src/fizgig/krea2/embedder.py", "generate_caption",
         {"max_new_tokens", "detailed", "seed", "instruction"}),
    ]
    for relative, name, kwargs in cases:
        tree = ast.parse((upstream / relative).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        supported = {arg.arg for arg in [*function.args.args, *function.args.kwonlyargs]}
        assert kwargs <= supported


def test_driver_cache_has_a_distinct_architecture(upstream):
    tree = ast.parse((upstream / "src/fizgig/families/krea2.py").read_text())
    description = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "KREA2" for t in n.targets))
    arch_id = next(ast.literal_eval(k.value) for k in description.keywords if k.arg == "arch_id")
    assert arch_id == "krea2drv", "cache layout changed; review reuse of existing dataset caches"


@pytest.mark.parametrize("clips", ["still", "motion"])
@pytest.mark.parametrize("steps", [0, 200])
def test_video_refmod_commands_match_pinned_upstream(upstream, clips, steps):
    from fizgig_s3 import VIDEO_SUFFIXES
    assert VIDEO_SUFFIXES == set(_constant(upstream, "src/fizgig/minimax/clip.py", "VIDEO_EXTENSIONS"))
    request = validate_training_request({"family": "minimax_h3", "dataset": "clips",
        "output_name": "video-refmod", "preset": "h3_refmod_lite", "steps": steps,
        "clips": clips, "token_cap": 5120})
    for phase, command in build_pipeline_commands(request, paths_for_request(request)):
        script = upstream / Path(command[1]).relative_to(FIZGIG_ROOT)
        parsed = _parser(upstream, script).parse_args(command[2:])
        if phase in {"caching_latents", "caching_references"}:
            assert parsed.clip_still is True
        if phase == "making_refmod":
            assert parsed.clips == clips and parsed.token_cap == 5120
            assert parsed.audio == "off"
