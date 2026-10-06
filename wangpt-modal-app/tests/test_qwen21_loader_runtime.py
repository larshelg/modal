"""Exercise the pinned MMGP file loader on CPU during the Modal image build."""

import ast
import json
import os
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Requires the Modal image's PyTorch/MMGP dependencies")
class Qwen21CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mmgp import offload, quant_router
        from safetensors.torch import save_file

        cls.offload = offload
        cls.save_file = staticmethod(save_file)
        quant_router.register_handler("shared.qtypes.int8_convrot")
        root = Path(os.environ.get("WANGP_TEST_ROOT", "/opt/Wan2GP"))
        tree = ast.parse((root / "models/qwen21/qwen21_handler.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "fast_load_transformers_model"
                 and any(k.arg == "modelClass" and isinstance(k.value, ast.Name)
                         and k.value.id == "QwenImage21Transformer2DModel"
                         for k in node.keywords)]
        assert len(calls) == 1
        # Use the actual handler argument; removing the wiring fails this test.
        cls.split_map = ast.literal_eval(next(
            k.value for k in calls[0].keywords if k.arg == "fused_split_map"))

    def model(self):
        model = torch.nn.Module()
        block = torch.nn.Module()
        block.img_mlp = torch.nn.Module()
        block.img_mlp.gate_layer = torch.nn.Linear(256, 3, bias=False, dtype=torch.bfloat16)
        block.img_mlp.proj = torch.nn.Linear(256, 3, bias=False, dtype=torch.bfloat16)
        model.transformer_blocks = torch.nn.ModuleList([block])
        return model

    def load(self, state, *, split=True):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "test.safetensors")
            self.save_file(state, path)
            model = self.model()
            self.offload.load_model_data(
                model, path, writable_tensors=False, default_dtype=torch.bfloat16,
                fused_split_map=self.split_map if split else None, verboseLevel=0)
            return model.transformer_blocks[0].img_mlp

    def int8_state(self, *, fused):
        data = (torch.arange(6 * 256).reshape(6, 256) % 127 - 63).to(torch.int8)
        scales = torch.tensor([0.125, 0.25, 0.5, 1, 2, 4], dtype=torch.float32)
        config = torch.tensor(list(json.dumps({
            "format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256,
        }).encode()), dtype=torch.uint8)
        state = {}
        parts = [("gate_up", data, scales)] if fused else [
            ("gate_layer", data[:3], scales[:3]), ("proj", data[3:], scales[3:])]
        for name, weight, scale in parts:
            base = "transformer_blocks.0.img_mlp." + name
            state[base + ".weight"] = weight.contiguous()
            state[base + ".weight_scale"] = scale.contiguous()
            state[base + ".comfy_quant"] = config.clone()
        return state, data, scales

    def assert_int8_preserved(self, loaded, data, scales):
        from shared.qtypes.int8_convrot import QLinearInt8ConvRot
        for module, start in [(loaded.gate_layer, 0), (loaded.proj, 3)]:
            # MMGP keeps its router class and delegates to the ConvRot handler.
            self.assertIs(module._router_forward_impl, QLinearInt8ConvRot.forward)
            self.assertEqual(module._convrot_group_size, 256)
            self.assertEqual(module.weight._data.dtype, torch.int8)
            torch.testing.assert_close(module.weight._data, data[start:start + 3])
            torch.testing.assert_close(module.weight._scale.reshape(-1),
                                       scales[start:start + 3].to(torch.bfloat16))

    def test_fused_int8_reproduces_failure_without_map(self):
        state, _, _ = self.int8_state(fused=True)
        with self.assertRaisesRegex(Exception, "Missing keys:.*img_mlp"):
            self.load(state, split=False)

    def test_fused_int8_preserves_weights_scales_and_rotation(self):
        state, data, scales = self.int8_state(fused=True)
        self.assert_int8_preserved(self.load(state), data, scales)

    def test_already_split_int8_is_unchanged(self):
        state, data, scales = self.int8_state(fused=False)
        self.assert_int8_preserved(self.load(state), data, scales)

    def test_fused_bf16_splits_gate_then_proj(self):
        weight = torch.arange(6 * 256, dtype=torch.float32).reshape(6, 256).to(torch.bfloat16)
        model = self.load({"transformer_blocks.0.img_mlp.gate_up.weight": weight})
        torch.testing.assert_close(model.gate_layer.weight, weight[:3])
        torch.testing.assert_close(model.proj.weight, weight[3:])


if __name__ == "__main__":
    unittest.main()
