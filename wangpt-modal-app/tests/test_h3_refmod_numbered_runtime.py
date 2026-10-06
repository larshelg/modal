"""Tensor/hook regressions, also run on CPU in the real Modal image build."""
import functools
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

from h3_refmod_numbered import REPORTS_ATTR, STATE_ATTR, install_numbered_hooks


class LatentSourceCompatibilityTests(unittest.TestCase):
    def test_pinned_latent_edits_match_native_generate(self):
        root = Path(os.environ.get("WANGP_TEST_ROOT", "/opt/Wan2GP"))
        plugin = Path(os.environ.get("WANGP_LATENT_PLUGIN_ROOT", str(root / "plugins/wan2gp-h3-latent-continue")))
        if not root.exists() or not plugin.exists():
            self.skipTest("Pinned sources are checked in the Modal image build")
        source = (root / "models/minimax_h3/pipeline.py").read_text()
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3Pipeline")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate")
        lines = source.splitlines(keepends=True)
        body = "".join(lines[method.lineno - 1:method.end_lineno])
        for edit in json.loads((plugin / "native_edits.json").read_text()):
            old, new = edit["old"], edit["new"]
            if not old.startswith("\n"):
                old, new = "\n" + old, "\n" + new
            self.assertEqual(body.count(old), 1, f"Latent insertion point no longer unique: {old!r}")
            body = body.replace(old, new, 1)
        # Syntax-check the edited body without importing the GPU pipeline.
        compile("class Pipeline:\n" + body, "<latent-edited-generate>", "exec")


@unittest.skipIf(torch is None, "Tensor tests run in the Modal image, which includes PyTorch")
class NumberedRuntimeTests(unittest.TestCase):
    def setUp(self):
        class Image:
            def __init__(self, latent):
                self.latent, self.hide_ref = latent, False

        class Video(Image):
            @property
            def shape(self):
                return (3, (self.latent.shape[2] - 1) * 4 + 1, 32, 32)

            def __getitem__(self, key):
                return Video(self.latent[:, :, :max(1, (key[1].stop - 1) // 4 + 1)])

        class Pipeline:
            reference_mode, audio_only, fixed_prompt = True, False, None
            device = "cpu"

            def __init__(self):
                self.decodes, self.encodings, self.latents = [], [], []
                self.vae = SimpleNamespace(decode=self.decode, _model_dtype=torch.float32)

            def _check_abort(self):
                pass

            def _use_shared_components(self):
                pass

            def decode(self, latent):
                self.decodes.append(latent.clone())
                frames = (latent.shape[2] - 1) * 4 + 1
                return torch.ones(1, 3, frames, 32, 32) * latent.mean()

            def _add_image_reference(self, image, width, height, size, presentation, visual_latents, refs):
                if isinstance(image, Image):
                    self.latents.append(image.latent)
                else:
                    presentation.append({"type": "image", "frames": image})
                refs.append({"kind": "image"})

            def _add_video_reference(self, video, soundtrack, fps, presentation, visual_latents, audio_latents, refs):
                if isinstance(video, Video):
                    self.latents.append(video.latent)
                else:
                    presentation.append({"type": "video", "frames": video})
                refs.append({"kind": "video"})

            def _encode_prompt(self, prompt, presentation):
                self.encodings.append(presentation)
                return prompt

            def generate(self, prompt, **kwargs):
                for _ in range(kwargs.get("phases", 1)):
                    presentation, refs = [], []
                    if kwargs.get("input_video") is not None and kwargs.get("prefix_frames_count", 0):
                        presentation.append({"type": "image", "frames": kwargs["input_video"][:, -1:]})
                    elif kwargs.get("image_start") is not None:
                        presentation.append({"type": "image", "frames": kwargs["image_start"]})
                    for image in kwargs.get("input_ref_images") or []:
                        self._add_image_reference(image, 32, 32, 100, presentation, [], refs)
                    for key in ("input_frames", "input_frames2", "input_frames3"):
                        video = kwargs.get(key)
                        if video is not None:
                            if kwargs.get("trim") and isinstance(video, Video):
                                video = video[:, :5]
                            self._add_video_reference(video, None, 24, presentation, [], [], refs)
                    self._encode_prompt(prompt, presentation)
                return "generated"

        native_generate = Pipeline.generate

        @functools.wraps(native_generate)
        def basic_generate(self, *args, **kwargs):
            return native_generate(self, *args, **kwargs)

        Pipeline.generate = basic_generate
        self.patches = SimpleNamespace(
            SETTING_EXTRACT="h3_refmod_extract", _RefModImageSentinel=Image,
            _RefModVideoSentinel=Video, _reset_gen_state=lambda pipeline: None,
        )
        self.mods = {}
        self.wgp = SimpleNamespace()
        self.storage = SimpleNamespace(load_refmod=lambda name: self.mods[name])
        self.h3 = SimpleNamespace(MiniMaxH3Pipeline=Pipeline, _qwen_frames=lambda video: video.permute(1, 2, 3, 0).add(1).mul(0.5))
        install_numbered_hooks(self.wgp, self.patches, self.storage, self.h3)
        self.pipeline = Pipeline()
        self.add_mod("alice")
        self.add_mod("walk", "video", 5)

    def add_mod(self, name, kind="image", frames=1):
        latent = torch.ones(1, 24, frames, 2, 2)
        self.mods[name] = SimpleNamespace(kind=kind, weighted_latent=lambda strength, curve=None: latent * strength)

    def run_numbered(self, rows=None, **kwargs):
        options = {"text_encode": True, "rows": rows or [{"mod": "alice"}]}
        return self.pipeline.generate("prompt", custom_settings={"h3_refmod_state": json.dumps(options)}, **kwargs)

    def test_image_and_video_presented_with_mixed_refs_and_same_weighted_latents(self):
        self.run_numbered([{"mod": "alice", "strength": 0.5}, {"mod": "walk", "reference_fps": 12}],
                          image_start=torch.zeros(1), input_ref_images=[torch.zeros(1)], input_frames=torch.zeros(1))
        refs = getattr(self.wgp, REPORTS_ATTR)[0]["references"]
        self.assertEqual([r["label"] for r in refs], ["<Picture 1>", "<Picture 2>", "<Picture 3>", "<Video 1>", "<Video 2>"])
        self.assertEqual(refs[2]["mod"], "alice")
        self.assertEqual(refs[4]["mod"], "walk")
        self.assertTrue(torch.equal(self.pipeline.decodes[0], self.pipeline.latents[0]))
        self.assertEqual(self.pipeline.decodes[0].mean().item(), 0.5)
        self.assertEqual(self.pipeline.encodings[0][-1]["timestamps"], [0, 0.5, 1])

    def test_video_only_and_two_phase_reuse_decoded_pixels(self):
        self.run_numbered([{"mod": "walk"}], phases=2)
        self.assertEqual(len(self.pipeline.decodes), 1)
        self.assertEqual(len(getattr(self.wgp, REPORTS_ATTR)), 1)
        self.assertEqual(getattr(self.wgp, REPORTS_ATTR)[0]["references"][0]["label"], "<Video 1>")

    def test_continuation_anchor_precedes_image_and_video_refmods(self):
        self.run_numbered([{"mod": "alice"}, {"mod": "walk"}],
                          input_video=torch.zeros(3, 18, 32, 32), prefix_frames_count=18, window_no=1)
        refs = getattr(self.wgp, REPORTS_ATTR)[0]["references"]
        self.assertEqual([r["label"] for r in refs], ["<Picture 1>", "<Picture 2>", "<Video 1>"])
        self.assertEqual(refs[0]["source"], "native")
        self.assertEqual([r["mod"] for r in refs[1:]], ["alice", "walk"])
        self.assertEqual(len(self.pipeline.decodes), 2)
        self.assertEqual(len(self.pipeline.latents), 2)
        # An independent request on the same pipeline starts its numbering afresh.
        self.run_numbered()
        self.assertEqual(getattr(self.wgp, REPORTS_ATTR)[-1]["references"][0]["label"], "<Picture 1>")

    def test_video_trimming_retains_name_and_decodes_matching_latent(self):
        self.run_numbered([{"mod": "walk"}], trim=True)
        self.assertEqual(self.pipeline.decodes[0].shape[2], 2)
        self.assertTrue(torch.equal(self.pipeline.decodes[0], self.pipeline.latents[0]))
        self.assertEqual(getattr(self.wgp, REPORTS_ATTR)[0]["references"][0]["mod"], "walk")

    def test_multi_image_and_copies_have_distinct_labels(self):
        self.add_mod("views", frames=2)
        self.run_numbered([{"mod": "views", "copies": 2}])
        refs = getattr(self.wgp, REPORTS_ATTR)[0]["references"]
        self.assertEqual([r["label"] for r in refs], [f"<Picture {i}>" for i in range(1, 5)])
        self.assertEqual([(r["copy"], r["frame"]) for r in refs], [(0, 0), (0, 1), (1, 0), (1, 1)])

    def test_zero_strength_does_not_load_or_take_a_number(self):
        self.run_numbered([{"mod": "missing", "strength": 0}, {"mod": "alice"}])
        refs = getattr(self.wgp, REPORTS_ATTR)[0]["references"]
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["label"], "<Picture 1>")

    def test_missing_mod_fails_and_cleans_pipeline_state(self):
        with self.assertRaises(KeyError):
            self.run_numbered([{"mod": "missing"}])
        self.assertFalse(hasattr(self.pipeline, STATE_ATTR))
        self.assertEqual(self.pipeline.encodings, [])

    def test_decode_failure_is_not_swallowed(self):
        def fail(_):
            raise RuntimeError("decode failed")
        self.pipeline.vae.decode = fail
        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            self.run_numbered()
        self.assertFalse(hasattr(self.pipeline, STATE_ATTR))
        self.assertEqual(self.pipeline.encodings, [])

    def test_basic_request_on_warm_pipeline_has_no_stale_refs(self):
        self.run_numbered()
        count = len(self.pipeline.decodes)
        self.pipeline.generate("basic")
        self.assertEqual(len(self.pipeline.decodes), count)
        self.assertEqual(self.pipeline.encodings[-1], [])

    def test_changed_strength_is_decoded_again_on_warm_pipeline(self):
        self.run_numbered()
        self.run_numbered([{"mod": "alice", "strength": 0.25}])
        self.assertEqual(self.pipeline.decodes[-1].mean().item(), 0.25)
        self.assertEqual(len(self.pipeline.decodes), 2)

    def test_no_double_hook_installation(self):
        install_numbered_hooks(self.wgp, self.patches, self.storage, self.h3)
        self.run_numbered()
        self.assertEqual(len(self.pipeline.latents), 1)
        self.assertEqual(len(self.pipeline.decodes), 1)

    def test_sliding_window_rejected_at_pipeline_boundary(self):
        with self.assertRaisesRegex(ValueError, "sliding windows"):
            self.run_numbered(window_no=2, input_video=torch.zeros(3, 1, 32, 32), prefix_frames_count=1)
        self.assertEqual(self.pipeline.decodes, [])

    def test_too_many_copies_rejected_before_decoding(self):
        with self.assertRaisesRegex(ValueError, "at most 9"):
            self.run_numbered([{"mod": "alice", "copies": 10}])
        self.assertEqual(self.pipeline.decodes, [])

    def test_more_than_three_videos_rejected(self):
        with self.assertRaisesRegex(ValueError, "at most 3"):
            self.run_numbered([{"mod": "walk"}] * 4)

    def test_audio_refmod_rejected(self):
        self.add_mod("voice", "audio")
        with self.assertRaisesRegex(ValueError, "image/video"):
            self.run_numbered([{"mod": "voice"}])

    def test_video_control_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "control videos"):
            self.run_numbered([{"mod": "walk"}], video_prompt_type="VU")


if __name__ == "__main__":
    unittest.main()
