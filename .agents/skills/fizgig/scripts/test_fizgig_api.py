from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fizgig_api import ClientError, RestClient, build_parser, execute, training_body


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        return {"ok": True}


class FizgigApiTests(unittest.TestCase):
    def test_training_flags_build_h3_request(self):
        args = build_parser().parse_args(
            [
                "training-submit",
                "--family",
                "minimax_h3",
                "--dataset",
                "anna",
                "--output-name",
                "anna_h3_v1",
                "--preset",
                "h3_character_quality",
            ]
        )
        self.assertEqual(
            training_body(args),
            {
                "family": "minimax_h3",
                "dataset": "anna",
                "output_name": "anna_h3_v1",
                "preset": "h3_character_quality",
            },
        )

    def test_training_lifecycle_operations_map_to_namespaced_routes(self):
        job_id = "5ed81512-87c4-4888-a438-d23693507c21"
        expected = {
            "training-status": ("GET", f"/training/jobs/{job_id}", None),
            "training-pause": ("POST", f"/training/jobs/{job_id}/pause", None),
            "training-resume": ("POST", f"/training/jobs/{job_id}/resume", None),
            "training-cancel": ("POST", f"/training/jobs/{job_id}/cancel", None),
        }
        for operation, call in expected.items():
            with self.subTest(operation=operation):
                client = FakeClient()
                args = build_parser().parse_args([operation, job_id])
                execute(client, args)
                self.assertEqual(client.calls, [call])

    def test_krea2_training_request_contains_only_public_intent(self):
        args = build_parser().parse_args(
            [
                "training-submit",
                "--family",
                "krea2",
                "--dataset",
                "linda",
                "--output-name",
                "linda_krea2_v1",
                "--preset",
                "krea2_defaults",
                "--trigger-word",
                "linda",
            ]
        )
        body = training_body(args)
        self.assertEqual(
            body,
            {
                "family": "krea2",
                "dataset": "linda",
                "output_name": "linda_krea2_v1",
                "preset": "krea2_defaults",
                "trigger_word": "linda",
            },
        )

    def test_generation_submission_loads_native_params(self):
        with tempfile.TemporaryDirectory() as directory:
            params_path = Path(directory) / "params.json"
            params_path.write_text(json.dumps({"prompt": "fox", "seed": 42}))
            args = build_parser().parse_args(
                [
                    "generation-submit",
                    "--model",
                    "krea2_turbo",
                    "--params",
                    str(params_path),
                ]
            )
            client = FakeClient()
            execute(client, args)
        self.assertEqual(
            client.calls,
            [
                (
                    "POST",
                    "/jobs",
                    {
                        "model": "krea2_turbo",
                        "params": {"prompt": "fox", "seed": 42},
                    },
                )
            ],
        )

    def test_missing_configuration_names_variables_without_values(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ClientError) as raised:
                RestClient()
        message = raised.exception.payload["error"]["message"]
        self.assertIn("FIZGIG_REST_URL", message)
        self.assertIn("MODAL_KEY", message)
        self.assertIn("MODAL_SECRET", message)


if __name__ == "__main__":
    unittest.main()
