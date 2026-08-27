#!/usr/bin/env python3
"""Authenticated client for the project's Fizgig and WanGP REST API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


class ClientError(RuntimeError):
    def __init__(self, payload: dict[str, Any]):
        super().__init__(str(payload))
        self.payload = payload


class RestClient:
    def __init__(self) -> None:
        base_url = os.environ.get("FIZGIG_REST_URL") or os.environ.get("WANGP_URL")
        missing = []
        if not base_url:
            missing.append("FIZGIG_REST_URL (or WANGP_URL)")
        modal_key = os.environ.get("MODAL_KEY")
        if not modal_key:
            missing.append("MODAL_KEY")
        modal_secret = os.environ.get("MODAL_SECRET")
        if not modal_secret:
            missing.append("MODAL_SECRET")
        if missing:
            raise ClientError(
                {
                    "error": {
                        "type": "configuration",
                        "message": "missing environment variables: " + ", ".join(missing),
                    }
                }
            )
        assert base_url is not None
        assert modal_key is not None
        assert modal_secret is not None
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Accept": "application/json",
            "Modal-Key": modal_key,
            "Modal-Secret": modal_secret,
        }

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> Any:
        data = None
        headers = dict(self.headers)
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                detail = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                detail = raw.decode("utf-8", errors="replace")
            raise ClientError(
                {
                    "error": {
                        "type": "http",
                        "status": exc.code,
                        "detail": detail,
                    }
                }
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ClientError(
                {
                    "error": {
                        "type": "transport",
                        "message": str(reason),
                    }
                }
            ) from exc
        try:
            return json.loads(payload) if payload else None
        except json.JSONDecodeError as exc:
            raise ClientError(
                {
                    "error": {
                        "type": "response",
                        "message": "API returned a non-JSON response",
                    }
                }
            ) from exc


def load_object(path: str, field: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClientError(
            {
                "error": {
                    "type": "input",
                    "message": f"could not read {field} JSON: {exc}",
                }
            }
        ) from exc
    if not isinstance(value, dict):
        raise ClientError(
            {
                "error": {
                    "type": "input",
                    "message": f"{field} JSON must contain an object",
                }
            }
        )
    return value


def training_body(args: argparse.Namespace) -> dict[str, Any]:
    body: dict[str, Any] = {
        "family": args.family,
        "dataset": args.dataset,
        "output_name": args.output_name,
        "preset": args.preset,
    }
    optional = {
        "trigger_word": args.trigger_word,
        "epochs": args.epochs,
    }
    body.update({key: value for key, value in optional.items() if value is not None})
    return body


def job_id(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("job ID must be a UUID returned by the API") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Operate Fizgig training and WanGP generation through REST."
    )
    commands = parser.add_subparsers(dest="operation", required=True)
    commands.add_parser("health")

    submit = commands.add_parser("training-submit")
    submit.add_argument(
        "--family",
        choices=("minimax_h3", "krea2"),
        required=True,
    )
    submit.add_argument("--dataset", required=True)
    submit.add_argument("--output-name", required=True)
    submit.add_argument(
        "--preset",
        choices=(
            "h3_character_fast",
            "h3_character_quality",
            "krea2_defaults",
            "krea2_ultra_fast",
        ),
        required=True,
    )
    submit.add_argument("--trigger-word")
    submit.add_argument("--epochs", type=int)

    for operation in (
        "training-status",
        "training-pause",
        "training-resume",
        "training-cancel",
        "generation-status",
        "generation-cancel",
    ):
        command = commands.add_parser(operation)
        command.add_argument("job_id", type=job_id)

    models = commands.add_parser("models")
    models.add_argument("--family")
    models.add_argument("--type", dest="model_type")

    for operation in ("model-defaults", "model-schema"):
        command = commands.add_parser(operation)
        command.add_argument("model")

    generate = commands.add_parser("generation-submit")
    generate.add_argument("--model", required=True)
    generate.add_argument(
        "--kind",
        choices=("image", "video", "audio"),
        help=(
            "Optional output kind. The API infers it from model metadata when "
            "omitted and rejects mismatches."
        ),
    )
    generate.add_argument(
        "--params",
        required=True,
        help="Path to a JSON object containing native WanGP parameters.",
    )
    return parser


def execute(client: RestClient, args: argparse.Namespace) -> Any:
    operation = args.operation
    if operation == "health":
        return client.request("GET", "/health")
    if operation == "training-submit":
        return client.request("POST", "/training/jobs", training_body(args))
    if operation.startswith("training-"):
        action = operation.removeprefix("training-")
        if action == "status":
            return client.request("GET", f"/training/jobs/{args.job_id}")
        return client.request("POST", f"/training/jobs/{args.job_id}/{action}")
    if operation == "models":
        query = {
            key: value
            for key, value in {
                "family": args.family,
                "type": args.model_type,
            }.items()
            if value is not None
        }
        suffix = "?" + urllib.parse.urlencode(query) if query else ""
        return client.request("GET", "/models" + suffix)
    if operation == "model-defaults":
        model = urllib.parse.quote(args.model, safe="")
        return client.request("GET", f"/models/{model}/defaults")
    if operation == "model-schema":
        model = urllib.parse.quote(args.model, safe="")
        return client.request("GET", f"/models/{model}/schema")
    if operation == "generation-submit":
        params = load_object(args.params, "params")
        body: dict[str, Any] = {"model": args.model, "params": params}
        if args.kind is not None:
            body["kind"] = args.kind
        return client.request(
            "POST",
            "/jobs",
            body,
        )
    if operation == "generation-status":
        return client.request("GET", f"/jobs/{args.job_id}")
    if operation == "generation-cancel":
        return client.request("POST", f"/jobs/{args.job_id}/cancel")
    raise AssertionError(f"unsupported operation: {operation}")


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = execute(RestClient(), args)
    except ClientError as exc:
        print(json.dumps(exc.payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
