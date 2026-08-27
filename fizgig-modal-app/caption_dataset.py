#!/usr/bin/env python3
"""Auto-caption a flat Fizgig dataset with its Krea2 Qwen3-VL encoder."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
CAPTION_TASKS = ("training", "short", "detailed", "exhaustive", "style")


def caption_candidates(image_dir: Path, overwrite: bool = False) -> list[Path]:
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if overwrite:
        return images
    candidates = []
    for path in images:
        caption_path = path.with_suffix(".txt")
        if not caption_path.is_file() or not caption_path.read_text(
            encoding="utf-8"
        ).strip():
            candidates.append(path)
    return candidates


def training_caption(generated: str, trigger_word: str | None) -> str:
    generated = " ".join(generated.split()).strip()
    if not generated:
        raise ValueError("caption model returned an empty caption")
    return f"{trigger_word}, {generated}" if trigger_word else generated


def write_caption(image_path: Path, caption: str) -> Path:
    caption_path = image_path.with_suffix(".txt")
    temporary = caption_path.with_suffix(".txt.tmp")
    temporary.write_text(caption + "\n", encoding="utf-8")
    os.replace(temporary, caption_path)
    return caption_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Caption missing Fizgig sidecars with Krea2's Qwen3-VL encoder."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--trigger-word")
    parser.add_argument("--task", choices=CAPTION_TASKS, default="training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {args.image_dir}")
    images = caption_candidates(args.image_dir, args.overwrite)
    if not images:
        print(json.dumps({"captioned": 0, "skipped": "all captions already exist"}))
        return 0

    import torch

    from fizgig.krea2.embedder import CAPTION_TASKS as UPSTREAM_CAPTION_TASKS
    from fizgig.krea2.embedder import generate_caption
    from fizgig.krea2.utils import load_krea2_text_encoder

    _label, instruction, max_new_tokens = UPSTREAM_CAPTION_TASKS[args.task]
    encoder = load_krea2_text_encoder(
        args.text_encoder,
        dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    for index, image_path in enumerate(images, start=1):
        generated = generate_caption(
            encoder,
            str(image_path),
            max_new_tokens=max_new_tokens,
            detailed=args.task == "exhaustive",
            seed=args.seed + index - 1,
            instruction=instruction,
        )
        caption_path = write_caption(
            image_path,
            training_caption(generated, args.trigger_word),
        )
        print(
            json.dumps(
                {
                    "captioned": index,
                    "total": len(images),
                    "image": image_path.name,
                    "caption": caption_path.name,
                }
            ),
            flush=True,
        )
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
