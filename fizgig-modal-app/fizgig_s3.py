"""S3 dataset imports and verified artifact uploads using WanGP's studio-s3 secret."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, Callable

from fizgig_common import normalize_dataset_s3, parse_dataset_s3, utc_now, validate_component

S3_SECRET_NAME = "studio-s3"
S3_REQUIRED_KEYS = (
    "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_ENDPOINT", "S3_BUCKET", "S3_REGION",
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
# Keep in sync with the pinned fizgig.minimax.clip.VIDEO_EXTENSIONS.
VIDEO_SUFFIXES = {".mp4"}


@dataclass(frozen=True)
class DatasetLimits:
    max_images: int = 5000
    max_objects: int = 20000
    max_image_bytes: int = 128 * 1024 * 1024
    max_caption_bytes: int = 1024 * 1024
    max_total_bytes: int = 10 * 1024 * 1024 * 1024
    max_videos: int = 256
    max_video_bytes: int = 512 * 1024 * 1024


def create_s3_client():
    import boto3
    from botocore.config import Config

    missing = [name for name in S3_REQUIRED_KEYS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Modal secret {S3_SECRET_NAME} is missing required keys: {', '.join(missing)}")
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        endpoint_url=os.environ["S3_ENDPOINT"],
        region_name=os.environ["S3_REGION"],
        config=Config(s3={"addressing_style": "path"}, connect_timeout=30, read_timeout=60),
    ), os.environ["S3_BUCKET"]


def plan_dataset(client, uri: str, bucket: str, limits: DatasetLimits = DatasetLimits(),
                 *, include_video: bool = False) -> dict[str, Any]:
    """List a prefix and pair media with captions; video is opt-in for RefMods."""
    _, prefix = parse_dataset_s3(uri, bucket)
    media, captions = {}, {}
    counts = {"image": 0, "video": 0}
    suffixes = IMAGE_SUFFIXES | (VIDEO_SUFFIXES if include_video else set())
    scanned = 0
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            scanned += 1
            if scanned > limits.max_objects:
                raise ValueError("S3 prefix contains too many objects; choose a narrower dataset folder")
            key = item["Key"]
            if not key.startswith(prefix):
                raise ValueError("S3 listing returned an object outside the requested prefix")
            relative = key[len(prefix):]
            if not relative or relative.endswith("/"):
                continue
            suffix = PurePosixPath(relative).suffix.lower()
            if suffix not in suffixes | {".txt"}:
                continue
            if ("\\" in relative or any(ord(char) < 32 for char in relative)
                    or any(part in {".", "..", ""} for part in relative.split("/"))):
                raise ValueError("S3 dataset contains an unsafe object path")
            # Media and its caption get the same hash stem, even across folders.
            kind = "image" if suffix in IMAGE_SUFFIXES else "video" if suffix in VIDEO_SUFFIXES else "caption"
            size = int(item["Size"])
            limit = {"image": limits.max_image_bytes, "video": limits.max_video_bytes,
                     "caption": limits.max_caption_bytes}[kind]
            if size < (0 if kind == "caption" else 1) or size > limit:
                raise ValueError(f"S3 {kind} is empty or exceeds its {limit}-byte limit: {key}")
            if not item.get("ETag"):
                raise ValueError(f"S3 listing omitted the ETag needed to snapshot: {key}")
            entry = {"key": key, "size_bytes": size, "etag": item["ETag"], "kind": kind}
            if kind != "caption":
                if relative not in media:
                    counts[kind] += 1
                media[relative] = entry
                count_limit = limits.max_images if kind == "image" else limits.max_videos
                if counts[kind] > count_limit:
                    raise ValueError(f"S3 dataset exceeds the {count_limit}-{kind} limit")
            else:
                stem = str(PurePosixPath(relative).with_suffix(""))
                if stem in captions:
                    raise ValueError(f"ambiguous caption files for media stem: {stem}")
                captions[stem] = entry
    if not media:
        raise ValueError("S3 dataset contains no supported images" + (" or MP4 videos" if include_video else ""))

    files = []
    caption_counts = {"image": 0, "video": 0}
    for relative, entry in sorted(media.items()):
        name = hashlib.sha256(relative.encode("utf-8")).hexdigest()
        # Preserve upstream's explicit mute marker when flattening clip names.
        if entry["kind"] == "video" and PurePosixPath(relative).stem.lower().endswith("_mute"):
            name += "_mute"
        files.append({**entry, "name": name + PurePosixPath(relative).suffix.lower()})
        caption = captions.get(str(PurePosixPath(relative).with_suffix("")))
        if caption is not None:
            files.append({**caption, "name": name + ".txt"})
            caption_counts[entry["kind"]] += 1
    total = sum(entry["size_bytes"] for entry in files)
    if total > limits.max_total_bytes:
        raise ValueError(f"S3 dataset exceeds the {limits.max_total_bytes}-byte total download limit")
    return {
        "schema_version": 2 if include_video else 1,
        "source": normalize_dataset_s3(uri),
        "image_count": counts["image"],
        "caption_count": sum(caption_counts.values()),
        "images_without_caption": counts["image"] - caption_counts["image"],
        **({"video_count": counts["video"], "media_count": len(media),
            "videos_without_caption": counts["video"] - caption_counts["video"],
            "media_without_caption": len(media) - sum(caption_counts.values())} if include_video else {}),
        "total_bytes": total,
        "files": files,
    }


def dataset_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: manifest[key] for key in (
        "source", "image_count", "video_count", "media_count", "caption_count",
        "images_without_caption", "videos_without_caption", "media_without_caption", "total_bytes",
    ) if key in manifest}


def _download(client, bucket: str, entry: dict[str, Any], destination: Path) -> str:
    # IfMatch prevents silently importing a different object after listing.
    response = client.get_object(Bucket=bucket, Key=entry["key"], IfMatch=entry["etag"])
    body = response["Body"]
    digest, size = hashlib.sha256(), 0
    try:
        if int(response["ContentLength"]) != entry["size_bytes"]:
            raise ValueError("S3 object size changed between listing and download")
        if response.get("ETag") != entry["etag"]:
            raise ValueError("S3 object ETag changed between listing and download")
        with destination.open("xb") as output:
            for chunk in iter(lambda: body.read(1024 * 1024), b""):
                size += len(chunk)
                if size > entry["size_bytes"]:
                    raise ValueError("S3 object exceeded its listed size during download")
                output.write(chunk)
                digest.update(chunk)
        if size != entry["size_bytes"]:
            raise ValueError("S3 object download was truncated")
        expected_digest = response.get("Metadata", {}).get("sha256")
        if expected_digest and digest.hexdigest() != expected_digest.lower():
            raise ValueError("S3 object failed SHA-256 metadata verification")
        if entry["kind"] == "caption":
            destination.read_text(encoding="utf-8")
        return digest.hexdigest()
    finally:
        body.close()


def materialize_dataset(
    client,
    uri: str,
    bucket: str,
    destination: Path,
    *,
    limits: DatasetLimits = DatasetLimits(),
    include_video: bool = False,
    progress: Callable[[int, int, int, int], None] | None = None,
) -> dict[str, Any]:
    """Publish a complete local snapshot, leaving no partial dataset on failure."""
    if destination.exists():
        raise FileExistsError("S3 dataset destination already exists")
    manifest = plan_dataset(client, uri, bucket, limits, include_video=include_video)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The temporary directory shares the destination filesystem so rename is
    # atomic. Never overwrite an existing run's snapshot or captions.
    with TemporaryDirectory(prefix=".s3-dataset-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        images = staging / "images"
        images.mkdir()
        downloaded = 0
        for index, entry in enumerate(manifest["files"], start=1):
            entry["sha256"] = _download(client, bucket, entry, images / entry["name"])
            downloaded += entry["size_bytes"]
            if progress:
                progress(index, len(manifest["files"]), downloaded, manifest["total_bytes"])
        manifest["created_at"] = utc_now()
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(destination)
    return manifest


def load_dataset_snapshot(destination: Path, uri: str, *, include_video: bool = False) -> dict[str, Any]:
    """Resume from persisted inputs and captions without consulting mutable S3."""
    try:
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("S3 dataset snapshot is missing or invalid; cannot resume safely") from exc
    schemas = {1, 2} if include_video else {1}
    if manifest.get("schema_version") not in schemas or manifest.get("source") != normalize_dataset_s3(uri):
        raise ValueError("S3 dataset snapshot does not match the requested source")
    media_names = set()
    suffixes = IMAGE_SUFFIXES | (VIDEO_SUFFIXES if include_video else set())
    for entry in manifest["files"]:
        # Captions are intentionally mutable: captioning/recaptioning writes
        # improved sidecars during training. Preserve those on resume.
        if entry["kind"] == "caption":
            continue
        if entry["kind"] not in ({"image", "video"} if include_video else {"image"}):
            raise ValueError("unsupported media in S3 dataset snapshot")
        name = entry["name"]
        if Path(name).name != name or Path(name).suffix.lower() not in suffixes:
            raise ValueError("invalid media path in S3 dataset snapshot")
        path = destination / "images" / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["size_bytes"]:
            raise ValueError("S3 dataset snapshot media is missing or changed")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError("S3 dataset snapshot media failed SHA-256 verification")
        media_names.add(name)
    actual = {path.name for path in (destination / "images").iterdir()
              if path.suffix.lower() in suffixes}
    if not media_names or media_names != actual:
        raise ValueError("S3 dataset snapshot media set changed")
    return manifest


S3_OUTPUT_PREFIX = "runninghub/fizgig"


def upload_artifact(client, bucket: str, job_id: str, path: Path, artifact_type: str) -> dict[str, Any]:
    """Publish one final artifact using WanGP's URI/size/SHA-256 output contract."""
    job_id = validate_component(job_id, "job_id")
    validate_component(path.stem, "artifact output_name")
    if artifact_type not in {"lora", "refmod"} or path.suffix != ".safetensors":
        raise ValueError("expected a final LoRA or RefMod safetensors artifact")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    if not size:
        raise ValueError("cannot publish an empty artifact")
    sha256 = digest.hexdigest()
    key = f"{S3_OUTPUT_PREFIX}/{job_id}/000-{path.name}"

    def matches(head):
        return (int(head.get("ContentLength", -1)) == size
                and head.get("Metadata", {}).get("sha256") == sha256)

    # Idempotent retry, but never silently replace a different object at this key.
    try:
        existing = client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if code not in {"404", "NoSuchKey", "NotFound"}:
            raise
        existing = None
    if existing is not None:
        if not matches(existing):
            raise FileExistsError(f"a different artifact already exists at s3://{bucket}/{key}")
    else:
        client.upload_file(str(path), bucket, key, ExtraArgs={
            "ContentType": "application/octet-stream", "Metadata": {"sha256": sha256},
        })
        if not matches(client.head_object(Bucket=bucket, Key=key)):
            raise RuntimeError(f"uploaded artifact verification failed for {key}")
    return {"storage": "s3", "bucket": bucket, "key": key,
            "uri": f"s3://{bucket}/{key}", "filename": path.name, "size_bytes": size,
            "media_type": "application/octet-stream", "sha256": sha256,
            "artifact_type": artifact_type}
