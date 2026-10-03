"""Bounded video transfers for Sol, independent of the WanGP application."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlsplit

from sol_refiner.common import MAX_INPUT_BYTES, validate_url

S3_KEYS = ("S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_ENDPOINT", "S3_BUCKET", "S3_REGION")


def s3_client():
    import boto3
    from botocore.config import Config

    missing = [key for key in S3_KEYS if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"studio-s3 is missing: {', '.join(missing)}")
    return boto3.client(
        "s3", endpoint_url=os.environ["S3_ENDPOINT"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        region_name=os.environ["S3_REGION"],
        config=Config(s3={"addressing_style": "path"}, connect_timeout=30, read_timeout=60),
    ), os.environ["S3_BUCKET"]


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_bounded(stream, destination: Path, expected: int | None, limit: int) -> None:
    if expected is not None and not 0 < expected <= limit:
        raise ValueError("input is empty or exceeds the download size limit")
    count, start = 0, time.monotonic()
    with destination.open("xb") as output:
        while block := stream.read(1024 * 1024):
            count += len(block)
            if count > limit or time.monotonic() - start > 600:
                raise ValueError("input exceeded the download size or time limit")
            output.write(block)
    if count == 0 or (expected is not None and count != expected):
        raise ValueError("input failed size verification")


def _public_https(url: str) -> None:
    validate_url(url)
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        raise ValueError("redirects must remain HTTPS")
    addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("HTTPS input must resolve to public addresses")


class _PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _public_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_input(url: str, destination: Path, client=None, bucket: str | None = None,
                   limit: int = MAX_INPUT_BYTES) -> Path:
    url = validate_url(url)
    parsed = urlsplit(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    try:
        if parsed.scheme == "s3":
            if client is None or bucket is None:
                client, bucket = s3_client()
            if parsed.netloc != bucket:
                raise ValueError("input bucket does not match configured S3_BUCKET")
            key = unquote(parsed.path[1:])
            if not key or key.endswith("/") or "\x00" in key:
                raise ValueError("S3 input must name a video object")
            response = client.get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            try:
                _copy_bounded(body, partial, int(response["ContentLength"]), limit)
            finally:
                body.close()
            expected_digest = response.get("Metadata", {}).get("sha256")
            if expected_digest and digest_file(partial) != expected_digest:
                raise ValueError("input failed SHA-256 verification")
        else:
            _public_https(url)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _PublicRedirect())
            try:
                with opener.open(url, timeout=60) as response:
                    length = response.headers.get("Content-Length")
                    _copy_bounded(response, partial, int(length) if length else None, limit)
            except urllib.error.URLError:
                # Do not put a presigned URL in a persisted job error.
                raise RuntimeError("HTTPS input download failed; check access and URL expiry") from None
        if destination.exists():
            raise FileExistsError("input destination already exists")
        partial.replace(destination)
        return destination
    finally:
        partial.unlink(missing_ok=True)


def upload_output(path: Path, job_id: str, client, bucket: str) -> dict:
    key = f"runninghub/sol/{job_id}/refined.mp4"
    digest, size = digest_file(path), path.stat().st_size
    client.upload_file(str(path), bucket, key, ExtraArgs={
        "ContentType": "video/mp4", "Metadata": {"sha256": digest},
    })
    head = client.head_object(Bucket=bucket, Key=key)
    if int(head.get("ContentLength", -1)) != size or head.get("Metadata", {}).get("sha256") != digest:
        raise RuntimeError("uploaded output failed size/SHA-256 metadata verification")
    return {"storage": "s3", "bucket": bucket, "key": key, "uri": f"s3://{bucket}/{key}",
            "filename": path.name, "media_type": "video/mp4", "size_bytes": size, "sha256": digest}
