"""Download the public baseline checkpoint with integrity verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Protocol


BASELINE_V1_URL = (
    "https://github.com/yuraira/ecg-guard/releases/download/"
    "baseline-v1/best_model.pt"
)
BASELINE_V1_SIZE_BYTES = 23_579_661
DOWNLOAD_LIMIT_BYTES = 64 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024


class Response(Protocol):
    """Minimal response interface used by the downloader."""

    headers: object

    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> Response: ...

    def __exit__(self, *args: object) -> None: ...


OpenUrl = Callable[..., Response]


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_v1_digest() -> str:
    """Read the locked checkpoint digest from the packaged protocol."""
    protocol_path = Path(__file__).with_name("baseline_v1_inference.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    digest = str(protocol.get("checkpoint_sha256", "")).lower()
    if len(digest) != 64:
        raise RuntimeError("the inference protocol has no valid checkpoint hash")
    return digest


def _content_length(response: Response) -> int | None:
    headers = response.headers
    value = getattr(headers, "get", lambda _name: None)("Content-Length")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError("checkpoint response has an invalid size") from error


def download_checkpoint(
    destination: Path,
    *,
    url: str = BASELINE_V1_URL,
    expected_digest: str | None = None,
    expected_size: int = BASELINE_V1_SIZE_BYTES,
    open_url: OpenUrl = urllib.request.urlopen,
) -> Path:
    """Atomically download and verify the public baseline checkpoint."""
    expected_digest = expected_digest or baseline_v1_digest()
    expected_digest = expected_digest.lower()
    if len(expected_digest) != 64:
        raise ValueError("expected_digest must be a SHA-256 hex digest")
    if expected_size <= 0 or expected_size > DOWNLOAD_LIMIT_BYTES:
        raise ValueError("expected_size is outside the download limit")

    destination = destination.expanduser().resolve()
    if (
        destination.is_file()
        and destination.stat().st_size == expected_size
        and sha256_file(destination) == expected_digest
    ):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "ecg-guard-baseline-v1"},
            )
            with open_url(request, timeout=60) as response:
                declared_size = _content_length(response)
                if declared_size is not None and declared_size != expected_size:
                    raise RuntimeError(
                        "checkpoint response size does not match the release manifest"
                    )
                digest = hashlib.sha256()
                downloaded = 0
                while chunk := response.read(CHUNK_SIZE):
                    downloaded += len(chunk)
                    if downloaded > DOWNLOAD_LIMIT_BYTES:
                        raise RuntimeError("checkpoint download exceeds the size limit")
                    temporary.write(chunk)
                    digest.update(chunk)

        if downloaded != expected_size:
            raise RuntimeError(
                "downloaded checkpoint size does not match the release manifest"
            )
        if digest.hexdigest() != expected_digest:
            raise RuntimeError(
                "downloaded checkpoint does not match the inference protocol"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    """Download the verified baseline checkpoint for an ephemeral host."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/ecg-guard/best_model.pt"),
        help="verified checkpoint output path",
    )
    args = parser.parse_args()
    destination = download_checkpoint(args.output)
    print(f"verified baseline checkpoint ready: {destination}")


if __name__ == "__main__":
    main()
