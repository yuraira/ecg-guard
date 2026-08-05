from __future__ import annotations

import hashlib
import io
from pathlib import Path
from urllib.request import Request

import pytest

from ecg_guard.resources.fetch_checkpoint import download_checkpoint


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}


def response_for(payload: bytes):
    def open_url(request: Request, timeout: int) -> FakeResponse:
        assert request.full_url.startswith("https://")
        assert timeout == 60
        return FakeResponse(payload)

    return open_url


def test_download_checkpoint_verifies_and_replaces(tmp_path: Path) -> None:
    payload = b"verified baseline fixture"
    destination = tmp_path / "models" / "best_model.pt"

    result = download_checkpoint(
        destination,
        expected_digest=hashlib.sha256(payload).hexdigest(),
        expected_size=len(payload),
        open_url=response_for(payload),
    )

    assert result == destination.resolve()
    assert destination.read_bytes() == payload
    assert list(destination.parent.glob("*.part")) == []


def test_download_checkpoint_reuses_verified_file(tmp_path: Path) -> None:
    payload = b"already verified"
    destination = tmp_path / "best_model.pt"
    destination.write_bytes(payload)

    def fail_if_called(*_args: object, **_kwargs: object) -> FakeResponse:
        raise AssertionError("network should not be used for a verified file")

    result = download_checkpoint(
        destination,
        expected_digest=hashlib.sha256(payload).hexdigest(),
        expected_size=len(payload),
        open_url=fail_if_called,
    )

    assert result == destination.resolve()


def test_download_checkpoint_rejects_digest_mismatch(tmp_path: Path) -> None:
    destination = tmp_path / "best_model.pt"
    payload = b"unexpected checkpoint"

    with pytest.raises(RuntimeError, match="does not match"):
        download_checkpoint(
            destination,
            expected_digest="0" * 64,
            expected_size=len(payload),
            open_url=response_for(payload),
        )

    assert not destination.exists()
    assert list(tmp_path.glob("*.part")) == []
