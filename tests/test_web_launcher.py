from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ecg_guard.web.launcher import verify_checkpoint


def test_verify_checkpoint_accepts_protocol_digest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_model.pt"
    checkpoint.write_bytes(b"locked checkpoint fixture")
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps({"checkpoint_sha256": expected}),
        encoding="utf-8",
    )

    assert verify_checkpoint(checkpoint, protocol) == expected


def test_verify_checkpoint_rejects_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_model.pt"
    checkpoint.write_bytes(b"unexpected checkpoint")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps({"checkpoint_sha256": "0" * 64}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="does not match"):
        verify_checkpoint(checkpoint, protocol)
