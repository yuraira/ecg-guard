from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from scripts.prepare_model_release import (
    EXPECTED_CLASSES,
    build_release_archive,
    sha256_file,
    validate_checkpoint,
)


def checkpoint_fixture() -> dict[str, object]:
    return {
        "epoch": 5,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "validation_metrics": {"macro_auroc": 0.9},
        "normalization": {"mean": 0.0, "std": 1.0},
        "config": {
            "classes": list(EXPECTED_CLASSES),
            "test_fold_used_for_model_selection": False,
            "data_dir": "data/raw/ptb-xl",
            "output_dir": "outputs/baseline",
        },
    }


def test_validate_checkpoint_accepts_locked_artifact(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save(checkpoint_fixture(), checkpoint_path)
    digest = sha256_file(checkpoint_path)

    summary = validate_checkpoint(
        checkpoint_path,
        {"checkpoint_sha256": digest},
    )

    assert summary["sha256"] == digest
    assert summary["classes"] == list(EXPECTED_CLASSES)


def test_validate_checkpoint_rejects_absolute_source_path(
    tmp_path: Path,
) -> None:
    checkpoint = checkpoint_fixture()
    checkpoint["config"]["data_dir"] = "C:\\private\\ptb-xl"
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save(checkpoint, checkpoint_path)
    digest = sha256_file(checkpoint_path)

    with pytest.raises(ValueError, match="absolute data_dir"):
        validate_checkpoint(
            checkpoint_path,
            {"checkpoint_sha256": digest},
        )


def test_release_zip_is_deterministic(tmp_path: Path) -> None:
    release_directory = tmp_path / "ecg-guard-baseline-v1"
    release_directory.mkdir()
    (release_directory / "README.md").write_text(
        "research only\n",
        encoding="utf-8",
    )
    (release_directory / "best_model.pt").write_bytes(b"model")

    first = build_release_archive(
        release_directory,
        tmp_path / "first.zip",
        epoch=1_700_000_000,
    )
    second = build_release_archive(
        release_directory,
        tmp_path / "second.zip",
        epoch=1_700_000_000,
    )

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()
