from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.prepare_model_release import (
    EXPECTED_CLASSES,
    build_release_archive,
    sha256_file,
    validate_checkpoint,
    validate_container_provenance,
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


def test_validate_container_provenance_cross_checks_sbom(
    tmp_path: Path,
) -> None:
    sbom_directory = tmp_path / "sbom"
    sbom_directory.mkdir()
    sbom_path = sbom_directory / "container-runtime.cdx.json"
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.7",
        "components": [
            {"name": "library", "purl": "pkg:pypi/library@1.0"},
            {"name": "libc", "purl": "pkg:deb/debian/libc@1.0"},
        ],
    }
    sbom_path.write_text(
        json.dumps(sbom),
        encoding="utf-8",
    )
    provenance = {
        "source_commit": "0" * 40,
        "sbom": {
            "format": "CycloneDX",
            "spec_version": "1.7",
            "sha256": sha256_file(sbom_path),
            "component_count": 2,
            "python_component_count": 1,
            "debian_component_count": 1,
        },
        "checkpoint": {"sha256": "1" * 64},
    }
    (sbom_directory / "container-runtime.provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )

    result = validate_container_provenance(
        tmp_path,
        {"checkpoint_sha256": "1" * 64},
    )

    assert result["sbom"]["component_count"] == 2
