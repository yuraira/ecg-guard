"""Verify and stage the frozen baseline checkpoint for a public release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch


REQUIRED_CHECKPOINT_KEYS = {
    "epoch",
    "model_state_dict",
    "optimizer_state_dict",
    "validation_metrics",
    "normalization",
    "config",
}
EXPECTED_CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_commit(repository: Path) -> str | None:
    """Read the current Git commit without making the release depend on Git."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def source_is_dirty(repository: Path) -> bool | None:
    """Return whether tracked or untracked source differs from HEAD."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def source_timestamp(repository: Path) -> int | None:
    """Return a reproducible release timestamp from the source commit."""
    configured = os.environ.get("SOURCE_DATE_EPOCH")
    if configured:
        return int(configured)
    result = subprocess.run(
        ["git", "show", "-s", "--format=%ct", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return int(value) if value.isdigit() else None


def source_contains_commit(repository: Path, commit: str) -> bool | None:
    """Return whether one commit is an ancestor of the release source."""
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def iso_timestamp(epoch: int) -> str:
    """Format one Unix timestamp as stable UTC metadata."""
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat()


def validate_container_provenance(
    repository: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Cross-check the final image provenance against its SBOM and model."""
    provenance_path = (
        repository / "sbom" / "container-runtime.provenance.json"
    )
    sbom_path = repository / "sbom" / "container-runtime.cdx.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    sbom = json.loads(sbom_path.read_text(encoding="utf-8"))

    expected_sbom_hash = str(provenance["sbom"]["sha256"]).lower()
    if sha256_file(sbom_path) != expected_sbom_hash:
        raise ValueError("container SBOM hash does not match provenance")
    if sbom.get("bomFormat") != provenance["sbom"]["format"]:
        raise ValueError("container SBOM format does not match provenance")
    if sbom.get("specVersion") != provenance["sbom"]["spec_version"]:
        raise ValueError("container SBOM version does not match provenance")

    components = list(sbom.get("components", ()))
    python_components = [
        component
        for component in components
        if str(component.get("purl", "")).startswith("pkg:pypi/")
    ]
    debian_components = [
        component
        for component in components
        if str(component.get("purl", "")).startswith("pkg:deb/debian/")
    ]
    counts = {
        "component_count": len(components),
        "python_component_count": len(python_components),
        "debian_component_count": len(debian_components),
    }
    for name, actual in counts.items():
        if actual != int(provenance["sbom"][name]):
            raise ValueError(f"container SBOM {name} does not match provenance")

    expected_checkpoint = str(protocol["checkpoint_sha256"]).lower()
    recorded_checkpoint = str(provenance["checkpoint"]["sha256"]).lower()
    if recorded_checkpoint != expected_checkpoint:
        raise ValueError("container provenance checkpoint hash is not locked")
    return provenance


def validate_checkpoint(
    checkpoint_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Validate artifact integrity and the minimum reproducibility contract."""
    expected_digest = str(protocol["checkpoint_sha256"]).lower()
    actual_digest = sha256_file(checkpoint_path)
    if actual_digest != expected_digest:
        raise ValueError(
            "checkpoint SHA-256 does not match the locked inference protocol: "
            f"expected {expected_digest}, got {actual_digest}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    missing = REQUIRED_CHECKPOINT_KEYS - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing)}")

    config = checkpoint["config"]
    if tuple(config.get("classes", ())) != EXPECTED_CLASSES:
        raise ValueError("checkpoint class order is not the frozen baseline order")
    if config.get("test_fold_used_for_model_selection") is not False:
        raise ValueError("checkpoint does not preserve the untouched-test contract")

    for key in ("data_dir", "output_dir"):
        value = str(config.get(key, ""))
        if (
            PureWindowsPath(value).is_absolute()
            or PurePosixPath(value).is_absolute()
        ):
            raise ValueError(f"checkpoint config contains an absolute {key}")

    return {
        "sha256": actual_digest,
        "bytes": checkpoint_path.stat().st_size,
        "epoch": int(checkpoint["epoch"]),
        "classes": list(EXPECTED_CLASSES),
        "normalization": checkpoint["normalization"],
        "validation_metrics": checkpoint["validation_metrics"],
    }


def copy_asset(source: Path, destination: Path) -> None:
    """Copy one required release file and fail on an absent source."""
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, destination)


def build_release_package(
    repository: Path,
    checkpoint_path: Path,
    output_directory: Path,
    *,
    allow_dirty: bool,
) -> Path:
    """Create a self-describing, checksummed model release directory."""
    dirty = source_is_dirty(repository)
    if dirty is None and not allow_dirty:
        raise RuntimeError("unable to verify that the Git worktree is clean")
    if dirty and not allow_dirty:
        raise RuntimeError(
            "refusing to package a dirty Git worktree; commit the intended "
            "release source or pass --allow-dirty for local inspection only"
        )

    protocol_path = (
        repository
        / "src"
        / "ecg_guard"
        / "resources"
        / "baseline_v1_inference.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    checkpoint_summary = validate_checkpoint(checkpoint_path, protocol)
    container_provenance = validate_container_provenance(
        repository,
        protocol,
    )
    image_source_commit = str(container_provenance["source_commit"])
    if source_contains_commit(repository, image_source_commit) is not True:
        raise RuntimeError(
            "container image source commit is not an ancestor of the "
            "release source"
        )

    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"release directory is not empty: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    assets = {
        checkpoint_path: output_directory / "best_model.pt",
        repository / "docs" / "model_card.md": output_directory
        / "MODEL_CARD.md",
        protocol_path: output_directory / "baseline_v1_inference.json",
        repository / "LICENSE": output_directory / "LICENSE",
        repository / "THIRD_PARTY_NOTICES.md": output_directory
        / "THIRD_PARTY_NOTICES.md",
        repository / "sbom" / "direct-dependencies.cdx.json": output_directory
        / "direct-dependencies.cdx.json",
        repository / "sbom" / "container-runtime.cdx.json": output_directory
        / "container-runtime.cdx.json",
        repository / "sbom" / "container-runtime.provenance.json": output_directory
        / "container-runtime.provenance.json",
    }
    for source, destination in assets.items():
        copy_asset(source, destination)

    release_readme = f"""# ECG Guard Baseline v1

This package contains the frozen ECG Guard baseline-v1 checkpoint.

- Architecture: residual 1D CNN
- Input: 100 Hz, 10-second, standard 12-lead ECG shaped `(12, 1000)`
- Outputs: NORM, MI, STTC, CD and HYP multilabel probabilities
- Checkpoint SHA-256: `{checkpoint_summary["sha256"]}`
- Intended use: research and education only
- Not validated for diagnosis, screening, treatment or emergency decisions
- `container-runtime.cdx.json` is the final Docker image SBOM; the direct
  dependency SBOM is retained separately for source-level review

Verify `SHA256SUMS.txt` before loading the PyTorch checkpoint. The source code is
licensed under Apache-2.0. PTB-XL attribution and separate CC BY 4.0 terms are
documented in `THIRD_PARTY_NOTICES.md`; the PTB-XL source data is not included.
"""
    (output_directory / "README.md").write_text(
        release_readme,
        encoding="utf-8",
    )

    manifest_files = []
    for path in sorted(output_directory.iterdir()):
        if path.is_file():
            manifest_files.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    created_at = source_timestamp(repository)
    if created_at is None:
        raise RuntimeError("unable to read a reproducible source timestamp")
    manifest = {
        "schema_version": "1.0",
        "model_version": str(protocol["model_version"]),
        "created_at": iso_timestamp(created_at),
        "source_repository": "https://github.com/yuraira/ecg-guard",
        "source_commit": source_commit(repository),
        "source_dirty": dirty,
        "checkpoint": checkpoint_summary,
        "container_image": container_provenance["image"],
        "container_sbom": container_provenance["sbom"],
        "license": "Apache-2.0",
        "third_party_dataset": {
            "name": "PTB-XL",
            "version": "1.0.3",
            "license": "CC-BY-4.0",
            "source_data_included": False,
        },
        "files": manifest_files,
    }
    manifest_path = output_directory / "release-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    checksum_paths = [
        path
        for path in sorted(output_directory.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.txt"
    ]
    checksum_text = "".join(
        f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths
    )
    (output_directory / "SHA256SUMS.txt").write_text(
        checksum_text,
        encoding="utf-8",
    )
    return output_directory


def build_release_archive(
    release_directory: Path,
    archive_path: Path,
    *,
    epoch: int,
) -> Path:
    """Create a deterministic ZIP containing the complete release directory."""
    if archive_path.exists():
        raise FileExistsError(f"release archive already exists: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    minimum_zip_epoch = 315532800
    timestamp = datetime.fromtimestamp(
        max(epoch, minimum_zip_epoch),
        tz=UTC,
    )
    zip_datetime = (
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
    )
    root_name = release_directory.name
    with zipfile.ZipFile(
        archive_path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(release_directory.iterdir()):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(
                f"{root_name}/{path.name}",
                date_time=zip_datetime,
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/baseline/best_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/ecg-guard-baseline-v1"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("dist/ecg-guard-baseline-v1.zip"),
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Build an inspection package that must not be published.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    checkpoint = (
        args.checkpoint
        if args.checkpoint.is_absolute()
        else repository / args.checkpoint
    )
    output_directory = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repository / args.output_dir
    )
    archive_path = (
        args.archive
        if args.archive.is_absolute()
        else repository / args.archive
    )
    built = build_release_package(
        repository,
        checkpoint,
        output_directory,
        allow_dirty=args.allow_dirty,
    )
    epoch = source_timestamp(repository)
    if epoch is None:
        raise RuntimeError("unable to read a reproducible source timestamp")
    archive = build_release_archive(
        built,
        archive_path,
        epoch=epoch,
    )
    print(f"release_directory={built}")
    print(f"release_archive={archive}")
    print(f"release_archive_sha256={sha256_file(archive)}")
    print(f"checkpoint_sha256={sha256_file(built / 'best_model.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
