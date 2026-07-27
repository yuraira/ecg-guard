"""Prepare PTB-XL metadata, labels, splits, waveforms, and a sample plot."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import wfdb

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DIAGNOSTIC_CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
LEAD_NAMES = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
EXPECTED_SAMPLING_FREQUENCY = 100
EXPECTED_SAMPLES = 1_000


def parse_scp_codes(value: str | Mapping[str, float]) -> dict[str, float]:
    """Parse the serialized PTB-XL SCP code dictionary without evaluating code."""
    if isinstance(value, Mapping):
        parsed = dict(value)
    elif isinstance(value, str):
        parsed = ast.literal_eval(value)
    else:
        raise TypeError(f"scp_codes must be a string or mapping, got {type(value).__name__}")

    if not isinstance(parsed, dict):
        raise ValueError("scp_codes must contain a dictionary")
    if not all(isinstance(code, str) for code in parsed):
        raise ValueError("every SCP code must be a string")
    return parsed


def load_diagnostic_mapping(data_dir: Path) -> dict[str, str]:
    """Map diagnostic SCP codes to the five PTB-XL diagnostic superclasses."""
    statements_path = data_dir / "scp_statements.csv"
    statements = pd.read_csv(statements_path, index_col=0)
    diagnostic = statements.loc[
        statements["diagnostic"].eq(1.0) & statements["diagnostic_class"].notna(),
        "diagnostic_class",
    ]
    mapping = diagnostic.astype(str).to_dict()

    unexpected = set(mapping.values()) - set(DIAGNOSTIC_CLASSES)
    if unexpected:
        raise ValueError(f"unexpected diagnostic classes: {sorted(unexpected)}")
    return mapping


def aggregate_diagnostic_classes(
    scp_codes: Mapping[str, float],
    diagnostic_mapping: Mapping[str, str],
) -> tuple[str, ...]:
    """Aggregate present SCP codes into a deterministic superclass tuple."""
    present = {
        diagnostic_mapping[code]
        for code in scp_codes
        if code in diagnostic_mapping
    }
    return tuple(label for label in DIAGNOSTIC_CLASSES if label in present)


def split_from_fold(strat_fold: int) -> str:
    """Apply the PTB-XL recommended train, validation, and test split."""
    fold = int(strat_fold)
    if 1 <= fold <= 8:
        return "train"
    if fold == 9:
        return "validation"
    if fold == 10:
        return "test"
    raise ValueError(f"strat_fold must be between 1 and 10, got {fold}")


def prepare_metadata(data_dir: Path) -> pd.DataFrame:
    """Load PTB-XL metadata and add parsed codes, labels, and split columns."""
    database_path = data_dir / "ptbxl_database.csv"
    metadata = pd.read_csv(database_path)

    required_columns = {
        "ecg_id",
        "patient_id",
        "scp_codes",
        "strat_fold",
        "filename_lr",
    }
    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(f"missing PTB-XL metadata columns: {sorted(missing)}")
    if metadata["patient_id"].isna().any():
        raise ValueError("patient_id contains missing values")

    diagnostic_mapping = load_diagnostic_mapping(data_dir)
    metadata["scp_codes_parsed"] = metadata["scp_codes"].map(parse_scp_codes)
    metadata["diagnostic_superclasses"] = metadata["scp_codes_parsed"].map(
        lambda codes: aggregate_diagnostic_classes(codes, diagnostic_mapping)
    )
    for label in DIAGNOSTIC_CLASSES:
        metadata[label] = metadata["diagnostic_superclasses"].map(
            lambda labels: int(label in labels)
        )
    metadata["has_diagnostic_superclass"] = metadata.loc[
        :, DIAGNOSTIC_CLASSES
    ].sum(axis=1).gt(0)
    metadata["split"] = metadata["strat_fold"].map(split_from_fold)

    validate_patient_splits(metadata)
    return metadata


def validate_patient_splits(metadata: pd.DataFrame) -> None:
    """Raise if a patient occurs in more than one dataset split."""
    required = {"patient_id", "split"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"missing split validation columns: {sorted(missing)}")

    patients_by_split = {
        split: set(group["patient_id"].astype(int))
        for split, group in metadata.groupby("split", sort=False)
    }
    expected_splits = {"train", "validation", "test"}
    if set(patients_by_split) != expected_splits:
        raise ValueError(
            f"expected splits {sorted(expected_splits)}, got {sorted(patients_by_split)}"
        )

    overlaps = {
        f"{left}/{right}": patients_by_split[left] & patients_by_split[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    }
    nonempty = {pair: ids for pair, ids in overlaps.items() if ids}
    if nonempty:
        counts = {pair: len(ids) for pair, ids in nonempty.items()}
        raise ValueError(f"patient leakage across splits: {counts}")


def make_label_matrix(metadata: pd.DataFrame) -> np.ndarray:
    """Return binary labels in DIAGNOSTIC_CLASSES order."""
    matrix = metadata.loc[:, DIAGNOSTIC_CLASSES].to_numpy(dtype=np.float32)
    if not np.isin(matrix, (0.0, 1.0)).all():
        raise ValueError("label matrix must be binary")
    return matrix


def load_waveform(record_path: str, data_dir: Path) -> np.ndarray:
    """Load and validate one 100 Hz waveform as (leads, samples)."""
    signal, fields = wfdb.rdsamp(str(data_dir / record_path))
    sampling_frequency = int(fields["fs"])
    signal_names = tuple(str(name).upper() for name in fields["sig_name"])

    if signal.shape != (EXPECTED_SAMPLES, len(LEAD_NAMES)):
        raise ValueError(f"unexpected waveform shape {signal.shape} for {record_path}")
    if sampling_frequency != EXPECTED_SAMPLING_FREQUENCY:
        raise ValueError(
            f"unexpected sampling frequency {sampling_frequency} for {record_path}"
        )
    if signal_names != LEAD_NAMES:
        raise ValueError(f"unexpected lead order {signal_names} for {record_path}")
    if not np.isfinite(signal).all():
        raise ValueError(f"non-finite waveform values in {record_path}")

    return signal.T.astype(np.float32, copy=False)


def load_waveforms(
    metadata: pd.DataFrame,
    data_dir: Path,
    *,
    limit: int | None = None,
) -> np.ndarray:
    """Load 100 Hz PTB-XL waveforms as (records, leads, samples)."""
    selected = metadata if limit is None else metadata.iloc[:limit]
    if selected.empty:
        raise ValueError("no waveform rows selected")

    waveforms: list[np.ndarray] = []
    for record_path in selected["filename_lr"]:
        waveforms.append(load_waveform(str(record_path), data_dir))

    return np.stack(waveforms)


def plot_twelve_lead_ecg(
    waveform: np.ndarray,
    output_path: Path,
    *,
    sampling_frequency: int = EXPECTED_SAMPLING_FREQUENCY,
) -> None:
    """Save a compact twelve-lead ECG plot."""
    expected_shape = (len(LEAD_NAMES), EXPECTED_SAMPLES)
    if waveform.shape != expected_shape:
        raise ValueError(f"expected waveform shape {expected_shape}, got {waveform.shape}")

    time_seconds = np.arange(waveform.shape[1]) / sampling_frequency
    figure, axes = plt.subplots(6, 2, figsize=(14, 12), sharex=True)
    for axis, lead_name, values in zip(axes.flat, LEAD_NAMES, waveform, strict=True):
        axis.plot(time_seconds, values, linewidth=0.8, color="#b42318")
        axis.set_title(lead_name, loc="left", fontsize=9)
        axis.set_ylabel("mV")
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")

    figure.suptitle("PTB-XL 12-lead ECG sample")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def summarize(
    metadata: pd.DataFrame,
    loaded_metadata: pd.DataFrame,
    waveforms: np.ndarray,
) -> str:
    """Build a deterministic, human-readable preparation summary."""
    labels = make_label_matrix(metadata)
    loaded_labels = make_label_matrix(loaded_metadata)
    class_counts = {
        label: int(count)
        for label, count in zip(
            DIAGNOSTIC_CLASSES,
            labels.sum(axis=0),
            strict=True,
        )
    }
    split_counts = metadata["split"].value_counts().reindex(
        ["train", "validation", "test"]
    )
    return "\n".join(
        (
            f"metadata_shape={metadata.shape}",
            f"waveform_shape={waveforms.shape}",
            f"loaded_label_shape={loaded_labels.shape}",
            f"split_counts={split_counts.astype(int).to_dict()}",
            f"class_counts={class_counts}",
            "records_without_diagnostic_superclass="
            f"{int((labels.sum(axis=1) == 0).sum())}",
            "patient_overlap=0",
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/ptb-xl"),
        help="PTB-XL 1.0.3 root directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="number of 100 Hz waveforms to load",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path("outputs/ptbxl_sample.png"),
        help="output path for the first twelve-lead ECG plot",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    metadata = prepare_metadata(args.data_dir)
    sample = metadata.iloc[: args.limit].copy()
    waveforms = load_waveforms(sample, args.data_dir)
    plot_twelve_lead_ecg(waveforms[0], args.plot_out)
    print(summarize(metadata, sample, waveforms))
    print(f"plot={args.plot_out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
