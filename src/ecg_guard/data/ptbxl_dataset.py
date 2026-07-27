"""PyTorch dataset and leakage-safe normalization for PTB-XL."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ecg_guard.data.prepare_ptbxl import (
    DIAGNOSTIC_CLASSES,
    load_waveform,
    make_label_matrix,
)


@dataclass(frozen=True)
class NormalizationStats:
    """Global scalar statistics fitted on training waveforms only."""

    mean: float
    std: float
    count: int
    records: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("normalization mean must be finite")
        if not np.isfinite(self.std) or self.std <= 0:
            raise ValueError("normalization std must be finite and positive")
        if self.count <= 0 or self.records <= 0:
            raise ValueError("normalization counts must be positive")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, float | int],
    ) -> NormalizationStats:
        return cls(
            mean=float(values["mean"]),
            std=float(values["std"]),
            count=int(values["count"]),
            records=int(values["records"]),
        )


def select_modeling_metadata(
    metadata: pd.DataFrame,
    split: str,
    *,
    limit: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Select labeled records from one official split.

    PTB-XL contains records without any of the five diagnostic superclasses.
    They remain in prepared metadata for auditability but are excluded from
    this five-label modeling task.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported split: {split}")

    required = {
        "ecg_id",
        "filename_lr",
        "split",
        "has_diagnostic_superclass",
        *DIAGNOSTIC_CLASSES,
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"missing modeling metadata columns: {sorted(missing)}")

    selected = metadata.loc[
        metadata["split"].eq(split) & metadata["has_diagnostic_superclass"]
    ].copy()
    if selected.empty:
        raise ValueError(f"no labeled records found for split={split}")

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected.sample(
            n=min(limit, len(selected)),
            random_state=seed,
            replace=False,
        ).sort_values("ecg_id")

    selected = selected.reset_index(drop=True)
    if make_label_matrix(selected).sum(axis=1).min() < 1:
        raise AssertionError("zero-label records entered the modeling cohort")
    return selected


def compute_training_normalization_stats(
    train_metadata: pd.DataFrame,
    data_dir: Path,
) -> NormalizationStats:
    """Fit one global mean and standard deviation using training records only."""
    if train_metadata.empty:
        raise ValueError("training metadata is empty")
    if set(train_metadata["split"].unique()) != {"train"}:
        raise ValueError("normalization statistics must use only the training split")
    if make_label_matrix(train_metadata).sum(axis=1).min() < 1:
        raise ValueError("normalization cohort contains zero-label records")

    total_count = 0
    running_mean = 0.0
    running_m2 = 0.0

    for record_path in train_metadata["filename_lr"]:
        waveform = load_waveform(str(record_path), data_dir).astype(
            np.float64,
            copy=False,
        )
        batch_count = waveform.size
        batch_mean = float(waveform.mean())
        batch_m2 = float(((waveform - batch_mean) ** 2).sum())

        if total_count == 0:
            running_mean = batch_mean
            running_m2 = batch_m2
            total_count = batch_count
            continue

        combined_count = total_count + batch_count
        delta = batch_mean - running_mean
        running_mean += delta * batch_count / combined_count
        running_m2 += (
            batch_m2
            + delta * delta * total_count * batch_count / combined_count
        )
        total_count = combined_count

    standard_deviation = float(np.sqrt(running_m2 / total_count))
    return NormalizationStats(
        mean=running_mean,
        std=standard_deviation,
        count=total_count,
        records=len(train_metadata),
    )


class PTBXLDataset(Dataset[dict[str, Tensor]]):
    """Lazily load normalized PTB-XL records and five-label targets."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        data_dir: Path,
        normalization: NormalizationStats,
    ) -> None:
        if metadata.empty:
            raise ValueError("dataset metadata is empty")
        labels = make_label_matrix(metadata)
        if labels.sum(axis=1).min() < 1:
            raise ValueError("dataset metadata contains zero-label records")

        self.metadata = metadata.reset_index(drop=True).copy()
        self.data_dir = Path(data_dir)
        self.normalization = normalization
        self.labels = labels

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.metadata.iloc[index]
        waveform = load_waveform(str(row["filename_lr"]), self.data_dir)
        normalized = (
            (waveform - self.normalization.mean) / self.normalization.std
        ).astype(np.float32, copy=False)

        age = float(row["age"]) if pd.notna(row.get("age")) else float("nan")
        sex = int(row["sex"]) if pd.notna(row.get("sex")) else -1
        return {
            "signal": torch.from_numpy(normalized),
            "target": torch.from_numpy(self.labels[index].copy()),
            "ecg_id": torch.tensor(int(row["ecg_id"]), dtype=torch.int64),
            "age": torch.tensor(age, dtype=torch.float32),
            "sex": torch.tensor(sex, dtype=torch.int64),
        }
