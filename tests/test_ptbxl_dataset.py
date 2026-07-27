from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ecg_guard.data.prepare_ptbxl import prepare_metadata
from ecg_guard.data.ptbxl_dataset import (
    PTBXLDataset,
    compute_training_normalization_stats,
    select_modeling_metadata,
)


DATA_DIR = Path("data/raw/ptb-xl")


@pytest.fixture(scope="module")
def prepared_metadata() -> pd.DataFrame:
    if not DATA_DIR.is_dir():
        pytest.skip("local PTB-XL data is not available")
    return prepare_metadata(DATA_DIR)


def test_modeling_cohort_counts(prepared_metadata: pd.DataFrame) -> None:
    assert {
        split: len(select_modeling_metadata(prepared_metadata, split))
        for split in ("train", "validation", "test")
    } == {
        "train": 17_084,
        "validation": 2_146,
        "test": 2_158,
    }


def test_limited_selection_is_deterministic(
    prepared_metadata: pd.DataFrame,
) -> None:
    first = select_modeling_metadata(
        prepared_metadata,
        "train",
        limit=8,
        seed=7,
    )
    second = select_modeling_metadata(
        prepared_metadata,
        "train",
        limit=8,
        seed=7,
    )

    assert first["ecg_id"].tolist() == second["ecg_id"].tolist()
    assert first["ecg_id"].is_monotonic_increasing


def test_normalization_and_dataset_item(
    prepared_metadata: pd.DataFrame,
) -> None:
    selected = select_modeling_metadata(
        prepared_metadata,
        "train",
        limit=3,
        seed=42,
    )
    stats = compute_training_normalization_stats(selected, DATA_DIR)
    dataset = PTBXLDataset(selected, DATA_DIR, stats)
    signals = torch.stack([dataset[index]["signal"] for index in range(len(dataset))])
    item = dataset[0]

    assert stats.records == 3
    assert stats.count == 3 * 12 * 1_000
    assert item["signal"].shape == (12, 1_000)
    assert item["signal"].dtype == torch.float32
    assert item["target"].shape == (5,)
    assert item["target"].sum() >= 1
    assert np.isclose(float(signals.mean()), 0.0, atol=1e-6)
    assert np.isclose(float(signals.std(correction=0)), 1.0, atol=1e-6)


def test_nontraining_normalization_is_rejected(
    prepared_metadata: pd.DataFrame,
) -> None:
    selected = select_modeling_metadata(
        prepared_metadata,
        "validation",
        limit=1,
    )
    with pytest.raises(ValueError, match="training split"):
        compute_training_normalization_stats(selected, DATA_DIR)
