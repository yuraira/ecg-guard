from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg_guard.data.prepare_ptbxl import (
    DIAGNOSTIC_CLASSES,
    aggregate_diagnostic_classes,
    load_diagnostic_mapping,
    load_waveforms,
    make_label_matrix,
    parse_scp_codes,
    prepare_metadata,
    split_from_fold,
    validate_patient_splits,
)


DATA_DIR = Path("data/raw/ptb-xl")


def test_parse_and_aggregate_diagnostic_codes() -> None:
    parsed = parse_scp_codes("{'NORM': 100.0, 'LVOLT': 0.0, 'SR': 0.0}")
    labels = aggregate_diagnostic_classes(
        parsed,
        {"NORM": "NORM", "LVOLT": "CD"},
    )

    assert parsed["LVOLT"] == 0.0
    assert labels == ("NORM", "CD")


@pytest.mark.parametrize(
    ("fold", "expected"),
    [(1, "train"), (8, "train"), (9, "validation"), (10, "test")],
)
def test_official_fold_mapping(fold: int, expected: str) -> None:
    assert split_from_fold(fold) == expected


def test_invalid_fold_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 1 and 10"):
        split_from_fold(0)


def test_patient_leakage_is_rejected() -> None:
    metadata = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 3],
            "split": ["train", "test", "validation", "test"],
        }
    )

    with pytest.raises(ValueError, match="patient leakage"):
        validate_patient_splits(metadata)


@pytest.fixture(scope="module")
def prepared_metadata() -> pd.DataFrame:
    if not DATA_DIR.is_dir():
        pytest.skip("local PTB-XL data is not available")
    return prepare_metadata(DATA_DIR)


def test_real_metadata_and_official_class_counts(
    prepared_metadata: pd.DataFrame,
) -> None:
    assert len(prepared_metadata) == 21_799
    assert prepared_metadata["split"].value_counts().to_dict() == {
        "train": 17_418,
        "test": 2_198,
        "validation": 2_183,
    }
    assert prepared_metadata.loc[:, DIAGNOSTIC_CLASSES].sum().astype(int).to_dict() == {
        "NORM": 9_514,
        "MI": 5_469,
        "STTC": 5_235,
        "CD": 4_898,
        "HYP": 2_649,
    }
    assert (
        prepared_metadata.loc[:, DIAGNOSTIC_CLASSES].sum(axis=1).eq(0).sum()
        == 411
    )
    assert prepared_metadata["has_diagnostic_superclass"].sum() == 21_388


def test_real_patient_splits_do_not_overlap(
    prepared_metadata: pd.DataFrame,
) -> None:
    validate_patient_splits(prepared_metadata)


def test_real_label_matrix_is_binary(
    prepared_metadata: pd.DataFrame,
) -> None:
    labels = make_label_matrix(prepared_metadata)

    assert labels.shape == (21_799, 5)
    assert labels.dtype == np.float32
    assert np.isin(labels, (0.0, 1.0)).all()


def test_real_waveform_shape_and_values(
    prepared_metadata: pd.DataFrame,
) -> None:
    waveforms = load_waveforms(prepared_metadata.iloc[:1], DATA_DIR)

    assert waveforms.shape == (1, 12, 1_000)
    assert waveforms.dtype == np.float32
    assert np.isfinite(waveforms).all()


def test_mapping_contains_only_expected_superclasses() -> None:
    if not DATA_DIR.is_dir():
        pytest.skip("local PTB-XL data is not available")

    mapping = load_diagnostic_mapping(DATA_DIR)
    assert set(mapping.values()) == set(DIAGNOSTIC_CLASSES)
