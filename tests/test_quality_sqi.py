from __future__ import annotations

import numpy as np

from ecg_guard.quality.sqi import (
    QUALITY_FEATURE_DIRECTIONS,
    extract_quality_features,
    fit_quality_reference,
    score_quality_features,
)


def test_quality_features_are_finite_for_flat_signal() -> None:
    features = extract_quality_features(np.zeros((12, 1_000)))

    assert set(features) == set(QUALITY_FEATURE_DIRECTIONS)
    assert all(np.isfinite(value) for value in features.values())
    assert features["flatline_fraction_max"] == 1.0


def test_spectral_features_respond_to_frequency() -> None:
    time = np.arange(1_000) / 100
    low = np.tile(np.sin(2 * np.pi * 0.2 * time), (12, 1))
    high = np.tile(np.sin(2 * np.pi * 40 * time), (12, 1))

    low_features = extract_quality_features(low)
    high_features = extract_quality_features(high)

    assert (
        low_features["baseline_wander_ratio_max"]
        > high_features["baseline_wander_ratio_max"]
    )
    assert (
        high_features["high_frequency_ratio_max"]
        > low_features["high_frequency_ratio_max"]
    )


def test_reference_flags_extreme_values() -> None:
    values = np.linspace(0.0, 1.0, 1_000)
    matrix = {
        name: values.copy()
        for name in QUALITY_FEATURE_DIRECTIONS
    }
    reference = fit_quality_reference(matrix)
    middle = score_quality_features(
        {name: 0.5 for name in QUALITY_FEATURE_DIRECTIONS},
        reference,
    )
    extreme_features = {
        name: (2.0 if direction == "high" else -1.0)
        for name, direction in QUALITY_FEATURE_DIRECTIONS.items()
    }
    extreme = score_quality_features(extreme_features, reference)

    assert middle["technical_quality_status"] == "within_reference"
    assert extreme["technical_quality_status"] == "extreme_outlier"
    assert (
        extreme["technical_quality_score"]
        < middle["technical_quality_score"]
    )
