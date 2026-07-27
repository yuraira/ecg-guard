"""Distribution-referenced technical indicators for 12-lead ECG signals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ecg_guard.data.prepare_ptbxl import (
    EXPECTED_SAMPLES,
    EXPECTED_SAMPLING_FREQUENCY,
    LEAD_NAMES,
)


QUALITY_FEATURE_DIRECTIONS = {
    "baseline_wander_ratio_max": "high",
    "high_frequency_ratio_max": "high",
    "flatline_fraction_max": "high",
    "amplitude_range_max": "high",
    "amplitude_range_min": "low",
    "derivative_ratio_max": "high",
    "lead_std_min": "low",
}


def extract_quality_features(
    waveform: np.ndarray,
    *,
    sampling_frequency: int = EXPECTED_SAMPLING_FREQUENCY,
) -> dict[str, float]:
    """Extract interpretable spectral and time-domain technical indicators."""
    waveform = np.asarray(waveform, dtype=np.float64)
    expected_shape = (len(LEAD_NAMES), EXPECTED_SAMPLES)
    if waveform.shape != expected_shape:
        raise ValueError(f"expected waveform shape {expected_shape}, got {waveform.shape}")
    if sampling_frequency != EXPECTED_SAMPLING_FREQUENCY:
        raise ValueError("quality indicators currently require 100 Hz signals")
    if not np.isfinite(waveform).all():
        raise ValueError("waveform contains non-finite values")

    centered = waveform - waveform.mean(axis=1, keepdims=True)
    window = np.hanning(waveform.shape[1])
    frequencies = np.fft.rfftfreq(
        waveform.shape[1],
        d=1.0 / sampling_frequency,
    )
    power = np.abs(np.fft.rfft(centered * window, axis=1)) ** 2
    total_band = (frequencies >= 0.1) & (frequencies <= 49.0)
    total_power = power[:, total_band].sum(axis=1) + np.finfo(float).eps
    baseline_band = (frequencies >= 0.1) & (frequencies < 0.7)
    high_frequency_band = frequencies >= 35.0
    baseline_ratio = power[:, baseline_band].sum(axis=1) / total_power
    high_frequency_ratio = (
        power[:, high_frequency_band].sum(axis=1) / total_power
    )

    differences = np.diff(waveform, axis=1)
    lead_standard_deviation = waveform.std(axis=1)
    derivative_ratio = np.sqrt((differences**2).mean(axis=1)) / (
        lead_standard_deviation + np.finfo(float).eps
    )
    flatline_fraction = (np.abs(differences) < 1e-5).mean(axis=1)
    amplitude_range = (
        np.quantile(waveform, 0.995, axis=1)
        - np.quantile(waveform, 0.005, axis=1)
    )

    features = {
        "baseline_wander_ratio_max": float(baseline_ratio.max()),
        "high_frequency_ratio_max": float(high_frequency_ratio.max()),
        "flatline_fraction_max": float(flatline_fraction.max()),
        "amplitude_range_max": float(amplitude_range.max()),
        "amplitude_range_min": float(amplitude_range.min()),
        "derivative_ratio_max": float(derivative_ratio.max()),
        "lead_std_min": float(lead_standard_deviation.min()),
    }
    if not all(np.isfinite(value) for value in features.values()):
        raise ValueError("quality feature extraction produced non-finite values")
    return features


def fit_quality_reference(
    feature_matrix: Mapping[str, np.ndarray],
    *,
    quantile_count: int = 1_001,
) -> dict[str, Any]:
    """Fit empirical quantile references using training waveforms only."""
    if quantile_count < 101:
        raise ValueError("quantile_count must be at least 101")
    if set(feature_matrix) != set(QUALITY_FEATURE_DIRECTIONS):
        raise ValueError("feature matrix does not match quality feature definitions")
    lengths = {len(np.asarray(values)) for values in feature_matrix.values()}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError("all quality feature arrays must have one positive length")

    probabilities = np.linspace(0.0, 1.0, quantile_count)
    return {
        "method": "training empirical quantile reference",
        "quantile_probabilities": probabilities.tolist(),
        "features": {
            name: {
                "direction": QUALITY_FEATURE_DIRECTIONS[name],
                "quantiles": np.quantile(
                    np.asarray(values, dtype=np.float64),
                    probabilities,
                ).tolist(),
            }
            for name, values in feature_matrix.items()
        },
    }


def score_quality_features(
    features: Mapping[str, float],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Score technical outlier severity relative to the training distribution."""
    if set(features) != set(QUALITY_FEATURE_DIRECTIONS):
        raise ValueError("features do not match quality feature definitions")
    probabilities = np.asarray(
        reference["quantile_probabilities"],
        dtype=np.float64,
    )
    component_tail_probabilities: dict[str, float] = {}
    review_flags: list[str] = []

    for name, direction in QUALITY_FEATURE_DIRECTIONS.items():
        quantiles = np.asarray(
            reference["features"][name]["quantiles"],
            dtype=np.float64,
        )
        percentile = float(
            np.interp(
                float(features[name]),
                quantiles,
                probabilities,
                left=0.0,
                right=1.0,
            )
        )
        tail_probability = percentile if direction == "high" else 1.0 - percentile
        component_tail_probabilities[name] = tail_probability
        if tail_probability >= 0.99:
            review_flags.append(name)

    review_score = max(component_tail_probabilities.values())
    quality_score = 100.0 * (
        1.0 - np.clip((review_score - 0.95) / 0.05, 0.0, 1.0)
    )
    if review_score >= 0.999:
        status = "extreme_outlier"
    elif review_score >= 0.99:
        status = "review"
    else:
        status = "within_reference"
    return {
        "technical_quality_score": float(quality_score),
        "technical_review_score": float(review_score),
        "technical_quality_status": status,
        "review_flags": review_flags,
        "component_tail_probabilities": component_tail_probabilities,
    }
