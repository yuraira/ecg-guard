from __future__ import annotations

import numpy as np

from ecg_guard.data.prepare_ptbxl import DIAGNOSTIC_CLASSES
from ecg_guard.uncertainty.selective import (
    decision_boundary_uncertainty,
    fit_coverage_cutoffs,
    predictive_entropy,
    selective_metrics,
)


THRESHOLDS = {
    class_name: value
    for class_name, value in zip(
        DIAGNOSTIC_CLASSES,
        (0.4, 0.3, 0.3, 0.25, 0.15),
        strict=True,
    )
}


def test_entropy_is_highest_at_half_probability() -> None:
    entropy = predictive_entropy(np.array([[0.01, 0.5, 0.99]]))

    assert entropy.shape == (1, 3)
    assert entropy[0, 1] > entropy[0, 0]
    assert np.isclose(entropy[0, 0], entropy[0, 2])


def test_uncertainty_is_highest_on_decision_boundary() -> None:
    threshold_vector = np.array(
        [THRESHOLDS[name] for name in DIAGNOSTIC_CLASSES]
    )
    probabilities = np.vstack(
        [threshold_vector, np.full(5, 0.99)]
    )
    record_uncertainty, class_uncertainty = decision_boundary_uncertainty(
        probabilities,
        THRESHOLDS,
    )

    assert np.allclose(class_uncertainty[0], 1.0)
    assert record_uncertainty[0] == 1.0
    assert record_uncertainty[1] < record_uncertainty[0]


def test_validation_coverage_cutoffs_are_ordered() -> None:
    scores = np.linspace(0.0, 1.0, 101)
    cutoffs = fit_coverage_cutoffs(scores, (0.9, 0.8, 0.7))

    assert cutoffs["0.70"] <= cutoffs["0.80"] <= cutoffs["0.90"]


def test_selective_metrics_report_retained_records() -> None:
    targets = np.tile(np.array([[1, 0, 1, 0, 0]]), (10, 1))
    probabilities = np.tile(np.array([[0.9, 0.1, 0.8, 0.1, 0.1]]), (10, 1))
    uncertainty = np.linspace(0.0, 0.9, 10)

    metrics = selective_metrics(
        targets,
        probabilities,
        THRESHOLDS,
        uncertainty,
        cutoff=0.45,
    )

    assert metrics["records"] == 5
    assert metrics["coverage"] == 0.5
    assert metrics["hamming_error_rate"] == 0.0
