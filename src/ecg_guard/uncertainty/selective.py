"""Decision-boundary uncertainty proxies for a frozen multilabel model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ecg_guard.data.prepare_ptbxl import DIAGNOSTIC_CLASSES
from ecg_guard.evaluation.metrics import evaluate_predictions


UNCERTAINTY_EPSILON = 1e-7


def predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Binary entropy per record and class, normalized to the [0, 1] range."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(
        probabilities,
        UNCERTAINTY_EPSILON,
        1.0 - UNCERTAINTY_EPSILON,
    )
    entropy = -(
        clipped * np.log(clipped)
        + (1.0 - clipped) * np.log(1.0 - clipped)
    )
    return entropy / np.log(2.0)


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        np.asarray(values, dtype=np.float64),
        UNCERTAINTY_EPSILON,
        1.0 - UNCERTAINTY_EPSILON,
    )
    return np.log(clipped / (1.0 - clipped))


def decision_boundary_uncertainty(
    probabilities: np.ndarray,
    thresholds: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return record and class uncertainty based on distance to locked thresholds.

    A value of one is exactly on a decision boundary. Values approach zero as
    the log-odds distance from every boundary grows.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(
        DIAGNOSTIC_CLASSES
    ):
        raise ValueError("unexpected probability matrix shape")
    if set(thresholds) != set(DIAGNOSTIC_CLASSES):
        raise ValueError("thresholds must cover every diagnostic class")
    threshold_vector = np.array(
        [thresholds[class_name] for class_name in DIAGNOSTIC_CLASSES],
        dtype=np.float64,
    )
    logit_distance = np.abs(
        _logit(probabilities) - _logit(threshold_vector)[None, :]
    )
    class_uncertainty = np.exp(-logit_distance)
    return class_uncertainty.max(axis=1), class_uncertainty


def fit_coverage_cutoffs(
    validation_uncertainty: np.ndarray,
    target_coverages: Sequence[float],
) -> dict[str, float]:
    """Fit keep-if-below cutoffs using validation uncertainty quantiles."""
    scores = np.asarray(validation_uncertainty, dtype=np.float64)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("validation uncertainty must be a nonempty vector")
    if not np.isfinite(scores).all():
        raise ValueError("validation uncertainty must be finite")

    cutoffs: dict[str, float] = {}
    for coverage in target_coverages:
        if not 0 < coverage <= 1:
            raise ValueError("target coverages must be in (0, 1]")
        cutoffs[f"{coverage:.2f}"] = float(
            np.quantile(scores, coverage, method="higher")
        )
    return cutoffs


def prediction_error_arrays(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    threshold_vector = np.array(
        [thresholds[class_name] for class_name in DIAGNOSTIC_CLASSES]
    )
    predictions = probabilities >= threshold_vector[None, :]
    errors = predictions != np.asarray(targets, dtype=bool)
    return errors.any(axis=1), errors.mean(axis=1)


def selective_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Mapping[str, float],
    uncertainty: np.ndarray,
    cutoff: float,
) -> dict[str, Any]:
    """Evaluate retained records at one pre-specified uncertainty cutoff."""
    selected = np.asarray(uncertainty) <= cutoff
    if not selected.any():
        raise ValueError("uncertainty cutoff retained no records")
    any_error, hamming_error = prediction_error_arrays(
        targets,
        probabilities,
        thresholds,
    )
    metrics = evaluate_predictions(
        targets[selected],
        probabilities[selected],
        thresholds,
    )
    return {
        "records": int(selected.sum()),
        "coverage": float(selected.mean()),
        "exact_match_error_rate": float(any_error[selected].mean()),
        "hamming_error_rate": float(hamming_error[selected].mean()),
        "macro": metrics["macro"],
        "per_class": metrics["per_class"],
    }


def risk_coverage_curve(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Mapping[str, float],
    uncertainty: np.ndarray,
    *,
    minimum_coverage: float = 0.1,
    points: int = 91,
) -> dict[str, list[float] | float]:
    """Measure error among increasingly large least-uncertain subsets."""
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    if points < 2:
        raise ValueError("points must be at least two")
    uncertainty = np.asarray(uncertainty)
    any_error, hamming_error = prediction_error_arrays(
        targets,
        probabilities,
        thresholds,
    )
    order = np.argsort(uncertainty, kind="stable")
    coverages = np.linspace(minimum_coverage, 1.0, points)
    exact_risks: list[float] = []
    hamming_risks: list[float] = []
    for coverage in coverages:
        retained = order[: max(1, int(np.ceil(coverage * len(order))))]
        exact_risks.append(float(any_error[retained].mean()))
        hamming_risks.append(float(hamming_error[retained].mean()))
    return {
        "coverage": coverages.tolist(),
        "exact_match_error_rate": exact_risks,
        "hamming_error_rate": hamming_risks,
        "exact_match_aurc_from_minimum_coverage": float(
            np.trapezoid(exact_risks, coverages)
            / (1.0 - minimum_coverage)
        ),
        "hamming_aurc_from_minimum_coverage": float(
            np.trapezoid(hamming_risks, coverages)
            / (1.0 - minimum_coverage)
        ),
    }
