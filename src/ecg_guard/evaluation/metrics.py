"""Metrics, calibration, thresholds, and clustered confidence intervals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from ecg_guard.data.prepare_ptbxl import DIAGNOSTIC_CLASSES


PROBABILITY_EPSILON = 1e-7


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable NumPy sigmoid."""
    logits = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(logits)
    positive = logits >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    negative_exponential = np.exp(logits[~positive])
    result[~positive] = negative_exponential / (1.0 + negative_exponential)
    return result


def binary_nll_from_logits(targets: np.ndarray, logits: np.ndarray) -> float:
    """Mean binary cross-entropy calculated stably from logits."""
    targets = np.asarray(targets, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    if targets.shape != logits.shape:
        raise ValueError("targets and logits must have the same shape")
    return float(np.mean(np.logaddexp(0.0, logits) - targets * logits))


def fit_temperature(
    validation_logits: np.ndarray,
    validation_targets: np.ndarray,
    *,
    minimum: float = 0.05,
    maximum: float = 10.0,
    grid_size: int = 1_001,
) -> float:
    """Fit one scalar temperature by validation binary cross-entropy."""
    if minimum <= 0 or maximum <= minimum:
        raise ValueError("temperature range must be positive and increasing")
    if grid_size < 3:
        raise ValueError("grid_size must be at least three")

    logits = np.asarray(validation_logits, dtype=np.float64)
    targets = np.asarray(validation_targets, dtype=np.float64)
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("validation logits and targets must be matching matrices")

    candidates = np.unique(
        np.concatenate(
            (
                np.geomspace(minimum, maximum, grid_size),
                np.array([1.0]),
            )
        )
    )
    losses = np.array(
        [binary_nll_from_logits(targets, logits / value) for value in candidates]
    )
    return float(candidates[int(np.argmin(losses))])


def fit_youden_thresholds(
    validation_targets: np.ndarray,
    validation_probabilities: np.ndarray,
) -> dict[str, float]:
    """Choose class thresholds maximizing Youden's J on validation data."""
    targets = np.asarray(validation_targets)
    probabilities = np.asarray(validation_probabilities)
    expected_shape = (targets.shape[0], len(DIAGNOSTIC_CLASSES))
    if targets.shape != expected_shape or probabilities.shape != expected_shape:
        raise ValueError("unexpected validation target/probability shape")

    thresholds: dict[str, float] = {}
    for class_index, class_name in enumerate(DIAGNOSTIC_CLASSES):
        class_targets = targets[:, class_index]
        if np.unique(class_targets).size < 2:
            raise ValueError(f"both outcomes are required for {class_name}")
        false_positive_rate, true_positive_rate, candidates = roc_curve(
            class_targets,
            probabilities[:, class_index],
            drop_intermediate=False,
        )
        youden = true_positive_rate - false_positive_rate
        best = np.flatnonzero(np.isclose(youden, youden.max(), atol=1e-12))
        finite_candidates = candidates[best][np.isfinite(candidates[best])]
        if finite_candidates.size == 0:
            raise ValueError(f"no finite threshold found for {class_name}")
        # Lowest tied threshold favors sensitivity without changing Youden's J.
        thresholds[class_name] = float(np.min(finite_candidates))
    return thresholds


def expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    """Equal-width expected calibration error for one binary outcome."""
    targets = np.asarray(targets, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.shape != probabilities.shape or targets.ndim != 1:
        raise ValueError("ECE inputs must be matching vectors")
    if bins <= 1:
        raise ValueError("bins must be greater than one")

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    error = 0.0
    for bin_id in range(bins):
        selected = bin_ids == bin_id
        if not selected.any():
            continue
        error += (
            selected.mean()
            * abs(targets[selected].mean() - probabilities[selected].mean())
        )
    return float(error)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _mean_available(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return float(np.mean(available)) if available else None


def evaluate_predictions(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Calculate discrimination, calibration, and operating-point metrics."""
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    expected_shape = (targets.shape[0], len(DIAGNOSTIC_CLASSES))
    if targets.shape != expected_shape or probabilities.shape != expected_shape:
        raise ValueError("unexpected target/probability shape")
    if set(thresholds) != set(DIAGNOSTIC_CLASSES):
        raise ValueError("thresholds must cover all diagnostic classes")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("targets must be binary")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")

    per_class: dict[str, dict[str, float | int | None]] = {}
    for class_index, class_name in enumerate(DIAGNOSTIC_CLASSES):
        class_targets = targets[:, class_index]
        class_probabilities = probabilities[:, class_index]
        threshold = float(thresholds[class_name])
        predictions = class_probabilities >= threshold
        positives = class_targets == 1
        negatives = ~positives
        true_positive = int((predictions & positives).sum())
        false_positive = int((predictions & negatives).sum())
        false_negative = int((~predictions & positives).sum())
        true_negative = int((~predictions & negatives).sum())

        if np.unique(class_targets).size >= 2:
            auroc: float | None = float(
                roc_auc_score(class_targets, class_probabilities)
            )
            average_precision: float | None = float(
                average_precision_score(class_targets, class_probabilities)
            )
        else:
            auroc = None
            average_precision = None

        precision = _safe_ratio(true_positive, true_positive + false_positive)
        sensitivity = _safe_ratio(true_positive, true_positive + false_negative)
        specificity = _safe_ratio(true_negative, true_negative + false_positive)
        f1_denominator = 2 * true_positive + false_positive + false_negative
        per_class[class_name] = {
            "records": int(len(class_targets)),
            "positives": int(positives.sum()),
            "prevalence": float(positives.mean()),
            "threshold": threshold,
            "auroc": auroc,
            "average_precision": average_precision,
            "nll": float(
                -np.mean(
                    class_targets
                    * np.log(
                        np.clip(
                            class_probabilities,
                            PROBABILITY_EPSILON,
                            1.0,
                        )
                    )
                    + (1 - class_targets)
                    * np.log(
                        np.clip(
                            1.0 - class_probabilities,
                            PROBABILITY_EPSILON,
                            1.0,
                        )
                    )
                )
            ),
            "brier": float(np.mean((class_probabilities - class_targets) ** 2)),
            "ece_15_bin": expected_calibration_error(
                class_targets,
                class_probabilities,
            ),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1": _safe_ratio(2 * true_positive, f1_denominator),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        }

    macro_fields = (
        "auroc",
        "average_precision",
        "nll",
        "brier",
        "ece_15_bin",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
    )
    macro = {
        field: _mean_available(
            [
                (
                    float(per_class[class_name][field])
                    if per_class[class_name][field] is not None
                    else None
                )
                for class_name in DIAGNOSTIC_CLASSES
            ]
        )
        for field in macro_fields
    }
    return {
        "records": int(targets.shape[0]),
        "macro": macro,
        "per_class": per_class,
    }


def _percentile_interval(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if finite.size == 0:
        return {"lower": None, "upper": None, "valid_replicates": 0}
    lower, upper = np.percentile(finite, (2.5, 97.5))
    return {
        "lower": float(lower),
        "upper": float(upper),
        "valid_replicates": int(finite.size),
    }


def patient_cluster_bootstrap(
    patient_ids: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Mapping[str, float],
    *,
    replicates: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Percentile confidence intervals resampling patients as clusters."""
    patient_ids = np.asarray(patient_ids)
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    if len(patient_ids) != len(targets) or len(targets) != len(probabilities):
        raise ValueError("patient IDs, targets, and probabilities must align")
    if replicates <= 0:
        raise ValueError("replicates must be positive")

    unique_patients, inverse = np.unique(patient_ids, return_inverse=True)
    indices_by_patient = [
        np.flatnonzero(inverse == patient_index)
        for patient_index in range(len(unique_patients))
    ]
    rng = np.random.default_rng(seed)
    macro_fields = (
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
        "f1",
        "brier",
    )
    class_fields = macro_fields
    macro_samples = {field: [] for field in macro_fields}
    class_samples = {
        class_name: {field: [] for field in class_fields}
        for class_name in DIAGNOSTIC_CLASSES
    }

    for _ in range(replicates):
        sampled_patients = rng.integers(
            0,
            len(unique_patients),
            size=len(unique_patients),
        )
        sampled_indices = np.concatenate(
            [indices_by_patient[index] for index in sampled_patients]
        )
        metrics = evaluate_predictions(
            targets[sampled_indices],
            probabilities[sampled_indices],
            thresholds,
        )
        for field in macro_fields:
            value = metrics["macro"][field]
            if value is not None:
                macro_samples[field].append(float(value))
        for class_name in DIAGNOSTIC_CLASSES:
            for field in class_fields:
                value = metrics["per_class"][class_name][field]
                if value is not None:
                    class_samples[class_name][field].append(float(value))

    return {
        "method": "patient-cluster percentile bootstrap",
        "confidence_level": 0.95,
        "requested_replicates": replicates,
        "seed": seed,
        "patients": int(len(unique_patients)),
        "macro": {
            field: _percentile_interval(values)
            for field, values in macro_samples.items()
        },
        "per_class": {
            class_name: {
                field: _percentile_interval(values)
                for field, values in fields.items()
            }
            for class_name, fields in class_samples.items()
        },
    }
