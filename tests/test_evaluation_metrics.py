from __future__ import annotations

import numpy as np

from ecg_guard.data.prepare_ptbxl import DIAGNOSTIC_CLASSES
from ecg_guard.evaluation.evaluate_baseline import age_group
from ecg_guard.evaluation.metrics import (
    binary_nll_from_logits,
    evaluate_predictions,
    fit_temperature,
    fit_youden_thresholds,
    patient_cluster_bootstrap,
    sigmoid,
)


def synthetic_predictions() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    targets = rng.integers(0, 2, size=(40, 5))
    logits = (targets * 2 - 1) * 1.5 + rng.normal(0, 1, size=(40, 5))
    return targets, logits


def test_temperature_does_not_worsen_validation_nll() -> None:
    targets, logits = synthetic_predictions()
    temperature = fit_temperature(logits, targets, grid_size=101)

    before = binary_nll_from_logits(targets, logits)
    after = binary_nll_from_logits(targets, logits / temperature)

    assert temperature > 0
    assert after <= before + 1e-12


def test_thresholds_and_metrics_cover_all_classes() -> None:
    targets, logits = synthetic_predictions()
    probabilities = sigmoid(logits)
    thresholds = fit_youden_thresholds(targets, probabilities)
    metrics = evaluate_predictions(targets, probabilities, thresholds)

    assert set(thresholds) == set(DIAGNOSTIC_CLASSES)
    assert set(metrics["per_class"]) == set(DIAGNOSTIC_CLASSES)
    assert 0 <= metrics["macro"]["auroc"] <= 1
    assert 0 <= metrics["macro"]["sensitivity"] <= 1
    assert 0 <= metrics["macro"]["specificity"] <= 1


def test_patient_cluster_bootstrap_is_deterministic() -> None:
    targets, logits = synthetic_predictions()
    probabilities = sigmoid(logits)
    thresholds = fit_youden_thresholds(targets, probabilities)
    patient_ids = np.repeat(np.arange(20), 2)

    first = patient_cluster_bootstrap(
        patient_ids,
        targets,
        probabilities,
        thresholds,
        replicates=20,
        seed=11,
    )
    second = patient_cluster_bootstrap(
        patient_ids,
        targets,
        probabilities,
        thresholds,
        replicates=20,
        seed=11,
    )

    assert first == second
    assert first["patients"] == 20
    assert first["macro"]["auroc"]["valid_replicates"] == 20


def test_masked_age_is_not_treated_as_real_age() -> None:
    assert age_group(39) == "under_40"
    assert age_group(40) == "40_to_59"
    assert age_group(59) == "40_to_59"
    assert age_group(60) == "60_plus"
    assert age_group(300) == "masked_90_plus"
