from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from ecg_guard.data.prepare_ptbxl import DIAGNOSTIC_CLASSES
from ecg_guard.data.ptbxl_dataset import NormalizationStats
from ecg_guard.inference.predict_record import (
    DEFAULT_PROTOCOL_PATH,
    determine_review_action,
    load_inference_protocol,
    predict_waveform,
)
from ecg_guard.quality.sqi import (
    extract_quality_features,
    fit_quality_reference,
)


class FixedLogitModel(nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer(
            "fixed_logits",
            torch.tensor(logits, dtype=torch.float32),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fixed_logits.repeat(inputs.shape[0], 1)


def make_protocol(waveform: np.ndarray) -> dict[str, object]:
    features = extract_quality_features(waveform)
    feature_matrix = {
        name: np.linspace(value * 0.5, value * 1.5 + 1e-6, 1_000)
        for name, value in features.items()
    }
    return {
        "schema_version": "1.0",
        "model_version": "test-model",
        "calibration": {"temperature": 1.0},
        "thresholds": {
            "values": {
                class_name: 0.5 for class_name in DIAGNOSTIC_CLASSES
            }
        },
        "selective_prediction": {
            "target_coverage": 0.8,
            "uncertainty_cutoff": 0.8,
        },
        "quality_reference": fit_quality_reference(feature_matrix),
    }


def test_repository_protocol_is_valid() -> None:
    protocol = load_inference_protocol(DEFAULT_PROTOCOL_PATH)

    assert protocol["model_version"] == "baseline-v1"
    assert tuple(protocol["classes"]) == DIAGNOSTIC_CLASSES


@pytest.mark.parametrize(
    ("uncertainty", "quality_status", "expected"),
    [
        (0.5, "within_reference", "auto_result"),
        (0.9, "within_reference", "review_uncertain"),
        (0.5, "review", "review_technical"),
        (0.9, "extreme_outlier", "review_both"),
    ],
)
def test_review_actions_preserve_independent_reasons(
    uncertainty: float,
    quality_status: str,
    expected: str,
) -> None:
    assert (
        determine_review_action(uncertainty, 0.8, quality_status)
        == expected
    )


def test_prediction_report_contains_calibrated_routing() -> None:
    time = np.arange(1_000) / 100
    waveform = np.stack(
        [
            0.5 * np.sin(2 * np.pi * (1.0 + index * 0.1) * time)
            for index in range(12)
        ]
    ).astype(np.float32)
    protocol = make_protocol(waveform)
    model = FixedLogitModel([0.0, 2.0, -2.0, 1.0, -1.0])
    normalization = NormalizationStats(
        mean=0.0,
        std=1.0,
        count=12_000,
        records=1,
    )

    report = predict_waveform(
        waveform,
        model,
        normalization,
        protocol,
        torch.device("cpu"),
    )

    assert report["warning"]
    assert len(report["predictions"]) == 5
    assert report["predictions"][1]["positive"] is True
    assert report["predictions"][2]["positive"] is False
    assert report["routing"]["action"] == "review_uncertain"
