from __future__ import annotations

import numpy as np
import torch

from ecg_guard.models import BaselineECGCNN
from ecg_guard.training.train_baseline import multilabel_metrics


def test_baseline_forward_shape_and_gradients() -> None:
    model = BaselineECGCNN()
    inputs = torch.randn(2, 12, 1_000)
    targets = torch.tensor(
        [[1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0]]
    )

    logits = model(inputs)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()

    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_multilabel_metrics_handles_absent_class() -> None:
    targets = np.array(
        [[0, 0, 0, 1, 0], [1, 0, 1, 0, 0], [1, 0, 0, 1, 1]],
        dtype=np.float32,
    )
    probabilities = np.array(
        [[0.1, 0.2, 0.3, 0.9, 0.2], [0.8, 0.1, 0.8, 0.1, 0.2], [0.9, 0.3, 0.2, 0.8, 0.7]],
        dtype=np.float32,
    )

    metrics = multilabel_metrics(targets, probabilities)

    assert metrics["per_class"]["MI"] == {
        "auroc": None,
        "average_precision": None,
    }
    assert metrics["macro_auroc"] is not None
