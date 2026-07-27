from __future__ import annotations

import pandas as pd

from ecg_guard.analysis.error_analysis import (
    error_masks,
    select_confident_errors,
)


def example_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ecg_id": [1, 2, 3, 4],
            "target_HYP": [0, 0, 1, 1],
            "prediction_HYP": [1, 1, 0, 1],
            "probability_HYP": [0.8, 0.9, 0.1, 0.7],
        }
    )


def test_error_masks_are_mutually_exclusive() -> None:
    masks = error_masks(example_frame(), "HYP")

    assert {name: int(mask.sum()) for name, mask in masks.items()} == {
        "true_positive": 1,
        "false_positive": 2,
        "false_negative": 1,
        "true_negative": 0,
    }
    assert sum(masks.values()).tolist() == [1, 1, 1, 1]


def test_confident_error_selection_order() -> None:
    frame = example_frame()

    false_positive = select_confident_errors(
        frame,
        "HYP",
        "false_positive",
        limit=2,
    )
    false_negative = select_confident_errors(
        frame,
        "HYP",
        "false_negative",
        limit=1,
    )

    assert false_positive["ecg_id"].tolist() == [2, 1]
    assert false_negative["ecg_id"].tolist() == [3]
