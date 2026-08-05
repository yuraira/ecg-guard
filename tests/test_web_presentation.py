from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from ecg_guard.data.prepare_ptbxl import LEAD_NAMES
from ecg_guard.web.presentation import (
    create_ecg_figure,
    create_probability_figure,
    create_synthetic_demo_waveform,
    prediction_table,
    save_uploaded_record,
    validate_upload_set,
    validate_wfdb_header,
)


@dataclass
class FakeUpload:
    name: str
    payload: bytes

    @property
    def size(self) -> int:
        return len(self.payload)

    def getvalue(self) -> bytes:
        return self.payload


def make_header(
    *,
    sampling_frequency: int = 100,
    samples: int = 1_000,
) -> bytes:
    signal_lines = [
        f"sample.dat 16 1000/mV 16 0 0 0 0 {lead}"
        for lead in LEAD_NAMES
    ]
    return (
        "\n".join(
            [
                f"sample 12 {sampling_frequency} {samples}",
                *signal_lines,
            ]
        )
        + "\n"
    ).encode("ascii")


def test_upload_set_rejects_paths_and_requires_pair() -> None:
    with pytest.raises(ValueError, match="경로"):
        validate_upload_set(["../sample.hea", "sample.dat"], [10, 10])
    with pytest.raises(ValueError, match="하나 이상의"):
        validate_upload_set(["sample.hea"], [10])


def test_header_validates_shape_and_references() -> None:
    validate_wfdb_header(
        make_header(),
        ["sample.hea", "sample.dat"],
    )

    with pytest.raises(ValueError, match="100Hz"):
        validate_wfdb_header(
            make_header(sampling_frequency=500),
            ["sample.hea", "sample.dat"],
        )
    with pytest.raises(ValueError, match="함께 업로드"):
        validate_wfdb_header(make_header(), ["sample.hea", "other.dat"])


def test_uploaded_record_is_saved_in_target_directory(
    tmp_path: Path,
) -> None:
    uploads = [
        FakeUpload("sample.hea", make_header()),
        FakeUpload("sample.dat", b"\x00" * 64),
    ]

    header = save_uploaded_record(uploads, tmp_path)

    assert header == tmp_path / "sample.hea"
    assert header.read_bytes() == make_header()
    assert (tmp_path / "sample.dat").read_bytes() == b"\x00" * 64


def test_synthetic_demo_waveform_is_deterministic_and_finite() -> None:
    first = create_synthetic_demo_waveform()
    second = create_synthetic_demo_waveform()

    assert first.shape == (12, 1_000)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)
    assert not np.array_equal(first[0], first[3])


def test_figures_and_table_follow_prediction_contract() -> None:
    waveform = np.zeros((12, 1_000), dtype=np.float32)
    report = {
        "predictions": [
            {
                "class": name,
                "probability": 0.8 if index == 0 else 0.1,
                "threshold": 0.5,
                "positive": index == 0,
                "decision_uncertainty": 0.2,
            }
            for index, name in enumerate(
                ("NORM", "MI", "STTC", "CD", "HYP")
            )
        ]
    }

    ecg_figure = create_ecg_figure(waveform)
    probability_figure = create_probability_figure(report)
    table = prediction_table(report)

    assert len(ecg_figure.axes) == 12
    assert len(probability_figure.axes) == 1
    assert list(table["분류 결과"]) == ["양성", "음성", "음성", "음성", "음성"]
