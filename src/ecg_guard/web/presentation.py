"""Validation and visual presentation helpers for the web demo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ecg_guard.data.prepare_ptbxl import (
    DIAGNOSTIC_CLASSES,
    EXPECTED_SAMPLES,
    EXPECTED_SAMPLING_FREQUENCY,
    LEAD_NAMES,
)


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
CLASS_LABELS_KO = {
    "NORM": "정상",
    "MI": "심근경색",
    "STTC": "ST-T 변화",
    "CD": "전도장애",
    "HYP": "심비대",
}
ACTION_LABELS_KO = {
    "auto_result": "자동 결과 대상",
    "review_uncertain": "불확실성 검토 필요",
    "review_technical": "신호 품질 검토 필요",
    "review_both": "불확실성·신호 품질 검토 필요",
}
QUALITY_FLAG_LABELS_KO = {
    "baseline_wander_ratio_max": "기준선 변동",
    "high_frequency_ratio_max": "고주파 성분",
    "flatline_fraction_max": "평탄 신호",
    "amplitude_range_max": "과도한 진폭 범위",
    "amplitude_range_min": "낮은 진폭 범위",
    "derivative_ratio_max": "급격한 신호 변화",
    "lead_std_min": "낮은 유도 변동성",
}


def create_synthetic_demo_waveform() -> np.ndarray:
    """Create a deterministic non-clinical waveform for UI demonstrations."""
    time = np.arange(EXPECTED_SAMPLES, dtype=np.float64) / 100.0
    phase = np.mod(time, 1.0)
    heartbeat = (
        0.10 * np.exp(-((phase - 0.16) / 0.045) ** 2)
        - 0.14 * np.exp(-((phase - 0.285) / 0.014) ** 2)
        + 1.00 * np.exp(-((phase - 0.31) / 0.012) ** 2)
        - 0.24 * np.exp(-((phase - 0.345) / 0.016) ** 2)
        + 0.25 * np.exp(-((phase - 0.58) / 0.09) ** 2)
    )
    baseline = 0.015 * np.sin(2 * np.pi * 0.2 * time)
    lead_scales = np.asarray(
        [
            0.72,
            1.0,
            0.66,
            -0.82,
            0.38,
            0.84,
            -0.42,
            0.34,
            0.78,
            1.0,
            0.88,
            0.70,
        ],
        dtype=np.float64,
    )
    waveform = lead_scales[:, None] * heartbeat[None, :] + baseline[None, :]
    return waveform.astype(np.float32)


class UploadedFileLike(Protocol):
    """Minimal interface shared by Streamlit uploads and unit-test doubles."""

    name: str
    size: int

    def getvalue(self) -> bytes: ...


def validate_upload_set(
    file_names: Sequence[str],
    file_sizes: Sequence[int],
    *,
    maximum_bytes: int = MAX_UPLOAD_BYTES,
) -> str:
    """Validate a bounded WFDB upload set and return the header filename."""
    if len(file_names) != len(file_sizes) or not file_names:
        raise ValueError("업로드 파일 이름과 크기 정보가 올바르지 않습니다.")
    if any(size < 0 for size in file_sizes):
        raise ValueError("업로드 파일 크기가 올바르지 않습니다.")
    if sum(file_sizes) > maximum_bytes:
        raise ValueError("업로드 파일 전체 크기는 50MB를 넘을 수 없습니다.")

    normalized: list[str] = []
    for name in file_names:
        if (
            not name
            or Path(name).name != name
            or "/" in name
            or "\\" in name
            or any(ord(character) < 32 for character in name)
        ):
            raise ValueError("파일 이름에는 경로나 제어 문자를 사용할 수 없습니다.")
        normalized.append(name)
    if len({name.casefold() for name in normalized}) != len(normalized):
        raise ValueError("이름이 중복되는 파일이 있습니다.")

    unsupported = [
        name
        for name in normalized
        if Path(name).suffix.lower() not in {".hea", ".dat"}
    ]
    if unsupported:
        raise ValueError("WFDB .hea와 .dat 파일만 업로드할 수 있습니다.")
    headers = [
        name for name in normalized if Path(name).suffix.lower() == ".hea"
    ]
    data_files = [
        name for name in normalized if Path(name).suffix.lower() == ".dat"
    ]
    if len(headers) != 1 or not data_files:
        raise ValueError(".hea 파일 1개와 하나 이상의 .dat 파일이 필요합니다.")
    return headers[0]


def validate_wfdb_header(
    header_bytes: bytes,
    uploaded_names: Sequence[str],
) -> None:
    """Validate the declared signal shape and referenced upload filenames."""
    try:
        header = header_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("WFDB 헤더는 ASCII 텍스트여야 합니다.") from error

    lines = [line.strip() for line in header.splitlines() if line.strip()]
    if not lines:
        raise ValueError("WFDB 헤더가 비어 있습니다.")
    fields = lines[0].split()
    if len(fields) < 4:
        raise ValueError("WFDB 헤더 첫 줄의 형식이 올바르지 않습니다.")
    try:
        signal_count = int(fields[1])
        sampling_frequency = int(float(fields[2].split("/")[0]))
        sample_count = int(fields[3])
    except ValueError as error:
        raise ValueError("WFDB 헤더의 신호 크기 정보를 읽을 수 없습니다.") from error

    if signal_count != len(LEAD_NAMES):
        raise ValueError("ECG Guard는 표준 12유도 입력만 지원합니다.")
    if sampling_frequency != EXPECTED_SAMPLING_FREQUENCY:
        raise ValueError("ECG Guard는 100Hz 입력만 지원합니다.")
    if sample_count != EXPECTED_SAMPLES:
        raise ValueError("ECG Guard는 유도당 1,000표본 입력만 지원합니다.")
    if len(lines) < signal_count + 1:
        raise ValueError("WFDB 헤더에 12개 신호 정의가 필요합니다.")

    uploaded = set(uploaded_names)
    signal_lines = [line.split() for line in lines[1 : signal_count + 1]]
    referenced_files = [fields[0] for fields in signal_lines if fields]
    if len(referenced_files) != signal_count:
        raise ValueError("WFDB 신호 정의를 읽을 수 없습니다.")
    for name in referenced_files:
        if Path(name).name != name or name not in uploaded:
            raise ValueError("헤더가 참조하는 .dat 파일을 함께 업로드해 주세요.")
    signal_names = tuple(fields[-1].upper() for fields in signal_lines)
    if signal_names != LEAD_NAMES:
        raise ValueError(
            "유도 순서는 I, II, III, aVR, aVL, aVF, V1~V6이어야 합니다."
        )


def save_uploaded_record(
    uploaded_files: Sequence[UploadedFileLike],
    directory: Path,
) -> Path:
    """Validate and save one upload set into an isolated temporary directory."""
    names = [file.name for file in uploaded_files]
    sizes = [file.size for file in uploaded_files]
    header_name = validate_upload_set(names, sizes)
    uploads = {file.name: file for file in uploaded_files}
    validate_wfdb_header(uploads[header_name].getvalue(), names)

    directory.mkdir(parents=True, exist_ok=True)
    for name, uploaded in uploads.items():
        (directory / name).write_bytes(uploaded.getvalue())
    return directory / header_name


def create_ecg_figure(waveform: np.ndarray) -> Figure:
    """Create a readable 12-lead, ten-second waveform overview."""
    waveform = np.asarray(waveform, dtype=np.float64)
    expected_shape = (len(LEAD_NAMES), EXPECTED_SAMPLES)
    if waveform.shape != expected_shape or not np.isfinite(waveform).all():
        raise ValueError(f"expected finite waveform shaped {expected_shape}")

    time = np.arange(EXPECTED_SAMPLES) / EXPECTED_SAMPLING_FREQUENCY
    figure, axes = plt.subplots(
        6,
        2,
        figsize=(14, 10),
        sharex=True,
        constrained_layout=True,
    )
    figure.patch.set_facecolor("#F7F9FC")
    for axis, lead_name, values in zip(
        axes.flat,
        LEAD_NAMES,
        waveform,
        strict=True,
    ):
        axis.set_facecolor("#FFFFFF")
        axis.plot(time, values, color="#0F5D63", linewidth=0.85)
        axis.set_title(lead_name, loc="left", fontsize=10, fontweight="bold")
        axis.grid(color="#CBD7DA", alpha=0.55, linewidth=0.55)
        axis.tick_params(labelsize=8, colors="#52656D")
        axis.set_ylabel("mV", fontsize=8, color="#52656D")
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)", fontsize=9, color="#52656D")
    return figure


def create_probability_figure(report: Mapping[str, Any]) -> Figure:
    """Visualize calibrated probabilities with their locked thresholds."""
    predictions = report["predictions"]
    if [row["class"] for row in predictions] != list(DIAGNOSTIC_CLASSES):
        raise ValueError("prediction class order is invalid")

    probabilities = np.asarray(
        [row["probability"] for row in predictions],
        dtype=np.float64,
    )
    thresholds = np.asarray(
        [row["threshold"] for row in predictions],
        dtype=np.float64,
    )
    positives = np.asarray(
        [row["positive"] for row in predictions],
        dtype=bool,
    )
    labels = list(DIAGNOSTIC_CLASSES)
    colors = np.where(positives, "#0F766E", "#91A4AA")

    figure, axis = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    figure.patch.set_facecolor("#F7F9FC")
    axis.set_facecolor("#FFFFFF")
    positions = np.arange(len(labels))
    axis.barh(positions, probabilities, color=colors, height=0.58)
    axis.scatter(
        thresholds,
        positions,
        marker="|",
        s=350,
        linewidths=2.2,
        color="#C75B39",
        label="Locked validation threshold",
        zorder=3,
    )
    for position, probability in zip(
        positions,
        probabilities,
        strict=True,
    ):
        axis.text(
            min(probability + 0.018, 0.94),
            position,
            f"{probability:.1%}",
            va="center",
            fontsize=9,
            color="#20343B",
        )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("Calibrated probability")
    axis.grid(axis="x", color="#D8E1E3", linewidth=0.6)
    axis.legend(loc="lower right", frameon=False, fontsize=8)
    return figure


def prediction_table(report: Mapping[str, Any]) -> pd.DataFrame:
    """Build the compact result table displayed and exported by the app."""
    rows = []
    for prediction in report["predictions"]:
        class_name = str(prediction["class"])
        rows.append(
            {
                "진단군": f"{class_name} · {CLASS_LABELS_KO[class_name]}",
                "보정 확률": float(prediction["probability"]),
                "임계값": float(prediction["threshold"]),
                "분류 결과": "양성" if prediction["positive"] else "음성",
                "경계 불확실성": float(
                    prediction["decision_uncertainty"]
                ),
            }
        )
    return pd.DataFrame(rows)


def quality_flag_text(flags: Sequence[str]) -> str:
    """Translate technical feature identifiers without inferring a diagnosis."""
    if not flags:
        return "학습 참조 분포의 99백분위 초과 항목 없음"
    return ", ".join(QUALITY_FLAG_LABELS_KO.get(flag, flag) for flag in flags)
