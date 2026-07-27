"""Fit training-referenced SQIs and audit their association with annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_guard.data.prepare_ptbxl import load_waveform, prepare_metadata
from ecg_guard.data.ptbxl_dataset import select_modeling_metadata
from ecg_guard.quality.sqi import (
    QUALITY_FEATURE_DIRECTIONS,
    extract_quality_features,
    fit_quality_reference,
    score_quality_features,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


QUALITY_ANNOTATION_COLUMNS = (
    "baseline_drift",
    "static_noise",
    "burst_noise",
    "electrodes_problems",
)


def extract_split_features(
    metadata: pd.DataFrame,
    data_dir: Path,
    *,
    progress_interval: int = 1_000,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(metadata.iterrows(), start=1):
        waveform = load_waveform(str(row["filename_lr"]), data_dir)
        rows.append(
            {
                "ecg_id": int(row["ecg_id"]),
                "patient_id": int(row["patient_id"]),
                "split": str(row["split"]),
                "artifact_annotation_present": bool(
                    row.loc[list(QUALITY_ANNOTATION_COLUMNS)].notna().any()
                ),
                **extract_quality_features(waveform),
            }
        )
        if progress_interval and position % progress_interval == 0:
            print(f"quality_features split={row['split']} records={position}")
    return pd.DataFrame(rows)


def apply_reference(
    frame: pd.DataFrame,
    reference: dict[str, Any],
) -> pd.DataFrame:
    scored = frame.copy()
    score_rows = []
    for _, row in scored.iterrows():
        features = {
            name: float(row[name])
            for name in QUALITY_FEATURE_DIRECTIONS
        }
        score_rows.append(score_quality_features(features, reference))
    scored["technical_quality_score"] = [
        row["technical_quality_score"] for row in score_rows
    ]
    scored["technical_review_score"] = [
        row["technical_review_score"] for row in score_rows
    ]
    scored["technical_quality_status"] = [
        row["technical_quality_status"] for row in score_rows
    ]
    scored["review_flags"] = [
        ";".join(row["review_flags"]) for row in score_rows
    ]
    return scored


def binary_association(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    cutoff = float(np.quantile(scores, 0.9))
    top = scores >= cutoff
    return {
        "records": int(len(labels)),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "top_decile_cutoff": cutoff,
        "top_decile_positive_rate": float(labels[top].mean()),
        "remaining_positive_rate": float(labels[~top].mean()),
    }


def prediction_error_association(
    quality_frame: pd.DataFrame,
    predictions_path: Path,
) -> dict[str, float | int]:
    predictions = pd.read_csv(predictions_path)
    prediction_columns = [
        column
        for column in predictions
        if column.startswith("prediction_")
    ]
    class_names = [column.removeprefix("prediction_") for column in prediction_columns]
    target_columns = [f"target_{name}" for name in class_names]
    predictions["any_class_error"] = (
        predictions[prediction_columns].to_numpy()
        != predictions[target_columns].to_numpy()
    ).any(axis=1)
    merged = quality_frame.merge(
        predictions.loc[:, ["ecg_id", "any_class_error"]],
        on="ecg_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(quality_frame):
        raise ValueError("quality rows and prediction rows do not align")
    return binary_association(
        merged["any_class_error"].to_numpy(),
        merged["technical_review_score"].to_numpy(),
    )


def plot_quality_audit(
    frames: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for axis, split in zip(
        axes[:2],
        ("validation", "test"),
        strict=True,
    ):
        frame = frames[split]
        for present, label, color in (
            (False, "annotation absent", "#167d8d"),
            (True, "annotation present", "#c43c35"),
        ):
            selected = frame["artifact_annotation_present"].eq(present)
            axis.hist(
                frame.loc[selected, "technical_review_score"],
                bins=30,
                alpha=0.55,
                density=True,
                label=label,
                color=color,
            )
        axis.set(
            title=f"{split}: technical review score",
            xlabel="Training-referenced tail score",
            ylabel="Density",
        )
        axis.grid(alpha=0.2)
        axis.legend()

    status_counts = (
        frames["test"]["technical_quality_status"]
        .value_counts()
        .reindex(["within_reference", "review", "extreme_outlier"], fill_value=0)
    )
    axes[2].bar(status_counts.index, status_counts.values, color="#526ca0")
    axes[2].set(
        title="Test technical quality status",
        ylabel="Records",
    )
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(axis="y", alpha=0.2)
    figure.suptitle("ECG Guard transparent signal-quality audit")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_report(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# 파형 기반 기술 품질 분석",
        "",
        "이 점수는 임상적 신호 품질 판정이나 artifact 검출 정답이 아닙니다. 학습",
        "파형 분포에서 설명 가능한 SQI가 얼마나 극단적인지 표시하는 검토 우선순위입니다.",
        "",
        "## Artifact 주석과의 약한 연관성 점검",
        "",
        "| 분할 | 기록 | 주석 비율 | AUROC | AP | 상위 10% 주석 비율 | 나머지 비율 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        values = results["artifact_annotation_association"][split]
        lines.append(
            f"| {split} | {values['records']} | {values['prevalence']:.4f} | "
            f"{values['auroc']:.4f} | {values['average_precision']:.4f} | "
            f"{values['top_decile_positive_rate']:.4f} | "
            f"{values['remaining_positive_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 모델 오류와의 연관성",
            "",
            "| 분할 | Any-label 오류율 | AUROC | AP | 상위 10% 오류율 | 나머지 오류율 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("validation", "test"):
        values = results["prediction_error_association"][split]
        lines.append(
            f"| {split} | {values['prevalence']:.4f} | "
            f"{values['auroc']:.4f} | {values['average_precision']:.4f} | "
            f"{values['top_decile_positive_rate']:.4f} | "
            f"{values['remaining_positive_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "시험 오류 탐지 AUROC가 무작위 수준이므로 기술 품질 점수를 모델",
            "신뢰도 점수로 사용하거나 uncertainty와 가중 합산하지 않습니다.",
            "",
            "## 품질 상태",
            "",
            "`review`는 하나 이상의 SQI가 학습 분포의 단측 99 percentile 이상,",
            "`extreme_outlier`는 99.9 percentile 이상인 경우입니다.",
            "",
        ]
    )
    for split in ("train", "validation", "test"):
        counts = results["status_counts"][split]
        lines.append(
            f"- {split}: within_reference={counts.get('within_reference', 0)}, "
            f"review={counts.get('review', 0)}, "
            f"extreme_outlier={counts.get('extreme_outlier', 0)}"
        )
    lines.extend(
        [
            "",
            "## 제한",
            "",
            "- 질병 형태 자체가 SQI의 극단값을 만들 수 있으므로 자동 제외에 쓰지 않습니다.",
            "- PTB-XL artifact 주석은 희소하고 주석 부재는 음성 정답이 아닙니다.",
            "- 100 Hz 입력에서는 50/60 Hz 전원선 간섭을 신뢰성 있게 분리하지 않습니다.",
            "- 점수는 PTB-XL 학습 분포 기준이므로 다른 장비·병원에 그대로 적용할 수 없습니다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ptb-xl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/quality-analysis"),
    )
    parser.add_argument(
        "--validation-predictions",
        type=Path,
        default=Path(
            "outputs/baseline-evaluation/validation_predictions.csv"
        ),
    )
    parser.add_argument(
        "--test-predictions",
        type=Path,
        default=Path("outputs/baseline-evaluation/test_predictions.csv"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = prepare_metadata(args.data_dir)
    split_metadata = {
        split: select_modeling_metadata(metadata, split)
        for split in ("train", "validation", "test")
    }
    raw_frames = {
        split: extract_split_features(frame, args.data_dir)
        for split, frame in split_metadata.items()
    }
    train_feature_matrix = {
        name: raw_frames["train"][name].to_numpy()
        for name in QUALITY_FEATURE_DIRECTIONS
    }
    reference = fit_quality_reference(train_feature_matrix)
    frames = {
        split: apply_reference(frame, reference)
        for split, frame in raw_frames.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        frame.to_csv(args.output_dir / f"{split}_quality_features.csv", index=False)
    (args.output_dir / "quality_reference.json").write_text(
        json.dumps(reference, indent=2) + "\n",
        encoding="utf-8",
    )

    results = {
        "method": (
            "transparent SQI tail score referenced to training waveform distribution"
        ),
        "automatic_rejection_allowed": False,
        "artifact_annotation_association": {
            split: binary_association(
                frames[split]["artifact_annotation_present"].to_numpy(),
                frames[split]["technical_review_score"].to_numpy(),
            )
            for split in ("validation", "test")
        },
        "prediction_error_association": {
            "validation": prediction_error_association(
                frames["validation"],
                args.validation_predictions,
            ),
            "test": prediction_error_association(
                frames["test"],
                args.test_predictions,
            ),
        },
        "status_counts": {
            split: {
                str(name): int(count)
                for name, count in frame[
                    "technical_quality_status"
                ].value_counts().items()
            }
            for split, frame in frames.items()
        },
    }
    (args.output_dir / "quality_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir / "quality_report.md", results)
    plot_quality_audit(
        frames,
        args.output_dir / "quality_audit.png",
    )
    print(
        "validation_artifact_auroc="
        f"{results['artifact_annotation_association']['validation']['auroc']:.6f} "
        "test_artifact_auroc="
        f"{results['artifact_annotation_association']['test']['auroc']:.6f}"
    )
    print(f"output={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
