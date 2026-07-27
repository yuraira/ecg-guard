"""Describe frozen-model errors without retuning the classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd

from ecg_guard.data.prepare_ptbxl import (
    DIAGNOSTIC_CLASSES,
    EXPECTED_SAMPLING_FREQUENCY,
    LEAD_NAMES,
    load_waveform,
    prepare_metadata,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DISPLAY_LEADS = ("II", "V1", "V5")


def error_masks(
    frame: pd.DataFrame,
    class_name: str,
) -> dict[str, np.ndarray]:
    target = frame[f"target_{class_name}"].to_numpy(dtype=int)
    prediction = frame[f"prediction_{class_name}"].to_numpy(dtype=int)
    return {
        "true_positive": (target == 1) & (prediction == 1),
        "false_positive": (target == 0) & (prediction == 1),
        "false_negative": (target == 1) & (prediction == 0),
        "true_negative": (target == 0) & (prediction == 0),
    }


def select_confident_errors(
    frame: pd.DataFrame,
    class_name: str,
    error_type: str,
    *,
    limit: int,
) -> pd.DataFrame:
    """Select strongest errors for descriptive inspection."""
    if error_type not in {"false_positive", "false_negative"}:
        raise ValueError("error_type must be false_positive or false_negative")
    if limit <= 0:
        raise ValueError("limit must be positive")
    selected = frame.loc[error_masks(frame, class_name)[error_type]].copy()
    probability_column = f"probability_{class_name}"
    ascending = error_type == "false_negative"
    return selected.sort_values(
        [probability_column, "ecg_id"],
        ascending=[ascending, True],
    ).head(limit)


def class_error_summary(
    frame: pd.DataFrame,
    class_name: str,
) -> dict[str, Any]:
    masks = error_masks(frame, class_name)
    result: dict[str, Any] = {
        name: int(mask.sum())
        for name, mask in masks.items()
    }
    for error_type in ("false_positive", "false_negative"):
        selected = masks[error_type]
        cohort = frame.loc[selected]
        result[f"{error_type}_artifact_annotation_rate"] = (
            float(cohort["artifact_annotation_present"].mean())
            if len(cohort)
            else None
        )
        result[f"{error_type}_mean_probability"] = (
            float(cohort[f"probability_{class_name}"].mean())
            if len(cohort)
            else None
        )
        result[f"{error_type}_coexisting_target_rate"] = {
            other: float(cohort[f"target_{other}"].mean())
            for other in DIAGNOSTIC_CLASSES
            if other != class_name and len(cohort)
        }
    return result


def plot_error_gallery(
    examples: pd.DataFrame,
    metadata: pd.DataFrame,
    data_dir: Path,
    class_name: str,
    output_path: Path,
) -> None:
    if examples.empty:
        raise ValueError("no examples supplied")
    metadata_by_id = metadata.set_index("ecg_id")
    lead_indices = [LEAD_NAMES.index(lead) for lead in DISPLAY_LEADS]
    time_seconds = np.arange(1_000) / EXPECTED_SAMPLING_FREQUENCY
    figure, axes = plt.subplots(
        len(examples),
        len(DISPLAY_LEADS),
        figsize=(15, 2.7 * len(examples)),
        sharex=True,
        squeeze=False,
    )

    for row_index, (_, example) in enumerate(examples.iterrows()):
        ecg_id = int(example["ecg_id"])
        metadata_row = metadata_by_id.loc[ecg_id]
        waveform = load_waveform(str(metadata_row["filename_lr"]), data_dir)
        target = int(example[f"target_{class_name}"])
        prediction = int(example[f"prediction_{class_name}"])
        error_name = "FP" if target == 0 and prediction == 1 else "FN"
        probability = float(example[f"probability_{class_name}"])
        for column_index, (lead_name, lead_index) in enumerate(
            zip(DISPLAY_LEADS, lead_indices, strict=True)
        ):
            axis = axes[row_index, column_index]
            axis.plot(
                time_seconds,
                waveform[lead_index],
                color="#a4262c",
                linewidth=0.7,
            )
            axis.grid(alpha=0.25)
            axis.set_title(
                f"ECG {ecg_id} · {error_name} · p={probability:.3f} · {lead_name}",
                fontsize=9,
            )
            if column_index == 0:
                axis.set_ylabel("mV")
            if row_index == len(examples) - 1:
                axis.set_xlabel("Time (s)")

    figure.suptitle(
        f"{class_name} strongest errors · descriptive post-test review",
        fontsize=14,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_report(
    output_path: Path,
    summaries: dict[str, dict[str, Any]],
    selected_examples: dict[str, list[dict[str, Any]]],
    overall_artifact_rate: float,
) -> None:
    lines = [
        "# Baseline v1 사후 오류 분석",
        "",
        "이 분석은 동결된 시험 결과를 설명하기 위한 기술 분석입니다. 결과를 이용해",
        "baseline v1을 다시 튜닝하거나 동일 시험 fold의 새 성능을 주장하지 않습니다.",
        "",
        f"- 전체 artifact annotation 비율: {overall_artifact_rate:.4f}",
        "",
        "## 클래스별 오류 수",
        "",
        "| 클래스 | TP | FP | FN | TN | FP artifact 비율 | FN artifact 비율 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for class_name in DIAGNOSTIC_CLASSES:
        values = summaries[class_name]
        lines.append(
            f"| {class_name} | {values['true_positive']} | "
            f"{values['false_positive']} | {values['false_negative']} | "
            f"{values['true_negative']} | "
            f"{values['false_positive_artifact_annotation_rate']:.4f} | "
            f"{values['false_negative_artifact_annotation_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 대표 오류",
            "",
            "확률이 높은 FP와 확률이 낮은 FN을 기계적으로 선택했습니다. 파형을 보고",
            "새 임상 진단이나 오류 원인을 추정하지 않습니다.",
            "",
        ]
    )
    for class_name, examples in selected_examples.items():
        lines.append(f"### {class_name}")
        lines.append("")
        lines.append("| ECG ID | 오류 | 확률 | 연령 그룹 | 성별 코드 | Artifact 주석 |")
        lines.append("|---:|---|---:|---|---:|---|")
        for example in examples:
            error_name = (
                "FP"
                if example[f"target_{class_name}"] == 0
                else "FN"
            )
            lines.append(
                f"| {example['ecg_id']} | {error_name} | "
                f"{example[f'probability_{class_name}']:.4f} | "
                f"{example['age_group']} | {example['sex']} | "
                f"{example['artifact_annotation_present']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 해석 제한",
            "",
            "- Artifact 주석 부재는 고품질 신호를 의미하지 않습니다.",
            "- 오류 집단의 동반 라벨과 인구통계 차이는 원인이나 편향의 증거가 아닙니다.",
            "- 파형 갤러리는 임상 판독을 대체하지 않습니다.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/baseline-evaluation/test_predictions.csv"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ptb-xl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/error-analysis"),
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        choices=DIAGNOSTIC_CLASSES,
        default=["HYP", "MI"],
    )
    parser.add_argument("--examples-per-error", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.examples_per_error <= 0:
        raise ValueError("--examples-per-error must be positive")

    frame = pd.read_csv(args.predictions)
    metadata = prepare_metadata(args.data_dir)
    summaries = {
        class_name: class_error_summary(frame, class_name)
        for class_name in DIAGNOSTIC_CLASSES
    }
    selected_examples: dict[str, list[dict[str, Any]]] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for class_name in args.classes:
        false_positives = select_confident_errors(
            frame,
            class_name,
            "false_positive",
            limit=args.examples_per_error,
        )
        false_negatives = select_confident_errors(
            frame,
            class_name,
            "false_negative",
            limit=args.examples_per_error,
        )
        examples = pd.concat(
            [false_positives, false_negatives],
            ignore_index=True,
        )
        selected_examples[class_name] = examples.to_dict(orient="records")
        plot_error_gallery(
            examples,
            metadata,
            args.data_dir,
            class_name,
            args.output_dir / f"{class_name.lower()}_error_gallery.png",
        )

    payload = {
        "analysis_type": "post-test descriptive analysis of frozen baseline v1",
        "records": len(frame),
        "overall_artifact_annotation_rate": float(
            frame["artifact_annotation_present"].mean()
        ),
        "classes": summaries,
        "selected_examples": selected_examples,
    }
    (args.output_dir / "error_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(
        args.output_dir / "error_analysis.md",
        summaries,
        selected_examples,
        payload["overall_artifact_annotation_rate"],
    )
    print(f"records={len(frame)} classes={args.classes}")
    print(f"output={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
