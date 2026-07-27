"""Audit uncertainty routing for the frozen ECG Guard baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_guard.data.prepare_ptbxl import DIAGNOSTIC_CLASSES
from ecg_guard.uncertainty.selective import (
    decision_boundary_uncertainty,
    fit_coverage_cutoffs,
    prediction_error_arrays,
    predictive_entropy,
    risk_coverage_curve,
    selective_metrics,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TARGET_COVERAGES = (0.9, 0.8, 0.7)
DEMO_COVERAGE = "0.80"


def prediction_arrays(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    targets = frame[
        [f"target_{name}" for name in DIAGNOSTIC_CLASSES]
    ].to_numpy(dtype=int)
    probabilities = frame[
        [f"probability_{name}" for name in DIAGNOSTIC_CLASSES]
    ].to_numpy(dtype=float)
    return targets, probabilities


def add_uncertainty_columns(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    targets, probabilities = prediction_arrays(frame)
    record_uncertainty, class_uncertainty = decision_boundary_uncertainty(
        probabilities,
        thresholds,
    )
    entropy = predictive_entropy(probabilities)
    any_error, hamming_error = prediction_error_arrays(
        targets,
        probabilities,
        thresholds,
    )
    result = frame.copy()
    result["decision_uncertainty"] = record_uncertainty
    result["maximum_predictive_entropy"] = entropy.max(axis=1)
    result["mean_predictive_entropy"] = entropy.mean(axis=1)
    result["any_class_error"] = any_error
    result["hamming_error"] = hamming_error
    for index, class_name in enumerate(DIAGNOSTIC_CLASSES):
        result[f"uncertainty_{class_name}"] = class_uncertainty[:, index]
    return result


def error_detection_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    labels = frame["any_class_error"].to_numpy(dtype=int)
    scores = frame["decision_uncertainty"].to_numpy(dtype=float)
    return {
        "records": int(len(frame)),
        "errors": int(labels.sum()),
        "error_prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def evaluate_cutoffs(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    cutoffs: dict[str, float],
) -> dict[str, Any]:
    targets, probabilities = prediction_arrays(frame)
    uncertainty = frame["decision_uncertainty"].to_numpy()
    return {
        target: selective_metrics(
            targets,
            probabilities,
            thresholds,
            uncertainty,
            cutoff,
        )
        for target, cutoff in cutoffs.items()
    }


def assign_review_action(
    frame: pd.DataFrame,
    cutoff: float,
) -> pd.Series:
    uncertain = frame["decision_uncertainty"].gt(cutoff)
    technical = frame["technical_quality_status"].ne("within_reference")
    actions = np.full(len(frame), "auto_result", dtype=object)
    actions[uncertain.to_numpy()] = "review_uncertain"
    actions[technical.to_numpy()] = "review_technical"
    actions[(uncertain & technical).to_numpy()] = "review_both"
    return pd.Series(actions, index=frame.index, dtype="string")


def plot_selective_audit(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    curves: dict[str, Any],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    for frame, split, color in (
        (validation, "validation", "#167d8d"),
        (test, "test", "#c43c35"),
    ):
        curve = curves[split]
        axes[0, 0].plot(
            curve["coverage"],
            curve["hamming_error_rate"],
            label=split,
            color=color,
        )
        axes[0, 1].plot(
            curve["coverage"],
            curve["exact_match_error_rate"],
            label=split,
            color=color,
        )
        for error, label, alpha in (
            (False, "correct", 0.45),
            (True, "any-label error", 0.55),
        ):
            selected = frame["any_class_error"].eq(error)
            axes[1, 0].hist(
                frame.loc[selected, "decision_uncertainty"],
                bins=30,
                density=True,
                alpha=alpha,
                label=f"{split} {label}",
            )

    axes[0, 0].set(
        title="Hamming risk-coverage",
        xlabel="Coverage",
        ylabel="Hamming error rate",
    )
    axes[0, 1].set(
        title="Exact-match risk-coverage",
        xlabel="Coverage",
        ylabel="Any-label error rate",
    )
    axes[1, 0].set(
        title="Uncertainty by prediction correctness",
        xlabel="Decision-boundary uncertainty",
        ylabel="Density",
    )
    for axis in axes.flat[:3]:
        axis.grid(alpha=0.25)
        axis.legend()

    action_counts = test["review_action"].value_counts()
    axes[1, 1].bar(action_counts.index, action_counts.values, color="#526ca0")
    axes[1, 1].set(title="Test routing at validation 80% cutoff", ylabel="Records")
    axes[1, 1].tick_params(axis="x", rotation=25)
    axes[1, 1].grid(axis="y", alpha=0.25)

    figure.suptitle("ECG Guard uncertainty and selective prediction audit")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_report(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# 예측 불확실성과 선택적 판정 분석",
        "",
        "단일 모델 확률의 결정 경계 근접도를 uncertainty proxy로 사용합니다.",
        "Epistemic uncertainty나 임상적 위험 확률로 해석하지 않습니다.",
        "",
        "## 오류 탐지",
        "",
        "| 분할 | 기록 | any-label 오류율 | AUROC | AP |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        values = results["error_detection"][split]
        lines.append(
            f"| {split} | {values['records']} | "
            f"{values['error_prevalence']:.4f} | {values['auroc']:.4f} | "
            f"{values['average_precision']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 검증에서 고정한 coverage cutoff의 시험 결과",
            "",
            f"전체 시험 Hamming 오류율은 "
            f"{results['risk_coverage']['test']['hamming_error_rate'][-1]:.4f}, "
            f"Any-label 오류율은 "
            f"{results['error_detection']['test']['error_prevalence']:.4f}입니다.",
            "",
            "| 목표 coverage | 실제 coverage | Hamming 오류율 | "
            "Any-label 오류율 | Macro AUROC | 민감도 | 특이도 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for target, values in results["selective_metrics"]["test"].items():
        macro = values["macro"]
        lines.append(
            f"| {target} | {values['coverage']:.4f} | "
            f"{values['hamming_error_rate']:.4f} | "
            f"{values['exact_match_error_rate']:.4f} | "
            f"{macro['auroc']:.4f} | {macro['sensitivity']:.4f} | "
            f"{macro['specificity']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 데모 라우팅",
            "",
        ]
    )
    for action, count in results["test_action_counts"].items():
        lines.append(f"- {action}: {count}건")
    lines.extend(
        [
            "",
            "기술 품질 경고는 오류 탐지력이 확인되지 않았기 때문에 uncertainty와",
            "하나의 수치로 합치지 않고 별도 review 사유로 유지합니다.",
            "",
            "## 제한",
            "",
            "- 단일 모델의 경계 근접도는 epistemic uncertainty를 측정하지 않습니다.",
            "- 판정 보류가 환자 안전이나 임상 효용을 높인다는 검증은 하지 않았습니다.",
            "- 보류 집단에서 질병 유병률과 하위집단 구성이 달라질 수 있습니다.",
            "- 80% coverage는 제품 데모용 운영점이며 임상적으로 승인된 기준이 아닙니다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path("outputs/baseline-evaluation"),
    )
    parser.add_argument(
        "--quality-dir",
        type=Path,
        default=Path("outputs/quality-analysis"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/selective-analysis"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = json.loads(
        (args.evaluation_dir / "protocol_lock.json").read_text(encoding="utf-8")
    )
    thresholds = {
        name: float(value)
        for name, value in protocol["thresholds"].items()
    }
    frames: dict[str, pd.DataFrame] = {}
    for split in ("validation", "test"):
        predictions = pd.read_csv(
            args.evaluation_dir / f"{split}_predictions.csv"
        )
        quality = pd.read_csv(
            args.quality_dir / f"{split}_quality_features.csv"
        )
        frame = predictions.merge(
            quality.loc[
                :,
                [
                    "ecg_id",
                    "technical_quality_score",
                    "technical_review_score",
                    "technical_quality_status",
                    "review_flags",
                ],
            ],
            on="ecg_id",
            how="inner",
            validate="one_to_one",
        )
        if len(frame) != len(predictions):
            raise ValueError(f"{split} predictions and quality rows do not align")
        frames[split] = add_uncertainty_columns(frame, thresholds)

    validation_cutoffs = fit_coverage_cutoffs(
        frames["validation"]["decision_uncertainty"].to_numpy(),
        TARGET_COVERAGES,
    )
    for frame in frames.values():
        frame["review_action"] = assign_review_action(
            frame,
            validation_cutoffs[DEMO_COVERAGE],
        )

    curves = {}
    for split, frame in frames.items():
        targets, probabilities = prediction_arrays(frame)
        curves[split] = risk_coverage_curve(
            targets,
            probabilities,
            thresholds,
            frame["decision_uncertainty"].to_numpy(),
        )
    results = {
        "uncertainty_proxy": (
            "maximum exp(-absolute log-odds distance to class decision boundary)"
        ),
        "epistemic_uncertainty": False,
        "validation_coverage_cutoffs": validation_cutoffs,
        "error_detection": {
            split: error_detection_metrics(frame)
            for split, frame in frames.items()
        },
        "selective_metrics": {
            split: evaluate_cutoffs(frame, thresholds, validation_cutoffs)
            for split, frame in frames.items()
        },
        "risk_coverage": curves,
        "demo_coverage": float(DEMO_COVERAGE),
        "test_action_counts": {
            str(action): int(count)
            for action, count in frames["test"][
                "review_action"
            ].value_counts().items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in frames.items():
        frame.to_csv(
            args.output_dir / f"{split}_selective_predictions.csv",
            index=False,
        )
    (args.output_dir / "selective_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir / "selective_report.md", results)
    plot_selective_audit(
        frames["validation"],
        frames["test"],
        curves,
        args.output_dir / "selective_audit.png",
    )
    print(
        "validation_error_detection_auroc="
        f"{results['error_detection']['validation']['auroc']:.6f} "
        "test_error_detection_auroc="
        f"{results['error_detection']['test']['auroc']:.6f}"
    )
    print(f"output={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
