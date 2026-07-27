"""Evaluate a locked baseline once on the held-out PTB-XL test fold."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ecg_guard.data.prepare_ptbxl import DIAGNOSTIC_CLASSES, prepare_metadata
from ecg_guard.data.ptbxl_dataset import (
    PTBXLDataset,
    NormalizationStats,
    select_modeling_metadata,
)
from ecg_guard.evaluation.metrics import (
    binary_nll_from_logits,
    evaluate_predictions,
    fit_temperature,
    fit_youden_thresholds,
    patient_cluster_bootstrap,
    sigmoid,
)
from ecg_guard.models import BaselineECGCNN
from ecg_guard.training.train_baseline import resolve_device, seed_worker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


QUALITY_COLUMNS = (
    "baseline_drift",
    "static_noise",
    "burst_noise",
    "electrodes_problems",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    *,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ecg_ids: list[np.ndarray] = []
    for batch in loader:
        signals = batch["signal"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            batch_logits = model(signals)
        logits.append(batch_logits.float().cpu().numpy())
        targets.append(batch["target"].numpy())
        ecg_ids.append(batch["ecg_id"].numpy())
    return (
        np.concatenate(logits),
        np.concatenate(targets).astype(np.int64),
        np.concatenate(ecg_ids),
    )


def make_loader(
    metadata: pd.DataFrame,
    data_dir: Path,
    normalization: NormalizationStats,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[dict[str, Tensor]]:
    return DataLoader(
        PTBXLDataset(metadata, data_dir, normalization),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )


def age_group(age: float) -> str:
    """Group valid ages while retaining PTB-XL's masked 90+ sentinel."""
    if age >= 300:
        return "masked_90_plus"
    if age < 40:
        return "under_40"
    if age < 60:
        return "40_to_59"
    return "60_plus"


def add_prediction_columns(
    metadata: pd.DataFrame,
    ecg_ids: np.ndarray,
    targets: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    if metadata["ecg_id"].tolist() != ecg_ids.astype(int).tolist():
        raise ValueError("prediction order does not match metadata")
    frame = metadata.loc[:, ["ecg_id", "patient_id", "age", "sex"]].copy()
    frame["age_group"] = metadata["age"].map(age_group)
    frame["artifact_annotation_present"] = (
        metadata.loc[:, QUALITY_COLUMNS].notna().any(axis=1).to_numpy()
    )
    for class_index, class_name in enumerate(DIAGNOSTIC_CLASSES):
        frame[f"target_{class_name}"] = targets[:, class_index]
        frame[f"raw_probability_{class_name}"] = raw_probabilities[:, class_index]
        frame[f"probability_{class_name}"] = calibrated_probabilities[:, class_index]
        frame[f"prediction_{class_name}"] = (
            calibrated_probabilities[:, class_index] >= thresholds[class_name]
        ).astype(int)
    return frame


def subgroup_evaluation(
    prediction_frame: pd.DataFrame,
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    groups = {
        "sex": {
            "sex_code_0": prediction_frame["sex"].eq(0).to_numpy(),
            "sex_code_1": prediction_frame["sex"].eq(1).to_numpy(),
        },
        "age": {
            name: prediction_frame["age_group"].eq(name).to_numpy()
            for name in ("under_40", "40_to_59", "60_plus")
        },
        "artifact_annotation": {
            "present": prediction_frame["artifact_annotation_present"].to_numpy(),
            "absent": ~prediction_frame[
                "artifact_annotation_present"
            ].to_numpy(),
        },
    }
    result: dict[str, Any] = {}
    patient_ids = prediction_frame["patient_id"].to_numpy()
    for family, family_groups in groups.items():
        result[family] = {}
        for name, selected in family_groups.items():
            metrics = evaluate_predictions(
                targets[selected],
                probabilities[selected],
                thresholds,
            )
            metrics["patients"] = int(np.unique(patient_ids[selected]).size)
            result[family][name] = metrics
    result["excluded"] = {
        "age_masked_90_plus_records": int(
            prediction_frame["age_group"].eq("masked_90_plus").sum()
        )
    }
    return result


def plot_evaluation(
    targets: np.ndarray,
    probabilities: np.ndarray,
    subgroup_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 11))
    colors = plt.cm.tab10.colors
    for index, class_name in enumerate(DIAGNOSTIC_CLASSES):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            targets[:, index],
            probabilities[:, index],
        )
        axes[0, 0].plot(
            false_positive_rate,
            true_positive_rate,
            label=class_name,
            color=colors[index],
        )
        precision, recall, _ = precision_recall_curve(
            targets[:, index],
            probabilities[:, index],
        )
        axes[0, 1].plot(
            recall,
            precision,
            label=class_name,
            color=colors[index],
        )

        bin_edges = np.linspace(0.0, 1.0, 11)
        bin_ids = np.minimum(
            np.digitize(probabilities[:, index], bin_edges[1:-1]),
            9,
        )
        observed: list[float] = []
        predicted: list[float] = []
        for bin_id in range(10):
            selected = bin_ids == bin_id
            if selected.any():
                observed.append(float(targets[selected, index].mean()))
                predicted.append(float(probabilities[selected, index].mean()))
        axes[1, 0].plot(
            predicted,
            observed,
            marker="o",
            label=class_name,
            color=colors[index],
        )

    axes[0, 0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0, 0].set(title="ROC curves", xlabel="False positive rate", ylabel="Sensitivity")
    axes[0, 1].set(title="Precision-recall curves", xlabel="Recall", ylabel="Precision")
    axes[1, 0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[1, 0].set(
        title="Reliability diagram",
        xlabel="Mean predicted probability",
        ylabel="Observed frequency",
    )
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.grid(alpha=0.25)
        axis.legend()

    labels: list[str] = []
    values: list[float] = []
    for family in ("sex", "age", "artifact_annotation"):
        for group_name, metrics in subgroup_metrics[family].items():
            labels.append(f"{family}\n{group_name}")
            values.append(float(metrics["macro"]["auroc"]))
    axes[1, 1].bar(range(len(values)), values, color="#167d8d")
    axes[1, 1].set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axes[1, 1].set_ylim(0.5, 1.0)
    axes[1, 1].set_ylabel("Macro AUROC")
    axes[1, 1].set_title("Test performance by subgroup")
    axes[1, 1].grid(axis="y", alpha=0.25)

    figure.suptitle("ECG Guard baseline test evaluation")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_json(path: Path, values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(values, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_summary(
    path: Path,
    test_metrics: dict[str, Any],
    bootstrap: dict[str, Any],
    temperature: float,
    thresholds: dict[str, float],
) -> None:
    macro = test_metrics["macro"]
    auroc_interval = bootstrap["macro"]["auroc"]
    ap_interval = bootstrap["macro"]["average_precision"]
    lines = [
        "# ECG Guard 기준 모델 시험 평가",
        "",
        "연구용 내부 벤치마크이며 임상적 유효성의 근거가 아닙니다.",
        "",
        f"- 기록: {test_metrics['records']}건, 환자: {bootstrap['patients']}명",
        f"- 검증에서 결정한 temperature: {temperature:.6f}",
        f"- 시험 macro AUROC: {macro['auroc']:.5f} "
        f"(95% patient-cluster bootstrap 구간 "
        f"{auroc_interval['lower']:.5f}–{auroc_interval['upper']:.5f})",
        f"- 시험 macro average precision: {macro['average_precision']:.5f} "
        f"(95% 구간 {ap_interval['lower']:.5f}–{ap_interval['upper']:.5f})",
        f"- 시험 macro 민감도: {macro['sensitivity']:.5f}",
        f"- 시험 macro 특이도: {macro['specificity']:.5f}",
        f"- 시험 macro Brier score: {macro['brier']:.5f}",
        "",
        "## 클래스별 결과",
        "",
        "| 클래스 | AUROC | Average precision | 민감도 | 특이도 | Precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for class_name in DIAGNOSTIC_CLASSES:
        metrics = test_metrics["per_class"][class_name]
        lines.append(
            f"| {class_name} | {metrics['auroc']:.5f} | "
            f"{metrics['average_precision']:.5f} | "
            f"{metrics['sensitivity']:.5f} | {metrics['specificity']:.5f} | "
            f"{metrics['precision']:.5f} |"
        )
    lines.extend(["", "## 검증에서 잠근 임계값", ""])
    lines.extend(
        f"- {class_name}: {thresholds[class_name]:.6f}"
        for class_name in DIAGNOSTIC_CLASSES
    )
    lines.extend(
        [
            "",
            "Temperature와 임계값은 검증 fold 9에서 고정했습니다. 시험 fold 10은 "
            "모델이나 평가 프로토콜 조정에 사용하지 않았습니다.",
            "",
            "Bootstrap 구간은 고정된 단일 seed 모델과 고정된 보정·임계값에 "
            "조건부이며, 학습 및 보정 과정 자체의 변동성은 포함하지 않습니다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/baseline/best_model.pt"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ptb-xl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/baseline-evaluation"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="exercise calibration and outputs without reading test waveforms",
    )
    parser.add_argument(
        "--allow-test-evaluation",
        action="store_true",
        help="explicitly permit the one-time held-out test evaluation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.validation_only and not args.allow_test_evaluation:
        raise RuntimeError(
            "test evaluation is locked; pass --allow-test-evaluation only "
            "after the protocol is fixed"
        )
    if (
        not args.validation_only
        and (args.output_dir / "test_metrics.json").exists()
    ):
        raise RuntimeError("test metrics already exist; refusing to evaluate again")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid DataLoader settings")
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")

    device = resolve_device(args.device)
    use_amp = device.type == "cuda"
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if tuple(checkpoint["config"]["classes"]) != DIAGNOSTIC_CLASSES:
        raise ValueError("checkpoint class order does not match the evaluator")
    if checkpoint["config"].get("test_fold_used_for_model_selection") is not False:
        raise ValueError("checkpoint does not document an untouched test fold")

    normalization = NormalizationStats.from_mapping(checkpoint["normalization"])
    model = BaselineECGCNN(
        dropout=float(checkpoint["config"]["dropout"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metadata = prepare_metadata(args.data_dir)
    validation_metadata = select_modeling_metadata(metadata, "validation")

    validation_loader = make_loader(
        validation_metadata,
        args.data_dir,
        normalization,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_logits, validation_targets, validation_ecg_ids = predict_logits(
        model,
        validation_loader,
        device,
        use_amp=use_amp,
    )
    temperature = fit_temperature(validation_logits, validation_targets)
    raw_validation_probabilities = sigmoid(validation_logits)
    validation_probabilities = sigmoid(validation_logits / temperature)
    thresholds = fit_youden_thresholds(
        validation_targets,
        validation_probabilities,
    )
    validation_metrics = evaluate_predictions(
        validation_targets,
        validation_probabilities,
        thresholds,
    )
    validation_calibration = {
        "temperature": temperature,
        "raw_nll": binary_nll_from_logits(
            validation_targets,
            validation_logits,
        ),
        "calibrated_nll": binary_nll_from_logits(
            validation_targets,
            validation_logits / temperature,
        ),
        "raw_metrics": evaluate_predictions(
            validation_targets,
            raw_validation_probabilities,
            {class_name: 0.5 for class_name in DIAGNOSTIC_CLASSES},
        ),
        "calibrated_metrics": validation_metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "validation_calibration.json", validation_calibration)
    validation_frame = add_prediction_columns(
        validation_metadata,
        validation_ecg_ids,
        validation_targets,
        raw_validation_probabilities,
        validation_probabilities,
        thresholds,
    )
    validation_frame.to_csv(
        args.output_dir / "validation_predictions.csv",
        index=False,
    )
    if args.validation_only:
        print(
            f"validation_records={len(validation_metadata)} "
            f"temperature={temperature:.6f} "
            f"macro_auroc={validation_metrics['macro']['auroc']:.6f}"
        )
        print("test_waveforms_read=False")
        print(f"output={args.output_dir.resolve()}")
        return 0

    test_metadata = select_modeling_metadata(metadata, "test")
    protocol_lock = {
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_records": len(validation_metadata),
        "test_records_expected": len(test_metadata),
        "classes": list(DIAGNOSTIC_CLASSES),
        "calibration": "single scalar temperature minimizing validation BCE",
        "temperature": temperature,
        "threshold_rule": "maximum validation Youden J; lowest tied threshold",
        "thresholds": thresholds,
        "confidence_intervals": (
            "95% percentile bootstrap resampling patient clusters"
        ),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "subgroups": {
            "sex": ["code 0", "code 1"],
            "age": ["<40", "40-59", "60+"],
            "age_exclusion": "age >=300 is masked 90+ and not assigned",
            "signal_quality": (
                "any PTB-XL artifact annotation present versus absent"
            ),
        },
    }
    write_json(args.output_dir / "protocol_lock.json", protocol_lock)
    test_loader = make_loader(
        test_metadata,
        args.data_dir,
        normalization,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_logits, test_targets, test_ecg_ids = predict_logits(
        model,
        test_loader,
        device,
        use_amp=use_amp,
    )
    raw_test_probabilities = sigmoid(test_logits)
    test_probabilities = sigmoid(test_logits / temperature)
    test_metrics = evaluate_predictions(
        test_targets,
        test_probabilities,
        thresholds,
    )
    test_metrics["raw_calibration"] = evaluate_predictions(
        test_targets,
        raw_test_probabilities,
        {class_name: 0.5 for class_name in DIAGNOSTIC_CLASSES},
    )["macro"]
    bootstrap = patient_cluster_bootstrap(
        test_metadata["patient_id"].to_numpy(),
        test_targets,
        test_probabilities,
        thresholds,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    test_frame = add_prediction_columns(
        test_metadata,
        test_ecg_ids,
        test_targets,
        raw_test_probabilities,
        test_probabilities,
        thresholds,
    )
    subgroups = subgroup_evaluation(
        test_frame,
        test_targets,
        test_probabilities,
        thresholds,
    )

    write_json(args.output_dir / "test_metrics.json", test_metrics)
    write_json(args.output_dir / "bootstrap_intervals.json", bootstrap)
    write_json(args.output_dir / "subgroup_metrics.json", subgroups)
    test_frame.to_csv(args.output_dir / "test_predictions.csv", index=False)
    plot_evaluation(
        test_targets,
        test_probabilities,
        subgroups,
        args.output_dir / "evaluation_curves.png",
    )
    write_summary(
        args.output_dir / "evaluation_summary.md",
        test_metrics,
        bootstrap,
        temperature,
        thresholds,
    )
    print(
        f"test_records={len(test_metadata)} "
        f"patients={test_metadata['patient_id'].nunique()} "
        f"macro_auroc={test_metrics['macro']['auroc']:.6f} "
        f"macro_ap={test_metrics['macro']['average_precision']:.6f}"
    )
    print(f"output={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
