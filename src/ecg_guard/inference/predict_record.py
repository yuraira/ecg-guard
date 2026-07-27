"""Run one locked ECG Guard baseline prediction with review routing."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ecg_guard.data.prepare_ptbxl import (
    DIAGNOSTIC_CLASSES,
    EXPECTED_SAMPLES,
    EXPECTED_SAMPLING_FREQUENCY,
    LEAD_NAMES,
    load_waveform,
)
from ecg_guard.data.ptbxl_dataset import NormalizationStats
from ecg_guard.evaluation.metrics import sigmoid
from ecg_guard.models import BaselineECGCNN
from ecg_guard.quality.sqi import (
    QUALITY_FEATURE_DIRECTIONS,
    extract_quality_features,
    score_quality_features,
)
from ecg_guard.uncertainty.selective import (
    decision_boundary_uncertainty,
    predictive_entropy,
)


RESEARCH_WARNING = (
    "연구·교육용 결과이며 의료 진단, 선별, 치료 또는 응급 의사결정에 "
    "사용할 수 없습니다."
)
DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "baseline_v1_inference.json"
)


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inference_protocol(path: Path) -> dict[str, Any]:
    """Load and strictly validate the versioned inference contract."""
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "1.0":
        raise ValueError("unsupported inference protocol schema")
    if tuple(protocol.get("classes", ())) != DIAGNOSTIC_CLASSES:
        raise ValueError("protocol class order does not match the model")

    input_contract = protocol.get("input", {})
    expected_input = {
        "sampling_frequency_hz": EXPECTED_SAMPLING_FREQUENCY,
        "samples_per_lead": EXPECTED_SAMPLES,
        "lead_order": list(LEAD_NAMES),
        "amplitude_unit": "mV",
    }
    for name, expected in expected_input.items():
        if input_contract.get(name) != expected:
            raise ValueError(f"protocol input field {name!r} is invalid")

    temperature = float(protocol.get("calibration", {}).get("temperature", 0))
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("protocol temperature must be finite and positive")

    thresholds = protocol.get("thresholds", {}).get("values", {})
    if set(thresholds) != set(DIAGNOSTIC_CLASSES):
        raise ValueError("protocol thresholds must cover every class")
    if not all(0 < float(value) < 1 for value in thresholds.values()):
        raise ValueError("protocol thresholds must be in (0, 1)")

    selective = protocol.get("selective_prediction", {})
    cutoff = float(selective.get("uncertainty_cutoff", -1))
    coverage = float(selective.get("target_coverage", -1))
    if not 0 <= cutoff <= 1 or not 0 < coverage <= 1:
        raise ValueError("protocol selective prediction settings are invalid")

    reference = protocol.get("quality_reference", {})
    probabilities = np.asarray(
        reference.get("quantile_probabilities", ()),
        dtype=np.float64,
    )
    if (
        probabilities.ndim != 1
        or len(probabilities) < 2
        or not np.isclose(probabilities[0], 0)
        or not np.isclose(probabilities[-1], 1)
        or np.any(np.diff(probabilities) <= 0)
    ):
        raise ValueError("protocol quality probabilities are invalid")
    feature_reference = reference.get("features", {})
    if set(feature_reference) != set(QUALITY_FEATURE_DIRECTIONS):
        raise ValueError("protocol quality reference has unexpected features")
    for name, direction in QUALITY_FEATURE_DIRECTIONS.items():
        values = np.asarray(
            feature_reference[name].get("quantiles", ()),
            dtype=np.float64,
        )
        if (
            feature_reference[name].get("direction") != direction
            or values.shape != probabilities.shape
            or not np.isfinite(values).all()
            or np.any(np.diff(values) < 0)
        ):
            raise ValueError(f"protocol quality feature {name!r} is invalid")

    expected_digest = str(protocol.get("checkpoint_sha256", "")).lower()
    if len(expected_digest) != 64 or any(
        character not in "0123456789abcdef" for character in expected_digest
    ):
        raise ValueError("protocol checkpoint SHA-256 is invalid")
    return protocol


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit device or choose CUDA when available."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_locked_model(
    checkpoint_path: Path,
    protocol: Mapping[str, Any],
    device: torch.device,
) -> tuple[nn.Module, NormalizationStats, str]:
    """Verify and load the exact checkpoint locked by the protocol."""
    actual_digest = sha256_file(checkpoint_path)
    expected_digest = str(protocol["checkpoint_sha256"]).lower()
    if actual_digest != expected_digest:
        raise ValueError(
            "checkpoint SHA-256 does not match the locked inference protocol"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint.get("config", {})
    if tuple(config.get("classes", ())) != DIAGNOSTIC_CLASSES:
        raise ValueError("checkpoint class order does not match the model")
    if config.get("test_fold_used_for_model_selection") is not False:
        raise ValueError("checkpoint does not document an untouched test fold")

    normalization = NormalizationStats.from_mapping(
        checkpoint["normalization"]
    )
    model = BaselineECGCNN(dropout=float(config["dropout"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, normalization, actual_digest


def determine_review_action(
    uncertainty: float,
    cutoff: float,
    technical_quality_status: str,
) -> str:
    """Assign the same independent review reasons used in offline analysis."""
    uncertain = uncertainty > cutoff
    technical = technical_quality_status != "within_reference"
    if uncertain and technical:
        return "review_both"
    if uncertain:
        return "review_uncertain"
    if technical:
        return "review_technical"
    return "auto_result"


@torch.inference_mode()
def predict_waveform(
    waveform: np.ndarray,
    model: nn.Module,
    normalization: NormalizationStats,
    protocol: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Build a calibrated prediction and transparent review-routing report."""
    quality_features = extract_quality_features(
        waveform,
        sampling_frequency=EXPECTED_SAMPLING_FREQUENCY,
    )
    quality = score_quality_features(
        quality_features,
        protocol["quality_reference"],
    )

    normalized = (
        (np.asarray(waveform, dtype=np.float32) - normalization.mean)
        / normalization.std
    ).astype(np.float32, copy=False)
    inputs = torch.from_numpy(normalized[None, ...]).to(device)
    logits = model(inputs).float().cpu().numpy()
    if logits.shape != (1, len(DIAGNOSTIC_CLASSES)):
        raise ValueError(f"model returned unexpected logits shape {logits.shape}")

    temperature = float(protocol["calibration"]["temperature"])
    probabilities = sigmoid(logits / temperature)
    thresholds = {
        name: float(value)
        for name, value in protocol["thresholds"]["values"].items()
    }
    record_uncertainty, class_uncertainty = decision_boundary_uncertainty(
        probabilities,
        thresholds,
    )
    entropy = predictive_entropy(probabilities)
    uncertainty = float(record_uncertainty[0])
    cutoff = float(
        protocol["selective_prediction"]["uncertainty_cutoff"]
    )
    action = determine_review_action(
        uncertainty,
        cutoff,
        str(quality["technical_quality_status"]),
    )

    predictions = []
    for index, class_name in enumerate(DIAGNOSTIC_CLASSES):
        probability = float(probabilities[0, index])
        threshold = thresholds[class_name]
        predictions.append(
            {
                "class": class_name,
                "probability": probability,
                "threshold": threshold,
                "positive": probability >= threshold,
                "decision_uncertainty": float(
                    class_uncertainty[0, index]
                ),
                "predictive_entropy": float(entropy[0, index]),
            }
        )

    technical_review = (
        quality["technical_quality_status"] != "within_reference"
    )
    classification_withheld = uncertainty > cutoff
    return {
        "schema_version": protocol["schema_version"],
        "model_version": protocol["model_version"],
        "warning": RESEARCH_WARNING,
        "routing": {
            "action": action,
            "classification_withheld": classification_withheld,
            "technical_review_recommended": technical_review,
            "reason": (
                "결정 경계 근접도와 기술 품질 사유를 독립적으로 평가한 "
                "연구용 라우팅 결과입니다."
            ),
        },
        "uncertainty": {
            "decision_uncertainty": uncertainty,
            "validation_locked_cutoff": cutoff,
            "target_coverage": float(
                protocol["selective_prediction"]["target_coverage"]
            ),
            "is_uncertain": classification_withheld,
            "interpretation": (
                "단일 모델의 결정 경계 근접도이며 epistemic uncertainty나 "
                "임상 위험 확률이 아닙니다."
            ),
        },
        "technical_quality": {
            **quality,
            "features": quality_features,
            "interpretation": (
                "PTB-XL 학습 분포 기반 기술 검토 지표이며 자동 입력 폐기 "
                "기준이나 임상적 신호 품질 정답이 아닙니다."
            ),
        },
        "predictions": predictions,
    }


def _record_base(path: Path) -> Path:
    if path.suffix.lower() in {".hea", ".dat"}:
        return path.with_suffix("")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        type=Path,
        required=True,
        help="WFDB record base path, with or without .hea/.dat",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/baseline/best_model.pt"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/inference/prediction.json"),
    )
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_inference_protocol(args.protocol)
    device = resolve_device(args.device)
    model, normalization, checkpoint_digest = load_locked_model(
        args.checkpoint,
        protocol,
        device,
    )

    record = _record_base(args.record)
    waveform = load_waveform(record.name, record.parent)
    report = predict_waveform(
        waveform,
        model,
        normalization,
        protocol,
        device,
    )
    report["input"] = {
        "record_id": record.name,
        "format": "WFDB",
        "sampling_frequency_hz": EXPECTED_SAMPLING_FREQUENCY,
        "shape": list(waveform.shape),
        "lead_order": list(LEAD_NAMES),
        "amplitude_unit": "mV",
    }
    report["provenance"] = {
        "checkpoint_sha256": checkpoint_digest,
        "protocol_file": args.protocol.name,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"record={record.name} action={report['routing']['action']} "
        f"output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
