"""Train the ECG Guard baseline without accessing the held-out test fold."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ecg_guard.data.prepare_ptbxl import DIAGNOSTIC_CLASSES, prepare_metadata
from ecg_guard.data.ptbxl_dataset import (
    PTBXLDataset,
    NormalizationStats,
    compute_training_normalization_stats,
    select_modeling_metadata,
)
from ecg_guard.models import BaselineECGCNN


def set_reproducibility(seed: int) -> None:
    """Seed random number generators and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic NumPy/Python seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def multilabel_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Calculate per-class and macro metrics, tolerating small smoke samples."""
    if targets.shape != probabilities.shape:
        raise ValueError("targets and probabilities must have the same shape")
    if targets.shape[1] != len(DIAGNOSTIC_CLASSES):
        raise ValueError("unexpected number of target classes")

    per_class: dict[str, dict[str, float | None]] = {}
    aurocs: list[float] = []
    average_precisions: list[float] = []
    for index, label in enumerate(DIAGNOSTIC_CLASSES):
        class_targets = targets[:, index]
        if np.unique(class_targets).size < 2:
            per_class[label] = {"auroc": None, "average_precision": None}
            continue
        auroc = float(roc_auc_score(class_targets, probabilities[:, index]))
        average_precision = float(
            average_precision_score(class_targets, probabilities[:, index])
        )
        per_class[label] = {
            "auroc": auroc,
            "average_precision": average_precision,
        }
        aurocs.append(auroc)
        average_precisions.append(average_precision)

    return {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
        "macro_average_precision": (
            float(np.mean(average_precisions)) if average_precisions else None
        ),
        "per_class": per_class,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    *,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_records = 0
    for batch in loader:
        signals = batch["signal"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            logits = model(signals)
            loss = loss_function(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = signals.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_records += batch_size
    return total_loss / total_records


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    loss_function: nn.Module,
    device: torch.device,
    *,
    use_amp: bool,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_records = 0
    target_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []

    for batch in loader:
        signals = batch["signal"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            logits = model(signals)
            loss = loss_function(logits, targets)

        batch_size = signals.shape[0]
        total_loss += float(loss) * batch_size
        total_records += batch_size
        target_batches.append(targets.cpu().numpy())
        probability_batches.append(torch.sigmoid(logits).float().cpu().numpy())

    targets_array = np.concatenate(target_batches)
    probabilities_array = np.concatenate(probability_batches)
    return {
        "loss": total_loss / total_records,
        **multilabel_metrics(targets_array, probabilities_array),
    }


def build_loaders(
    train_metadata: Any,
    validation_metadata: Any,
    data_dir: Path,
    normalization: NormalizationStats,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> tuple[DataLoader[dict[str, Tensor]], DataLoader[dict[str, Tensor]]]:
    train_dataset = PTBXLDataset(train_metadata, data_dir, normalization)
    validation_dataset = PTBXLDataset(
        validation_metadata,
        data_dir,
        normalization,
    )
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common,
    )
    return train_loader, validation_loader


def write_json(path: Path, values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(values, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ptb-xl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="disable bfloat16 autocast on CUDA",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    positive_integer_fields = ("epochs", "batch_size", "patience")
    for field in positive_integer_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1)")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_arguments(args)
    set_reproducibility(args.seed)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    metadata = prepare_metadata(args.data_dir)
    train_metadata = select_modeling_metadata(
        metadata,
        "train",
        limit=args.train_limit,
        seed=args.seed,
    )
    validation_metadata = select_modeling_metadata(
        metadata,
        "validation",
        limit=args.validation_limit,
        seed=args.seed,
    )
    normalization = compute_training_normalization_stats(
        train_metadata,
        args.data_dir,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "dropout": args.dropout,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": str(device),
        "amp_dtype": "bfloat16" if use_amp else None,
        "train_limit": args.train_limit,
        "validation_limit": args.validation_limit,
        "train_records": len(train_metadata),
        "validation_records": len(validation_metadata),
        "test_fold_used_for_model_selection": False,
        "classes": list(DIAGNOSTIC_CLASSES),
    }
    write_json(args.output_dir / "config.json", config)
    write_json(
        args.output_dir / "normalization.json",
        normalization.to_dict(),
    )

    train_loader, validation_loader = build_loaders(
        train_metadata,
        validation_metadata,
        args.data_dir,
        normalization,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )
    model = BaselineECGCNN(dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(
        f"device={device} amp={use_amp} parameters={parameter_count} "
        f"train={len(train_metadata)} validation={len(validation_metadata)}"
    )
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    epochs_without_improvement = 0
    checkpoint_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
            use_amp=use_amp,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
            use_amp=use_amp,
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": validation_metrics,
        }
        history.append(epoch_metrics)
        write_json(args.output_dir / "history.json", history)

        macro_auroc = validation_metrics["macro_auroc"]
        score = float(macro_auroc) if macro_auroc is not None else -math.inf
        improved = score > best_score
        if improved:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": validation_metrics,
                    "normalization": normalization.to_dict(),
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"val_macro_auroc={macro_auroc} improved={improved}"
        )
        if epochs_without_improvement >= args.patience:
            print(f"early_stopping_epoch={epoch}")
            break

    if not checkpoint_path.is_file():
        raise RuntimeError("no checkpoint was saved")
    write_json(
        args.output_dir / "summary.json",
        {
            "best_validation_macro_auroc": best_score,
            "epochs_completed": len(history),
            "checkpoint": str(checkpoint_path),
            "test_fold_used_for_model_selection": False,
        },
    )
    print(f"checkpoint={checkpoint_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
