"""Compact residual 1D CNN baseline for twelve-lead ECG classification."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ecg_guard.data.prepare_ptbxl import DIAGNOSTIC_CLASSES, LEAD_NAMES


class ResidualBlock1D(nn.Module):
    """Two-convolution residual block with optional downsampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.convolutions(inputs) + self.shortcut(inputs))


class BaselineECGCNN(nn.Module):
    """Predict five PTB-XL diagnostic superclasses from a 10-second ECG."""

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(
                len(LEAD_NAMES),
                64,
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 256, stride=2),
            ResidualBlock1D(256, 256, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, len(DIAGNOSTIC_CLASSES)),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[1] != len(LEAD_NAMES):
            raise ValueError(
                "expected inputs shaped (batch, 12, samples), "
                f"got {tuple(inputs.shape)}"
            )
        return self.classifier(self.features(inputs))
