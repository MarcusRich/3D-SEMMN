"""Vision/audio encoders and lobe injection."""

from __future__ import annotations

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AudioEncoder1D(nn.Module):
    """1D Conv encoder over spectrogram time sequences."""

    def __init__(self, out_dim: int = 256, freq_bins: int = 112) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(freq_bins, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # Input expected `(B, 1, F, T)`; use `(B, F, T)` for Conv1d.
        x = spec.squeeze(1)
        return self.net(x)


class LobeInjector(nn.Module):
    """Project modality features into selected columns."""

    def __init__(self, in_dim: int, num_columns: int, state_dim: int, target_columns: torch.Tensor) -> None:
        super().__init__()
        self.num_columns = num_columns
        self.state_dim = state_dim
        self.register_buffer("target_columns", target_columns.long())
        out_dim = max(1, self.target_columns.numel()) * state_dim
        self.project = nn.Linear(in_dim, out_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b = features.shape[0]
        injected = features.new_zeros(b, self.num_columns, self.state_dim)
        projected = self.project(features).view(b, -1, self.state_dim)
        injected[:, self.target_columns, :] = projected[:, : self.target_columns.numel(), :]
        return injected
