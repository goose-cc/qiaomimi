#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TCN baseline for continuous a1 + gamma inverse regression.

The public interface intentionally matches PairwiseInverseMLP:
    forward(g) -> (a1, log10(gamma))
    set_input_normalization(mean, scale)

That allows Exp33 to change only the network backbone while keeping the
dataset, parameterization, regression loss and optimizer protocol unchanged.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


def _valid_group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(int(preferred), int(channels)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DilatedResidualBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
        *,
        kernel_size: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        padding = int(dilation) * (int(kernel_size) - 1) // 2
        groups = _valid_group_count(int(channels))

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        x = self.dropout(x)
        return self.activation(x + residual)


class PairwiseInverseTCN(nn.Module):
    """TCN regressor for g(q^2) -> [a1, log10(gamma)]."""

    def __init__(
        self,
        *,
        input_points: int = 100,
        channels: int = 64,
        dilations: Sequence[int] = (1, 2, 4, 8, 16),
        kernel_size: int = 3,
        dropout: float = 0.05,
        head_hidden: int = 128,
        second_min: float,
        second_max: float,
        gamma_min: float = 1e-3,
        gamma_max: float = 1.0,
        use_coordinate_channel: bool = True,
    ) -> None:
        super().__init__()
        self.input_points = int(input_points)
        self.second_min = float(second_min)
        self.second_max = float(second_max)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.use_coordinate_channel = bool(use_coordinate_channel)

        if self.input_points < 2:
            raise ValueError("input_points must be >= 2")
        if not (self.second_min < self.second_max):
            raise ValueError("require second_min < second_max")
        if not (0.0 < self.gamma_min < self.gamma_max):
            raise ValueError("require 0 < gamma_min < gamma_max")
        if channels <= 0 or head_hidden <= 0:
            raise ValueError("channels/head_hidden must be positive")
        if not dilations or any(int(d) <= 0 for d in dilations):
            raise ValueError("dilations must contain positive integers")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")

        # Use exactly the same fixed train-set, per-position standardization
        # as the MLP baseline.
        self.register_buffer(
            "input_mean",
            torch.zeros(self.input_points, dtype=torch.float32),
        )
        self.register_buffer(
            "input_scale",
            torch.ones(self.input_points, dtype=torch.float32),
        )
        self.register_buffer(
            "_second_min",
            torch.tensor(self.second_min, dtype=torch.float32),
        )
        self.register_buffer(
            "_second_max",
            torch.tensor(self.second_max, dtype=torch.float32),
        )
        self.register_buffer(
            "_log_gamma_min",
            torch.tensor(math.log10(self.gamma_min), dtype=torch.float32),
        )
        self.register_buffer(
            "_log_gamma_max",
            torch.tensor(math.log10(self.gamma_max), dtype=torch.float32),
        )

        # Convolution is translation-equivariant.  A normalized index channel
        # gives it absolute q-position information; it adds no target/physics
        # information and is deterministic for every sample.
        coord = torch.linspace(-1.0, 1.0, self.input_points, dtype=torch.float32)
        self.register_buffer("coordinate", coord.view(1, 1, -1))

        in_channels = 2 if self.use_coordinate_channel else 1
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, int(channels), kernel_size=5, padding=2),
            nn.GroupNorm(_valid_group_count(int(channels)), int(channels)),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                DilatedResidualBlock1D(
                    int(channels),
                    int(d),
                    kernel_size=int(kernel_size),
                    dropout=float(dropout),
                )
                for d in dilations
            ]
        )

        # Average and max pooling summarize broad and local shape information.
        self.head = nn.Sequential(
            nn.Linear(2 * int(channels), int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden), int(head_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden), 2),
        )

        self.receptive_field_estimate = int(
            5 + 2 * (int(kernel_size) - 1) * sum(int(d) for d in dilations)
        )

    @torch.no_grad()
    def set_input_normalization(
        self,
        mean: torch.Tensor | list[float],
        scale: torch.Tensor | list[float],
    ) -> None:
        mean_t = torch.as_tensor(mean, dtype=self.input_mean.dtype).reshape(-1)
        scale_t = torch.as_tensor(scale, dtype=self.input_scale.dtype).reshape(-1)
        if mean_t.numel() != self.input_points or scale_t.numel() != self.input_points:
            raise ValueError("mean/scale length must equal input_points")
        if not torch.isfinite(mean_t).all() or not torch.isfinite(scale_t).all():
            raise ValueError("input normalization contains non-finite values")
        if torch.any(scale_t <= 0):
            raise ValueError("all input scales must be > 0")
        self.input_mean.copy_(mean_t.to(self.input_mean.device))
        self.input_scale.copy_(scale_t.to(self.input_scale.device))

    def _prepare_input(self, gy: torch.Tensor) -> torch.Tensor:
        if gy.ndim == 3:
            if gy.shape[1] != 1:
                raise ValueError(f"expected [B,1,N], got {tuple(gy.shape)}")
            gy = gy[:, 0, :]
        if gy.ndim != 2:
            raise ValueError(f"expected [B,N] or [B,1,N], got {tuple(gy.shape)}")
        if gy.shape[1] != self.input_points:
            raise ValueError(
                f"expected {self.input_points} input points, got {gy.shape[1]}"
            )
        mean = self.input_mean.to(device=gy.device, dtype=gy.dtype)
        scale = self.input_scale.to(device=gy.device, dtype=gy.dtype)
        x = ((gy - mean) / scale).unsqueeze(1)
        if self.use_coordinate_channel:
            coord = self.coordinate.to(device=x.device, dtype=x.dtype)
            coord = coord.expand(x.shape[0], -1, -1)
            x = torch.cat([x, coord], dim=1)
        return x

    def forward(self, gy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (a1, log10(gamma)), each shape [B]."""
        x = self.stem(self._prepare_input(gy))
        x = self.blocks(x)
        pooled = torch.cat([torch.mean(x, dim=2), torch.amax(x, dim=2)], dim=1)
        logits = self.head(pooled)

        second_unit = torch.sigmoid(logits[:, 0])
        second_lo = self._second_min.to(second_unit)
        second_hi = self._second_max.to(second_unit)
        second = second_lo + second_unit * (second_hi - second_lo)

        gamma_unit = torch.sigmoid(logits[:, 1])
        gamma_lo = self._log_gamma_min.to(gamma_unit)
        gamma_hi = self._log_gamma_max.to(gamma_unit)
        log_gamma = gamma_lo + gamma_unit * (gamma_hi - gamma_lo)
        return second, log_gamma

    def predict_physical(self, gy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        second, log_gamma = self.forward(gy)
        return second, torch.pow(10.0, log_gamma)
