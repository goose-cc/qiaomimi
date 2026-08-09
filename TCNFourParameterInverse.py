#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp17 model: TCN nuisance-parameter regressor.

The network intentionally does NOT predict gamma.

Input:
    100-point g(q^2)

Output:
    [a1, a2, a3, m]

Design choices:
1. Global input scaling instead of per-sample RMS normalization, preserving
   sample-to-sample amplitude information (Exp11).
2. q^2 coordinate is provided as a second input channel.
3. Dilated residual Conv1d blocks give a receptive field covering the full
   100-point observation.
4. Gamma is inferred later by the exact physical forward model, not by this
   network.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from mc_parametric import parameter_bounds


NUISANCE_NAMES = ("a1", "a2", "a3", "m")
NUISANCE_INDICES = (0, 1, 2, 3)


def _valid_group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
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
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        if dilation <= 0:
            raise ValueError("dilation must be positive")

        padding = dilation * (kernel_size - 1) // 2
        groups = _valid_group_count(channels)

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout(x)
        return self.activation(x + residual)


class TCNFourParameterInverse1D(nn.Module):
    """Predict [a1,a2,a3,m] while leaving gamma to physics profiling."""

    def __init__(
        self,
        *,
        input_length: int = 100,
        channels: int = 64,
        dilations: Sequence[int] = (1, 2, 4, 8, 16),
        kernel_size: int = 3,
        dropout: float = 0.05,
        head_hidden: int = 128,
        q2_min: float = -100.0,
        q2_max: float = -6.0,
        global_input_scale: float = 0.0175,
        rms_eps: float = 1e-8,
    ):
        super().__init__()

        self.input_length = int(input_length)
        self.global_input_scale = float(global_input_scale)
        self.rms_eps = float(rms_eps)

        if self.input_length <= 1:
            raise ValueError("input_length must be > 1")
        if self.global_input_scale <= 0:
            raise ValueError("global_input_scale must be > 0")
        if self.rms_eps <= 0:
            raise ValueError("rms_eps must be > 0")

        # Input channels:
        #   0: globally-scaled g
        #   1: normalized q^2 coordinate
        self.stem = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=5, padding=2),
            nn.GroupNorm(_valid_group_count(channels), channels),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[
                DilatedResidualBlock1D(
                    channels,
                    int(d),
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for d in dilations
            ]
        )

        # Average pooling preserves global shape; max pooling can preserve local
        # deviations. log(RMS) is retained as an explicit scalar feature.
        self.head = nn.Sequential(
            nn.Linear(2 * channels + 1, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 4),
        )

        q = torch.linspace(float(q2_min), float(q2_max), self.input_length)
        q_norm = 2.0 * (q - float(q2_min)) / max(
            float(q2_max) - float(q2_min), 1e-12
        ) - 1.0
        self.register_buffer("q_norm", q_norm.view(1, 1, -1))

        lower5, upper5 = parameter_bounds()
        idx = torch.tensor(NUISANCE_INDICES, dtype=torch.long)
        self.register_buffer("parameter_lower", lower5.index_select(0, idx))
        self.register_buffer("parameter_upper", upper5.index_select(0, idx))

        # Effective receptive field for two k=3 convolutions per residual block:
        # RF ~= 1 + 4*sum(dilations), plus the 5-point stem.
        self.receptive_field_estimate = int(
            5 + 2 * (kernel_size - 1) * sum(int(d) for d in dilations)
        )

    def predict_nuisance(self, gy: torch.Tensor) -> torch.Tensor:
        if gy.ndim != 3 or gy.shape[1] != 1:
            raise ValueError(
                f"Expected gy shape [B,1,N], got {tuple(gy.shape)}"
            )
        if gy.shape[2] != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, got {gy.shape[2]}"
            )

        batch = gy.shape[0]
        rms = torch.sqrt(
            torch.mean(gy.square(), dim=2).clamp_min(self.rms_eps**2)
        )  # [B,1]

        g_scaled = gy / self.global_input_scale
        q = self.q_norm.expand(batch, -1, -1).to(
            dtype=g_scaled.dtype, device=g_scaled.device
        )
        x = torch.cat([g_scaled, q], dim=1)

        x = self.stem(x)
        x = self.blocks(x)

        avg = torch.mean(x, dim=2)
        mx = torch.amax(x, dim=2)
        log_rms = torch.log(rms.clamp_min(self.rms_eps))

        logits = self.head(torch.cat([avg, mx, log_rms], dim=1))
        unit = torch.sigmoid(logits)

        lower = self.parameter_lower.to(
            dtype=unit.dtype, device=unit.device
        )
        upper = self.parameter_upper.to(
            dtype=unit.dtype, device=unit.device
        )
        return lower + unit * (upper - lower)

    def forward(self, gy: torch.Tensor) -> torch.Tensor:
        return self.predict_nuisance(gy)
