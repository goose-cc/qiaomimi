#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp30 a1+gamma TCN inverse model.

This adapts the existing TCNFourParameterInverse design to the Exp30
two-parameter inverse problem:

    g(q^2) -> [a1, gamma]

Key points:
- Dilated residual Conv1d TCN backbone.
- Supports 1000-point Exp30 observations.
- Uses the same train-set input normalization supplied by
  training_input_normalization(...), so the MLP/TCN comparison keeps the
  data normalization pipeline unchanged.
- Keeps q^2 as an optional second channel, following the existing TCN design.
- Bounds a1 linearly and gamma logarithmically inside their physical ranges.
- Can mimic the output style of the existing PairwiseInverseMLP
  ("tensor", "tuple", or "dict") so existing Exp20/Exp26 helper functions
  can be reused without modification.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


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
            channels, channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation
        )
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation
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


class PairwiseInverseTCN(nn.Module):
    """Predict Exp30 [a1, gamma] from g(q^2)."""

    def __init__(
        self,
        *,
        input_points: int,
        channels: int = 64,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.05,
        head_hidden: int = 128,
        second_min: float = 0.05,
        second_max: float = 0.20,
        gamma_min: float = 0.01,
        gamma_max: float = 1.0,
        q2_min: float = -100.0,
        q2_max: float = -6.0,
        use_q2_channel: bool = True,
        rms_eps: float = 1e-8,
    ):
        super().__init__()

        self.input_points = int(input_points)
        self.channels = int(channels)
        self.dilations = tuple(int(d) for d in dilations)
        self.kernel_size = int(kernel_size)
        self.head_hidden = int(head_hidden)
        self.use_q2_channel = bool(use_q2_channel)
        self.rms_eps = float(rms_eps)

        self.second_min = float(second_min)
        self.second_max = float(second_max)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)

        if self.input_points <= 1:
            raise ValueError("input_points must be > 1")
        if not (self.second_max > self.second_min):
            raise ValueError("second_max must be > second_min")
        if not (self.gamma_max > self.gamma_min > 0):
            raise ValueError("Require 0 < gamma_min < gamma_max")
        if not self.dilations or any(d <= 0 for d in self.dilations):
            raise ValueError("dilations must contain positive integers")

        in_channels = 2 if self.use_q2_channel else 1
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2),
            nn.GroupNorm(_valid_group_count(channels), channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                DilatedResidualBlock1D(
                    channels, d, kernel_size=kernel_size, dropout=dropout
                )
                for d in self.dilations
            ]
        )

        # avg-pool + max-pool + raw log-RMS scalar
        self.head = nn.Sequential(
            nn.Linear(2 * channels + 1, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 2),
        )

        q = torch.linspace(float(q2_min), float(q2_max), self.input_points)
        q_norm = 2.0 * (q - float(q2_min)) / max(
            float(q2_max) - float(q2_min), 1e-12
        ) - 1.0
        self.register_buffer("q_norm", q_norm.view(1, 1, -1))

        # These are replaced by set_input_normalization().
        self.register_buffer("input_mean", torch.zeros(1, self.input_points))
        self.register_buffer("input_scale", torch.ones(1, self.input_points))

        self._return_style = "tensor"
        self._dict_keys = ("second", "gamma")

        # Two k=3 convolutions per residual block plus the 5-point stem.
        self.receptive_field_estimate = int(
            5 + 2 * (self.kernel_size - 1) * sum(self.dilations)
        )

    def set_input_normalization(self, mean, scale) -> None:
        """Install the same train-set normalization used by the MLP pipeline."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        mean_t = torch.as_tensor(mean, dtype=dtype, device=device).reshape(-1)
        scale_t = torch.as_tensor(scale, dtype=dtype, device=device).reshape(-1)

        if mean_t.numel() == 1:
            mean_t = mean_t.expand(self.input_points)
        if scale_t.numel() == 1:
            scale_t = scale_t.expand(self.input_points)

        if mean_t.numel() != self.input_points:
            raise ValueError(
                f"Normalization mean has {mean_t.numel()} values; "
                f"expected 1 or {self.input_points}"
            )
        if scale_t.numel() != self.input_points:
            raise ValueError(
                f"Normalization scale has {scale_t.numel()} values; "
                f"expected 1 or {self.input_points}"
            )

        scale_t = scale_t.clamp_min(1e-8)
        self.input_mean = mean_t.view(1, -1)
        self.input_scale = scale_t.view(1, -1)

    def set_return_style(self, style: str, dict_keys=None) -> None:
        """Match the existing PairwiseInverseMLP forward-output convention."""
        style = str(style).strip().lower()
        if style not in {"tensor", "tuple", "dict"}:
            raise ValueError("return style must be tensor, tuple, or dict")
        self._return_style = style
        if dict_keys is not None:
            keys = tuple(dict_keys)
            if len(keys) != 2:
                raise ValueError("dict_keys must contain exactly two keys")
            self._dict_keys = keys

    def _as_bn(self, gy: torch.Tensor) -> torch.Tensor:
        if gy.ndim == 3:
            if gy.shape[1] != 1:
                raise ValueError(
                    f"Expected gy [B,1,N] or [B,N], got {tuple(gy.shape)}"
                )
            gy = gy[:, 0, :]
        if gy.ndim != 2:
            raise ValueError(
                f"Expected gy [B,N] or [B,1,N], got {tuple(gy.shape)}"
            )
        if gy.shape[1] != self.input_points:
            raise ValueError(
                f"Expected input length {self.input_points}, got {gy.shape[1]}"
            )
        return gy

    def predict_parameters(self, gy: torch.Tensor) -> torch.Tensor:
        gy = self._as_bn(gy)
        batch = gy.shape[0]

        # Keep an explicit amplitude feature from the raw observation.
        raw_rms = torch.sqrt(
            torch.mean(gy.square(), dim=1, keepdim=True)
            .clamp_min(self.rms_eps ** 2)
        )

        mean = self.input_mean.to(dtype=gy.dtype, device=gy.device)
        scale = self.input_scale.to(dtype=gy.dtype, device=gy.device)
        g_norm = (gy - mean) / scale
        x = g_norm.unsqueeze(1)

        if self.use_q2_channel:
            q = self.q_norm.expand(batch, -1, -1).to(
                dtype=x.dtype, device=x.device
            )
            x = torch.cat([x, q], dim=1)

        x = self.stem(x)
        x = self.blocks(x)

        avg = torch.mean(x, dim=2)
        mx = torch.amax(x, dim=2)
        log_rms = torch.log(raw_rms.clamp_min(self.rms_eps))
        logits = self.head(torch.cat([avg, mx, log_rms], dim=1))

        second_unit = torch.sigmoid(logits[:, 0])
        gamma_unit = torch.sigmoid(logits[:, 1])

        second = self.second_min + second_unit * (
            self.second_max - self.second_min
        )

        log_gmin = math.log(self.gamma_min)
        log_gmax = math.log(self.gamma_max)
        gamma = torch.exp(log_gmin + gamma_unit * (log_gmax - log_gmin))

        return torch.stack([second, gamma], dim=1)

    def forward(self, gy: torch.Tensor):
        out = self.predict_parameters(gy)
        if self._return_style == "tuple":
            return out[:, 0], out[:, 1]
        if self._return_style == "dict":
            return {
                self._dict_keys[0]: out[:, 0],
                self._dict_keys[1]: out[:, 1],
            }
        return out
