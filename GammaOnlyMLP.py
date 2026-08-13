#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple gamma-only MLP for Exp19A.

Input:
    g(q^2) sampled on a fixed grid.

Output:
    log10(gamma), constrained to the physical gamma interval.

Normalization:
    A fixed affine normalization is estimated ONCE from the training set:
        x_j = (g_j - mean_train_j) / std_train_j
    This is not per-sample RMS normalization. It is a fixed train-set transform,
    so sample-to-sample amplitude information is preserved.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


class GammaOnlyMLP(nn.Module):
    def __init__(
        self,
        *,
        input_points: int = 100,
        hidden_sizes: Sequence[int] = (256, 256, 128),
        dropout: float = 0.05,
        gamma_min: float = 1e-3,
        gamma_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_points = int(input_points)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)

        if self.input_points < 2:
            raise ValueError("input_points must be >= 2")
        if not (0.0 < self.gamma_min < self.gamma_max):
            raise ValueError("require 0 < gamma_min < gamma_max")
        if not hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty")

        self.register_buffer(
            "input_mean",
            torch.zeros(self.input_points, dtype=torch.float32),
        )
        self.register_buffer(
            "input_scale",
            torch.ones(self.input_points, dtype=torch.float32),
        )
        self.register_buffer(
            "log_gamma_min",
            torch.tensor(math.log10(self.gamma_min), dtype=torch.float32),
        )
        self.register_buffer(
            "log_gamma_max",
            torch.tensor(math.log10(self.gamma_max), dtype=torch.float32),
        )

        layers: list[nn.Module] = []
        in_features = self.input_points
        for width in hidden_sizes:
            width = int(width)
            if width <= 0:
                raise ValueError("all hidden sizes must be positive")
            layers.extend(
                [
                    nn.Linear(in_features, width),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_features = width

        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(in_features, 1)

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
        return (gy - mean) / scale

    def forward(self, gy: torch.Tensor) -> torch.Tensor:
        """Return predicted log10(gamma), shape [B]."""
        x = self._prepare_input(gy)
        raw = self.output(self.backbone(x)).squeeze(-1)
        unit = torch.sigmoid(raw)
        lo = self.log_gamma_min.to(device=unit.device, dtype=unit.dtype)
        hi = self.log_gamma_max.to(device=unit.device, dtype=unit.dtype)
        return lo + unit * (hi - lo)

    def predict_gamma(self, gy: torch.Tensor) -> torch.Tensor:
        return torch.pow(10.0, self.forward(gy))
