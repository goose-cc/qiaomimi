#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared MLP for Exp20 pairwise curriculum experiments.

Modes
-----
a1gamma:
    g(q^2) -> [a1, log10(gamma)]

mgamma:
    g(q^2) -> [m, log10(gamma)]

The backbone intentionally stays close to Exp19B so that the main controlled
change is adding exactly one extra unknown physical parameter.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


class PairwiseInverseMLP(nn.Module):
    def __init__(
        self,
        *,
        input_points: int = 100,
        hidden_sizes: Sequence[int] = (256, 256, 128),
        dropout: float = 0.05,
        second_min: float,
        second_max: float,
        gamma_min: float = 1e-3,
        gamma_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_points = int(input_points)
        self.second_min = float(second_min)
        self.second_max = float(second_max)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)

        if self.input_points < 2:
            raise ValueError("input_points must be >= 2")
        if not (self.second_min < self.second_max):
            raise ValueError("require second_min < second_max")
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
        self.second_head = nn.Linear(in_features, 1)
        self.gamma_head = nn.Linear(in_features, 1)

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

    def forward(self, gy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (second_parameter, log10(gamma)), each shape [B]."""
        x = self.backbone(self._prepare_input(gy))

        second_unit = torch.sigmoid(self.second_head(x).squeeze(-1))
        second_lo = self._second_min.to(second_unit)
        second_hi = self._second_max.to(second_unit)
        second = second_lo + second_unit * (second_hi - second_lo)

        gamma_unit = torch.sigmoid(self.gamma_head(x).squeeze(-1))
        gamma_lo = self._log_gamma_min.to(gamma_unit)
        gamma_hi = self._log_gamma_max.to(gamma_unit)
        log_gamma = gamma_lo + gamma_unit * (gamma_hi - gamma_lo)
        return second, log_gamma

    def second_to_unit(self, value: torch.Tensor) -> torch.Tensor:
        lo = self._second_min.to(value)
        hi = self._second_max.to(value)
        return (value - lo) / (hi - lo)

    def predict_physical(self, gy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        second, log_gamma = self.forward(gy)
        return second, torch.pow(10.0, log_gamma)
