#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Coarse-to-fine inverse MLP for Exp24.

The model predicts:
1) continuous a1;
2) a coarse gamma interval (classification);
3) a within-interval coordinate (regression).

At inference, gamma is reconstructed *inside the predicted interval*, so when
the coarse class is correct the model cannot jump to a completely unrelated
gamma branch.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class CoarseToFineA1GammaMLP(nn.Module):
    def __init__(
        self,
        input_points=1000,
        hidden_sizes=(256, 256, 128),
        dropout=0.05,
        a1_min=0.05,
        a1_max=0.2,
        gamma_edges=(0.01, 0.0316227766, 0.1, 0.316227766, 1.0),
    ):
        super().__init__()
        self.input_points = int(input_points)
        self.a1_min = float(a1_min)
        self.a1_max = float(a1_max)
        self.gamma_edges_list = tuple(float(x) for x in gamma_edges)
        if len(self.gamma_edges_list) < 3:
            raise ValueError("gamma_edges must contain at least 3 values")
        if any(x <= 0 for x in self.gamma_edges_list):
            raise ValueError("gamma edges must be positive")
        if any(self.gamma_edges_list[i] >= self.gamma_edges_list[i + 1]
               for i in range(len(self.gamma_edges_list) - 1)):
            raise ValueError("gamma edges must be strictly increasing")
        if not self.a1_min < self.a1_max:
            raise ValueError("require a1_min < a1_max")

        self.n_classes = len(self.gamma_edges_list) - 1

        self.register_buffer("input_mean", torch.zeros(self.input_points))
        self.register_buffer("input_scale", torch.ones(self.input_points))
        self.register_buffer("_a1_min", torch.tensor(self.a1_min, dtype=torch.float32))
        self.register_buffer("_a1_max", torch.tensor(self.a1_max, dtype=torch.float32))
        self.register_buffer(
            "_log_gamma_edges",
            torch.log10(torch.tensor(self.gamma_edges_list, dtype=torch.float32)),
        )

        layers = []
        width_in = self.input_points
        for width in hidden_sizes:
            width = int(width)
            layers.extend([
                nn.Linear(width_in, width),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ])
            width_in = width
        self.backbone = nn.Sequential(*layers)
        self.a1_head = nn.Linear(width_in, 1)
        self.gamma_class_head = nn.Linear(width_in, self.n_classes)
        self.gamma_residual_head = nn.Linear(width_in, self.n_classes)

    @torch.no_grad()
    def set_input_normalization(self, mean, scale):
        mean = torch.as_tensor(mean, dtype=self.input_mean.dtype).reshape(-1)
        scale = torch.as_tensor(scale, dtype=self.input_scale.dtype).reshape(-1)
        if mean.numel() != self.input_points or scale.numel() != self.input_points:
            raise ValueError("normalization length mismatch")
        if torch.any(scale <= 0):
            raise ValueError("input scale must be positive")
        self.input_mean.copy_(mean.to(self.input_mean.device))
        self.input_scale.copy_(scale.to(self.input_scale.device))

    def _prepare(self, gy):
        if gy.ndim == 3:
            gy = gy[:, 0, :]
        if gy.ndim != 2 or gy.shape[1] != self.input_points:
            raise ValueError("expected [B,%d] input" % self.input_points)
        return (gy - self.input_mean.to(gy)) / self.input_scale.to(gy)

    def forward(self, gy):
        h = self.backbone(self._prepare(gy))
        a1_unit = torch.sigmoid(self.a1_head(h).squeeze(-1))
        a1 = self._a1_min.to(a1_unit) + a1_unit * (
            self._a1_max.to(a1_unit) - self._a1_min.to(a1_unit)
        )
        logits = self.gamma_class_head(h)
        residual_unit_all = torch.sigmoid(self.gamma_residual_head(h))
        return a1, logits, residual_unit_all

    def gamma_class(self, gamma):
        """Return integer interval index in [0, n_classes-1]."""
        gamma = gamma.reshape(-1)
        internal = torch.pow(10.0, self._log_gamma_edges[1:-1]).to(gamma)
        # Exact boundaries go to the interval on their right.
        return torch.sum(
            gamma[:, None] >= internal[None, :], dim=1
        ).to(torch.long)

    def residual_target(self, gamma, class_index):
        log_g = torch.log10(gamma.reshape(-1))
        edges = self._log_gamma_edges.to(log_g)
        lo = edges[class_index]
        hi = edges[class_index + 1]
        return torch.clamp((log_g - lo) / (hi - lo), 0.0, 1.0)

    def gamma_from_class_residual(self, class_index, residual_unit):
        edges = self._log_gamma_edges.to(residual_unit)
        lo = edges[class_index]
        hi = edges[class_index + 1]
        log_g = lo + torch.clamp(residual_unit, 0.0, 1.0) * (hi - lo)
        return torch.pow(10.0, log_g)

    def predict_physical(self, gy):
        a1, logits, residual_all = self.forward(gy)
        pred_class = torch.argmax(logits, dim=1)
        pred_u = residual_all.gather(1, pred_class[:, None]).squeeze(1)
        gamma = self.gamma_from_class_residual(pred_class, pred_u)
        return a1, gamma, pred_class, logits, residual_all
