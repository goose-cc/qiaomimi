#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp34 model: frozen Exp33 TCN + amplitude-aware residual correction for a1.

Motivation
----------
Exp33 showed:
- gamma already benefits strongly from the TCN shape branch;
- a1 remains the main limiter of joint recovery.

This model therefore DOES NOT replace the successful TCN.  It keeps the
Exp33 TCN path exactly as the base predictor and adds one small branch that
only corrects the a1 logit using amplitude-preserving features.

Important properties
--------------------
1. forward(g) -> (a1, log10(gamma)) exactly matches the existing interface.
2. The correction head is zero-initialized.  Before residual training, the
   model reproduces the loaded Exp33 TCN predictions exactly.
3. The base TCN can be frozen, so gamma predictions remain unchanged while
   the new branch learns only an a1 correction.
4. a1 is still predicted directly by the neural network; no analytic/VarPro
   solution is used here.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from PairwiseInverseTCN import PairwiseInverseTCN


def _parse_hidden(hidden: Sequence[int]) -> tuple[int, ...]:
    vals = tuple(int(v) for v in hidden)
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("amplitude hidden sizes must be positive")
    return vals


class PairwiseInverseTCNA1Residual(PairwiseInverseTCN):
    """Exp33 TCN with an amplitude-aware residual branch dedicated to a1."""

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
        amplitude_bins: int = 16,
        amplitude_hidden: Sequence[int] = (96, 64),
        correction_limit: float = 2.0,
    ) -> None:
        super().__init__(
            input_points=input_points,
            channels=channels,
            dilations=dilations,
            kernel_size=kernel_size,
            dropout=dropout,
            head_hidden=head_hidden,
            second_min=second_min,
            second_max=second_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            use_coordinate_channel=use_coordinate_channel,
        )

        self.amplitude_bins = int(amplitude_bins)
        self.amplitude_hidden = _parse_hidden(amplitude_hidden)
        self.correction_limit = float(correction_limit)
        if self.amplitude_bins < 2:
            raise ValueError("amplitude_bins must be >= 2")
        if self.correction_limit <= 0:
            raise ValueError("correction_limit must be > 0")

        # A single scale is deliberately used in this branch.  Unlike
        # per-position standardization, a single global scale preserves
        # absolute amplitude differences across the g curve.
        self.register_buffer(
            "global_input_scale",
            torch.tensor(1.0, dtype=torch.float32),
        )

        # Features:
        #   adaptive avg pool of centered globally-scaled g       [bins]
        #   adaptive max pool of centered globally-scaled g       [bins]
        #   global statistics                                     [8]
        #   frozen base a1-unit and gamma-unit predictions         [2]
        feature_dim = 2 * self.amplitude_bins + 10

        layers: list[nn.Module] = []
        in_features = feature_dim
        for width in self.amplitude_hidden:
            layers.extend(
                [
                    nn.Linear(in_features, width),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_features = width
        self.amplitude_encoder = nn.Sequential(*layers)
        self.a1_logit_correction = nn.Linear(in_features, 1)

        # Crucial: Exp34 starts exactly from the Exp33 TCN prediction.
        nn.init.zeros_(self.a1_logit_correction.weight)
        nn.init.zeros_(self.a1_logit_correction.bias)

        self._base_frozen = False

    @torch.no_grad()
    def set_input_normalization(
        self,
        mean: torch.Tensor | list[float],
        scale: torch.Tensor | list[float],
        global_rms: float | torch.Tensor | None = None,
    ) -> None:
        super().set_input_normalization(mean, scale)
        if global_rms is None:
            # Fallback for compatibility.  The Exp34 trainer passes the
            # train-set global RMS explicitly.
            mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
            scale_t = torch.as_tensor(scale, dtype=torch.float32).reshape(-1)
            global_rms_t = torch.sqrt(torch.mean(mean_t.square() + scale_t.square()))
        else:
            global_rms_t = torch.as_tensor(global_rms, dtype=torch.float32).reshape(())
        if not torch.isfinite(global_rms_t) or float(global_rms_t) <= 0:
            raise ValueError("global_rms must be finite and > 0")
        self.global_input_scale.copy_(global_rms_t.to(self.global_input_scale.device))

    def freeze_base(self) -> None:
        """Freeze the Exp33 TCN path; only the new a1 residual branch trains."""
        for module in (self.stem, self.blocks, self.head):
            for p in module.parameters():
                p.requires_grad_(False)
        self._base_frozen = True
        self.enforce_frozen_base_eval()

    def unfreeze_base(self) -> None:
        for module in (self.stem, self.blocks, self.head):
            for p in module.parameters():
                p.requires_grad_(True)
        self._base_frozen = False

    def enforce_frozen_base_eval(self) -> None:
        """Disable dropout in the frozen base even while residual branch trains."""
        if self._base_frozen:
            self.stem.eval()
            self.blocks.eval()
            self.head.eval()

    def residual_parameters(self):
        for p in self.amplitude_encoder.parameters():
            if p.requires_grad:
                yield p
        for p in self.a1_logit_correction.parameters():
            if p.requires_grad:
                yield p

    def _coerce_gy(self, gy: torch.Tensor) -> torch.Tensor:
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
        return gy

    def _amplitude_features(
        self,
        gy2: torch.Tensor,
        base_a1_unit: torch.Tensor,
        gamma_unit: torch.Tensor,
    ) -> torch.Tensor:
        mean = self.input_mean.to(device=gy2.device, dtype=gy2.dtype)
        global_scale = self.global_input_scale.to(
            device=gy2.device, dtype=gy2.dtype
        ).clamp_min(torch.finfo(gy2.dtype).eps)

        centered = (gy2 - mean) / global_scale
        centered_1d = centered.unsqueeze(1)

        avg_pool = F.adaptive_avg_pool1d(
            centered_1d, self.amplitude_bins
        ).squeeze(1)
        max_pool = F.adaptive_max_pool1d(
            centered_1d, self.amplitude_bins
        ).squeeze(1)

        mean_c = torch.mean(centered, dim=1)
        std_c = torch.std(centered, dim=1, unbiased=False)
        rms_c = torch.sqrt(torch.mean(centered.square(), dim=1).clamp_min(1e-12))
        mean_abs_c = torch.mean(torch.abs(centered), dim=1)
        max_c = torch.amax(centered, dim=1)
        min_c = torch.amin(centered, dim=1)
        ptp_c = max_c - min_c

        raw_rms = torch.sqrt(torch.mean(gy2.square(), dim=1).clamp_min(1e-12))
        log_raw_rms = torch.log(raw_rms / global_scale + 1e-12)

        stats = torch.stack(
            [
                mean_c,
                std_c,
                rms_c,
                mean_abs_c,
                max_c,
                min_c,
                ptp_c,
                log_raw_rms,
                base_a1_unit.detach(),
                gamma_unit.detach(),
            ],
            dim=1,
        )
        return torch.cat([avg_pool, max_pool, stats], dim=1)

    def forward(self, gy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (a1, log10(gamma)), each shape [B]."""
        gy2 = self._coerce_gy(gy)

        # Exact Exp33 TCN path.
        x = self.stem(self._prepare_input(gy2))
        x = self.blocks(x)
        pooled = torch.cat([torch.mean(x, dim=2), torch.amax(x, dim=2)], dim=1)
        base_logits = self.head(pooled)

        base_a1_unit = torch.sigmoid(base_logits[:, 0])
        gamma_unit = torch.sigmoid(base_logits[:, 1])

        # New branch predicts only a bounded logit correction for a1.
        amp_features = self._amplitude_features(
            gy2, base_a1_unit=base_a1_unit, gamma_unit=gamma_unit
        )
        amp_latent = self.amplitude_encoder(amp_features)
        raw_delta = self.a1_logit_correction(amp_latent).squeeze(-1)
        limit = self.correction_limit
        delta_logit = limit * torch.tanh(raw_delta / limit)

        a1_unit = torch.sigmoid(base_logits[:, 0] + delta_logit)
        second_lo = self._second_min.to(a1_unit)
        second_hi = self._second_max.to(a1_unit)
        a1 = second_lo + a1_unit * (second_hi - second_lo)

        gamma_lo = self._log_gamma_min.to(gamma_unit)
        gamma_hi = self._log_gamma_max.to(gamma_unit)
        log_gamma = gamma_lo + gamma_unit * (gamma_hi - gamma_lo)
        return a1, log_gamma
