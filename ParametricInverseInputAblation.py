#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp11 model: input-normalization ablation for Exp7 parameter regression.

The original Exp7 ParametricInverseTransformer1D performs:
    g -> g / per-sample RMS
and only restores absolute-scale information through one scalar log(RMS).

That preprocessing can be harmful when a1 itself is an amplitude parameter.
This class keeps the architecture/head identical while allowing:

    token_normalization="rms"
        Exact Exp7-style token preprocessing:
        tokens = g / RMS(g)

    token_normalization="global"
        Preserve between-sample amplitude:
        tokens = g / fixed_global_scale

In BOTH modes log(RMS) is still appended to the pooled encoder feature, so
the only intended ablation is whether token-level amplitude is removed.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ParametricInverseTransformerInputAblation1D(nn.Module):
    def __init__(
        self,
        input_length=100,
        output_length=1000,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        y_min=-100.0,
        y_max=-6.0,
        x_min=0.1764,
        x_max=6.0,
        shift=400.0,
        data_scale=160000.0,
        rms_eps=1e-8,
        token_normalization="global",
        global_input_scale=0.0175,
    ):
        super().__init__()
        from mc_parametric import parameter_bounds

        self.input_length = int(input_length)
        self.output_length = int(output_length)
        self.shift = float(shift)
        self.data_scale = float(data_scale)
        self.rms_eps = float(rms_eps)
        self.token_normalization = str(token_normalization).lower()
        self.global_input_scale = float(global_input_scale)

        if self.input_length <= 1 or self.output_length <= 1:
            raise ValueError("input_length and output_length must be > 1")
        if self.rms_eps <= 0:
            raise ValueError("rms_eps must be positive")
        if self.token_normalization not in ("rms", "global"):
            raise ValueError("token_normalization must be 'rms' or 'global'")
        if self.global_input_scale <= 0:
            raise ValueError("global_input_scale must be > 0")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.g_value_embed = nn.Linear(1, d_model)
        self.y_pos_embed = nn.Linear(1, d_model)
        self.src_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_encoder_layers,
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        # Keep the same d_model+1 head as Exp7 in BOTH normalization modes.
        # The extra scalar is always log(RMS).
        self.parameter_head = nn.Sequential(
            nn.Linear(d_model + 1, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 5),
        )

        y_grid = torch.linspace(-1.0, 1.0, self.input_length)
        s_grid = torch.linspace(float(x_min), float(x_max), self.output_length)
        lower, upper = parameter_bounds()

        self.register_buffer(
            "y_grid",
            y_grid.view(1, self.input_length, 1),
        )
        self.register_buffer("s_grid", s_grid)
        self.register_buffer("parameter_lower", lower)
        self.register_buffer("parameter_upper", upper)

    def predict_parameters(self, gy):
        if gy.dim() != 3 or gy.shape[1] != 1:
            raise ValueError(
                f"Expected gy shape [B, 1, Ny], but got {tuple(gy.shape)}"
            )
        if gy.shape[2] != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, got {gy.shape[2]}"
            )

        batch = gy.shape[0]
        rms = torch.sqrt(
            torch.mean(
                gy.square(),
                dim=2,
                keepdim=True,
            ).clamp_min(self.rms_eps**2)
        )

        if self.token_normalization == "rms":
            # Original Exp7 behavior: removes per-sample amplitude from all tokens.
            gy_for_tokens = gy / rms
        else:
            # New ablation: one common scale for every sample.
            # Relative amplitude between samples is therefore preserved.
            gy_for_tokens = gy / self.global_input_scale

        tokens = gy_for_tokens.transpose(1, 2)
        pos = self.y_grid.expand(batch, -1, -1).to(
            dtype=tokens.dtype,
            device=tokens.device,
        )

        src = self.src_norm(
            self.g_value_embed(tokens)
            + self.y_pos_embed(pos)
        )

        encoded = self.encoder_norm(self.encoder(src))
        pooled = encoded.mean(dim=1)

        # Keep this feature in BOTH modes for a clean ablation.
        log_rms = torch.log(
            rms[:, 0, 0].clamp_min(self.rms_eps)
        ).unsqueeze(1)

        logits = self.parameter_head(
            torch.cat([pooled, log_rms], dim=1)
        )

        unit = torch.sigmoid(logits)

        lower = self.parameter_lower.to(
            dtype=unit.dtype,
            device=unit.device,
        )
        upper = self.parameter_upper.to(
            dtype=unit.dtype,
            device=unit.device,
        )

        return lower + unit * (upper - lower)

    def forward(self, gy):
        return self.predict_parameters(gy)
