#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Float64 mirror of the project forward observation for Exp36 diagnostics.

The project physics is not changed.  This helper mirrors
mc_physics.scaled_forward_observation_numpy but deliberately keeps float64
outputs so very small alias residuals can be measured without float32
subtraction/cancellation artifacts.
"""
from __future__ import annotations

import numpy as np

from mc_physics import legendre_rule, output_grids_numpy
from mc_pool_config import DEFAULT_PHYSICS, PhysicsConfig


def _as_parameter_array(parameters: np.ndarray) -> np.ndarray:
    p = np.asarray(parameters, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 5:
        raise ValueError(f"parameters must have shape [N,5], got {p.shape}")
    return p


def scaled_forward_observation_float64(
    parameters: np.ndarray,
    integration_points: int = 128,
    config: PhysicsConfig = DEFAULT_PHYSICS,
    q_block_size: int = 25,
) -> np.ndarray:
    """Numerically equivalent forward observation, retained in float64."""
    p = _as_parameter_array(parameters)
    _, q2 = output_grids_numpy(config)
    q2 = np.asarray(q2, dtype=np.float64)
    x, w = legendre_rule(int(integration_points))
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    s_mid = 0.5 * (config.s_max + config.s_min)
    s_half = 0.5 * (config.s_max - config.s_min)
    s_fixed = s_mid + s_half * x
    s_weights = s_half * w
    background = (
        p[:, 1:2] * s_fixed.reshape(1, -1) + p[:, 2:3]
    ) / (s_fixed.reshape(1, -1) + config.shift) ** 2
    background_kernel = (
        s_weights.reshape(-1, 1)
        / (s_fixed.reshape(-1, 1) - q2.reshape(1, -1))
    )
    result = background @ background_kernel

    a1 = p[:, 0:1]
    mass = p[:, 3:4]
    width = p[:, 3:4] * p[:, 4:5]
    z0 = np.arctan((config.s_min - mass) / width)
    z1 = np.arctan((config.s_max - mass) / width)
    z_mid = 0.5 * (z0 + z1)
    z_half = 0.5 * (z1 - z0)
    z = z_mid + z_half * x.reshape(1, -1)
    s_res = mass + width * np.tan(z)
    coefficient = (
        (a1 / np.pi)
        * z_half
        * w.reshape(1, -1)
        / (s_res + config.shift) ** 2
    )

    for start in range(0, len(q2), int(q_block_size)):
        stop = min(start + int(q_block_size), len(q2))
        denominator = s_res[:, :, None] - q2[None, None, start:stop]
        result[:, start:stop] += np.sum(
            coefficient[:, :, None] / denominator, axis=1
        )

    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite float64 forward observations")
    return config.data_scale * result


def forward64_batched(parameters: np.ndarray, integration_points: int, batch_size: int = 768) -> np.ndarray:
    p = _as_parameter_array(parameters)
    parts = []
    for start in range(0, len(p), int(batch_size)):
        parts.append(scaled_forward_observation_float64(
            p[start:start+int(batch_size)], integration_points=integration_points
        ))
    return np.concatenate(parts, axis=0) if parts else np.empty((0, 0), dtype=np.float64)
