from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import numpy as np

from mc_pool_config import DEFAULT_PHYSICS, PhysicsConfig


def _as_parameter_array(parameters: np.ndarray) -> np.ndarray:
    array = np.asarray(parameters, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 5:
        raise ValueError(
            f"parameters must have shape [N, 5], received {array.shape}"
        )
    return array


def rho_numpy(parameters: np.ndarray, s: np.ndarray | float) -> np.ndarray:
    """Evaluate rho(s) for a batch of [a1, a2, a3, m, gamma]."""
    p = _as_parameter_array(parameters)
    s_array = np.asarray(s, dtype=np.float64)
    a1, a2, a3, mass, gamma = [p[:, i:i + 1] for i in range(5)]
    width = mass * gamma
    center = mass * mass
    values = (
        (a1 / np.pi)
        * width
        / ((s_array.reshape(1, -1) - center) ** 2 + width**2)
        + a2 * s_array.reshape(1, -1)
        + a3
    )
    return values


def _rho_at_per_row(parameters: np.ndarray, s_values: np.ndarray) -> np.ndarray:
    p = _as_parameter_array(parameters)
    s_values = np.asarray(s_values, dtype=np.float64)
    if s_values.shape != (len(p),):
        raise ValueError("s_values must have one scalar for every parameter row")
    a1, a2, a3, mass, gamma = [p[:, i] for i in range(5)]
    width = mass * gamma
    center = mass * mass
    return (
        (a1 / np.pi)
        * width
        / ((s_values - center) ** 2 + width**2)
        + a2 * s_values
        + a3
    )


def minimum_rho_on_interval_numpy(
    parameters: np.ndarray,
    s_min: float = DEFAULT_PHYSICS.s_min,
    s_max: float = DEFAULT_PHYSICS.s_max,
    bisection_iterations: int = 40,
) -> np.ndarray:
    """
    Continuous-interval minimum for the specific rho model.

    rho'(s) is always positive to the left of m^2. To the right of m^2 it may
    contain a local maximum followed by one local minimum. The function checks
    both interval endpoints and solves for that possible local minimum with a
    vectorized bisection. This avoids relying on a sparse non-negativity grid.
    """
    p = _as_parameter_array(parameters)
    if len(p) == 0:
        return np.empty(0, dtype=np.float64)

    a1, a2, _, mass, gamma = [p[:, i] for i in range(5)]
    width = mass * gamma
    center = mass * mass
    left = _rho_at_per_row(p, np.full(len(p), s_min, dtype=np.float64))
    right = _rho_at_per_row(p, np.full(len(p), s_max, dtype=np.float64))
    minimum = np.minimum(left, right)

    # d rho / ds = a2 - 2*c*x/(x^2+w^2)^2, x=s-m^2, c=a1*w/pi.
    x_right = s_max - center
    x_left = np.maximum(s_min - center, 0.0)
    x_peak = width / math.sqrt(3.0)
    lo_all = np.maximum(x_left, x_peak)
    c = a1 * width / np.pi

    def derivative(x: np.ndarray, ids: np.ndarray) -> np.ndarray:
        w = width[ids]
        return a2[ids] - (
            2.0 * c[ids] * x / ((x * x + w * w) ** 2)
        )

    base_mask = (
        (a1 > 0.0)
        & (a2 > 0.0)
        & (width > 0.0)
        & (x_right > lo_all)
    )
    ids = np.flatnonzero(base_mask)
    if len(ids):
        d_lo = derivative(lo_all[ids], ids)
        d_hi = derivative(x_right[ids], ids)
        eligible_local = (d_lo < 0.0) & (d_hi >= 0.0)
        ids = ids[eligible_local]

    if len(ids):
        lo = lo_all[ids].copy()
        hi = x_right[ids].copy()
        for _ in range(int(bisection_iterations)):
            mid = 0.5 * (lo + hi)
            d_mid = derivative(mid, ids)
            # The second root crosses from negative to positive.
            negative = d_mid < 0.0
            lo[negative] = mid[negative]
            hi[~negative] = mid[~negative]
        stationary_s = center[ids] + 0.5 * (lo + hi)
        stationary_rho = _rho_at_per_row(p[ids], stationary_s)
        minimum[ids] = np.minimum(minimum[ids], stationary_rho)

    return minimum


def valid_parameter_mask_numpy(
    parameters: np.ndarray,
    config: PhysicsConfig = DEFAULT_PHYSICS,
    tolerance: float = -1e-12,
) -> np.ndarray:
    p = _as_parameter_array(parameters)
    finite = np.isfinite(p).all(axis=1)
    ranges = (
        (p[:, 0] >= config.a1_min)
        & (p[:, 0] <= config.a1_max)
        & (p[:, 1] >= config.a2_min)
        & (p[:, 1] <= config.a2_max)
        & (p[:, 2] >= config.a3_min)
        & (p[:, 2] <= config.a3_max)
        & (p[:, 3] > config.m_min_open)
        & (p[:, 3] <= config.m_max)
        & (p[:, 4] > config.gamma_min_open)
        & (p[:, 4] <= config.gamma_max)
    )
    result = np.zeros(len(p), dtype=bool)
    candidate_ids = np.flatnonzero(finite & ranges)
    if len(candidate_ids):
        minimum = minimum_rho_on_interval_numpy(
            p[candidate_ids],
            s_min=config.s_min,
            s_max=config.s_max,
        )
        result[candidate_ids] = minimum >= tolerance
    return result


@lru_cache(maxsize=32)
def legendre_rule(points: int) -> Tuple[np.ndarray, np.ndarray]:
    if points < 8:
        raise ValueError("integration points must be at least 8")
    nodes, weights = np.polynomial.legendre.leggauss(int(points))
    return nodes.astype(np.float64), weights.astype(np.float64)


def output_grids_numpy(
    config: PhysicsConfig = DEFAULT_PHYSICS,
) -> tuple[np.ndarray, np.ndarray]:
    s = np.linspace(
        config.s_min, config.s_max, config.output_points, dtype=np.float64
    )
    q2 = np.linspace(
        config.q2_min, config.q2_max, config.q2_points, dtype=np.float64
    )
    return s, q2


def scaled_target_u_numpy(
    parameters: np.ndarray,
    config: PhysicsConfig = DEFAULT_PHYSICS,
) -> np.ndarray:
    s_output, _ = output_grids_numpy(config)
    rho = rho_numpy(parameters, s_output)
    u = rho / (s_output.reshape(1, -1) + config.shift) ** 2
    return (config.data_scale * u).astype(np.float32)


def scaled_forward_observation_at_q2_numpy(
    parameters: np.ndarray,
    q2: np.ndarray,
    integration_points: int = 128,
    config: PhysicsConfig = DEFAULT_PHYSICS,
    q_block_size: int = 25,
    output_dtype=np.float64,
) -> np.ndarray:
    """Compute scaled g_clean on an arbitrary Euclidean q^2 design.

    This is the single NumPy source of truth for the corrected resonance center
    m^2.  Observation-design experiments should call this function instead of
    reimplementing the spectral formula.
    """
    p = _as_parameter_array(parameters)
    q2 = np.asarray(q2, dtype=np.float64).reshape(-1)
    if q2.size == 0 or not np.isfinite(q2).all():
        raise ValueError("q2 must contain finite observation points")
    if np.any(q2 >= config.s_min):
        raise ValueError("q2 points must stay below the spectral integration interval")
    x, w = legendre_rule(int(integration_points))

    # Smooth background: (a2*s+a3)/(s+shift)^2/(s-q2).
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

    # Corrected Lorentzian resonance: center=m^2, width=m*gamma.
    a1 = p[:, 0:1]
    mass = p[:, 3:4]
    center = mass * mass
    width = mass * p[:, 4:5]
    if np.any(width <= 0.0):
        raise ValueError(
            "m and gamma must be strictly positive for numerical forward integration; "
            "the nominal zero endpoints are singular/degenerate boundaries"
        )
    z0 = np.arctan((config.s_min - center) / width)
    z1 = np.arctan((config.s_max - center) / width)
    z_mid = 0.5 * (z0 + z1)
    z_half = 0.5 * (z1 - z0)
    z = z_mid + z_half * x.reshape(1, -1)
    s_res = center + width * np.tan(z)
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
        raise FloatingPointError("non-finite forward observations were produced")
    scaled = config.data_scale * result
    return scaled.astype(output_dtype, copy=False)


def scaled_forward_observation_numpy(
    parameters: np.ndarray,
    integration_points: int = 128,
    config: PhysicsConfig = DEFAULT_PHYSICS,
    q_block_size: int = 25,
) -> np.ndarray:
    """Compute scaled g_clean(q^2) on the configured observation grid."""
    _, q2 = output_grids_numpy(config)
    return scaled_forward_observation_at_q2_numpy(
        parameters,
        q2=q2,
        integration_points=integration_points,
        config=config,
        q_block_size=q_block_size,
        output_dtype=np.float32,
    )


def scaled_curves_numpy(
    parameters: np.ndarray,
    integration_points: int = 128,
    config: PhysicsConfig = DEFAULT_PHYSICS,
) -> tuple[np.ndarray, np.ndarray]:
    target = scaled_target_u_numpy(parameters, config=config)
    observation = scaled_forward_observation_numpy(
        parameters,
        integration_points=integration_points,
        config=config,
    )
    return target, observation


def add_rms_noise_numpy(
    clean: np.ndarray,
    noise_level: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    clean = np.asarray(clean, dtype=np.float32)
    rms = np.sqrt(np.mean(clean.astype(np.float64) ** 2, axis=1, keepdims=True))
    noise = (
        float(noise_level)
        * rms
        * rng.standard_normal(clean.shape)
    ).astype(np.float32)
    return clean + noise, noise


def scaled_curves_torch(
    parameters,
    integration_points: int = 128,
    config: PhysicsConfig = DEFAULT_PHYSICS,
    q_block_size: int = 25,
):
    """Torch equivalent used by the daily trainer on CPU, CUDA, or XPU."""
    import torch

    if parameters.ndim != 2 or parameters.shape[1] != 5:
        raise ValueError("parameters must have shape [N, 5]")
    device = parameters.device
    dtype = parameters.dtype
    x_np, w_np = legendre_rule(int(integration_points))
    x = torch.as_tensor(x_np, device=device, dtype=dtype)
    w = torch.as_tensor(w_np, device=device, dtype=dtype)
    s_output = torch.linspace(
        config.s_min, config.s_max, config.output_points,
        device=device, dtype=dtype
    )
    q2 = torch.linspace(
        config.q2_min, config.q2_max, config.q2_points,
        device=device, dtype=dtype
    )

    a1 = parameters[:, 0:1]
    a2 = parameters[:, 1:2]
    a3 = parameters[:, 2:3]
    mass = parameters[:, 3:4]
    gamma = parameters[:, 4:5]
    width = mass * gamma
    center = mass * mass

    rho_output = (
        (a1 / math.pi)
        * width
        / ((s_output[None, :] - center) ** 2 + width**2)
        + a2 * s_output[None, :]
        + a3
    )
    target = (
        config.data_scale
        * rho_output
        / (s_output[None, :] + config.shift) ** 2
    )

    s_mid = 0.5 * (config.s_max + config.s_min)
    s_half = 0.5 * (config.s_max - config.s_min)
    s_fixed = s_mid + s_half * x
    s_weights = s_half * w
    background = (
        a2 * s_fixed[None, :] + a3
    ) / (s_fixed[None, :] + config.shift) ** 2
    background_kernel = (
        s_weights[:, None]
        / (s_fixed[:, None] - q2[None, :])
    )
    result = background @ background_kernel

    z0 = torch.atan((config.s_min - center) / width)
    z1 = torch.atan((config.s_max - center) / width)
    z_mid = 0.5 * (z0 + z1)
    z_half = 0.5 * (z1 - z0)
    z = z_mid + z_half * x[None, :]
    s_res = center + width * torch.tan(z)
    coefficient = (
        (a1 / math.pi)
        * z_half
        * w[None, :]
        / (s_res + config.shift) ** 2
    )
    for start in range(0, config.q2_points, int(q_block_size)):
        stop = min(start + int(q_block_size), config.q2_points)
        denominator = (
            s_res[:, :, None] - q2[None, None, start:stop]
        )
        result[:, start:stop] = result[:, start:stop] + torch.sum(
            coefficient[:, :, None] / denominator, dim=1
        )

    observation = config.data_scale * result
    if not torch.isfinite(target).all() or not torch.isfinite(observation).all():
        raise FloatingPointError("non-finite curves were produced")
    return target.to(torch.float32), observation.to(torch.float32)


def add_rms_noise_torch(clean, noise_level: float):
    import torch

    rms = torch.sqrt(torch.mean(clean.to(torch.float64) ** 2, dim=1, keepdim=True))
    rms = rms.to(clean.dtype)
    noise = torch.randn_like(clean) * (float(noise_level) * rms)
    return clean + noise
