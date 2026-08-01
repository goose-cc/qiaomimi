from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from dataset_config import DatasetConfig


PARAM_NAMES = np.array(["a1", "a2", "a3", "m", "gamma"])


def calculate_rho(
    s: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    a3: np.ndarray,
    m: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    """
    Evaluate
        rho(s) = a1/pi * (m*gamma) /
                 ((s-m)^2 + (m*gamma)^2) + a2*s + a3

    Inputs are broadcast using NumPy rules. All calculations are float64.
    """
    s = np.asarray(s, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    a2 = np.asarray(a2, dtype=np.float64)
    a3 = np.asarray(a3, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)

    width = m * gamma
    denominator = (s - m) ** 2 + width**2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        resonance = (a1 / np.pi) * width / denominator
        rho = resonance + a2 * s + a3
    return rho


def _h_derivative_term(
    x: np.ndarray,
    a1: np.ndarray,
    width: np.ndarray,
) -> np.ndarray:
    """
    h(x) = 2*(a1/pi)*width*x / (x^2+width^2)^2, x >= 0.
    rho'(m+x) = a2 - h(x).
    """
    x = np.asarray(x, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    width = np.asarray(width, dtype=np.float64)
    den = x * x + width * width
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return 2.0 * (a1 / np.pi) * width * x / (den * den)


def continuous_minimum_rho(
    params: np.ndarray,
    s_min: float,
    s_max: float,
    bisection_iterations: int = 64,
) -> np.ndarray:
    """
    Compute the continuous minimum of rho(s) on [s_min, s_max].

    Because a2 >= 0:
      * rho is increasing to the left of m;
      * to the right of m, rho' = a2 - h(x), where h is unimodal;
      * therefore the global minimum can only occur at an endpoint or at the
        second derivative root on the descending branch of h.

    This avoids relying only on a coarse grid and catches local negative dips.
    """
    params = np.asarray(params, dtype=np.float64)
    if params.ndim != 2 or params.shape[1] != 5:
        raise ValueError("params must have shape [N, 5]")

    a1, a2, a3, m, gamma = params.T
    width = m * gamma

    left = calculate_rho(
        np.full_like(m, s_min), a1, a2, a3, m, gamma
    )
    right = calculate_rho(
        np.full_like(m, s_max), a1, a2, a3, m, gamma
    )
    minimum = np.minimum(left, right)

    x_interval_left = np.maximum(s_min - m, 0.0)
    x_interval_right = s_max - m
    x_peak = width / np.sqrt(3.0)
    lo = np.maximum(x_interval_left, x_peak)
    hi = x_interval_right.copy()

    h_lo = _h_derivative_term(lo, a1, width)
    h_hi = _h_derivative_term(hi, a1, width)

    candidate = (
        (a1 > 0.0)
        & (a2 > 0.0)
        & (width > 0.0)
        & (hi > lo)
        & np.isfinite(h_lo)
        & np.isfinite(h_hi)
        & (h_lo > a2)
        & (h_hi <= a2)
    )

    if np.any(candidate):
        idx = np.flatnonzero(candidate)
        lo_c = lo[idx].copy()
        hi_c = hi[idx].copy()
        a1_c = a1[idx]
        a2_c = a2[idx]
        width_c = width[idx]

        for _ in range(bisection_iterations):
            mid = 0.5 * (lo_c + hi_c)
            h_mid = _h_derivative_term(mid, a1_c, width_c)
            # h is decreasing on this branch. If h(mid) > a2, the root is right.
            move_right = h_mid > a2_c
            lo_c = np.where(move_right, mid, lo_c)
            hi_c = np.where(move_right, hi_c, mid)

        x_root = 0.5 * (lo_c + hi_c)
        s_root = m[idx] + x_root
        rho_root = calculate_rho(
            s_root, a1[idx], a2[idx], a3[idx], m[idx], gamma[idx]
        )
        minimum[idx] = np.minimum(minimum[idx], rho_root)

    return minimum


def valid_parameter_mask(
    params: np.ndarray,
    cfg: DatasetConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (valid_mask, continuous_minimum_rho).

    The theoretical ranges remain unchanged. Exact zero for m or gamma is
    rejected; no artificial positive lower bound is introduced.
    """
    params = np.asarray(params, dtype=np.float64)
    a1, a2, a3, m, gamma = params.T

    in_range = (
        (a1 >= cfg.a1_min)
        & (a1 <= cfg.a1_max)
        & (a2 >= cfg.a2_min)
        & (a2 <= cfg.a2_max)
        & (a3 >= cfg.a3_min)
        & (a3 <= cfg.a3_max)
        & (m >= cfg.m_min)
        & (m <= cfg.m_max)
        & (gamma >= cfg.gamma_min)
        & (gamma <= cfg.gamma_max)
    )
    basic = in_range & (m > 0.0) & (gamma > 0.0)
    finite_params = np.all(np.isfinite(params), axis=1)

    minimum = np.full(params.shape[0], -np.inf, dtype=np.float64)
    eligible = basic & finite_params
    if np.any(eligible):
        minimum[eligible] = continuous_minimum_rho(
            params[eligible], cfg.s_min, cfg.s_max
        )

    valid = (
        eligible
        & np.isfinite(minimum)
        & (minimum >= -cfg.rho_negative_tolerance)
    )
    return valid, minimum


@dataclass
class ForwardOperator:
    cfg: DatasetConfig

    def __post_init__(self) -> None:
        self.s_output = np.linspace(
            self.cfg.s_min,
            self.cfg.s_max,
            self.cfg.signal_length,
            dtype=np.float64,
        )
        self.q2_grid = np.linspace(
            self.cfg.q2_min,
            self.cfg.q2_max,
            self.cfg.observation_length,
            dtype=np.float64,
        )

        # Smooth background basis integrals:
        # I0(q2) = int ds / ((s+400)^2 (s-q2))
        # I1(q2) = int s ds / ((s+400)^2 (s-q2))
        bg_nodes, bg_weights = np.polynomial.legendre.leggauss(
            self.cfg.background_quadrature_order
        )
        s_mid = 0.5 * (self.cfg.s_min + self.cfg.s_max)
        s_half = 0.5 * (self.cfg.s_max - self.cfg.s_min)
        s_bg = s_mid + s_half * bg_nodes
        w_bg = s_half * bg_weights
        kernel = (
            w_bg[:, None]
            / ((s_bg[:, None] + 400.0) ** 2)
            / (s_bg[:, None] - self.q2_grid[None, :])
        )
        self.background_i0 = np.sum(kernel, axis=0)
        self.background_i1 = np.sum(s_bg[:, None] * kernel, axis=0)

        # Resonance is integrated after z = atan((s-m)/(m*gamma)).
        self.res_nodes, self.res_weights = np.polynomial.legendre.leggauss(
            self.cfg.resonance_quadrature_order
        )

    def compute_batch(
        self,
        params: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return rho(s_output), u(s_output), and g_clean(q2), all float64.

        The resonance integral uses:
            z = atan((s-m)/(m*gamma))
        which removes the arbitrarily narrow Lorentzian peak from the numerical
        integration. This is substantially more stable than a fixed sparse s-grid.
        """
        params = np.asarray(params, dtype=np.float64)
        if params.ndim != 2 or params.shape[1] != 5:
            raise ValueError("params must have shape [N, 5]")

        a1, a2, a3, m, gamma = params.T
        width = m * gamma
        if np.any(width <= 0.0):
            raise ValueError("m and gamma must be strictly positive")

        rho = calculate_rho(
            self.s_output[None, :],
            a1[:, None],
            a2[:, None],
            a3[:, None],
            m[:, None],
            gamma[:, None],
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            u = rho / (self.s_output[None, :] + 400.0) ** 2

        # Smooth linear background contribution.
        g_background = (
            a2[:, None] * self.background_i1[None, :]
            + a3[:, None] * self.background_i0[None, :]
        )

        # Peak-aware transformed quadrature for the resonance contribution.
        z_lo = np.arctan2(self.cfg.s_min - m, width)
        z_hi = np.arctan2(self.cfg.s_max - m, width)
        z_mid = 0.5 * (z_lo + z_hi)
        z_half = 0.5 * (z_hi - z_lo)

        z = z_mid[:, None] + z_half[:, None] * self.res_nodes[None, :]
        s_res = m[:, None] + width[:, None] * np.tan(z)
        w_res = z_half[:, None] * self.res_weights[None, :]

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            denominator = (
                (s_res[:, :, None] + 400.0) ** 2
                * (s_res[:, :, None] - self.q2_grid[None, None, :])
            )
            integrand = w_res[:, :, None] / denominator
            g_resonance = (
                (a1 / np.pi)[:, None] * np.sum(integrand, axis=1)
            )

        g_clean = g_resonance + g_background
        return rho, u, g_clean


def storage_safety_mask(
    rho: np.ndarray,
    u: np.ndarray,
    g_clean: np.ndarray,
    storage_dtype: str,
    safety_fraction: float,
) -> np.ndarray:
    """
    Reject NaN/Inf and values too close to the storage dtype's finite limit.
    """
    dtype = np.dtype(storage_dtype)
    if dtype.kind != "f":
        raise ValueError("storage_dtype must be a floating dtype")
    finite_limit = np.finfo(dtype).max * float(safety_fraction)

    mask = (
        np.all(np.isfinite(rho), axis=1)
        & np.all(np.isfinite(u), axis=1)
        & np.all(np.isfinite(g_clean), axis=1)
        & (np.max(np.abs(rho), axis=1) <= finite_limit)
        & (np.max(np.abs(u), axis=1) <= finite_limit)
        & (np.max(np.abs(g_clean), axis=1) <= finite_limit)
    )
    return mask
