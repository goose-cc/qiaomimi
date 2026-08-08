from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from mc_parametric import normalize_parameters


TARGET_NAMES = ("a1", "a2", "a3", "m", "log_gamma")


def transformed_parameter_coordinates(
    params: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    gamma_floor: float = 1e-5,
) -> torch.Tensor:
    """Return [a1,a2,a3,m,log_gamma] normalized coordinates."""
    lo = lower.to(device=params.device, dtype=params.dtype)
    hi = upper.to(device=params.device, dtype=params.dtype)
    linear = normalize_parameters(params, lo, hi)

    gamma = params[:, 4].clamp_min(float(gamma_floor))
    log_lo = math.log(float(gamma_floor))
    log_hi = math.log(float(hi[4]))
    gamma_n = (torch.log(gamma) - log_lo) / max(log_hi - log_lo, 1e-12)

    return torch.stack(
        [linear[:, 0], linear[:, 1], linear[:, 2], linear[:, 3], gamma_n],
        dim=1,
    )


def resonance_visibility(
    resonance: torch.Tensor,
    total_curve: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """RMS resonance fraction, clipped to [0,1].

    This is a training-time target derived only from the true curve.
    It is NOT an extra physical assumption and is not used at inference.
    """
    rr = torch.sqrt(torch.mean(resonance.square(), dim=1))
    tt = torch.sqrt(torch.mean(total_curve.square(), dim=1)).clamp_min(float(eps))
    return (rr / tt).clamp(0.0, 1.0)


def smooth_visibility_gate(
    visibility: torch.Tensor,
    low: float = 0.05,
    high: float = 0.20,
) -> torch.Tensor:
    """Smoothly map weak resonance -> 0 and clearly visible resonance -> 1."""
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("visibility thresholds must satisfy 0 <= low < high <= 1")
    t = ((visibility - float(low)) / float(high - low)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    values = values.reshape(-1)
    weights = weights.reshape(-1).to(device=values.device, dtype=values.dtype)
    return torch.sum(values * weights) / torch.sum(weights).clamp_min(float(eps))


def conditional_parameter_mse(
    pred: torch.Tensor,
    true: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    base_weights: torch.Tensor,
    resonance_boost_weights: torch.Tensor,
    resonance_gate: torch.Tensor,
    *,
    gamma_floor: float = 1e-5,
    boost_scale: float = 1.0,
) -> torch.Tensor:
    """Parameter loss with m/log-gamma emphasis only when a true resonance is visible.

    base_weights apply to every sample.
    resonance_boost_weights are multiplied by the per-sample visibility gate.
    """
    pn = transformed_parameter_coordinates(pred, lower, upper, gamma_floor)
    tn = transformed_parameter_coordinates(true, lower, upper, gamma_floor)

    bw = base_weights.to(device=pred.device, dtype=pred.dtype).view(1, 5)
    rw = resonance_boost_weights.to(device=pred.device, dtype=pred.dtype).view(1, 5)
    gate = resonance_gate.to(device=pred.device, dtype=pred.dtype).view(-1, 1)
    effective = bw + float(boost_scale) * gate * rw

    per_sample = torch.sum((pn - tn).square() * effective, dim=1)
    per_sample = per_sample / torch.sum(effective, dim=1).clamp_min(1e-12)
    return torch.mean(per_sample)


def per_sample_relative_mse(
    pred: torch.Tensor,
    true: torch.Tensor,
    *,
    floor: torch.Tensor | None = None,
    floor_fraction: float = 0.01,
    eps: float = 1e-10,
) -> torch.Tensor:
    num = torch.mean((pred - true).square(), dim=1)
    den = torch.mean(true.square(), dim=1)
    if floor is not None:
        den = den + float(floor_fraction) * torch.mean(floor.square(), dim=1)
    return num / den.clamp_min(float(eps))


def relative_mse(
    pred: torch.Tensor,
    true: torch.Tensor,
    *,
    floor: torch.Tensor | None = None,
    floor_fraction: float = 0.01,
    eps: float = 1e-10,
) -> torch.Tensor:
    return torch.mean(
        per_sample_relative_mse(
            pred,
            true,
            floor=floor,
            floor_fraction=floor_fraction,
            eps=eps,
        )
    )


def gated_relative_mse(
    pred: torch.Tensor,
    true: torch.Tensor,
    sample_gate: torch.Tensor,
    *,
    floor: torch.Tensor | None = None,
    floor_fraction: float = 0.01,
    eps: float = 1e-10,
) -> torch.Tensor:
    per = per_sample_relative_mse(
        pred,
        true,
        floor=floor,
        floor_fraction=floor_fraction,
        eps=eps,
    )
    return _weighted_mean(per, sample_gate)


def gradient_relative_mse(
    pred: torch.Tensor,
    true: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    dp = pred[:, 1:] - pred[:, :-1]
    dt = true[:, 1:] - true[:, :-1]
    ref = reference[:, 1:] - reference[:, :-1]
    num = torch.mean((dp - dt).square(), dim=1)
    den = torch.mean(dt.square(), dim=1) + 0.01 * torch.mean(ref.square(), dim=1)
    return torch.mean(num / den.clamp_min(float(eps)))


def per_sample_log_width_error_sq(
    pred_params: torch.Tensor,
    true_params: torch.Tensor,
    width_floor: float = 1e-10,
) -> torch.Tensor:
    pw = (pred_params[:, 3] * pred_params[:, 4]).clamp_min(float(width_floor))
    tw = (true_params[:, 3] * true_params[:, 4]).clamp_min(float(width_floor))
    return ((torch.log10(pw) - torch.log10(tw)) ** 2)


def gated_log_width_mse(
    pred_params: torch.Tensor,
    true_params: torch.Tensor,
    gate: torch.Tensor,
    width_floor: float = 1e-10,
) -> torch.Tensor:
    return _weighted_mean(
        per_sample_log_width_error_sq(pred_params, true_params, width_floor),
        gate,
    )


def resonance_peak_height(
    params: torch.Tensor,
    shift: float,
    data_scale: float,
    width_floor: float = 1e-10,
) -> torch.Tensor:
    a1 = params[:, 0]
    m = params[:, 3]
    width = (params[:, 3] * params[:, 4]).clamp_min(float(width_floor))
    return float(data_scale) * (a1 / math.pi) / width / (m + float(shift)).square()


def per_sample_log_peak_height_error_sq(
    pred_params: torch.Tensor,
    true_params: torch.Tensor,
    shift: float,
    data_scale: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    ph = resonance_peak_height(pred_params, shift, data_scale).clamp_min(float(eps))
    th = resonance_peak_height(true_params, shift, data_scale).clamp_min(float(eps))
    return (torch.log10(ph) - torch.log10(th)) ** 2


def gated_log_peak_height_mse(
    pred_params: torch.Tensor,
    true_params: torch.Tensor,
    gate: torch.Tensor,
    shift: float,
    data_scale: float,
) -> torch.Tensor:
    return _weighted_mean(
        per_sample_log_peak_height_error_sq(
            pred_params, true_params, shift, data_scale
        ),
        gate,
    )


def visibility_mse(
    pred_resonance: torch.Tensor,
    pred_total: torch.Tensor,
    true_resonance: torch.Tensor,
    true_total: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Directly teach the model how much resonance should be present.

    This is the main anti-false-peak signal for weak/background-like samples.
    """
    pv = resonance_visibility(pred_resonance, pred_total)
    tv = resonance_visibility(true_resonance, true_total)
    return torch.mean((pv - tv).square()), pv, tv


def weak_resonance_excess_mse(
    pred_resonance: torch.Tensor,
    true_resonance: torch.Tensor,
    true_total: torch.Tensor,
    weak_gate: torch.Tensor,
    *,
    tolerance_fraction: float = 0.02,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Penalize only EXCESS predicted resonance on weak-resonance samples.

    It does not force the resonance to zero.  The prediction may match the true
    small resonance plus a small tolerance proportional to the total curve RMS.
    """
    pred_rms = torch.sqrt(torch.mean(pred_resonance.square(), dim=1))
    true_rms = torch.sqrt(torch.mean(true_resonance.square(), dim=1))
    total_rms = torch.sqrt(torch.mean(true_total.square(), dim=1)).clamp_min(float(eps))

    allowed = true_rms + float(tolerance_fraction) * total_rms
    excess = torch.relu(pred_rms - allowed) / total_rms
    return _weighted_mean(excess.square(), weak_gate)


def width_metrics(
    pred_params: torch.Tensor,
    true_params: torch.Tensor,
    width_floor: float = 1e-10,
) -> Dict[str, torch.Tensor]:
    pw = (pred_params[:, 3] * pred_params[:, 4]).clamp_min(float(width_floor))
    tw = (true_params[:, 3] * true_params[:, 4]).clamp_min(float(width_floor))
    return {
        "width_abs_error": torch.abs(pw - tw),
        "width_relative_error": torch.abs(pw - tw) / tw,
        "width_log10_abs_error": torch.abs(torch.log10(pw) - torch.log10(tw)),
    }
