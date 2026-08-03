from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


PARAMETER_NAMES = ("a1", "a2", "a3", "m", "gamma")


def parameter_bounds(
    *,
    a1_min: float = 0.0,
    a1_max: float = 0.2,
    a2_min: float = 0.0,
    a2_max: float = 0.05,
    a3_min: float = -0.05,
    a3_max: float = 0.05,
    m_min_open: float = 0.0,
    m_max: float = 2.0,
    gamma_min_open: float = 0.0,
    gamma_max: float = 1.0,
    open_bound_eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return finite lower/upper bounds in [a1,a2,a3,m,gamma] order.

    m and gamma have open lower physical bounds. A neural network cannot emit an
    exact open endpoint through a sigmoid anyway, so a tiny numerical epsilon is
    used without changing the practical physical domain.
    """
    lower = torch.tensor(
        [
            float(a1_min),
            float(a2_min),
            float(a3_min),
            max(float(m_min_open) + float(open_bound_eps), float(open_bound_eps)),
            max(float(gamma_min_open) + float(open_bound_eps), float(open_bound_eps)),
        ],
        dtype=torch.float32,
    )
    upper = torch.tensor(
        [float(a1_max), float(a2_max), float(a3_max), float(m_max), float(gamma_max)],
        dtype=torch.float32,
    )
    if not torch.all(upper > lower):
        raise ValueError("all parameter upper bounds must be greater than lower bounds")
    return lower, upper


def normalize_parameters(
    parameters: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    lower = lower.to(device=parameters.device, dtype=parameters.dtype)
    upper = upper.to(device=parameters.device, dtype=parameters.dtype)
    return (parameters - lower) / (upper - lower)


def denormalize_parameters(
    normalized: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    lower = lower.to(device=normalized.device, dtype=normalized.dtype)
    upper = upper.to(device=normalized.device, dtype=normalized.dtype)
    return lower + normalized * (upper - lower)


def scaled_spectral_components(
    parameters: torch.Tensor,
    s_grid: torch.Tensor,
    *,
    shift: float,
    data_scale: float,
    width_eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode physical parameters into scaled total/resonance/background curves.

    This is exactly the data-generating physical family:
        rho(s) = a1/pi * (m*gamma)/((s-m)^2+(m*gamma)^2) + a2*s + a3
        f(s)   = data_scale * rho(s)/(s+shift)^2
    """
    if parameters.ndim != 2 or parameters.shape[1] != 5:
        raise ValueError("parameters must have shape [B, 5]")
    if s_grid.ndim != 1:
        raise ValueError("s_grid must be one-dimensional")

    dtype = parameters.dtype
    device = parameters.device
    s = s_grid.to(device=device, dtype=dtype).view(1, -1)
    a1 = parameters[:, 0:1]
    a2 = parameters[:, 1:2]
    a3 = parameters[:, 2:3]
    mass = parameters[:, 3:4]
    gamma = parameters[:, 4:5]
    width = (mass * gamma).clamp_min(float(width_eps))

    rho_resonance = (
        (a1 / math.pi)
        * width
        / ((s - mass).square() + width.square())
    )
    rho_background = a2 * s + a3
    denominator = (s + float(shift)).square()
    resonance = float(data_scale) * rho_resonance / denominator
    background = float(data_scale) * rho_background / denominator
    total = resonance + background
    return total, resonance, background


def normalized_parameter_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    weights: torch.Tensor,
    sample_parameter_weights: torch.Tensor = None,
) -> torch.Tensor:
    pred_n = normalize_parameters(predicted, lower, upper)
    target_n = normalize_parameters(target, lower, upper)
    global_weights = weights.to(device=predicted.device, dtype=predicted.dtype).view(1, -1)
    if sample_parameter_weights is None:
        effective = global_weights.expand_as(pred_n)
    else:
        effective = global_weights * sample_parameter_weights.to(
            device=predicted.device, dtype=predicted.dtype
        )
        if effective.shape != pred_n.shape:
            raise ValueError("sample_parameter_weights must have shape [B, 5]")
    numerator = torch.sum((pred_n - target_n).square() * effective, dim=1)
    denominator = torch.sum(effective, dim=1).clamp_min(1e-12)
    return torch.mean(numerator / denominator)


def resonance_visibility(
    resonance: torch.Tensor,
    total_curve: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    resonance_rms = torch.sqrt(torch.mean(resonance.square(), dim=1))
    total_rms = torch.sqrt(torch.mean(total_curve.square(), dim=1)).clamp_min(float(eps))
    return (resonance_rms / total_rms).clamp(0.0, 1.0)


def normalized_parameter_rmse_per_sample(
    predicted: torch.Tensor,
    target: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    weights: torch.Tensor,
    sample_parameter_weights: torch.Tensor = None,
) -> torch.Tensor:
    pred_n = normalize_parameters(predicted, lower, upper)
    target_n = normalize_parameters(target, lower, upper)
    global_weights = weights.to(device=predicted.device, dtype=predicted.dtype).view(1, -1)
    if sample_parameter_weights is None:
        effective = global_weights.expand_as(pred_n)
    else:
        effective = global_weights * sample_parameter_weights.to(
            device=predicted.device, dtype=predicted.dtype
        )
    numerator = torch.sum((pred_n - target_n).square() * effective, dim=1)
    denominator = torch.sum(effective, dim=1).clamp_min(1e-12)
    return torch.sqrt(numerator / denominator)


def relative_curve_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    scale_floor: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-sample relative MSE with a physically meaningful floor.

    The floor prevents nearly-zero resonances (a1 approximately zero) from
    dominating the loss while still making visible peaks important.
    """
    numerator = torch.mean((predicted - target).square(), dim=1)
    denominator = torch.mean(target.square(), dim=1)
    denominator = denominator + 0.01 * torch.mean(scale_floor.square(), dim=1)
    return torch.mean(numerator / denominator.clamp_min(float(eps)))


def parameter_error_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pred_n = normalize_parameters(predicted, lower, upper)
    true_n = normalize_parameters(target, lower, upper)
    diff_n = pred_n - true_n
    result: Dict[str, torch.Tensor] = {
        "parameter_normalized_rmse": torch.sqrt(torch.mean(diff_n.square(), dim=1)),
    }
    for index, name in enumerate(PARAMETER_NAMES):
        result[name + "_abs_error"] = torch.abs(predicted[:, index] - target[:, index])
        result[name + "_normalized_abs_error"] = torch.abs(diff_n[:, index])
    true_width = target[:, 3] * target[:, 4]
    pred_width = predicted[:, 3] * predicted[:, 4]
    result["width_abs_error"] = torch.abs(pred_width - true_width)
    result["width_relative_error"] = torch.abs(pred_width - true_width) / true_width.abs().clamp_min(1e-8)
    return result
