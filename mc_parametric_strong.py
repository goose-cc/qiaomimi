from __future__ import annotations

import math
from typing import Dict

import torch

from mc_parametric import normalize_parameters

STRONG_TARGET_NAMES = ('a1', 'a2', 'a3', 'm', 'log_gamma')


def transformed_parameter_coordinates(params, lower, upper, gamma_floor=1e-5):
    """Return [a1,a2,a3,m,log_gamma] in approximately [0,1] coordinates."""
    p = params
    lo = lower.to(device=p.device, dtype=p.dtype)
    hi = upper.to(device=p.device, dtype=p.dtype)
    linear = normalize_parameters(p, lo, hi)
    gamma = p[:, 4].clamp_min(float(gamma_floor))
    log_lo = math.log(float(gamma_floor))
    log_hi = math.log(float(hi[4]))
    gamma_n = (torch.log(gamma) - log_lo) / max(log_hi - log_lo, 1e-12)
    return torch.stack([linear[:,0], linear[:,1], linear[:,2], linear[:,3], gamma_n], dim=1)


def strong_parameter_mse(pred, true, lower, upper, weights, gamma_floor=1e-5):
    pn = transformed_parameter_coordinates(pred, lower, upper, gamma_floor)
    tn = transformed_parameter_coordinates(true, lower, upper, gamma_floor)
    w = weights.to(device=pred.device, dtype=pred.dtype).view(1,-1)
    return torch.mean(torch.sum((pn-tn).square()*w,dim=1)/w.sum().clamp_min(1e-12))


def relative_mse(pred, true, floor=None, floor_fraction=0.01, eps=1e-10):
    num = torch.mean((pred-true).square(),dim=1)
    den = torch.mean(true.square(),dim=1)
    if floor is not None:
        den = den + float(floor_fraction)*torch.mean(floor.square(),dim=1)
    return torch.mean(num/den.clamp_min(float(eps)))


def gradient_relative_mse(pred, true, reference, eps=1e-10):
    dp = pred[:,1:] - pred[:,:-1]
    dt = true[:,1:] - true[:,:-1]
    ref = reference[:,1:] - reference[:,:-1]
    num = torch.mean((dp-dt).square(),dim=1)
    den = torch.mean(dt.square(),dim=1) + 0.01*torch.mean(ref.square(),dim=1)
    return torch.mean(num/den.clamp_min(float(eps)))


def log_width_mse(pred_params, true_params, width_floor=1e-10):
    pw=(pred_params[:,3]*pred_params[:,4]).clamp_min(float(width_floor))
    tw=(true_params[:,3]*true_params[:,4]).clamp_min(float(width_floor))
    # Divide by ln(10): error is measured in decades.  A factor-of-10 miss -> 1.
    return torch.mean(((torch.log(pw)-torch.log(tw))/math.log(10.0)).square())


def resonance_peak_height(params, shift, data_scale, width_floor=1e-10):
    a1=params[:,0]
    m=params[:,3]
    w=(params[:,3]*params[:,4]).clamp_min(float(width_floor))
    return float(data_scale)*(a1/math.pi)/w/(m+float(shift)).square()


def log_peak_height_mse(pred_params, true_params, shift, data_scale, eps=1e-12):
    ph=resonance_peak_height(pred_params,shift,data_scale).clamp_min(float(eps))
    th=resonance_peak_height(true_params,shift,data_scale).clamp_min(float(eps))
    return torch.mean(((torch.log(ph)-torch.log(th))/math.log(10.0)).square())


def width_metrics(pred_params, true_params, width_floor=1e-10) -> Dict[str,torch.Tensor]:
    pw=(pred_params[:,3]*pred_params[:,4]).clamp_min(float(width_floor))
    tw=(true_params[:,3]*true_params[:,4]).clamp_min(float(width_floor))
    return {
        'width_abs_error': torch.abs(pw-tw),
        'width_relative_error': torch.abs(pw-tw)/tw,
        'width_log10_abs_error': torch.abs(torch.log10(pw)-torch.log10(tw)),
    }
