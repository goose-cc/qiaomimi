from __future__ import annotations

import math

import torch

from mc_inverse_loss import MonteCarloInverseLoss


def make_loss(profile: str) -> MonteCarloInverseLoss:
    return MonteCarloInverseLoss(
        s_min=0.1764,
        s_max=6.0,
        q2_min=-100.0,
        q2_max=-6.0,
        output_points=100,
        input_points=100,
        profile=profile,
        normalization="relative",
        lambda_grad=0.05,
        lambda_physics=0.01,
        lambda_peak=0.5,
        lambda_resonance=0.1,
        peak_alpha=5.0,
        min_resonance_visibility=0.05,
        min_width_grid_cells=0.5,
        data_scale=160000.0,
        shift=400.0,
        lambda_smooth=0.01,
        lambda_tv=0.01,
        lambda_tikhonov=0.01,
    )


def curves_from_params(params: torch.Tensor) -> torch.Tensor:
    s = torch.linspace(0.1764, 6.0, 100).view(1, 1, -1)
    a1 = params[:, 0].view(-1, 1, 1)
    a2 = params[:, 1].view(-1, 1, 1)
    a3 = params[:, 2].view(-1, 1, 1)
    m = params[:, 3].view(-1, 1, 1)
    gamma = params[:, 4].view(-1, 1, 1)
    width = m * gamma
    rho = (
        (a1 / math.pi) * width / ((s - m).square() + width.square())
        + a2 * s + a3
    )
    return 160000.0 * rho / (s + 400.0).square()


def main() -> None:
    torch.manual_seed(20260802)
    params = torch.tensor(
        [
            [0.16, 0.020, 0.010, 0.50, 0.30],
            [0.12, 0.030, 0.005, 0.80, 0.25],
            [0.18, 0.015, 0.015, 1.10, 0.20],
            [0.10, 0.040, 0.010, 1.40, 0.35],
            [0.15, 0.025, 0.020, 0.65, 0.45],
            [0.13, 0.020, 0.010, 0.95, 0.50],
            [0.17, 0.010, 0.020, 1.30, 0.40],
            [0.11, 0.035, 0.005, 1.70, 0.30],
        ],
        dtype=torch.float32,
    )
    f_true = curves_from_params(params)
    reference_loss = make_loss("peak_pinn")
    g_clean = reference_loss.physics_forward_integral(f_true)

    for profile in sorted(MonteCarloInverseLoss.VALID_PROFILES):
        loss_fn = make_loss(profile)
        perfect, perfect_logs = loss_fn(f_true.clone(), f_true, g_clean, params=params)
        for key in ("data_mse", "grad", "physics", "peak_weighted", "resonance"):
            if perfect_logs[key] > 1e-8:
                raise RuntimeError(
                    "%s: perfect prediction should have zero %s, got %s"
                    % (profile, key, perfect_logs[key])
                )
        if not torch.isfinite(perfect):
            raise RuntimeError("%s: non-finite perfect loss" % profile)

        prediction = (f_true + 0.02 * torch.randn_like(f_true)).requires_grad_(True)
        loss, logs = loss_fn(prediction, f_true, g_clean, params=params)
        if not torch.isfinite(loss):
            raise RuntimeError("%s: non-finite loss" % profile)
        loss.backward()
        if prediction.grad is None or not torch.isfinite(prediction.grad).all():
            raise RuntimeError("%s: invalid gradient" % profile)
        print("%-16s loss=%.6e logs=%s" % (profile, float(loss.detach()), logs))

    print("all loss profiles passed")


if __name__ == "__main__":
    main()
