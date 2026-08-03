from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonteCarloInverseLoss(nn.Module):
    """Loss for g(q^2) -> f(s) on the Monte-Carlo inverse problem.

    ``peak_finetune`` is a conservative V2 fine-tuning profile.  It keeps the
    original full-spectrum relative MSE as the dominant term, adds a small
    gradient term, adds a local loss only around visible/resolvable true peaks,
    and adds a weak forward-consistency term to protect the already-good V2
    mapping.

    Expected shapes:
        f_pred, f_true: [B, 1, Ns]
        g_reference:    [B, 1, Nq]
        params:         [B, 5] = [a1, a2, a3, m, gamma]
    """

    VALID_PROFILES = {
        "base",
        "mse_grad",
        "huber_grad",
        "pinn",
        "pinn_smooth",
        "pinn_tikhonov",
        "pinn_huber",
        "pinn_full",
        "peak_finetune",
    }

    def __init__(
        self,
        *,
        s_min: float,
        s_max: float,
        q2_min: float,
        q2_max: float,
        output_points: int,
        input_points: int,
        profile: str = "peak_finetune",
        normalization: str = "relative",
        lambda_grad: float = 0.02,
        lambda_physics: float = 0.03,
        lambda_peak: float = 0.10,
        peak_window_widths: float = 3.0,
        peak_width_min: float = 0.06,
        peak_width_max: float = 0.50,
        min_resonance_visibility: float = 0.10,
        peak_in_domain_only: bool = True,
        data_scale: float = 160000.0,
        shift: float = 400.0,
        lambda_smooth: float = 0.0,
        lambda_tv: float = 0.0,
        lambda_tikhonov: float = 0.0,
        huber_beta: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if profile not in self.VALID_PROFILES:
            raise ValueError("unknown loss profile: %s" % profile)
        if normalization not in {"relative", "absolute"}:
            raise ValueError("normalization must be 'relative' or 'absolute'")
        if output_points < 2 or input_points < 1:
            raise ValueError("output_points must be >=2 and input_points must be >=1")
        if not s_max > s_min:
            raise ValueError("s_max must be greater than s_min")
        for name, value in (
            ("lambda_grad", lambda_grad),
            ("lambda_physics", lambda_physics),
            ("lambda_peak", lambda_peak),
            ("peak_window_widths", peak_window_widths),
            ("peak_width_min", peak_width_min),
            ("peak_width_max", peak_width_max),
            ("min_resonance_visibility", min_resonance_visibility),
        ):
            if float(value) < 0:
                raise ValueError("%s must be nonnegative" % name)
        if peak_width_max > 0 and peak_width_max < peak_width_min:
            raise ValueError("peak_width_max must be >= peak_width_min or 0 for no upper bound")

        self.profile = profile
        self.normalization = normalization
        self.lambda_grad = float(lambda_grad)
        self.lambda_physics = float(lambda_physics)
        self.lambda_peak = float(lambda_peak)
        self.peak_window_widths = float(peak_window_widths)
        self.peak_width_min = float(peak_width_min)
        self.peak_width_max = float(peak_width_max)
        self.min_resonance_visibility = float(min_resonance_visibility)
        self.peak_in_domain_only = bool(peak_in_domain_only)
        self.data_scale = float(data_scale)
        self.shift = float(shift)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_tv = float(lambda_tv)
        self.lambda_tikhonov = float(lambda_tikhonov)
        self.huber_beta = float(huber_beta)
        self.eps = float(eps)

        s = torch.linspace(float(s_min), float(s_max), int(output_points))
        q2 = torch.linspace(float(q2_min), float(q2_max), int(input_points))
        ds = (float(s_max) - float(s_min)) / (int(output_points) - 1)
        trap_weights = torch.full((int(output_points),), ds)
        trap_weights[0] *= 0.5
        trap_weights[-1] *= 0.5
        forward_matrix = trap_weights[:, None] / (s[:, None] - q2[None, :])

        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.ds = float(ds)
        self.register_buffer("s_grid", s, persistent=False)
        self.register_buffer("q2_grid", q2, persistent=False)
        self.register_buffer("forward_matrix", forward_matrix, persistent=False)

    @staticmethod
    def _check_shape(name: str, z: torch.Tensor, length: int) -> None:
        if z.ndim != 3 or z.shape[1] != 1 or z.shape[2] != length:
            raise ValueError(
                "%s must have shape [B, 1, %d], got %s"
                % (name, length, tuple(z.shape))
            )

    @staticmethod
    def first_diff(z: torch.Tensor) -> torch.Tensor:
        return z[..., 1:] - z[..., :-1]

    @staticmethod
    def second_diff(z: torch.Tensor) -> torch.Tensor:
        return z[..., 2:] - 2.0 * z[..., 1:-1] + z[..., :-2]

    def physics_forward_integral(self, f_scaled: torch.Tensor) -> torch.Tensor:
        self._check_shape("f_scaled", f_scaled, self.forward_matrix.shape[0])
        matrix = self.forward_matrix.to(device=f_scaled.device, dtype=f_scaled.dtype)
        return (f_scaled.squeeze(1) @ matrix).unsqueeze(1)

    def _relative_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        scale_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reduce_dims = tuple(range(1, pred.ndim))
        numerator = torch.mean((pred - target).square(), dim=reduce_dims)
        if self.normalization == "absolute":
            return numerator.mean()
        reference = target if scale_target is None else scale_target
        reference_dims = tuple(range(1, reference.ndim))
        denominator = torch.mean(reference.square(), dim=reference_dims).clamp_min(self.eps)
        return torch.mean(numerator / denominator)

    def _relative_huber(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.normalization == "relative":
            reduce_dims = tuple(range(1, target.ndim))
            rms = torch.sqrt(
                torch.mean(target.square(), dim=reduce_dims, keepdim=True).clamp_min(self.eps)
            )
            pred = pred / rms
            target = target / rms
        return F.smooth_l1_loss(pred, target, beta=self.huber_beta)

    def _true_resonance_and_peak_info(
        self,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if params.ndim != 2 or params.shape[1] != 5:
            raise ValueError("params must have shape [B,5]")
        dtype = params.dtype
        device = params.device
        s = self.s_grid.to(device=device, dtype=dtype).view(1, 1, -1)
        a1 = params[:, 0].view(-1, 1, 1)
        mass = params[:, 3].view(-1, 1, 1)
        gamma = params[:, 4].view(-1, 1, 1)
        width = mass * gamma
        resonance_rho = (
            (a1 / math.pi)
            * width
            / ((s - mass).square() + width.square())
        )
        resonance = self.data_scale * resonance_rho / (s + self.shift).square()
        return resonance, mass.view(-1), width.view(-1)

    def _local_peak_loss(
        self,
        f_pred: torch.Tensor,
        f_true: torch.Tensor,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stable local full-spectrum loss around eligible true peaks.

        The target remains the complete spectrum.  We do not subtract a true
        background and do not normalize by tiny resonance energy, avoiding the
        instability seen in the aggressive V2.6 experiment.
        """
        resonance, mass, width = self._true_resonance_and_peak_info(params)
        dtype = f_pred.dtype
        device = f_pred.device
        s = self.s_grid.to(device=device, dtype=dtype).view(1, 1, -1)
        mass_view = mass.to(device=device, dtype=dtype).view(-1, 1, 1)
        width_view = width.to(device=device, dtype=dtype).view(-1, 1, 1)

        resonance_energy = torch.mean(resonance.to(dtype=dtype).square(), dim=(1, 2))
        full_energy = torch.mean(f_true.square(), dim=(1, 2)).clamp_min(self.eps)
        visibility = torch.sqrt(resonance_energy.clamp_min(self.eps)) / torch.sqrt(full_energy)

        eligible = width.to(device=device) >= self.peak_width_min
        if self.peak_width_max > 0:
            eligible = eligible & (width.to(device=device) < self.peak_width_max)
        eligible = eligible & (visibility >= self.min_resonance_visibility)
        if self.peak_in_domain_only:
            eligible = eligible & (mass.to(device=device) >= self.s_min)
            eligible = eligible & (mass.to(device=device) <= self.s_max)

        half_width = self.peak_window_widths * torch.maximum(
            width_view,
            torch.full_like(width_view, self.ds),
        )
        point_mask = (torch.abs(s - mass_view) <= half_width).to(dtype=dtype)
        point_count = torch.sum(point_mask, dim=(1, 2)).clamp_min(1.0)
        local_mse = torch.sum(point_mask * (f_pred - f_true).square(), dim=(1, 2)) / point_count
        if self.normalization == "relative":
            local_mse = local_mse / full_energy

        eligible_float = eligible.to(dtype=dtype)
        eligible_fraction = torch.mean(eligible_float)
        if torch.any(eligible):
            local_loss = torch.sum(local_mse * eligible_float) / torch.sum(
                eligible_float
            ).clamp_min(1.0)
        else:
            local_loss = f_pred.new_zeros(())
        return local_loss, eligible_fraction

    def forward(
        self,
        f_pred: torch.Tensor,
        f_true: torch.Tensor,
        g_reference: torch.Tensor,
        params: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self._check_shape("f_pred", f_pred, self.forward_matrix.shape[0])
        self._check_shape("f_true", f_true, self.forward_matrix.shape[0])
        self._check_shape("g_reference", g_reference, self.forward_matrix.shape[1])

        zero = f_pred.new_zeros(())
        data_mse = self._relative_mse(f_pred, f_true)

        use_huber = self.profile in {"huber_grad", "pinn_huber"}
        use_grad = self.profile in {
            "mse_grad", "huber_grad", "pinn", "pinn_smooth",
            "pinn_huber", "pinn_full", "peak_finetune",
        }
        use_physics = self.profile in {
            "pinn", "pinn_smooth", "pinn_tikhonov",
            "pinn_huber", "pinn_full", "peak_finetune",
        }
        use_peak = self.profile == "peak_finetune"
        use_smooth = self.profile in {"pinn_smooth", "pinn_full"}
        use_tv = self.profile == "pinn_full"
        use_tikhonov = self.profile in {"pinn_tikhonov", "pinn_full"}

        data_huber = self._relative_huber(f_pred, f_true) if use_huber else zero
        grad = (
            self._relative_mse(
                self.first_diff(f_pred),
                self.first_diff(f_true),
                scale_target=f_true,
            )
            if use_grad else zero
        )
        physics = (
            self._relative_mse(self.physics_forward_integral(f_pred), g_reference)
            if use_physics else zero
        )
        if use_peak:
            if params is None:
                raise ValueError("peak_finetune requires true [a1,a2,a3,m,gamma] params")
            peak_local, eligible_fraction = self._local_peak_loss(
                f_pred, f_true, params
            )
        else:
            peak_local = zero
            eligible_fraction = zero

        if use_smooth:
            d2 = self.second_diff(f_pred)
            smooth = torch.mean(d2.square()) if d2.numel() else zero
        else:
            smooth = zero
        tv = torch.mean(torch.abs(self.first_diff(f_pred))) if use_tv else zero
        tikhonov = torch.mean(f_pred.square()) if use_tikhonov else zero

        if self.profile == "base":
            total = data_mse
        elif self.profile == "mse_grad":
            total = data_mse + self.lambda_grad * grad
        elif self.profile == "huber_grad":
            total = data_huber + self.lambda_grad * grad
        elif self.profile == "pinn":
            total = data_mse + self.lambda_grad * grad + self.lambda_physics * physics
        elif self.profile == "pinn_smooth":
            total = (
                data_mse + self.lambda_grad * grad
                + self.lambda_physics * physics
                + self.lambda_smooth * smooth
            )
        elif self.profile == "pinn_tikhonov":
            total = data_mse + self.lambda_physics * physics + self.lambda_tikhonov * tikhonov
        elif self.profile == "pinn_huber":
            total = data_huber + self.lambda_grad * grad + self.lambda_physics * physics
        elif self.profile == "pinn_full":
            total = (
                data_mse + self.lambda_grad * grad + self.lambda_physics * physics
                + self.lambda_smooth * smooth + self.lambda_tv * tv
                + self.lambda_tikhonov * tikhonov
            )
        elif self.profile == "peak_finetune":
            total = (
                data_mse
                + self.lambda_grad * grad
                + self.lambda_peak * peak_local
                + self.lambda_physics * physics
            )
        else:
            raise AssertionError(self.profile)

        logs = {
            "total": float(total.detach().cpu()),
            "data_mse": float(data_mse.detach().cpu()),
            "data_huber": float(data_huber.detach().cpu()),
            "grad": float(grad.detach().cpu()),
            "physics": float(physics.detach().cpu()),
            "peak_local": float(peak_local.detach().cpu()),
            "eligible_fraction": float(eligible_fraction.detach().cpu()),
            "smooth": float(smooth.detach().cpu()),
            "tv": float(tv.detach().cpu()),
            "tikhonov": float(tikhonov.detach().cpu()),
        }
        return total, logs
