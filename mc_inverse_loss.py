from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonteCarloInverseLoss(nn.Module):
    """Loss for the V2 free-curve inverse model.

    The old profiles are kept unchanged.  Two new peak-aware profiles are added:

      peak_grad = data + grad + peak_weighted + resonance
      peak_pinn = peak_grad + weak forward-physics consistency

    The peak-aware terms use the known synthetic parameters only during training.
    They do not alter the physical forward problem or the target curves.
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
        "peak_grad",
        "peak_pinn",
        "narrow_peak",
        "narrow_peak_pinn",
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
        profile: str = "peak_grad",
        normalization: str = "relative",
        lambda_grad: float = 0.05,
        lambda_physics: float = 0.01,
        lambda_peak: float = 0.5,
        lambda_resonance: float = 0.02,
        lambda_background: float = 0.2,
        lambda_peak_shape: float = 1.0,
        lambda_peak_grad: float = 0.05,
        lambda_peak_area: float = 0.25,
        lambda_peak_center: float = 0.05,
        lambda_peak_distribution: float = 0.2,
        peak_window_min_cells: float = 2.0,
        peak_softargmax_beta: float = 20.0,
        peak_alpha: float = 5.0,
        min_resonance_visibility: float = 0.10,
        min_width_grid_cells: float = 1.0,
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
            ("lambda_resonance", lambda_resonance),
            ("lambda_background", lambda_background),
            ("lambda_peak_shape", lambda_peak_shape),
            ("lambda_peak_grad", lambda_peak_grad),
            ("lambda_peak_area", lambda_peak_area),
            ("lambda_peak_center", lambda_peak_center),
            ("lambda_peak_distribution", lambda_peak_distribution),
            ("peak_window_min_cells", peak_window_min_cells),
            ("peak_softargmax_beta", peak_softargmax_beta),
            ("peak_alpha", peak_alpha),
            ("min_resonance_visibility", min_resonance_visibility),
            ("min_width_grid_cells", min_width_grid_cells),
        ):
            if float(value) < 0:
                raise ValueError("%s must be nonnegative" % name)

        self.profile = profile
        self.normalization = normalization
        self.lambda_grad = float(lambda_grad)
        self.lambda_physics = float(lambda_physics)
        self.lambda_peak = float(lambda_peak)
        self.lambda_resonance = float(lambda_resonance)
        self.lambda_background = float(lambda_background)
        self.lambda_peak_shape = float(lambda_peak_shape)
        self.lambda_peak_grad = float(lambda_peak_grad)
        self.lambda_peak_area = float(lambda_peak_area)
        self.lambda_peak_center = float(lambda_peak_center)
        self.lambda_peak_distribution = float(lambda_peak_distribution)
        self.peak_window_min_cells = float(peak_window_min_cells)
        self.peak_softargmax_beta = float(peak_softargmax_beta)
        self.peak_alpha = float(peak_alpha)
        self.min_resonance_visibility = float(min_resonance_visibility)
        self.min_width_grid_cells = float(min_width_grid_cells)
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
        self.register_buffer("trap_weights", trap_weights, persistent=False)

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

    def _decompose_true_curve(
        self,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if params.ndim != 2 or params.shape[1] != 5:
            raise ValueError("params must have shape [B,5]")
        dtype = params.dtype
        device = params.device
        s = self.s_grid.to(device=device, dtype=dtype).view(1, 1, -1)
        a1 = params[:, 0].view(-1, 1, 1)
        a2 = params[:, 1].view(-1, 1, 1)
        a3 = params[:, 2].view(-1, 1, 1)
        mass = params[:, 3].view(-1, 1, 1)
        gamma = params[:, 4].view(-1, 1, 1)
        width = mass * gamma

        resonance_rho = (
            (a1 / math.pi)
            * width
            / ((s - mass).square() + width.square())
        )
        background_rho = a2 * s + a3
        denominator = (s + self.shift).square()
        resonance = self.data_scale * resonance_rho / denominator
        background = self.data_scale * background_rho / denominator
        return resonance, background, mass.squeeze(-1).squeeze(-1), width.squeeze(-1).squeeze(-1)

    def _peak_terms(
        self,
        f_pred: torch.Tensor,
        f_true: torch.Tensor,
        params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        resonance, background, mass, width = self._decompose_true_curve(params)

        # Weighted full-curve MSE: the original complete target remains the target,
        # but points carrying true resonance information receive larger weight.
        peak_max = torch.amax(resonance, dim=2, keepdim=True).clamp_min(self.eps)
        point_weight = 1.0 + self.peak_alpha * resonance / peak_max
        weighted_numerator = torch.sum(
            point_weight * (f_pred - f_true).square(), dim=(1, 2)
        ) / torch.sum(point_weight, dim=(1, 2)).clamp_min(self.eps)
        if self.normalization == "relative":
            full_scale = torch.mean(f_true.square(), dim=(1, 2)).clamp_min(self.eps)
            peak_weighted = torch.mean(weighted_numerator / full_scale)
        else:
            peak_weighted = torch.mean(weighted_numerator)

        # Strict resonance diagnostic/supervision.  Using the known synthetic
        # background prevents the many background points from hiding a bad peak.
        resonance_pred = f_pred - background
        resonance_num = torch.mean((resonance_pred - resonance).square(), dim=(1, 2))
        resonance_den = torch.mean(resonance.square(), dim=(1, 2)).clamp_min(self.eps)
        resonance_rel = resonance_num / resonance_den

        visibility = torch.sqrt(resonance_den) / torch.sqrt(
            torch.mean(f_true.square(), dim=(1, 2)).clamp_min(self.eps)
        )
        eligible = visibility >= self.min_resonance_visibility
        eligible = eligible & ((width / self.ds) >= self.min_width_grid_cells)
        if self.peak_in_domain_only:
            eligible = eligible & (mass >= self.s_min) & (mass <= self.s_max)

        eligible_float = eligible.to(dtype=f_pred.dtype)
        eligible_fraction = torch.mean(eligible_float)
        if torch.any(eligible):
            resonance_loss = torch.sum(resonance_rel * eligible_float) / torch.sum(
                eligible_float
            ).clamp_min(1.0)
        else:
            resonance_loss = f_pred.new_zeros(())
        return peak_weighted, resonance_loss, eligible_fraction

    def _narrow_component_terms(
        self,
        f_true: torch.Tensor,
        params: torch.Tensor,
        background_pred: torch.Tensor,
        resonance_pred: torch.Tensor,
        peak_location_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._check_shape("background_pred", background_pred, self.forward_matrix.shape[0])
        self._check_shape("resonance_pred", resonance_pred, self.forward_matrix.shape[0])
        resonance_true, background_true, mass, width = self._decompose_true_curve(params)

        # 背景分支使用完整谱尺度归一化，避免接近零的背景样本导致数值爆炸。
        background_loss = self._relative_mse(
            background_pred, background_true, scale_target=f_true
        )

        s = self.s_grid.to(device=f_true.device, dtype=f_true.dtype).view(1, 1, -1)
        mass_view = mass.view(-1, 1, 1)
        width_view = width.view(-1, 1, 1)
        sigma = torch.maximum(
            width_view,
            f_true.new_tensor(self.peak_window_min_cells * self.ds),
        )
        window = torch.exp(-0.5 * ((s - mass_view) / sigma).square())
        window_sum = torch.sum(window, dim=(1, 2)).clamp_min(self.eps)

        shape_num = torch.sum(
            window * (resonance_pred - resonance_true).square(), dim=(1, 2)
        ) / window_sum
        shape_den = torch.sum(
            window * resonance_true.square(), dim=(1, 2)
        ) / window_sum
        shape_rel = shape_num / shape_den.clamp_min(self.eps)

        pred_grad = self.first_diff(resonance_pred)
        true_grad = self.first_diff(resonance_true)
        grad_window = 0.5 * (window[..., 1:] + window[..., :-1])
        grad_window_sum = torch.sum(grad_window, dim=(1, 2)).clamp_min(self.eps)
        grad_num = torch.sum(
            grad_window * (pred_grad - true_grad).square(), dim=(1, 2)
        ) / grad_window_sum
        grad_den = torch.sum(
            grad_window * true_grad.square(), dim=(1, 2)
        ) / grad_window_sum
        peak_grad_rel = grad_num / grad_den.clamp_min(self.eps)

        trap = self.trap_weights.to(
            device=f_true.device, dtype=f_true.dtype
        ).view(1, 1, -1)
        pred_area = torch.sum(trap * resonance_pred, dim=(1, 2))
        true_area = torch.sum(trap * resonance_true, dim=(1, 2))
        peak_area_rel = (
            (pred_area - true_area).square()
            / true_area.square().clamp_min(self.eps)
        )

        peak_scale = torch.amax(resonance_pred, dim=2, keepdim=True).clamp_min(self.eps)
        normalized_peak = resonance_pred / peak_scale
        logits = self.peak_softargmax_beta * normalized_peak + torch.log(
            window.clamp_min(1e-6)
        )
        probability = torch.softmax(logits, dim=2)
        center_pred = torch.sum(probability * s, dim=2).squeeze(1)
        center_scale = torch.maximum(
            width, f_true.new_tensor(2.0 * self.ds)
        ).clamp_min(self.eps)
        center_rel = ((center_pred - mass) / center_scale).square()

        if peak_location_logits is not None:
            self._check_shape(
                "peak_location_logits", peak_location_logits, self.forward_matrix.shape[0]
            )
            target_distribution = resonance_true.clamp_min(0.0)
            target_distribution = target_distribution / torch.sum(
                target_distribution, dim=2, keepdim=True
            ).clamp_min(self.eps)
            target_log = torch.log(target_distribution.clamp_min(self.eps))
            pred_log = torch.log_softmax(peak_location_logits, dim=2)
            distribution_kl = torch.sum(
                target_distribution * (target_log - pred_log), dim=(1, 2)
            )
        else:
            distribution_kl = torch.zeros_like(center_rel)

        resonance_rms = torch.sqrt(
            torch.mean(resonance_true.square(), dim=(1, 2)).clamp_min(self.eps)
        )
        full_rms = torch.sqrt(
            torch.mean(f_true.square(), dim=(1, 2)).clamp_min(self.eps)
        )
        visibility = resonance_rms / full_rms
        eligible = visibility >= self.min_resonance_visibility
        eligible = eligible & ((width / self.ds) >= self.min_width_grid_cells)
        if self.peak_in_domain_only:
            eligible = eligible & (mass >= self.s_min) & (mass <= self.s_max)

        eligible_float = eligible.to(dtype=f_true.dtype)
        eligible_fraction = torch.mean(eligible_float)

        def eligible_mean(values: torch.Tensor) -> torch.Tensor:
            if torch.any(eligible):
                return torch.sum(values * eligible_float) / torch.sum(
                    eligible_float
                ).clamp_min(1.0)
            return f_true.new_zeros(())

        return (
            background_loss,
            eligible_mean(shape_rel),
            eligible_mean(torch.log1p(peak_grad_rel)),
            eligible_mean(peak_area_rel),
            eligible_mean(center_rel),
            eligible_mean(distribution_kl),
            eligible_fraction,
        )

    def forward(
        self,
        f_pred: torch.Tensor,
        f_true: torch.Tensor,
        g_reference: torch.Tensor,
        params: Optional[torch.Tensor] = None,
        background_pred: Optional[torch.Tensor] = None,
        resonance_pred: Optional[torch.Tensor] = None,
        peak_location_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self._check_shape("f_pred", f_pred, self.forward_matrix.shape[0])
        self._check_shape("f_true", f_true, self.forward_matrix.shape[0])
        self._check_shape("g_reference", g_reference, self.forward_matrix.shape[1])

        zero = f_pred.new_zeros(())
        data_mse = self._relative_mse(f_pred, f_true)

        use_huber = self.profile in {"huber_grad", "pinn_huber"}
        use_grad = self.profile in {
            "mse_grad", "huber_grad", "pinn", "pinn_smooth",
            "pinn_huber", "pinn_full", "peak_grad", "peak_pinn",
            "narrow_peak", "narrow_peak_pinn",
        }
        use_physics = self.profile in {
            "pinn", "pinn_smooth", "pinn_tikhonov",
            "pinn_huber", "pinn_full", "peak_pinn", "narrow_peak_pinn",
        }
        use_peak = self.profile in {"peak_grad", "peak_pinn", "narrow_peak", "narrow_peak_pinn"}
        use_narrow_components = self.profile in {"narrow_peak", "narrow_peak_pinn"}
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
                raise ValueError("peak-aware loss requires the true [a1,a2,a3,m,gamma] params")
            peak_weighted, resonance, eligible_fraction = self._peak_terms(
                f_pred, f_true, params
            )
        else:
            peak_weighted = zero
            resonance = zero
            eligible_fraction = zero

        background_component = zero
        peak_shape_component = zero
        peak_grad_component = zero
        peak_area_component = zero
        peak_center_component = zero
        peak_distribution_component = zero
        if use_narrow_components:
            if background_pred is None or resonance_pred is None:
                raise ValueError(
                    "narrow_peak loss requires explicit background_pred and resonance_pred"
                )
            (
                background_component,
                peak_shape_component,
                peak_grad_component,
                peak_area_component,
                peak_center_component,
                peak_distribution_component,
                eligible_fraction,
            ) = self._narrow_component_terms(
                f_true, params, background_pred, resonance_pred, peak_location_logits
            )

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
            total = data_mse + self.lambda_grad * grad + self.lambda_physics * physics + self.lambda_smooth * smooth
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
        elif self.profile == "peak_grad":
            total = (
                data_mse
                + self.lambda_grad * grad
                + self.lambda_peak * peak_weighted
                + self.lambda_resonance * resonance
            )
        elif self.profile == "peak_pinn":
            total = (
                data_mse
                + self.lambda_grad * grad
                + self.lambda_peak * peak_weighted
                + self.lambda_resonance * resonance
                + self.lambda_physics * physics
            )
        elif self.profile == "narrow_peak":
            total = (
                data_mse
                + self.lambda_grad * grad
                + self.lambda_peak * peak_weighted
                + self.lambda_background * background_component
                + self.lambda_peak_shape * peak_shape_component
                + self.lambda_peak_grad * peak_grad_component
                + self.lambda_peak_area * peak_area_component
                + self.lambda_peak_center * peak_center_component
                + self.lambda_peak_distribution * peak_distribution_component
            )
        elif self.profile == "narrow_peak_pinn":
            total = (
                data_mse
                + self.lambda_grad * grad
                + self.lambda_peak * peak_weighted
                + self.lambda_background * background_component
                + self.lambda_peak_shape * peak_shape_component
                + self.lambda_peak_grad * peak_grad_component
                + self.lambda_peak_area * peak_area_component
                + self.lambda_peak_center * peak_center_component
                + self.lambda_peak_distribution * peak_distribution_component
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
            "peak_weighted": float(peak_weighted.detach().cpu()),
            "resonance": float(resonance.detach().cpu()),
            "background_component": float(background_component.detach().cpu()),
            "peak_shape_component": float(peak_shape_component.detach().cpu()),
            "peak_grad_component": float(peak_grad_component.detach().cpu()),
            "peak_area_component": float(peak_area_component.detach().cpu()),
            "peak_center_component": float(peak_center_component.detach().cpu()),
            "peak_distribution_component": float(peak_distribution_component.detach().cpu()),
            "eligible_fraction": float(eligible_fraction.detach().cpu()),
            "smooth": float(smooth.detach().cpu()),
            "tv": float(tv.detach().cpu()),
            "tikhonov": float(tikhonov.detach().cpu()),
        }
        return total, logs
