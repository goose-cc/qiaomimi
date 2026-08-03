from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonteCarloInverseLoss(nn.Module):
    """Loss for g(q^2) -> f(s) on the Monte-Carlo parameter-pool problem.

    All tensors are expected to use the same scaling convention:
        f_scaled = data_scale * f
        g_scaled = data_scale * g

    Shapes:
        f_pred, f_true: [B, 1, Ns]
        g_reference:    [B, 1, Nq]
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
        profile: str = "pinn",
        normalization: str = "relative",
        lambda_grad: float = 0.1,
        lambda_physics: float = 0.1,
        lambda_smooth: float = 0.0,
        lambda_tv: float = 0.0,
        lambda_tikhonov: float = 0.0,
        huber_beta: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if profile not in self.VALID_PROFILES:
            raise ValueError(f"unknown loss profile: {profile}")
        if normalization not in {"relative", "absolute"}:
            raise ValueError("normalization must be 'relative' or 'absolute'")
        if output_points < 2 or input_points < 1:
            raise ValueError("output_points must be >=2 and input_points must be >=1")
        if not s_max > s_min:
            raise ValueError("s_max must be greater than s_min")

        self.profile = profile
        self.normalization = normalization
        self.lambda_grad = float(lambda_grad)
        self.lambda_physics = float(lambda_physics)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_tv = float(lambda_tv)
        self.lambda_tikhonov = float(lambda_tikhonov)
        self.huber_beta = float(huber_beta)
        self.eps = float(eps)

        s = torch.linspace(float(s_min), float(s_max), int(output_points))
        q2 = torch.linspace(float(q2_min), float(q2_max), int(input_points))

        # Uniform-grid trapezoidal weights.  The physical forward map is
        #     g(q^2) = integral f(s)/(s-q^2) ds.
        ds = (float(s_max) - float(s_min)) / (int(output_points) - 1)
        trap_weights = torch.full((int(output_points),), ds)
        trap_weights[0] *= 0.5
        trap_weights[-1] *= 0.5
        forward_matrix = trap_weights[:, None] / (s[:, None] - q2[None, :])

        self.register_buffer("s_grid", s, persistent=False)
        self.register_buffer("q2_grid", q2, persistent=False)
        self.register_buffer("forward_matrix", forward_matrix, persistent=False)

    @staticmethod
    def _check_shape(name: str, z: torch.Tensor, length: int) -> None:
        if z.ndim != 3 or z.shape[1] != 1 or z.shape[2] != length:
            raise ValueError(
                f"{name} must have shape [B, 1, {length}], got {tuple(z.shape)}"
            )

    @staticmethod
    def first_diff(z: torch.Tensor) -> torch.Tensor:
        return z[..., 1:] - z[..., :-1]

    @staticmethod
    def second_diff(z: torch.Tensor) -> torch.Tensor:
        return z[..., 2:] - 2.0 * z[..., 1:-1] + z[..., :-2]

    def physics_forward_integral(self, f_scaled: torch.Tensor) -> torch.Tensor:
        """Apply the sampled-grid forward operator to f_scaled."""
        self._check_shape("f_scaled", f_scaled, self.forward_matrix.shape[0])
        matrix = self.forward_matrix.to(device=f_scaled.device, dtype=f_scaled.dtype)
        g_scaled = f_scaled.squeeze(1) @ matrix
        return g_scaled.unsqueeze(1)

    def _relative_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        scale_target: torch.Tensor | None = None,
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

    def forward(
        self,
        f_pred: torch.Tensor,
        f_true: torch.Tensor,
        g_reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self._check_shape("f_pred", f_pred, self.forward_matrix.shape[0])
        self._check_shape("f_true", f_true, self.forward_matrix.shape[0])
        self._check_shape("g_reference", g_reference, self.forward_matrix.shape[1])

        zero = f_pred.new_zeros(())
        data_mse = self._relative_mse(f_pred, f_true)

        use_huber = self.profile in {"huber_grad", "pinn_huber"}
        use_grad = self.profile in {
            "mse_grad", "huber_grad", "pinn", "pinn_smooth",
            "pinn_huber", "pinn_full",
        }
        use_physics = self.profile in {
            "pinn", "pinn_smooth", "pinn_tikhonov",
            "pinn_huber", "pinn_full",
        }
        use_smooth = self.profile in {"pinn_smooth", "pinn_full"}
        use_tv = self.profile == "pinn_full"
        use_tikhonov = self.profile in {"pinn_tikhonov", "pinn_full"}

        data_huber = self._relative_huber(f_pred, f_true) if use_huber else zero

        if use_grad:
            # 用 f_true 的整体 RMS 定标，而不是用一阶差分自身的 RMS。
            # 后者在平坦样本上会接近零，导致 gradient loss 异常放大。
            grad = self._relative_mse(
                self.first_diff(f_pred),
                self.first_diff(f_true),
                scale_target=f_true,
            )
        else:
            grad = zero

        if use_physics:
            g_pred = self.physics_forward_integral(f_pred)
            physics = self._relative_mse(g_pred, g_reference)
        else:
            physics = zero

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
                data_mse
                + self.lambda_grad * grad
                + self.lambda_physics * physics
                + self.lambda_smooth * smooth
            )
        elif self.profile == "pinn_tikhonov":
            total = data_mse + self.lambda_physics * physics + self.lambda_tikhonov * tikhonov
        elif self.profile == "pinn_huber":
            total = data_huber + self.lambda_grad * grad + self.lambda_physics * physics
        elif self.profile == "pinn_full":
            total = (
                data_mse
                + self.lambda_grad * grad
                + self.lambda_physics * physics
                + self.lambda_smooth * smooth
                + self.lambda_tv * tv
                + self.lambda_tikhonov * tikhonov
            )
        else:  # guarded in __init__
            raise AssertionError(self.profile)

        logs = {
            "total": float(total.detach().cpu()),
            "data_mse": float(data_mse.detach().cpu()),
            "data_huber": float(data_huber.detach().cpu()),
            "grad": float(grad.detach().cpu()),
            "physics": float(physics.detach().cpu()),
            "smooth": float(smooth.detach().cpu()),
            "tv": float(tv.detach().cpu()),
            "tikhonov": float(tikhonov.detach().cpu()),
        }
        return total, logs
