import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseProblemLoss(nn.Module):
    """反问题训练损失函数。

    支持的 loss_profile：
    - base: 只使用 MSE(f_pred, f_true)
    - mse_grad: MSE + 一阶梯度匹配
    - huber_grad: Huber + 一阶梯度匹配
    - pinn: MSE + 一阶梯度匹配 + 物理一致性
    - pinn_smooth: MSE + 一阶梯度匹配 + 物理一致性 + 二阶平滑
    - pinn_tikhonov: MSE + 物理一致性 + Tikhonov 正则
    - pinn_huber: Huber + 一阶梯度匹配 + 物理一致性
    - pinn_full: MSE + 一阶梯度匹配 + 物理一致性 + 平滑 + TV + Tikhonov
    """

    def __init__(self, config, profile='base'):
        super().__init__()
        self.config = config
        self.profile = profile

        self.mse = nn.MSELoss()

        self.lambda_grad = getattr(config, 'lambda_grad', 0.1)
        self.lambda_physics = getattr(config, 'lambda_physics', 0.01)
        self.lambda_smooth = getattr(config, 'lambda_smooth', 0.01)
        self.lambda_tv = getattr(config, 'lambda_tv', 0.001)
        self.lambda_tikhonov = getattr(config, 'lambda_tikhonov', 0.0001)

        self.num_x_points = config.num_x_points
        self.num_y_points = config.num_y_points
        self.normalize_gy_per_sample = getattr(config, 'normalize_gy_per_sample', True)

        self.x_min = getattr(config, 'x_min', 0.0)
        self.x_max = getattr(config, 'x_max', 2.0)
        self.y_min = getattr(config, 'y_min', 3.0)
        self.y_max = getattr(config, 'y_max', 8.0)

    def first_diff(self, z):
        return z[:, :, 1:] - z[:, :, :-1]

    def second_diff(self, z):
        return z[:, :, 2:] - 2.0 * z[:, :, 1:-1] + z[:, :, :-2]

    def gradient_loss(self, pred, target):
        return self.mse(self.first_diff(pred), self.first_diff(target))

    def smooth_loss(self, pred):
        return torch.mean(self.second_diff(pred) ** 2)

    def tv_loss(self, pred):
        return torch.mean(torch.abs(self.first_diff(pred)))

    def tikhonov_loss(self, pred):
        return torch.mean(pred ** 2)

    def normalize_per_sample(self, z):
        mean = z.mean(dim=-1, keepdim=True)
        std = z.std(dim=-1, keepdim=True)
        return (z - mean) / (std + 1e-8)

    def physics_forward_integral(self, fx_pred):
        """由预测的 f(x) 重新计算 g(y)。

        g(y_i) ≈ ∫ f(x)/(y_i-x) dx
        fx_pred: [B, 1, Nx]
        return:  [B, 1, Ny]
        """
        device = fx_pred.device
        dtype = fx_pred.dtype

        fx = fx_pred.squeeze(1)  # [B, Nx]

        x = torch.linspace(self.x_min, self.x_max, self.num_x_points, device=device, dtype=dtype)
        y = torch.linspace(self.y_min, self.y_max, self.num_y_points, device=device, dtype=dtype)

        Y, X = torch.meshgrid(y, x, indexing='ij')
        denominator = Y - X

        sign = torch.sign(denominator)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        denominator = torch.where(torch.abs(denominator) < 1e-6, sign * 1e-6, denominator)

        integrand = fx.unsqueeze(1) / denominator.unsqueeze(0)  # [B, Ny, Nx]
        gy_pred = torch.trapz(integrand, x, dim=-1)            # [B, Ny]
        return gy_pred.unsqueeze(1)                            # [B, 1, Ny]

    def physics_loss(self, fx_pred, gy_input):
        gy_pred = self.physics_forward_integral(fx_pred)

        # 如果 Dataset 中对 gy 做了逐样本标准化，这里也对 gy_pred 和 gy_input 做同样比较
        if self.normalize_gy_per_sample:
            gy_pred = self.normalize_per_sample(gy_pred)
            gy_input = self.normalize_per_sample(gy_input)

        return self.mse(gy_pred, gy_input)

    def forward(self, fx_pred, fx_true, gy_input):
        mse_loss = self.mse(fx_pred, fx_true)
        huber_loss = F.smooth_l1_loss(fx_pred, fx_true, beta=0.1)
        grad_loss = self.gradient_loss(fx_pred, fx_true)
        physics_loss = self.physics_loss(fx_pred, gy_input)
        smooth_loss = self.smooth_loss(fx_pred)
        tv_loss = self.tv_loss(fx_pred)
        tikhonov_loss = self.tikhonov_loss(fx_pred)

        if self.profile == 'base':
            total_loss = mse_loss

        elif self.profile == 'mse_grad':
            total_loss = mse_loss + self.lambda_grad * grad_loss

        elif self.profile == 'huber_grad':
            total_loss = huber_loss + self.lambda_grad * grad_loss

        elif self.profile == 'pinn':
            total_loss = mse_loss + self.lambda_grad * grad_loss + self.lambda_physics * physics_loss

        elif self.profile == 'pinn_smooth':
            total_loss = (
                mse_loss
                + self.lambda_grad * grad_loss
                + self.lambda_physics * physics_loss
                + self.lambda_smooth * smooth_loss
            )

        elif self.profile == 'pinn_tikhonov':
            total_loss = mse_loss + self.lambda_physics * physics_loss + self.lambda_tikhonov * tikhonov_loss

        elif self.profile == 'pinn_huber':
            total_loss = huber_loss + self.lambda_grad * grad_loss + self.lambda_physics * physics_loss

        elif self.profile == 'pinn_full':
            total_loss = (
                mse_loss
                + self.lambda_grad * grad_loss
                + self.lambda_physics * physics_loss
                + self.lambda_smooth * smooth_loss
                + self.lambda_tv * tv_loss
                + self.lambda_tikhonov * tikhonov_loss
            )

        else:
            raise ValueError(f'Unknown loss profile: {self.profile}')

        loss_logs = {
            'total': float(total_loss.detach().cpu()),
            'mse': float(mse_loss.detach().cpu()),
            'huber': float(huber_loss.detach().cpu()),
            'grad': float(grad_loss.detach().cpu()),
            'physics': float(physics_loss.detach().cpu()),
            'smooth': float(smooth_loss.detach().cpu()),
            'tv': float(tv_loss.detach().cpu()),
            'tikhonov': float(tikhonov_loss.detach().cpu()),
        }
        return total_loss, loss_logs
