import torch
import torch.nn as nn


class InverseProblemLoss(nn.Module):
    """
    反问题训练损失函数。

    目标：
    输入 g(y)，模型预测 f(x)。
    由于该问题是病态反问题，所以除了基础 MSE 外，
    额外加入梯度约束、物理一致性约束、平滑正则、TV 正则和 Tikhonov 正则。

    支持的 loss_profile：
    1. base
       只使用 MSE(f_pred, f_true)

    2. mse_grad
       MSE + 一阶梯度约束

    3. pinn
       MSE + 一阶梯度约束 + 物理一致性约束

    4. pinn_smooth
       MSE + 一阶梯度约束 + 物理一致性约束 + 二阶平滑正则

    5. pinn_tikhonov
       MSE + 物理一致性约束 + Tikhonov 正则

    6. pinn_full
       MSE + 一阶梯度约束 + 物理一致性约束 + 二阶平滑正则 + TV 正则 + Tikhonov 正则
    """

    def __init__(self, config, profile="pinn_smooth"):
        super().__init__()

        self.config = config
        self.profile = profile
        self.mse = nn.MSELoss()

        # 从 config 中读取权重；如果 config.py 中没有这些参数，就使用默认值
        self.lambda_grad = getattr(config, "lambda_grad", 0.1)
        self.lambda_physics = getattr(config, "lambda_physics", 0.05)
        self.lambda_smooth = getattr(config, "lambda_smooth", 0.01)
        self.lambda_tv = getattr(config, "lambda_tv", 0.001)
        self.lambda_tikhonov = getattr(config, "lambda_tikhonov", 0.0001)

        self.num_x_points = config.num_x_points
        self.num_y_points = config.num_y_points

    def first_diff(self, z):
        """
        一阶差分，用来约束曲线变化趋势。
        z shape: [B, 1, L]
        """
        return z[:, :, 1:] - z[:, :, :-1]

    def second_diff(self, z):
        """
        二阶差分，用来约束曲线平滑性。
        z shape: [B, 1, L]
        """
        return z[:, :, 2:] - 2 * z[:, :, 1:-1] + z[:, :, :-2]

    def gradient_loss(self, pred, target):
        """
        一阶梯度匹配损失：
        希望预测曲线和真实曲线的变化趋势一致。
        """
        return self.mse(self.first_diff(pred), self.first_diff(target))

    def smooth_loss(self, pred):
        """
        二阶平滑正则：
        抑制预测 f(x) 出现剧烈局部震荡。
        """
        return torch.mean(self.second_diff(pred) ** 2)

    def tv_loss(self, pred):
        """
        TV 正则：
        Total Variation，总变差正则。
        可以让预测曲线更加稳定。
        """
        return torch.mean(torch.abs(self.first_diff(pred)))

    def tikhonov_loss(self, pred):
        """
        Tikhonov 正则：
        控制预测 f(x) 的整体幅值，避免病态反问题中误差被过度放大。
        """
        return torch.mean(pred ** 2)

    def normalize_per_sample(self, z):
        """
        对每个样本单独标准化。
        用于物理一致性 loss 中比较 g(y)。

        因为 PIDataset.py 中通常会对输入 g(y) 做标准化，
        所以这里也把由 f_pred 积分得到的 g_pred 做同样的标准化比较。
        """
        mean = z.mean(dim=-1, keepdim=True)
        std = z.std(dim=-1, keepdim=True)
        return (z - mean) / (std + 1e-8)

    def physics_forward_integral(self, fx_pred):
        """
        根据预测得到的 f(x)，通过积分关系重新计算 g(y)。

        g(y) = integral_a^b f(x) / (y - x) dx

        fx_pred shape: [B, 1, num_x_points]
        return shape: [B, 1, num_y_points]
        """
        device = fx_pred.device

        fx = fx_pred.squeeze(1)  # [B, num_x_points]

        # 这里使用项目中原来的积分区间
        x = torch.linspace(0, 2, self.num_x_points, device=device)
        y = torch.linspace(2.1, 10, self.num_y_points, device=device)

        # Y, X shape: [num_y_points, num_x_points]
        Y, X = torch.meshgrid(y, x, indexing="ij")

        denominator = Y - X

        # 避免分母过小造成数值不稳定
        sign = torch.sign(denominator)
        sign = torch.where(sign == 0, torch.tensor(1.0, device=device), sign)
        denominator = torch.where(
            torch.abs(denominator) < 1e-6,
            sign * 1e-6,
            denominator
        )

        # integrand shape: [B, num_y_points, num_x_points]
        integrand = fx.unsqueeze(1) / denominator.unsqueeze(0)

        # 沿 x 方向积分
        gy_pred = torch.trapz(integrand, x, dim=-1)  # [B, num_y_points]

        return gy_pred.unsqueeze(1)  # [B, 1, num_y_points]

    def physics_loss(self, fx_pred, gy_input):
        """
        物理一致性损失：
        由预测 f(x) 积分得到 g_pred(y)，
        再与模型输入的 g(y) 进行比较。
        """
        gy_pred = self.physics_forward_integral(fx_pred)

        gy_pred_norm = self.normalize_per_sample(gy_pred)
        gy_input_norm = self.normalize_per_sample(gy_input)

        return self.mse(gy_pred_norm, gy_input_norm)

    def forward(self, fx_pred, fx_true, gy_input):
        """
        根据 profile 组合不同损失项。

        fx_pred: 模型预测的 f(x)
        fx_true: 数据集中真实的 f(x)
        gy_input: 模型输入的 g(y)
        """
        mse_loss = self.mse(fx_pred, fx_true)
        grad_loss = self.gradient_loss(fx_pred, fx_true)
        physics_loss = self.physics_loss(fx_pred, gy_input)
        smooth_loss = self.smooth_loss(fx_pred)
        tv_loss = self.tv_loss(fx_pred)
        tikhonov_loss = self.tikhonov_loss(fx_pred)

        if self.profile == "base":
            total_loss = mse_loss

        elif self.profile == "mse_grad":
            total_loss = (
                mse_loss
                + self.lambda_grad * grad_loss
            )

        elif self.profile == "pinn":
            total_loss = (
                mse_loss
                + self.lambda_grad * grad_loss
                + self.lambda_physics * physics_loss
            )

        elif self.profile == "pinn_smooth":
            total_loss = (
                mse_loss
                + self.lambda_grad * grad_loss
                + self.lambda_physics * physics_loss
                + self.lambda_smooth * smooth_loss
            )

        elif self.profile == "pinn_tikhonov":
            total_loss = (
                mse_loss
                + self.lambda_physics * physics_loss
                + self.lambda_tikhonov * tikhonov_loss
            )

        elif self.profile == "pinn_full":
            total_loss = (
                mse_loss
                + self.lambda_grad * grad_loss
                + self.lambda_physics * physics_loss
                + self.lambda_smooth * smooth_loss
                + self.lambda_tv * tv_loss
                + self.lambda_tikhonov * tikhonov_loss
            )

        else:
            raise ValueError(f"Unknown loss profile: {self.profile}")

        loss_logs = {
            "total": total_loss.item(),
            "mse": mse_loss.item(),
            "grad": grad_loss.item(),
            "physics": physics_loss.item(),
            "smooth": smooth_loss.item(),
            "tv": tv_loss.item(),
            "tikhonov": tikhonov_loss.item(),
        }

        return total_loss, loss_logs
