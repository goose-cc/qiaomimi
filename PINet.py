import torch.nn as nn
import torch
import torch.nn.functional as F
from ResidualBlock import ResidualBlock


class PeakInversionCNN(nn.Module):
    """峰值反演 CNN 模型。

    注意：新数据中的 f(x) 可能为负，因此最后不再使用 ReLU 强制非负。
    """

    def __init__(self, input_length=100, output_length=100, num_channels=64):
        super(PeakInversionCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, num_channels // 4, 5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv1d(num_channels // 4, num_channels // 2, 5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(num_channels // 2, num_channels, 5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True),

            ResidualBlock(num_channels),
            ResidualBlock(num_channels),
        )

        self.mid_conv = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, 3, dilation=2, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(num_channels, num_channels, 3, dilation=4, padding=4, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(num_channels, num_channels // 2, 5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(num_channels // 2, num_channels // 4, 5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv1d(num_channels // 4, 1, 5, padding=2, padding_mode='replicate'),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_length)

    def forward(self, x):
        features = self.encoder(x)
        features = self.mid_conv(features)
        output = self.decoder(features)
        output = self.adaptive_pool(output)
        return output


class PhysicsInformedNetwork(nn.Module):
    """保留旧类名以兼容，但 PINN 的核心应在 loss，而不是网络类名。"""

    def __init__(self, input_length=100, output_length=100, num_channels=64):
        super(PhysicsInformedNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, num_channels // 4, 5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels // 4, num_channels // 2, 5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels // 2, num_channels, 5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(
            ResidualBlock(num_channels),
            ResidualBlock(num_channels),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(num_channels, num_channels // 2, 5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels // 2, num_channels // 4, 5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels // 4, 1, 5, padding=2, padding_mode='replicate'),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_length)

    def forward(self, x):
        features = self.encoder(x)
        features = self.residual(features)
        output = self.decoder(features)
        output = self.adaptive_pool(output)
        return output
