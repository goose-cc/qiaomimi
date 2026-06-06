import torch.nn as nn
from ResidualBlock import ResidualBlock
import torch.nn.functional as F

class PeakInversionCNN(nn.Module):
    """峰值反演CNN模型"""
    def __init__(self, input_length=200, output_length=200, num_channels=64):
        super(PeakInversionCNN, self).__init__()
        
        # 编码器部分：提取g(y)特征
        self.encoder = nn.Sequential(
            nn.Conv1d(1, num_channels//4, 5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels//4),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(num_channels//4, num_channels//2, 5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels//2),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(num_channels//2, num_channels, 5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True),
            
            ResidualBlock(num_channels),
            ResidualBlock(num_channels),
        )
        
        # 中间处理：使用空洞卷积扩大感受野
        self.mid_conv = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, 3, dilation=2, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(num_channels, num_channels, 3, dilation=4, padding=4, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(inplace=True),
        )
        
        # 解码器部分：重建f(x)
        self.decoder = nn.Sequential(
            nn.Conv1d(num_channels, num_channels//2, 5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels//2),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(num_channels//2, num_channels//4, 5, padding=2, padding_mode='replicate'),
            nn.BatchNorm1d(num_channels//4),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(num_channels//4, 1, 5, padding=2, padding_mode='replicate'),
        )
        
        # 自适应池化处理维度不匹配
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_length)
        
    def forward(self, x):
        # x: [batch_size, 1, input_length]
        features = self.encoder(x)
        features = self.mid_conv(features)
        output = self.decoder(features)
        output = self.adaptive_pool(output)  # 确保输出长度正确
        output = F.relu(output)  # 保证输出非负
        return output