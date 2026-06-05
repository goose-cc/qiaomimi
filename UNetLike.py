import torch.nn as nn
import torch
import torch.nn.functional as F


class UNetLikeModel(nn.Module):
    """UNet风格模型，适合处理反问题"""
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super(UNetLikeModel, self).__init__()
        
        self.encoder1 = self._block(in_channels, features[0])
        self.pool1 = nn.MaxPool1d(2)
        self.encoder2 = self._block(features[0], features[1])
        self.pool2 = nn.MaxPool1d(2)
        self.encoder3 = self._block(features[1], features[2])
        self.pool3 = nn.MaxPool1d(2)
        
        self.bottleneck = self._block(features[2], features[3])
        
        self.upconv3 = nn.ConvTranspose1d(features[3], features[2], 2, stride=2, output_padding=1)
        self.decoder3 = self._block(features[2]*2, features[2])
        self.upconv2 = nn.ConvTranspose1d(features[2], features[1], 2, stride=2)
        self.decoder2 = self._block(features[1]*2, features[1])
        self.upconv1 = nn.ConvTranspose1d(features[1], features[0], 2, stride=2)
        self.decoder1 = self._block(features[0]*2, features[0])
        
        self.final_conv = nn.Conv1d(features[0], out_channels, 1)
        
    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv1d(in_channels, features, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
            nn.Conv1d(features, features, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return F.relu(self.final_conv(dec1))
