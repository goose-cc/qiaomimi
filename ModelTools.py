import os
from collections import defaultdict

import torch
import torch.nn as nn

from PINet import PeakInversionCNN
from UNetLike import UNetLikeModel
from TransformerInverse import InverseTransformer1D
from losses import InverseProblemLoss


class ModelTools:
    """峰值反演模型训练器。"""

    def __init__(self, config, model_type='cnn', loss_profile=None, LoadModel=False):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('device is', self.device)

        self.model_type = model_type
        self.loss_profile = loss_profile or getattr(config, 'loss_profile', 'base')

        # =====================
        # 选择网络结构
        # =====================
        if model_type == 'cnn':
            self.model = PeakInversionCNN(
                input_length=config.num_y_points,
                output_length=config.num_x_points,
            ).to(self.device)

        elif model_type == 'unet':
            self.model = UNetLikeModel().to(self.device)

        elif model_type == 'transformer':
            self.model = InverseTransformer1D(
                input_length=config.num_y_points,
                output_length=config.num_x_points,
                d_model=config.transformer_d_model,
                nhead=config.transformer_nhead,
                num_encoder_layers=config.transformer_num_layers,
                num_decoder_layers=config.transformer_num_layers,
                dim_feedforward=config.transformer_dim_feedforward,
                dropout=config.transformer_dropout,
                x_min=config.x_min,
                x_max=config.x_max,
                y_min=config.y_min,
                y_max=config.y_max,
            ).to(self.device)

        else:
            raise ValueError(f'未知网络结构: {model_type}')

        os.makedirs(getattr(config, 'model_dir', './model'), exist_ok=True)
        exp_name = getattr(config, "exp", "exp")
        self.model_path = os.path.join(
            getattr(config, "model_dir", "./model"),
            f"{exp_name}_{self.model_type}_{self.loss_profile}.pth"
        )

        if LoadModel:
            self.load_model(self.model_path)

        # 训练 loss：可切换 base / mse_grad / pinn / pinn_smooth 等
        self.loss_fn = InverseProblemLoss(config, profile=self.loss_profile).to(self.device)

        # 评估指标统一使用纯 MSE，方便不同 loss_profile 之间比较
        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        print(f'model_type = {self.model_type}')
        print(f'loss_profile = {self.loss_profile}')
        print(f'model_path = {self.model_path}')

    def load_model(self, model_path):
        print(f'尝试加载模型: {model_path}')
        try:
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            except TypeError:
                state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f'成功加载模型: {model_path}')
        except Exception as e:
            print(f'加载模型失败: {e}')
            print('将使用随机初始化的模型')

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        logs_sum = defaultdict(float)
        n_batches = 0

        for gy, fx in train_loader:
            gy = gy.to(self.device)
            fx = fx.to(self.device)

            self.optimizer.zero_grad()
            fx_pred = self.model(gy)

            loss, loss_logs = self.loss_fn(fx_pred, fx, gy)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            for k, v in loss_logs.items():
                logs_sum[k] += v
            n_batches += 1

        avg_logs = {k: v / max(n_batches, 1) for k, v in logs_sum.items()}
        return total_loss / max(len(train_loader), 1), avg_logs

    def validate(self, val_loader):
        """验证/测试时只计算纯 MSE。"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for gy, fx in val_loader:
                gy = gy.to(self.device)
                fx = fx.to(self.device)

                fx_pred = self.model(gy)
                loss = self.criterion(fx_pred, fx)
                total_loss += loss.item()

        return total_loss / max(len(val_loader), 1)

    def train(self, train_loader, val_loader, num_epochs=100):
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f'epoch {epoch + 1} start')

            train_loss, train_logs = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)

            print(
                f'Epoch [{epoch + 1}/{num_epochs}], '
                f'Train Loss: {train_loss:.6f}, '
                f'Val MSE: {val_loss:.6f}, '
                f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}'
            )
            print(
                '  loss parts: '
                f'mse={train_logs.get("mse", 0):.6f}, '
                f'grad={train_logs.get("grad", 0):.6f}, '
                f'physics={train_logs.get("physics", 0):.6f}, '
                f'smooth={train_logs.get("smooth", 0):.6f}, '
                f'tv={train_logs.get("tv", 0):.6f}, '
                f'tikhonov={train_logs.get("tikhonov", 0):.6f}'
            )

    def predict(self, predict_loader, index):
        self.model.eval()

        predict_loader_list = list(predict_loader)
        if index < 0 or index >= len(predict_loader_list):
            raise IndexError(f'index={index} 超出范围，测试集大小={len(predict_loader_list)}')

        gy, fx = predict_loader_list[index]
        gy = gy.to(self.device)
        fx = fx.to(self.device)

        with torch.no_grad():
            fx_pred = self.model(gy)
            loss = self.criterion(fx_pred, fx)

        return fx_pred, loss.item()
