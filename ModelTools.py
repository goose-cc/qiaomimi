import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from PINet import PeakInversionCNN, PhysicsInformedNetwork
from UNetLike import UNetLikeModel

class ModelTools:
    """峰值反演模型训练器"""
    def __init__(self, config, model_type='cnn', LoadModel=False):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device is cuda")
        
        # 选择模型
        if model_type == 'cnn':
            self.model = PeakInversionCNN(
                input_length=config.num_y_points,
                output_length=config.num_x_points
            ).to(self.device)
            # print(self.model)
        elif model_type == 'unet':
            self.model = UNetLikeModel().to(self.device)
        elif model_type == 'pinn':
            self.model = PhysicsInformedNetwork(
                input_length=config.num_y_points,
                output_length=config.num_x_points
            ).to(self.device)

        self.model_type = model_type
        self.x = torch.linspace(0, 2, config.num_x_points)
        self.y = torch.linspace(2.1, 10, config.num_y_points)

        self.load_model(config.model_path) if LoadModel else None
        
        # 损失函数：MSE + 一阶梯度约束
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

    def load_model(self, model_path):
        """加载预训练模型权重"""
        print(f"尝试加载模型: {model_path}")
        try:
            # 加载模型权重
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("将使用随机初始化的模型")

    def gradient_loss(self, pred, target):
        """梯度匹配损失，使重建信号更平滑"""
        pred_grad = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
        target_grad = torch.abs(target[:, :, 1:] - target[:, :, :-1])
        return F.mse_loss(pred_grad, target_grad)

    def physics_loss(self, fx_pred, gy_noisy):
        """物理约束损失：根据预测 f(x) 计算 g(y) 并与观测的 g(y) 比较"""
        fx_pred = fx_pred.squeeze(1)  # [B, num_x]
        # x 和 y 在整个训练过程中都是固定的网格
        x = self.x.to(self.device)
        y = self.y.to(self.device)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        denominator = Y - X
        sign = torch.sign(denominator)
        sign = torch.where(sign == 0, torch.tensor(1.0, device=self.device), sign)
        denominator = torch.where(torch.abs(denominator) < 1e-6, sign * 1e-6, denominator)
        integrand = fx_pred.unsqueeze(1) / denominator.unsqueeze(0)
        gy_pred = torch.trapz(integrand, x, dim=-1)
        gy_pred = gy_pred.unsqueeze(1)
        return self.criterion(gy_pred, gy_noisy)
    
    def train_epoch(self, train_loader):
        '''
        训练一个epoch
        
        :self: 自身实例
        :param train_loader: 训练数据加载器
        '''
        self.model.train()      # 设置模型为训练模式
        total_loss = 0
        
        for gy_noisy, fx in train_loader:
            # 移动数据到 cuda 设备
            gy_noisy = gy_noisy.to(self.device)
            fx = fx.to(self.device)
            
            # 清空梯度，否则会累积错误
            self.optimizer.zero_grad()
            
            # 前向传播
            fx_pred = self.model(gy_noisy)
            
            # 计算损失
            mse_loss = self.criterion(fx_pred, fx)          # MSE损失
            grad_loss = self.gradient_loss(fx_pred, fx)     # 梯度损失
            if self.model_type == 'pinn':
                phys_loss = self.physics_loss(fx_pred, gy_noisy)
                loss = mse_loss + 0.1 * grad_loss + 0.05 * phys_loss
            else:
                loss = mse_loss + 0.1 * grad_loss               # 组合损失
            
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()                           # 更新参数
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()       # 设置模型为评估模式
        total_loss = 0          # 初始化总损失
        
        with torch.no_grad():   # 禁用梯度计算，不会生成计算图，节省资源加速计算
            for gy_noisy, fx in val_loader:
                # 移动数据到 cuda 设备
                gy_noisy = gy_noisy.to(self.device)
                fx = fx.to(self.device)
                
                # 前向传播
                fx_pred = self.model(gy_noisy)
                # 计算损失
                loss = self.criterion(fx_pred, fx)  # MSE损失
                # 累计验证集上的总损失
                total_loss += loss.item()
        # 平均损失
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, num_epochs=100):
        '''
        训练指定轮数的模型
        
        :param self: 自身实例
        :param train_loader: 训练数据加载器
        :param val_loader: 验证数据加载器
        :param num_epochs: 训练轮数
        '''
        
        # 初始化为正无穷大，确保第一个验证损失一定比它小
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f'epoch {epoch + 1} start')

            train_loss = self.train_epoch(train_loader) # 训练一个epoch，一个epoch要把所有训练数据都过一遍

            val_loss = self.validate(val_loader)        # 用验证集验证该epoch训练后的模型
            
            self.scheduler.step(val_loss)               # 调整学习率
            
            # 更新损失, 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), './model/best_model.pth')
            
            # if (epoch + 1) % 10 == 0:
            
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                    f'Train Loss: {train_loss:.6f}, '
                    f'Val Loss: {val_loss:.6f}, '
                    f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
    
    def predict(self, predict_loader, index):
        self.model.eval()       # 设置模型为评估模式
        
        predict_loader_list = list(predict_loader)
        gy_noisy, fx = predict_loader_list[index]
        gy_noisy = gy_noisy.to(self.device)
        fx = fx.to(self.device)
                
        # 前向传播
        fx_pred = self.model(gy_noisy)
        # 计算损失
        loss = self.criterion(fx_pred, fx)  # MSE损失
                
        return fx_pred, loss
