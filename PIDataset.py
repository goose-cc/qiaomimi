from torch.utils.data import Dataset
import os
import numpy as np
import torch

class PeakInversionDataset(Dataset):
    """峰值反演数据集"""
    def __init__(self, data_dir):
        '''
        构造函数
        
        :param self: 说明
        :param data_dir: 说明
        '''
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        '''
        获取数据项
        
        :param idx: 数据项索引

        :return:    返回数据, 标签
        '''
        data = np.load(os.path.join(self.data_dir, self.files[idx]))
        
        # 输入: 带噪声的g(y) [num_y_points]
        # 输出: 原始的f(x) [num_x_points]
        gy_noisy = data['gy_noisy'].astype(np.float32)
        fx = data['fx'].astype(np.float32)
        
        # 标准化输入，如果训练时就标准化了数据，那么预测时也需要标准化数据再输入
        gy_noisy = (gy_noisy - gy_noisy.mean()) / (gy_noisy.std() + 1e-8)
        
        # 转换为PyTorch张量并增加通道维度
        gy_noisy = torch.FloatTensor(gy_noisy).unsqueeze(0)  # [1, num_y_points]
        fx = torch.FloatTensor(fx).unsqueeze(0)  # [1, num_x_points]
        
        return gy_noisy, fx
    
    def __getx__(self, idx):
        '''
        获取x的分布
        :index: 数据项索引
        :return: x的分布
        '''
        data = np.load(os.path.join(self.data_dir, self.files[idx]))
        
        x = data['x'].astype(np.float32)

        return x