from torch.utils.data import Dataset
import numpy as np
import torch


class PeakInversionDataset(Dataset):
    """
    峰值反演数据集
    读取 data/train.npz、data/val.npz、data/test.npz
    输入: g(y)
    输出: f(x)
    """

    def __init__(self, data_path):
        data = np.load(data_path)

        self.fx = data["fx"].astype(np.float32)

        if "gy_noisy" in data:
            self.gy = data["gy_noisy"].astype(np.float32)
        elif "gy" in data:
            self.gy = data["gy"].astype(np.float32)
        else:
            raise KeyError("数据文件中没有找到 gy 或 gy_noisy")

        self.x = data["x"].astype(np.float32) if "x" in data else None

        assert len(self.fx) == len(self.gy), "fx 和 gy 样本数量不一致"

    def __len__(self):
        return len(self.fx)

    def __getitem__(self, idx):
        gy = self.gy[idx]
        fx = self.fx[idx]

        gy = torch.tensor(gy, dtype=torch.float32).unsqueeze(0)
        fx = torch.tensor(fx, dtype=torch.float32).unsqueeze(0)

        return gy, fx

    def __getx__(self, idx=0):
        if self.x is None:
            return None
        return self.x


PIDataset = PeakInversionDataset
