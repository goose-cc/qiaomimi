from torch.utils.data import Dataset
import os
import numpy as np
import torch


class PeakInversionDataset(Dataset):
    """峰值反演数据集。

    兼容两种格式：
    1. 旧格式：文件夹中包含 sample_00000.npz 等多个文件
    2. 新格式：单个 npz 文件，例如 ./data/train.npz，内部包含 fx / gy 等数组

    重要说明：
    - normalize_gy 默认 False，不再对每条 gy 做逐样本标准化，避免丢失幅值信息。
    - 如果确实需要旧版逐样本标准化，可在创建 Dataset 时传 normalize_gy=True。
    """

    def __init__(self, data_path, max_samples=None, normalize_gy=False):
        self.data_path = data_path
        self.max_samples = max_samples
        self.normalize_gy = normalize_gy
        self.mode = None
        self.gy_key = None
        self.x_all = None
        self.y_all = None

        if os.path.isfile(data_path):
            self.mode = "single_npz"
            self.data = np.load(data_path, allow_pickle=True)

            print(f"Loaded npz file: {data_path}")
            print(f"Keys in npz: {self.data.files}")

            if "gy_noisy" in self.data.files:
                self.gy_key = "gy_noisy"
            elif "gy" in self.data.files:
                self.gy_key = "gy"
            else:
                raise KeyError(
                    f"npz文件里找不到 gy_noisy 或 gy，当前字段有: {self.data.files}"
                )

            if "fx" not in self.data.files:
                raise KeyError(
                    f"npz文件里找不到 fx，当前字段有: {self.data.files}"
                )

            self.gy_all = self.data[self.gy_key].astype(np.float32)
            self.fx_all = self.data["fx"].astype(np.float32)

            # 如果是单条数据 [L]，扩展成 [1, L]
            if self.gy_all.ndim == 1:
                self.gy_all = self.gy_all[None, :]
            if self.fx_all.ndim == 1:
                self.fx_all = self.fx_all[None, :]

            print("raw gy shape:", self.gy_all.shape)
            print("raw fx shape:", self.fx_all.shape)

            if max_samples is not None:
                self.gy_all = self.gy_all[:max_samples]
                self.fx_all = self.fx_all[:max_samples]
                print(f"use first {max_samples} samples for: {data_path}")

            print("use gy shape:", self.gy_all.shape)
            print("use fx shape:", self.fx_all.shape)
            print("normalize_gy:", self.normalize_gy)

            if "x" in self.data.files:
                self.x_all = self.data["x"].astype(np.float32)
            if "y" in self.data.files:
                self.y_all = self.data["y"].astype(np.float32)

        elif os.path.isdir(data_path):
            self.mode = "folder"
            self.files = [f for f in os.listdir(data_path) if f.endswith(".npz")]
            self.files.sort()

            if max_samples is not None:
                self.files = self.files[:max_samples]

            print(f"Loaded folder: {data_path}, samples: {len(self.files)}")
            print("normalize_gy:", self.normalize_gy)

        else:
            raise FileNotFoundError(f"数据路径不存在: {data_path}")

    def __len__(self):
        if self.mode == "single_npz":
            return len(self.gy_all)
        return len(self.files)

    def _normalize_gy(self, gy):
        """是否对单条 gy 做逐样本标准化。

        默认不标准化，因为反问题中 gy 的幅值信息对恢复 fx 很重要。
        """
        gy = gy.astype(np.float32)

        if not self.normalize_gy:
            return gy

        return (gy - gy.mean()) / (gy.std() + 1e-8)

    def _load_one_from_folder(self, idx):
        data = np.load(os.path.join(self.data_path, self.files[idx]), allow_pickle=True)

        if "gy_noisy" in data.files:
            gy_key = "gy_noisy"
        elif "gy" in data.files:
            gy_key = "gy"
        else:
            raise KeyError(
                f"样本文件里找不到 gy_noisy 或 gy，当前字段有: {data.files}"
            )

        if "fx" not in data.files:
            raise KeyError(f"样本文件里找不到 fx，当前字段有: {data.files}")

        gy = data[gy_key].astype(np.float32)
        fx = data["fx"].astype(np.float32)
        return data, gy_key, gy, fx

    def __getitem__(self, idx):
        if self.mode == "single_npz":
            gy = self.gy_all[idx]
            fx = self.fx_all[idx]
        else:
            _, _, gy, fx = self._load_one_from_folder(idx)

        # 默认不做逐样本标准化；是否标准化由 normalize_gy 控制
        gy = self._normalize_gy(gy)
        fx = fx.astype(np.float32)

        gy = torch.from_numpy(gy).float()
        fx = torch.from_numpy(fx).float()

        # 保证形状是 [1, num_points]
        if gy.ndim == 1:
            gy = gy.unsqueeze(0)
        if fx.ndim == 1:
            fx = fx.unsqueeze(0)

        return gy, fx

    def __getx__(self, idx):
        if self.mode == "single_npz":
            if self.x_all is None:
                return np.linspace(0, 2, self.fx_all.shape[-1]).astype(np.float32)
            if self.x_all.ndim == 1:
                return self.x_all
            return self.x_all[idx]

        data, _, _, fx = self._load_one_from_folder(idx)
        if "x" in data.files:
            return data["x"].astype(np.float32)
        return np.linspace(0, 2, fx.shape[-1]).astype(np.float32)

    def __gety__(self, idx):
        if self.mode == "single_npz":
            if self.y_all is None:
                return np.linspace(3, 8, self.gy_all.shape[-1]).astype(np.float32)
            if self.y_all.ndim == 1:
                return self.y_all
            return self.y_all[idx]

        data, gy_key, gy, _ = self._load_one_from_folder(idx)
        if "y" in data.files:
            return data["y"].astype(np.float32)
        return np.linspace(3, 8, gy.shape[-1]).astype(np.float32)
