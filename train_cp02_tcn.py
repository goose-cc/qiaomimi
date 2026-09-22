#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CP02 TCN baseline.

This file is a TCN replacement experiment:
- keep the same data
- keep the same train/val/test split
- keep the same target parameters
- replace MLP feature extractor with 1D temporal convolution blocks

Input:
    [batch, 239]

TCN expects:
    [batch, channel=1, length=239]
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class CP02TCN(nn.Module):
    def __init__(self, out_dim=3):
        super().__init__()
        self.feature = nn.Sequential(
            TemporalBlock(1, 32, dilation=1),
            TemporalBlock(32, 64, dilation=2),
            TemporalBlock(64, 128, dilation=4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.feature(x)
        return self.head(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)

    print("TCN model initialized.")
    print("Please connect this backbone to the existing CP02 Dataset loader.")
    print("Dataset directory:", args.data_dir)

    model = CP02TCN().to(args.device)
    print(model)


if __name__ == "__main__":
    main()
