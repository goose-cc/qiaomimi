#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速验证全参数池随机采样函数的范围、顺序和可复现性。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from train_mc_parameter_pool_transformer_loss import sample_random_parameter_chunk


def main() -> None:
    row_count = 1000
    chunk_size = 128

    with tempfile.TemporaryDirectory() as temporary_dir:
        data_path = Path(temporary_dir) / "parameters.dat"

        # 第一列写入唯一行号，便于检查抽到了哪些行。
        rows = np.zeros((row_count, 5), dtype=np.float32)
        rows[:, 0] = np.arange(row_count, dtype=np.float32)
        rows.tofile(data_path)

        pool = np.memmap(
            data_path,
            dtype=np.float32,
            mode="r",
            shape=(row_count, 5),
        )

        np.random.seed(20260721)
        first = sample_random_parameter_chunk(pool, row_count, chunk_size)

        np.random.seed(20260721)
        repeated = sample_random_parameter_chunk(pool, row_count, chunk_size)

        if not np.array_equal(first, repeated):
            raise AssertionError("相同随机种子没有得到相同样本")
        if first.shape != (chunk_size, 5):
            raise AssertionError(f"采样形状错误: {first.shape}")
        if np.any(first[:, 0] < 0) or np.any(first[:, 0] >= row_count):
            raise AssertionError("采样索引超出参数池范围")
        if np.all(np.diff(first[:, 0]) >= 0):
            raise AssertionError("样本仍按参数池行号单调排列，随机顺序恢复失败")
        if np.unique(first[:, 0]).size < chunk_size * 0.85:
            raise AssertionError("小规模测试中重复样本异常偏多")

        print("RANDOM SAMPLING TEST PASSED")
        print(f"rows={row_count}, chunk_size={chunk_size}")
        print("first 12 sampled row ids:", first[:12, 0].astype(np.int64).tolist())
        # Windows 不允许删除仍被 np.memmap 占用的文件。
        # 在TemporaryDirectory 清理前显式关闭底层 mmap 文件句柄。
        mmap_handle = getattr(pool, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()
        del pool


if __name__ == "__main__":
    main()
