#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证分块打乱、无放回采样、轮间重排和断点可复现性。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from train_mc_parameter_pool_transformer_loss import ShuffledNoReplacementSampler


def close_memmap(array: np.memmap) -> None:
    handle = getattr(array, "_mmap", None)
    if handle is not None:
        handle.close()


def collect_cycle(
    sampler: ShuffledNoReplacementSampler,
    *,
    cycle: int,
    row_count: int,
    chunk_size: int,
) -> np.ndarray:
    chunks = []
    progress = 0
    while progress < row_count:
        take = min(chunk_size, row_count - progress)
        chunk = sampler.sample_chunk(
            cycle=cycle,
            progress=progress,
            chunk_size=take,
        )
        chunks.append(chunk[:, 0].astype(np.int64))
        progress += take
    return np.concatenate(chunks)


def main() -> None:
    row_count = 1000
    shuffle_block_size = 137
    chunk_size = 53

    with tempfile.TemporaryDirectory() as temporary_dir:
        data_path = Path(temporary_dir) / "parameters.dat"

        rows = np.zeros((row_count, 5), dtype=np.float32)
        rows[:, 0] = np.arange(row_count, dtype=np.float32)
        rows.tofile(data_path)

        pool = np.memmap(
            data_path,
            dtype=np.float32,
            mode="r",
            shape=(row_count, 5),
        )
        try:
            sampler = ShuffledNoReplacementSampler(
                pool=pool,
                usable_rows=row_count,
                block_size=shuffle_block_size,
                seed=20260721,
            )

            first_cycle = collect_cycle(
                sampler,
                cycle=0,
                row_count=row_count,
                chunk_size=chunk_size,
            )
            second_cycle = collect_cycle(
                sampler,
                cycle=1,
                row_count=row_count,
                chunk_size=chunk_size,
            )

            if first_cycle.size != row_count:
                raise AssertionError("第一轮样本数量错误")
            if np.unique(first_cycle).size != row_count:
                raise AssertionError("第一轮存在重复或遗漏，不满足无放回要求")
            if set(first_cycle.tolist()) != set(range(row_count)):
                raise AssertionError("第一轮没有完整覆盖参数池")
            if np.all(first_cycle == np.arange(row_count)):
                raise AssertionError("第一轮仍按原始顺序读取")
            if np.array_equal(first_cycle, second_cycle):
                raise AssertionError("不同 pool_cycle 没有重新打乱")

            # 模拟在任意进度恢复：同一 seed/cycle/progress 必须返回完全相同的数据。
            resume_progress = 417
            resume_size = 91
            expected = sampler.sample_chunk(
                cycle=3,
                progress=resume_progress,
                chunk_size=resume_size,
            )
            resumed_sampler = ShuffledNoReplacementSampler(
                pool=pool,
                usable_rows=row_count,
                block_size=shuffle_block_size,
                seed=20260721,
            )
            actual = resumed_sampler.sample_chunk(
                cycle=3,
                progress=resume_progress,
                chunk_size=resume_size,
            )
            if not np.array_equal(expected, actual):
                raise AssertionError("断点恢复后采样顺序不一致")

            print("SHUFFLE WITHOUT REPLACEMENT TEST PASSED")
            print(f"rows={row_count}, shuffle_block_size={shuffle_block_size}")
            print("first 12 sampled row ids:", first_cycle[:12].tolist())
            print("unique rows in cycle 0:", np.unique(first_cycle).size)
        finally:
            close_memmap(pool)


if __name__ == "__main__":
    main()
