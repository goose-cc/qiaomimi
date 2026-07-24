"""
生成“两组固定真值 + 每组 10000 条 10% 高斯白噪声”的 Model 3 数据集。

老师要求：
真值 1: a1=a2=1, m=0.8, gamma=0.2
真值 2: a1=a2=1, m=0.4, gamma=0.2
每个真值生成 10000 条独立的 10% 白噪声观测，共 20000 条训练数据。

数据流程：params -> f(x) -> integral -> g_clean(y) -> Gaussian white noise -> g_noisy(y)

注意：按老师当前要求，train.npz 使用全部 20000 条；val/test 是从这 20000 条中
固定随机抽取的子集，因此它评估的是训练分布内重建，不是严格的独立泛化测试。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from config import Config


PARAM_NAMES = np.array(["a1", "a2", "m", "gamma"])
TRUTH_NAMES = np.array(["truth_m0p8_gamma0p2", "truth_m0p4_gamma0p2"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate two fixed-truth Model 3 datasets with 10% Gaussian white noise."
    )
    parser.add_argument(
        "--output_dir",
        default="./data_two_truth10",
        help="输出目录，默认 ./data_two_truth10",
    )
    parser.add_argument(
        "--samples_per_truth",
        type=int,
        default=10000,
        help="每个真值的噪声样本数，默认 10000",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.10,
        help="相对于每条 g_clean RMS 的噪声比例，默认 0.10",
    )
    parser.add_argument("--val_size", type=int, default=2000)
    parser.add_argument("--test_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的 train/val/test.npz",
    )
    return parser.parse_args()


def build_grids_and_kernel(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.num_x_points, dtype=np.float32)
    y = np.linspace(cfg.y_min, cfg.y_max, cfg.num_y_points, dtype=np.float32)

    dx = (x[-1] - x[0]) / (len(x) - 1)
    weights = np.full_like(x, dx, dtype=np.float32)
    weights[0] *= 0.5
    weights[-1] *= 0.5

    kernel = weights[None, :] / (y[:, None] - x[None, :])
    return x, y, kernel.astype(np.float32)


def model3_fx(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """与原项目 data_generate_model3_param_sweep.py 保持相同的 Model 3 公式。"""
    x_grid = x[None, :]
    a1 = params[:, 0:1]
    a2 = params[:, 1:2]
    m = params[:, 2:3]
    gamma = params[:, 3:4]

    width = a1 * m * gamma
    peak = (1.0 / np.pi) * width / ((x_grid - m**2) ** 2 + width**2)
    background = a2 * x_grid / 5.0
    return (peak + background).astype(np.float32)


def add_relative_gaussian_white_noise(
    gy_clean: np.ndarray,
    noise_level: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对每条 g_clean 独立加零均值高斯白噪声：
        sigma_i = noise_level * RMS(g_clean_i)
        noise_i ~ N(0, sigma_i^2)
    """
    rms = np.sqrt(np.mean(gy_clean**2, axis=1, keepdims=True))
    sigma = np.maximum(rms, 1e-12) * np.float32(noise_level)
    standard_noise = rng.standard_normal(gy_clean.shape).astype(np.float32)
    noise = standard_noise * sigma.astype(np.float32)
    gy_noisy = gy_clean + noise
    return gy_noisy.astype(np.float32), noise.astype(np.float32)


def build_full_dataset(
    cfg: Config,
    samples_per_truth: int,
    noise_level: float,
    seed: int,
) -> dict[str, np.ndarray]:
    if samples_per_truth <= 0:
        raise ValueError("samples_per_truth 必须大于 0")
    if noise_level < 0:
        raise ValueError("noise_level 不能为负数")

    x, y, kernel = build_grids_and_kernel(cfg)

    truth_params = np.array(
        [
            [1.0, 1.0, 0.8, 0.2],
            [1.0, 1.0, 0.4, 0.2],
        ],
        dtype=np.float32,
    )

    params = np.repeat(truth_params, samples_per_truth, axis=0)
    truth_id = np.repeat(np.arange(2, dtype=np.int8), samples_per_truth)

    # 两个固定真值只需各计算一次 clean f(x) 和 g(y)，随后复制。
    fx_truths = model3_fx(x, truth_params)
    gy_truths = (fx_truths @ kernel.T).astype(np.float32)
    fx = fx_truths[truth_id].copy()
    gy_clean = gy_truths[truth_id].copy()

    rng = np.random.default_rng(seed)
    gy_noisy, noise = add_relative_gaussian_white_noise(
        gy_clean=gy_clean,
        noise_level=noise_level,
        rng=rng,
    )

    # 打乱两类样本，避免前 10000 条全是同一真值。
    order = rng.permutation(len(params))
    params = params[order]
    truth_id = truth_id[order]
    fx = fx[order]
    gy_clean = gy_clean[order]
    gy_noisy = gy_noisy[order]
    noise = noise[order]

    measured_noise_ratio = np.sqrt(np.mean(noise**2, axis=1)) / np.maximum(
        np.sqrt(np.mean(gy_clean**2, axis=1)), 1e-12
    )

    return {
        "x": x,
        "y": y,
        "fx": fx,
        "gy_clean": gy_clean,
        "gy_noisy": gy_noisy,
        "noise": noise,
        "a1": params[:, 0],
        "a2": params[:, 1],
        "m": params[:, 2],
        "gamma": params[:, 3],
        "params": params,
        "param_names": PARAM_NAMES,
        "truth_id": truth_id,
        "truth_names": TRUTH_NAMES,
        "truth_params": truth_params,
        "noise_level": np.float32(noise_level),
        "noise_definition": np.array("sigma = noise_level * RMS(g_clean), Gaussian N(0,sigma^2)"),
        "measured_noise_ratio": measured_noise_ratio.astype(np.float32),
        "samples_per_truth": np.int32(samples_per_truth),
        "random_seed": np.int64(seed),
        "source": np.array("model3_two_fixed_truths_10percent"),
    }


def take_subset(dataset: dict[str, np.ndarray], indices: np.ndarray, role: str) -> dict[str, np.ndarray]:
    total = len(dataset["fx"])
    result: dict[str, np.ndarray] = {}

    for key, value in dataset.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == total:
            result[key] = value[indices]
        else:
            result[key] = value

    result["subset_indices_in_train"] = indices.astype(np.int32)
    result["split_role"] = np.array(role)
    result["is_independent_split"] = np.array(False)
    return result


def save_npz(path: Path, dataset: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **dataset)
    ids, counts = np.unique(dataset["truth_id"], return_counts=True)
    class_counts = {int(i): int(c) for i, c in zip(ids, counts)}
    print(f"Saved: {path}")
    print(f"  samples: {len(dataset['fx'])}")
    print(f"  truth counts: {class_counts}")
    print(f"  fx shape: {dataset['fx'].shape}")
    print(f"  gy_noisy shape: {dataset['gy_noisy'].shape}")


def main() -> None:
    args = parse_args()
    cfg = Config()
    seed = cfg.random_seed if args.seed is None else args.seed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train.npz",
        "val": output_dir / "val.npz",
        "test": output_dir / "test.npz",
    }

    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "以下文件已存在，为避免覆盖已停止：\n"
            + "\n".join(existing)
            + "\n需要覆盖时加 --overwrite"
        )

    dataset = build_full_dataset(
        cfg=cfg,
        samples_per_truth=args.samples_per_truth,
        noise_level=args.noise_level,
        seed=seed,
    )
    total = len(dataset["fx"])

    if args.val_size <= 0 or args.test_size <= 0:
        raise ValueError("val_size 和 test_size 必须大于 0")
    if args.val_size > total or args.test_size > total:
        raise ValueError("val_size/test_size 不能超过训练集总数")

    # train 按老师要求保留全部 20000 条。
    train_data = dict(dataset)
    train_data["split_role"] = np.array("train_all_20000")
    train_data["is_independent_split"] = np.array(False)

    # val/test 按固定随机种子从训练样本中抽取；二者互不重叠。
    split_rng = np.random.default_rng(seed + 1)
    subset_order = split_rng.permutation(total)
    if args.val_size + args.test_size > total:
        raise ValueError("val_size + test_size 不能超过总样本数，需保证二者互不重叠")
    val_indices = subset_order[: args.val_size]
    test_indices = subset_order[args.val_size : args.val_size + args.test_size]
    val_data = take_subset(dataset, val_indices, "validation_subset_of_train")
    test_data = take_subset(dataset, test_indices, "test_subset_of_train")

    save_npz(paths["train"], train_data)
    save_npz(paths["val"], val_data)
    save_npz(paths["test"], test_data)

    mean_ratio = float(np.mean(dataset["measured_noise_ratio"]))
    print("\n全部生成完成。")
    print("真值 1: a1=a2=1, m=0.8, gamma=0.2, 10000 条")
    print("真值 2: a1=a2=1, m=0.4, gamma=0.2, 10000 条")
    print(f"噪声: zero-mean Gaussian white noise, target={args.noise_level:.1%}")
    print(f"实测平均 RMS 噪声比例: {mean_ratio:.4%}")
    print("train 使用全部样本；val/test 是 train 的固定随机子集。")
    print("训练命令使用: --exp two_truth10")


if __name__ == "__main__":
    main()
