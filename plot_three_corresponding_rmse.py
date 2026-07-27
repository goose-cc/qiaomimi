from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from TransformerInverse import InverseTransformer1D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从 two_truth10 测试集中抽取 3 个对应样本：每个样本分别绘制 "
            "True/Predicted f(x) 和 Clean/10%-noisy g(y)，指标使用 RMSE。"
        )
    )
    parser.add_argument("--data_path", default="./data_two_truth10/test.npz")
    parser.add_argument(
        "--model_path", default="./model/two_truth10_transformer_pinn.pth"
    )
    parser.add_argument(
        "--output_dir", default="./picture/three_corresponding_rmse"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--mode",
        choices=["balanced", "random"],
        default="balanced",
        help="balanced 保证两种真值都至少出现一次；random 完全随机抽 3 条。",
    )
    return parser.parse_args()


def load_state_dict_safely(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError("权重文件不是可识别的 state_dict 格式。")
    return checkpoint


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def choose_three_indices(
    m_values: np.ndarray,
    gamma_values: np.ndarray,
    seed: int,
    mode: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = len(m_values)
    if total < 3:
        raise ValueError("测试集少于 3 条，无法抽取 3 个样本。")

    if mode == "random":
        return rng.choice(total, size=3, replace=False).astype(np.int64)

    pairs = np.round(np.column_stack([m_values, gamma_values]), decimals=6)
    unique_pairs = np.unique(pairs, axis=0)
    if len(unique_pairs) < 2:
        return rng.choice(total, size=3, replace=False).astype(np.int64)

    # 每种真值先抽一条，再从剩余样本中抽第三条。
    selected: list[int] = []
    for target in unique_pairs[:2]:
        candidates = np.flatnonzero(np.all(pairs == target, axis=1))
        selected.append(int(rng.choice(candidates)))

    remaining = np.setdiff1d(np.arange(total), np.asarray(selected), assume_unique=False)
    selected.append(int(rng.choice(remaining)))
    return np.asarray(selected, dtype=np.int64)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"找不到测试数据：{data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型权重：{model_path}")

    data = np.load(data_path, allow_pickle=True)
    required = {"fx", "gy_clean", "gy_noisy", "m", "gamma"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"测试数据缺少字段：{missing}\n现有字段：{data.files}")

    fx_true = np.asarray(data["fx"], dtype=np.float32)
    gy_clean = np.asarray(data["gy_clean"], dtype=np.float32)
    gy_noisy = np.asarray(data["gy_noisy"], dtype=np.float32)
    m_values = np.asarray(data["m"], dtype=np.float32)
    gamma_values = np.asarray(data["gamma"], dtype=np.float32)

    if "x" in data.files:
        x = np.asarray(data["x"], dtype=np.float32).reshape(-1)
    else:
        x = np.linspace(0.0, 2.0, fx_true.shape[1], dtype=np.float32)

    if "y" in data.files:
        y = np.asarray(data["y"], dtype=np.float32).reshape(-1)
    else:
        y = np.arange(gy_clean.shape[1], dtype=np.float32)

    indices = choose_three_indices(
        m_values=m_values,
        gamma_values=gamma_values,
        seed=args.seed,
        mode=args.mode,
    )

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InverseTransformer1D(
        input_length=cfg.num_y_points,
        output_length=cfg.num_x_points,
        d_model=cfg.transformer_d_model,
        nhead=cfg.transformer_nhead,
        num_encoder_layers=cfg.transformer_num_layers,
        num_decoder_layers=cfg.transformer_num_layers,
        dim_feedforward=cfg.transformer_dim_feedforward,
        dropout=cfg.transformer_dropout,
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        y_min=cfg.y_min,
        y_max=cfg.y_max,
    ).to(device)
    model.load_state_dict(load_state_dict_safely(model_path, device))
    model.eval()

    batch = torch.from_numpy(gy_noisy[indices]).unsqueeze(1).to(device)
    with torch.no_grad():
        fx_pred = model(batch).squeeze(1).cpu().numpy()

    rows: list[dict[str, object]] = []
    overview, axes = plt.subplots(3, 2, figsize=(14, 14), squeeze=False)

    print("\n抽取的 3 组对应样本（同一 index 的 F 与 G 一一对应）：")
    for order, (idx, pred) in enumerate(zip(indices, fx_pred), start=1):
        true_f = fx_true[idx]
        clean_g = gy_clean[idx]
        noisy_g = gy_noisy[idx]
        m = float(m_values[idx])
        gamma = float(gamma_values[idx])
        peak_center = m**2  # 当前数据生成公式中的峰中心

        f_rmse = rmse(pred, true_f)
        g_rmse = rmse(noisy_g, clean_g)
        clean_rms = float(np.sqrt(np.mean(clean_g**2)))
        relative_g_rmse = 100.0 * g_rmse / max(clean_rms, 1e-12)

        # 图 1：真值 F 和预测 F
        fig_f, ax_f = plt.subplots(figsize=(8, 5))
        ax_f.plot(x, true_f, label="True f(x)")
        ax_f.plot(x, pred, "--", label="Predicted f(x)")
        ax_f.axvline(
            peak_center,
            linestyle=":",
            linewidth=1.2,
            label=f"Peak center m^2={peak_center:.3f}",
        )
        ax_f.set_xlabel("x")
        ax_f.set_ylabel("f(x)")
        ax_f.set_title(
            f"Sample {order} | index={idx} | m={m:.3f}, gamma={gamma:.3f}\n"
            f"F RMSE={f_rmse:.6f}"
        )
        ax_f.legend()
        ax_f.grid(alpha=0.25)
        fig_f.tight_layout()
        f_path = output_dir / f"sample_{order}_index{idx}_F_true_vs_pred_RMSE.png"
        fig_f.savefig(f_path, dpi=220)
        plt.close(fig_f)

        # 图 2：未加噪 G 和 10% 加噪 G
        fig_g, ax_g = plt.subplots(figsize=(8, 5))
        ax_g.plot(y, clean_g, label="Clean g(y)")
        ax_g.plot(y, noisy_g, "--", label="10% noisy g(y)")
        ax_g.set_xlabel("y")
        ax_g.set_ylabel("g(y)")
        ax_g.set_title(
            f"Sample {order} | index={idx} | m={m:.3f}, gamma={gamma:.3f}\n"
            f"G noise RMSE={g_rmse:.6f} (relative={relative_g_rmse:.2f}%)"
        )
        ax_g.legend()
        ax_g.grid(alpha=0.25)
        fig_g.tight_layout()
        g_path = output_dir / f"sample_{order}_index{idx}_G_clean_vs_10pct_noise_RMSE.png"
        fig_g.savefig(g_path, dpi=220)
        plt.close(fig_g)

        # 汇总图：每行一组对应样本，左 F、右 G。
        ax_left = axes[order - 1, 0]
        ax_left.plot(x, true_f, label="True f(x)")
        ax_left.plot(x, pred, "--", label="Predicted f(x)")
        ax_left.axvline(peak_center, linestyle=":", linewidth=1.0, label=f"m^2={peak_center:.3f}")
        ax_left.set_title(
            f"Sample {order}: F | m={m:.3f}, gamma={gamma:.3f}\nRMSE={f_rmse:.6f}"
        )
        ax_left.set_xlabel("x")
        ax_left.set_ylabel("f(x)")
        ax_left.legend()
        ax_left.grid(alpha=0.25)

        ax_right = axes[order - 1, 1]
        ax_right.plot(y, clean_g, label="Clean g(y)")
        ax_right.plot(y, noisy_g, "--", label="10% noisy g(y)")
        ax_right.set_title(
            f"Sample {order}: G | same index={idx}\n"
            f"RMSE={g_rmse:.6f}, relative={relative_g_rmse:.2f}%"
        )
        ax_right.set_xlabel("y")
        ax_right.set_ylabel("g(y)")
        ax_right.legend()
        ax_right.grid(alpha=0.25)

        rows.append(
            {
                "sample_order": order,
                "test_index": int(idx),
                "m": m,
                "gamma": gamma,
                "peak_center_m_squared": peak_center,
                "f_rmse": f_rmse,
                "g_noise_rmse": g_rmse,
                "g_relative_rmse_percent": relative_g_rmse,
                "f_figure": str(f_path),
                "g_figure": str(g_path),
            }
        )

        print(
            f"  第 {order} 组: index={idx}, m={m:.3f}, gamma={gamma:.3f}, "
            f"峰中心 m^2={peak_center:.3f}, F_RMSE={f_rmse:.6f}, "
            f"G噪声_RMSE={g_rmse:.6f}, 相对RMSE={relative_g_rmse:.2f}%"
        )

    overview.suptitle(
        "Three corresponding samples: F reconstruction (left) and G noise comparison (right)",
        fontsize=14,
    )
    overview.tight_layout(rect=[0, 0, 1, 0.975])
    overview_path = output_dir / "overview_three_corresponding_samples_RMSE.png"
    overview.savefig(overview_path, dpi=220)
    plt.close(overview)

    csv_path = output_dir / "three_corresponding_samples_RMSE.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    npz_path = output_dir / "three_corresponding_samples_RMSE.npz"
    np.savez_compressed(
        npz_path,
        selected_indices=indices.astype(np.int32),
        m=m_values[indices],
        gamma=gamma_values[indices],
        fx_true=fx_true[indices],
        fx_pred=fx_pred.astype(np.float32),
        gy_clean=gy_clean[indices],
        gy_noisy=gy_noisy[indices],
        f_rmse=np.asarray([row["f_rmse"] for row in rows], dtype=np.float32),
        g_noise_rmse=np.asarray([row["g_noise_rmse"] for row in rows], dtype=np.float32),
        g_relative_rmse_percent=np.asarray(
            [row["g_relative_rmse_percent"] for row in rows], dtype=np.float32
        ),
    )

    print(f"\n汇总图：{overview_path}")
    print(f"指标表：{csv_path}")
    print(f"数据文件：{npz_path}")
    print("每组都有两张单独图片：F 真值/预测图 + G 未加噪/10%加噪图。")


if __name__ == "__main__":
    main()
