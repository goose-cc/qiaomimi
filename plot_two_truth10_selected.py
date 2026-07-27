from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from TransformerInverse import InverseTransformer1D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 two_truth10 测试集中按 (m, gamma) 分组抽样，并批量生成预测图片。"
    )
    parser.add_argument(
        "--data_path",
        default="./data_two_truth10/test.npz",
        help="测试集 npz 路径。",
    )
    parser.add_argument(
        "--model_path",
        default="./model/two_truth10_transformer_pinn.pth",
        help="训练好的 Transformer 权重路径。",
    )
    parser.add_argument(
        "--samples_per_group",
        type=int,
        default=3,
        help="每种 (m, gamma) 参数组合抽取多少条噪声样本。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="抽样随机种子。",
    )
    parser.add_argument(
        "--output_dir",
        default="./picture/two_truth10_selected",
        help="图片和汇总文件输出目录。",
    )
    return parser.parse_args()


def load_state_dict_safely(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    # 兼容纯 state_dict，以及 {"state_dict": ...} / {"model_state_dict": ...} 格式。
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if not isinstance(checkpoint, dict):
        raise TypeError("权重文件不是可识别的 state_dict 格式。")
    return checkpoint


def filename_number(value: float) -> str:
    text = f"{value:g}".replace("-", "minus").replace(".", "p")
    return re.sub(r"[^0-9A-Za-z_]+", "_", text)


def main() -> None:
    args = parse_args()
    if args.samples_per_group <= 0:
        raise ValueError("--samples_per_group 必须大于 0")

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"找不到测试集: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型权重: {model_path}")

    data = np.load(data_path, allow_pickle=True)
    required = {"fx", "m", "gamma"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"测试集缺少字段: {missing}；现有字段: {data.files}")

    if "gy_noisy" in data.files:
        gy_key = "gy_noisy"
    elif "gy_clean" in data.files:
        gy_key = "gy_clean"
    elif "gy" in data.files:
        gy_key = "gy"
    else:
        raise KeyError("测试集中找不到 gy_noisy、gy_clean 或 gy")

    gy = data[gy_key].astype(np.float32)
    fx_true = data["fx"].astype(np.float32)
    m_values = data["m"].astype(np.float32)
    gamma_values = data["gamma"].astype(np.float32)

    if "x" in data.files:
        x = np.asarray(data["x"], dtype=np.float32)
        if x.ndim > 1:
            x = x[0]
    else:
        x = np.linspace(0.0, 2.0, fx_true.shape[-1], dtype=np.float32)

    if gy.ndim != 2 or fx_true.ndim != 2:
        raise ValueError(f"期望 gy/fx 形状为 [N, L]，实际是 {gy.shape} 和 {fx_true.shape}")

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

    pairs = np.column_stack([m_values, gamma_values])
    unique_pairs = np.unique(np.round(pairs, decimals=6), axis=0)
    rng = np.random.default_rng(args.seed)

    selected_indices: list[int] = []
    for m_target, gamma_target in unique_pairs:
        candidates = np.flatnonzero(
            np.isclose(m_values, m_target) & np.isclose(gamma_values, gamma_target)
        )
        count = min(args.samples_per_group, len(candidates))
        chosen = rng.choice(candidates, size=count, replace=False)
        selected_indices.extend(int(index) for index in chosen)

    selected_indices.sort(key=lambda idx: (float(m_values[idx]), float(gamma_values[idx]), idx))

    gy_batch = torch.from_numpy(gy[selected_indices]).unsqueeze(1).to(device)
    with torch.no_grad():
        fx_pred = model(gy_batch).squeeze(1).cpu().numpy()

    true_batch = fx_true[selected_indices]
    mse_values = np.mean((fx_pred - true_batch) ** 2, axis=1)

    summary_rows: list[dict[str, object]] = []
    for position, index in enumerate(selected_indices):
        m_value = float(m_values[index])
        gamma_value = float(gamma_values[index])
        mse = float(mse_values[position])

        figure_path = output_dir / (
            f"prediction_m{filename_number(m_value)}_"
            f"gamma{filename_number(gamma_value)}_index{index}.png"
        )

        plt.figure(figsize=(8, 5))
        plt.plot(x, true_batch[position], label="True f(x)")
        plt.plot(x, fx_pred[position], linestyle="--", label="Predicted f(x)")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(
            f"m={m_value:g}, gamma={gamma_value:g}, "
            f"test index={index}, MSE={mse:.6f}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_path, dpi=200)
        plt.close()

        summary_rows.append(
            {
                "test_index": index,
                "m": m_value,
                "gamma": gamma_value,
                "mse": mse,
                "image": str(figure_path),
            }
        )

    # 再保存一张总览图，便于报告中同时展示多个样本。
    columns = min(3, max(1, len(selected_indices)))
    rows = int(np.ceil(len(selected_indices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4 * rows), squeeze=False)
    for axis in axes.flat:
        axis.set_visible(False)

    for position, index in enumerate(selected_indices):
        axis = axes.flat[position]
        axis.set_visible(True)
        axis.plot(x, true_batch[position], label="True f(x)")
        axis.plot(x, fx_pred[position], linestyle="--", label="Predicted f(x)")
        axis.set_title(
            f"m={float(m_values[index]):g}, gamma={float(gamma_values[index]):g}\n"
            f"index={index}, MSE={float(mse_values[position]):.6f}"
        )
        axis.set_xlabel("x")
        axis.set_ylabel("f(x)")
        axis.legend()

    fig.tight_layout()
    overview_path = output_dir / "selected_predictions_overview.png"
    fig.savefig(overview_path, dpi=200)
    plt.close(fig)

    csv_path = output_dir / "selected_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["test_index", "m", "gamma", "mse", "image"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    npz_path = output_dir / "selected_predictions.npz"
    np.savez_compressed(
        npz_path,
        selected_indices=np.asarray(selected_indices, dtype=np.int32),
        m=m_values[selected_indices],
        gamma=gamma_values[selected_indices],
        x=x,
        gy=gy[selected_indices],
        fx_true=true_batch,
        fx_pred=fx_pred,
        mse=mse_values.astype(np.float32),
        gy_key=np.array(gy_key),
        model_path=np.array(str(model_path)),
    )

    print(f"device: {device}")
    print(f"输入字段: {gy_key}")
    print(f"发现的参数组合: {unique_pairs.tolist()}")
    print("抽取结果:")
    for row in summary_rows:
        print(
            f"  index={row['test_index']}, m={row['m']:g}, "
            f"gamma={row['gamma']:g}, MSE={row['mse']:.6f}"
        )
    print(f"总览图片: {overview_path}")
    print(f"单图目录: {output_dir}")
    print(f"汇总表: {csv_path}")
    print(f"预测数组: {npz_path}")


if __name__ == "__main__":
    main()
