from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from TransformerInverse import InverseTransformer1D
from mc_inverse_loss import MonteCarloInverseLoss
from train_mc_parameter_pool_transformer_loss import OnlinePhysics, normalize_prediction_shape


PARAMETER_NAMES = ("a1", "a2", "a3", "m", "gamma")
PARAMETER_COLUMNS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a trained Transformer on a fixed Monte-Carlo parameter pool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--validation-pool-dir",
        required=True,
        help="独立验证参数池目录，里面应有 parameters.dat 和 metadata.json",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="训练检查点目录，里面应有 latest_checkpoint.pth",
    )
    parser.add_argument(
        "--weights",
        choices=("best", "latest"),
        default="best",
        help="best 使用 best_model.pth；latest 使用 latest_checkpoint.pth 中的模型权重",
    )
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help="验证噪声比例；不填时沿用训练 checkpoint 中的 noise_level",
    )
    parser.add_argument(
        "--output-dir",
        default="./validation_results",
        help="指标、CSV 和图片输出目录",
    )
    parser.add_argument("--plot-count", type=int, default=12)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 torch.cuda.is_available() 为 False。")
    return torch.device(name)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    return value if isinstance(value, dict) else {}


def open_parameter_pool(pool_dir: Path) -> Tuple[np.memmap, int, Dict[str, Any]]:
    data_path = pool_dir / "parameters.dat"
    metadata_path = pool_dir / "metadata.json"
    if not data_path.exists():
        raise FileNotFoundError(f"找不到验证参数池：{data_path}")

    row_bytes = PARAMETER_COLUMNS * np.dtype(np.float32).itemsize
    file_size = data_path.stat().st_size
    if file_size <= 0 or file_size % row_bytes != 0:
        raise RuntimeError(f"parameters.dat 文件大小异常：{file_size} 字节")

    file_rows = file_size // row_bytes
    metadata = load_json(metadata_path)
    generated_rows = metadata.get("generated_truths", file_rows)
    try:
        usable_rows = min(file_rows, int(generated_rows))
    except (TypeError, ValueError):
        usable_rows = file_rows

    if usable_rows <= 0:
        raise RuntimeError("验证参数池中没有可用参数。")

    pool = np.memmap(
        data_path,
        dtype=np.float32,
        mode="r",
        shape=(int(file_rows), PARAMETER_COLUMNS),
    )
    return pool, int(usable_rows), metadata


def build_model(saved_args: Dict[str, Any], device: torch.device) -> InverseTransformer1D:
    if saved_args.get("model_type", "transformer") != "transformer":
        raise RuntimeError("此验证脚本当前只用于 Transformer checkpoint。")

    model = InverseTransformer1D(
        input_length=int(saved_args.get("input_points", 100)),
        output_length=int(saved_args.get("output_points", 100)),
        d_model=int(saved_args.get("transformer_d_model", 64)),
        nhead=int(saved_args.get("transformer_nhead", 4)),
        num_encoder_layers=int(saved_args.get("transformer_num_layers", 3)),
        num_decoder_layers=int(saved_args.get("transformer_num_layers", 3)),
        dim_feedforward=int(saved_args.get("transformer_dim_feedforward", 128)),
        dropout=float(saved_args.get("transformer_dropout", 0.1)),
        x_min=float(saved_args.get("s_min", 0.1764)),
        x_max=float(saved_args.get("s_max", 6.0)),
        y_min=float(saved_args.get("q2_min", -100.0)),
        y_max=float(saved_args.get("q2_max", -6.0)),
    )
    return model.to(device)


def load_model_and_config(
    checkpoint_dir: Path,
    weights_choice: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    latest_path = checkpoint_dir / "latest_checkpoint.pth"
    if not latest_path.exists():
        raise FileNotFoundError(f"找不到：{latest_path}")

    try:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(latest_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("latest_checkpoint.pth 格式不符合训练脚本保存格式。")

    saved_args = dict(checkpoint.get("args", {}))
    model = build_model(saved_args, device)

    if weights_choice == "latest":
        state_dict = checkpoint["model_state_dict"]
        weights_path = latest_path
    else:
        best_path = checkpoint_dir / "best_model.pth"
        if not best_path.exists():
            raise FileNotFoundError(f"找不到：{best_path}")
        try:
            state_dict = torch.load(best_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(best_path, map_location="cpu")
        weights_path = best_path

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    checkpoint_info = {
        "weights_path": str(weights_path.resolve()),
        "latest_global_step": int(checkpoint.get("global_step", 0)),
        "latest_pool_cycle": int(checkpoint.get("pool_cycle", 0)),
        "latest_next_parameter_id": int(checkpoint.get("next_parameter_id", 0)),
        "saved_model_class": checkpoint.get("model_class", ""),
    }
    return model, saved_args, checkpoint_info


def set_fixed_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    numerator = torch.linalg.vector_norm(pred - target, dim=1)
    denominator = torch.linalg.vector_norm(target, dim=1).clamp_min(eps)
    return numerator / denominator


def summarize_vector(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def save_curve_plot(
    x: np.ndarray,
    curves: List[Tuple[str, np.ndarray]],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    for label, y in curves:
        ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples 必须大于 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")
    if args.plot_count < 0:
        raise ValueError("--plot-count 不能为负数")

    set_fixed_seed(args.seed)
    device = choose_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    validation_pool_dir = Path(args.validation_pool_dir)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    model, saved_args, checkpoint_info = load_model_and_config(
        checkpoint_dir,
        args.weights,
        device,
    )

    # 验证必须沿用训练时的物理网格、缩放和模型结构。
    validation_args = dict(saved_args)
    validation_args["noise_level"] = (
        float(args.noise_level)
        if args.noise_level is not None
        else float(saved_args.get("noise_level", 0.09))
    )
    validation_args["physics_dtype"] = saved_args.get("physics_dtype", "float32")
    physics = OnlinePhysics(SimpleNamespace(**validation_args), device)

    criterion = MonteCarloInverseLoss(
        s_min=float(saved_args.get("s_min", 0.1764)),
        s_max=float(saved_args.get("s_max", 6.0)),
        q2_min=float(saved_args.get("q2_min", -100.0)),
        q2_max=float(saved_args.get("q2_max", -6.0)),
        output_points=int(saved_args.get("output_points", 100)),
        input_points=int(saved_args.get("input_points", 100)),
        profile="pinn",
        normalization="relative",
        lambda_grad=0.0,
        lambda_physics=0.0,
    ).to(device)

    pool, usable_rows, metadata = open_parameter_pool(validation_pool_dir)
    sample_count = min(int(args.num_samples), usable_rows)
    rng = np.random.default_rng(args.seed)
    noise_rng = np.random.default_rng(args.seed + 1)
    indices = np.sort(rng.choice(usable_rows, size=sample_count, replace=False))
    params_np = np.asarray(pool[indices], dtype=np.float32)

    training_pool_string = str(saved_args.get("pool_dir", ""))
    same_pool_warning = False
    try:
        same_pool_warning = (
            validation_pool_dir.resolve()
            == Path(training_pool_string).resolve()
        )
    except Exception:
        same_pool_warning = False

    all_params: List[np.ndarray] = []
    all_f_true: List[np.ndarray] = []
    all_f_pred: List[np.ndarray] = []
    all_g_clean: List[np.ndarray] = []
    all_g_noisy: List[np.ndarray] = []
    all_g_pred: List[np.ndarray] = []
    all_g_true_discrete: List[np.ndarray] = []

    print("=" * 72)
    print("开始固定验证")
    print(f"设备: {device}")
    print(f"权重: {checkpoint_info['weights_path']}")
    print(f"验证参数池: {validation_pool_dir.resolve()}")
    print(f"验证样本数: {sample_count:,}")
    print(f"固定随机种子: {args.seed}")
    print(f"噪声比例: {validation_args['noise_level']:.2%}")
    if same_pool_warning:
        print("警告：验证池与训练池路径相同；结果只能反映拟合情况，不能代表泛化能力。")
    print("=" * 72)

    with torch.inference_mode():
        for start in range(0, sample_count, args.batch_size):
            stop = min(start + args.batch_size, sample_count)
            params = torch.from_numpy(params_np[start:stop]).to(device)
            f_true_2d, g_clean_2d = physics.make_clean_batch(params)

            # 使用 NumPy 固定生成器产生验证噪声。这样在相同 seed 下，
            # 验证噪声不依赖 CPU/GPU 或 batch-size，便于公平比较 best/latest。
            g_clean_np_batch = as_numpy(g_clean_2d).astype(np.float64, copy=False)
            g_rms_np = np.sqrt(np.mean(g_clean_np_batch ** 2, axis=1, keepdims=True))
            noise_np = (
                float(validation_args["noise_level"])
                * g_rms_np
                * noise_rng.standard_normal(g_clean_np_batch.shape)
            ).astype(np.float32)
            g_noisy_2d = g_clean_2d + torch.from_numpy(noise_np).to(
                device=device, dtype=g_clean_2d.dtype
            )

            f_true = f_true_2d.unsqueeze(1)
            g_clean = g_clean_2d.unsqueeze(1)
            g_noisy = g_noisy_2d.unsqueeze(1)

            f_pred = model(g_noisy)
            f_pred = normalize_prediction_shape(f_pred, f_true)
            g_pred = criterion.physics_forward_integral(f_pred)
            g_true_discrete = criterion.physics_forward_integral(f_true)

            all_params.append(params_np[start:stop].copy())
            all_f_true.append(as_numpy(f_true.squeeze(1)))
            all_f_pred.append(as_numpy(f_pred.squeeze(1)))
            all_g_clean.append(as_numpy(g_clean.squeeze(1)))
            all_g_noisy.append(as_numpy(g_noisy.squeeze(1)))
            all_g_pred.append(as_numpy(g_pred.squeeze(1)))
            all_g_true_discrete.append(as_numpy(g_true_discrete.squeeze(1)))

            print(f"已验证: {stop:,}/{sample_count:,}")

    params_arr = np.concatenate(all_params, axis=0)
    f_true_arr = np.concatenate(all_f_true, axis=0)
    f_pred_arr = np.concatenate(all_f_pred, axis=0)
    g_clean_arr = np.concatenate(all_g_clean, axis=0)
    g_noisy_arr = np.concatenate(all_g_noisy, axis=0)
    g_pred_arr = np.concatenate(all_g_pred, axis=0)
    g_true_discrete_arr = np.concatenate(all_g_true_discrete, axis=0)

    f_true_t = torch.from_numpy(f_true_arr)
    f_pred_t = torch.from_numpy(f_pred_arr)
    g_clean_t = torch.from_numpy(g_clean_arr)
    g_pred_t = torch.from_numpy(g_pred_arr)
    g_true_discrete_t = torch.from_numpy(g_true_discrete_arr)

    f_error_arr = f_pred_arr - f_true_arr
    f_rmse = np.sqrt(np.mean(f_error_arr.astype(np.float64) ** 2, axis=1))
    f_true_rms = np.sqrt(np.mean(f_true_arr.astype(np.float64) ** 2, axis=1))
    f_nrmse = f_rmse / np.maximum(f_true_rms, 1e-12)

    # 对等长向量，RMSE/RMS(target) 与 relative L2 数值完全相同。
    f_rel = as_numpy(relative_l2(f_pred_t, f_true_t))
    g_rel_clean = as_numpy(relative_l2(g_pred_t, g_clean_t))
    g_rel_discrete = as_numpy(relative_l2(g_pred_t, g_true_discrete_t))
    representation_gap = as_numpy(relative_l2(g_true_discrete_t, g_clean_t))

    s_grid = np.linspace(
        float(saved_args.get("s_min", 0.1764)),
        float(saved_args.get("s_max", 6.0)),
        int(saved_args.get("output_points", 100)),
    )
    q2_grid = np.linspace(
        float(saved_args.get("q2_min", -100.0)),
        float(saved_args.get("q2_max", -6.0)),
        int(saved_args.get("input_points", 100)),
    )

    true_peak_idx = np.argmax(f_true_arr, axis=1)
    pred_peak_idx = np.argmax(f_pred_arr, axis=1)
    true_peak_s = s_grid[true_peak_idx]
    pred_peak_s = s_grid[pred_peak_idx]
    peak_grid_error = np.abs(pred_peak_s - true_peak_s)
    true_argmax_to_m_error = np.abs(true_peak_s - params_arr[:, 3])
    pred_argmax_to_m_error = np.abs(pred_peak_s - params_arr[:, 3])

    true_peak_height = np.max(f_true_arr, axis=1)
    pred_peak_height = np.max(f_pred_arr, axis=1)
    peak_height_rel = (
        np.abs(pred_peak_height - true_peak_height)
        / np.maximum(np.abs(true_peak_height), 1e-12)
    )
    negative_point_fraction_per_sample = np.mean(f_pred_arr < 0.0, axis=1)

    metrics_path = output_dir / "validation_metrics.csv"
    fieldnames = [
        "validation_row_id",
        *PARAMETER_NAMES,
        "m_gamma",
        "f_rmse",
        "f_true_rms",
        "f_nrmse_rms",
        "f_relative_l2",
        "g_relative_l2_vs_clean",
        "g_relative_l2_vs_discrete_clean",
        "clean_discrete_representation_gap",
        "true_peak_s_grid",
        "pred_peak_s_grid",
        "peak_grid_error",
        "true_argmax_to_m_error",
        "pred_argmax_to_m_error",
        "true_peak_height",
        "pred_peak_height",
        "peak_height_relative_error",
        "negative_point_fraction",
    ]
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(sample_count):
            row = {
                "validation_row_id": int(indices[i]),
                "a1": float(params_arr[i, 0]),
                "a2": float(params_arr[i, 1]),
                "a3": float(params_arr[i, 2]),
                "m": float(params_arr[i, 3]),
                "gamma": float(params_arr[i, 4]),
                "m_gamma": float(params_arr[i, 3] * params_arr[i, 4]),
                "f_rmse": float(f_rmse[i]),
                "f_true_rms": float(f_true_rms[i]),
                "f_nrmse_rms": float(f_nrmse[i]),
                "f_relative_l2": float(f_rel[i]),
                "g_relative_l2_vs_clean": float(g_rel_clean[i]),
                "g_relative_l2_vs_discrete_clean": float(g_rel_discrete[i]),
                "clean_discrete_representation_gap": float(representation_gap[i]),
                "true_peak_s_grid": float(true_peak_s[i]),
                "pred_peak_s_grid": float(pred_peak_s[i]),
                "peak_grid_error": float(peak_grid_error[i]),
                "true_argmax_to_m_error": float(true_argmax_to_m_error[i]),
                "pred_argmax_to_m_error": float(pred_argmax_to_m_error[i]),
                "true_peak_height": float(true_peak_height[i]),
                "pred_peak_height": float(pred_peak_height[i]),
                "peak_height_relative_error": float(peak_height_rel[i]),
                "negative_point_fraction": float(negative_point_fraction_per_sample[i]),
            }
            writer.writerow(row)

    summary = {
        "checkpoint": checkpoint_info,
        "validation_pool": str(validation_pool_dir.resolve()),
        "same_as_training_pool_warning": bool(same_pool_warning),
        "validation_samples": int(sample_count),
        "seed": int(args.seed),
        "noise_level": float(validation_args["noise_level"]),
        "weights_choice": args.weights,
        "validation_noise_is_batch_and_device_independent": True,
        "f_rmse": summarize_vector(f_rmse),
        "f_true_rms": summarize_vector(f_true_rms),
        "f_nrmse_rms": summarize_vector(f_nrmse),
        "f_relative_l2": summarize_vector(f_rel),
        "g_relative_l2_vs_clean": summarize_vector(g_rel_clean),
        "g_relative_l2_vs_discrete_clean": summarize_vector(g_rel_discrete),
        "clean_discrete_representation_gap": summarize_vector(representation_gap),
        "peak_grid_error": summarize_vector(peak_grid_error),
        "true_argmax_to_m_error": summarize_vector(true_argmax_to_m_error),
        "pred_argmax_to_m_error": summarize_vector(pred_argmax_to_m_error),
        "peak_height_relative_error": summarize_vector(peak_height_rel),
        "negative_prediction_point_fraction": float(np.mean(f_pred_arr < 0.0)),
        "negative_prediction_sample_fraction": float(
            np.mean(np.any(f_pred_arr < 0.0, axis=1))
        ),
        "notes": {
            "f_rmse": "预测 f 与真实 f 的绝对均方根误差。",
            "f_nrmse_rms": (
                "f_rmse / RMS(f_true)，表示相对重建误差；对等长向量，"
                "它与 f_relative_l2 数值相同。"
            ),
            "f_relative_l2": "越小越好；比较预测 f 与真实 f。",
            "argmax_to_m_warning": (
                "m 是 Lorentz 项中心，不一定等于完整 u(s) 的全局最大点；"
                "线性背景和 (s+shift)^-2 因子都可能使 argmax 偏移。"
            ),
            "g_relative_l2_vs_clean": "越小越好；比较 K(f_pred) 与高精度 g_clean。",
            "g_relative_l2_vs_discrete_clean": "越小越好；比较 K(f_pred) 与 K(f_true_100点)。",
            "clean_discrete_representation_gap": (
                "衡量100点 f 标签与高精度 g_clean 的先天不一致；"
                "此值大时，clean physics target 可能与100点输出表示冲突。"
            ),
        },
    }
    summary_path = output_dir / "validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 选取最差样本和随机样本画图，便于直接观察峰是否被压平或错位。
    selected: List[int] = []
    if args.plot_count > 0:
        worst_f = np.argsort(f_rel)[::-1]
        worst_g = np.argsort(g_rel_clean)[::-1]
        for candidate in list(worst_f[: max(1, args.plot_count // 3)]) + list(
            worst_g[: max(1, args.plot_count // 3)]
        ):
            if int(candidate) not in selected:
                selected.append(int(candidate))
            if len(selected) >= args.plot_count:
                break
        remaining = [i for i in range(sample_count) if i not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: max(0, args.plot_count - len(selected))])

    for rank, i in enumerate(selected, start=1):
        row_id = int(indices[i])
        title_prefix = (
            f"row={row_id}, f_rel={f_rel[i]:.3g}, "
            f"g_rel_clean={g_rel_clean[i]:.3g}, mγ={params_arr[i,3]*params_arr[i,4]:.3g}"
        )
        save_curve_plot(
            s_grid,
            [
                ("f_true", f_true_arr[i]),
                ("f_pred", f_pred_arr[i]),
            ],
            title=f"f validation: {title_prefix}",
            xlabel="s",
            ylabel="scaled f(s)",
            path=plots_dir / f"{rank:02d}_row_{row_id}_f.png",
        )
        save_curve_plot(
            q2_grid,
            [
                ("g_noisy input", g_noisy_arr[i]),
                ("g_clean", g_clean_arr[i]),
                ("K(f_pred)", g_pred_arr[i]),
                ("K(f_true_100)", g_true_discrete_arr[i]),
            ],
            title=f"g validation: {title_prefix}",
            xlabel="q^2",
            ylabel="scaled g(q^2)",
            path=plots_dir / f"{rank:02d}_row_{row_id}_g.png",
        )

    print("=" * 72)
    print("验证完成")
    print(f"f RMSE mean          : {summary['f_rmse']['mean']:.6g}")
    print(f"f NRMSE mean         : {summary['f_nrmse_rms']['mean']:.6g}")
    print(f"f relative L2 mean   : {summary['f_relative_l2']['mean']:.6g}")
    print(f"f relative L2 median : {summary['f_relative_l2']['median']:.6g}")
    print(f"f relative L2 p90    : {summary['f_relative_l2']['p90']:.6g}")
    print(f"g vs clean mean      : {summary['g_relative_l2_vs_clean']['mean']:.6g}")
    print(f"g vs discrete mean   : {summary['g_relative_l2_vs_discrete_clean']['mean']:.6g}")
    print(f"representation gap   : {summary['clean_discrete_representation_gap']['mean']:.6g}")
    print(f"peak grid error mean : {summary['peak_grid_error']['mean']:.6g}")
    print(f"negative point frac  : {summary['negative_prediction_point_fraction']:.6%}")
    print(f"summary: {summary_path.resolve()}")
    print(f"metrics: {metrics_path.resolve()}")
    print(f"plots  : {plots_dir.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
