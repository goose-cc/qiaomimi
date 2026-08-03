from __future__ import annotations

"""
Peak-aware ranked validation and plotting for the V2 free-curve Transformer.

This script intentionally does not replace validate_mc_transformer.py.  It uses
exactly the same checkpoint loading, online physics, validation sampling, and
fixed-noise convention, but lets the user explicitly select best/typical/worst
samples and writes the numerical curves for every selected sample.

Python 3.8 compatible.
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mc_inverse_loss import MonteCarloInverseLoss
from train_mc_parameter_pool_transformer_loss import OnlinePhysics, normalize_prediction_shape
from validate_mc_transformer import (
    PARAMETER_NAMES,
    as_numpy,
    choose_device,
    load_model_and_config,
    open_parameter_pool,
    relative_l2,
    set_fixed_seed,
)


METRIC_DIRECTIONS = {
    "f_relative_l2": "lower",
    "f_nrmse_rms": "lower",
    "f_rmse_scaled": "lower",
    "f_mse_scaled": "lower",
    "g_relative_l2_vs_clean": "lower",
    "resonance_relative_l2": "lower",
    "resonance_rmse_scaled": "lower",
    "peak_center_abs_error": "lower",
    "peak_height_relative_error": "lower",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate V2 Transformer and save peak-aware ranked plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--validation-pool-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--weights", choices=("best", "latest"), default="latest")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--noise-level", type=float, default=None)
    parser.add_argument("--output-dir", default="./validation_results/ranked")
    parser.add_argument("--plot-count", type=int, default=12)
    parser.add_argument(
        "--selection",
        choices=("best", "typical", "representative", "worst", "mixed"),
        default="best",
        help=(
            "best=误差最低；typical=最接近中位数；representative=覆盖多个分位数；"
            "worst=误差最高；mixed=最好/典型/最差各一部分"
        ),
    )
    parser.add_argument(
        "--rank-metric",
        choices=tuple(METRIC_DIRECTIONS.keys()),
        default="f_relative_l2",
        help="用于挑选样本的排序指标",
    )
    parser.add_argument(
        "--peak-only",
        action="store_true",
        help="只在峰中心位于区间内、峰可分辨且共振可见的样本中排序",
    )
    parser.add_argument(
        "--min-resonance-visibility",
        type=float,
        default=0.10,
        help="共振 L2 范数占总谱 L2 范数的最低比例",
    )
    parser.add_argument(
        "--min-width-grid-cells",
        type=float,
        default=1.0,
        help="要求 m*gamma 至少覆盖多少个输出网格间隔",
    )
    parser.add_argument(
        "--allow-outside-center",
        action="store_true",
        help="peak-only 时允许 m 位于绘图区间之外（通常不建议）",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--save-selected-data",
        dest="save_selected_data",
        action="store_true",
        help="为每个选中样本保存 f/g 数值 CSV",
    )
    parser.add_argument(
        "--no-save-selected-data",
        dest="save_selected_data",
        action="store_false",
    )
    parser.set_defaults(save_selected_data=True)
    return parser.parse_args()


def ensure_positive_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num-samples 必须大于 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")
    if args.plot_count < 0:
        raise ValueError("--plot-count 不能为负数")


def pick_closest_unused(
    sorted_indices: np.ndarray,
    target_position: int,
    used: set,
) -> int:
    n = len(sorted_indices)
    for radius in range(n):
        for pos in (target_position - radius, target_position + radius):
            if 0 <= pos < n:
                candidate = int(sorted_indices[pos])
                if candidate not in used:
                    used.add(candidate)
                    return candidate
    raise RuntimeError("无法选取未使用样本")


def select_indices(metric: np.ndarray, count: int, mode: str) -> List[int]:
    count = min(int(count), int(metric.size))
    if count <= 0:
        return []

    order = np.argsort(metric)  # lower is always better for supported metrics
    used = set()
    selected: List[int] = []

    if mode == "best":
        return [int(i) for i in order[:count]]
    if mode == "worst":
        return [int(i) for i in order[::-1][:count]]

    if mode == "typical":
        center = (len(order) - 1) // 2
        for radius in range(len(order)):
            for pos in (center - radius, center + radius):
                if 0 <= pos < len(order):
                    idx = int(order[pos])
                    if idx not in used:
                        used.add(idx)
                        selected.append(idx)
                    if len(selected) >= count:
                        return selected
        return selected

    if mode == "representative":
        # Avoid exact endpoints; cover the distribution from good to poor.
        quantiles = np.linspace(0.05, 0.95, count)
        for q in quantiles:
            pos = int(round(q * (len(order) - 1)))
            selected.append(pick_closest_unused(order, pos, used))
        return selected

    if mode == "mixed":
        n_best = max(1, count // 3)
        n_worst = max(1, count // 3)
        n_typical = count - n_best - n_worst
        if n_typical < 0:
            n_typical = 0

        for idx in order[:n_best]:
            idx_int = int(idx)
            if idx_int not in used:
                used.add(idx_int)
                selected.append(idx_int)

        if n_typical:
            middle_positions = np.linspace(0.40, 0.60, n_typical)
            for q in middle_positions:
                pos = int(round(q * (len(order) - 1)))
                selected.append(pick_closest_unused(order, pos, used))

        for idx in order[::-1]:
            idx_int = int(idx)
            if idx_int not in used:
                used.add(idx_int)
                selected.append(idx_int)
            if len(selected) >= count:
                break
        return selected[:count]

    raise ValueError("未知 selection: %s" % mode)


def save_f_plot(
    *,
    s_grid: np.ndarray,
    f_true: np.ndarray,
    f_pred: np.ndarray,
    row_id: int,
    rank: int,
    params: np.ndarray,
    metrics: Dict[str, float],
    path: Path,
) -> None:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(s_grid, f_true, label="f_true", linewidth=2.0)
    ax.plot(s_grid, f_pred, label="f_pred", linewidth=2.0)
    ax.set_xlabel("s")
    ax.set_ylabel("scaled f(s)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    score = metrics["relative_reconstruction_score_percent"]
    ax.set_title(
        "V2 ranked validation | rank=%d row=%d | %s" %
        (rank, row_id, metrics["selection_label"])
    )
    info = (
        "MSE(scaled) = %.6g\n"
        "RMSE(scaled) = %.6g\n"
        "NRMSE / relative L2 = %.4f\n"
        "relative reconstruction score = %.2f%%\n"
        "g relative error vs clean = %.4f\n"
        "a1=%.4g, a2=%.4g, a3=%.4g\n"
        "m=%.4g, gamma=%.4g, m*gamma=%.4g"
    ) % (
        metrics["f_mse_scaled"],
        metrics["f_rmse_scaled"],
        metrics["f_relative_l2"],
        score,
        metrics["g_relative_l2_vs_clean"],
        params[0], params[1], params[2], params[3], params[4], params[3] * params[4],
    )
    ax.text(
        0.985,
        0.03,
        info,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_g_plot(
    *,
    q2_grid: np.ndarray,
    g_noisy: np.ndarray,
    g_clean: np.ndarray,
    g_pred: np.ndarray,
    g_true_discrete: np.ndarray,
    row_id: int,
    rank: int,
    metrics: Dict[str, float],
    path: Path,
) -> None:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(q2_grid, g_noisy, label="g_noisy input", linewidth=1.5)
    ax.plot(q2_grid, g_clean, label="g_clean", linewidth=2.0)
    ax.plot(q2_grid, g_pred, label="K(f_pred)", linewidth=2.0)
    ax.plot(q2_grid, g_true_discrete, label="K(f_true_100)", linewidth=1.5)
    ax.set_xlabel("q^2")
    ax.set_ylabel("scaled g(q^2)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(
        "Forward consistency | rank=%d row=%d | g_rel_clean=%.5f" %
        (rank, row_id, metrics["g_relative_l2_vs_clean"])
    )
    info = (
        "g MSE vs clean (scaled) = %.6g\n"
        "g RMSE vs clean (scaled) = %.6g\n"
        "g relative L2 vs clean = %.5f\n"
        "representation gap = %.5f"
    ) % (
        metrics["g_mse_vs_clean_scaled"],
        metrics["g_rmse_vs_clean_scaled"],
        metrics["g_relative_l2_vs_clean"],
        metrics["clean_discrete_representation_gap"],
    )
    ax.text(
        0.985,
        0.03,
        info,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_resonance_plot(
    *,
    s_grid: np.ndarray,
    resonance_true: np.ndarray,
    resonance_pred_diagnostic: np.ndarray,
    row_id: int,
    rank: int,
    metrics: Dict[str, float],
    path: Path,
) -> None:
    """Plot the true resonance and a diagnostic predicted residual.

    The predicted residual subtracts the *true validation background* from
    f_pred.  This is intentionally a diagnostic only; it is not an inference
    procedure available for unknown real data.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(s_grid, resonance_true, label="true resonance", linewidth=2.0)
    ax.plot(
        s_grid,
        resonance_pred_diagnostic,
        label="f_pred - true background (diagnostic)",
        linewidth=2.0,
    )
    ax.axvline(metrics["true_peak_s_grid"], linestyle="--", linewidth=1.2, label="true peak grid")
    ax.axvline(metrics["pred_peak_s_grid"], linestyle=":", linewidth=1.2, label="pred peak grid")
    ax.set_xlabel("s")
    ax.set_ylabel("scaled resonance component")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(
        "Peak-aware validation | rank=%d row=%d | %s" %
        (rank, row_id, metrics["selection_label"])
    )
    info = (
        "resonance relative L2 = %.4f\n"
        "resonance RMSE(scaled) = %.6g\n"
        "visibility = %.4f\n"
        "peak center abs error = %.4f\n"
        "peak height relative error = %.4f\n"
        "width/grid = %.2f cells"
    ) % (
        metrics["resonance_relative_l2"],
        metrics["resonance_rmse_scaled"],
        metrics["resonance_visibility"],
        metrics["peak_center_abs_error"],
        metrics["peak_height_relative_error"],
        metrics["width_grid_cells"],
    )
    ax.text(
        0.985, 0.03, info, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_curve_csv(path: Path, headers: Sequence[str], columns: Sequence[np.ndarray]) -> None:
    data = np.column_stack(columns)
    np.savetxt(
        str(path),
        data,
        delimiter=",",
        header=",".join(headers),
        comments="",
        fmt="%.10g",
    )


def main() -> None:
    args = parse_args()
    ensure_positive_args(args)
    set_fixed_seed(args.seed)

    device = choose_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    validation_pool_dir = Path(args.validation_pool_dir)
    output_dir = Path(args.output_dir)
    plots_f_dir = output_dir / "plots_f"
    plots_g_dir = output_dir / "plots_g"
    plots_resonance_dir = output_dir / "plots_resonance"
    selected_data_dir = output_dir / "selected_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_f_dir.mkdir(parents=True, exist_ok=True)
    plots_g_dir.mkdir(parents=True, exist_ok=True)
    plots_resonance_dir.mkdir(parents=True, exist_ok=True)
    if args.save_selected_data:
        selected_data_dir.mkdir(parents=True, exist_ok=True)

    for folder in (plots_f_dir, plots_g_dir, plots_resonance_dir):
        for old in folder.glob("*.png"):
            old.unlink()

    model, saved_args, checkpoint_info = load_model_and_config(
        checkpoint_dir, args.weights, device
    )

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
    metadata_names = metadata.get("parameter_names")
    if metadata_names and list(metadata_names) != list(PARAMETER_NAMES):
        raise RuntimeError(
            "验证参数池 parameter_names 顺序不一致: %r != %r" %
            (metadata_names, list(PARAMETER_NAMES))
        )

    sample_count = min(int(args.num_samples), usable_rows)
    rng = np.random.default_rng(args.seed)
    noise_rng = np.random.default_rng(args.seed + 1)
    validation_rows = np.sort(rng.choice(usable_rows, size=sample_count, replace=False))
    params_np = np.asarray(pool[validation_rows], dtype=np.float32)

    all_f_true: List[np.ndarray] = []
    all_f_pred: List[np.ndarray] = []
    all_g_clean: List[np.ndarray] = []
    all_g_noisy: List[np.ndarray] = []
    all_g_pred: List[np.ndarray] = []
    all_g_true_discrete: List[np.ndarray] = []

    print("=" * 72)
    print("开始 ranked validation")
    print("设备:", device)
    print("权重:", checkpoint_info["weights_path"])
    print("selected model step:", "%s" % format(checkpoint_info["selected_global_step"], ","))
    print("验证样本数:", "%s" % format(sample_count, ","))
    print("噪声比例: %.2f%%" % (100.0 * validation_args["noise_level"]))
    print("selection:", args.selection)
    print("rank metric:", args.rank_metric)
    print("peak only:", args.peak_only)
    print("=" * 72)

    with torch.inference_mode():
        for start in range(0, sample_count, args.batch_size):
            stop = min(start + args.batch_size, sample_count)
            params = torch.from_numpy(params_np[start:stop]).to(device)
            f_true_2d, g_clean_2d = physics.make_clean_batch(params)

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
            f_pred = normalize_prediction_shape(model(g_noisy), f_true)
            g_pred = criterion.physics_forward_integral(f_pred)
            g_true_discrete = criterion.physics_forward_integral(f_true)

            all_f_true.append(as_numpy(f_true.squeeze(1)))
            all_f_pred.append(as_numpy(f_pred.squeeze(1)))
            all_g_clean.append(as_numpy(g_clean.squeeze(1)))
            all_g_noisy.append(as_numpy(g_noisy.squeeze(1)))
            all_g_pred.append(as_numpy(g_pred.squeeze(1)))
            all_g_true_discrete.append(as_numpy(g_true_discrete.squeeze(1)))
            print("已验证: %s/%s" % (format(stop, ","), format(sample_count, ",")))

    f_true_arr = np.concatenate(all_f_true, axis=0)
    f_pred_arr = np.concatenate(all_f_pred, axis=0)
    g_clean_arr = np.concatenate(all_g_clean, axis=0)
    g_noisy_arr = np.concatenate(all_g_noisy, axis=0)
    g_pred_arr = np.concatenate(all_g_pred, axis=0)
    g_true_discrete_arr = np.concatenate(all_g_true_discrete, axis=0)

    f_err = f_pred_arr.astype(np.float64) - f_true_arr.astype(np.float64)
    g_err = g_pred_arr.astype(np.float64) - g_clean_arr.astype(np.float64)
    f_mse = np.mean(f_err ** 2, axis=1)
    f_rmse = np.sqrt(f_mse)
    f_true_rms = np.sqrt(np.mean(f_true_arr.astype(np.float64) ** 2, axis=1))
    f_nrmse = f_rmse / np.maximum(f_true_rms, 1e-12)

    f_rel = as_numpy(relative_l2(torch.from_numpy(f_pred_arr), torch.from_numpy(f_true_arr)))
    g_rel_clean = as_numpy(relative_l2(torch.from_numpy(g_pred_arr), torch.from_numpy(g_clean_arr)))
    g_rel_discrete = as_numpy(
        relative_l2(torch.from_numpy(g_pred_arr), torch.from_numpy(g_true_discrete_arr))
    )
    representation_gap = as_numpy(
        relative_l2(torch.from_numpy(g_true_discrete_arr), torch.from_numpy(g_clean_arr))
    )
    g_mse = np.mean(g_err ** 2, axis=1)
    g_rmse = np.sqrt(g_mse)

    # Peak-aware diagnostics.  The validation parameters are known here, so
    # the exact background may be subtracted for diagnostic evaluation.
    s_min = float(saved_args.get("s_min", 0.1764))
    s_max = float(saved_args.get("s_max", 6.0))
    shift = float(saved_args.get("shift", 400.0))
    data_scale = float(saved_args.get("data_scale", 160000.0))
    output_points = int(saved_args.get("output_points", 100))
    s_grid = np.linspace(s_min, s_max, output_points)
    ds = float((s_max - s_min) / max(output_points - 1, 1))

    a1 = params_np[:, 0:1].astype(np.float64)
    a2 = params_np[:, 1:2].astype(np.float64)
    a3 = params_np[:, 2:3].astype(np.float64)
    m = params_np[:, 3:4].astype(np.float64)
    gamma = params_np[:, 4:5].astype(np.float64)
    width = m * gamma
    s_row = s_grid.reshape(1, -1)
    background_true = (
        data_scale * (a2 * s_row + a3) / (s_row + shift) ** 2
    )
    resonance_true = (
        data_scale
        * (a1 / np.pi)
        * width
        / ((s_row - m) ** 2 + width ** 2)
        / (s_row + shift) ** 2
    )
    # Equal to f_pred minus the known validation background.  This isolates
    # whether the free-curve output actually contains the true resonance.
    resonance_pred_diag = f_pred_arr.astype(np.float64) - background_true
    resonance_err = resonance_pred_diag - resonance_true
    resonance_mse = np.mean(resonance_err ** 2, axis=1)
    resonance_rmse = np.sqrt(resonance_mse)
    resonance_norm = np.linalg.norm(resonance_true, axis=1)
    resonance_rel = np.linalg.norm(resonance_err, axis=1) / np.maximum(resonance_norm, 1e-12)
    resonance_visibility = resonance_norm / np.maximum(
        np.linalg.norm(f_true_arr.astype(np.float64), axis=1), 1e-12
    )

    true_peak_idx = np.argmax(resonance_true, axis=1)
    pred_peak_idx = np.argmax(resonance_pred_diag, axis=1)
    true_peak_s_grid = s_grid[true_peak_idx]
    pred_peak_s_grid = s_grid[pred_peak_idx]
    peak_center_abs_error = np.abs(pred_peak_s_grid - true_peak_s_grid)
    true_peak_height = resonance_true[np.arange(sample_count), true_peak_idx]
    pred_height_at_true_peak = resonance_pred_diag[np.arange(sample_count), true_peak_idx]
    peak_height_relative_error = np.abs(pred_height_at_true_peak - true_peak_height) / np.maximum(
        np.abs(true_peak_height), 1e-12
    )
    width_1d = width[:, 0]
    width_grid_cells = width_1d / max(ds, 1e-12)

    center_in_domain = (params_np[:, 3] >= s_min) & (params_np[:, 3] <= s_max)
    eligible_peak = (
        (resonance_visibility >= float(args.min_resonance_visibility))
        & (width_grid_cells >= float(args.min_width_grid_cells))
    )
    if not args.allow_outside_center:
        eligible_peak &= center_in_domain

    metric_arrays: Dict[str, np.ndarray] = {
        "f_relative_l2": f_rel,
        "f_nrmse_rms": f_nrmse,
        "f_rmse_scaled": f_rmse,
        "f_mse_scaled": f_mse,
        "g_relative_l2_vs_clean": g_rel_clean,
        "resonance_relative_l2": resonance_rel,
        "resonance_rmse_scaled": resonance_rmse,
        "peak_center_abs_error": peak_center_abs_error,
        "peak_height_relative_error": peak_height_relative_error,
    }
    rank_metric_values = metric_arrays[args.rank_metric]
    if args.peak_only:
        candidate_indices = np.flatnonzero(eligible_peak)
        if candidate_indices.size == 0:
            raise RuntimeError(
                "没有样本满足 peak-only 条件；请降低 --min-resonance-visibility "
                "或 --min-width-grid-cells"
            )
        selected_local = select_indices(
            rank_metric_values[candidate_indices], args.plot_count, args.selection
        )
        selected = [int(candidate_indices[j]) for j in selected_local]
    else:
        candidate_indices = np.arange(sample_count)
        selected = select_indices(rank_metric_values, args.plot_count, args.selection)

    q2_grid = np.linspace(
        float(saved_args.get("q2_min", -100.0)),
        float(saved_args.get("q2_max", -6.0)),
        int(saved_args.get("input_points", 100)),
    )

    selected_csv_path = output_dir / "selected_samples.csv"
    fields = [
        "display_rank",
        "selection",
        "rank_metric",
        "rank_metric_value",
        "validation_row_id",
        *PARAMETER_NAMES,
        "m_gamma",
        "resonance_visibility",
        "width_grid_cells",
        "resonance_mse_scaled",
        "resonance_rmse_scaled",
        "resonance_relative_l2",
        "true_peak_s_grid",
        "pred_peak_s_grid",
        "peak_center_abs_error",
        "peak_height_relative_error",
        "f_mse_scaled",
        "f_rmse_scaled",
        "f_nrmse_rms",
        "f_relative_l2",
        "relative_reconstruction_score_percent",
        "g_mse_vs_clean_scaled",
        "g_rmse_vs_clean_scaled",
        "g_relative_l2_vs_clean",
        "g_relative_l2_vs_discrete_clean",
        "clean_discrete_representation_gap",
    ]

    selected_rows: List[Dict[str, Any]] = []
    with selected_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for display_rank, i in enumerate(selected, start=1):
            row_id = int(validation_rows[i])
            score_percent = 100.0 * (1.0 - float(f_nrmse[i]))
            row: Dict[str, Any] = {
                "display_rank": display_rank,
                "selection": args.selection,
                "rank_metric": args.rank_metric,
                "rank_metric_value": float(rank_metric_values[i]),
                "validation_row_id": row_id,
                "a1": float(params_np[i, 0]),
                "a2": float(params_np[i, 1]),
                "a3": float(params_np[i, 2]),
                "m": float(params_np[i, 3]),
                "gamma": float(params_np[i, 4]),
                "m_gamma": float(params_np[i, 3] * params_np[i, 4]),
                "resonance_visibility": float(resonance_visibility[i]),
                "width_grid_cells": float(width_grid_cells[i]),
                "resonance_mse_scaled": float(resonance_mse[i]),
                "resonance_rmse_scaled": float(resonance_rmse[i]),
                "resonance_relative_l2": float(resonance_rel[i]),
                "true_peak_s_grid": float(true_peak_s_grid[i]),
                "pred_peak_s_grid": float(pred_peak_s_grid[i]),
                "peak_center_abs_error": float(peak_center_abs_error[i]),
                "peak_height_relative_error": float(peak_height_relative_error[i]),
                "f_mse_scaled": float(f_mse[i]),
                "f_rmse_scaled": float(f_rmse[i]),
                "f_nrmse_rms": float(f_nrmse[i]),
                "f_relative_l2": float(f_rel[i]),
                "relative_reconstruction_score_percent": score_percent,
                "g_mse_vs_clean_scaled": float(g_mse[i]),
                "g_rmse_vs_clean_scaled": float(g_rmse[i]),
                "g_relative_l2_vs_clean": float(g_rel_clean[i]),
                "g_relative_l2_vs_discrete_clean": float(g_rel_discrete[i]),
                "clean_discrete_representation_gap": float(representation_gap[i]),
            }
            writer.writerow(row)
            selected_rows.append(row)

            plot_metrics = dict(row)
            plot_metrics["selection_label"] = "%s by %s" % (
                args.selection, args.rank_metric
            )
            save_f_plot(
                s_grid=s_grid,
                f_true=f_true_arr[i],
                f_pred=f_pred_arr[i],
                row_id=row_id,
                rank=display_rank,
                params=params_np[i],
                metrics=plot_metrics,
                path=plots_f_dir / (
                    "%02d_row_%d_f_%s.png" % (display_rank, row_id, args.selection)
                ),
            )
            save_resonance_plot(
                s_grid=s_grid,
                resonance_true=resonance_true[i],
                resonance_pred_diagnostic=resonance_pred_diag[i],
                row_id=row_id,
                rank=display_rank,
                metrics=plot_metrics,
                path=plots_resonance_dir / (
                    "%02d_row_%d_resonance_%s.png" %
                    (display_rank, row_id, args.selection)
                ),
            )
            save_g_plot(
                q2_grid=q2_grid,
                g_noisy=g_noisy_arr[i],
                g_clean=g_clean_arr[i],
                g_pred=g_pred_arr[i],
                g_true_discrete=g_true_discrete_arr[i],
                row_id=row_id,
                rank=display_rank,
                metrics=plot_metrics,
                path=plots_g_dir / (
                    "%02d_row_%d_g_%s.png" % (display_rank, row_id, args.selection)
                ),
            )

            if args.save_selected_data:
                write_curve_csv(
                    selected_data_dir / ("%02d_row_%d_f.csv" % (display_rank, row_id)),
                    ("s", "f_true", "f_pred", "error", "abs_error", "squared_error"),
                    (
                        s_grid,
                        f_true_arr[i],
                        f_pred_arr[i],
                        f_pred_arr[i] - f_true_arr[i],
                        np.abs(f_pred_arr[i] - f_true_arr[i]),
                        (f_pred_arr[i] - f_true_arr[i]) ** 2,
                    ),
                )
                write_curve_csv(
                    selected_data_dir / ("%02d_row_%d_resonance.csv" % (display_rank, row_id)),
                    (
                        "s", "true_background", "true_resonance",
                        "pred_resonance_diagnostic", "resonance_error"
                    ),
                    (
                        s_grid, background_true[i], resonance_true[i],
                        resonance_pred_diag[i], resonance_err[i]
                    ),
                )
                write_curve_csv(
                    selected_data_dir / ("%02d_row_%d_g.csv" % (display_rank, row_id)),
                    ("q2", "g_noisy", "g_clean", "g_from_f_pred", "g_from_f_true_100"),
                    (
                        q2_grid,
                        g_noisy_arr[i],
                        g_clean_arr[i],
                        g_pred_arr[i],
                        g_true_discrete_arr[i],
                    ),
                )

    selection_info = {
        "checkpoint": checkpoint_info,
        "validation_pool": str(validation_pool_dir.resolve()),
        "validation_samples": int(sample_count),
        "noise_level": float(validation_args["noise_level"]),
        "seed": int(args.seed),
        "selection": args.selection,
        "rank_metric": args.rank_metric,
        "plot_count": int(len(selected)),
        "peak_only": bool(args.peak_only),
        "eligible_peak_samples": int(np.count_nonzero(eligible_peak)),
        "min_resonance_visibility": float(args.min_resonance_visibility),
        "min_width_grid_cells": float(args.min_width_grid_cells),
        "selected_samples_csv": str(selected_csv_path.resolve()),
        "plots_f": str(plots_f_dir.resolve()),
        "plots_g": str(plots_g_dir.resolve()),
        "plots_resonance": str(plots_resonance_dir.resolve()),
        "selected_data": str(selected_data_dir.resolve()) if args.save_selected_data else None,
        "metric_notes": {
            "mse_rmse_units": "图中 MSE/RMSE 使用训练时 scaled f 单位。",
            "relative_reconstruction_score_percent": (
                "定义为 (1 - NRMSE) * 100%，仅为自定义相对重建得分，"
                "不是分类准确率，也不是统计学置信度。"
            ),
            "selection_bias": (
                "按总谱误差选择 best 会偏向弱峰/宽峰样本；研究峰恢复时应使用 "
                "--peak-only --rank-metric resonance_relative_l2。"
            ),
            "predicted_resonance_diagnostic": (
                "pred resonance 图使用 f_pred 减去验证样本的真实背景，仅用于诊断，"
                "不能用于未知真实数据的推断。"
            ),
        },
        "selected_rows": selected_rows,
    }
    with (output_dir / "selection_info.json").open("w", encoding="utf-8") as f:
        json.dump(selection_info, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("ranked validation complete")
    print("selected model step:", "%s" % format(checkpoint_info["selected_global_step"], ","))
    print("weights choice:", args.weights)
    print("selection:", args.selection)
    print("rank metric:", args.rank_metric)
    print("peak only:", args.peak_only)
    print("eligible peak samples:", int(np.count_nonzero(eligible_peak)))
    print("selected samples:", selected_csv_path.resolve())
    print("f plots:", plots_f_dir.resolve())
    print("g plots:", plots_g_dir.resolve())
    print("resonance plots:", plots_resonance_dir.resolve())
    if args.save_selected_data:
        print("curve data:", selected_data_dir.resolve())
    print("注意：relative reconstruction score=(1-NRMSE)*100%，不是标准 accuracy。")
    print("=" * 72)


if __name__ == "__main__":
    main()
