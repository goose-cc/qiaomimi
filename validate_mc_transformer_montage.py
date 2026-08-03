from __future__ import annotations

"""
Create four publication-style montage figures for the V2 free-curve Transformer.

The script validates the checkpoint once on a fixed validation sample, then builds:

1. all_best_montage.png
   Pick three configured ranks from the best N samples over the full validation set,
   ranked by total-spectrum relative L2 error.

2. all_typical_montage.png
   Pick three configured ranks from the N samples closest to the median total-spectrum
   relative L2 error.

3. peak_best_montage.png
   Restrict to visible/resolvable in-domain resonance samples, rank by resonance
   relative L2 error, and pick three configured ranks from the best N.

4. peak_typical_montage.png
   Restrict to visible/resolvable in-domain resonance samples, find the median
   resonance relative L2 inside that subset, and pick three configured ranks from
   the N candidates closest to that median.

Each montage uses one row per sample and four columns:
    total f(s) | resonance diagnostic | forward g(q^2) | metrics/parameters

The resonance diagnostic subtracts the TRUE validation background from f_pred.
It is only an evaluation diagnostic and is not available for an unknown real sample.

Python 3.8 compatible.
"""

import argparse
import csv
import json
import math
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


GROUP_ALL_BEST = "all_best"
GROUP_ALL_TYPICAL = "all_typical"
GROUP_PEAK_BEST = "peak_best"
GROUP_PEAK_TYPICAL = "peak_typical"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create V2 best/typical/peak-best/peak-typical montage figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--validation-pool-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--weights", choices=("best", "latest"), default="latest")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--noise-level", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        default="./validation_results/base_v2_montages",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=8,
        help="每组先选出的候选样本数",
    )
    parser.add_argument(
        "--pick-ranks",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="从每组候选中放入合并图的 1-based 候选排名",
    )
    parser.add_argument(
        "--min-resonance-visibility",
        type=float,
        default=0.10,
        help="明显峰样本要求：共振 L2 / 总谱 L2 的最低值",
    )
    parser.add_argument(
        "--min-width-grid-cells",
        type=float,
        default=1.0,
        help="明显峰样本要求：m*gamma 至少覆盖多少个输出网格间隔",
    )
    parser.add_argument(
        "--allow-outside-center",
        action="store_true",
        help="明显峰筛选时允许 m 位于 s 绘图区间外（通常不建议）",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--save-selected-data",
        dest="save_selected_data",
        action="store_true",
        help="保存合并图中每个样本的 f/resonance/g 数值 CSV",
    )
    parser.add_argument(
        "--no-save-selected-data",
        dest="save_selected_data",
        action="store_false",
    )
    parser.set_defaults(save_selected_data=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num-samples 必须大于 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")
    if args.candidate_count <= 0:
        raise ValueError("--candidate-count 必须大于 0")
    if not args.pick_ranks:
        raise ValueError("--pick-ranks 至少需要一个排名")
    if len(set(args.pick_ranks)) != len(args.pick_ranks):
        raise ValueError("--pick-ranks 不能重复")
    for rank in args.pick_ranks:
        if rank < 1 or rank > args.candidate_count:
            raise ValueError(
                "--pick-ranks 中的排名必须位于 1..candidate-count；收到 %d" % rank
            )
    if args.dpi <= 0:
        raise ValueError("--dpi 必须大于 0")


def select_best(metric: np.ndarray, count: int, eligible: np.ndarray = None) -> List[int]:
    if eligible is None:
        pool = np.arange(metric.size)
    else:
        pool = np.asarray(eligible, dtype=np.int64)
    if pool.size == 0:
        return []
    count = min(int(count), int(pool.size))
    local_order = np.argsort(metric[pool])
    return [int(pool[j]) for j in local_order[:count]]


def select_typical(
    metric: np.ndarray,
    count: int,
    eligible: np.ndarray = None,
) -> List[int]:
    """Return samples closest to the median inside the selected population.

    If ``eligible`` is None, the population is the full metric array. Otherwise,
    the median is computed only over those eligible sample indices. Returned
    indices always refer to the original validation array.
    """
    if eligible is None:
        pool = np.arange(metric.size, dtype=np.int64)
    else:
        pool = np.asarray(eligible, dtype=np.int64)
    if pool.size == 0:
        return []
    count = min(int(count), int(pool.size))
    median = float(np.median(metric[pool]))
    distance = np.abs(metric[pool] - median)
    # Stable tie-break: lower metric first, then original sample index.
    local_order = np.lexsort((pool, metric[pool], distance))
    return [int(pool[j]) for j in local_order[:count]]


def pick_from_candidates(candidates: Sequence[int], pick_ranks: Sequence[int]) -> List[int]:
    return [int(candidates[rank - 1]) for rank in pick_ranks]


def safe_relative_l2_np(pred: np.ndarray, target: np.ndarray, axis: int = 1) -> np.ndarray:
    numerator = np.linalg.norm(pred - target, axis=axis)
    denominator = np.maximum(np.linalg.norm(target, axis=axis), 1e-12)
    return numerator / denominator


def write_curve_csv(path: Path, headers: Sequence[str], columns: Sequence[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack(columns)
    np.savetxt(
        str(path),
        data,
        delimiter=",",
        header=",".join(headers),
        comments="",
        fmt="%.10g",
    )


def make_record(
    *,
    group: str,
    candidate_rank: int,
    sample_index: int,
    validation_rows: np.ndarray,
    params_np: np.ndarray,
    arrays: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    i = int(sample_index)
    params = params_np[i]
    record: Dict[str, Any] = {
        "group": group,
        "candidate_rank": int(candidate_rank),
        "validation_row_id": int(validation_rows[i]),
        "sample_index_in_validation": i,
        "a1": float(params[0]),
        "a2": float(params[1]),
        "a3": float(params[2]),
        "m": float(params[3]),
        "gamma": float(params[4]),
        "m_gamma": float(params[3] * params[4]),
    }
    scalar_names = (
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
        "resonance_visibility",
        "width_grid_cells",
        "resonance_mse_scaled",
        "resonance_rmse_scaled",
        "resonance_relative_l2",
        "true_peak_s_grid",
        "pred_peak_s_grid",
        "peak_center_abs_error",
        "peak_height_relative_error",
    )
    for name in scalar_names:
        record[name] = float(arrays[name][i])
    return record


def metrics_text(record: Dict[str, Any], include_peak: bool = True) -> str:
    lines = [
        "row=%d | candidate rank=%d" % (
            record["validation_row_id"], record["candidate_rank"]
        ),
        "",
        "TOTAL SPECTRUM",
        "MSE(scaled)   = %.6g" % record["f_mse_scaled"],
        "RMSE(scaled)  = %.6g" % record["f_rmse_scaled"],
        "NRMSE/rel-L2  = %.4f" % record["f_relative_l2"],
        "recon score   = %.2f%%" % record["relative_reconstruction_score_percent"],
        "",
        "FORWARD g",
        "g rel clean   = %.5f" % record["g_relative_l2_vs_clean"],
        "repr. gap     = %.5f" % record["clean_discrete_representation_gap"],
    ]
    if include_peak:
        lines.extend([
            "",
            "RESONANCE DIAGNOSTIC",
            "res rel-L2    = %.4f" % record["resonance_relative_l2"],
            "peak center Δ = %.4f" % record["peak_center_abs_error"],
            "peak height Δ = %.4f" % record["peak_height_relative_error"],
            "visibility    = %.4f" % record["resonance_visibility"],
            "width/grid    = %.2f" % record["width_grid_cells"],
        ])
    lines.extend([
        "",
        "PARAMETERS",
        "a1=%.5g  a2=%.5g" % (record["a1"], record["a2"]),
        "a3=%.5g" % record["a3"],
        "m=%.5g  gamma=%.5g" % (record["m"], record["gamma"]),
        "m*gamma=%.5g" % record["m_gamma"],
    ])
    return "\n".join(lines)


def plot_montage(
    *,
    group: str,
    selected: Sequence[int],
    candidate_ranks: Sequence[int],
    validation_rows: np.ndarray,
    params_np: np.ndarray,
    s_grid: np.ndarray,
    q2_grid: np.ndarray,
    data: Dict[str, np.ndarray],
    output_path: Path,
    dpi: int,
) -> List[Dict[str, Any]]:
    n_rows = len(selected)
    if n_rows == 0:
        raise RuntimeError("组 %s 没有选中样本" % group)

    group_titles = {
        GROUP_ALL_BEST: (
            "Full validation set: selected samples from the best candidates "
            "by total-spectrum relative L2"
        ),
        GROUP_ALL_TYPICAL: (
            "Full validation set: selected samples from candidates closest "
            "to the median total-spectrum relative L2"
        ),
        GROUP_PEAK_BEST: (
            "Visible-peak subset: selected samples from the best candidates "
            "by resonance relative L2"
        ),
        GROUP_PEAK_TYPICAL: (
            "Visible-peak subset: selected samples from candidates closest "
            "to the median resonance relative L2"
        ),
    }

    fig_height = max(4.0 * n_rows, 7.5)
    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(23.0, fig_height),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.25, 1.25, 1.25, 0.90]},
    )
    records: List[Dict[str, Any]] = []

    for row_pos, (i, candidate_rank) in enumerate(zip(selected, candidate_ranks)):
        record = make_record(
            group=group,
            candidate_rank=int(candidate_rank),
            sample_index=int(i),
            validation_rows=validation_rows,
            params_np=params_np,
            arrays=data,
        )
        records.append(record)
        row_id = record["validation_row_id"]

        # Column 1: total spectrum.
        ax_f = axes[row_pos, 0]
        ax_f.plot(s_grid, data["f_true"][i], label="f_true", linewidth=2.0)
        ax_f.plot(s_grid, data["f_pred"][i], label="f_pred", linewidth=2.0)
        ax_f.set_xlabel("s")
        ax_f.set_ylabel("scaled f(s)")
        ax_f.grid(True, alpha=0.3)
        ax_f.legend(fontsize=8)
        ax_f.set_title(
            "Total spectrum | candidate rank=%d | row=%d\n"
            "MSE=%.4g, RMSE=%.4g, NRMSE=%.4f, score=%.2f%%" % (
                candidate_rank,
                row_id,
                record["f_mse_scaled"],
                record["f_rmse_scaled"],
                record["f_relative_l2"],
                record["relative_reconstruction_score_percent"],
            ),
            fontsize=10,
        )

        # Column 2: diagnostic resonance component.
        ax_r = axes[row_pos, 1]
        ax_r.plot(
            s_grid,
            data["resonance_true"][i],
            label="true resonance",
            linewidth=2.0,
        )
        ax_r.plot(
            s_grid,
            data["resonance_pred_diag"][i],
            label="f_pred - true background",
            linewidth=2.0,
        )
        ax_r.axvline(
            record["true_peak_s_grid"], linestyle="--", linewidth=1.1,
            label="true peak grid",
        )
        ax_r.axvline(
            record["pred_peak_s_grid"], linestyle=":", linewidth=1.1,
            label="pred peak grid",
        )
        ax_r.set_xlabel("s")
        ax_r.set_ylabel("scaled resonance")
        ax_r.grid(True, alpha=0.3)
        ax_r.legend(fontsize=7)
        ax_r.set_title(
            "Resonance diagnostic\n"
            "rel-L2=%.4f, center Δ=%.4f, height Δ=%.4f" % (
                record["resonance_relative_l2"],
                record["peak_center_abs_error"],
                record["peak_height_relative_error"],
            ),
            fontsize=10,
        )

        # Column 3: forward g consistency.
        ax_g = axes[row_pos, 2]
        ax_g.plot(q2_grid, data["g_noisy"][i], label="g_noisy input", linewidth=1.2)
        ax_g.plot(q2_grid, data["g_clean"][i], label="g_clean", linewidth=2.0)
        ax_g.plot(q2_grid, data["g_pred"][i], label="K(f_pred)", linewidth=2.0)
        ax_g.plot(
            q2_grid,
            data["g_true_discrete"][i],
            label="K(f_true_100)",
            linewidth=1.2,
        )
        ax_g.set_xlabel("q^2")
        ax_g.set_ylabel("scaled g(q^2)")
        ax_g.grid(True, alpha=0.3)
        ax_g.legend(fontsize=7)
        ax_g.set_title(
            "Forward consistency\n"
            "g rel clean=%.5f, representation gap=%.5f" % (
                record["g_relative_l2_vs_clean"],
                record["clean_discrete_representation_gap"],
            ),
            fontsize=10,
        )

        # Column 4: metrics and physical parameters.
        ax_t = axes[row_pos, 3]
        ax_t.axis("off")
        ax_t.text(
            0.01,
            0.99,
            metrics_text(record, include_peak=True),
            transform=ax_t.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
        )

    fig.suptitle(group_titles[group], fontsize=16, y=0.997)
    fig.text(
        0.5,
        0.004,
        "Note: reconstruction score=(1-NRMSE)*100% is a custom score, not classification accuracy. "
        "The resonance panel subtracts the known true validation background and is diagnostic only.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.01, 0.018, 0.995, 0.982))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return records


def save_selected_curves(
    *,
    output_dir: Path,
    group: str,
    selected: Sequence[int],
    candidate_ranks: Sequence[int],
    validation_rows: np.ndarray,
    s_grid: np.ndarray,
    q2_grid: np.ndarray,
    data: Dict[str, np.ndarray],
) -> None:
    group_dir = output_dir / "selected_data" / group
    group_dir.mkdir(parents=True, exist_ok=True)
    for i, candidate_rank in zip(selected, candidate_ranks):
        row_id = int(validation_rows[i])
        stem = "candidate_%02d_row_%d" % (int(candidate_rank), row_id)
        write_curve_csv(
            group_dir / (stem + "_f.csv"),
            ("s", "f_true", "f_pred", "error", "abs_error", "squared_error"),
            (
                s_grid,
                data["f_true"][i],
                data["f_pred"][i],
                data["f_pred"][i] - data["f_true"][i],
                np.abs(data["f_pred"][i] - data["f_true"][i]),
                (data["f_pred"][i] - data["f_true"][i]) ** 2,
            ),
        )
        write_curve_csv(
            group_dir / (stem + "_resonance.csv"),
            (
                "s",
                "true_background",
                "true_resonance",
                "pred_resonance_diagnostic",
                "resonance_error",
            ),
            (
                s_grid,
                data["background_true"][i],
                data["resonance_true"][i],
                data["resonance_pred_diag"][i],
                data["resonance_pred_diag"][i] - data["resonance_true"][i],
            ),
        )
        write_curve_csv(
            group_dir / (stem + "_g.csv"),
            ("q2", "g_noisy", "g_clean", "g_from_f_pred", "g_from_f_true_100"),
            (
                q2_grid,
                data["g_noisy"][i],
                data["g_clean"][i],
                data["g_pred"][i],
                data["g_true_discrete"][i],
            ),
        )


def write_records_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_fixed_seed(args.seed)

    device = choose_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    validation_pool_dir = Path(args.validation_pool_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    if sample_count < args.candidate_count:
        raise ValueError(
            "验证样本数 %d 小于 candidate-count %d" %
            (sample_count, args.candidate_count)
        )

    rng = np.random.default_rng(args.seed)
    noise_rng = np.random.default_rng(args.seed + 1)
    validation_rows = np.sort(
        rng.choice(usable_rows, size=sample_count, replace=False)
    )
    params_np = np.asarray(pool[validation_rows], dtype=np.float32)

    all_f_true: List[np.ndarray] = []
    all_f_pred: List[np.ndarray] = []
    all_g_clean: List[np.ndarray] = []
    all_g_noisy: List[np.ndarray] = []
    all_g_pred: List[np.ndarray] = []
    all_g_true_discrete: List[np.ndarray] = []

    print("=" * 72)
    print("开始 V2 montage validation")
    print("设备:", device)
    print("权重:", checkpoint_info["weights_path"])
    print("selected model step:", "%s" % format(checkpoint_info["selected_global_step"], ","))
    print("验证参数池:", validation_pool_dir.resolve())
    print("验证样本数:", "%s" % format(sample_count, ","))
    print("固定随机种子:", args.seed)
    print("噪声比例: %.2f%%" % (100.0 * validation_args["noise_level"]))
    print("candidate count:", args.candidate_count)
    print("pick ranks:", args.pick_ranks)
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

    # Total-spectrum metrics.
    f_true64 = f_true_arr.astype(np.float64)
    f_pred64 = f_pred_arr.astype(np.float64)
    g_clean64 = g_clean_arr.astype(np.float64)
    g_pred64 = g_pred_arr.astype(np.float64)
    g_true_discrete64 = g_true_discrete_arr.astype(np.float64)

    f_error = f_pred64 - f_true64
    f_mse = np.mean(f_error ** 2, axis=1)
    f_rmse = np.sqrt(f_mse)
    f_true_rms = np.sqrt(np.mean(f_true64 ** 2, axis=1))
    f_nrmse = f_rmse / np.maximum(f_true_rms, 1e-12)
    f_rel = safe_relative_l2_np(f_pred64, f_true64)
    reconstruction_score = 100.0 * (1.0 - f_nrmse)

    g_error = g_pred64 - g_clean64
    g_mse = np.mean(g_error ** 2, axis=1)
    g_rmse = np.sqrt(g_mse)
    g_rel_clean = safe_relative_l2_np(g_pred64, g_clean64)
    g_rel_discrete = safe_relative_l2_np(g_pred64, g_true_discrete64)
    representation_gap = safe_relative_l2_np(g_true_discrete64, g_clean64)

    # Peak-aware diagnostics using the known validation parameters.
    s_min = float(saved_args.get("s_min", 0.1764))
    s_max = float(saved_args.get("s_max", 6.0))
    shift = float(saved_args.get("shift", 400.0))
    data_scale = float(saved_args.get("data_scale", 160000.0))
    output_points = int(saved_args.get("output_points", 100))
    input_points = int(saved_args.get("input_points", 100))
    q2_min = float(saved_args.get("q2_min", -100.0))
    q2_max = float(saved_args.get("q2_max", -6.0))

    s_grid = np.linspace(s_min, s_max, output_points)
    q2_grid = np.linspace(q2_min, q2_max, input_points)
    ds = float((s_max - s_min) / max(output_points - 1, 1))

    a1 = params_np[:, 0:1].astype(np.float64)
    a2 = params_np[:, 1:2].astype(np.float64)
    a3 = params_np[:, 2:3].astype(np.float64)
    m = params_np[:, 3:4].astype(np.float64)
    gamma = params_np[:, 4:5].astype(np.float64)
    width = m * gamma
    s_row = s_grid.reshape(1, -1)

    background_true = data_scale * (a2 * s_row + a3) / (s_row + shift) ** 2
    resonance_true = (
        data_scale
        * (a1 / np.pi)
        * width
        / ((s_row - m) ** 2 + width ** 2)
        / (s_row + shift) ** 2
    )
    resonance_pred_diag = f_pred64 - background_true
    resonance_error = resonance_pred_diag - resonance_true
    resonance_mse = np.mean(resonance_error ** 2, axis=1)
    resonance_rmse = np.sqrt(resonance_mse)
    resonance_rel = safe_relative_l2_np(resonance_pred_diag, resonance_true)
    resonance_visibility = np.linalg.norm(resonance_true, axis=1) / np.maximum(
        np.linalg.norm(f_true64, axis=1), 1e-12
    )

    true_peak_idx = np.argmax(resonance_true, axis=1)
    pred_peak_idx = np.argmax(resonance_pred_diag, axis=1)
    true_peak_s_grid = s_grid[true_peak_idx]
    pred_peak_s_grid = s_grid[pred_peak_idx]
    peak_center_abs_error = np.abs(pred_peak_s_grid - true_peak_s_grid)
    true_peak_height = resonance_true[np.arange(sample_count), true_peak_idx]
    pred_height_at_true_peak = resonance_pred_diag[np.arange(sample_count), true_peak_idx]
    peak_height_relative_error = np.abs(
        pred_height_at_true_peak - true_peak_height
    ) / np.maximum(np.abs(true_peak_height), 1e-12)

    width_1d = width[:, 0]
    width_grid_cells = width_1d / max(ds, 1e-12)
    center_in_domain = (params_np[:, 3] >= s_min) & (params_np[:, 3] <= s_max)
    eligible_peak = (
        (resonance_visibility >= float(args.min_resonance_visibility))
        & (width_grid_cells >= float(args.min_width_grid_cells))
    )
    if not args.allow_outside_center:
        eligible_peak &= center_in_domain
    eligible_peak_indices = np.flatnonzero(eligible_peak)
    if eligible_peak_indices.size < args.candidate_count:
        raise RuntimeError(
            "明显峰样本只有 %d 个，小于 candidate-count=%d。请降低 "
            "--min-resonance-visibility 或 --min-width-grid-cells。" %
            (eligible_peak_indices.size, args.candidate_count)
        )

    data: Dict[str, np.ndarray] = {
        "f_true": f_true_arr,
        "f_pred": f_pred_arr,
        "g_clean": g_clean_arr,
        "g_noisy": g_noisy_arr,
        "g_pred": g_pred_arr,
        "g_true_discrete": g_true_discrete_arr,
        "background_true": background_true,
        "resonance_true": resonance_true,
        "resonance_pred_diag": resonance_pred_diag,
        "f_mse_scaled": f_mse,
        "f_rmse_scaled": f_rmse,
        "f_nrmse_rms": f_nrmse,
        "f_relative_l2": f_rel,
        "relative_reconstruction_score_percent": reconstruction_score,
        "g_mse_vs_clean_scaled": g_mse,
        "g_rmse_vs_clean_scaled": g_rmse,
        "g_relative_l2_vs_clean": g_rel_clean,
        "g_relative_l2_vs_discrete_clean": g_rel_discrete,
        "clean_discrete_representation_gap": representation_gap,
        "resonance_visibility": resonance_visibility,
        "width_grid_cells": width_grid_cells,
        "resonance_mse_scaled": resonance_mse,
        "resonance_rmse_scaled": resonance_rmse,
        "resonance_relative_l2": resonance_rel,
        "true_peak_s_grid": true_peak_s_grid,
        "pred_peak_s_grid": pred_peak_s_grid,
        "peak_center_abs_error": peak_center_abs_error,
        "peak_height_relative_error": peak_height_relative_error,
    }

    # Build the four candidate sets (each normally contains 8 samples).
    candidates_by_group: Dict[str, List[int]] = {
        GROUP_ALL_BEST: select_best(f_rel, args.candidate_count),
        GROUP_ALL_TYPICAL: select_typical(f_rel, args.candidate_count),
        GROUP_PEAK_BEST: select_best(
            resonance_rel,
            args.candidate_count,
            eligible=eligible_peak_indices,
        ),
        GROUP_PEAK_TYPICAL: select_typical(
            resonance_rel,
            args.candidate_count,
            eligible=eligible_peak_indices,
        ),
    }
    selected_by_group: Dict[str, List[int]] = {}
    for group, candidates in candidates_by_group.items():
        if len(candidates) < args.candidate_count:
            raise RuntimeError(
                "组 %s 只有 %d 个候选，少于 %d" %
                (group, len(candidates), args.candidate_count)
            )
        selected_by_group[group] = pick_from_candidates(candidates, args.pick_ranks)

    # Save all candidate rows, not only the three shown in each montage.
    candidate_records: List[Dict[str, Any]] = []
    for group, candidates in candidates_by_group.items():
        for candidate_rank, i in enumerate(candidates, start=1):
            candidate_records.append(
                make_record(
                    group=group,
                    candidate_rank=candidate_rank,
                    sample_index=i,
                    validation_rows=validation_rows,
                    params_np=params_np,
                    arrays=data,
                )
            )
    write_records_csv(output_dir / "candidate_sets.csv", candidate_records)

    montage_paths = {
        GROUP_ALL_BEST: output_dir / "01_all_best_montage.png",
        GROUP_ALL_TYPICAL: output_dir / "02_all_typical_montage.png",
        GROUP_PEAK_BEST: output_dir / "03_peak_best_montage.png",
        GROUP_PEAK_TYPICAL: output_dir / "04_peak_typical_montage.png",
    }

    selected_records: List[Dict[str, Any]] = []
    for group in (
        GROUP_ALL_BEST,
        GROUP_ALL_TYPICAL,
        GROUP_PEAK_BEST,
        GROUP_PEAK_TYPICAL,
    ):
        group_records = plot_montage(
            group=group,
            selected=selected_by_group[group],
            candidate_ranks=args.pick_ranks,
            validation_rows=validation_rows,
            params_np=params_np,
            s_grid=s_grid,
            q2_grid=q2_grid,
            data=data,
            output_path=montage_paths[group],
            dpi=args.dpi,
        )
        selected_records.extend(group_records)
        if args.save_selected_data:
            save_selected_curves(
                output_dir=output_dir,
                group=group,
                selected=selected_by_group[group],
                candidate_ranks=args.pick_ranks,
                validation_rows=validation_rows,
                s_grid=s_grid,
                q2_grid=q2_grid,
                data=data,
            )

    write_records_csv(output_dir / "selected_samples.csv", selected_records)

    summary = {
        "checkpoint": checkpoint_info,
        "validation_pool": str(validation_pool_dir.resolve()),
        "validation_samples": int(sample_count),
        "noise_level": float(validation_args["noise_level"]),
        "seed": int(args.seed),
        "candidate_count": int(args.candidate_count),
        "pick_ranks": [int(x) for x in args.pick_ranks],
        "eligible_peak_samples": int(eligible_peak_indices.size),
        "min_resonance_visibility": float(args.min_resonance_visibility),
        "min_width_grid_cells": float(args.min_width_grid_cells),
        "allow_outside_center": bool(args.allow_outside_center),
        "montages": {k: str(v.resolve()) for k, v in montage_paths.items()},
        "candidate_sets_csv": str((output_dir / "candidate_sets.csv").resolve()),
        "selected_samples_csv": str((output_dir / "selected_samples.csv").resolve()),
        "selected_data_dir": (
            str((output_dir / "selected_data").resolve())
            if args.save_selected_data else None
        ),
        "notes": {
            "all_best": (
                "全验证集按完整谱 f_relative_l2 从低到高取前 candidate_count，"
                "再按 pick_ranks 选入合并图。"
            ),
            "all_typical": (
                "全验证集中按与 f_relative_l2 中位数的距离从近到远取 candidate_count，"
                "再按 pick_ranks 选入合并图。"
            ),
            "peak_best": (
                "先筛选峰中心在区间内、共振可见且宽度可分辨的样本，"
                "再按 resonance_relative_l2 从低到高取前 candidate_count。"
            ),
            "peak_typical": (
                "先使用与 peak_best 相同的可见峰筛选，再在该子集内计算 "
                "resonance_relative_l2 的中位数；按与这个中位数的距离从近到远 "
                "取 candidate_count，再按 pick_ranks 选入合并图。"
            ),
            "resonance_diagnostic": (
                "预测共振使用 f_pred 减去验证样本真实背景，仅用于已知真值的诊断，"
                "不是未知样本可直接使用的推断方法。"
            ),
            "reconstruction_score": (
                "定义为 (1-NRMSE)*100%，是自定义相对重建得分，不是分类准确率。"
            ),
        },
    }
    with (output_dir / "montage_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("montage validation complete")
    print("selected model step:", "%s" % format(checkpoint_info["selected_global_step"], ","))
    print("weights choice:", args.weights)
    print("eligible peak samples:", int(eligible_peak_indices.size))
    print("all best montage:", montage_paths[GROUP_ALL_BEST].resolve())
    print("all typical montage:", montage_paths[GROUP_ALL_TYPICAL].resolve())
    print("peak best montage:", montage_paths[GROUP_PEAK_BEST].resolve())
    print("peak typical montage:", montage_paths[GROUP_PEAK_TYPICAL].resolve())
    print("candidate sets:", (output_dir / "candidate_sets.csv").resolve())
    print("selected samples:", (output_dir / "selected_samples.csv").resolve())
    if args.save_selected_data:
        print("selected curve data:", (output_dir / "selected_data").resolve())
    print("=" * 72)


if __name__ == "__main__":
    main()
