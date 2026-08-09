#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp7 follow-up diagnostic: parameter error conditioned on true resonance visibility.

This script does NOT retrain the model and does NOT modify the original loss.
It loads an existing Exp7 checkpoint, re-runs validation, and answers:

    Are m/gamma only bad on weak-resonance samples, or are they also bad
    when the resonance is clearly visible?

Outputs:
- visibility_binned_summary.csv
- visibility_binned_summary.json
- visibility_per_sample.csv
- m_gamma_error_vs_visibility.png
- parameter_error_by_visibility.png
- true_vs_pred_visibility.png

The visibility definition is intentionally the same one already present in
mc_parametric.py:
    RMS(resonance_component) / RMS(total_spectrum)
computed on the Exp7 output s-grid.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from TransformerInverse import ParametricInverseTransformer1D
from mc_online_physics import OnlinePhysics
from mc_parametric import normalize_parameters, resonance_visibility
from train_mc_parameter_pool_transformer_loss import choose_device, open_parameter_pool


PARAMETER_NAMES = ("a1", "a2", "a3", "m", "gamma")
DEFAULT_VISIBILITY_BINS = (0.0, 0.01, 0.05, 0.10, 0.20, 0.40, 0.70, 1.000001)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze Exp7 parameter errors as a function of true resonance visibility."
    )
    p.add_argument("--validation-pool-dir", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--weights", choices=("best", "latest"), default="best")
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--noise-level", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20260802)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    p.add_argument(
        "--visibility-bins",
        default="0,0.01,0.05,0.10,0.20,0.40,0.70,1.000001",
        help="Comma-separated bin edges. The final edge should be >1 because visibility is clamped to 1.",
    )
    p.add_argument(
        "--weak-threshold",
        type=float,
        default=0.05,
        help="True visibility <= this value is reported as weak resonance.",
    )
    p.add_argument(
        "--visible-threshold",
        type=float,
        default=0.20,
        help="True visibility >= this value is reported as clearly visible resonance.",
    )
    p.add_argument(
        "--scatter-max-points",
        type=int,
        default=20000,
        help="Maximum points used in scatter plots; metrics always use all samples.",
    )
    return p.parse_args()


def parse_visibility_bins(text: str) -> np.ndarray:
    try:
        edges = np.asarray([float(x.strip()) for x in text.split(",") if x.strip()], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("--visibility-bins must contain only numbers") from exc
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("--visibility-bins needs at least two edges")
    if not np.all(np.isfinite(edges)):
        raise ValueError("--visibility-bins contains non-finite values")
    if not np.all(edges[1:] > edges[:-1]):
        raise ValueError("--visibility-bins must be strictly increasing")
    if edges[0] > 0.0 or edges[-1] <= 1.0:
        raise ValueError("visibility bins must start at <=0 and end at >1")
    return edges


def load_training_args(checkpoint_dir: Path) -> Tuple[dict, dict]:
    latest_path = checkpoint_dir / "latest_checkpoint.pth"
    if not latest_path.exists():
        raise FileNotFoundError(latest_path)
    try:
        latest_ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
    except TypeError:
        latest_ckpt = torch.load(latest_path, map_location="cpu")
    return latest_ckpt, dict(latest_ckpt.get("args", {}))


def build_model(saved: dict, device: torch.device) -> ParametricInverseTransformer1D:
    return ParametricInverseTransformer1D(
        input_length=int(saved.get("input_points", 100)),
        output_length=int(saved.get("output_points", 1000)),
        d_model=int(saved.get("transformer_d_model", 64)),
        nhead=int(saved.get("transformer_nhead", 4)),
        num_encoder_layers=int(saved.get("transformer_num_layers", 3)),
        dim_feedforward=int(saved.get("transformer_dim_feedforward", 128)),
        dropout=float(saved.get("transformer_dropout", 0.1)),
        y_min=float(saved.get("q2_min", -100.0)),
        y_max=float(saved.get("q2_max", -6.0)),
        x_min=float(saved.get("s_min", 0.1764)),
        x_max=float(saved.get("s_max", 6.0)),
        shift=float(saved.get("shift", 400.0)),
        data_scale=float(saved.get("data_scale", 160000.0)),
    ).to(device)


def make_physics(saved: dict, noise_level: float, device: torch.device) -> OnlinePhysics:
    physics_args = SimpleNamespace(
        physics_dtype=str(saved.get("physics_dtype", "float32")),
        noise_level=float(noise_level),
        data_scale=float(saved.get("data_scale", 160000.0)),
        shift=float(saved.get("shift", 400.0)),
        s_min=float(saved.get("s_min", 0.1764)),
        s_max=float(saved.get("s_max", 6.0)),
        q2_min=float(saved.get("q2_min", -100.0)),
        q2_max=float(saved.get("q2_max", -6.0)),
        output_points=int(saved.get("output_points", 1000)),
        input_points=int(saved.get("input_points", 100)),
        integration_points=int(saved.get("integration_points", 128)),
    )
    return OnlinePhysics(physics_args, device)


def rel_l2(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = torch.linalg.vector_norm(pred - true, dim=1)
    den = torch.linalg.vector_norm(true, dim=1).clamp_min(eps)
    return num / den


def _stat(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def visibility_bin_indices(visibility: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return bin id in [0, len(edges)-2], or -1 if outside the configured range."""
    visibility = np.asarray(visibility, dtype=np.float64)
    ids = np.searchsorted(edges, visibility, side="right") - 1
    valid = (ids >= 0) & (ids < len(edges) - 1)
    result = np.full(visibility.shape, -1, dtype=np.int64)
    result[valid] = ids[valid]
    return result


def summarize_bins(
    *,
    edges: np.ndarray,
    true_visibility: np.ndarray,
    pred_visibility: np.ndarray,
    norm_abs: np.ndarray,
    log_gamma_abs: np.ndarray,
    width_rel: np.ndarray,
    f_rel: np.ndarray,
    g_rel: np.ndarray,
) -> List[dict]:
    bin_ids = visibility_bin_indices(true_visibility, edges)
    rows: List[dict] = []
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        count = int(np.sum(mask))
        row = {
            "bin_id": b,
            "visibility_lo": float(edges[b]),
            "visibility_hi": float(edges[b + 1]),
            "count": count,
            "fraction": float(count / len(true_visibility)) if len(true_visibility) else 0.0,
        }
        if count:
            row["true_visibility_mean"] = float(np.mean(true_visibility[mask]))
            row["pred_visibility_mean"] = float(np.mean(pred_visibility[mask]))
            row["visibility_abs_error_mean"] = float(
                np.mean(np.abs(pred_visibility[mask] - true_visibility[mask]))
            )
            for k, name in enumerate(PARAMETER_NAMES):
                row[f"{name}_norm_abs_mean"] = float(np.mean(norm_abs[mask, k]))
                row[f"{name}_norm_abs_median"] = float(np.median(norm_abs[mask, k]))
            row["log_gamma_abs_mean"] = float(np.mean(log_gamma_abs[mask]))
            row["log_gamma_abs_median"] = float(np.median(log_gamma_abs[mask]))
            row["width_rel_mean"] = float(np.mean(width_rel[mask]))
            row["f_rel_l2_mean"] = float(np.mean(f_rel[mask]))
            row["g_rel_clean_mean"] = float(np.mean(g_rel[mask]))
        else:
            row["true_visibility_mean"] = float("nan")
            row["pred_visibility_mean"] = float("nan")
            row["visibility_abs_error_mean"] = float("nan")
            for name in PARAMETER_NAMES:
                row[f"{name}_norm_abs_mean"] = float("nan")
                row[f"{name}_norm_abs_median"] = float("nan")
            row["log_gamma_abs_mean"] = float("nan")
            row["log_gamma_abs_median"] = float("nan")
            row["width_rel_mean"] = float("nan")
            row["f_rel_l2_mean"] = float("nan")
            row["g_rel_clean_mean"] = float("nan")
        rows.append(row)
    return rows


def threshold_summary(
    mask: np.ndarray,
    *,
    true_visibility: np.ndarray,
    pred_visibility: np.ndarray,
    norm_abs: np.ndarray,
    log_gamma_abs: np.ndarray,
    width_rel: np.ndarray,
    f_rel: np.ndarray,
    g_rel: np.ndarray,
) -> dict:
    count = int(np.sum(mask))
    result = {
        "count": count,
        "fraction": float(count / len(mask)) if len(mask) else 0.0,
    }
    if count == 0:
        return result
    result["true_visibility"] = _stat(true_visibility[mask])
    result["pred_visibility"] = _stat(pred_visibility[mask])
    result["visibility_abs_error"] = _stat(
        np.abs(pred_visibility[mask] - true_visibility[mask])
    )
    result["parameters"] = {
        name: _stat(norm_abs[mask, k]) for k, name in enumerate(PARAMETER_NAMES)
    }
    result["log_gamma_abs_error"] = _stat(log_gamma_abs[mask])
    result["width_relative_error"] = _stat(width_rel[mask])
    result["f_relative_l2"] = _stat(f_rel[mask])
    result["g_relative_l2_vs_clean"] = _stat(g_rel[mask])
    return result


def save_bin_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_m_gamma_vs_visibility(
    out: Path,
    true_visibility: np.ndarray,
    norm_abs: np.ndarray,
    bin_rows: Sequence[dict],
    max_points: int,
    seed: int,
) -> None:
    n = len(true_visibility)
    if n == 0:
        return
    rng = np.random.default_rng(seed)
    if n > max_points:
        ids = np.sort(rng.choice(n, size=max_points, replace=False))
    else:
        ids = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(true_visibility[ids], norm_abs[ids, 3], s=8, alpha=0.18, label="m normalized abs error")
    ax.scatter(true_visibility[ids], norm_abs[ids, 4], s=8, alpha=0.18, label="gamma normalized abs error")

    centers = []
    m_means = []
    g_means = []
    for row in bin_rows:
        if row["count"] <= 0:
            continue
        centers.append(row["true_visibility_mean"])
        m_means.append(row["m_norm_abs_mean"])
        g_means.append(row["gamma_norm_abs_mean"])
    if centers:
        ax.plot(centers, m_means, marker="o", linewidth=2.0, label="m bin mean")
        ax.plot(centers, g_means, marker="o", linewidth=2.0, label="gamma bin mean")

    ax.set_xlabel("true resonance visibility")
    ax.set_ylabel("normalized absolute error")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Exp7: m/gamma error vs true resonance visibility")
    fig.tight_layout()
    fig.savefig(out / "m_gamma_error_vs_visibility.png", dpi=170)
    plt.close(fig)


def plot_parameter_errors_by_bin(out: Path, bin_rows: Sequence[dict]) -> None:
    usable = [row for row in bin_rows if row["count"] > 0]
    if not usable:
        return
    x = np.arange(len(usable), dtype=np.float64)
    labels = [
        f"[{row['visibility_lo']:.2g},{row['visibility_hi']:.2g})\nN={row['count']}"
        for row in usable
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    for name in PARAMETER_NAMES:
        y = [row[f"{name}_norm_abs_mean"] for row in usable]
        ax.plot(x, y, marker="o", label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean normalized absolute error")
    ax.set_xlabel("true resonance visibility bin")
    ax.grid(alpha=0.25)
    ax.legend(ncol=5)
    ax.set_title("Exp7: parameter error by true resonance visibility")
    fig.tight_layout()
    fig.savefig(out / "parameter_error_by_visibility.png", dpi=170)
    plt.close(fig)


def plot_true_vs_pred_visibility(
    out: Path,
    true_visibility: np.ndarray,
    pred_visibility: np.ndarray,
    max_points: int,
    seed: int,
) -> None:
    n = len(true_visibility)
    if n == 0:
        return
    rng = np.random.default_rng(seed + 1)
    if n > max_points:
        ids = np.sort(rng.choice(n, size=max_points, replace=False))
    else:
        ids = np.arange(n)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(true_visibility[ids], pred_visibility[ids], s=8, alpha=0.22)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)
    ax.set_xlabel("true resonance visibility")
    ax.set_ylabel("predicted resonance visibility")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.set_title("Exp7: true vs predicted resonance visibility")
    fig.tight_layout()
    fig.savefig(out / "true_vs_pred_visibility.png", dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.noise_level < 0:
        raise ValueError("--noise-level must be >= 0")
    if not 0 <= args.weak_threshold <= 1:
        raise ValueError("--weak-threshold must be in [0,1]")
    if not 0 <= args.visible_threshold <= 1:
        raise ValueError("--visible-threshold must be in [0,1]")
    if args.weak_threshold >= args.visible_threshold:
        raise ValueError("--weak-threshold must be smaller than --visible-threshold")

    edges = parse_visibility_bins(args.visibility_bins)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    latest_ckpt, saved = load_training_args(checkpoint_dir)
    model = build_model(saved, device)

    if args.weights == "best":
        weight_path = checkpoint_dir / "best_model.pth"
        if not weight_path.exists():
            raise FileNotFoundError(weight_path)
        try:
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(weight_path, map_location="cpu")
        info_path = checkpoint_dir / "best_model_info.json"
        selected_step = (
            json.loads(info_path.read_text(encoding="utf-8")).get("global_step")
            if info_path.exists()
            else None
        )
    else:
        state = latest_ckpt["model_state_dict"]
        selected_step = latest_ckpt.get("global_step")

    model.load_state_dict(state)
    model.eval()

    pool, _, usable = open_parameter_pool(
        Path(args.validation_pool_dir), require_complete=False
    )
    n = min(int(args.num_samples), int(usable))
    true_np = np.array(pool[:n], dtype=np.float32, copy=True)

    physics = make_physics(saved, args.noise_level, device)
    rng = np.random.default_rng(args.seed)

    all_true: List[torch.Tensor] = []
    all_pred: List[torch.Tensor] = []
    f_errs: List[torch.Tensor] = []
    g_errs: List[torch.Tensor] = []
    width_rels: List[torch.Tensor] = []
    true_vis_list: List[torch.Tensor] = []
    pred_vis_list: List[torch.Tensor] = []

    cursor = 0
    with torch.no_grad():
        while cursor < n:
            stop = min(cursor + args.batch_size, n)
            true_params = torch.from_numpy(true_np[cursor:stop]).to(device)

            true_f, true_res, _ = physics.components(true_params)
            g_clean = physics.forward_from_parameters(true_params)

            if args.noise_level > 0:
                g_np = g_clean.cpu().numpy().astype(np.float64)
                rms = np.sqrt(np.mean(g_np * g_np, axis=1, keepdims=True))
                g_in_np = (
                    g_np
                    + args.noise_level
                    * rms
                    * rng.standard_normal(g_np.shape)
                )
                g_input = torch.from_numpy(g_in_np.astype(np.float32)).to(device)
            else:
                g_input = g_clean

            pred_params = model.predict_parameters(g_input.unsqueeze(1))
            pred_f, pred_res, _ = physics.components(pred_params)
            pred_g = physics.forward_from_parameters(pred_params)

            true_vis = resonance_visibility(true_res, true_f)
            pred_vis = resonance_visibility(pred_res, pred_f)

            true_width = true_params[:, 3] * true_params[:, 4]
            pred_width = pred_params[:, 3] * pred_params[:, 4]
            width_rel = (
                torch.abs(pred_width - true_width)
                / true_width.abs().clamp_min(1e-8)
            )

            all_true.append(true_params.cpu())
            all_pred.append(pred_params.cpu())
            f_errs.append(rel_l2(pred_f, true_f).cpu())
            g_errs.append(rel_l2(pred_g, g_clean).cpu())
            width_rels.append(width_rel.cpu())
            true_vis_list.append(true_vis.cpu())
            pred_vis_list.append(pred_vis.cpu())
            cursor = stop

    true = torch.cat(all_true, dim=0)
    pred = torch.cat(all_pred, dim=0)
    f_rel = torch.cat(f_errs).numpy()
    g_rel = torch.cat(g_errs).numpy()
    width_rel = torch.cat(width_rels).numpy()
    true_visibility = torch.cat(true_vis_list).numpy()
    pred_visibility = torch.cat(pred_vis_list).numpy()

    lower = model.parameter_lower.detach().cpu()
    upper = model.parameter_upper.detach().cpu()
    true_n = normalize_parameters(true, lower, upper)
    pred_n = normalize_parameters(pred, lower, upper)
    norm_abs = torch.abs(pred_n - true_n).numpy()
    abs_err = torch.abs(pred - true).numpy()

    gamma_true = true[:, 4].numpy().astype(np.float64)
    gamma_pred = pred[:, 4].numpy().astype(np.float64)
    log_gamma_abs = np.abs(
        np.log(np.clip(gamma_pred, 1e-12, None))
        - np.log(np.clip(gamma_true, 1e-12, None))
    )

    bin_rows = summarize_bins(
        edges=edges,
        true_visibility=true_visibility,
        pred_visibility=pred_visibility,
        norm_abs=norm_abs,
        log_gamma_abs=log_gamma_abs,
        width_rel=width_rel,
        f_rel=f_rel,
        g_rel=g_rel,
    )
    save_bin_csv(out / "visibility_binned_summary.csv", bin_rows)

    weak_mask = true_visibility <= args.weak_threshold
    visible_mask = true_visibility >= args.visible_threshold
    middle_mask = (~weak_mask) & (~visible_mask)

    summary = {
        "weights": args.weights,
        "selected_model_step": selected_step,
        "validation_noise_level": float(args.noise_level),
        "num_samples": int(n),
        "visibility_definition": "RMS(true_resonance)/RMS(true_total), clamped to [0,1], on Exp7 output s-grid",
        "visibility_bins": [float(x) for x in edges],
        "weak_threshold": float(args.weak_threshold),
        "visible_threshold": float(args.visible_threshold),
        "all_samples": threshold_summary(
            np.ones(n, dtype=bool),
            true_visibility=true_visibility,
            pred_visibility=pred_visibility,
            norm_abs=norm_abs,
            log_gamma_abs=log_gamma_abs,
            width_rel=width_rel,
            f_rel=f_rel,
            g_rel=g_rel,
        ),
        "weak_resonance": threshold_summary(
            weak_mask,
            true_visibility=true_visibility,
            pred_visibility=pred_visibility,
            norm_abs=norm_abs,
            log_gamma_abs=log_gamma_abs,
            width_rel=width_rel,
            f_rel=f_rel,
            g_rel=g_rel,
        ),
        "middle_resonance": threshold_summary(
            middle_mask,
            true_visibility=true_visibility,
            pred_visibility=pred_visibility,
            norm_abs=norm_abs,
            log_gamma_abs=log_gamma_abs,
            width_rel=width_rel,
            f_rel=f_rel,
            g_rel=g_rel,
        ),
        "visible_resonance": threshold_summary(
            visible_mask,
            true_visibility=true_visibility,
            pred_visibility=pred_visibility,
            norm_abs=norm_abs,
            log_gamma_abs=log_gamma_abs,
            width_rel=width_rel,
            f_rel=f_rel,
            g_rel=g_rel,
        ),
        "bins": bin_rows,
    }
    (out / "visibility_binned_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    bin_ids = visibility_bin_indices(true_visibility, edges)
    with (out / "visibility_per_sample.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        header = [
            "row",
            "visibility_bin",
            "true_visibility",
            "pred_visibility",
            "visibility_abs_err",
        ]
        for name in PARAMETER_NAMES:
            header += [
                f"{name}_true",
                f"{name}_pred",
                f"{name}_abs_err",
                f"{name}_norm_abs_err",
            ]
        header += [
            "log_gamma_abs_err",
            "width_rel_err",
            "f_rel_l2",
            "g_rel_clean",
        ]
        writer.writerow(header)

        true_np_final = true.numpy()
        pred_np_final = pred.numpy()
        for i in range(n):
            row = [
                i,
                int(bin_ids[i]),
                float(true_visibility[i]),
                float(pred_visibility[i]),
                float(abs(pred_visibility[i] - true_visibility[i])),
            ]
            for k in range(5):
                row += [
                    float(true_np_final[i, k]),
                    float(pred_np_final[i, k]),
                    float(abs_err[i, k]),
                    float(norm_abs[i, k]),
                ]
            row += [
                float(log_gamma_abs[i]),
                float(width_rel[i]),
                float(f_rel[i]),
                float(g_rel[i]),
            ]
            writer.writerow(row)

    plot_m_gamma_vs_visibility(
        out,
        true_visibility,
        norm_abs,
        bin_rows,
        args.scatter_max_points,
        args.seed,
    )
    plot_parameter_errors_by_bin(out, bin_rows)
    plot_true_vs_pred_visibility(
        out,
        true_visibility,
        pred_visibility,
        args.scatter_max_points,
        args.seed,
    )

    print("=" * 80)
    print("Exp7 visibility-conditioned analysis finished")
    print(f"weights / step          : {args.weights} / {selected_step}")
    print(f"validation noise        : {100 * args.noise_level:.2f}%")
    print(f"samples                 : {n}")
    print(f"weak  V<={args.weak_threshold:g}      : {int(np.sum(weak_mask))}")
    print(f"visible V>={args.visible_threshold:g} : {int(np.sum(visible_mask))}")
    print("-" * 80)
    for label, mask in (
        ("ALL", np.ones(n, dtype=bool)),
        ("WEAK", weak_mask),
        ("VISIBLE", visible_mask),
    ):
        if not np.any(mask):
            print(f"{label:8s}: no samples")
            continue
        print(
            f"{label:8s}: "
            f"m={np.mean(norm_abs[mask, 3]):.5f}  "
            f"gamma={np.mean(norm_abs[mask, 4]):.5f}  "
            f"log_gamma={np.mean(log_gamma_abs[mask]):.5f}  "
            f"f={np.mean(f_rel[mask]):.5f}  "
            f"g={np.mean(g_rel[mask]):.5f}"
        )
    print("-" * 80)
    print("Interpretation:")
    print("1) If m/gamma errors fall sharply as true visibility increases,")
    print("   the global ~0.23 error is dominated by weak/meaningless resonance labels.")
    print("2) If m/gamma remain poor even for V>=visible-threshold,")
    print("   then the next experiment should locate which parameter coupling causes the collapse.")
    print(f"output                  : {out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
