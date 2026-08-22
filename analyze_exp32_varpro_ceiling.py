#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp32 physics ceiling: profile gamma and solve a1 analytically.

For the current a1+gamma stage, a2/a3/m are fixed and the observation has the
form

    g(gamma, a1) = B + a1 * R(gamma).

For every candidate gamma, the least-squares a1 has a closed form.  Therefore
we only scan one nonlinear parameter (gamma), solve a1 analytically, and pick
the minimum residual.  This gives a network-independent reference for the
recoverability of the current filtered dataset.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from mc_pool_config import DEFAULT_PHYSICS
from mc_physics import scaled_curves_torch
from train_exp20_pairwise import resolve_device, write_csv
from train_exp32_joint_continuous import continuous_metrics


def parse_args():
    p = argparse.ArgumentParser(
        description="Exp32 a1+gamma variable-projection ceiling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--noise-dir", default="noise_0p2pct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gamma-grid-points", type=int, default=1201)
    p.add_argument("--gamma-min", type=float, default=0.0, help="0 means use dataset/original bound")
    p.add_argument("--gamma-max", type=float, default=0.0, help="0 means use dataset/original bound")
    p.add_argument("--a1-min", type=float, default=float("nan"), help="NaN means use dataset/original bound")
    p.add_argument("--a1-max", type=float, default=float("nan"), help="NaN means use dataset/original bound")
    p.add_argument("--integration-points", type=int, default=0, help="0 means use dataset metadata")
    p.add_argument("--forward-batch-size", type=int, default=256)
    p.add_argument("--projection-batch-size", type=int, default=1024)
    p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--skip-refinement", action="store_true")
    return p.parse_args()


def original_bounds(meta: dict):
    a1 = [float(x) for x in meta.get("original_a1_range", meta["a1_range"])]
    gamma = [float(x) for x in meta.get("original_gamma_range", meta["gamma_range"])]
    return a1, gamma


@torch.no_grad()
def forward_g_batched(params_np, *, device, config, integration_points, batch_size):
    parts = []
    params_np = np.asarray(params_np, dtype=np.float32)
    for start in range(0, len(params_np), int(batch_size)):
        stop = min(start + int(batch_size), len(params_np))
        p = torch.from_numpy(params_np[start:stop]).to(device=device, dtype=torch.float32)
        _, g = scaled_curves_torch(
            p,
            integration_points=int(integration_points),
            config=config,
        )
        parts.append(g.detach().cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float64)


@torch.no_grad()
def forward_fg_batched(params_np, *, device, config, integration_points, batch_size):
    f_parts, g_parts = [], []
    params_np = np.asarray(params_np, dtype=np.float32)
    for start in range(0, len(params_np), int(batch_size)):
        stop = min(start + int(batch_size), len(params_np))
        p = torch.from_numpy(params_np[start:stop]).to(device=device, dtype=torch.float32)
        f, g = scaled_curves_torch(
            p,
            integration_points=int(integration_points),
            config=config,
        )
        f_parts.append(f.detach().cpu().numpy())
        g_parts.append(g.detach().cpu().numpy())
    return np.concatenate(f_parts).astype(np.float64), np.concatenate(g_parts).astype(np.float64)


def build_b_and_r(
    gamma_values,
    *,
    fixed,
    device,
    config,
    integration_points,
    batch_size,
    a1_reference=0.1,
):
    a2, a3, m = fixed
    bg = np.zeros((1, 5), dtype=np.float32)
    bg[0] = [0.0, a2, a3, m, float(gamma_values[len(gamma_values) // 2])]
    B = forward_g_batched(
        bg,
        device=device,
        config=config,
        integration_points=integration_points,
        batch_size=1,
    )[0]

    p = np.zeros((len(gamma_values), 5), dtype=np.float32)
    p[:, 0] = float(a1_reference)
    p[:, 3] = float(m)
    p[:, 4] = np.asarray(gamma_values, dtype=np.float32)
    resonance_g = forward_g_batched(
        p,
        device=device,
        config=config,
        integration_points=integration_points,
        batch_size=batch_size,
    )
    R = resonance_g / float(a1_reference)
    return B, R


def project_grid(y, B, R, a1_min, a1_max, batch_size):
    y = np.asarray(y, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    r2 = np.sum(R * R, axis=1)
    r2 = np.maximum(r2, 1e-30)

    best_idx = np.empty(len(y), dtype=np.int64)
    best_a1 = np.empty(len(y), dtype=np.float64)
    best_sse = np.empty(len(y), dtype=np.float64)
    neighbor_sse = np.full((len(y), 3), np.nan, dtype=np.float64)

    for start in range(0, len(y), int(batch_size)):
        stop = min(start + int(batch_size), len(y))
        Y = y[start:stop] - B[None, :]
        dot = Y @ R.T
        a = np.clip(dot / r2[None, :], float(a1_min), float(a1_max))
        y2 = np.sum(Y * Y, axis=1, keepdims=True)
        sse = y2 - 2.0 * a * dot + (a * a) * r2[None, :]
        idx = np.argmin(sse, axis=1)
        rr = np.arange(stop - start)
        best_idx[start:stop] = idx
        best_a1[start:stop] = a[rr, idx]
        best_sse[start:stop] = sse[rr, idx]
        for off, col in [(-1, 0), (0, 1), (1, 2)]:
            jj = np.clip(idx + off, 0, R.shape[0] - 1)
            neighbor_sse[start:stop, col] = sse[rr, jj]

    return best_idx, best_a1, best_sse, neighbor_sse


def refine_log_gamma(gamma_grid, best_idx, sse3):
    logg = np.log10(np.asarray(gamma_grid, dtype=np.float64))
    if len(logg) < 3:
        return np.asarray(gamma_grid, dtype=np.float64)[best_idx]
    step = float(np.median(np.diff(logg)))
    center = logg[best_idx]
    left, mid, right = sse3[:, 0], sse3[:, 1], sse3[:, 2]
    denom = left - 2.0 * mid + right
    delta = np.zeros(len(best_idx), dtype=np.float64)
    ok = (best_idx > 0) & (best_idx < len(logg) - 1) & (np.abs(denom) > 1e-30)
    delta[ok] = 0.5 * (left[ok] - right[ok]) / denom[ok]
    delta = np.clip(delta, -1.0, 1.0)
    return np.power(10.0, center + delta * step)


def project_a1_with_per_sample_r(y, B, R, a1_min, a1_max):
    Y = np.asarray(y, np.float64) - np.asarray(B, np.float64)[None, :]
    R = np.asarray(R, np.float64)
    dot = np.sum(Y * R, axis=1)
    r2 = np.maximum(np.sum(R * R, axis=1), 1e-30)
    a = np.clip(dot / r2, float(a1_min), float(a1_max))
    residual = Y - a[:, None] * R
    sse = np.sum(residual * residual, axis=1)
    return a, sse


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("mode") != "a1gamma":
        raise ValueError("variable-projection ceiling requires a1gamma data")

    test_path = data_dir / args.noise_dir / "test.npz"
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    with np.load(test_path, allow_pickle=False) as z:
        y = np.asarray(z["gy_noisy"], dtype=np.float64)
        y_clean = np.asarray(z["gy_clean"], dtype=np.float64) if "gy_clean" in z else None
        fx = np.asarray(z["fx"], dtype=np.float64) if "fx" in z else None
        params_true = np.asarray(z["parameters"], dtype=np.float64)
        true_a1 = np.asarray(z["a1"], dtype=np.float64).reshape(-1)
        true_gamma = np.asarray(z["gamma"], dtype=np.float64).reshape(-1)
        noise_level = np.asarray(z["noise_level"], dtype=np.float64).reshape(-1)

    if args.max_test_samples and len(y) > int(args.max_test_samples):
        n = int(args.max_test_samples)
        y = y[:n]
        true_a1 = true_a1[:n]
        true_gamma = true_gamma[:n]
        params_true = params_true[:n]
        noise_level = noise_level[:n]
        if y_clean is not None:
            y_clean = y_clean[:n]
        if fx is not None:
            fx = fx[:n]

    a1_bounds, gamma_bounds = original_bounds(meta)
    a1_min = float(args.a1_min) if np.isfinite(args.a1_min) else a1_bounds[0]
    a1_max = float(args.a1_max) if np.isfinite(args.a1_max) else a1_bounds[1]
    gamma_min = float(args.gamma_min) if args.gamma_min > 0 else gamma_bounds[0]
    gamma_max = float(args.gamma_max) if args.gamma_max > 0 else gamma_bounds[1]
    if not (0 < gamma_min < gamma_max):
        raise ValueError("invalid gamma bounds")
    if not (a1_min < a1_max):
        raise ValueError("invalid a1 bounds")

    fixed_meta = meta["fixed_parameters"]
    fixed = (
        float(fixed_meta["a2"]),
        float(fixed_meta["a3"]),
        float(fixed_meta["m"]),
    )
    output_points = int(fx.shape[1]) if fx is not None else DEFAULT_PHYSICS.output_points
    config = replace(DEFAULT_PHYSICS, q2_points=int(y.shape[1]), output_points=output_points)
    integration_points = int(args.integration_points or meta.get("integration_points", 512))
    device = resolve_device(args.device)

    gamma_grid = np.logspace(
        math.log10(gamma_min),
        math.log10(gamma_max),
        int(args.gamma_grid_points),
        dtype=np.float64,
    )

    print("=" * 100)
    print("Exp32 variable-projection ceiling")
    print("test                       : %s" % test_path)
    print("samples                    : %d" % len(y))
    print("a1 scan/projection bounds  : [%.6g, %.6g]" % (a1_min, a1_max))
    print("gamma scan bounds          : [%.6g, %.6g]" % (gamma_min, gamma_max))
    print("gamma grid points          : %d" % len(gamma_grid))
    print("integration points         : %d" % integration_points)
    print("=" * 100)

    B, R = build_b_and_r(
        gamma_grid,
        fixed=fixed,
        device=device,
        config=config,
        integration_points=integration_points,
        batch_size=args.forward_batch_size,
    )

    idx, pred_a1, residual_sse, sse3 = project_grid(
        y, B, R, a1_min, a1_max, args.projection_batch_size
    )
    pred_gamma = gamma_grid[idx]

    if not args.skip_refinement:
        refined_gamma = refine_log_gamma(gamma_grid, idx, sse3)
        _, R_ref = build_b_and_r(
            refined_gamma,
            fixed=fixed,
            device=device,
            config=config,
            integration_points=integration_points,
            batch_size=args.forward_batch_size,
        )
        # build_b_and_r returns one resonance row per gamma, exactly what is needed here.
        pred_a1_ref, residual_sse_ref = project_a1_with_per_sample_r(
            y, B, R_ref, a1_min, a1_max
        )
        improve = residual_sse_ref <= residual_sse
        pred_gamma = np.where(improve, refined_gamma, pred_gamma)
        pred_a1 = np.where(improve, pred_a1_ref, pred_a1)
        residual_sse = np.where(improve, residual_sse_ref, residual_sse)

    cm = continuous_metrics(
        pred_a1,
        true_a1,
        pred_gamma,
        true_gamma,
        a1_reference_span=a1_bounds[1] - a1_bounds[0],
    )

    params_pred = params_true.copy()
    params_pred[:, 0] = pred_a1
    params_pred[:, 4] = pred_gamma
    pred_f, pred_g = forward_fg_batched(
        params_pred,
        device=device,
        config=config,
        integration_points=integration_points,
        batch_size=args.forward_batch_size,
    )

    if fx is not None:
        f_rel = np.linalg.norm(pred_f - fx, axis=1) / np.maximum(np.linalg.norm(fx, axis=1), 1e-30)
    else:
        f_rel = np.full(len(y), np.nan)
    if y_clean is not None:
        g_rel = np.linalg.norm(pred_g - y_clean, axis=1) / np.maximum(np.linalg.norm(y_clean, axis=1), 1e-30)
    else:
        g_rel = np.full(len(y), np.nan)

    noisy_residual_rms = np.sqrt(residual_sse / max(y.shape[1], 1))
    summary = {
        **cm,
        "noise_dir": args.noise_dir,
        "noise_level_median": float(np.median(noise_level)),
        "median_profile_noisy_g_rms_residual": float(np.median(noisy_residual_rms)),
        "p90_profile_noisy_g_rms_residual": float(np.quantile(noisy_residual_rms, 0.90)),
        "median_f_relative_l2": float(np.nanmedian(f_rel)),
        "p90_f_relative_l2": float(np.nanquantile(f_rel, 0.90)),
        "median_g_clean_relative_l2": float(np.nanmedian(g_rel)),
        "p90_g_clean_relative_l2": float(np.nanquantile(g_rel, 0.90)),
        "gamma_grid_points": int(len(gamma_grid)),
        "refinement_enabled": int(not args.skip_refinement),
    }
    write_csv(out / "exp32_varpro_summary.csv", [summary])

    sample_rows = []
    for i in range(len(y)):
        glog = float(np.log10(max(pred_gamma[i], 1e-30)) - np.log10(max(true_gamma[i], 1e-30)))
        sample_rows.append(
            {
                "sample_index": int(i),
                "noise_level": float(noise_level[i]),
                "true_a1": float(true_a1[i]),
                "pred_a1": float(pred_a1[i]),
                "a1_signed_error": float(pred_a1[i] - true_a1[i]),
                "a1_abs_error": float(abs(pred_a1[i] - true_a1[i])),
                "true_gamma": float(true_gamma[i]),
                "pred_gamma": float(pred_gamma[i]),
                "gamma_signed_log10_error": glog,
                "gamma_abs_log10_error": abs(glog),
                "gamma_factor_error": float(
                    np.exp(abs(np.log(max(pred_gamma[i], 1e-30)) - np.log(max(true_gamma[i], 1e-30))))
                ),
                "profile_noisy_g_rms_residual": float(noisy_residual_rms[i]),
                "f_relative_l2": float(f_rel[i]),
                "g_clean_relative_l2": float(g_rel[i]),
            }
        )
    write_csv(out / "exp32_varpro_samples.csv", sample_rows)

    (out / "exp32_varpro_metadata.json").write_text(
        json.dumps(
            {
                "experiment": "Exp32 a1+gamma variable-projection ceiling",
                "data_dir": str(data_dir),
                "noise_dir": args.noise_dir,
                "a1_bounds": [a1_min, a1_max],
                "gamma_bounds": [gamma_min, gamma_max],
                "gamma_grid_points": len(gamma_grid),
                "integration_points": integration_points,
                "refinement_enabled": not args.skip_refinement,
                "interpretation": (
                    "Network-independent reference: gamma is profiled and a1 is solved "
                    "analytically at each gamma. Compare continuous errors against MLP."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("Exp32 variable-projection ceiling complete")
    print("a1 MAE              : %.6g" % cm["a1_mae"])
    print("a1 P90 abs          : %.6g" % cm["a1_p90_abs_error"])
    print("gamma log10 MAE     : %.6g" % cm["gamma_log10_mae"])
    print("gamma P90 factor    : %.6g" % cm["gamma_p90_factor"])
    print("Read first: %s" % (out / "exp32_varpro_summary.csv"))
    print("=" * 100)


if __name__ == "__main__":
    main()
