#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp21: construct an a1+gamma dataset by enforcing g-space separation.

This is the direct test of the "make parameter intervals larger so different
physical states produce distinguishable g(q^2)" idea.

Instead of choosing parameter counts first and merely measuring separation
afterward, this script:

1. scans regular candidate grids in (a1, gamma);
2. evaluates every pair of clean g(q^2) curves;
3. keeps only grid resolutions whose *minimum* full-curve separation SNR is
   above a requested threshold;
4. chooses the largest feasible regular grid;
5. generates train/val/test data on that grid.

The physical forward formula is unchanged.  The training script remains the
same pairwise MLP used by Exp20 so the controlled change is the data geometry.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np

from data_generate_exp20_pairwise import (
    DEFAULT_PHYSICS,
    FIXED_A2,
    FIXED_A3,
    FIXED_M,
    arithmetic_midpoints,
    build_parameter_rows,
    forward_bank,
    geometric_midpoints,
    make_split_arrays,
    noise_dir_name,
    pair_metrics,
    save_npz,
    write_csv,
)


def parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exp21 g-separation-controlled a1+gamma dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default="data_exp21_gsep_a1_gamma")
    p.add_argument("--target-snr", type=float, default=5.0,
                   help="minimum allowed full-curve ||g1-g2||_2 / sigma_ref")
    p.add_argument("--reference-noise", type=float, default=0.002,
                   help="noise fraction used only to design the separation-controlled grid")
    p.add_argument("--a1-min", type=float, default=0.05)
    p.add_argument("--a1-max", type=float, default=0.20)
    p.add_argument("--gamma-min", type=float, default=0.01)
    p.add_argument("--gamma-max", type=float, default=1.0)
    p.add_argument("--min-a1-count", type=int, default=3)
    p.add_argument("--max-a1-count", type=int, default=11)
    p.add_argument("--min-gamma-count", type=int, default=3)
    p.add_argument("--max-gamma-count", type=int, default=11)
    p.add_argument("--noise-levels", type=parse_float_list, default=[0.0, 0.002, 0.01])
    p.add_argument("--train-noise-level", type=float, default=0.002)
    p.add_argument("--target-train-samples", type=int, default=20000)
    p.add_argument("--target-val-samples", type=int, default=2500)
    p.add_argument("--target-test-seen-samples", type=int, default=4400)
    p.add_argument("--target-test-interp-samples", type=int, default=2200)
    p.add_argument("--q2-points", type=int, default=100)
    p.add_argument("--integration-points", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260813)
    p.add_argument("--compressed", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def full_pair_diagnostics(
    a1_values: list[float],
    gamma_values: list[float],
    *,
    physics,
    integration_points: int,
    reference_noise: float,
) -> tuple[np.ndarray, list[dict], list[dict], dict]:
    params = build_parameter_rows("a1gamma", a1_values, gamma_values)
    _, g_bank = forward_bank(
        params,
        physics=physics,
        integration_points=integration_points,
    )
    n_gamma = len(gamma_values)
    state_rows: list[dict] = []
    for ia, a1 in enumerate(a1_values):
        for ig, gamma in enumerate(gamma_values):
            state_rows.append(
                {
                    "state_index": ia * n_gamma + ig,
                    "a1": float(a1),
                    "gamma": float(gamma),
                }
            )

    pair_rows: list[dict] = []
    nearest = [None] * len(state_rows)
    for i, j in combinations(range(len(state_rows)), 2):
        metrics = pair_metrics(g_bank[i], g_bank[j], reference_noise)
        row = {
            "state_1": i,
            "state_2": j,
            "a1_1": state_rows[i]["a1"],
            "a1_2": state_rows[j]["a1"],
            "gamma_1": state_rows[i]["gamma"],
            "gamma_2": state_rows[j]["gamma"],
            "reference_noise_level": float(reference_noise),
            **metrics,
        }
        pair_rows.append(row)
        for state, other in ((i, j), (j, i)):
            current = nearest[state]
            if current is None or metrics["snr_sep"] < current["snr_sep"]:
                nearest[state] = {
                    "state": state,
                    "nearest_state": other,
                    "a1": state_rows[state]["a1"],
                    "nearest_a1": state_rows[other]["a1"],
                    "gamma": state_rows[state]["gamma"],
                    "nearest_gamma": state_rows[other]["gamma"],
                    "relative_l2_difference": metrics["relative_l2_difference"],
                    "snr_sep": metrics["snr_sep"],
                    "snr_sep_rms": metrics["snr_sep_rms"],
                }

    nearest_rows = [x for x in nearest if x is not None]
    snr = np.asarray([r["snr_sep"] for r in nearest_rows], dtype=np.float64)
    snr_rms = np.asarray([r["snr_sep_rms"] for r in nearest_rows], dtype=np.float64)
    rel = np.asarray([r["relative_l2_difference"] for r in nearest_rows], dtype=np.float64)
    summary = {
        "state_count": len(state_rows),
        "min_snr": float(np.min(snr)),
        "median_nearest_snr": float(np.median(snr)),
        "p90_nearest_snr": float(np.quantile(snr, 0.90)),
        "min_snr_rms": float(np.min(snr_rms)),
        "median_nearest_snr_rms": float(np.median(snr_rms)),
        "min_relative_l2": float(np.min(rel)),
        "median_nearest_relative_l2": float(np.median(rel)),
    }
    return g_bank, pair_rows, nearest_rows, summary


def search_regular_grid(args: argparse.Namespace, physics) -> tuple[list[float], list[float], list[dict]]:
    rows: list[dict] = []
    feasible: list[dict] = []

    for a_count in range(args.min_a1_count, args.max_a1_count + 1):
        a1_values = np.linspace(args.a1_min, args.a1_max, a_count, dtype=np.float64).tolist()
        for g_count in range(args.min_gamma_count, args.max_gamma_count + 1):
            gamma_values = np.geomspace(
                args.gamma_min, args.gamma_max, g_count, dtype=np.float64
            ).tolist()
            _, _, _, summary = full_pair_diagnostics(
                a1_values,
                gamma_values,
                physics=physics,
                integration_points=args.integration_points,
                reference_noise=args.reference_noise,
            )
            row = {
                "a1_count": a_count,
                "gamma_count": g_count,
                "state_count": a_count * g_count,
                "a1_step": (args.a1_max - args.a1_min) / (a_count - 1),
                "gamma_ratio": (args.gamma_max / args.gamma_min) ** (1.0 / (g_count - 1)),
                "target_snr": args.target_snr,
                "feasible": int(summary["min_snr"] >= args.target_snr),
                **summary,
            }
            rows.append(row)
            if row["feasible"]:
                feasible.append(row)

    if not feasible:
        best = max(rows, key=lambda r: r["min_snr"])
        raise RuntimeError(
            "No regular grid satisfies the requested target SNR. "
            f"Best scanned grid: a1_count={best['a1_count']}, "
            f"gamma_count={best['gamma_count']}, min_snr={best['min_snr']:.4g}. "
            "Lower --target-snr or reduce the minimum grid counts."
        )

    # Largest feasible state count first.  On ties prefer a more balanced grid,
    # then more gamma coverage, then larger safety margin.
    chosen = max(
        feasible,
        key=lambda r: (
            r["state_count"],
            min(r["a1_count"], r["gamma_count"]),
            r["gamma_count"],
            r["min_snr"],
        ),
    )
    a1_values = np.linspace(
        args.a1_min, args.a1_max, int(chosen["a1_count"]), dtype=np.float64
    ).tolist()
    gamma_values = np.geomspace(
        args.gamma_min, args.gamma_max, int(chosen["gamma_count"]), dtype=np.float64
    ).tolist()
    return a1_values, gamma_values, rows


def count_for_target(target: int, n_states: int) -> int:
    return max(1, int(round(float(target) / max(int(n_states), 1))))


def plot_diagnostics(output_dir: Path, grid_rows: list[dict], nearest_rows: list[dict], a1_values: list[float], gamma_values: list[float]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: plotting skipped: {exc}")
        return

    # Search landscape.
    xs = np.asarray([r["gamma_count"] for r in grid_rows], dtype=float)
    ys = np.asarray([r["a1_count"] for r in grid_rows], dtype=float)
    cs = np.asarray([r["min_snr"] for r in grid_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(xs, ys, c=cs, s=100)
    fig.colorbar(sc, ax=ax, label="minimum full-curve separation SNR")
    ax.set_xlabel("gamma anchor count")
    ax.set_ylabel("a1 anchor count")
    ax.set_title("Exp21 grid search: minimum pairwise g-separation")
    fig.tight_layout()
    fig.savefig(output_dir / "grid_search_min_snr.png", dpi=160)
    plt.close(fig)

    # Selected physical states.
    aa, gg = np.meshgrid(np.asarray(a1_values), np.asarray(gamma_values), indexing="ij")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(aa.ravel(), gg.ravel())
    ax.set_yscale("log")
    ax.set_xlabel("a1")
    ax.set_ylabel("gamma")
    ax.set_title("Exp21 selected training parameter grid")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_parameter_grid.png", dpi=160)
    plt.close(fig)

    # Nearest-neighbour separation distribution.
    snr = np.asarray([r["snr_sep"] for r in nearest_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(snr, bins=min(30, max(5, len(snr) // 2)))
    ax.set_xlabel("nearest-state full-curve separation SNR")
    ax.set_ylabel("state count")
    ax.set_title("Exp21 nearest g-space neighbour separation")
    fig.tight_layout()
    fig.savefig(output_dir / "nearest_state_snr_hist.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.target_snr <= 0:
        raise ValueError("--target-snr must be > 0")
    if args.reference_noise <= 0:
        raise ValueError("--reference-noise must be > 0")
    if not (0 < args.a1_min < args.a1_max <= 0.2 + 1e-12):
        raise ValueError("require 0 < a1_min < a1_max <= 0.2")
    if not (0 < args.gamma_min < args.gamma_max <= 1.0 + 1e-12):
        raise ValueError("require 0 < gamma_min < gamma_max <= 1")
    if args.min_a1_count < 2 or args.min_gamma_count < 2:
        raise ValueError("minimum grid counts must be >= 2")
    if args.max_a1_count < args.min_a1_count or args.max_gamma_count < args.min_gamma_count:
        raise ValueError("max grid counts must be >= min grid counts")
    if not any(np.isclose(args.train_noise_level, x, atol=1e-12, rtol=0) for x in args.noise_levels):
        raise ValueError("--train-noise-level must be included in --noise-levels")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} is not empty; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    physics = replace(DEFAULT_PHYSICS, q2_points=int(args.q2_points))

    print("=" * 96)
    print("Exp21: g-separation-controlled a1 + gamma dataset")
    print("Physical formula: UNCHANGED")
    print(f"reference noise     : {100*args.reference_noise:.4g}%")
    print(f"target full-curve SNR: >= {args.target_snr:.4g}")
    print(
        f"grid search         : a1 count {args.min_a1_count}..{args.max_a1_count}, "
        f"gamma count {args.min_gamma_count}..{args.max_gamma_count}"
    )
    print("=" * 96)

    a1_values, gamma_values, grid_rows = search_regular_grid(args, physics)
    write_csv(output_dir / "grid_search.csv", grid_rows)

    _, pair_rows, nearest_rows, selected_summary = full_pair_diagnostics(
        a1_values,
        gamma_values,
        physics=physics,
        integration_points=args.integration_points,
        reference_noise=args.reference_noise,
    )
    write_csv(output_dir / "state_pair_separation.csv", pair_rows)
    write_csv(output_dir / "state_nearest_neighbor_separation.csv", nearest_rows)

    train_states = []
    idx = 0
    for ia, a1 in enumerate(a1_values):
        for ig, gamma in enumerate(gamma_values):
            train_states.append(
                {
                    "state_index": idx,
                    "a1_index": ia,
                    "gamma_index": ig,
                    "a1": float(a1),
                    "gamma": float(gamma),
                }
            )
            idx += 1
    write_csv(output_dir / "train_states.csv", train_states)

    a1_mid = arithmetic_midpoints(a1_values)
    gamma_mid = geometric_midpoints(gamma_values)

    n_train_states = len(a1_values) * len(gamma_values)
    split_defs = {
        "test_seen": (a1_values, gamma_values, True, True),
        "test_interp_gamma": (a1_values, gamma_mid, True, False),
        "test_interp_second": (a1_mid, gamma_values, False, True),
        "test_interp_both": (a1_mid, gamma_mid, False, False),
    }

    per_state_counts = {
        "train": count_for_target(args.target_train_samples, n_train_states),
        "val": count_for_target(args.target_val_samples, n_train_states),
        "test_seen": count_for_target(
            args.target_test_seen_samples, len(a1_values) * len(gamma_values)
        ),
        "test_interp_gamma": count_for_target(
            args.target_test_interp_samples, len(a1_values) * len(gamma_mid)
        ),
        "test_interp_second": count_for_target(
            args.target_test_interp_samples, len(a1_mid) * len(gamma_values)
        ),
        "test_interp_both": count_for_target(
            args.target_test_interp_samples, len(a1_mid) * len(gamma_mid)
        ),
    }

    for noise_level in args.noise_levels:
        d = output_dir / noise_dir_name(noise_level)
        d.mkdir(parents=True, exist_ok=True)

        if np.isclose(noise_level, args.train_noise_level, atol=1e-12, rtol=0):
            for split in ("train", "val"):
                arrays = make_split_arrays(
                    mode="a1gamma",
                    split=split,
                    second_values=a1_values,
                    gamma_values=gamma_values,
                    count_per_state=per_state_counts[split],
                    noise_level=noise_level,
                    physics=physics,
                    integration_points=args.integration_points,
                    base_seed=args.seed,
                    second_seen=True,
                    gamma_seen=True,
                )
                save_npz(d / f"{split}.npz", arrays, args.compressed)
                print(f"saved {d / (split + '.npz')} samples={len(arrays['gamma']):,}")

        for split, (a_vals, g_vals, a_seen, g_seen) in split_defs.items():
            arrays = make_split_arrays(
                mode="a1gamma",
                split=split,
                second_values=a_vals,
                gamma_values=g_vals,
                count_per_state=per_state_counts[split],
                noise_level=noise_level,
                physics=physics,
                integration_points=args.integration_points,
                base_seed=args.seed,
                second_seen=a_seen,
                gamma_seen=g_seen,
            )
            save_npz(d / f"{split}.npz", arrays, args.compressed)
            print(f"saved {d / (split + '.npz')} samples={len(arrays['gamma']):,}")

    plot_diagnostics(output_dir, grid_rows, nearest_rows, a1_values, gamma_values)

    metadata = {
        "experiment": "Exp21 g-separation-controlled a1+gamma",
        "mode": "a1gamma",
        "second_parameter": "a1",
        "fixed_parameters": {"a2": FIXED_A2, "a3": FIXED_A3, "m": FIXED_M},
        "selection_rule": {
            "type": "largest regular Cartesian grid satisfying minimum full-curve g separation",
            "metric": "snr_sep = ||g_i-g_j||_2 / (reference_noise * mean_RMS(g_i,g_j))",
            "target_snr": float(args.target_snr),
            "reference_noise": float(args.reference_noise),
            "candidate_a1_count_range": [args.min_a1_count, args.max_a1_count],
            "candidate_gamma_count_range": [args.min_gamma_count, args.max_gamma_count],
        },
        "second_train_values": [float(x) for x in a1_values],
        "second_interp_values": [float(x) for x in a1_mid],
        "gamma_train_values": [float(x) for x in gamma_values],
        "gamma_interp_values": [float(x) for x in gamma_mid],
        "training_state_count": n_train_states,
        "selected_separation": selected_summary,
        "noise_levels": [float(x) for x in args.noise_levels],
        "train_noise_level": float(args.train_noise_level),
        "reference_noise": float(args.reference_noise),
        "q2_range": [physics.q2_min, physics.q2_max],
        "q2_points": physics.q2_points,
        "output_points": physics.output_points,
        "integration_points": args.integration_points,
        "data_scale": physics.data_scale,
        "counts_per_state": per_state_counts,
        "approx_split_sizes": {
            "train": per_state_counts["train"] * n_train_states,
            "val": per_state_counts["val"] * n_train_states,
            "test_seen": per_state_counts["test_seen"] * len(a1_values) * len(gamma_values),
            "test_interp_gamma": per_state_counts["test_interp_gamma"] * len(a1_values) * len(gamma_mid),
            "test_interp_second": per_state_counts["test_interp_second"] * len(a1_mid) * len(gamma_values),
            "test_interp_both": per_state_counts["test_interp_both"] * len(a1_mid) * len(gamma_mid),
        },
        "noise_definition": "g_noisy = g_clean + noise_level * RMS(g_clean) * N(0,1)",
        "notes": [
            "Physical formula is unchanged.",
            "Unlike Exp20, parameter-grid density is selected from g-space separation before training.",
            "The minimum separation criterion is enforced over all pairs of training states, including diagonal a1-gamma compensation pairs.",
            "Interpolation splits are diagnostics outside the guaranteed separation-controlled training grid.",
        ],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("=" * 96)
    print("Exp21 data ready")
    print(f"selected a1 anchors   : {len(a1_values)} -> {np.array(a1_values)}")
    print(f"selected gamma anchors: {len(gamma_values)} -> {np.array(gamma_values)}")
    print(f"training states       : {n_train_states}")
    print(
        "minimum g-separation : "
        f"SNR={selected_summary['min_snr']:.4g}, "
        f"RMS-SNR={selected_summary['min_snr_rms']:.4g}, "
        f"relative-L2={selected_summary['min_relative_l2']:.4g}"
    )
    print(f"output: {output_dir}")
    print("Read before training:")
    print(f"  {output_dir / 'grid_search.csv'}")
    print(f"  {output_dir / 'state_nearest_neighbor_separation.csv'}")
    print(f"  {output_dir / 'grid_search_min_snr.png'}")
    print("=" * 96)


if __name__ == "__main__":
    main()
