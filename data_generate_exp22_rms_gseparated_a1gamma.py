#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp22A: construct an a1+gamma dataset using RMS-SNR separation.

Purpose
-------
Exp21 selected the regular parameter grid using the *full-curve* L2 SNR

    ||g_i-g_j||_2 / sigma_ref,

which grows like sqrt(N_q) when the number of q^2 samples increases.  Exp22A
uses the sampling-density-normalized criterion

    RMS(g_i-g_j) / sigma_ref

instead.  This directly asks whether the typical pointwise difference between
two clean input curves is large compared with the reference noise.

The physical forward model, MLP architecture, loss, training noise, and q^2
range are unchanged.  Only the training-state selection rule is changed.
"""
from __future__ import annotations

import argparse
import json
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
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exp22A RMS-SNR-controlled a1+gamma dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default="data_exp22a_rms_gsep_a1_gamma")
    p.add_argument(
        "--target-rms-snr",
        type=float,
        default=2.0,
        help=(
            "minimum allowed RMS(g_i-g_j) / sigma_ref over all training-state pairs; "
            "ignored when --target-g-gap-percent is supplied"
        ),
    )
    p.add_argument(
        "--target-g-gap-percent",
        type=float,
        default=None,
        help=(
            "direct g-space target: require RMS(g_i-g_j) / mean_RMS(g_i,g_j) "
            "to be at least this percent. Example: 0.4 means a 0.4%% RMS curve gap. "
            "The script converts this to the equivalent RMS-SNR using --reference-noise."
        ),
    )
    p.add_argument(
        "--reference-noise",
        type=float,
        default=0.002,
        help="noise fraction used only to design the separation-controlled grid",
    )
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
    p.add_argument(
        "--q2-points",
        type=int,
        default=100,
        help="Exp22A control should normally remain 100; later sampling-density tests can change this",
    )
    p.add_argument("--integration-points", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260813)
    p.add_argument("--compressed", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def effective_target_rms_snr(args: argparse.Namespace) -> float:
    """Resolve the requested g-space separation into the dimensionless RMS-SNR.

    target_g_gap_percent is the most direct user-facing quantity:

        100 * RMS(g_i-g_j) / mean_RMS(g_i,g_j).

    Since sigma_ref = reference_noise * mean_RMS(g_i,g_j),

        RMS-SNR = (gap_percent / 100) / reference_noise.
    """
    if args.target_g_gap_percent is not None:
        if args.target_g_gap_percent <= 0:
            raise ValueError("--target-g-gap-percent must be > 0")
        return (float(args.target_g_gap_percent) / 100.0) / float(args.reference_noise)
    return float(args.target_rms_snr)


def pair_diagnostics(
    a1_values: list[float],
    gamma_values: list[float],
    *,
    physics,
    integration_points: int,
    reference_noise: float,
) -> tuple[list[dict], list[dict], dict]:
    params = build_parameter_rows("a1gamma", a1_values, gamma_values)
    _, g_bank = forward_bank(
        params,
        physics=physics,
        integration_points=integration_points,
    )

    n_gamma = len(gamma_values)
    states: list[dict] = []
    for ia, a1 in enumerate(a1_values):
        for ig, gamma in enumerate(gamma_values):
            states.append(
                {
                    "state_index": ia * n_gamma + ig,
                    "a1": float(a1),
                    "gamma": float(gamma),
                }
            )

    pair_rows: list[dict] = []
    nearest: list[dict | None] = [None] * len(states)

    for i, j in combinations(range(len(states)), 2):
        metrics = pair_metrics(g_bank[i], g_bank[j], reference_noise)
        pair_rows.append(
            {
                "state_1": i,
                "state_2": j,
                "a1_1": states[i]["a1"],
                "a1_2": states[j]["a1"],
                "gamma_1": states[i]["gamma"],
                "gamma_2": states[j]["gamma"],
                "reference_noise_level": float(reference_noise),
                "rms_g_relative": (
                    metrics["rms_g_difference"] / metrics["mean_rms_g"]
                    if metrics["mean_rms_g"] > 0 else float("inf")
                ),
                "rms_g_gap_percent": (
                    100.0 * metrics["rms_g_difference"] / metrics["mean_rms_g"]
                    if metrics["mean_rms_g"] > 0 else float("inf")
                ),
                **metrics,
            }
        )

        # RMS-SNR is the Exp22A design metric.
        for state, other in ((i, j), (j, i)):
            current = nearest[state]
            if current is None or metrics["snr_sep_rms"] < current["snr_sep_rms"]:
                nearest[state] = {
                    "state": state,
                    "nearest_state": other,
                    "a1": states[state]["a1"],
                    "nearest_a1": states[other]["a1"],
                    "gamma": states[state]["gamma"],
                    "nearest_gamma": states[other]["gamma"],
                    "relative_l2_difference": metrics["relative_l2_difference"],
                    "snr_sep": metrics["snr_sep"],
                    "snr_sep_rms": metrics["snr_sep_rms"],
                    "rms_g_difference": metrics["rms_g_difference"],
                    "mean_rms_g": metrics["mean_rms_g"],
                    "rms_g_relative": (
                        metrics["rms_g_difference"] / metrics["mean_rms_g"]
                        if metrics["mean_rms_g"] > 0 else float("inf")
                    ),
                    "rms_g_gap_percent": (
                        100.0 * metrics["rms_g_difference"] / metrics["mean_rms_g"]
                        if metrics["mean_rms_g"] > 0 else float("inf")
                    ),
                    "sigma_reference": metrics["sigma_reference"],
                }

    nearest_rows = [row for row in nearest if row is not None]
    full = np.asarray([row["snr_sep"] for row in nearest_rows], dtype=np.float64)
    rms = np.asarray([row["snr_sep_rms"] for row in nearest_rows], dtype=np.float64)
    rel = np.asarray([row["relative_l2_difference"] for row in nearest_rows], dtype=np.float64)
    rms_rel = np.asarray([row["rms_g_relative"] for row in nearest_rows], dtype=np.float64)

    summary = {
        "state_count": len(states),
        "min_snr_rms": float(np.min(rms)),
        "median_nearest_snr_rms": float(np.median(rms)),
        "p90_nearest_snr_rms": float(np.quantile(rms, 0.90)),
        "min_snr": float(np.min(full)),
        "median_nearest_snr": float(np.median(full)),
        "min_relative_l2": float(np.min(rel)),
        "median_nearest_relative_l2": float(np.median(rel)),
        "min_rms_g_relative": float(np.min(rms_rel)),
        "median_nearest_rms_g_relative": float(np.median(rms_rel)),
        "min_rms_g_gap_percent": float(100.0 * np.min(rms_rel)),
        "median_nearest_rms_g_gap_percent": float(100.0 * np.median(rms_rel)),
    }
    return pair_rows, nearest_rows, summary


def search_regular_grid(
    args: argparse.Namespace,
    physics,
) -> tuple[list[float], list[float], list[dict], dict]:
    rows: list[dict] = []
    feasible: list[dict] = []
    target_rms_snr = effective_target_rms_snr(args)

    for a_count in range(args.min_a1_count, args.max_a1_count + 1):
        a1_values = np.linspace(
            args.a1_min, args.a1_max, a_count, dtype=np.float64
        ).tolist()

        for g_count in range(args.min_gamma_count, args.max_gamma_count + 1):
            gamma_values = np.geomspace(
                args.gamma_min, args.gamma_max, g_count, dtype=np.float64
            ).tolist()

            _, _, summary = pair_diagnostics(
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
                "target_rms_snr": float(target_rms_snr),
                "target_g_gap_percent": float(100.0 * target_rms_snr * args.reference_noise),
                "feasible": int(summary["min_snr_rms"] >= target_rms_snr),
                **summary,
            }
            rows.append(row)
            if row["feasible"]:
                feasible.append(row)

    if not feasible:
        best = max(rows, key=lambda row: row["min_snr_rms"])
        raise RuntimeError(
            "No regular grid satisfies the requested RMS-SNR threshold. "
            f"Best scanned grid: a1_count={best['a1_count']}, "
            f"gamma_count={best['gamma_count']}, "
            f"min_rms_snr={best['min_snr_rms']:.4g}. "
            "Lower --target-rms-snr or reduce the minimum grid counts."
        )

    # Main scientific objective: retain as many separated classes as possible.
    # Tie-breaks prefer more gamma classes, then a more balanced grid, then margin.
    chosen = max(
        feasible,
        key=lambda row: (
            row["state_count"],
            row["gamma_count"],
            min(row["a1_count"], row["gamma_count"]),
            row["min_snr_rms"],
        ),
    )

    a1_values = np.linspace(
        args.a1_min, args.a1_max, int(chosen["a1_count"]), dtype=np.float64
    ).tolist()
    gamma_values = np.geomspace(
        args.gamma_min, args.gamma_max, int(chosen["gamma_count"]), dtype=np.float64
    ).tolist()
    return a1_values, gamma_values, rows, chosen


def count_for_target(target: int, n_states: int) -> int:
    return max(1, int(round(float(target) / max(int(n_states), 1))))


def plot_diagnostics(
    output_dir: Path,
    grid_rows: list[dict],
    nearest_rows: list[dict],
    a1_values: list[float],
    gamma_values: list[float],
    target_rms_snr: float,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: plotting skipped: {exc}")
        return

    xs = np.asarray([row["gamma_count"] for row in grid_rows], dtype=float)
    ys = np.asarray([row["a1_count"] for row in grid_rows], dtype=float)
    cs = np.asarray([row["min_snr_rms"] for row in grid_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(xs, ys, c=cs, s=100)
    fig.colorbar(sc, ax=ax, label="minimum nearest-pair RMS-SNR")
    ax.set_xlabel("gamma anchor count")
    ax.set_ylabel("a1 anchor count")
    ax.set_title("Exp22A grid search: minimum RMS-SNR")
    fig.tight_layout()
    fig.savefig(output_dir / "grid_search_min_rms_snr.png", dpi=160)
    plt.close(fig)

    aa, gg = np.meshgrid(
        np.asarray(a1_values), np.asarray(gamma_values), indexing="ij"
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(aa.ravel(), gg.ravel())
    ax.set_yscale("log")
    ax.set_xlabel("a1")
    ax.set_ylabel("gamma")
    ax.set_title("Exp22A selected training parameter grid")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_parameter_grid.png", dpi=160)
    plt.close(fig)

    rms = np.asarray([row["snr_sep_rms"] for row in nearest_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rms, bins=min(30, max(5, len(rms) // 2)))
    ax.axvline(target_rms_snr, linestyle="--", linewidth=1.5)
    ax.set_xlabel("nearest-state RMS-SNR")
    ax.set_ylabel("state count")
    ax.set_title("Exp22A nearest g-space neighbour separation")
    fig.tight_layout()
    fig.savefig(output_dir / "nearest_state_rms_snr_hist.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.target_rms_snr <= 0:
        raise ValueError("--target-rms-snr must be > 0")
    target_rms_snr = effective_target_rms_snr(args)
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
    if args.q2_points < 2:
        raise ValueError("--q2-points must be >= 2")
    if not any(
        np.isclose(args.train_noise_level, x, atol=1e-12, rtol=0)
        for x in args.noise_levels
    ):
        raise ValueError("--train-noise-level must be included in --noise-levels")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} is not empty; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    physics = replace(DEFAULT_PHYSICS, q2_points=int(args.q2_points))

    print("=" * 96)
    print("Exp22A: RMS-SNR-controlled a1 + gamma dataset")
    print("Physical formula : UNCHANGED")
    print(f"q2 points        : {physics.q2_points}")
    print(f"reference noise  : {100 * args.reference_noise:.4g}%")
    print(f"target g RMS gap: >= {100.0 * target_rms_snr * args.reference_noise:.4g}% of mean RMS(g)")
    print(f"target RMS-SNR   : >= {target_rms_snr:.4g}")
    print(
        f"grid search      : a1 count {args.min_a1_count}..{args.max_a1_count}, "
        f"gamma count {args.min_gamma_count}..{args.max_gamma_count}"
    )
    print("=" * 96)

    a1_values, gamma_values, grid_rows, chosen = search_regular_grid(args, physics)
    write_csv(output_dir / "grid_search.csv", grid_rows)

    pair_rows, nearest_rows, selected_summary = pair_diagnostics(
        a1_values,
        gamma_values,
        physics=physics,
        integration_points=args.integration_points,
        reference_noise=args.reference_noise,
    )
    write_csv(output_dir / "state_pair_separation.csv", pair_rows)
    write_csv(output_dir / "state_nearest_neighbor_separation.csv", nearest_rows)

    train_states = []
    state_index = 0
    for ia, a1 in enumerate(a1_values):
        for ig, gamma in enumerate(gamma_values):
            train_states.append(
                {
                    "state_index": state_index,
                    "a1_index": ia,
                    "gamma_index": ig,
                    "a1": float(a1),
                    "gamma": float(gamma),
                }
            )
            state_index += 1
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
        noise_dir = output_dir / noise_dir_name(noise_level)
        noise_dir.mkdir(parents=True, exist_ok=True)

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
                save_npz(noise_dir / f"{split}.npz", arrays, args.compressed)
                print(
                    f"saved {noise_dir / (split + '.npz')} "
                    f"samples={len(arrays['gamma']):,}"
                )

        for split, (a_values, g_values, a_seen, g_seen) in split_defs.items():
            arrays = make_split_arrays(
                mode="a1gamma",
                split=split,
                second_values=a_values,
                gamma_values=g_values,
                count_per_state=per_state_counts[split],
                noise_level=noise_level,
                physics=physics,
                integration_points=args.integration_points,
                base_seed=args.seed,
                second_seen=a_seen,
                gamma_seen=g_seen,
            )
            save_npz(noise_dir / f"{split}.npz", arrays, args.compressed)
            print(
                f"saved {noise_dir / (split + '.npz')} "
                f"samples={len(arrays['gamma']):,}"
            )

    plot_diagnostics(
        output_dir,
        grid_rows,
        nearest_rows,
        a1_values,
        gamma_values,
        target_rms_snr,
    )

    metadata = {
        "experiment": "Exp22A RMS-SNR-controlled a1+gamma",
        "mode": "a1gamma",
        "second_parameter": "a1",
        "fixed_parameters": {"a2": FIXED_A2, "a3": FIXED_A3, "m": FIXED_M},
        "selection_rule": {
            "type": "largest regular Cartesian grid satisfying minimum RMS g separation",
            "metric": (
                "snr_sep_rms = RMS(g_i-g_j) / "
                "(reference_noise * mean_RMS(g_i,g_j))"
            ),
            "target_rms_snr": float(target_rms_snr),
            "target_g_gap_percent": float(100.0 * target_rms_snr * args.reference_noise),
            "reference_noise": float(args.reference_noise),
            "candidate_a1_count_range": [
                args.min_a1_count,
                args.max_a1_count,
            ],
            "candidate_gamma_count_range": [
                args.min_gamma_count,
                args.max_gamma_count,
            ],
        },
        "selected_grid_row": chosen,
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
        "noise_definition": (
            "g_noisy = g_clean + noise_level * RMS(g_clean) * N(0,1)"
        ),
        "notes": [
            "Physical formula is unchanged.",
            "MLP architecture and training loss are unchanged.",
            "Exp22A does not edit or rescale g values by hand. Every g is generated by the unchanged physical forward model.",
            "The user specifies a minimum physical g-space separation; the script then automatically chooses the densest regular parameter grid whose generated g curves satisfy it.",
            "Exp22A changes only the training-state selection metric from full-curve SNR to RMS-SNR.",
            "RMS-SNR is normalized against q2 sample count and is therefore the appropriate screening metric before a later q2-density comparison.",
            "Interpolation splits are diagnostic and are not guaranteed to satisfy the training-grid RMS-SNR threshold.",
        ],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("=" * 96)
    print("Exp22A data ready")
    print(f"selected a1 anchors   : {len(a1_values)} -> {np.array(a1_values)}")
    print(f"selected gamma anchors: {len(gamma_values)} -> {np.array(gamma_values)}")
    print(f"training states       : {n_train_states}")
    print(
        "minimum g-separation : "
        f"RMS gap={selected_summary['min_rms_g_gap_percent']:.4g}%, "
        f"RMS-SNR={selected_summary['min_snr_rms']:.4g}, "
        f"full-SNR={selected_summary['min_snr']:.4g}, "
        f"relative-L2={selected_summary['min_relative_l2']:.4g}"
    )
    print(f"output: {output_dir}")
    print("Read before training:")
    print(f"  {output_dir / 'grid_search.csv'}")
    print(f"  {output_dir / 'state_nearest_neighbor_separation.csv'}")
    print(f"  {output_dir / 'grid_search_min_rms_snr.png'}")
    print("=" * 96)


if __name__ == "__main__":
    main()
