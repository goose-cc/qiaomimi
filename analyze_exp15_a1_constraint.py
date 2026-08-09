#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp15: Does an independent constraint on a1 rescue gamma identifiability?

Motivation
----------
Exp12-Exp14 showed a strong a1-gamma degeneracy.  This experiment asks the
next causal question:

    Is gamma hard mainly because a1 can compensate it,
    or is g(q^2) intrinsically too insensitive to gamma even if a1 is known?

No neural network is trained.

For each true gamma and each assumed external uncertainty on a1, we restrict
a1 to

    a1_true * (1-u) <= a1 <= a1_true * (1+u),

clipped to the physical a1 range.  "free" uses the original [a1_min,a1_max],
and u=0 means a1 is known exactly.

For each candidate gamma we compute

    E_profile(gamma; a1 constraint)
      = min_allowed_a1 ||g(a1,gamma)-g_true||_2 / ||g_true||_2.

Main diagnostics
----------------
1) factor-2 wrong-gamma separation:
   among gamma values at least a factor 2 away from the truth, how close can
   the wrong model still get after the allowed a1 compensation?
   Higher is better.

2) compatible gamma interval at 0.1%, 0.5%, 1%, 5%, 9% g tolerance.

Interpretation
--------------
- If gamma becomes sharply identifiable once a1 is known to e.g. +/-1% or
  +/-5%, then the dominant problem is the a1-gamma compensation and an
  independent measurement/prior on a1 is valuable.
- If even exact a1 leaves a broad gamma-compatible region at 5%-9% tolerance,
  then the current observable itself is intrinsically too insensitive to
  gamma at realistic noise levels.

The project's existing OnlinePhysics forward implementation is reused.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

from mc_online_physics import OnlinePhysics


THRESHOLDS = (0.001, 0.005, 0.01, 0.05, 0.09)


def parse_float_list(text: str) -> list[float]:
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    if not out:
        raise ValueError("empty numeric list")
    return out


def parse_a1_constraints(text: str) -> list[tuple[str, float | None]]:
    """
    Examples:
        free,0.5,0.2,0.1,0.05,0.02,0.01,0
    returns:
        ("free", None), ("50%", 0.5), ..., ("exact", 0.0)
    """
    result = []
    seen = set()

    for raw in str(text).split(","):
        token = raw.strip().lower()
        if not token:
            continue

        if token == "free":
            key = ("free", None)
        else:
            value = float(token)
            if value < 0:
                raise ValueError("a1 relative uncertainty must be >= 0")
            if value == 0:
                key = ("exact", 0.0)
            else:
                key = (f"±{100.0*value:g}%", value)

        if key[0] not in seen:
            result.append(key)
            seen.add(key[0])

    if not result:
        raise ValueError("no a1 constraints supplied")
    return result


def parse_args():
    p = argparse.ArgumentParser(
        description="Exp15 a1-constraint -> gamma-identifiability diagnostic"
    )

    p.add_argument(
        "--output-dir",
        default=r".\validation_results\exp15_a1_constraint",
    )

    p.add_argument("--a1-true", type=float, default=0.10)
    p.add_argument("--a2", type=float, default=0.025)
    p.add_argument("--a3", type=float, default=0.0)
    p.add_argument("--m", type=float, default=0.8)
    p.add_argument(
        "--true-gammas",
        default="0.01,0.05,0.2,0.5",
    )

    p.add_argument("--a1-min", type=float, default=0.05)
    p.add_argument("--a1-max", type=float, default=0.20)

    p.add_argument(
        "--a1-constraints",
        default="free,0.5,0.2,0.1,0.05,0.02,0.01,0",
        help=(
            "Comma-separated relative a1 uncertainties. "
            "Example: free,0.2,0.1,0.05,0.01,0"
        ),
    )

    p.add_argument("--gamma-min", type=float, default=0.001)
    p.add_argument("--gamma-max", type=float, default=1.0)
    p.add_argument("--gamma-points", type=int, default=1601)
    p.add_argument("--far-gamma-factor", type=float, default=2.0)

    # Keep the original observable by default.
    p.add_argument("--q2-min", type=float, default=-100.0)
    p.add_argument("--q2-max", type=float, default=-6.0)
    p.add_argument("--input-points", type=int, default=100)
    p.add_argument("--output-points", type=int, default=1000)
    p.add_argument("--integration-points", type=int, default=128)
    p.add_argument("--data-scale", type=float, default=160000.0)
    p.add_argument("--shift", type=float, default=400.0)
    p.add_argument("--s-min", type=float, default=0.1764)
    p.add_argument("--s-max", type=float, default=6.0)
    p.add_argument(
        "--physics-dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "xpu"),
        default="cuda",
    )
    p.add_argument("--forward-batch-size", type=int, default=512)

    return p.parse_args()


def choose_device(name: str) -> torch.device:
    name = str(name).lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but CUDA is unavailable")
        return torch.device("cuda")
    if name == "xpu":
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU requested but XPU is unavailable")
        return torch.device("xpu")
    return torch.device("cpu")


def make_physics(args, device: torch.device) -> OnlinePhysics:
    physics_args = SimpleNamespace(
        physics_dtype=args.physics_dtype,
        noise_level=0.0,
        data_scale=args.data_scale,
        shift=args.shift,
        s_min=args.s_min,
        s_max=args.s_max,
        q2_min=args.q2_min,
        q2_max=args.q2_max,
        output_points=args.output_points,
        input_points=args.input_points,
        integration_points=args.integration_points,
        resonance_grid_points=max(args.output_points, 1000),
    )
    return OnlinePhysics(physics_args, device)


def parameter_rows(
    a1: np.ndarray | float,
    gamma: np.ndarray | float,
    *,
    a2: float,
    a3: float,
    m: float,
) -> np.ndarray:
    a1_arr, gamma_arr = np.broadcast_arrays(
        np.asarray(a1, dtype=np.float64),
        np.asarray(gamma, dtype=np.float64),
    )
    flat_a1 = a1_arr.reshape(-1)
    flat_gamma = gamma_arr.reshape(-1)

    p = np.empty((len(flat_a1), 5), dtype=np.float32)
    p[:, 0] = flat_a1
    p[:, 1] = float(a2)
    p[:, 2] = float(a3)
    p[:, 3] = float(m)
    p[:, 4] = flat_gamma
    return p


@torch.no_grad()
def forward_numpy(
    physics: OnlinePhysics,
    params_np: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    params_np = np.asarray(params_np, dtype=np.float32)
    outputs = []
    for start in range(0, len(params_np), int(batch_size)):
        stop = min(start + int(batch_size), len(params_np))
        params = torch.from_numpy(params_np[start:stop]).to(
            device=physics.device,
            dtype=torch.float32,
        )
        g = physics.forward_from_parameters(params)
        outputs.append(g.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outputs, axis=0)


def build_gamma_grid(args, true_gammas):
    grid = np.logspace(
        np.log10(args.gamma_min),
        np.log10(args.gamma_max),
        int(args.gamma_points),
        dtype=np.float64,
    )
    grid = np.unique(
        np.concatenate([grid, np.asarray(true_gammas, dtype=np.float64)])
    )
    grid.sort()
    return grid


def build_responses(
    physics: OnlinePhysics,
    args,
    gamma_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Current model is linear in a1.  Use the project forward code to obtain
    # the background and unit-a1 resonance response at every gamma.
    bg = forward_numpy(
        physics,
        parameter_rows(
            0.0,
            gamma_grid[0],
            a2=args.a2,
            a3=args.a3,
            m=args.m,
        ),
        1,
    )[0]

    a1_basis = 0.10
    basis = forward_numpy(
        physics,
        parameter_rows(
            np.full_like(gamma_grid, a1_basis),
            gamma_grid,
            a2=args.a2,
            a3=args.a3,
            m=args.m,
        ),
        args.forward_batch_size,
    )
    resonance = (basis - bg.reshape(1, -1)) / a1_basis
    return bg, resonance


def relative_l2_rows(candidate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64).reshape(1, -1)
    denom = np.linalg.norm(truth[0])
    return (
        np.linalg.norm(candidate - truth, axis=1)
        / max(denom, 1e-300)
    )


def allowed_a1_bounds(
    label: str,
    uncertainty: float | None,
    args,
) -> tuple[float, float]:
    if label == "free":
        return float(args.a1_min), float(args.a1_max)

    assert uncertainty is not None
    if uncertainty == 0:
        return float(args.a1_true), float(args.a1_true)

    lo = max(
        float(args.a1_min),
        float(args.a1_true) * (1.0 - float(uncertainty)),
    )
    hi = min(
        float(args.a1_max),
        float(args.a1_true) * (1.0 + float(uncertainty)),
    )
    if lo > hi:
        raise ValueError("invalid a1 constraint after clipping")
    return lo, hi


def profiled_error(
    resonance_response: np.ndarray,
    g_background: np.ndarray,
    g_true: np.ndarray,
    a1_lo: float,
    a1_hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(resonance_response, dtype=np.float64)
    d = np.asarray(g_true - g_background, dtype=np.float64)

    if abs(a1_hi - a1_lo) < 1e-15:
        a1_star = np.full(len(r), a1_lo, dtype=np.float64)
    else:
        numerator = np.sum(r * d.reshape(1, -1), axis=1)
        denominator = np.sum(r * r, axis=1)
        a1_star = numerator / np.maximum(denominator, 1e-300)
        a1_star = np.clip(a1_star, a1_lo, a1_hi)

    g_best = g_background.reshape(1, -1) + a1_star[:, None] * r
    profile = relative_l2_rows(g_best, g_true)
    return a1_star, profile


def compatible_interval(
    gamma_grid: np.ndarray,
    profile: np.ndarray,
    threshold: float,
) -> dict:
    mask = profile <= float(threshold)
    if not np.any(mask):
        return {
            "gamma_min": None,
            "gamma_max": None,
            "factor_span": None,
            "count": 0,
        }
    vals = gamma_grid[mask]
    return {
        "gamma_min": float(vals.min()),
        "gamma_max": float(vals.max()),
        "factor_span": float(vals.max() / vals.min()),
        "count": int(mask.sum()),
    }


def far_gamma_result(
    gamma_grid: np.ndarray,
    profile: np.ndarray,
    best_a1: np.ndarray,
    true_gamma: float,
    factor: float,
) -> dict:
    mask = (
        (gamma_grid <= true_gamma / factor)
        | (gamma_grid >= true_gamma * factor)
    )
    ids = np.flatnonzero(mask)
    if len(ids) == 0:
        raise RuntimeError("no far-gamma candidates")
    j = ids[np.argmin(profile[ids])]
    return {
        "alt_gamma": float(gamma_grid[j]),
        "alt_a1": float(best_a1[j]),
        "g_relative_l2": float(profile[j]),
        "g_error_percent": float(100.0 * profile[j]),
        "ratio_alt_over_true": float(gamma_grid[j] / true_gamma),
    }


def constraint_x_value(
    label: str,
    uncertainty: float | None,
) -> float:
    # For plotting only. "free" is placed at 100% relative uncertainty.
    if label == "free":
        return 1.0
    assert uncertainty is not None
    # Exact a1 at x=0.
    return float(uncertainty)


def plot_profiles(
    path: Path,
    gamma_grid: np.ndarray,
    curves: list[tuple[str, np.ndarray]],
    true_gamma: float,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Avoid clutter: show only the most informative subset if many constraints.
    preferred = {"free", "±10%", "±5%", "±1%", "exact"}
    shown = [item for item in curves if item[0] in preferred]
    if len(shown) < 3:
        shown = curves

    for label, profile in shown:
        ax.plot(
            gamma_grid,
            100.0 * np.maximum(profile, 1e-12),
            label=label,
        )

    ax.axvline(
        true_gamma,
        linestyle=":",
        linewidth=1.1,
        label="true gamma",
    )
    ax.axhline(5.0, linestyle="--", linewidth=0.8, label="5%")
    ax.axhline(9.0, linestyle="--", linewidth=0.8, label="9%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("candidate gamma")
    ax.set_ylabel("best g relative L2 error (%)")
    ax.set_title(
        f"Gamma profile as a1 becomes known "
        f"(true gamma={true_gamma:g})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_far_separation(
    path: Path,
    aggregate_rows: list[dict],
):
    # Aggregate row has worst case over true gammas for each constraint.
    rows = sorted(
        aggregate_rows,
        key=lambda r: r["plot_x"],
        reverse=True,
    )

    x = np.arange(len(rows))
    labels = [r["a1_constraint"] for r in rows]
    y = [r["worst_case_far_error_percent"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, marker="o")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(
        "worst-case factor-2 wrong-gamma g error (%)"
    )
    ax.set_xlabel("external knowledge of a1")
    ax.set_title(
        "Does constraining a1 make wrong gamma distinguishable? "
        "(higher is better)"
    )
    ax.axhline(5.0, linestyle="--", linewidth=0.8, label="5% noise")
    ax.axhline(9.0, linestyle="--", linewidth=0.8, label="9% noise")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_compatible_span(
    path: Path,
    rows: list[dict],
    threshold: float,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    true_gammas = sorted(set(float(r["true_gamma"]) for r in rows))
    constraints = []
    for r in rows:
        name = r["a1_constraint"]
        if name not in constraints:
            constraints.append(name)

    x = np.arange(len(constraints))

    for true_gamma in true_gammas:
        ys = []
        for name in constraints:
            matches = [
                r for r in rows
                if float(r["true_gamma"]) == true_gamma
                and r["a1_constraint"] == name
                and abs(float(r["threshold"]) - threshold) < 1e-15
            ]
            if not matches:
                ys.append(np.nan)
            else:
                span = matches[0]["compatible_factor_span"]
                ys.append(
                    float(span) if span is not None else np.nan
                )
        ax.plot(
            x,
            ys,
            marker="o",
            label=f"true gamma={true_gamma:g}",
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(constraints, rotation=30, ha="right")
    ax.set_ylabel("compatible gamma factor span")
    ax.set_xlabel("external knowledge of a1")
    ax.set_title(
        f"Gamma ambiguity at {100*threshold:g}% g tolerance "
        "(lower is better)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    true_gammas = parse_float_list(args.true_gammas)
    constraints = parse_a1_constraints(args.a1_constraints)

    if not (0 < args.a1_min < args.a1_max):
        raise ValueError("require 0 < a1_min < a1_max")
    if not (args.a1_min <= args.a1_true <= args.a1_max):
        raise ValueError("a1_true outside physical a1 range")
    if not (0 < args.gamma_min < args.gamma_max):
        raise ValueError("invalid gamma range")
    if args.far_gamma_factor <= 1:
        raise ValueError("--far-gamma-factor must be > 1")

    for g in true_gammas:
        if not (args.gamma_min <= g <= args.gamma_max):
            raise ValueError(f"true gamma={g} outside scan")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    physics = make_physics(args, device)
    gamma_grid = build_gamma_grid(args, true_gammas)
    g_background, resonance_response = build_responses(
        physics,
        args,
        gamma_grid,
    )

    print("=" * 90)
    print("Exp15: independent a1 constraint -> gamma identifiability")
    print("NO neural-network training")
    print(f"device        : {device}")
    print(f"q^2           : [{args.q2_min}, {args.q2_max}] x {args.input_points}")
    print(f"true a1       : {args.a1_true}")
    print(f"true gammas   : {true_gammas}")
    print(f"a1 constraints: {[x[0] for x in constraints]}")
    print("=" * 90)

    detail_rows = []
    far_rows = []
    per_truth_curves = {g: [] for g in true_gammas}

    for true_gamma in true_gammas:
        true_params = parameter_rows(
            args.a1_true,
            true_gamma,
            a2=args.a2,
            a3=args.a3,
            m=args.m,
        )
        g_true = forward_numpy(physics, true_params, 1)[0]

        for label, uncertainty in constraints:
            a1_lo, a1_hi = allowed_a1_bounds(
                label,
                uncertainty,
                args,
            )

            best_a1, profile = profiled_error(
                resonance_response,
                g_background,
                g_true,
                a1_lo,
                a1_hi,
            )
            per_truth_curves[true_gamma].append((label, profile))

            far = far_gamma_result(
                gamma_grid,
                profile,
                best_a1,
                true_gamma,
                args.far_gamma_factor,
            )

            far_rows.append(
                {
                    "true_gamma": float(true_gamma),
                    "a1_constraint": label,
                    "a1_relative_uncertainty": (
                        None if uncertainty is None else float(uncertainty)
                    ),
                    "a1_allowed_min": a1_lo,
                    "a1_allowed_max": a1_hi,
                    "far_alt_gamma": far["alt_gamma"],
                    "far_alt_a1": far["alt_a1"],
                    "far_alt_gamma_ratio": far["ratio_alt_over_true"],
                    "far_alt_g_error_percent": far["g_error_percent"],
                }
            )

            for threshold in THRESHOLDS:
                comp = compatible_interval(
                    gamma_grid,
                    profile,
                    threshold,
                )
                detail_rows.append(
                    {
                        "true_gamma": float(true_gamma),
                        "a1_constraint": label,
                        "a1_relative_uncertainty": (
                            None
                            if uncertainty is None
                            else float(uncertainty)
                        ),
                        "a1_allowed_min": a1_lo,
                        "a1_allowed_max": a1_hi,
                        "threshold": float(threshold),
                        "threshold_percent": float(100.0 * threshold),
                        "compatible_gamma_min": comp["gamma_min"],
                        "compatible_gamma_max": comp["gamma_max"],
                        "compatible_factor_span": comp["factor_span"],
                        "compatible_grid_count": comp["count"],
                        "far_alt_gamma": far["alt_gamma"],
                        "far_alt_a1": far["alt_a1"],
                        "far_alt_g_error_percent": far[
                            "g_error_percent"
                        ],
                    }
                )

        tag = (
            "gamma_"
            + f"{true_gamma:.8g}".replace("-", "m").replace(".", "p")
        )
        subdir = outdir / tag
        subdir.mkdir(parents=True, exist_ok=True)
        plot_profiles(
            subdir / "profiles_vs_a1_constraint.png",
            gamma_grid,
            per_truth_curves[true_gamma],
            true_gamma,
        )

    # Aggregate by constraint: worst case is the minimum separation across truths.
    aggregate_rows = []
    for label, uncertainty in constraints:
        subset = [r for r in far_rows if r["a1_constraint"] == label]
        vals = np.asarray(
            [r["far_alt_g_error_percent"] for r in subset],
            dtype=np.float64,
        )
        aggregate_rows.append(
            {
                "a1_constraint": label,
                "a1_relative_uncertainty": (
                    None if uncertainty is None else float(uncertainty)
                ),
                "plot_x": constraint_x_value(label, uncertainty),
                "worst_case_far_error_percent": float(np.min(vals)),
                "median_far_error_percent": float(np.median(vals)),
                "geometric_mean_far_error_percent": float(
                    np.exp(np.mean(np.log(np.maximum(vals, 1e-300))))
                ),
            }
        )

    plot_far_separation(
        outdir / "far_gamma_separation_vs_a1_constraint.png",
        aggregate_rows,
    )
    plot_compatible_span(
        outdir / "compatible_gamma_span_5pct.png",
        detail_rows,
        0.05,
    )
    plot_compatible_span(
        outdir / "compatible_gamma_span_9pct.png",
        detail_rows,
        0.09,
    )
    plot_compatible_span(
        outdir / "compatible_gamma_span_0p1pct.png",
        detail_rows,
        0.001,
    )

    write_csv(
        outdir / "exp15_a1_constraint_detail.csv",
        detail_rows,
    )
    write_csv(
        outdir / "exp15_far_gamma_separation.csv",
        far_rows,
    )
    write_csv(
        outdir / "exp15_ranking.csv",
        sorted(
            aggregate_rows,
            key=lambda r: r["worst_case_far_error_percent"],
            reverse=True,
        ),
    )

    summary = {
        "settings": vars(args),
        "interpretation": {
            "far_gamma_metric": (
                "minimum g error among candidate gammas at least factor "
                f"{args.far_gamma_factor:g} away from truth, after allowed "
                "a1 compensation; higher is better"
            ),
            "decision_rule": (
                "If even exact a1 leaves factor-2 gamma separation far below "
                "5%-9%, the observable itself is too insensitive to gamma at "
                "realistic noise levels."
            ),
        },
        "aggregate": aggregate_rows,
    }
    with (outdir / "exp15_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("-" * 90)
    print("Headline: worst-case factor-2 wrong-gamma separation")
    for row in aggregate_rows:
        print(
            f"{row['a1_constraint']:>8s} : "
            f"{row['worst_case_far_error_percent']:.8g}% "
            f"(median {row['median_far_error_percent']:.8g}%)"
        )

    print("=" * 90)
    print("Exp15 finished")
    print(f"output: {outdir}")
    print("")
    print("FIRST READ:")
    print("  exp15_ranking.csv")
    print("  exp15_a1_constraint_detail.csv")
    print("")
    print("PLOTS:")
    print("  far_gamma_separation_vs_a1_constraint.png")
    print("  compatible_gamma_span_5pct.png")
    print("  compatible_gamma_span_9pct.png")
    print("  gamma_*/profiles_vs_a1_constraint.png")
    print("=" * 90)


if __name__ == "__main__":
    main()
