#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp12: pure-physics a1-gamma identifiability / degeneracy scan.

This script DOES NOT train a neural network.

Question
--------
With the current physical formula and the same 100-point g(q^2) forward map,
if a1 is unknown, how well can gamma be distinguished?

For several true gamma values it:
  1) fixes a2, a3, m and a1_true;
  2) scans candidate (a1, gamma);
  3) computes relative-L2 error in g;
  4) profiles out a1:
         E_profile(gamma) = min_a1 ||g(a1,gamma)-g_true|| / ||g_true||;
  5) reports gamma ranges that remain below 0.1%, 0.5%, 1%, 5%, 9%;
  6) finds a deliberately far-away gamma (>= factor 2 different) that best
     reproduces g_true after re-optimizing a1;
  7) plots both g and f for the true and degenerate parameter sets.

Important implementation detail
-------------------------------
The current forward model is exactly linear in a1 when (m, gamma, a2, a3)
are fixed. We exploit that exact linearity to make the 2-D scan fast:

    g(a1,gamma) = g_background + a1 * r_gamma

where r_gamma is obtained from the project's own OnlinePhysics forward code.

No approximation to the physics is introduced by this acceleration.
A direct-forward sanity check is included.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

from mc_online_physics import OnlinePhysics


THRESHOLDS = (0.001, 0.005, 0.01, 0.05, 0.09)


def parse_gamma_list(text: str) -> list[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0:
            raise ValueError("all true gamma values must be > 0")
        values.append(value)
    if not values:
        raise ValueError("at least one true gamma is required")
    return values


def parse_args():
    p = argparse.ArgumentParser(
        description="Exp12 pure-physics a1-gamma degeneracy landscape"
    )

    p.add_argument(
        "--output-dir",
        default=r".\validation_results\exp12_a1_gamma_landscape",
    )

    # True/fixed physical point.
    p.add_argument("--a1-true", type=float, default=0.10)
    p.add_argument("--a2", type=float, default=0.025)
    p.add_argument("--a3", type=float, default=0.0)
    p.add_argument("--m", type=float, default=0.8)
    p.add_argument(
        "--true-gammas",
        default="0.01,0.05,0.2,0.5",
        help="comma-separated true gamma values",
    )

    # Scan domain.
    p.add_argument("--a1-min", type=float, default=0.05)
    p.add_argument("--a1-max", type=float, default=0.20)
    p.add_argument("--a1-points", type=int, default=241)
    p.add_argument("--gamma-min", type=float, default=0.001)
    p.add_argument("--gamma-max", type=float, default=1.0)
    p.add_argument("--gamma-points", type=int, default=361)
    p.add_argument(
        "--far-gamma-factor",
        type=float,
        default=2.0,
        help=(
            "Degenerate alternative must differ from true gamma by at least "
            "this multiplicative factor."
        ),
    )

    # Same physics settings as Exp7/Exp11.
    p.add_argument("--input-points", type=int, default=100)
    p.add_argument("--output-points", type=int, default=1000)
    p.add_argument("--integration-points", type=int, default=128)
    p.add_argument("--data-scale", type=float, default=160000.0)
    p.add_argument("--shift", type=float, default=400.0)
    p.add_argument("--s-min", type=float, default=0.1764)
    p.add_argument("--s-max", type=float, default=6.0)
    p.add_argument("--q2-min", type=float, default=-100.0)
    p.add_argument("--q2-max", type=float, default=-6.0)
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
    p.add_argument(
        "--forward-batch-size",
        type=int,
        default=512,
    )
    p.add_argument(
        "--linearity-checks",
        type=int,
        default=5,
    )
    p.add_argument("--seed", type=int, default=20260808)

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
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
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


@torch.no_grad()
def curves_numpy(
    physics: OnlinePhysics,
    params_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    params = torch.from_numpy(
        np.asarray(params_np, dtype=np.float32)
    ).to(
        device=physics.device,
        dtype=torch.float32,
    )
    f, g = physics.make_clean_batch(params)
    return (
        f.detach().cpu().numpy().astype(np.float64),
        g.detach().cpu().numpy().astype(np.float64),
    )


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


def relative_l2_rows(candidate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.float64).reshape(1, -1)
    candidate = np.asarray(candidate, dtype=np.float64)
    denom = np.linalg.norm(truth[0])
    if denom <= 0:
        raise ValueError("true g has zero norm")
    return np.linalg.norm(candidate - truth, axis=1) / denom


def gamma_tag(value: float) -> str:
    text = f"{value:.8g}"
    return "gamma_" + text.replace("-", "m").replace(".", "p")


def atomic_json_dump(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def threshold_interval(
    gamma_grid: np.ndarray,
    profile: np.ndarray,
    threshold: float,
) -> dict:
    mask = profile <= float(threshold)
    if not np.any(mask):
        return {
            "threshold": float(threshold),
            "has_solution": False,
            "gamma_min": None,
            "gamma_max": None,
            "factor_span": None,
            "count": 0,
        }

    values = gamma_grid[mask]
    return {
        "threshold": float(threshold),
        "has_solution": True,
        "gamma_min": float(values.min()),
        "gamma_max": float(values.max()),
        "factor_span": float(values.max() / values.min()),
        "count": int(mask.sum()),
    }


def continuous_best_a1(
    resonance_response: np.ndarray,
    g_background: np.ndarray,
    g_true: np.ndarray,
    a1_min: float,
    a1_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each gamma, analytically minimize ||g_bg + a1*r_gamma - g_true||_2
    over a1, then clamp to the scan's physical a1 interval.
    """
    r = np.asarray(resonance_response, dtype=np.float64)
    d = np.asarray(g_true - g_background, dtype=np.float64)

    numerator = np.sum(r * d.reshape(1, -1), axis=1)
    denominator = np.sum(r * r, axis=1)

    a1_star = numerator / np.maximum(denominator, 1e-300)
    a1_star = np.clip(a1_star, float(a1_min), float(a1_max))

    g_best = g_background.reshape(1, -1) + a1_star[:, None] * r
    profile = relative_l2_rows(g_best, g_true)
    return a1_star, profile


def grid_error_landscape(
    resonance_response: np.ndarray,
    g_background: np.ndarray,
    g_true: np.ndarray,
    a1_grid: np.ndarray,
) -> np.ndarray:
    """
    Return [N_gamma, N_a1] relative-L2 errors without materializing
    [N_gamma, N_a1, N_q] all at once.
    """
    denom = np.linalg.norm(g_true)
    result = np.empty(
        (len(resonance_response), len(a1_grid)),
        dtype=np.float64,
    )

    d = g_background - g_true
    for i, r in enumerate(resonance_response):
        # candidate - true = d + a1*r
        diff = d.reshape(1, -1) + a1_grid[:, None] * r.reshape(1, -1)
        result[i, :] = np.linalg.norm(diff, axis=1) / denom
    return result


def run_linearity_check(
    physics: OnlinePhysics,
    gamma_grid: np.ndarray,
    g_background: np.ndarray,
    resonance_response: np.ndarray,
    args,
) -> dict:
    rng = np.random.default_rng(args.seed)
    checks = max(0, int(args.linearity_checks))
    if checks == 0:
        return {
            "checks": 0,
            "max_relative_difference": None,
            "mean_relative_difference": None,
        }

    rels = []
    for _ in range(checks):
        gi = int(rng.integers(0, len(gamma_grid)))
        a1 = float(rng.uniform(args.a1_min, args.a1_max))
        gamma = float(gamma_grid[gi])

        direct = forward_numpy(
            physics,
            parameter_rows(
                a1,
                gamma,
                a2=args.a2,
                a3=args.a3,
                m=args.m,
            ),
            batch_size=1,
        )[0]

        reconstructed = (
            g_background + a1 * resonance_response[gi]
        )
        rel = np.linalg.norm(direct - reconstructed) / max(
            np.linalg.norm(direct),
            1e-300,
        )
        rels.append(float(rel))

    return {
        "checks": int(checks),
        "max_relative_difference": float(max(rels)),
        "mean_relative_difference": float(np.mean(rels)),
    }


def save_profile_csv(
    path: Path,
    gamma_grid: np.ndarray,
    best_a1: np.ndarray,
    continuous_profile: np.ndarray,
    grid_best_a1: np.ndarray,
    grid_profile: np.ndarray,
):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "gamma",
                "continuous_best_a1",
                "continuous_g_relative_l2",
                "continuous_g_error_percent",
                "grid_best_a1",
                "grid_g_relative_l2",
                "grid_g_error_percent",
            ]
        )
        for i in range(len(gamma_grid)):
            writer.writerow(
                [
                    f"{gamma_grid[i]:.12g}",
                    f"{best_a1[i]:.12g}",
                    f"{continuous_profile[i]:.12g}",
                    f"{100.0 * continuous_profile[i]:.12g}",
                    f"{grid_best_a1[i]:.12g}",
                    f"{grid_profile[i]:.12g}",
                    f"{100.0 * grid_profile[i]:.12g}",
                ]
            )


def plot_heatmap(
    path: Path,
    a1_grid: np.ndarray,
    gamma_grid: np.ndarray,
    landscape: np.ndarray,
    true_a1: float,
    true_gamma: float,
):
    fig, ax = plt.subplots(figsize=(9, 7))

    # log10(error) makes the low-error degeneracy valley visible.
    shown = np.log10(np.maximum(landscape, 1e-8))
    mesh = ax.pcolormesh(
        a1_grid,
        gamma_grid,
        shown,
        shading="auto",
    )
    ax.set_yscale("log")
    ax.set_xlabel("candidate a1")
    ax.set_ylabel("candidate gamma")
    ax.set_title(
        f"a1-gamma forward degeneracy landscape\n"
        f"true a1={true_a1:g}, true gamma={true_gamma:g}"
    )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("log10(g relative L2 error)")

    # Useful practical noise/error contours.
    try:
        contours = ax.contour(
            a1_grid,
            gamma_grid,
            landscape,
            levels=list(THRESHOLDS),
        )
        ax.clabel(
            contours,
            inline=True,
            fontsize=8,
            fmt=lambda x: f"{100*x:g}%",
        )
    except ValueError:
        pass

    ax.scatter(
        [true_a1],
        [true_gamma],
        marker="x",
        s=90,
        linewidths=2,
        label="truth",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_profile(
    path: Path,
    gamma_grid: np.ndarray,
    profile: np.ndarray,
    best_a1: np.ndarray,
    true_gamma: float,
):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(gamma_grid, 100.0 * profile, label="min over a1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("candidate gamma")
    ax.set_ylabel("best g relative L2 error (%)")
    ax.set_title(
        f"Profiled gamma distinguishability (true gamma={true_gamma:g})"
    )

    for threshold in THRESHOLDS:
        ax.axhline(
            100.0 * threshold,
            linestyle="--",
            linewidth=0.8,
            label=f"{100*threshold:g}% threshold",
        )

    ax.axvline(
        true_gamma,
        linestyle=":",
        linewidth=1.2,
        label="true gamma",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(gamma_grid, best_a1)
    ax.set_xscale("log")
    ax.set_xlabel("candidate gamma")
    ax.set_ylabel("best compensating a1")
    ax.set_title(
        f"Best a1 compensation versus gamma (true gamma={true_gamma:g})"
    )
    fig.tight_layout()
    second_path = path.with_name(
        path.stem + "_best_a1" + path.suffix
    )
    fig.savefig(second_path, dpi=180)
    plt.close(fig)


def plot_comparisons(
    subdir: Path,
    physics: OnlinePhysics,
    true_params: np.ndarray,
    alt_params: np.ndarray,
    true_gamma: float,
    alt_gamma: float,
    alt_a1: float,
    alt_g_error: float,
):
    f, g = curves_numpy(
        physics,
        np.vstack([true_params, alt_params]),
    )
    f_true, f_alt = f[0], f[1]
    g_true, g_alt = g[0], g[1]

    q2 = physics.q2_grid.detach().cpu().numpy()
    s = physics.s_output.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(q2, g_true, label="true")
    ax.plot(q2, g_alt, linestyle="--", label="degenerate alternative")
    ax.set_xlabel("q^2")
    ax.set_ylabel("scaled g(q^2)")
    ax.set_title(
        "Nearly identical g from different gamma\n"
        f"true gamma={true_gamma:g}; alt gamma={alt_gamma:g}, "
        f"alt a1={alt_a1:.5g}; g error={100*alt_g_error:.4g}%"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        subdir / "degenerate_g_comparison.png",
        dpi=180,
    )
    plt.close(fig)

    f_rel = np.linalg.norm(f_alt - f_true) / max(
        np.linalg.norm(f_true),
        1e-300,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(s, f_true, label="true")
    ax.plot(s, f_alt, linestyle="--", label="degenerate alternative")
    ax.set_xlabel("s")
    ax.set_ylabel("scaled f(s)")
    ax.set_title(
        "Spectra corresponding to the nearly identical g curves\n"
        f"f relative L2 difference={100*f_rel:.4g}%"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        subdir / "degenerate_f_comparison.png",
        dpi=180,
    )
    plt.close(fig)

    return float(f_rel)


def main():
    args = parse_args()

    if not (0 < args.a1_min < args.a1_max):
        raise ValueError("require 0 < a1_min < a1_max")
    if not (0 < args.gamma_min < args.gamma_max):
        raise ValueError("require 0 < gamma_min < gamma_max")
    if args.a1_points < 3 or args.gamma_points < 3:
        raise ValueError("scan point counts must be >= 3")
    if args.far_gamma_factor <= 1:
        raise ValueError("--far-gamma-factor must be > 1")

    true_gammas = parse_gamma_list(args.true_gammas)

    for gamma in true_gammas:
        if not (args.gamma_min <= gamma <= args.gamma_max):
            raise ValueError(
                f"true gamma={gamma} lies outside scan range "
                f"[{args.gamma_min}, {args.gamma_max}]"
            )
    if not (args.a1_min <= args.a1_true <= args.a1_max):
        raise ValueError("a1_true must lie inside the a1 scan range")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    physics = make_physics(args, device)

    a1_grid = np.linspace(
        args.a1_min,
        args.a1_max,
        args.a1_points,
        dtype=np.float64,
    )

    # Include every exact truth in the common log grid.
    gamma_grid = np.logspace(
        np.log10(args.gamma_min),
        np.log10(args.gamma_max),
        args.gamma_points,
        dtype=np.float64,
    )
    gamma_grid = np.unique(
        np.concatenate(
            [
                gamma_grid,
                np.asarray(true_gammas, dtype=np.float64),
            ]
        )
    )
    gamma_grid.sort()

    print("=" * 80)
    print("Exp12 pure-physics a1-gamma landscape")
    print("NO neural-network training")
    print(f"device              : {device}")
    print(f"true a1             : {args.a1_true}")
    print(f"fixed m             : {args.m}")
    print(f"fixed a2 / a3       : {args.a2} / {args.a3}")
    print(f"true gammas         : {true_gammas}")
    print(
        f"a1 scan             : [{args.a1_min}, {args.a1_max}] "
        f"x {len(a1_grid)}"
    )
    print(
        f"gamma scan          : [{args.gamma_min}, {args.gamma_max}] "
        f"x {len(gamma_grid)} (log-spaced + truths)"
    )
    print("=" * 80)

    # Background: a1 = 0, gamma is irrelevant.
    bg_params = parameter_rows(
        0.0,
        true_gammas[0],
        a2=args.a2,
        a3=args.a3,
        m=args.m,
    )
    g_background = forward_numpy(
        physics,
        bg_params,
        batch_size=1,
    )[0]

    # One forward per gamma at a1_basis, then divide out a1.
    a1_basis = 0.10
    basis_params = parameter_rows(
        np.full_like(gamma_grid, a1_basis),
        gamma_grid,
        a2=args.a2,
        a3=args.a3,
        m=args.m,
    )
    g_basis = forward_numpy(
        physics,
        basis_params,
        batch_size=args.forward_batch_size,
    )
    resonance_response = (
        g_basis - g_background.reshape(1, -1)
    ) / a1_basis

    linearity = run_linearity_check(
        physics,
        gamma_grid,
        g_background,
        resonance_response,
        args,
    )
    print(
        "a1 linearity check  : "
        f"max rel diff={linearity['max_relative_difference']}"
    )

    root_summary = {
        "settings": {
            "a1_true": args.a1_true,
            "a2": args.a2,
            "a3": args.a3,
            "m": args.m,
            "a1_min": args.a1_min,
            "a1_max": args.a1_max,
            "a1_points": len(a1_grid),
            "gamma_min": args.gamma_min,
            "gamma_max": args.gamma_max,
            "gamma_points": len(gamma_grid),
            "thresholds": list(THRESHOLDS),
            "far_gamma_factor": args.far_gamma_factor,
            "physics_dtype": args.physics_dtype,
            "device": str(device),
        },
        "linearity_check": linearity,
        "truths": [],
    }

    summary_rows = []

    for true_gamma in true_gammas:
        subdir = outdir / gamma_tag(true_gamma)
        subdir.mkdir(parents=True, exist_ok=True)

        true_params = parameter_rows(
            args.a1_true,
            true_gamma,
            a2=args.a2,
            a3=args.a3,
            m=args.m,
        )[0]

        g_true = forward_numpy(
            physics,
            true_params.reshape(1, 5),
            batch_size=1,
        )[0]

        # Full grid for heatmap.
        landscape = grid_error_landscape(
            resonance_response,
            g_background,
            g_true,
            a1_grid,
        )

        grid_best_idx = np.argmin(landscape, axis=1)
        grid_profile = landscape[
            np.arange(len(gamma_grid)),
            grid_best_idx,
        ]
        grid_best_a1 = a1_grid[grid_best_idx]

        # Continuous analytical profile over a1.
        best_a1, profile = continuous_best_a1(
            resonance_response,
            g_background,
            g_true,
            args.a1_min,
            args.a1_max,
        )

        intervals = [
            threshold_interval(
                gamma_grid,
                profile,
                threshold,
            )
            for threshold in THRESHOLDS
        ]

        far_mask = (
            (gamma_grid <= true_gamma / args.far_gamma_factor)
            | (gamma_grid >= true_gamma * args.far_gamma_factor)
        )

        if np.any(far_mask):
            far_ids = np.flatnonzero(far_mask)
            alt_id = far_ids[
                np.argmin(profile[far_ids])
            ]
        else:
            # This should not happen with the default domain, but keep it safe.
            distance = np.abs(
                np.log(gamma_grid / true_gamma)
            )
            alt_id = int(np.argmax(distance))

        alt_gamma = float(gamma_grid[alt_id])
        alt_a1 = float(best_a1[alt_id])
        alt_g_error = float(profile[alt_id])

        alt_params = parameter_rows(
            alt_a1,
            alt_gamma,
            a2=args.a2,
            a3=args.a3,
            m=args.m,
        )[0]

        f_rel = plot_comparisons(
            subdir,
            physics,
            true_params,
            alt_params,
            true_gamma,
            alt_gamma,
            alt_a1,
            alt_g_error,
        )

        save_profile_csv(
            subdir / "gamma_profile.csv",
            gamma_grid,
            best_a1,
            profile,
            grid_best_a1,
            grid_profile,
        )

        np.savez_compressed(
            subdir / "a1_gamma_landscape.npz",
            a1_grid=a1_grid,
            gamma_grid=gamma_grid,
            g_relative_l2=landscape,
            continuous_best_a1=best_a1,
            continuous_profile=profile,
            grid_best_a1=grid_best_a1,
            grid_profile=grid_profile,
        )

        plot_heatmap(
            subdir / "a1_gamma_heatmap.png",
            a1_grid,
            gamma_grid,
            landscape,
            args.a1_true,
            true_gamma,
        )
        plot_profile(
            subdir / "gamma_profile.png",
            gamma_grid,
            profile,
            best_a1,
            true_gamma,
        )

        truth_summary = {
            "true_parameters": {
                "a1": float(args.a1_true),
                "a2": float(args.a2),
                "a3": float(args.a3),
                "m": float(args.m),
                "gamma": float(true_gamma),
            },
            "threshold_intervals": intervals,
            "far_degenerate_alternative": {
                "criterion": (
                    f"gamma differs by at least factor "
                    f"{args.far_gamma_factor:g}"
                ),
                "a1": alt_a1,
                "gamma": alt_gamma,
                "gamma_ratio_alt_over_true": float(
                    alt_gamma / true_gamma
                ),
                "g_relative_l2": alt_g_error,
                "g_error_percent": float(100.0 * alt_g_error),
                "f_relative_l2": f_rel,
                "f_difference_percent": float(100.0 * f_rel),
            },
        }

        atomic_json_dump(
            truth_summary,
            subdir / "summary.json",
        )
        root_summary["truths"].append(truth_summary)

        interval_map = {
            entry["threshold"]: entry
            for entry in intervals
        }

        row = {
            "true_gamma": true_gamma,
            "alt_gamma": alt_gamma,
            "alt_a1": alt_a1,
            "alt_g_error_percent": 100.0 * alt_g_error,
            "alt_f_difference_percent": 100.0 * f_rel,
        }
        for threshold in THRESHOLDS:
            entry = interval_map[threshold]
            tag = f"{100*threshold:g}pct"
            row[f"{tag}_gamma_min"] = entry["gamma_min"]
            row[f"{tag}_gamma_max"] = entry["gamma_max"]
            row[f"{tag}_factor_span"] = entry["factor_span"]
        summary_rows.append(row)

        print("-" * 80)
        print(f"true gamma          : {true_gamma:g}")
        print(
            "far alternative     : "
            f"a1={alt_a1:.6g}, gamma={alt_gamma:.6g} "
            f"(x{alt_gamma/true_gamma:.4g})"
        )
        print(
            "far alternative g   : "
            f"{100.0 * alt_g_error:.6g}% relative L2"
        )
        print(
            "far alternative f   : "
            f"{100.0 * f_rel:.6g}% relative L2 difference"
        )
        for entry in intervals:
            if entry["has_solution"]:
                print(
                    f"<= {100*entry['threshold']:g}% g error : "
                    f"gamma in [{entry['gamma_min']:.6g}, "
                    f"{entry['gamma_max']:.6g}] "
                    f"(span x{entry['factor_span']:.4g})"
                )
            else:
                print(
                    f"<= {100*entry['threshold']:g}% g error : none"
                )

    atomic_json_dump(
        root_summary,
        outdir / "exp12_summary.json",
    )

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with (outdir / "exp12_summary.csv").open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    print("=" * 80)
    print("Exp12 finished")
    print(f"output: {outdir}")
    print("Main files:")
    print("  exp12_summary.csv")
    print("  exp12_summary.json")
    print("  gamma_*/a1_gamma_heatmap.png")
    print("  gamma_*/gamma_profile.png")
    print("  gamma_*/degenerate_g_comparison.png")
    print("  gamma_*/degenerate_f_comparison.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
