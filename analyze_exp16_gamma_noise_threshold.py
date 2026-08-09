#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp16: gamma observability threshold under the CURRENT physics.

Goal
----
Do not change the physical formula.  Instead ask:

1) If we use the statistically optimal information already present in all
   100 g(q^2) points, how detectable is a factor-2 change in gamma?
2) How low must the effective measurement noise become before gamma can be
   recovered usefully?
3) If the current single-measurement noise is 5% or 9%, how many independent
   repeated measurements would be needed (ideal 1/sqrt(M) averaging) to reach
   that effective noise floor?

This experiment does NOT train Transformer/TCN/LSTM.

Two optimistic estimators are compared:
  - exact_a1:
      a1,a2,a3,m are all known exactly; only gamma is unknown.
      This is an upper bound on what any network can do from the same g.
  - profile_a1:
      a2,a3,m are known, but a1 is unknown and is analytically profiled out
      for every candidate gamma.

For iid Gaussian noise, minimizing the full 100-point squared residual is the
maximum-likelihood estimator.  Thus this experiment already performs the
"matched-filter / coherent accumulation" that a neural network would have to
learn.

Important interpretation
------------------------
Multiplying g by 100 cannot improve SNR because signal and noise both scale.
A real improvement without changing the physics requires lower EFFECTIVE noise
(e.g. repeated independent measurements, better experimental precision, or an
additional independent observable).

Outputs
-------
- exp16_noise_sweep.csv
- exp16_critical_noise_summary.csv
- exp16_factor2_matched_filter.csv
- noise_sweep_factor2_success.png
- noise_sweep_median_factor_error.png
- gamma_*/factor2_residual_shape.png
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


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("empty numeric list")
    return values


def parse_args():
    p = argparse.ArgumentParser(
        description="Exp16 current-physics gamma noise-threshold experiment"
    )
    p.add_argument(
        "--output-dir",
        default=r".\validation_results\exp16_gamma_noise_threshold",
    )

    # Truth point, same controlled setting as Exp12-Exp15.
    p.add_argument("--a1-true", type=float, default=0.10)
    p.add_argument("--a2", type=float, default=0.025)
    p.add_argument("--a3", type=float, default=0.0)
    p.add_argument("--m", type=float, default=0.8)
    p.add_argument(
        "--true-gammas",
        default="0.01,0.05,0.2,0.5",
    )

    # Physical scan.
    p.add_argument("--a1-min", type=float, default=0.05)
    p.add_argument("--a1-max", type=float, default=0.20)
    p.add_argument("--gamma-min", type=float, default=0.001)
    p.add_argument("--gamma-max", type=float, default=1.0)
    p.add_argument("--gamma-points", type=int, default=1201)
    p.add_argument("--far-gamma-factor", type=float, default=2.0)

    # Noise sweep. Fractions: 0.05 = 5%.
    p.add_argument(
        "--noise-levels",
        default="0.09,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0",
    )
    p.add_argument(
        "--base-noise-levels",
        default="0.05,0.09",
        help="Used only to convert an effective noise into ideal repeat counts.",
    )
    p.add_argument("--trials", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260809)

    # Current observable.
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
    p = np.empty((a1_arr.size, 5), dtype=np.float32)
    p[:, 0] = a1_arr.reshape(-1)
    p[:, 1] = float(a2)
    p[:, 2] = float(a3)
    p[:, 3] = float(m)
    p[:, 4] = gamma_arr.reshape(-1)
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
    # Current forward map is exactly linear in a1 when the other parameters
    # are fixed.  Reuse OnlinePhysics to build that exact response.
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


def profile_a1_for_observation(
    g_obs: np.ndarray,
    g_background: np.ndarray,
    resonance_response: np.ndarray,
    a1_min: float,
    a1_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For one observed curve, analytically minimize the 100-point SSE over a1
    for every candidate gamma.
    """
    d = np.asarray(g_obs - g_background, dtype=np.float64)
    r = np.asarray(resonance_response, dtype=np.float64)

    rd = r @ d
    rr = np.sum(r * r, axis=1)
    a1_star = rd / np.maximum(rr, 1e-300)
    a1_star = np.clip(a1_star, float(a1_min), float(a1_max))

    dd = float(np.dot(d, d))
    sse = dd - 2.0 * a1_star * rd + (a1_star ** 2) * rr
    return a1_star, np.maximum(sse, 0.0)


def exact_a1_sse(
    g_obs: np.ndarray,
    g_background: np.ndarray,
    resonance_response: np.ndarray,
    a1_true: float,
) -> np.ndarray:
    candidates = (
        g_background.reshape(1, -1)
        + float(a1_true) * resonance_response
    )
    diff = candidates - np.asarray(g_obs, dtype=np.float64).reshape(1, -1)
    return np.sum(diff * diff, axis=1)


def factor_error(pred_gamma: np.ndarray, true_gamma: float) -> np.ndarray:
    pred = np.asarray(pred_gamma, dtype=np.float64)
    ratio = np.maximum(pred / float(true_gamma), 1e-300)
    return np.exp(np.abs(np.log(ratio)))


def make_noisy_batch(
    g_true: np.ndarray,
    noise_level: float,
    trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    g_true = np.asarray(g_true, dtype=np.float64)
    if noise_level == 0:
        return g_true.reshape(1, -1).copy()
    rms = float(np.sqrt(np.mean(g_true * g_true)))
    noise = (
        float(noise_level)
        * rms
        * rng.standard_normal((int(trials), len(g_true)))
    )
    return g_true.reshape(1, -1) + noise


def evaluate_estimator(
    estimator: str,
    noisy_batch: np.ndarray,
    gamma_grid: np.ndarray,
    g_background: np.ndarray,
    resonance_response: np.ndarray,
    args,
) -> np.ndarray:
    pred = np.empty(len(noisy_batch), dtype=np.float64)

    for i, g_obs in enumerate(noisy_batch):
        if estimator == "exact_a1":
            sse = exact_a1_sse(
                g_obs,
                g_background,
                resonance_response,
                args.a1_true,
            )
        elif estimator == "profile_a1":
            _, sse = profile_a1_for_observation(
                g_obs,
                g_background,
                resonance_response,
                args.a1_min,
                args.a1_max,
            )
        else:
            raise ValueError(estimator)

        pred[i] = gamma_grid[int(np.argmin(sse))]
    return pred


def hardest_factor2_alternative(
    gamma_grid: np.ndarray,
    g_true: np.ndarray,
    g_background: np.ndarray,
    resonance_response: np.ndarray,
    true_gamma: float,
    args,
    estimator: str,
) -> dict:
    if estimator == "exact_a1":
        candidates = (
            g_background.reshape(1, -1)
            + float(args.a1_true) * resonance_response
        )
        diff = candidates - g_true.reshape(1, -1)
        sse = np.sum(diff * diff, axis=1)
        best_a1 = np.full(len(gamma_grid), args.a1_true)
    elif estimator == "profile_a1":
        best_a1, sse = profile_a1_for_observation(
            g_true,
            g_background,
            resonance_response,
            args.a1_min,
            args.a1_max,
        )
        candidates = (
            g_background.reshape(1, -1)
            + best_a1[:, None] * resonance_response
        )
    else:
        raise ValueError(estimator)

    far = (
        (gamma_grid <= true_gamma / args.far_gamma_factor)
        | (gamma_grid >= true_gamma * args.far_gamma_factor)
    )
    ids = np.flatnonzero(far)
    j = ids[int(np.argmin(sse[ids]))]

    alt_g = candidates[j]
    delta = alt_g - g_true
    g_norm = max(float(np.linalg.norm(g_true)), 1e-300)
    rel = float(np.linalg.norm(delta) / g_norm)

    return {
        "alt_gamma": float(gamma_grid[j]),
        "alt_a1": float(best_a1[j]),
        "relative_l2": rel,
        "error_percent": 100.0 * rel,
        "delta": delta,
    }


def matched_filter_snr(
    delta: np.ndarray,
    g_true: np.ndarray,
    noise_level: float,
) -> float:
    if noise_level <= 0:
        return float("inf")
    rms = max(
        float(np.sqrt(np.mean(np.asarray(g_true, dtype=np.float64) ** 2))),
        1e-300,
    )
    sigma = float(noise_level) * rms
    return float(np.linalg.norm(delta) / sigma)


def required_noise_for_snr(
    delta: np.ndarray,
    g_true: np.ndarray,
    target_snr: float,
) -> float:
    """
    Return noise_level fraction such that ||delta|| / (noise*RMS(g)) = target.
    """
    rms = max(
        float(np.sqrt(np.mean(np.asarray(g_true, dtype=np.float64) ** 2))),
        1e-300,
    )
    return float(
        np.linalg.norm(delta)
        / (float(target_snr) * rms)
    )


def repeats_for_effective_noise(
    base_noise: float,
    target_noise: float,
) -> float:
    if target_noise <= 0:
        return float("inf")
    if target_noise >= base_noise:
        return 1.0
    return float((base_noise / target_noise) ** 2)


def gamma_tag(value: float) -> str:
    return "gamma_" + (
        f"{value:.8g}".replace("-", "m").replace(".", "p")
    )


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_residual(
    path: Path,
    q2: np.ndarray,
    delta_exact: np.ndarray,
    delta_profile: np.ndarray,
    true_gamma: float,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(q2, delta_exact, label="factor-2 residual, exact a1")
    ax.plot(q2, delta_profile, label="factor-2 residual, profiled a1")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("q^2")
    ax.set_ylabel("delta g")
    ax.set_title(
        f"Gamma signal shape across all q^2 points "
        f"(true gamma={true_gamma:g})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_noise_metric(
    path: Path,
    rows: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    log_y: bool = False,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    combos = []
    for r in rows:
        key = (float(r["true_gamma"]), r["estimator"])
        if key not in combos:
            combos.append(key)

    for true_gamma, estimator in combos:
        subset = [
            r for r in rows
            if float(r["true_gamma"]) == true_gamma
            and r["estimator"] == estimator
            and float(r["noise_level"]) > 0
        ]
        subset.sort(key=lambda x: float(x["noise_level"]))
        x = [100.0 * float(r["noise_level"]) for r in subset]
        y = [float(r[metric]) for r in subset]
        ax.plot(
            x,
            y,
            marker="o",
            label=f"gamma={true_gamma:g}, {estimator}",
        )

    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("effective noise (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()

    true_gammas = parse_float_list(args.true_gammas)
    noise_levels = parse_float_list(args.noise_levels)
    base_noise_levels = parse_float_list(args.base_noise_levels)

    if args.trials < 20:
        raise ValueError("--trials should be >= 20")
    if args.gamma_points < 100:
        raise ValueError("--gamma-points should be >= 100")
    if any(n < 0 for n in noise_levels):
        raise ValueError("noise levels must be >= 0")
    if any(n <= 0 for n in base_noise_levels):
        raise ValueError("base noise levels must be > 0")

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
    q2 = physics.q2_grid.detach().cpu().numpy().astype(np.float64)

    print("=" * 92)
    print("Exp16: gamma observability threshold under current physics")
    print("NO neural-network training")
    print(f"device       : {device}")
    print(f"q^2          : [{args.q2_min}, {args.q2_max}] x {args.input_points}")
    print(f"true gammas  : {true_gammas}")
    print(f"noise levels : {[100*n for n in noise_levels]} %")
    print(f"trials       : {args.trials}")
    print("=" * 92)

    noise_rows = []
    mf_rows = []
    rng = np.random.default_rng(args.seed)

    for true_gamma in true_gammas:
        true_params = parameter_rows(
            args.a1_true,
            true_gamma,
            a2=args.a2,
            a3=args.a3,
            m=args.m,
        )
        g_true = forward_numpy(physics, true_params, 1)[0]

        exact_alt = hardest_factor2_alternative(
            gamma_grid,
            g_true,
            g_background,
            resonance_response,
            true_gamma,
            args,
            "exact_a1",
        )
        profile_alt = hardest_factor2_alternative(
            gamma_grid,
            g_true,
            g_background,
            resonance_response,
            true_gamma,
            args,
            "profile_a1",
        )

        subdir = outdir / gamma_tag(true_gamma)
        subdir.mkdir(parents=True, exist_ok=True)
        plot_residual(
            subdir / "factor2_residual_shape.png",
            q2,
            exact_alt["delta"],
            profile_alt["delta"],
            true_gamma,
        )

        for estimator, alt in (
            ("exact_a1", exact_alt),
            ("profile_a1", profile_alt),
        ):
            row = {
                "true_gamma": float(true_gamma),
                "estimator": estimator,
                "hardest_factor2_alt_gamma": alt["alt_gamma"],
                "hardest_factor2_alt_a1": alt["alt_a1"],
                "factor2_g_error_percent": alt["error_percent"],
            }

            for noise in (0.09, 0.05, 0.01, 0.005, 0.001):
                row[f"matched_filter_snr_at_{100*noise:g}pct"] = (
                    matched_filter_snr(
                        alt["delta"],
                        g_true,
                        noise,
                    )
                )

            for target_snr in (1.0, 3.0, 5.0):
                required = required_noise_for_snr(
                    alt["delta"],
                    g_true,
                    target_snr,
                )
                row[f"required_noise_percent_for_snr{target_snr:g}"] = (
                    100.0 * required
                )
                for base in base_noise_levels:
                    row[
                        f"ideal_repeats_from_{100*base:g}pct_for_snr{target_snr:g}"
                    ] = repeats_for_effective_noise(base, required)

            mf_rows.append(row)

        # Same noisy realizations for the two estimators at each noise level.
        for noise_level in noise_levels:
            noisy = make_noisy_batch(
                g_true,
                noise_level,
                args.trials,
                rng,
            )

            for estimator in ("exact_a1", "profile_a1"):
                pred = evaluate_estimator(
                    estimator,
                    noisy,
                    gamma_grid,
                    g_background,
                    resonance_response,
                    args,
                )

                fac = factor_error(pred, true_gamma)
                log_abs = np.abs(np.log(np.maximum(pred / true_gamma, 1e-300)))

                row = {
                    "true_gamma": float(true_gamma),
                    "estimator": estimator,
                    "noise_level": float(noise_level),
                    "noise_percent": float(100.0 * noise_level),
                    "trials": int(len(pred)),
                    "median_pred_gamma": float(np.median(pred)),
                    "median_factor_error": float(np.median(fac)),
                    "p90_factor_error": float(np.quantile(fac, 0.90)),
                    "mean_abs_log_error": float(np.mean(log_abs)),
                    "success_within_x1p2": float(np.mean(fac <= 1.2)),
                    "success_within_x1p5": float(np.mean(fac <= 1.5)),
                    "success_within_x2": float(np.mean(fac <= 2.0)),
                }

                for base in base_noise_levels:
                    row[
                        f"ideal_repeats_equiv_from_{100*base:g}pct"
                    ] = (
                        1.0
                        if noise_level == 0
                        else repeats_for_effective_noise(
                            base,
                            noise_level,
                        )
                    )

                noise_rows.append(row)

    write_csv(
        outdir / "exp16_noise_sweep.csv",
        noise_rows,
    )
    write_csv(
        outdir / "exp16_factor2_matched_filter.csv",
        mf_rows,
    )

    # Critical noise: highest tested noise level that reaches the chosen
    # performance criterion. This is deliberately conservative and empirical.
    critical_rows = []
    for true_gamma in true_gammas:
        for estimator in ("exact_a1", "profile_a1"):
            subset = [
                r for r in noise_rows
                if float(r["true_gamma"]) == true_gamma
                and r["estimator"] == estimator
                and float(r["noise_level"]) > 0
            ]
            subset.sort(
                key=lambda r: float(r["noise_level"]),
                reverse=True,
            )

            criteria = {
                "90pct_within_x2": lambda r: r["success_within_x2"] >= 0.90,
                "90pct_within_x1p5": lambda r: r["success_within_x1p5"] >= 0.90,
                "median_within_x1p2": lambda r: r["median_factor_error"] <= 1.2,
            }

            for name, predicate in criteria.items():
                ok = [r for r in subset if predicate(r)]
                if ok:
                    best = ok[0]  # highest noise that still passes
                    noise = float(best["noise_level"])
                    row = {
                        "true_gamma": float(true_gamma),
                        "estimator": estimator,
                        "criterion": name,
                        "critical_noise_percent": 100.0 * noise,
                    }
                    for base in base_noise_levels:
                        row[
                            f"ideal_repeats_from_{100*base:g}pct"
                        ] = repeats_for_effective_noise(base, noise)
                else:
                    row = {
                        "true_gamma": float(true_gamma),
                        "estimator": estimator,
                        "criterion": name,
                        "critical_noise_percent": None,
                    }
                    for base in base_noise_levels:
                        row[
                            f"ideal_repeats_from_{100*base:g}pct"
                        ] = None
                critical_rows.append(row)

    write_csv(
        outdir / "exp16_critical_noise_summary.csv",
        critical_rows,
    )

    plot_noise_metric(
        outdir / "noise_sweep_factor2_success.png",
        noise_rows,
        "success_within_x2",
        "fraction of trials within factor 2",
        "Gamma recovery versus effective noise",
        log_y=False,
    )
    plot_noise_metric(
        outdir / "noise_sweep_median_factor_error.png",
        noise_rows,
        "median_factor_error",
        "median multiplicative gamma error",
        "Gamma factor error versus effective noise",
        log_y=True,
    )

    summary = {
        "settings": vars(args),
        "interpretation": {
            "exact_a1": (
                "optimistic upper bound: all nuisance parameters except gamma "
                "are known"
            ),
            "profile_a1": (
                "a1 is unknown but optimally fitted for every gamma candidate"
            ),
            "matched_filter_snr": (
                "optimal known-shape SNR across all q points under iid Gaussian "
                "noise; if this is <<1, no neural architecture can reliably "
                "distinguish those two gamma hypotheses from one measurement"
            ),
            "repeat_count": (
                "ideal independent-repeat averaging only; correlated/systematic "
                "errors will not improve as 1/sqrt(M)"
            ),
        },
    }
    with (outdir / "exp16_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("-" * 92)
    print("Factor-2 matched-filter upper-bound diagnostics:")
    for row in mf_rows:
        print(
            f"gamma={row['true_gamma']:g} | {row['estimator']:10s} | "
            f"g diff={row['factor2_g_error_percent']:.6g}% | "
            f"SNR@5%={row['matched_filter_snr_at_5pct']:.4g} | "
            f"noise for SNR3="
            f"{row['required_noise_percent_for_snr3']:.6g}%"
        )

    print("=" * 92)
    print("Exp16 finished")
    print(f"output: {outdir}")
    print("")
    print("FIRST READ:")
    print("  exp16_factor2_matched_filter.csv")
    print("  exp16_critical_noise_summary.csv")
    print("  exp16_noise_sweep.csv")
    print("")
    print("PLOTS:")
    print("  noise_sweep_factor2_success.png")
    print("  noise_sweep_median_factor_error.png")
    print("  gamma_*/factor2_residual_shape.png")
    print("=" * 92)


if __name__ == "__main__":
    main()
