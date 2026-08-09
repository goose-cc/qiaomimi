#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp14: gamma prior / compatible-interval / posterior diagnostic.

Purpose
-------
Exp12/Exp13 showed that, for the current single observable g(q^2), different
gamma values can produce almost indistinguishable g after a1 compensation.

This experiment does NOT train a neural network.  It asks the next practical
question:

    If the observable cannot uniquely determine gamma, how much can a
    physically justified gamma prior help, and what gamma interval / posterior
    is actually supported by the data?

The experiment reuses the project's exact OnlinePhysics forward map.

Outputs
-------
1) Profiled physics error:
       E_profile(gamma) = min_a1 ||g(a1,gamma)-g_true|| / ||g_true||

2) Prior-shrinkage diagnostic:
   For an oracle-centered prior window
       gamma_true/F <= gamma <= gamma_true*F
   and an observational tolerance epsilon, compute the compatible set
       C = {gamma in prior : E_profile(gamma) <= epsilon}.
   The quantity
       log-width(C) / log-width(prior)
   tells how much the data narrows the prior.
       ~1 : data contributes almost no gamma information.
       <<1: data meaningfully narrows the prior.

   The oracle-centered prior is ONLY a diagnostic for "how strong would the
   prior need to be?".  It is not a deployable prior because it uses the truth
   to center the window.

3) Gaussian-likelihood posterior:
   Using the same noise model as the project,
       sigma = noise_level * RMS(g_true)
   and uniform priors in a1 and gamma over the selected prior window,
   numerically marginalize over a1 and report the posterior p(gamma|g).

   The observed curve is kept equal to clean g_true.  This is an optimistic
   expected-identifiability test: if the posterior is already broad before
   drawing a noisy realization, adding actual noise will not make gamma more
   identifiable.

No Transformer/TCN/LSTM is used here.
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


PARAMETER_NAMES = ("a1", "a2", "a3", "m", "gamma")


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
        description="Exp14 gamma prior and compatible-interval diagnostic"
    )

    p.add_argument(
        "--output-dir",
        default=r".\validation_results\exp14_gamma_prior_interval",
    )

    # Fixed truth settings, matched to Exp12/Exp13.
    p.add_argument("--a1-true", type=float, default=0.10)
    p.add_argument("--a2", type=float, default=0.025)
    p.add_argument("--a3", type=float, default=0.0)
    p.add_argument("--m", type=float, default=0.8)
    p.add_argument(
        "--true-gammas",
        default="0.01,0.05,0.2,0.5",
    )

    # Current physical parameter domain.
    p.add_argument("--a1-min", type=float, default=0.05)
    p.add_argument("--a1-max", type=float, default=0.20)
    p.add_argument("--a1-points", type=int, default=401)
    p.add_argument("--gamma-min", type=float, default=0.001)
    p.add_argument("--gamma-max", type=float, default=1.0)
    p.add_argument("--gamma-points", type=int, default=1601)

    # Oracle-centered prior factors used ONLY to quantify required prior strength.
    p.add_argument(
        "--prior-factors",
        default="1000,100,30,10,5,3,2,1.5,1.2",
        help=(
            "For diagnostic factor F, prior is "
            "[gamma_true/F, gamma_true*F], clipped to physical bounds."
        ),
    )

    # Compatibility thresholds in RELATIVE units, not percent.
    p.add_argument(
        "--compatibility-thresholds",
        default="0.001,0.005,0.01,0.05,0.09",
        help="0.001=0.1%, 0.05=5%, 0.09=9%",
    )

    # Posterior assumes the project's iid RMS-scaled Gaussian noise.
    p.add_argument(
        "--posterior-noise-levels",
        default="0.05,0.09",
        help="Noise fractions, e.g. 0.05=5%.",
    )

    # Use the ORIGINAL observable by default, because Exp13 showed q redesign
    # helps only weakly. These remain configurable for comparison.
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
    p.add_argument("--seed", type=int, default=20260809)

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
        outputs.append(
            g.detach().cpu().numpy().astype(np.float64)
        )
    return np.concatenate(outputs, axis=0)


def build_gamma_grid(args, true_gammas: list[float]) -> np.ndarray:
    # Log spacing resolves the narrow-gamma region while exact truths are inserted.
    gamma_grid = np.logspace(
        np.log10(args.gamma_min),
        np.log10(args.gamma_max),
        int(args.gamma_points),
        dtype=np.float64,
    )
    gamma_grid = np.unique(
        np.concatenate(
            [gamma_grid, np.asarray(true_gammas, dtype=np.float64)]
        )
    )
    gamma_grid.sort()
    return gamma_grid


def build_responses(
    physics: OnlinePhysics,
    args,
    gamma_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bg = forward_numpy(
        physics,
        parameter_rows(
            0.0,
            gamma_grid[0],
            a2=args.a2,
            a3=args.a3,
            m=args.m,
        ),
        batch_size=1,
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
        batch_size=args.forward_batch_size,
    )
    resonance = (basis - bg.reshape(1, -1)) / a1_basis
    return bg, resonance


def relative_l2_rows(
    candidate: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64).reshape(1, -1)
    denom = np.linalg.norm(truth[0])
    return (
        np.linalg.norm(candidate - truth, axis=1)
        / max(denom, 1e-300)
    )


def continuous_best_a1(
    resonance_response: np.ndarray,
    g_background: np.ndarray,
    g_true: np.ndarray,
    a1_min: float,
    a1_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(resonance_response, dtype=np.float64)
    d = np.asarray(g_true - g_background, dtype=np.float64)

    numerator = np.sum(r * d.reshape(1, -1), axis=1)
    denominator = np.sum(r * r, axis=1)
    a1_star = numerator / np.maximum(denominator, 1e-300)
    a1_star = np.clip(a1_star, float(a1_min), float(a1_max))

    g_best = g_background.reshape(1, -1) + a1_star[:, None] * r
    profile = relative_l2_rows(g_best, g_true)
    return a1_star, profile


def prior_bounds(
    true_gamma: float,
    factor: float,
    gamma_min: float,
    gamma_max: float,
) -> tuple[float, float]:
    if factor <= 1.0:
        raise ValueError("prior factor must be > 1")
    lo = max(float(gamma_min), float(true_gamma) / float(factor))
    hi = min(float(gamma_max), float(true_gamma) * float(factor))
    return lo, hi


def log_width(lo: float, hi: float) -> float:
    if not (0 < lo < hi):
        return 0.0
    return float(np.log(hi / lo))


def compatible_interval(
    gamma_grid: np.ndarray,
    profile: np.ndarray,
    prior_lo: float,
    prior_hi: float,
    threshold: float,
) -> dict:
    prior_mask = (
        (gamma_grid >= float(prior_lo))
        & (gamma_grid <= float(prior_hi))
    )
    compatible = prior_mask & (profile <= float(threshold))

    if not np.any(compatible):
        return {
            "has_solution": False,
            "gamma_min": None,
            "gamma_max": None,
            "factor_span": None,
            "prior_log_width": log_width(prior_lo, prior_hi),
            "compatible_log_width": 0.0,
            "shrinkage_ratio": 0.0,
        }

    vals = gamma_grid[compatible]
    c_lo = float(vals.min())
    c_hi = float(vals.max())
    p_width = log_width(prior_lo, prior_hi)
    c_width = log_width(c_lo, c_hi)

    # 1 means the data did not narrow the prior at all.
    ratio = c_width / p_width if p_width > 0 else 0.0

    return {
        "has_solution": True,
        "gamma_min": c_lo,
        "gamma_max": c_hi,
        "factor_span": float(c_hi / c_lo) if c_lo > 0 else None,
        "prior_log_width": p_width,
        "compatible_log_width": c_width,
        "shrinkage_ratio": float(ratio),
    }


def trapz_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 1:
        return np.ones(1, dtype=np.float64)
    w = np.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    if len(x) > 2:
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return np.maximum(w, 1e-300)


def posterior_gamma(
    gamma_grid: np.ndarray,
    resonance_response: np.ndarray,
    g_background: np.ndarray,
    g_observed: np.ndarray,
    *,
    a1_min: float,
    a1_max: float,
    a1_points: int,
    prior_lo: float,
    prior_hi: float,
    noise_level: float,
) -> dict:
    """
    Marginalize a1 under uniform priors:
        p(gamma | g) ∝ ∫ p(g | a1,gamma) p(a1) da1
    with iid sigma = noise_level * RMS(g_observed).

    The gamma prior density is uniform in gamma inside [prior_lo, prior_hi].
    Non-uniform gamma-grid spacing is handled by trapezoidal integration weights.
    """
    if noise_level <= 0:
        raise ValueError("posterior noise level must be > 0")

    gamma_mask = (
        (gamma_grid >= float(prior_lo))
        & (gamma_grid <= float(prior_hi))
    )
    idx = np.flatnonzero(gamma_mask)
    if len(idx) < 2:
        raise ValueError(
            "prior window contains fewer than two gamma grid points"
        )

    gamma = gamma_grid[idx]
    r = resonance_response[idx]

    a1_grid = np.linspace(
        float(a1_min),
        float(a1_max),
        int(a1_points),
        dtype=np.float64,
    )

    # residual = (g_background - g_observed) + a1 * r_gamma
    d = np.asarray(g_background - g_observed, dtype=np.float64)
    d2 = float(np.dot(d, d))
    dr = r @ d
    rr = np.sum(r * r, axis=1)

    # [N_gamma, N_a1]
    sse = (
        d2
        + 2.0 * dr[:, None] * a1_grid[None, :]
        + rr[:, None] * (a1_grid[None, :] ** 2)
    )
    sse = np.maximum(sse, 0.0)

    rms = float(np.sqrt(np.mean(np.asarray(g_observed) ** 2)))
    sigma = max(float(noise_level) * rms, 1e-300)

    loglike = -0.5 * sse / (sigma * sigma)

    # Marginalize a1 with uniform prior. Constant prior density cancels.
    a1_w = trapz_weights(a1_grid)
    log_a1_w = np.log(a1_w)
    m = np.max(loglike + log_a1_w[None, :], axis=1)
    marginal = np.exp(m) * np.sum(
        np.exp(loglike + log_a1_w[None, :] - m[:, None]),
        axis=1,
    )

    # Uniform density in gamma; grid quadrature weight converts density to mass.
    gamma_w = trapz_weights(gamma)
    mass = marginal * gamma_w
    total = float(np.sum(mass))
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("posterior normalization failed")
    mass /= total

    cdf = np.cumsum(mass)

    def quantile(q):
        j = int(np.searchsorted(cdf, float(q), side="left"))
        j = min(max(j, 0), len(gamma) - 1)
        return float(gamma[j])

    mean = float(np.sum(gamma * mass))
    median = quantile(0.5)
    ci68 = (quantile(0.16), quantile(0.84))
    ci95 = (quantile(0.025), quantile(0.975))

    # Posterior-vs-uniform-prior information gain in nats, using discrete masses.
    prior_mass = gamma_w / np.sum(gamma_w)
    kl = float(
        np.sum(
            mass
            * np.log(
                np.maximum(mass, 1e-300)
                / np.maximum(prior_mass, 1e-300)
            )
        )
    )

    return {
        "gamma": gamma,
        "posterior_mass": mass,
        "mean": mean,
        "median": median,
        "ci68_low": ci68[0],
        "ci68_high": ci68[1],
        "ci95_low": ci95[0],
        "ci95_high": ci95[1],
        "ci95_factor_span": float(ci95[1] / ci95[0]),
        "kl_from_uniform_prior_nats": kl,
        "noise_level": float(noise_level),
        "sigma_absolute": sigma,
    }


def gamma_tag(value: float) -> str:
    return "gamma_" + (
        f"{value:.8g}".replace("-", "m").replace(".", "p")
    )


def plot_profile(
    path: Path,
    gamma_grid: np.ndarray,
    profile: np.ndarray,
    true_gamma: float,
    thresholds: list[float],
):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(gamma_grid, 100.0 * profile)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(true_gamma, linestyle=":", label="true gamma")

    for threshold in thresholds:
        ax.axhline(
            100.0 * threshold,
            linestyle="--",
            linewidth=0.8,
            label=f"{100*threshold:g}%",
        )

    ax.set_xlabel("candidate gamma")
    ax.set_ylabel("best g relative L2 error (%)")
    ax.set_title(
        f"Exp14 profiled gamma compatibility (true gamma={true_gamma:g})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_shrinkage(
    path: Path,
    rows: list[dict],
    true_gamma: float,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    thresholds = sorted(
        set(float(r["threshold"]) for r in rows)
    )
    for threshold in thresholds:
        subset = [
            r for r in rows
            if abs(float(r["threshold"]) - threshold) < 1e-15
        ]
        subset = sorted(
            subset,
            key=lambda r: float(r["prior_factor"]),
        )
        x = [float(r["prior_factor"]) for r in subset]
        y = [float(r["shrinkage_ratio"]) for r in subset]
        ax.plot(
            x,
            y,
            marker="o",
            label=f"{100*threshold:g}% tolerance",
        )

    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("oracle-centered prior factor F")
    ax.set_ylabel("compatible log-width / prior log-width")
    ax.set_title(
        f"How much does g narrow gamma beyond the prior? "
        f"(true gamma={true_gamma:g})"
    )
    ax.axhline(1.0, linestyle=":", linewidth=1.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_posterior(
    path: Path,
    posterior_results: list[tuple[str, dict]],
    true_gamma: float,
    prior_lo: float,
    prior_hi: float,
):
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, result in posterior_results:
        gamma = result["gamma"]
        mass = result["posterior_mass"]
        # Plot mass density per log-x display only for visualization.
        density = mass / np.maximum(trapz_weights(gamma), 1e-300)
        density /= max(float(np.max(density)), 1e-300)
        ax.plot(gamma, density, label=label)

    ax.axvline(true_gamma, linestyle=":", linewidth=1.2, label="truth")
    ax.set_xscale("log")
    ax.set_xlabel("gamma")
    ax.set_ylabel("posterior density (normalized to peak=1)")
    ax.set_title(
        f"Gamma posterior with uniform prior "
        f"[{prior_lo:.4g}, {prior_hi:.4g}]"
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
    prior_factors = parse_float_list(args.prior_factors)
    thresholds = parse_float_list(args.compatibility_thresholds)
    noise_levels = parse_float_list(args.posterior_noise_levels)

    if not (0 < args.a1_min < args.a1_max):
        raise ValueError("require 0 < a1_min < a1_max")
    if not (0 < args.gamma_min < args.gamma_max):
        raise ValueError("require 0 < gamma_min < gamma_max")
    if args.a1_points < 3 or args.gamma_points < 20:
        raise ValueError("increase scan resolution")
    if any(f <= 1.0 for f in prior_factors):
        raise ValueError("all prior factors must be > 1")
    if any(t <= 0 for t in thresholds):
        raise ValueError("compatibility thresholds must be > 0")
    if any(n <= 0 for n in noise_levels):
        raise ValueError("posterior noise levels must be > 0")

    for g in true_gammas:
        if not (args.gamma_min <= g <= args.gamma_max):
            raise ValueError(
                f"true gamma {g} outside physical gamma range"
            )

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

    print("=" * 88)
    print("Exp14 gamma prior / interval / posterior diagnostic")
    print("NO neural-network training")
    print(f"device          : {device}")
    print(f"q^2 range       : [{args.q2_min}, {args.q2_max}]")
    print(f"q points        : {args.input_points}")
    print(f"true gammas     : {true_gammas}")
    print(f"prior factors   : {prior_factors}")
    print(f"thresholds      : {[100*t for t in thresholds]} %")
    print(f"posterior noise : {[100*n for n in noise_levels]} %")
    print("=" * 88)

    shrinkage_rows = []
    posterior_rows = []
    root_summary = {
        "settings": vars(args),
        "interpretation": {
            "shrinkage_ratio": (
                "compatible_log_width / prior_log_width; "
                "1 means the data adds almost no narrowing beyond the prior"
            ),
            "posterior_note": (
                "Posterior uses clean g_true as the observed curve but assumes "
                "the stated RMS-scaled Gaussian noise. This is optimistic."
            ),
        },
        "truths": [],
    }

    # Use the broadest diagnostic prior for the headline posterior plot.
    headline_factor = max(prior_factors)

    for true_gamma in true_gammas:
        subdir = outdir / gamma_tag(true_gamma)
        subdir.mkdir(parents=True, exist_ok=True)

        true_params = parameter_rows(
            args.a1_true,
            true_gamma,
            a2=args.a2,
            a3=args.a3,
            m=args.m,
        )
        g_true = forward_numpy(
            physics,
            true_params,
            batch_size=1,
        )[0]

        best_a1, profile = continuous_best_a1(
            resonance_response,
            g_background,
            g_true,
            args.a1_min,
            args.a1_max,
        )

        plot_profile(
            subdir / "gamma_profile.png",
            gamma_grid,
            profile,
            true_gamma,
            thresholds,
        )

        truth_rows = []

        for factor in prior_factors:
            p_lo, p_hi = prior_bounds(
                true_gamma,
                factor,
                args.gamma_min,
                args.gamma_max,
            )

            for threshold in thresholds:
                comp = compatible_interval(
                    gamma_grid,
                    profile,
                    p_lo,
                    p_hi,
                    threshold,
                )
                row = {
                    "true_gamma": float(true_gamma),
                    "prior_factor": float(factor),
                    "prior_gamma_min": float(p_lo),
                    "prior_gamma_max": float(p_hi),
                    "threshold": float(threshold),
                    "threshold_percent": float(100.0 * threshold),
                    "compatible_gamma_min": comp["gamma_min"],
                    "compatible_gamma_max": comp["gamma_max"],
                    "compatible_factor_span": comp["factor_span"],
                    "prior_log_width": comp["prior_log_width"],
                    "compatible_log_width": comp[
                        "compatible_log_width"
                    ],
                    "shrinkage_ratio": comp["shrinkage_ratio"],
                }
                shrinkage_rows.append(row)
                truth_rows.append(row)

        plot_shrinkage(
            subdir / "prior_shrinkage.png",
            truth_rows,
            true_gamma,
        )

        # Posterior summaries for EVERY prior factor and requested noise level.
        headline_posteriors = []
        for factor in prior_factors:
            p_lo, p_hi = prior_bounds(
                true_gamma,
                factor,
                args.gamma_min,
                args.gamma_max,
            )

            for noise_level in noise_levels:
                post = posterior_gamma(
                    gamma_grid,
                    resonance_response,
                    g_background,
                    g_true,
                    a1_min=args.a1_min,
                    a1_max=args.a1_max,
                    a1_points=args.a1_points,
                    prior_lo=p_lo,
                    prior_hi=p_hi,
                    noise_level=noise_level,
                )

                posterior_rows.append(
                    {
                        "true_gamma": float(true_gamma),
                        "prior_factor": float(factor),
                        "prior_gamma_min": float(p_lo),
                        "prior_gamma_max": float(p_hi),
                        "noise_level": float(noise_level),
                        "noise_percent": float(100.0 * noise_level),
                        "posterior_mean": post["mean"],
                        "posterior_median": post["median"],
                        "ci68_low": post["ci68_low"],
                        "ci68_high": post["ci68_high"],
                        "ci95_low": post["ci95_low"],
                        "ci95_high": post["ci95_high"],
                        "ci95_factor_span": post["ci95_factor_span"],
                        "kl_from_uniform_prior_nats": post[
                            "kl_from_uniform_prior_nats"
                        ],
                    }
                )

                if abs(factor - headline_factor) < 1e-12:
                    headline_posteriors.append(
                        (
                            f"{100*noise_level:g}% noise",
                            post,
                        )
                    )

        headline_lo, headline_hi = prior_bounds(
            true_gamma,
            headline_factor,
            args.gamma_min,
            args.gamma_max,
        )
        plot_posterior(
            subdir / "gamma_posterior_broad_prior.png",
            headline_posteriors,
            true_gamma,
            headline_lo,
            headline_hi,
        )

        # Print the two practical noise thresholds from the broad prior.
        broad_5 = [
            r for r in truth_rows
            if abs(r["prior_factor"] - headline_factor) < 1e-12
            and abs(r["threshold"] - 0.05) < 1e-12
        ]
        broad_9 = [
            r for r in truth_rows
            if abs(r["prior_factor"] - headline_factor) < 1e-12
            and abs(r["threshold"] - 0.09) < 1e-12
        ]

        print("-" * 88)
        print(f"true gamma = {true_gamma:g}")
        if broad_5:
            r = broad_5[0]
            print(
                "5% compatible interval  : "
                f"[{r['compatible_gamma_min']:.6g}, "
                f"{r['compatible_gamma_max']:.6g}] | "
                f"shrinkage={r['shrinkage_ratio']:.4f}"
            )
        if broad_9:
            r = broad_9[0]
            print(
                "9% compatible interval  : "
                f"[{r['compatible_gamma_min']:.6g}, "
                f"{r['compatible_gamma_max']:.6g}] | "
                f"shrinkage={r['shrinkage_ratio']:.4f}"
            )

        truth_summary = {
            "true_gamma": float(true_gamma),
            "profile_min_error": float(np.min(profile)),
            "profile_best_gamma": float(
                gamma_grid[int(np.argmin(profile))]
            ),
            "profile_best_a1": float(
                best_a1[int(np.argmin(profile))]
            ),
        }
        root_summary["truths"].append(truth_summary)

    write_csv(
        outdir / "exp14_prior_shrinkage.csv",
        shrinkage_rows,
    )
    write_csv(
        outdir / "exp14_posterior_summary.csv",
        posterior_rows,
    )

    with (outdir / "exp14_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            root_summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 88)
    print("Exp14 finished")
    print(f"output: {outdir}")
    print("")
    print("FIRST READ:")
    print("  exp14_prior_shrinkage.csv")
    print("  exp14_posterior_summary.csv")
    print("")
    print("For each gamma_* folder:")
    print("  gamma_profile.png")
    print("  prior_shrinkage.png")
    print("  gamma_posterior_broad_prior.png")
    print("")
    print(
        "Interpretation: if shrinkage_ratio is close to 1 at 5%-9% tolerance, "
        "the observable contributes almost no narrowing beyond the prior."
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
