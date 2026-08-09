#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp13: q^2 observation-design study for gamma identifiability.

This experiment does NOT train a neural network.

It answers two questions before changing the training pipeline:

1) Can changing the q^2 observation interval make g(q^2) more sensitive to gamma?
2) With the same number of q^2 points, can non-uniform sampling improve
   a1-gamma distinguishability?

The metric is the same profiled physics metric used in Exp12:

    E_profile(gamma) =
        min_a1 ||g(a1, gamma) - g_true||_2 / ||g_true||_2

For each true gamma, the script also finds a deliberately far alternative
(gamma differs by at least a factor of 2) and reports how different its g is
after optimally compensating with a1.

Higher "far alternative g error" is BETTER:
it means a wrong gamma can no longer hide behind a small change in a1.

Important:
- No neural network is involved.
- The project's existing OnlinePhysics forward implementation is reused.
- q^2 ranges tested by this script are numerical candidates. Before using a
  newly found range in a physical training experiment, confirm with the
  physics setup that the range is actually observable/allowed.
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
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("empty float list")
    return values


def parse_ranges(text: str) -> list[tuple[float, float]]:
    ranges = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        pieces = [x.strip() for x in item.split(",")]
        if len(pieces) != 2:
            raise ValueError(
                f"Invalid range '{item}'. Use qmin,qmax;qmin,qmax;..."
            )
        qmin = float(pieces[0])
        qmax = float(pieces[1])
        if not qmin < qmax:
            raise ValueError(f"Require qmin < qmax, got {item}")
        if qmax >= 0:
            raise ValueError(
                f"This diagnostic keeps q^2 negative to avoid crossing the "
                f"spectral integration support; got qmax={qmax}"
            )
        ranges.append((qmin, qmax))
    if not ranges:
        raise ValueError("No q^2 ranges were provided")
    return ranges


def parse_args():
    p = argparse.ArgumentParser(
        description="Exp13 pure-physics q^2 observation-design scan"
    )

    p.add_argument(
        "--output-dir",
        default=r".\validation_results\exp13_q_observation_design",
    )

    # Physics truth used for the design diagnostic.
    p.add_argument("--a1-true", type=float, default=0.10)
    p.add_argument("--a2", type=float, default=0.025)
    p.add_argument("--a3", type=float, default=0.0)
    p.add_argument("--m", type=float, default=0.8)
    p.add_argument(
        "--true-gammas",
        default="0.01,0.05,0.2,0.5",
    )

    # a1/gamma scan used to measure degeneracy.
    p.add_argument("--a1-min", type=float, default=0.05)
    p.add_argument("--a1-max", type=float, default=0.20)
    p.add_argument("--gamma-min", type=float, default=0.001)
    p.add_argument("--gamma-max", type=float, default=1.0)
    p.add_argument("--gamma-points", type=int, default=361)
    p.add_argument("--far-gamma-factor", type=float, default=2.0)

    # q^2 observation design.
    p.add_argument(
        "--ranges",
        default=(
            "-200,-6;"
            "-100,-6;"
            "-50,-6;"
            "-20,-6;"
            "-10,-6;"
            "-20,-1;"
            "-10,-1;"
            "-6,-0.5"
        ),
        help=(
            "Semicolon-separated q^2 intervals. Example: "
            "'-100,-6;-20,-1;-6,-0.5'"
        ),
    )
    p.add_argument(
        "--baseline-range",
        default="-100,-6",
        help="Reference interval used for improvement ratios.",
    )
    p.add_argument("--input-points", type=int, default=100)
    p.add_argument(
        "--sampling-strategies",
        default="uniform,qmax_power2,qmax_power4,chebyshev,sensitivity",
    )
    p.add_argument(
        "--sensitivity-candidate-points",
        type=int,
        default=2001,
    )
    p.add_argument(
        "--sensitivity-fd-rel",
        type=float,
        default=0.02,
        help="Relative finite-difference step for gamma sensitivity.",
    )

    # Same forward settings as Exp12.
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


def make_physics(
    args,
    device: torch.device,
    q_points: np.ndarray,
) -> OnlinePhysics:
    q_points = np.asarray(q_points, dtype=np.float64)
    if q_points.ndim != 1 or len(q_points) < 2:
        raise ValueError("q_points must be a 1-D array with >= 2 values")
    if not np.all(np.diff(q_points) > 0):
        raise ValueError("q_points must be strictly increasing")
    if np.max(q_points) >= 0:
        raise ValueError("q_points must stay negative in Exp13")

    physics_args = SimpleNamespace(
        physics_dtype=args.physics_dtype,
        noise_level=0.0,
        data_scale=args.data_scale,
        shift=args.shift,
        s_min=args.s_min,
        s_max=args.s_max,
        q2_min=float(q_points[0]),
        q2_max=float(q_points[-1]),
        output_points=args.output_points,
        input_points=len(q_points),
        integration_points=args.integration_points,
        resonance_grid_points=max(args.output_points, 1000),
    )
    physics = OnlinePhysics(physics_args, device)

    # OnlinePhysics currently constructs a uniform linspace.  For this
    # diagnostic we replace only q2_grid; the forward integral itself remains
    # exactly the project's implementation.
    physics.q2_grid = torch.as_tensor(
        q_points,
        dtype=physics.dtype,
        device=device,
    )
    physics.input_points = int(len(q_points))
    physics.q2_min = float(q_points[0])
    physics.q2_max = float(q_points[-1])
    return physics


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


def relative_l2_rows(candidate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64).reshape(1, -1)
    denom = np.linalg.norm(truth[0])
    return np.linalg.norm(candidate - truth, axis=1) / max(denom, 1e-300)


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


def threshold_interval(
    gamma_grid: np.ndarray,
    profile: np.ndarray,
    threshold: float,
) -> dict:
    mask = profile <= float(threshold)
    if not np.any(mask):
        return {
            "threshold": float(threshold),
            "gamma_min": None,
            "gamma_max": None,
            "factor_span": None,
        }
    values = gamma_grid[mask]
    return {
        "threshold": float(threshold),
        "gamma_min": float(values.min()),
        "gamma_max": float(values.max()),
        "factor_span": float(values.max() / values.min()),
    }


def make_gamma_grid(args, true_gammas):
    grid = np.logspace(
        np.log10(args.gamma_min),
        np.log10(args.gamma_max),
        args.gamma_points,
        dtype=np.float64,
    )
    grid = np.unique(
        np.concatenate([grid, np.asarray(true_gammas, dtype=np.float64)])
    )
    grid.sort()
    return grid


def make_uniform_q(qmin: float, qmax: float, n: int) -> np.ndarray:
    return np.linspace(qmin, qmax, int(n), dtype=np.float64)


def make_q_strategy(
    strategy: str,
    qmin: float,
    qmax: float,
    n: int,
) -> np.ndarray:
    strategy = strategy.strip().lower()
    n = int(n)
    u = np.linspace(0.0, 1.0, n, dtype=np.float64)
    span = qmax - qmin

    if strategy == "uniform":
        return qmin + span * u

    if strategy == "qmax_power2":
        # Concentrate points near qmax (the end closest to zero).
        return qmin + span * (1.0 - (1.0 - u) ** 2)

    if strategy == "qmax_power4":
        return qmin + span * (1.0 - (1.0 - u) ** 4)

    if strategy == "chebyshev":
        k = np.arange(n, dtype=np.float64)
        x = np.cos((2.0 * k + 1.0) * math.pi / (2.0 * n))
        q = 0.5 * (qmin + qmax) + 0.5 * span * x
        q.sort()
        return q

    raise ValueError(f"Unknown non-sensitivity strategy: {strategy}")


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


def evaluate_design(
    label: str,
    q_points: np.ndarray,
    args,
    device,
    gamma_grid: np.ndarray,
    true_gammas: list[float],
) -> dict:
    physics = make_physics(args, device, q_points)
    g_background, resonance_response = build_responses(
        physics,
        args,
        gamma_grid,
    )

    per_truth = []
    far_errors = []

    for true_gamma in true_gammas:
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

        far_mask = (
            (gamma_grid <= true_gamma / args.far_gamma_factor)
            | (gamma_grid >= true_gamma * args.far_gamma_factor)
        )
        far_ids = np.flatnonzero(far_mask)
        if len(far_ids) == 0:
            raise RuntimeError("No far-gamma candidates exist in the scan")
        alt_id = far_ids[np.argmin(profile[far_ids])]

        alt_gamma = float(gamma_grid[alt_id])
        alt_a1 = float(best_a1[alt_id])
        alt_error = float(profile[alt_id])

        intervals = {
            f"{100*t:g}%": threshold_interval(
                gamma_grid,
                profile,
                t,
            )
            for t in THRESHOLDS
        }

        per_truth.append(
            {
                "true_gamma": float(true_gamma),
                "far_alt_gamma": alt_gamma,
                "far_alt_a1": alt_a1,
                "far_alt_ratio": float(alt_gamma / true_gamma),
                "far_alt_g_relative_l2": alt_error,
                "far_alt_g_error_percent": float(100.0 * alt_error),
                "threshold_intervals": intervals,
            }
        )
        far_errors.append(100.0 * alt_error)

    far_errors = np.asarray(far_errors, dtype=np.float64)
    robust_min = float(np.min(far_errors))
    geometric = float(
        np.exp(np.mean(np.log(np.maximum(far_errors, 1e-300))))
    )
    median = float(np.median(far_errors))

    return {
        "label": label,
        "qmin": float(q_points[0]),
        "qmax": float(q_points[-1]),
        "n_points": int(len(q_points)),
        "robust_min_far_error_percent": robust_min,
        "geometric_mean_far_error_percent": geometric,
        "median_far_error_percent": median,
        "per_truth": per_truth,
    }


def gamma_derivative(
    q_points: np.ndarray,
    true_gamma: float,
    args,
    device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    physics = make_physics(args, device, q_points)

    # a1 derivative is exact to numerical precision because the current
    # forward model is linear in a1.
    da = min(1e-3, 0.1 * (args.a1_max - args.a1_min))
    p_plus = parameter_rows(
        args.a1_true + da,
        true_gamma,
        a2=args.a2,
        a3=args.a3,
        m=args.m,
    )
    p_minus = parameter_rows(
        args.a1_true - da,
        true_gamma,
        a2=args.a2,
        a3=args.a3,
        m=args.m,
    )
    g_ap = forward_numpy(physics, p_plus, 1)[0]
    g_am = forward_numpy(physics, p_minus, 1)[0]
    j_a1 = (g_ap - g_am) / (2.0 * da)

    rel = float(args.sensitivity_fd_rel)
    dg = max(true_gamma * rel, 1e-5)
    low = max(args.gamma_min, true_gamma - dg)
    high = min(args.gamma_max, true_gamma + dg)
    if not low < true_gamma < high:
        # one-sided fallback near scan boundary
        low = true_gamma
        high = min(args.gamma_max, true_gamma + max(dg, 1e-5))

    p_low = parameter_rows(
        args.a1_true,
        low,
        a2=args.a2,
        a3=args.a3,
        m=args.m,
    )
    p_high = parameter_rows(
        args.a1_true,
        high,
        a2=args.a2,
        a3=args.a3,
        m=args.m,
    )
    g_low = forward_numpy(physics, p_low, 1)[0]
    g_high = forward_numpy(physics, p_high, 1)[0]
    j_gamma = (g_high - g_low) / max(high - low, 1e-12)

    p_true = parameter_rows(
        args.a1_true,
        true_gamma,
        a2=args.a2,
        a3=args.a3,
        m=args.m,
    )
    g_true = forward_numpy(physics, p_true, 1)[0]
    return g_true, j_a1, j_gamma


def make_sensitivity_q(
    qmin: float,
    qmax: float,
    n: int,
    true_gammas: list[float],
    args,
    device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a non-uniform q grid targeted to gamma information that cannot be
    explained by changing a1.

    On a dense candidate grid, for each true gamma we compute normalized
    Jacobian columns d g / d a1 and d g / d gamma.  We remove the component of
    d g / d gamma parallel to d g / d a1:

        j_gamma_perp =
            j_gamma - proj_{j_a1}(j_gamma)

    The pointwise RMS magnitude of this residual over several true gammas is
    used as a sampling density.  Equal-mass quantiles of that density give the
    final q points.

    The subsequent Exp12-style global profile scan validates whether this local
    sensitivity-based design actually improves identifiability.
    """
    dense_q = np.linspace(
        qmin,
        qmax,
        int(args.sensitivity_candidate_points),
        dtype=np.float64,
    )

    score_sq = np.zeros_like(dense_q)

    delta_a1 = args.a1_max - args.a1_min
    delta_gamma = args.gamma_max - args.gamma_min

    for true_gamma in true_gammas:
        g_true, j_a1, j_gamma = gamma_derivative(
            dense_q,
            true_gamma,
            args,
            device,
        )
        scale = max(
            float(np.sqrt(np.mean(g_true ** 2))),
            1e-300,
        )

        ja = j_a1 * delta_a1 / scale
        jg = j_gamma * delta_gamma / scale

        coeff = float(
            np.dot(jg, ja) / max(np.dot(ja, ja), 1e-300)
        )
        residual = jg - coeff * ja
        score_sq += residual ** 2

    score = np.sqrt(score_sq / max(len(true_gammas), 1))

    # Add a tiny floor so the design does not collapse onto only one q region.
    floor = max(float(np.max(score)) * 1e-4, 1e-15)
    density = score + floor
    cumulative = np.cumsum(density)
    cumulative /= cumulative[-1]

    targets = (np.arange(int(n), dtype=np.float64) + 0.5) / int(n)
    indices = np.searchsorted(cumulative, targets, side="left")
    indices = np.clip(indices, 0, len(dense_q) - 1)

    # Guarantee uniqueness. If quantiles collide in a very sharp region, fill
    # remaining points using the highest-scoring unused candidate locations.
    unique = list(dict.fromkeys(int(i) for i in indices))
    if len(unique) < int(n):
        ranked = np.argsort(score)[::-1]
        used = set(unique)
        for idx in ranked:
            idx = int(idx)
            if idx not in used:
                unique.append(idx)
                used.add(idx)
                if len(unique) == int(n):
                    break

    unique = np.asarray(sorted(unique[: int(n)]), dtype=np.int64)
    q_points = dense_q[unique]
    return q_points, dense_q, score


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_result_rows(results: list[dict], stage: str) -> list[dict]:
    rows = []
    for result in results:
        for item in result["per_truth"]:
            row = {
                "stage": stage,
                "design": result["label"],
                "qmin": result["qmin"],
                "qmax": result["qmax"],
                "n_points": result["n_points"],
                "robust_min_far_error_percent": result[
                    "robust_min_far_error_percent"
                ],
                "geometric_mean_far_error_percent": result[
                    "geometric_mean_far_error_percent"
                ],
                "median_far_error_percent": result[
                    "median_far_error_percent"
                ],
                "true_gamma": item["true_gamma"],
                "far_alt_gamma": item["far_alt_gamma"],
                "far_alt_a1": item["far_alt_a1"],
                "far_alt_g_error_percent": item[
                    "far_alt_g_error_percent"
                ],
            }
            for threshold in THRESHOLDS:
                key = f"{100*threshold:g}%"
                interval = item["threshold_intervals"][key]
                tag = f"{100*threshold:g}pct"
                row[f"{tag}_gamma_min"] = interval["gamma_min"]
                row[f"{tag}_gamma_max"] = interval["gamma_max"]
                row[f"{tag}_factor_span"] = interval["factor_span"]
            rows.append(row)
    return rows


def plot_design_scores(
    results: list[dict],
    path: Path,
    title: str,
):
    labels = [r["label"] for r in results]
    values = [r["robust_min_far_error_percent"] for r in results]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("worst-case factor-2 wrong-gamma g error (%)")
    ax.set_title(title)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sampling_points(
    q_designs: dict[str, np.ndarray],
    path: Path,
):
    fig, ax = plt.subplots(figsize=(11, 6))
    for row, (name, q) in enumerate(q_designs.items()):
        y = np.full_like(q, row, dtype=np.float64)
        ax.scatter(q, y, s=8, label=name)
    ax.set_yticks(np.arange(len(q_designs)))
    ax.set_yticklabels(list(q_designs.keys()))
    ax.set_xlabel("q^2")
    ax.set_title("100-point q^2 sampling designs")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()

    true_gammas = parse_float_list(args.true_gammas)
    ranges = parse_ranges(args.ranges)
    baseline = parse_ranges(args.baseline_range)[0]
    strategies = [
        x.strip().lower()
        for x in args.sampling_strategies.split(",")
        if x.strip()
    ]

    if args.input_points < 4:
        raise ValueError("--input-points must be >= 4")
    if args.far_gamma_factor <= 1:
        raise ValueError("--far-gamma-factor must be > 1")
    if not (args.a1_min <= args.a1_true <= args.a1_max):
        raise ValueError("a1_true must lie inside the a1 scan range")
    for g in true_gammas:
        if not (args.gamma_min <= g <= args.gamma_max):
            raise ValueError(
                f"true gamma {g} lies outside gamma scan range"
            )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    gamma_grid = make_gamma_grid(args, true_gammas)

    print("=" * 90)
    print("Exp13 q^2 observation-design study")
    print("NO neural-network training")
    print(f"device        : {device}")
    print(f"true gammas   : {true_gammas}")
    print(f"q points      : {args.input_points}")
    print(f"range tests   : {ranges}")
    print("=" * 90)

    # ------------------------------------------------------------------
    # Stage A: q^2 interval scan with the SAME 100-point uniform sampling.
    # ------------------------------------------------------------------
    range_results = []
    baseline_result = None

    for qmin, qmax in ranges:
        label = f"[{qmin:g},{qmax:g}] uniform"
        q_points = make_uniform_q(qmin, qmax, args.input_points)
        result = evaluate_design(
            label,
            q_points,
            args,
            device,
            gamma_grid,
            true_gammas,
        )
        range_results.append(result)

        if (
            abs(qmin - baseline[0]) < 1e-12
            and abs(qmax - baseline[1]) < 1e-12
        ):
            baseline_result = result

        print(
            f"RANGE {label:24s} | "
            f"worst far-separation = "
            f"{result['robust_min_far_error_percent']:.6g}% | "
            f"geom = "
            f"{result['geometric_mean_far_error_percent']:.6g}%"
        )

    if baseline_result is None:
        q_points = make_uniform_q(
            baseline[0],
            baseline[1],
            args.input_points,
        )
        baseline_result = evaluate_design(
            f"[{baseline[0]:g},{baseline[1]:g}] uniform (baseline)",
            q_points,
            args,
            device,
            gamma_grid,
            true_gammas,
        )
        range_results.append(baseline_result)

    # Higher worst-case far-gamma separation is better.
    best_range_result = max(
        range_results,
        key=lambda r: r["robust_min_far_error_percent"],
    )
    best_range = (
        float(best_range_result["qmin"]),
        float(best_range_result["qmax"]),
    )

    baseline_score = max(
        baseline_result["robust_min_far_error_percent"],
        1e-300,
    )

    for result in range_results:
        result["improvement_vs_baseline"] = float(
            result["robust_min_far_error_percent"] / baseline_score
        )

    plot_design_scores(
        range_results,
        outdir / "range_scan_worst_case_separation.png",
        "q^2 interval scan: higher is better",
    )

    # ------------------------------------------------------------------
    # Stage B: same number of q points, different sampling inside best range.
    # ------------------------------------------------------------------
    print("-" * 90)
    print(
        f"Best interval from Stage A: "
        f"[{best_range[0]:g}, {best_range[1]:g}]"
    )

    q_designs = {}
    sensitivity_dense_q = None
    sensitivity_score = None

    for strategy in strategies:
        if strategy == "sensitivity":
            q, dense_q, score = make_sensitivity_q(
                best_range[0],
                best_range[1],
                args.input_points,
                true_gammas,
                args,
                device,
            )
            q_designs[strategy] = q
            sensitivity_dense_q = dense_q
            sensitivity_score = score
        else:
            q_designs[strategy] = make_q_strategy(
                strategy,
                best_range[0],
                best_range[1],
                args.input_points,
            )

    sampling_results = []
    for strategy, q_points in q_designs.items():
        result = evaluate_design(
            strategy,
            q_points,
            args,
            device,
            gamma_grid,
            true_gammas,
        )
        sampling_results.append(result)
        print(
            f"SAMPLE {strategy:16s} | "
            f"worst far-separation = "
            f"{result['robust_min_far_error_percent']:.6g}% | "
            f"geom = "
            f"{result['geometric_mean_far_error_percent']:.6g}%"
        )

    best_sampling = max(
        sampling_results,
        key=lambda r: r["robust_min_far_error_percent"],
    )

    # Improvement relative to uniform sampling in the selected interval.
    uniform_best_range = next(
        (r for r in sampling_results if r["label"] == "uniform"),
        None,
    )
    uniform_score = (
        uniform_best_range["robust_min_far_error_percent"]
        if uniform_best_range is not None
        else best_range_result["robust_min_far_error_percent"]
    )
    uniform_score = max(uniform_score, 1e-300)

    for result in sampling_results:
        result["improvement_vs_uniform_same_range"] = float(
            result["robust_min_far_error_percent"] / uniform_score
        )

    plot_design_scores(
        sampling_results,
        outdir / "sampling_scan_worst_case_separation.png",
        (
            f"Sampling design inside "
            f"[{best_range[0]:g}, {best_range[1]:g}]: higher is better"
        ),
    )
    plot_sampling_points(
        q_designs,
        outdir / "q_sampling_designs.png",
    )

    if sensitivity_dense_q is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sensitivity_dense_q, sensitivity_score)
        ax.scatter(
            q_designs["sensitivity"],
            np.interp(
                q_designs["sensitivity"],
                sensitivity_dense_q,
                sensitivity_score,
            ),
            s=12,
        )
        ax.set_xlabel("q^2")
        ax.set_ylabel("gamma sensitivity not explained by a1")
        ax.set_title(
            "Physics-based gamma sensitivity score "
            "(a1 component projected out)"
        )
        fig.tight_layout()
        fig.savefig(
            outdir / "gamma_sensitivity_score.png",
            dpi=180,
        )
        plt.close(fig)

    # Save chosen q points for later training only after the physics result is
    # accepted.
    with (outdir / "q_sampling_points.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["strategy"] + [f"q{i:03d}" for i in range(args.input_points)]
        )
        for name, q in q_designs.items():
            writer.writerow([name] + [f"{x:.12g}" for x in q])

    # Full detail CSV.
    all_rows = (
        flatten_result_rows(range_results, "range_scan")
        + flatten_result_rows(sampling_results, "sampling_scan")
    )
    fieldnames = list(all_rows[0].keys())
    write_csv(
        outdir / "exp13_detailed_results.csv",
        all_rows,
        fieldnames,
    )

    # Compact ranking CSV.
    ranking_rows = []
    for stage, results in (
        ("range_scan", range_results),
        ("sampling_scan", sampling_results),
    ):
        ordered = sorted(
            results,
            key=lambda r: r["robust_min_far_error_percent"],
            reverse=True,
        )
        for rank, r in enumerate(ordered, start=1):
            ranking_rows.append(
                {
                    "stage": stage,
                    "rank": rank,
                    "design": r["label"],
                    "qmin": r["qmin"],
                    "qmax": r["qmax"],
                    "n_points": r["n_points"],
                    "worst_case_far_error_percent": r[
                        "robust_min_far_error_percent"
                    ],
                    "geometric_mean_far_error_percent": r[
                        "geometric_mean_far_error_percent"
                    ],
                    "median_far_error_percent": r[
                        "median_far_error_percent"
                    ],
                    "improvement_ratio": r.get(
                        "improvement_vs_baseline",
                        r.get(
                            "improvement_vs_uniform_same_range",
                            1.0,
                        ),
                    ),
                }
            )
    write_csv(
        outdir / "exp13_ranking.csv",
        ranking_rows,
        list(ranking_rows[0].keys()),
    )

    summary = {
        "interpretation": {
            "metric": (
                "factor-2 wrong-gamma g error after optimally profiling out a1"
            ),
            "direction": "higher is better",
            "note": (
                "A new q^2 range must be checked for physical observability "
                "before it is used for training."
            ),
        },
        "baseline": {
            "range": [baseline[0], baseline[1]],
            "uniform_points": args.input_points,
            "worst_case_far_error_percent": baseline_result[
                "robust_min_far_error_percent"
            ],
        },
        "best_range": {
            "qmin": best_range[0],
            "qmax": best_range[1],
            "label": best_range_result["label"],
            "worst_case_far_error_percent": best_range_result[
                "robust_min_far_error_percent"
            ],
            "improvement_vs_baseline": best_range_result[
                "improvement_vs_baseline"
            ],
        },
        "best_sampling_inside_best_range": {
            "strategy": best_sampling["label"],
            "worst_case_far_error_percent": best_sampling[
                "robust_min_far_error_percent"
            ],
            "improvement_vs_uniform_same_range": best_sampling[
                "improvement_vs_uniform_same_range"
            ],
        },
        "range_results": range_results,
        "sampling_results": sampling_results,
    }
    with (outdir / "exp13_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 90)
    print("Exp13 finished")
    print(f"Output: {outdir}")
    print("")
    print("FIRST READ:")
    print("  exp13_ranking.csv")
    print("")
    print("Then inspect:")
    print("  range_scan_worst_case_separation.png")
    print("  sampling_scan_worst_case_separation.png")
    print("  q_sampling_designs.png")
    print("  gamma_sensitivity_score.png")
    print("")
    print(
        "IMPORTANT: do not retrain the Transformer yet. "
        "First decide whether any physically allowed q^2 design actually "
        "raises gamma separation enough to matter relative to 5%-9% noise."
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
