#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp30 step 1: build a continuous candidate pool with an explicit remote-alias score.

Why this exists
---------------
Earlier g-separation experiments guaranteed only that the SELECTED physical
truths were separated from one another.  A continuous regressor can output an
unselected (a1, gamma), so a physically remote off-bank solution can still
produce almost the same g(q^2).

For the current a1+gamma problem, with a2/a3/m fixed, the forward observable is
linear in a1:

    g(a1, gamma) = B + a1 * R(gamma).

That lets us audit remote aliases much more densely than a brute-force random
probe pool.  We construct a dense log-gamma basis and, for every candidate
truth, analytically profile the best a1 at each gamma.  A remote alias is any
point satisfying either

    |delta a1| >= remote_a1_abs
or
    gamma factor >= remote_gamma_factor.

The candidate's ``remote_alias_rms_snr`` is the minimum observed-g RMS-SNR over
all such remote points on the dense gamma grid.  Larger is safer / more
identifiable.

This does NOT modify g and does NOT change the forward physics.  It only scores
which legitimate physical states are locally unique against the continuous
parameter domain.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from data_generate_exp20_pairwise import (
    DEFAULT_PHYSICS,
    FIXED_A2,
    FIXED_A3,
    FIXED_M,
    forward_bank,
)
from data_generate_exp22c_fixed_capacity import canonical_indices


def parse_args():
    p = argparse.ArgumentParser(
        description="Build Exp30 continuous alias-scored candidate pool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default="data_exp30_alias_pool")
    p.add_argument("--candidate-count", type=int, default=20000)
    p.add_argument("--alias-gamma-points", type=int, default=1601)
    p.add_argument("--reference-noise", type=float, default=0.002)
    p.add_argument("--a1-min", type=float, default=0.05)
    p.add_argument("--a1-max", type=float, default=0.20)
    p.add_argument("--gamma-min", type=float, default=0.01)
    p.add_argument("--gamma-max", type=float, default=1.0)
    p.add_argument("--remote-a1-abs", type=float, default=0.03)
    p.add_argument("--remote-gamma-factor", type=float, default=2.0)
    p.add_argument("--observation-points", type=int, default=500)
    p.add_argument("--model-input-points", type=int, default=1000)
    p.add_argument("--integration-points", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260830)
    p.add_argument("--profile-chunk", type=int, default=256)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def latin_hypercube_1d(n, rng):
    x = (np.arange(int(n), dtype=np.float64) + rng.random(int(n))) / float(n)
    rng.shuffle(x)
    return x


def build_basis(args, physics):
    """Return dense gamma grid plus g = B + a1*R(gamma) basis."""
    gamma_grid = np.geomspace(
        float(args.gamma_min),
        float(args.gamma_max),
        int(args.alias_gamma_points),
        dtype=np.float64,
    )

    p0 = np.zeros((1, 5), dtype=np.float64)
    p0[0] = [0.0, FIXED_A2, FIXED_A3, FIXED_M, gamma_grid[0]]
    _, g0 = forward_bank(
        p0, physics=physics, integration_points=int(args.integration_points)
    )
    B = np.asarray(g0[0], dtype=np.float64)

    # Stay inside the physical a1 range when constructing the response basis.
    a_ref = float(args.a1_max)
    p = np.zeros((len(gamma_grid), 5), dtype=np.float64)
    p[:, 0] = a_ref
    p[:, 1] = FIXED_A2
    p[:, 2] = FIXED_A3
    p[:, 3] = FIXED_M
    p[:, 4] = gamma_grid
    _, g_ref = forward_bank(
        p, physics=physics, integration_points=int(args.integration_points)
    )
    R = (np.asarray(g_ref, dtype=np.float64) - B[None, :]) / a_ref
    return gamma_grid, B, R


def candidate_parameters(args, gamma_grid):
    n = int(args.candidate_count)
    if n < 500:
        raise ValueError("--candidate-count should be >= 500")
    rng = np.random.default_rng(int(args.seed))
    u = latin_hypercube_1d(n, rng)
    v = latin_hypercube_1d(n, rng)

    a1 = float(args.a1_min) + (float(args.a1_max) - float(args.a1_min)) * u
    # Snap truth gamma to a very dense grid so the continuous alias profile can
    # use an exact response Gram matrix. With 1601 log points this is far denser
    # than any training anchor grid used earlier.
    gi = np.minimum(
        (v * len(gamma_grid)).astype(np.int64),
        len(gamma_grid) - 1,
    )
    gamma = gamma_grid[gi]

    params = np.zeros((n, 5), dtype=np.float64)
    params[:, 0] = a1
    params[:, 1] = FIXED_A2
    params[:, 2] = FIXED_A3
    params[:, 3] = FIXED_M
    params[:, 4] = gamma
    return params, gi.astype(np.int32)


def basis_linearity_check(args, physics, gamma_grid, B, R):
    rng = np.random.default_rng(int(args.seed) + 99)
    k = min(24, len(gamma_grid))
    gi = rng.integers(0, len(gamma_grid), size=k)
    a1 = rng.uniform(float(args.a1_min), float(args.a1_max), size=k)
    p = np.zeros((k, 5), dtype=np.float64)
    p[:, 0] = a1
    p[:, 1] = FIXED_A2
    p[:, 2] = FIXED_A3
    p[:, 3] = FIXED_M
    p[:, 4] = gamma_grid[gi]
    _, direct = forward_bank(
        p, physics=physics, integration_points=int(args.integration_points)
    )
    recon = B[None, :] + a1[:, None] * R[gi]
    direct = np.asarray(direct, dtype=np.float64)
    rel = np.linalg.norm(direct - recon, axis=1) / np.maximum(
        np.linalg.norm(direct, axis=1), 1e-30
    )
    return {
        "max_relative_l2": float(np.max(rel)),
        "median_relative_l2": float(np.median(rel)),
        "sample_count": int(k),
    }


def profiled_remote_alias_scores(
    *,
    params,
    gamma_index,
    gamma_grid,
    B_master,
    R_master,
    obs_idx,
    reference_noise,
    a1_min,
    a1_max,
    remote_a1_abs,
    remote_gamma_factor,
    chunk_size,
):
    """Profile the best physically remote a1 at every dense gamma grid point."""
    R_obs = np.asarray(R_master[:, obs_idx], dtype=np.float64)
    q = float(R_obs.shape[1])

    # C[i,j] = mean_q R(gamma_i) R(gamma_j)
    C = (R_obs @ R_obs.T) / q
    diag = np.maximum(np.diag(C), 1e-30)

    Bm = np.asarray(B_master, dtype=np.float64)
    Rm = np.asarray(R_master, dtype=np.float64)
    B2m = float(np.mean(Bm * Bm))
    BRm = np.mean(Rm * Bm[None, :], axis=1)
    R2m = np.mean(Rm * Rm, axis=1)

    n = len(params)
    score = np.empty(n, dtype=np.float64)
    best_a1 = np.empty(n, dtype=np.float64)
    best_gamma = np.empty(n, dtype=np.float64)
    best_gamma_factor = np.empty(n, dtype=np.float64)

    ggrid = np.asarray(gamma_grid, dtype=np.float64)
    logg = np.log(ggrid)

    for start in range(0, n, int(chunk_size)):
        stop = min(start + int(chunk_size), n)
        a_true = params[start:stop, 0].astype(np.float64)
        g_true = params[start:stop, 4].astype(np.float64)
        ti = gamma_index[start:stop].astype(np.int64)

        Crows = C[ti]  # [B, G]
        dot = a_true[:, None] * Crows
        v2 = (a_true * a_true)[:, None] * diag[ti, None]
        a0 = np.clip(dot / diag[None, :], float(a1_min), float(a1_max))

        # Gamma-remote points are always allowed. Otherwise a1 itself must be remote.
        gf = np.maximum(
            ggrid[None, :] / g_true[:, None],
            g_true[:, None] / ggrid[None, :],
        )
        remote_by_gamma = gf >= float(remote_gamma_factor)

        # Best point in the remote lower-a1 interval.
        lower_hi = np.minimum(float(a1_max), a_true - float(remote_a1_abs))
        lower_valid = lower_hi >= float(a1_min)
        a_lower = np.clip(
            a0,
            float(a1_min),
            np.maximum(lower_hi[:, None], float(a1_min)),
        )
        r_lower = v2 - 2.0 * a_lower * dot + (a_lower * a_lower) * diag[None, :]
        r_lower[~lower_valid, :] = np.inf

        # Best point in the remote upper-a1 interval.
        upper_lo = np.maximum(float(a1_min), a_true + float(remote_a1_abs))
        upper_valid = upper_lo <= float(a1_max)
        a_upper = np.clip(
            a0,
            np.minimum(upper_lo[:, None], float(a1_max)),
            float(a1_max),
        )
        r_upper = v2 - 2.0 * a_upper * dot + (a_upper * a_upper) * diag[None, :]
        r_upper[~upper_valid, :] = np.inf

        use_upper = r_upper < r_lower
        a_remote = np.where(use_upper, a_upper, a_lower)

        # If gamma itself is remote, unconstrained a0 is the true profile optimum.
        a_best = np.where(remote_by_gamma, a0, a_remote)

        res2 = np.maximum(
            v2 - 2.0 * a_best * dot + (a_best * a_best) * diag[None, :],
            0.0,
        )
        rms_diff = np.sqrt(res2)

        truth_rms2 = (
            B2m
            + 2.0 * a_true[:, None] * BRm[ti, None]
            + (a_true * a_true)[:, None] * R2m[ti, None]
        )
        truth_rms = np.sqrt(np.maximum(truth_rms2[:, 0], 1e-30))

        alias_rms2 = (
            B2m
            + 2.0 * a_best * BRm[None, :]
            + (a_best * a_best) * R2m[None, :]
        )
        alias_rms = np.sqrt(np.maximum(alias_rms2, 1e-30))

        snr = rms_diff / np.maximum(
            float(reference_noise) * 0.5 * (truth_rms[:, None] + alias_rms),
            1e-30,
        )

        j = np.argmin(snr, axis=1)
        rr = np.arange(stop - start)
        score[start:stop] = snr[rr, j]
        best_a1[start:stop] = a_best[rr, j]
        best_gamma[start:stop] = ggrid[j]
        best_gamma_factor[start:stop] = gf[rr, j]

    return score, best_a1, best_gamma, best_gamma_factor


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_alias_distribution(output_dir, scores):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=60)
    for x in (0.25, 0.5, 0.75, 1.0):
        ax.axvline(x, linestyle="--", label="alias min %.2f" % x)
    ax.set_xlabel("continuous remote-alias minimum RMS-SNR")
    ax.set_ylabel("candidate count")
    ax.set_title("Exp30 continuous identifiability score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "alias_score_distribution.png", dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    if not (0 < args.a1_min < args.a1_max):
        raise ValueError("require 0 < a1_min < a1_max")
    if not (0 < args.gamma_min < args.gamma_max):
        raise ValueError("require 0 < gamma_min < gamma_max")
    if args.remote_a1_abs <= 0 or args.remote_gamma_factor <= 1:
        raise ValueError("remote criteria must be positive / >1")
    if args.reference_noise <= 0:
        raise ValueError("reference noise must be positive")
    if args.alias_gamma_points < 201:
        raise ValueError("use at least 201 alias gamma points")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError("%s is not empty; use --overwrite" % output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    physics = replace(DEFAULT_PHYSICS, q2_points=int(args.model_input_points))
    obs_idx = canonical_indices(
        int(args.model_input_points), int(args.observation_points)
    )

    print("=" * 100)
    print("Exp30 continuous alias-scored candidate pool")
    print("candidate count       : %d" % args.candidate_count)
    print("dense gamma grid      : %d" % args.alias_gamma_points)
    print("remote criterion      : |da1| >= %.4g OR gamma factor >= %.4g"
          % (args.remote_a1_abs, args.remote_gamma_factor))
    print("reference noise       : %.4g (%.3f%%)"
          % (args.reference_noise, 100 * args.reference_noise))
    print("=" * 100)

    gamma_grid, B, R = build_basis(args, physics)
    linearity = basis_linearity_check(args, physics, gamma_grid, B, R)
    if linearity["max_relative_l2"] > 5e-5:
        raise RuntimeError(
            "a1 linearity check failed: max relL2=%.6g"
            % linearity["max_relative_l2"]
        )

    params, gamma_index = candidate_parameters(args, gamma_grid)
    score, alias_a1, alias_gamma, alias_gamma_factor = profiled_remote_alias_scores(
        params=params,
        gamma_index=gamma_index,
        gamma_grid=gamma_grid,
        B_master=B,
        R_master=R,
        obs_idx=obs_idx,
        reference_noise=args.reference_noise,
        a1_min=args.a1_min,
        a1_max=args.a1_max,
        remote_a1_abs=args.remote_a1_abs,
        remote_gamma_factor=args.remote_gamma_factor,
        chunk_size=args.profile_chunk,
    )

    np.savez_compressed(
        output_dir / "alias_pool.npz",
        candidate_a1=params[:, 0].astype(np.float32),
        candidate_gamma=params[:, 4].astype(np.float32),
        candidate_gamma_index=gamma_index.astype(np.int32),
        remote_alias_rms_snr=score.astype(np.float32),
        alias_a1=alias_a1.astype(np.float32),
        alias_gamma=alias_gamma.astype(np.float32),
        alias_gamma_factor=alias_gamma_factor.astype(np.float32),
        gamma_grid=gamma_grid.astype(np.float32),
        B_master=B.astype(np.float32),
        R_master=R.astype(np.float32),
        observation_indices=np.asarray(obs_idx, dtype=np.int32),
    )

    qs = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    summary = {
        "candidate_count": int(len(score)),
        "alias_score_min": float(np.min(score)),
        "alias_score_median": float(np.median(score)),
        "alias_score_max": float(np.max(score)),
    }
    for q in qs:
        summary["alias_score_q%03d" % int(round(100*q))] = float(np.quantile(score, q))
    for t in (0.25, 0.50, 0.75, 1.0):
        summary["count_alias_score_ge_%.2f" % t] = int(np.sum(score >= t))
        summary["fraction_alias_score_ge_%.2f" % t] = float(np.mean(score >= t))

    write_csv(output_dir / "alias_pool_summary.csv", [summary])

    # A compact candidate table is useful for auditing without loading the NPZ.
    order = np.argsort(-score)
    rows = []
    for rank, i in enumerate(order[: min(2000, len(order))]):
        rows.append({
            "safety_rank": rank + 1,
            "candidate_index": int(i),
            "a1": float(params[i, 0]),
            "gamma": float(params[i, 4]),
            "remote_alias_rms_snr": float(score[i]),
            "alias_a1": float(alias_a1[i]),
            "alias_gamma": float(alias_gamma[i]),
            "alias_gamma_factor": float(alias_gamma_factor[i]),
        })
    write_csv(output_dir / "top_alias_safe_candidates.csv", rows)
    plot_alias_distribution(output_dir, score)

    metadata = {
        "experiment": "Exp30 continuous alias-scored a1+gamma candidate pool",
        "mode": "a1gamma",
        "fixed_parameters": {"a2": FIXED_A2, "a3": FIXED_A3, "m": FIXED_M},
        "candidate_count": int(args.candidate_count),
        "alias_gamma_points": int(args.alias_gamma_points),
        "a1_range": [float(args.a1_min), float(args.a1_max)],
        "gamma_range": [float(args.gamma_min), float(args.gamma_max)],
        "reference_noise": float(args.reference_noise),
        "remote_a1_abs": float(args.remote_a1_abs),
        "remote_gamma_factor": float(args.remote_gamma_factor),
        "physical_observation_points": int(args.observation_points),
        "model_input_points": int(args.model_input_points),
        "observation_indices": [int(x) for x in obs_idx],
        "integration_points": int(args.integration_points),
        "linearity_check": linearity,
        "seed": int(args.seed),
        "interpretation": (
            "remote_alias_rms_snr is the profiled minimum observed-g RMS-SNR to "
            "a physically remote continuous state on a dense log-gamma grid. "
            "Higher scores mean fewer near-degenerate remote solutions."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("linearity max relL2 : %.3g" % linearity["max_relative_l2"])
    print("alias score median  : %.5g" % np.median(score))
    for t in (0.25, 0.50, 0.75, 1.0):
        print("alias >= %.2f        : %6d / %d"
              % (t, np.sum(score >= t), len(score)))
    print("Read first:")
    print("  %s" % (output_dir / "alias_pool_summary.csv"))
    print("  %s" % (output_dir / "alias_score_distribution.png"))
    print("=" * 100)


if __name__ == "__main__":
    main()
