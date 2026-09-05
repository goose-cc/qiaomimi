#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp35 step 1: build a three-parameter (a1, m, gamma) alias-scored candidate pool.

The forward physics is unchanged.  a2 and a3 stay fixed at their established
values, while a1, m and gamma vary continuously.

Unlike the 2-D Exp30 audit, an exhaustive dense (m,gamma) profile is too large
to evaluate cheaply.  Exp35 therefore performs a high-recall sampled audit:

1. draw a large Latin-hypercube candidate pool in (a1, m, log10(gamma));
2. compute the exact physical g(q^2) for every candidate;
3. use several deterministic random projections + cKDTree to retrieve many
   nearby g-space candidates for each truth;
4. among candidates that are OUTSIDE the declared recovery tolerances, compute
   the exact RMS-SNR separation in the original observed g-space;
5. keep the minimum exact separation as ``remote_alias_rms_snr``.

The projection is used only for neighbor retrieval.  The reported alias score
is always computed from the original physical g values.

Important: the score is a sampled/high-recall approximation, not a mathematical
proof over the full continuous 3-D parameter domain.  The metadata records that
fact explicitly so later experiments can tighten candidate count / neighbor-k.
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
from scipy.spatial import cKDTree

from data_generate_exp20_pairwise import (
    DEFAULT_PHYSICS,
    FIXED_A2,
    FIXED_A3,
    forward_bank,
)
from data_generate_exp22c_fixed_capacity import canonical_indices


def write_csv(path, rows):
    if not rows:
        return
    fields, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k); fields.append(k)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(
        description="Build Exp35 sampled continuous 3P alias pool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default="data_exp35_3p_alias_pool")
    p.add_argument("--candidate-count", type=int, default=20000)
    p.add_argument("--reference-noise", type=float, default=0.002)

    p.add_argument("--a1-min", type=float, default=0.05)
    p.add_argument("--a1-max", type=float, default=0.20)
    p.add_argument("--m-min", type=float, default=0.40)
    p.add_argument("--m-max", type=float, default=1.20)
    p.add_argument("--gamma-min", type=float, default=0.01)
    p.add_argument("--gamma-max", type=float, default=1.0)

    # "Remote" means outside the intended 3P recovery tolerance in at least one
    # coordinate.  These are configurable because later Exp35 scans may revise
    # the stage tolerances.
    p.add_argument("--remote-a1-abs", type=float, default=0.005)
    p.add_argument("--remote-m-abs", type=float, default=0.03)
    p.add_argument("--remote-gamma-factor", type=float, default=1.20)

    p.add_argument("--observation-points", type=int, default=500)
    p.add_argument("--model-input-points", type=int, default=1000)
    p.add_argument("--integration-points", type=int, default=128)
    p.add_argument("--forward-chunk", type=int, default=512)

    p.add_argument("--projection-dim", type=int, default=24)
    p.add_argument("--projection-repeats", type=int, default=3)
    p.add_argument("--neighbor-k", type=int, default=384)
    p.add_argument("--query-batch", type=int, default=256)
    p.add_argument("--fallback-random", type=int, default=512)

    p.add_argument("--seed", type=int, default=20260860)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def latin_hypercube_1d(n, rng):
    x = (np.arange(int(n), dtype=np.float64) + rng.random(int(n))) / float(n)
    rng.shuffle(x)
    return x


def candidate_parameters(args):
    n = int(args.candidate_count)
    if n < 500:
        raise ValueError("--candidate-count must be >= 500")
    rng = np.random.default_rng(int(args.seed))
    u1 = latin_hypercube_1d(n, rng)
    u2 = latin_hypercube_1d(n, rng)
    u3 = latin_hypercube_1d(n, rng)

    a1 = args.a1_min + (args.a1_max - args.a1_min) * u1
    mass = args.m_min + (args.m_max - args.m_min) * u2
    lg0, lg1 = math.log10(args.gamma_min), math.log10(args.gamma_max)
    gamma = np.power(10.0, lg0 + (lg1 - lg0) * u3)

    # Eight exact domain corners make support checks unambiguous.
    corners = [
        (a, m, g)
        for a in (args.a1_min, args.a1_max)
        for m in (args.m_min, args.m_max)
        for g in (args.gamma_min, args.gamma_max)
    ]
    for i, (a, m, g) in enumerate(corners):
        if i >= n:
            break
        a1[i], mass[i], gamma[i] = a, m, g

    params = np.zeros((n, 5), dtype=np.float64)
    params[:, 0] = a1
    params[:, 1] = FIXED_A2
    params[:, 2] = FIXED_A3
    params[:, 3] = mass
    params[:, 4] = gamma
    return params


def forward_in_chunks(params, physics, integration_points, chunk):
    fx_parts, g_parts = [], []
    for start in range(0, len(params), int(chunk)):
        stop = min(start + int(chunk), len(params))
        fx, g = forward_bank(
            params[start:stop],
            physics=physics,
            integration_points=int(integration_points),
        )
        fx_parts.append(np.asarray(fx, dtype=np.float32))
        g_parts.append(np.asarray(g, dtype=np.float32))
        print("forward %6d / %d" % (stop, len(params)))
    return np.concatenate(fx_parts, axis=0), np.concatenate(g_parts, axis=0)


def remote_mask(params, i, js, da1, dm, gamma_factor):
    p = params[int(i)]
    q = params[np.asarray(js, dtype=np.int64)]
    gf = np.maximum(
        q[:, 4] / max(float(p[4]), 1e-30),
        float(p[4]) / np.maximum(q[:, 4], 1e-30),
    )
    return (
        (np.abs(q[:, 0] - p[0]) >= float(da1))
        | (np.abs(q[:, 3] - p[3]) >= float(dm))
        | (gf >= float(gamma_factor))
    )


def exact_rms_snr(g_obs, master_rms, reference_noise, i, js):
    js = np.asarray(js, dtype=np.int64)
    d = np.asarray(g_obs[js], dtype=np.float64) - np.asarray(
        g_obs[int(i)], dtype=np.float64
    )[None, :]
    rms_d = np.sqrt(np.mean(d * d, axis=1))
    sigma = float(reference_noise) * 0.5 * (
        np.asarray(master_rms[js], dtype=np.float64)
        + float(master_rms[int(i)])
    )
    return rms_d / np.maximum(sigma, 1e-30)


def build_projection(g_obs, dim, seed):
    x = np.asarray(g_obs, dtype=np.float64)
    mean = np.mean(x, axis=0)
    rng = np.random.default_rng(int(seed))
    proj = rng.standard_normal((x.shape[1], int(dim))).astype(np.float64)
    proj /= math.sqrt(float(dim))
    z = (x - mean[None, :]) @ proj
    # One scalar scale keeps the projection numerically well conditioned without
    # changing relative geometry across projected coordinates.
    s = float(np.std(z))
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    return (z / s).astype(np.float32)


def sampled_alias_scores(args, params, g_master, obs_idx):
    g_obs = np.asarray(g_master[:, obs_idx], dtype=np.float32)
    master_rms = np.sqrt(np.mean(np.asarray(g_master, dtype=np.float64) ** 2, axis=1))
    n = len(params)
    k = min(max(int(args.neighbor_k), 8), n)
    query_batch = max(1, int(args.query_batch))

    neighbor_sets = [set() for _ in range(n)]
    for r in range(int(args.projection_repeats)):
        z = build_projection(
            g_obs, int(args.projection_dim), int(args.seed) + 1009 * (r + 1)
        )
        tree = cKDTree(z)
        for start in range(0, n, query_batch):
            stop = min(start + query_batch, n)
            _, idx = tree.query(z[start:stop], k=k, workers=-1)
            idx = np.asarray(idx)
            if idx.ndim == 1:
                idx = idx[:, None]
            for local, row in enumerate(idx):
                i = start + local
                neighbor_sets[i].update(int(j) for j in row if int(j) != i)
        print("projection neighbor pass %d / %d" % (r + 1, args.projection_repeats))

    score = np.full(n, np.inf, dtype=np.float64)
    alias_index = np.full(n, -1, dtype=np.int32)
    retrieved_remote_count = np.zeros(n, dtype=np.int32)
    rng = np.random.default_rng(int(args.seed) + 91919)

    for i in range(n):
        js = np.fromiter(neighbor_sets[i], dtype=np.int64)
        if len(js):
            keep = remote_mask(
                params, i, js,
                args.remote_a1_abs,
                args.remote_m_abs,
                args.remote_gamma_factor,
            )
            js = js[keep]
        retrieved_remote_count[i] = int(len(js))

        # Fallback random remote candidates reduce the chance that an acceptable
        # local cloud hides all remote alternatives from the projected kNN list.
        if int(args.fallback_random) > 0:
            rr = rng.integers(0, n, size=int(args.fallback_random) * 2)
            rr = np.unique(rr[rr != i])
            if len(rr):
                keep = remote_mask(
                    params, i, rr,
                    args.remote_a1_abs,
                    args.remote_m_abs,
                    args.remote_gamma_factor,
                )
                rr = rr[keep][: int(args.fallback_random)]
                if len(rr):
                    js = np.unique(np.concatenate([js, rr])) if len(js) else rr

        if len(js):
            s = exact_rms_snr(g_obs, master_rms, args.reference_noise, i, js)
            jloc = int(np.argmin(s))
            score[i] = float(s[jloc])
            alias_index[i] = int(js[jloc])

        if (i + 1) % 1000 == 0 or i + 1 == n:
            print("alias audit %6d / %d" % (i + 1, n))

    if np.any(alias_index < 0):
        bad = int(np.sum(alias_index < 0))
        raise RuntimeError(
            "%d candidates found no sampled remote alternative; increase pool/neighbor-k"
            % bad
        )
    return score, alias_index, retrieved_remote_count


def plot_summary(output_dir, params, scores):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores[np.isfinite(scores)], bins=60)
    for x in (0.20, 0.30, 0.40, 0.50, 0.75, 1.0):
        ax.axvline(x, linestyle="--", linewidth=1)
    ax.set_xlabel("sampled remote-alias RMS-SNR")
    ax.set_ylabel("candidate count")
    ax.set_title("Exp35 3P sampled identifiability")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "alias_score_distribution.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(params[:, 3], params[:, 4], c=scores, s=8)
    ax.set_yscale("log")
    ax.set_xlabel("m")
    ax.set_ylabel("gamma")
    ax.set_title("Exp35 alias safety across m-gamma")
    fig.colorbar(sc, ax=ax, label="remote alias RMS-SNR")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "alias_safety_m_gamma.png", dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    if not (0 < args.a1_min < args.a1_max):
        raise ValueError("require 0 < a1_min < a1_max")
    if not (0 < args.m_min < args.m_max):
        raise ValueError("require 0 < m_min < m_max")
    if not (0 < args.gamma_min < args.gamma_max):
        raise ValueError("require 0 < gamma_min < gamma_max")
    if args.reference_noise <= 0:
        raise ValueError("--reference-noise must be > 0")
    if args.remote_a1_abs <= 0 or args.remote_m_abs <= 0:
        raise ValueError("remote absolute tolerances must be > 0")
    if args.remote_gamma_factor <= 1:
        raise ValueError("--remote-gamma-factor must be > 1")
    if args.projection_dim < 4 or args.projection_repeats < 1:
        raise ValueError("projection settings are too small")

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise FileExistsError("%s is not empty; use --overwrite" % out)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    physics = replace(DEFAULT_PHYSICS, q2_points=int(args.model_input_points))
    obs_idx = canonical_indices(
        int(args.model_input_points), int(args.observation_points)
    )
    params = candidate_parameters(args)

    print("=" * 100)
    print("Exp35 3P candidate pool: a1 + m + gamma")
    print("candidate count   : %d" % len(params))
    print("ranges            : a1=[%.4g, %.4g], m=[%.4g, %.4g], gamma=[%.4g, %.4g]"
          % (args.a1_min, args.a1_max, args.m_min, args.m_max,
             args.gamma_min, args.gamma_max))
    print("remote criterion  : |da1|>=%.4g OR |dm|>=%.4g OR gamma-factor>=%.4g"
          % (args.remote_a1_abs, args.remote_m_abs, args.remote_gamma_factor))
    print("=" * 100)

    _, g_master = forward_in_chunks(
        params, physics, args.integration_points, args.forward_chunk
    )
    scores, alias_idx, retrieved_remote_count = sampled_alias_scores(
        args, params, g_master, obs_idx
    )
    alias_params = params[alias_idx]
    gamma_factor = np.maximum(
        alias_params[:, 4] / np.maximum(params[:, 4], 1e-30),
        params[:, 4] / np.maximum(alias_params[:, 4], 1e-30),
    )

    np.savez_compressed(
        out / "alias_pool_3p.npz",
        candidate_parameters=params.astype(np.float32),
        g_master=g_master.astype(np.float32),
        remote_alias_rms_snr=scores.astype(np.float32),
        alias_candidate_index=alias_idx.astype(np.int32),
        alias_a1=alias_params[:, 0].astype(np.float32),
        alias_m=alias_params[:, 3].astype(np.float32),
        alias_gamma=alias_params[:, 4].astype(np.float32),
        alias_gamma_factor=gamma_factor.astype(np.float32),
        retrieved_remote_count=retrieved_remote_count.astype(np.int32),
        observation_indices=np.asarray(obs_idx, dtype=np.int32),
    )

    summary = {
        "candidate_count": int(len(scores)),
        "alias_score_min": float(np.min(scores)),
        "alias_score_median": float(np.median(scores)),
        "alias_score_p90": float(np.quantile(scores, 0.90)),
        "alias_score_p95": float(np.quantile(scores, 0.95)),
        "alias_score_max": float(np.max(scores)),
        "median_retrieved_remote_count": float(np.median(retrieved_remote_count)),
    }
    for t in (0.20, 0.30, 0.40, 0.50, 0.75, 1.0):
        summary["count_alias_ge_%.2f" % t] = int(np.sum(scores >= t))
        summary["fraction_alias_ge_%.2f" % t] = float(np.mean(scores >= t))
    write_csv(out / "alias_pool_summary.csv", [summary])

    order = np.argsort(-scores)
    rows = []
    for rank, i in enumerate(order[: min(3000, len(order))]):
        j = int(alias_idx[i])
        rows.append({
            "safety_rank": rank + 1,
            "candidate_index": int(i),
            "a1": float(params[i, 0]),
            "m": float(params[i, 3]),
            "gamma": float(params[i, 4]),
            "remote_alias_rms_snr": float(scores[i]),
            "alias_candidate_index": j,
            "alias_a1": float(params[j, 0]),
            "alias_m": float(params[j, 3]),
            "alias_gamma": float(params[j, 4]),
            "alias_gamma_factor": float(gamma_factor[i]),
            "retrieved_remote_count": int(retrieved_remote_count[i]),
        })
    write_csv(out / "top_alias_safe_candidates.csv", rows)
    plot_summary(out, params, scores)

    metadata = {
        "experiment": "Exp35 3P sampled alias pool",
        "mode": "a1mgamma",
        "free_parameters": ["a1", "m", "gamma"],
        "fixed_parameters": {"a2": FIXED_A2, "a3": FIXED_A3},
        "parameter_ranges": {
            "a1": [float(args.a1_min), float(args.a1_max)],
            "m": [float(args.m_min), float(args.m_max)],
            "gamma": [float(args.gamma_min), float(args.gamma_max)],
        },
        "reference_noise": float(args.reference_noise),
        "remote_criteria": {
            "a1_abs": float(args.remote_a1_abs),
            "m_abs": float(args.remote_m_abs),
            "gamma_factor": float(args.remote_gamma_factor),
        },
        "candidate_count": int(args.candidate_count),
        "physical_observation_points": int(args.observation_points),
        "model_input_points": int(args.model_input_points),
        "observation_indices": [int(x) for x in obs_idx],
        "integration_points": int(args.integration_points),
        "alias_method": {
            "name": "multi-random-projection kNN with exact original-g reranking",
            "projection_dim": int(args.projection_dim),
            "projection_repeats": int(args.projection_repeats),
            "neighbor_k_per_projection": int(args.neighbor_k),
            "fallback_random": int(args.fallback_random),
            "important_limitation": (
                "remote_alias_rms_snr is a sampled/high-recall approximation over "
                "the generated 3P candidate pool, not an exhaustive proof over the "
                "full continuous parameter domain."
            ),
        },
        "seed": int(args.seed),
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("alias median        : %.5g" % np.median(scores))
    print("alias >= 0.40       : %d / %d" % (np.sum(scores >= 0.40), len(scores)))
    print("Read first:")
    print("  %s" % (out / "alias_pool_summary.csv"))
    print("  %s" % (out / "alias_score_distribution.png"))
    print("=" * 100)


if __name__ == "__main__":
    main()
