#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared numerical core for Exp37.

The project physics is not changed. This module provides:
- a float64 mirror of the existing forward observation;
- recovery-aligned grid alias profiling;
- direct-verified continuous alias refinement;
- stable g-space distances and parameter normalization;
- low-dimensional VarPro utilities.

Free inverse parameters: a1, m, gamma.
Fixed: a2=0.025, a3=0.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

from mc_physics import legendre_rule, output_grids_numpy
from mc_pool_config import DEFAULT_PHYSICS, PhysicsConfig


REGIONS = ("a1_low", "a1_high", "m_low", "m_high", "gamma_low", "gamma_high")


def _as_params(parameters: np.ndarray) -> np.ndarray:
    p = np.asarray(parameters, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 5:
        raise ValueError(f"parameters must have shape [N,5], got {p.shape}")
    return p


def forward64(
    parameters: np.ndarray,
    integration_points: int = 128,
    config: PhysicsConfig = DEFAULT_PHYSICS,
    q_block_size: int = 25,
) -> np.ndarray:
    """Float64 mirror of mc_physics.scaled_forward_observation_numpy."""
    p = _as_params(parameters)
    _, q2 = output_grids_numpy(config)
    q2 = np.asarray(q2, dtype=np.float64)
    x, w = legendre_rule(int(integration_points))
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    s_mid = 0.5 * (config.s_max + config.s_min)
    s_half = 0.5 * (config.s_max - config.s_min)
    s_fixed = s_mid + s_half * x
    s_weights = s_half * w

    background = (
        p[:, 1:2] * s_fixed.reshape(1, -1) + p[:, 2:3]
    ) / (s_fixed.reshape(1, -1) + config.shift) ** 2
    background_kernel = (
        s_weights.reshape(-1, 1)
        / (s_fixed.reshape(-1, 1) - q2.reshape(1, -1))
    )
    result = background @ background_kernel

    a1 = p[:, 0:1]
    mass = p[:, 3:4]
    width = p[:, 3:4] * p[:, 4:5]
    z0 = np.arctan((config.s_min - mass) / width)
    z1 = np.arctan((config.s_max - mass) / width)
    z_mid = 0.5 * (z0 + z1)
    z_half = 0.5 * (z1 - z0)
    z = z_mid + z_half * x.reshape(1, -1)
    s_res = mass + width * np.tan(z)
    coefficient = (
        (a1 / np.pi)
        * z_half
        * w.reshape(1, -1)
        / (s_res + config.shift) ** 2
    )
    for start in range(0, len(q2), int(q_block_size)):
        stop = min(start + int(q_block_size), len(q2))
        denominator = s_res[:, :, None] - q2[None, None, start:stop]
        result[:, start:stop] += np.sum(
            coefficient[:, :, None] / denominator, axis=1
        )
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite float64 forward observations")
    return config.data_scale * result


def forward64_batched(parameters: np.ndarray, integration_points: int, batch_size: int = 768) -> np.ndarray:
    p = _as_params(parameters)
    parts = []
    for start in range(0, len(p), int(batch_size)):
        parts.append(forward64(p[start:start+int(batch_size)], integration_points=integration_points))
    if not parts:
        return np.empty((0, DEFAULT_PHYSICS.q2_points), dtype=np.float64)
    return np.concatenate(parts, axis=0)


def rms_rows(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return np.sqrt(np.mean(a * a, axis=1))


def normalize_free_params(params: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    p = _as_params(params)
    pr = cfg["parameter_ranges"]
    a1 = (p[:, 0] - pr["a1"][0]) / (pr["a1"][1] - pr["a1"][0])
    m = (p[:, 3] - pr["m"][0]) / (pr["m"][1] - pr["m"][0])
    lg = np.log10(p[:, 4])
    lo, hi = math.log10(pr["gamma"][0]), math.log10(pr["gamma"][1])
    g = (lg - lo) / (hi - lo)
    return np.column_stack([a1, m, g]).astype(np.float64)


def make_profile_bank(cfg: dict[str, Any], integration_points: int,
                      m_points: int | None = None, log_gamma_points: int | None = None):
    """Build unit-resonance bank g = background + a1 * R(m,gamma)."""
    pr = cfg["parameter_ranges"]
    fixed = cfg["fixed_parameters"]
    pg = cfg["profile_grid"]
    M = int(m_points or pg["m_points"])
    G = int(log_gamma_points or pg["log_gamma_points"])
    m_grid = np.linspace(pr["m"][0], pr["m"][1], M, dtype=np.float64)
    lg_grid = np.linspace(math.log10(pr["gamma"][0]), math.log10(pr["gamma"][1]), G, dtype=np.float64)
    mm, ll = np.meshgrid(m_grid, lg_grid, indexing="ij")
    gamma = 10.0 ** ll.ravel()

    background_p = np.array(
        [[0.0, fixed["a2"], fixed["a3"], 0.5 * (pr["m"][0] + pr["m"][1]), 0.5]],
        dtype=np.float64,
    )
    background = forward64_batched(background_p, integration_points)[0]
    profile_p = np.column_stack([
        np.ones(mm.size),
        np.full(mm.size, fixed["a2"]),
        np.full(mm.size, fixed["a3"]),
        mm.ravel(),
        gamma,
    ]).astype(np.float64)
    g1 = forward64_batched(profile_p, integration_points, batch_size=512)
    resonance = g1 - background[None, :]
    r2 = np.einsum("ij,ij->i", resonance, resonance, dtype=np.float64)
    if np.any(~np.isfinite(r2)) or np.any(r2 <= 0):
        raise RuntimeError("invalid unit-resonance profile bank")
    return m_grid, lg_grid, background, resonance, r2


def _direct_sse(target_y: np.ndarray, rr: np.ndarray, a1: float) -> float:
    resid = np.asarray(target_y, dtype=np.float64) - float(a1) * np.asarray(rr, dtype=np.float64)
    return float(np.dot(resid, resid))


def best_a1_and_sse(target_y: np.ndarray, rr: np.ndarray, bounds: tuple[float, float]) -> tuple[float, float]:
    r = np.asarray(rr, dtype=np.float64)
    r2 = float(np.dot(r, r))
    if not np.isfinite(r2) or r2 <= 0:
        return math.nan, math.inf
    a = float(np.dot(np.asarray(target_y, dtype=np.float64), r) / r2)
    a = min(max(a, float(bounds[0])), float(bounds[1]))
    return a, _direct_sse(target_y, r, a)


def _best_region(dot, y2, r2, shape, m_grid, lg_grid, a1_bounds,
                 region, true_a1, true_m, true_lg, tol):
    amin, amax = map(float, a1_bounds)
    M, G = shape
    dot = np.asarray(dot, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    free_a = np.clip(dot / r2, amin, amax)
    sse_free = np.maximum(y2 + free_a * free_a * r2 - 2.0 * free_a * dot, 0.0)

    if region == "a1_low":
        hi = true_a1 - float(tol["a1_abs"])
        if hi < amin:
            return math.inf, -1, math.nan
        a = np.clip(dot / r2, amin, min(hi, amax))
        sse = np.maximum(y2 + a * a * r2 - 2.0 * a * dot, 0.0)
        idx = int(np.argmin(sse))
        return float(sse[idx]), idx, float(a[idx])
    if region == "a1_high":
        lo = true_a1 + float(tol["a1_abs"])
        if lo > amax:
            return math.inf, -1, math.nan
        a = np.clip(dot / r2, max(lo, amin), amax)
        sse = np.maximum(y2 + a * a * r2 - 2.0 * a * dot, 0.0)
        idx = int(np.argmin(sse))
        return float(sse[idx]), idx, float(a[idx])

    mat = sse_free.reshape(M, G)
    if region == "m_low":
        lim = true_m - float(tol["m_abs"])
        ids = np.flatnonzero(m_grid <= lim + 1e-14)
        if len(ids) == 0:
            return math.inf, -1, math.nan
        sub = mat[:ids[-1] + 1, :]
        local = int(np.argmin(sub))
        i, j = np.unravel_index(local, sub.shape)
    elif region == "m_high":
        lim = true_m + float(tol["m_abs"])
        ids = np.flatnonzero(m_grid >= lim - 1e-14)
        if len(ids) == 0:
            return math.inf, -1, math.nan
        i0 = int(ids[0])
        sub = mat[i0:, :]
        local = int(np.argmin(sub))
        ii, j = np.unravel_index(local, sub.shape)
        i = i0 + ii
    elif region == "gamma_low":
        lim = true_lg - math.log10(float(tol["gamma_factor"]))
        ids = np.flatnonzero(lg_grid <= lim + 1e-14)
        if len(ids) == 0:
            return math.inf, -1, math.nan
        sub = mat[:, :ids[-1] + 1]
        local = int(np.argmin(sub))
        i, j = np.unravel_index(local, sub.shape)
    elif region == "gamma_high":
        lim = true_lg + math.log10(float(tol["gamma_factor"]))
        ids = np.flatnonzero(lg_grid >= lim - 1e-14)
        if len(ids) == 0:
            return math.inf, -1, math.nan
        j0 = int(ids[0])
        sub = mat[:, j0:]
        local = int(np.argmin(sub))
        i, jj = np.unravel_index(local, sub.shape)
        j = j0 + jj
    else:
        raise ValueError(region)
    idx = int(i * G + j)
    return float(sse_free[idx]), idx, float(free_a[idx])


def grid_region_winners(target_y, true_a1, true_m, true_lg,
                        m_grid, lg_grid, resonance, r2, a1_bounds, tol):
    dot = np.asarray(resonance, dtype=np.float64) @ np.asarray(target_y, dtype=np.float64)
    y2 = float(np.dot(target_y, target_y))
    M, G = len(m_grid), len(lg_grid)
    winners = []
    for region in REGIONS:
        sf, idx, aa = _best_region(
            dot, y2, r2, (M, G), m_grid, lg_grid, a1_bounds,
            region, true_a1, true_m, true_lg, tol,
        )
        if idx < 0 or not np.isfinite(sf):
            continue
        i, j = divmod(idx, G)
        sd = _direct_sse(target_y, resonance[idx], aa)
        winners.append({
            "region": region, "idx": idx, "i": i, "j": j,
            "a1": float(aa), "m": float(m_grid[i]), "lg": float(lg_grid[j]),
            "sse_formula": float(sf), "sse_direct": float(sd),
        })
    return winners


def profile_alias_margins(clean_g64, params, cfg, background, m_grid, lg_grid,
                          resonance, r2, tol, chunk_size: int = 128):
    """Fast grid profile for all candidates; saved margins are direct-verified."""
    ref_noise = float(cfg["reference_noise"])
    pr = cfg["parameter_ranges"]
    a1_bounds = tuple(map(float, pr["a1"]))
    M, G = len(m_grid), len(lg_grid)
    n = len(params)

    margin = np.full(n, np.inf, dtype=np.float64)
    alias_a1 = np.full(n, np.nan)
    alias_m = np.full(n, np.nan)
    alias_gamma = np.full(n, np.nan)
    trigger = np.empty(n, dtype=object)
    trigger[:] = ""

    R = np.asarray(resonance, dtype=np.float64)
    y_all = np.asarray(clean_g64, dtype=np.float64) - np.asarray(background, dtype=np.float64)[None, :]
    g_rms = rms_rows(clean_g64)

    for start in range(0, n, int(chunk_size)):
        stop = min(start + int(chunk_size), n)
        y = y_all[start:stop]
        dots = y @ R.T
        y2s = np.einsum("ij,ij->i", y, y, dtype=np.float64)
        for bi in range(stop-start):
            gi = start + bi
            true_a1 = float(params[gi, 0])
            true_m = float(params[gi, 3])
            true_lg = math.log10(float(params[gi, 4]))
            candidates = []
            for region in REGIONS:
                sf, idx, aa = _best_region(
                    dots[bi], float(y2s[bi]), r2, (M, G), m_grid, lg_grid,
                    a1_bounds, region, true_a1, true_m, true_lg, tol,
                )
                if idx >= 0 and np.isfinite(sf):
                    sd = _direct_sse(y[bi], R[idx], aa)
                    candidates.append((sd, idx, aa, region))
            if not candidates:
                continue
            sd, idx, aa, region = min(candidates, key=lambda x: x[0])
            i, j = divmod(idx, G)
            denom = max(ref_noise * float(g_rms[gi]), 1e-30)
            margin[gi] = math.sqrt(max(sd, 0.0) / clean_g64.shape[1]) / denom
            alias_a1[gi] = aa
            alias_m[gi] = m_grid[i]
            alias_gamma[gi] = 10.0 ** lg_grid[j]
            trigger[gi] = region
        if start == 0 or stop == n or (start // int(chunk_size)) % 20 == 0:
            print(f"  grid aliases: {stop}/{n}")
    return {
        "margin": margin,
        "alias_a1": alias_a1,
        "alias_m": alias_m,
        "alias_gamma": alias_gamma,
        "trigger": trigger,
    }


def _region_specs(true_a1, true_m, true_lg, cfg, tol):
    pr = cfg["parameter_ranges"]
    amin, amax = map(float, pr["a1"])
    lgmin, lgmax = math.log10(pr["gamma"][0]), math.log10(pr["gamma"][1])
    specs = []
    if true_a1 - tol["a1_abs"] >= amin:
        specs.append(("a1_low", (amin, true_a1 - tol["a1_abs"]), tuple(pr["m"]), (lgmin, lgmax)))
    if true_a1 + tol["a1_abs"] <= amax:
        specs.append(("a1_high", (true_a1 + tol["a1_abs"], amax), tuple(pr["m"]), (lgmin, lgmax)))
    if true_m - tol["m_abs"] >= pr["m"][0]:
        specs.append(("m_low", (amin, amax), (pr["m"][0], true_m - tol["m_abs"]), (lgmin, lgmax)))
    if true_m + tol["m_abs"] <= pr["m"][1]:
        specs.append(("m_high", (amin, amax), (true_m + tol["m_abs"], pr["m"][1]), (lgmin, lgmax)))
    d = math.log10(float(tol["gamma_factor"]))
    if true_lg - d >= lgmin:
        specs.append(("gamma_low", (amin, amax), tuple(pr["m"]), (lgmin, true_lg - d)))
    if true_lg + d <= lgmax:
        specs.append(("gamma_high", (amin, amax), tuple(pr["m"]), (true_lg + d, lgmax)))
    return specs


def refine_alias_ids(ids, params, clean_g64, cfg, background, m_grid, lg_grid,
                     resonance, r2, grid_diag, integration_points,
                     maxiter: int | None = None, multistart: int | None = None,
                     progress_every: int = 100):
    """Direct-verified continuous refinement for a bank of candidate states."""
    ids = np.asarray(ids, dtype=np.int64)
    pr = cfg["parameter_ranges"]
    tol = cfg["recovery_tolerances"]
    ref_noise = float(cfg["reference_noise"])
    amin, amax = map(float, pr["a1"])
    R = np.asarray(resonance, dtype=np.float64)
    interp = RegularGridInterpolator(
        (m_grid, lg_grid), R.reshape(len(m_grid), len(lg_grid), -1),
        method="linear", bounds_error=False, fill_value=None,
    )
    fixed = cfg["fixed_parameters"]
    ccfg = cfg.get("continuous_prebank", {})
    maxiter = int(maxiter or ccfg.get("continuous_maxiter", 60))
    multistart = int(multistart or ccfg.get("continuous_multistart", 3))

    def direct_rr(mv, lgv):
        pp = np.array([[1.0, fixed["a2"], fixed["a3"], mv, 10.0 ** lgv]], dtype=np.float64)
        return forward64(pp, integration_points=integration_points)[0] - background

    rows = []
    for rank, sid in enumerate(ids):
        p = params[sid]
        true_a1 = float(p[0])
        true_m = float(p[3])
        true_gam = float(p[4])
        true_lg = math.log10(true_gam)
        target_y = np.asarray(clean_g64[sid], dtype=np.float64) - background
        target_rms = float(rms_rows(clean_g64[sid:sid+1])[0])
        denom = max(ref_noise * target_rms, 1e-30)

        grid_winners = grid_region_winners(
            target_y, true_a1, true_m, true_lg,
            m_grid, lg_grid, R, r2, (amin, amax), tol,
        )
        if not grid_winners:
            raise RuntimeError(f"state {sid}: no legal unacceptable alias region")
        gw = min(grid_winners, key=lambda x: x["sse_direct"])
        grid_margin = math.sqrt(max(gw["sse_direct"], 0.0) / clean_g64.shape[1]) / denom
        saved = float(grid_diag["margin"][sid])
        if abs(grid_margin - saved) > max(1e-10, 1e-7 * max(grid_margin, saved, 1.0)):
            raise RuntimeError(f"state {sid}: grid recompute mismatch {grid_margin} vs {saved}")

        spec_map = {x[0]: x[1:] for x in _region_specs(true_a1, true_m, true_lg, cfg, tol)}
        candidates = []
        for w in grid_winners:
            region = w["region"]
            if region not in spec_map:
                continue
            a_bounds, m_bounds, lg_bounds = spec_map[region]
            candidates.append((w["sse_direct"], w["a1"], w["m"], 10.0**w["lg"], region, "verified_grid"))

            def objective(x):
                rr = np.asarray(interp([[float(x[0]), float(x[1])]])[0], dtype=np.float64)
                return best_a1_and_sse(target_y, rr, tuple(map(float, a_bounds)))[1]

            grid_seed = np.array([w["m"], w["lg"]], dtype=np.float64)
            boundary_seed = np.array([
                min(max(true_m, float(m_bounds[0])), float(m_bounds[1])),
                min(max(true_lg, float(lg_bounds[0])), float(lg_bounds[1])),
            ], dtype=np.float64)
            seeds = [grid_seed, boundary_seed]
            if multistart >= 3 and np.linalg.norm(grid_seed - boundary_seed) > 1e-12:
                seeds.append(0.5 * (grid_seed + boundary_seed))
            seeds = seeds[:max(1, multistart)]

            for seed in seeds:
                opt = minimize(
                    objective, x0=seed, method="L-BFGS-B",
                    bounds=[tuple(map(float, m_bounds)), tuple(map(float, lg_bounds))],
                    options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-10},
                )
                mv, lgv = float(opt.x[0]), float(opt.x[1])
                rr = direct_rr(mv, lgv)
                aa, sse = best_a1_and_sse(target_y, rr, tuple(map(float, a_bounds)))
                if np.isfinite(sse):
                    candidates.append((sse, aa, mv, 10.0**lgv, region, "continuous_direct_verified"))

        best = min(candidates, key=lambda x: x[0])
        sse, aa, mm, gg, region, source = best
        refined = math.sqrt(max(float(sse), 0.0) / clean_g64.shape[1]) / denom
        refined = min(refined, grid_margin)
        rows.append({
            "state_id": int(sid),
            "recovery_alias_margin_grid_verified": float(grid_margin),
            "recovery_alias_margin_refined": float(refined),
            "refined_minus_grid_margin": float(refined-grid_margin),
            "alias_source_refined": source,
            "alias_trigger_refined": region,
            "alias_a1_refined": float(aa),
            "alias_m_refined": float(mm),
            "alias_gamma_refined": float(gg),
            "alias_delta_a1_refined": abs(float(aa)-true_a1),
            "alias_delta_m_refined": abs(float(mm)-true_m),
            "alias_gamma_factor_refined": max(float(gg)/true_gam, true_gam/float(gg)),
        })
        if (rank + 1) % int(progress_every) == 0 or rank + 1 == len(ids):
            print(f"  continuous aliases: {rank+1}/{len(ids)}")
    return rows


def symmetric_g_distance(g1, g2, rms1, rms2, ref_noise):
    d = np.sqrt(np.mean((np.asarray(g1, dtype=np.float64) - np.asarray(g2, dtype=np.float64)) ** 2, axis=1))
    scale = float(ref_noise) * np.sqrt(np.maximum(np.asarray(rms1, dtype=np.float64) * float(rms2), 1e-60))
    return d / np.maximum(scale, 1e-30)


def pairwise_g_distance(clean: np.ndarray, ref_noise: float) -> np.ndarray:
    g = np.asarray(clean, dtype=np.float64)
    n = len(g)
    r = rms_rows(g)
    D = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        diff = g - g[i:i+1]
        d = rms_rows(diff)
        den = float(ref_noise) * np.sqrt(np.maximum(r * r[i], 1e-60))
        D[i] = (d / np.maximum(den, 1e-30)).astype(np.float32)
    np.fill_diagonal(D, np.inf)
    return D


def coverage_radius(reference_norm: np.ndarray, selected_norm: np.ndarray, chunk: int = 2048) -> float:
    ref = np.asarray(reference_norm, dtype=np.float64)
    sel = np.asarray(selected_norm, dtype=np.float64)
    worst = 0.0
    for start in range(0, len(ref), int(chunk)):
        x = ref[start:start+int(chunk)]
        d2 = np.sum((x[:, None, :] - sel[None, :, :]) ** 2, axis=2)
        worst = max(worst, float(np.sqrt(np.min(d2, axis=1)).max()))
    return worst


def unacceptable(true_p: np.ndarray, alias_p: np.ndarray, tol: dict[str, float], eps: float = 2e-10) -> bool:
    da = abs(float(alias_p[0]) - float(true_p[0]))
    dm = abs(float(alias_p[3]) - float(true_p[3]))
    gf = max(float(alias_p[4]) / float(true_p[4]), float(true_p[4]) / float(alias_p[4]))
    return (
        da + eps >= float(tol["a1_abs"])
        or dm + eps >= float(tol["m_abs"])
        or gf + eps >= float(tol["gamma_factor"])
    )


def alias_margin_direct(true_p: np.ndarray, alias_p: np.ndarray, cfg: dict[str, Any], integration_points: int) -> float:
    gg = forward64(np.vstack([true_p, alias_p]), integration_points=integration_points)
    den = max(float(cfg["reference_noise"]) * float(rms_rows(gg[:1])[0]), 1e-30)
    return float(np.sqrt(np.mean((gg[1] - gg[0]) ** 2)) / den)


def varpro_grid_seeds(observation: np.ndarray, background: np.ndarray, resonance: np.ndarray,
                      r2: np.ndarray, a1_bounds: tuple[float, float], seed_count: int):
    y = np.asarray(observation, dtype=np.float64) - np.asarray(background, dtype=np.float64)
    R = np.asarray(resonance, dtype=np.float64)
    dots = R @ y
    y2 = float(np.dot(y, y))
    a = np.clip(dots / r2, float(a1_bounds[0]), float(a1_bounds[1]))
    sse = np.maximum(y2 + a*a*r2 - 2.0*a*dots, 0.0)
    k = min(max(1, int(seed_count) * 8), len(sse))
    raw = np.argpartition(sse, k-1)[:k]
    raw = raw[np.argsort(sse[raw])]
    return raw, a, sse


def varpro_predict_one(observation: np.ndarray, cfg: dict[str, Any], background: np.ndarray,
                       m_grid: np.ndarray, lg_grid: np.ndarray, resonance: np.ndarray, r2: np.ndarray,
                       integration_points: int, seed_count: int = 3, maxiter: int = 60):
    """Profile a1 analytically and optimize m,log10(gamma); direct-verify final fit."""
    pr = cfg["parameter_ranges"]
    fixed = cfg["fixed_parameters"]
    amin, amax = map(float, pr["a1"])
    R = np.asarray(resonance, dtype=np.float64)
    interp = RegularGridInterpolator(
        (m_grid, lg_grid), R.reshape(len(m_grid), len(lg_grid), -1),
        method="linear", bounds_error=False, fill_value=None,
    )
    y = np.asarray(observation, dtype=np.float64) - background
    raw, agrid, ssegrid = varpro_grid_seeds(observation, background, R, r2, (amin, amax), seed_count)
    G = len(lg_grid)

    # De-duplicate adjacent grid seeds to probe distinct basins.
    seeds = []
    for idx in raw:
        i, j = divmod(int(idx), G)
        if all(abs(i-ii) + abs(j-jj) >= 2 for ii, jj, _ in seeds):
            seeds.append((i, j, int(idx)))
        if len(seeds) >= int(seed_count):
            break
    if not seeds:
        idx = int(raw[0])
        i, j = divmod(idx, G)
        seeds = [(i, j, idx)]

    candidates = []
    for i, j, idx in seeds:
        # Preserve direct grid candidate.
        aa = float(agrid[idx])
        candidates.append((_direct_sse(y, R[idx], aa), aa, float(m_grid[i]), 10.0**float(lg_grid[j]), "grid"))

        def objective(x):
            rr = np.asarray(interp([[float(x[0]), float(x[1])]])[0], dtype=np.float64)
            return best_a1_and_sse(y, rr, (amin, amax))[1]

        opt = minimize(
            objective,
            x0=np.array([m_grid[i], lg_grid[j]], dtype=np.float64),
            method="L-BFGS-B",
            bounds=[tuple(map(float, pr["m"])),
                    (math.log10(pr["gamma"][0]), math.log10(pr["gamma"][1]))],
            options={"maxiter": int(maxiter), "ftol": 1e-15, "gtol": 1e-10},
        )
        mv, lgv = float(opt.x[0]), float(opt.x[1])
        pp = np.array([[1.0, fixed["a2"], fixed["a3"], mv, 10.0**lgv]], dtype=np.float64)
        rr = forward64(pp, integration_points=integration_points)[0] - background
        aa, sse = best_a1_and_sse(y, rr, (amin, amax))
        candidates.append((sse, aa, mv, 10.0**lgv, "continuous"))

    sse, aa, mm, gg, source = min(candidates, key=lambda x: x[0])
    fitted = background + aa * (
        forward64(
            np.array([[1.0, fixed["a2"], fixed["a3"], mm, gg]], dtype=np.float64),
            integration_points=integration_points,
        )[0] - background
    )
    rel = float(np.linalg.norm(fitted - np.asarray(observation, dtype=np.float64)) /
                max(np.linalg.norm(np.asarray(observation, dtype=np.float64)), 1e-30))
    return {
        "pred_a1": float(aa), "pred_m": float(mm), "pred_gamma": float(gg),
        "fit_sse": float(sse), "fit_relL2": rel, "source": source,
    }
