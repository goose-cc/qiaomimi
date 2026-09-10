#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp36 data builder: recovery-aligned identifiable 3P data.

Free parameters:
    a1, m, gamma
Fixed:
    a2=0.025, a3=0

The key difference from the earlier generalized-alias pipeline is that the
primary alias definition is aligned with the actual recovery criterion:
    |da1| >= 0.005 OR |dm| >= 0.03 OR gamma-factor >= 1.2.

For each candidate state, the closest unacceptable alternative is profiled over
a regular (m, log10(gamma)) grid.  Because the forward map is linear in a1 for
fixed (m, gamma), the best a1 is solved analytically.  Final selected states are
then refined in continuous (m, log-gamma) space with multi-start interpolation.
Every grid and refined winner is re-scored from a direct float64 residual.  The
refined result is never allowed to be worse than the verified grid result.

This file is self-contained apart from mc_physics.py and mc_pool_config.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import qmc

from mc_physics import (
    add_rms_noise_numpy,
    scaled_forward_observation_numpy,
    valid_parameter_mask_numpy,
)
from mc_pool_config import DEFAULT_PHYSICS
from exp36_forward64 import forward64_batched, scaled_forward_observation_float64


PARAM_NAMES = ("a1", "a2", "a3", "m", "gamma")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def forward_batched(parameters: np.ndarray, integration_points: int, batch_size: int = 768) -> np.ndarray:
    parts = []
    for start in range(0, len(parameters), batch_size):
        parts.append(
            scaled_forward_observation_numpy(
                parameters[start:start + batch_size],
                integration_points=integration_points,
            )
        )
    return np.concatenate(parts, axis=0)


def rms_rows(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2, axis=1))


def normalize_params(a1: np.ndarray, m: np.ndarray, gamma: np.ndarray, cfg: dict) -> np.ndarray:
    pr = cfg["parameter_ranges"]
    a1n = (a1 - pr["a1"][0]) / (pr["a1"][1] - pr["a1"][0])
    mn = (m - pr["m"][0]) / (pr["m"][1] - pr["m"][0])
    lg = np.log10(gamma)
    lo, hi = np.log10(pr["gamma"][0]), np.log10(pr["gamma"][1])
    gn = (lg - lo) / (hi - lo)
    return np.stack([a1n, mn, gn], axis=1)


def generate_candidates(cfg: dict, count: int, seed: int) -> np.ndarray:
    pr = cfg["parameter_ranges"]
    fixed = cfg["fixed_parameters"]

    # Sobol gives much better 3D space filling than plain pseudo-random draws.
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    # Generate a power-of-two Sobol block and truncate it. This preserves the
    # balance properties and avoids SciPy's non-power-of-two warning.
    m2 = int(math.ceil(math.log2(max(int(count), 2))))
    u = sampler.random_base2(m2)[:count]
    a1 = pr["a1"][0] + u[:, 0] * (pr["a1"][1] - pr["a1"][0])
    m = pr["m"][0] + u[:, 1] * (pr["m"][1] - pr["m"][0])
    lg0, lg1 = math.log10(pr["gamma"][0]), math.log10(pr["gamma"][1])
    gamma = 10.0 ** (lg0 + u[:, 2] * (lg1 - lg0))

    p = np.column_stack([
        a1,
        np.full(count, float(fixed["a2"])),
        np.full(count, float(fixed["a3"])),
        m,
        gamma,
    ]).astype(np.float64)

    # Explicit anchors make the documented full parameter range reproducible.
    anchors = np.array([
        [pr["a1"][0], fixed["a2"], fixed["a3"], 0.5*(pr["m"][0]+pr["m"][1]), math.sqrt(pr["gamma"][0]*pr["gamma"][1])],
        [pr["a1"][1], fixed["a2"], fixed["a3"], 0.5*(pr["m"][0]+pr["m"][1]), math.sqrt(pr["gamma"][0]*pr["gamma"][1])],
        [0.5*(pr["a1"][0]+pr["a1"][1]), fixed["a2"], fixed["a3"], pr["m"][0], math.sqrt(pr["gamma"][0]*pr["gamma"][1])],
        [0.5*(pr["a1"][0]+pr["a1"][1]), fixed["a2"], fixed["a3"], pr["m"][1], math.sqrt(pr["gamma"][0]*pr["gamma"][1])],
        [0.5*(pr["a1"][0]+pr["a1"][1]), fixed["a2"], fixed["a3"], 0.5*(pr["m"][0]+pr["m"][1]), pr["gamma"][0]],
        [0.5*(pr["a1"][0]+pr["a1"][1]), fixed["a2"], fixed["a3"], 0.5*(pr["m"][0]+pr["m"][1]), pr["gamma"][1]],
    ], dtype=np.float64)
    p[:len(anchors)] = anchors

    valid = valid_parameter_mask_numpy(p)
    if not np.all(valid):
        bad = int(np.sum(~valid))
        raise RuntimeError(f"{bad} generated candidate states violate the physics validity mask")
    return p


def make_profile_bank(cfg: dict, integration_points: int):
    """Build a float64 unit-resonance bank on (m, log10 gamma)."""
    pr = cfg["parameter_ranges"]
    fixed = cfg["fixed_parameters"]
    pg = cfg["profile_grid"]
    m_grid = np.linspace(pr["m"][0], pr["m"][1], int(pg["m_points"]), dtype=np.float64)
    lg_grid = np.linspace(math.log10(pr["gamma"][0]), math.log10(pr["gamma"][1]), int(pg["log_gamma_points"]), dtype=np.float64)
    mm, ll = np.meshgrid(m_grid, lg_grid, indexing="ij")
    gamma = 10.0 ** ll.ravel()

    # a1=0 gives the smooth background; it is independent of m and gamma.
    background_p = np.array([[0.0, fixed["a2"], fixed["a3"], 0.5*(pr["m"][0]+pr["m"][1]), 0.5]], dtype=np.float64)
    background = forward64_batched(background_p, integration_points)[0]

    profile_p = np.column_stack([
        np.ones(mm.size, dtype=np.float64),
        np.full(mm.size, fixed["a2"], dtype=np.float64),
        np.full(mm.size, fixed["a3"], dtype=np.float64),
        mm.ravel(),
        gamma,
    ]).astype(np.float64)
    g1 = forward64_batched(profile_p, integration_points, batch_size=512)
    resonance = g1 - background[None, :]
    r2 = np.einsum("ij,ij->i", resonance, resonance, dtype=np.float64)
    if np.any(~np.isfinite(r2)) or np.any(r2 <= 0):
        raise RuntimeError("profile bank contains invalid/zero resonance vectors")
    return m_grid, lg_grid, background.astype(np.float64), resonance.astype(np.float64), r2.astype(np.float64)


def _best_region(
    dot: np.ndarray,
    y2: float,
    r2: np.ndarray,
    resonance_shape: tuple[int, int],
    m_grid: np.ndarray,
    lg_grid: np.ndarray,
    a1_bounds: tuple[float, float],
    region: str,
    true_a1: float,
    true_m: float,
    true_lg: float,
    tol: dict,
):
    """Approximate best grid point for one unacceptable region in float64."""
    amin, amax = a1_bounds
    M, G = resonance_shape
    dot = np.asarray(dot, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    free_a = np.clip(dot / r2, amin, amax)
    sse_free = y2 + free_a * free_a * r2 - 2.0 * free_a * dot
    # Roundoff at exact/near-exact fits can make the algebraic expression tiny
    # negative.  This value is only used for ranking; the winner is verified by
    # an explicit residual vector below.
    sse_free = np.maximum(sse_free, 0.0)

    if region == "a1_low":
        hi = true_a1 - float(tol["a1_abs"])
        if hi < amin:
            return math.inf, -1, math.nan
        a = np.clip(dot / r2, amin, min(hi, amax))
        sse = np.maximum(y2 + a*a*r2 - 2*a*dot, 0.0)
        idx = int(np.argmin(sse))
        return float(sse[idx]), idx, float(a[idx])

    if region == "a1_high":
        lo = true_a1 + float(tol["a1_abs"])
        if lo > amax:
            return math.inf, -1, math.nan
        a = np.clip(dot / r2, max(lo, amin), amax)
        sse = np.maximum(y2 + a*a*r2 - 2*a*dot, 0.0)
        idx = int(np.argmin(sse))
        return float(sse[idx]), idx, float(a[idx])

    mat = sse_free.reshape(M, G)
    if region == "m_low":
        lim = true_m - float(tol["m_abs"])
        ids = np.flatnonzero(m_grid <= lim + 1e-14)
        if len(ids) == 0: return math.inf, -1, math.nan
        sub = mat[:ids[-1] + 1, :]
        local = int(np.argmin(sub)); i, j = np.unravel_index(local, sub.shape)
    elif region == "m_high":
        lim = true_m + float(tol["m_abs"])
        ids = np.flatnonzero(m_grid >= lim - 1e-14)
        if len(ids) == 0: return math.inf, -1, math.nan
        i0 = int(ids[0]); sub = mat[i0:, :]
        local = int(np.argmin(sub)); ii, j = np.unravel_index(local, sub.shape); i = i0 + ii
    elif region == "gamma_low":
        lim = true_lg - math.log10(float(tol["gamma_factor"]))
        ids = np.flatnonzero(lg_grid <= lim + 1e-14)
        if len(ids) == 0: return math.inf, -1, math.nan
        sub = mat[:, :ids[-1] + 1]
        local = int(np.argmin(sub)); i, j = np.unravel_index(local, sub.shape)
    elif region == "gamma_high":
        lim = true_lg + math.log10(float(tol["gamma_factor"]))
        ids = np.flatnonzero(lg_grid >= lim - 1e-14)
        if len(ids) == 0: return math.inf, -1, math.nan
        j0 = int(ids[0]); sub = mat[:, j0:]
        local = int(np.argmin(sub)); i, jj = np.unravel_index(local, sub.shape); j = j0 + jj
    else:
        raise ValueError(region)

    idx = int(i * G + j)
    return float(sse_free[idx]), idx, float(free_a[idx])


def _direct_residual_sse(target_y: np.ndarray, rr: np.ndarray, a1: float) -> float:
    resid = np.asarray(target_y, dtype=np.float64) - float(a1) * np.asarray(rr, dtype=np.float64)
    return float(np.dot(resid, resid))


def _best_a1_and_sse(target_y: np.ndarray, rr: np.ndarray, bounds: tuple[float, float]) -> tuple[float, float]:
    rr = np.asarray(rr, dtype=np.float64)
    rr2 = float(np.dot(rr, rr))
    if not np.isfinite(rr2) or rr2 <= 0:
        return math.nan, math.inf
    aa = float(np.dot(np.asarray(target_y, dtype=np.float64), rr) / rr2)
    aa = min(max(aa, float(bounds[0])), float(bounds[1]))
    return aa, _direct_residual_sse(target_y, rr, aa)


def _grid_region_winners_for_state(
    target_y: np.ndarray,
    true_a1: float,
    true_m: float,
    true_lg: float,
    m_grid: np.ndarray,
    lg_grid: np.ndarray,
    resonance: np.ndarray,
    r2: np.ndarray,
    a1_bounds: tuple[float, float],
    tol: dict,
):
    """Return direct-residual-verified winner for every valid OR region."""
    dot = np.asarray(resonance, dtype=np.float64) @ np.asarray(target_y, dtype=np.float64)
    y2 = float(np.dot(target_y, target_y))
    M, G = len(m_grid), len(lg_grid)
    regions = ("a1_low","a1_high","m_low","m_high","gamma_low","gamma_high")
    winners = []
    for region in regions:
        sse_formula, idx, aa = _best_region(
            dot, y2, r2, (M,G), m_grid, lg_grid, a1_bounds, region,
            true_a1, true_m, true_lg, tol,
        )
        if idx < 0 or not np.isfinite(sse_formula):
            continue
        i, j = divmod(idx, G)
        sse_direct = _direct_residual_sse(target_y, resonance[idx], aa)
        winners.append({
            "region": region,
            "idx": int(idx), "i": int(i), "j": int(j),
            "a1": float(aa), "m": float(m_grid[i]), "lg": float(lg_grid[j]),
            "sse_formula": float(sse_formula), "sse_direct": float(sse_direct),
        })
    return winners


def profile_alias_margins(
    clean_g64: np.ndarray,
    params: np.ndarray,
    cfg: dict,
    background: np.ndarray,
    m_grid: np.ndarray,
    lg_grid: np.ndarray,
    resonance: np.ndarray,
    r2: np.ndarray,
    tol: dict,
    *,
    chunk_size: int = 128,
):
    """Profile unacceptable aliases using float64 ranking + direct verification."""
    ref_noise = float(cfg["reference_noise"])
    pr = cfg["parameter_ranges"]
    a1_bounds = tuple(float(x) for x in pr["a1"])
    M, G = len(m_grid), len(lg_grid)
    regions = ("a1_low","a1_high","m_low","m_high","gamma_low","gamma_high")

    n = len(params)
    margin = np.full(n, np.inf, dtype=np.float64)
    margin_formula = np.full(n, np.inf, dtype=np.float64)
    alias_a1 = np.full(n, np.nan, dtype=np.float64)
    alias_m = np.full(n, np.nan, dtype=np.float64)
    alias_gamma = np.full(n, np.nan, dtype=np.float64)
    trigger = np.empty(n, dtype=object)
    trigger[:] = ""
    formula_direct_margin_delta = np.full(n, np.nan, dtype=np.float64)

    R = np.asarray(resonance, dtype=np.float64)
    RT = R.T
    g_rms = rms_rows(clean_g64)
    y_all = np.asarray(clean_g64, dtype=np.float64) - np.asarray(background, dtype=np.float64)[None, :]

    for start in range(0, n, int(chunk_size)):
        stop = min(start + int(chunk_size), n)
        y = y_all[start:stop]
        dots = y @ RT
        y2s = np.einsum("ij,ij->i", y, y, dtype=np.float64)

        for bi in range(stop-start):
            gi = start + bi
            true_a1 = float(params[gi,0]); true_m = float(params[gi,3]); true_lg = math.log10(float(params[gi,4]))
            formula_candidates = []
            for region in regions:
                sf, idx, aa = _best_region(
                    dots[bi], float(y2s[bi]), r2, (M,G), m_grid, lg_grid,
                    a1_bounds, region, true_a1, true_m, true_lg, tol,
                )
                if idx >= 0 and np.isfinite(sf):
                    formula_candidates.append((float(sf), int(idx), float(aa), region))
            if not formula_candidates:
                continue

            # Verify every region winner from an explicit float64 residual vector;
            # do not use the cancellation-prone quadratic expression as the saved result.
            verified = []
            for sf, idx, aa, region in formula_candidates:
                sd = _direct_residual_sse(y[bi], R[idx], aa)
                verified.append((sd, sf, idx, aa, region))
            sd, sf, idx, aa, region = min(verified, key=lambda x: x[0])
            i, j = divmod(idx, G)
            denom = max(ref_noise * float(g_rms[gi]), 1e-30)
            m_direct = math.sqrt(max(sd,0.0)/clean_g64.shape[1]) / denom
            m_formula = math.sqrt(max(sf,0.0)/clean_g64.shape[1]) / denom
            margin[gi] = m_direct
            margin_formula[gi] = m_formula
            formula_direct_margin_delta[gi] = m_formula - m_direct
            alias_a1[gi] = aa; alias_m[gi] = m_grid[i]; alias_gamma[gi] = 10.0**lg_grid[j]; trigger[gi] = region

        if start == 0 or stop == n or (start // int(chunk_size)) % 20 == 0:
            print(f"  profiled aliases: {stop}/{n}")

    return {
        "margin": margin,
        "margin_formula": margin_formula,
        "formula_direct_margin_delta": formula_direct_margin_delta,
        "alias_a1": alias_a1,
        "alias_m": alias_m,
        "alias_gamma": alias_gamma,
        "trigger": trigger,
    }

def symmetric_g_distance(g1: np.ndarray, g2: np.ndarray, rms1: np.ndarray, rms2: float, ref_noise: float) -> np.ndarray:
    d = np.sqrt(np.mean((g1.astype(np.float64) - g2.astype(np.float64))**2, axis=1))
    scale = ref_noise * np.sqrt(np.maximum(rms1 * float(rms2), 1e-60))
    return d / np.maximum(scale, 1e-30)


def farthest_select(
    eligible: np.ndarray,
    params_norm: np.ndarray,
    clean_g: np.ndarray,
    margins: np.ndarray,
    target_count: int,
    min_g_sep: float,
    ref_noise: float,
):
    ids = np.asarray(eligible, dtype=np.int64)
    if len(ids) < target_count:
        return None

    g_rms_all = rms_rows(clean_g)
    selected: list[int] = []
    available = np.ones(len(ids), dtype=bool)
    min_param = np.full(len(ids), np.inf, dtype=np.float64)

    # Start at the safest candidate. This is deterministic.
    first_local = int(np.nanargmax(margins[ids]))
    first = int(ids[first_local])
    selected.append(first)
    available[first_local] = False

    while len(selected) < target_count:
        last = selected[-1]
        pd = np.linalg.norm(params_norm[ids] - params_norm[last][None,:], axis=1)
        min_param = np.minimum(min_param, pd)

        gd = symmetric_g_distance(
            clean_g[ids], clean_g[last],
            g_rms_all[ids], g_rms_all[last], ref_noise
        )
        available &= gd >= float(min_g_sep) - 1e-12
        if not np.any(available):
            return None

        # Farthest parameter coverage first; alias margin breaks close ties.
        score = min_param.copy()
        finite_m = margins[ids]
        if np.any(np.isfinite(finite_m)):
            rank_bonus = np.nan_to_num(finite_m / max(np.nanmax(finite_m), 1e-30), nan=0.0)
            score += 1e-3 * rank_bonus
        score[~available] = -np.inf
        loc = int(np.argmax(score))
        selected.append(int(ids[loc]))
        available[loc] = False

    return np.asarray(selected, dtype=np.int64)


def choose_alias_cutoff_and_select(params_norm, clean_g, margins, cfg):
    target = int(cfg["selected_count"])
    ref_noise = float(cfg["reference_noise"])
    min_sep = float(cfg["min_selected_g_separation_rms_snr"])
    finite = margins[np.isfinite(margins)]
    if len(finite) < target:
        raise RuntimeError("fewer finite recovery-alias margins than selected_count")

    # Candidate thresholds from desired retention factors, high to low.
    factors = (2,3,4,5,6,8,10,12,16,20,30,50)
    thresholds = []
    for factor in factors:
        keep = min(len(finite), target * factor)
        if keep >= target:
            thresholds.append(float(np.partition(finite, len(finite)-keep)[len(finite)-keep]))
    thresholds.extend([0.5,0.25,0.10,0.05,0.025,0.01,0.0])
    thresholds = sorted(set(max(0.0,t) for t in thresholds), reverse=True)

    scan = []
    chosen = None
    chosen_cutoff = None
    for threshold in thresholds:
        eligible = np.flatnonzero(margins >= threshold)
        row = {
            "recovery_alias_cutoff": threshold,
            "margin_basis": "verified_grid_float64_direct_residual",
            "eligible_count": int(len(eligible)),
            "eligible_fraction": float(len(eligible)/len(margins)),
        }
        if len(eligible) >= target:
            sel = farthest_select(
                eligible, params_norm, clean_g, margins,
                target, min_sep, ref_noise
            )
            row["selection_success"] = int(sel is not None)
            row["selected_count_if_success"] = int(target if sel is not None else 0)
            if sel is not None and chosen is None:
                chosen = sel
                chosen_cutoff = threshold
        else:
            row["selection_success"] = 0
            row["selected_count_if_success"] = 0
        scan.append(row)

    if chosen is None:
        raise RuntimeError(
            "Could not select the requested number of states while satisfying "
            f"g-separation >= {min_sep}. Lower min separation or enlarge candidate pool."
        )
    return chosen, float(chosen_cutoff), scan


def refine_selected_continuous(
    selected_ids: np.ndarray,
    params: np.ndarray,
    clean_g64: np.ndarray,
    cfg: dict,
    background: np.ndarray,
    m_grid: np.ndarray,
    lg_grid: np.ndarray,
    resonance: np.ndarray,
    r2: np.ndarray,
    grid_diag: dict,
    integration_points: int,
):
    """Conservative continuous refinement of selected-state alias margins.

    For every OR region we start from its direct-verified grid winner and from
    the closest legal boundary point.  Interpolation is used only to *propose*
    an optimum.  Every proposed winner is re-scored with the direct float64
    forward operator.  The verified grid winner is always kept as a candidate,
    so the final refined margin is guaranteed not to exceed the verified grid
    margin except for machine epsilon.
    """
    pr = cfg["parameter_ranges"]
    tol = cfg["recovery_tolerances"]
    ref_noise = float(cfg["reference_noise"])
    amin, amax = map(float, pr["a1"])
    lgmin, lgmax = math.log10(pr["gamma"][0]), math.log10(pr["gamma"][1])
    R = np.asarray(resonance, dtype=np.float64)
    interp = RegularGridInterpolator(
        (m_grid, lg_grid), R.reshape(len(m_grid), len(lg_grid), -1),
        method="linear", bounds_error=False, fill_value=None,
    )
    fixed = cfg["fixed_parameters"]
    opt_cfg = cfg.get("numerics", {})
    maxiter = int(opt_cfg.get("continuous_maxiter", 80))
    consistency_tol = float(opt_cfg.get("grid_refined_consistency_margin_tol", 1e-9))

    def direct_rr(mv: float, lgv: float) -> np.ndarray:
        pp = np.array([[1.0, fixed["a2"], fixed["a3"], mv, 10.0**lgv]], dtype=np.float64)
        return scaled_forward_observation_float64(pp, integration_points=integration_points)[0] - background

    def region_specs(true_a1, true_m, true_lg):
        specs=[]
        if true_a1-tol["a1_abs"] >= amin:
            specs.append(("a1_low",(amin,true_a1-tol["a1_abs"]),(pr["m"][0],pr["m"][1]),(lgmin,lgmax)))
        if true_a1+tol["a1_abs"] <= amax:
            specs.append(("a1_high",(true_a1+tol["a1_abs"],amax),(pr["m"][0],pr["m"][1]),(lgmin,lgmax)))
        if true_m-tol["m_abs"] >= pr["m"][0]:
            specs.append(("m_low",(amin,amax),(pr["m"][0],true_m-tol["m_abs"]),(lgmin,lgmax)))
        if true_m+tol["m_abs"] <= pr["m"][1]:
            specs.append(("m_high",(amin,amax),(true_m+tol["m_abs"],pr["m"][1]),(lgmin,lgmax)))
        d=math.log10(float(tol["gamma_factor"]))
        if true_lg-d >= lgmin:
            specs.append(("gamma_low",(amin,amax),(pr["m"][0],pr["m"][1]),(lgmin,true_lg-d)))
        if true_lg+d <= lgmax:
            specs.append(("gamma_high",(amin,amax),(pr["m"][0],pr["m"][1]),(true_lg+d,lgmax)))
        return specs

    rows=[]
    for rank,sid in enumerate(selected_ids):
        p=params[sid]
        true_a1,true_m,true_gam=float(p[0]),float(p[3]),float(p[4]); true_lg=math.log10(true_gam)
        target_y=np.asarray(clean_g64[sid],dtype=np.float64)-background
        target_rms=float(rms_rows(clean_g64[sid:sid+1])[0])
        denom=max(ref_noise*target_rms,1e-30)

        grid_winners=_grid_region_winners_for_state(
            target_y,true_a1,true_m,true_lg,m_grid,lg_grid,R,r2,(amin,amax),tol
        )
        if not grid_winners:
            raise RuntimeError(f"state {sid}: no valid unacceptable alias region")
        specs={x[0]:x[1:] for x in region_specs(true_a1,true_m,true_lg)}

        # Direct-verified global grid winner.
        gw=min(grid_winners,key=lambda x:x["sse_direct"])
        grid_sse=float(gw["sse_direct"])
        grid_margin=math.sqrt(max(grid_sse,0.0)/clean_g64.shape[1])/denom
        # Cross-check against the all-candidate profile saved before selection.
        profile_grid_margin=float(grid_diag["margin"][sid])
        if abs(grid_margin-profile_grid_margin) > max(1e-10,1e-7*max(grid_margin,profile_grid_margin,1.0)):
            raise RuntimeError(
                f"state {sid}: selected-state grid recomputation disagrees with profile "
                f"({grid_margin} vs {profile_grid_margin})"
            )

        candidates=[]
        for w in grid_winners:
            region=w["region"]
            if region not in specs: continue
            a_bounds,m_bounds,lg_bounds=specs[region]
            # Always preserve direct-verified grid candidate.
            candidates.append((w["sse_direct"],w["a1"],w["m"],10.0**w["lg"],region,"verified_grid"))

            def interp_eval(x):
                rr=np.asarray(interp([[float(x[0]),float(x[1])]])[0],dtype=np.float64)
                return _best_a1_and_sse(target_y,rr,a_bounds)[1]

            boundary=np.array([
                min(max(true_m,float(m_bounds[0])),float(m_bounds[1])),
                min(max(true_lg,float(lg_bounds[0])),float(lg_bounds[1])),
            ],dtype=np.float64)
            grid_seed=np.array([w["m"],w["lg"]],dtype=np.float64)
            seeds=[grid_seed,boundary]
            # A midpoint gives a cheap third basin probe when the seeds differ.
            if np.linalg.norm(grid_seed-boundary)>1e-12:
                seeds.append(0.5*(grid_seed+boundary))

            for seed in seeds:
                opt=minimize(
                    interp_eval,x0=seed,method="L-BFGS-B",
                    bounds=[tuple(map(float,m_bounds)),tuple(map(float,lg_bounds))],
                    options={"maxiter":maxiter,"ftol":1e-15,"gtol":1e-10},
                )
                mv,lgv=float(opt.x[0]),float(opt.x[1])
                rr=direct_rr(mv,lgv)
                aa,sse=_best_a1_and_sse(target_y,rr,a_bounds)
                if np.isfinite(sse):
                    candidates.append((sse,aa,mv,10.0**lgv,region,"continuous_direct_verified"))

        best=min(candidates,key=lambda x:x[0])
        sse,aa,mm,gg,region,source=best
        refined_margin=math.sqrt(max(float(sse),0.0)/clean_g64.shape[1])/denom
        violation=refined_margin-grid_margin
        if violation > consistency_tol:
            raise RuntimeError(
                f"state {sid}: refined margin {refined_margin} > verified grid margin {grid_margin}"
            )
        # Explicitly cap microscopic positive roundoff, never a substantive difference.
        refined_margin=min(refined_margin,grid_margin)
        rows.append({
            "state_id":int(sid),
            "recovery_alias_margin_grid_verified":float(grid_margin),
            "recovery_alias_margin_grid_formula":float(grid_diag["margin_formula"][sid]),
            "grid_alias_trigger_verified":gw["region"],
            "grid_alias_a1_verified":gw["a1"],
            "grid_alias_m_verified":gw["m"],
            "grid_alias_gamma_verified":10.0**gw["lg"],
            "recovery_alias_margin_refined":float(refined_margin),
            "refined_minus_grid_margin":float(refined_margin-grid_margin),
            "alias_source_refined":source,
            "alias_trigger_refined":region,
            "alias_a1_refined":float(aa),
            "alias_m_refined":float(mm),
            "alias_gamma_refined":float(gg),
            "alias_delta_a1_refined":abs(float(aa)-true_a1),
            "alias_delta_m_refined":abs(float(mm)-true_m),
            "alias_gamma_factor_refined":max(float(gg)/true_gam,true_gam/float(gg)),
        })
        if (rank+1)%25==0 or rank+1==len(selected_ids):
            print(f"  continuous refinement: {rank+1}/{len(selected_ids)}")
    return rows

def pairwise_selected_distance(clean: np.ndarray, ref_noise: float) -> np.ndarray:
    n = len(clean)
    r = rms_rows(clean)
    D = np.full((n,n), np.inf, dtype=np.float64)
    for i in range(n):
        diff = clean - clean[i:i+1]
        d = rms_rows(diff)
        denom = ref_noise * np.sqrt(np.maximum(r*r[i],1e-60))
        D[i,:] = d / np.maximum(denom,1e-30)
        D[i,i] = np.inf
    return D


def split_states(selected_ids, selected_params_norm, selected_clean, cfg, seed):
    """Split physical states without leakage.

    The hard coverage constraint is imposed in normalized parameter space.
    g-space holdout->train distance is still reported, but kept as a preferred
    diagnostic rather than a hard requirement; making it a hard upper bound
    can make a well-separated selected bank impossible to split.
    """
    counts = cfg["split_counts"]
    n_train, n_val, n_test = int(counts["train"]), int(counts["val"]), int(counts["test"])
    if n_train+n_val+n_test != len(selected_ids):
        raise ValueError("split counts must sum to selected_count")

    Dg = pairwise_selected_distance(selected_clean, float(cfg["reference_noise"]))
    Dp = np.sqrt(np.sum((selected_params_norm[:,None,:]-selected_params_norm[None,:,:])**2,axis=2))
    np.fill_diagonal(Dp,np.inf)
    max_param_cover = float(cfg.get("max_holdout_to_train_parameter_distance",0.35))
    n_hold = n_val+n_test
    H: list[int] = []
    available = list(range(len(selected_ids)))

    for _ in range(n_hold):
        best_i, best_score = None, -np.inf
        for c in available:
            proposed = set(H+[c])
            train_set = [j for j in range(len(selected_ids)) if j not in proposed]
            if not train_set:
                continue
            # Every holdout keeps at least one reasonably nearby training state
            # in normalized (a1,m,log-gamma) parameter space.
            if any(float(np.min(Dp[h,train_set])) > max_param_cover + 1e-12 for h in proposed):
                continue
            # Make the holdout bank itself broad in parameter space.
            score = 1.0 if not H else float(np.min(np.linalg.norm(selected_params_norm[c]-selected_params_norm[H],axis=1)))
            if score > best_score:
                best_score, best_i = score, c
        if best_i is None:
            raise RuntimeError(
                f"Could only choose {len(H)} holdout states with normalized-parameter "
                f"holdout->train <= {max_param_cover}. Increase "
                "max_holdout_to_train_parameter_distance slightly."
            )
        H.append(best_i)
        available.remove(best_i)

    train_local = np.array([i for i in range(len(selected_ids)) if i not in set(H)], dtype=int)
    H = np.array(H,dtype=int)

    # Search deterministic partitions of the broad holdout bank and maximize
    # the worst normalized span over a1,m,log(gamma).
    rng = np.random.default_rng(seed)
    best = None
    full_span = np.ptp(selected_params_norm,axis=0)
    full_span = np.where(full_span>0,full_span,1.0)
    for _ in range(4000):
        perm = rng.permutation(H)
        test = perm[:n_test]
        val = perm[n_test:]
        test_span = np.ptp(selected_params_norm[test],axis=0)/full_span
        val_span = np.ptp(selected_params_norm[val],axis=0)/full_span
        score = min(float(np.min(test_span)), float(np.min(val_span)))
        if best is None or score > best[0]:
            best = (score,val.copy(),test.copy(),val_span,test_span)
    _, val_local, test_local, val_span, test_span = best

    hold_local=np.concatenate([val_local,test_local])
    param_hold_to_train=np.min(Dp[np.ix_(hold_local,train_local)],axis=1)
    g_hold_to_train=np.min(Dg[np.ix_(hold_local,train_local)],axis=1)
    return {
        "train_local": train_local,
        "val_local": val_local,
        "test_local": test_local,
        "distance_matrix": Dg,
        "parameter_distance_matrix": Dp,
        "val_span_fraction": val_span,
        "test_span_fraction": test_span,
        "max_holdout_to_train_parameter_distance": float(np.max(param_hold_to_train)),
        "max_holdout_to_train_rms_snr": float(np.max(g_hold_to_train)),
    }

def noise_tag(level: float) -> str:
    if abs(level) < 1e-15:
        return "noise_0pct"
    pct = 100.0*level
    text = ("%g" % pct).replace(".","p")
    return f"noise_{text}pct"


def reps_for_level(cfg, level: float, split: str) -> int:
    key = "0" if abs(level)<1e-15 else ("%g" % level)
    table = cfg["noise_repetitions"]
    if key not in table:
        # tolerate JSON formatting differences such as "0.0020"
        key = min(table.keys(), key=lambda k: abs(float(k)-level))
    return int(table[key][split])


def save_split_npz(path: Path, base_params: np.ndarray, base_clean: np.ndarray, state_ids: np.ndarray, level: float, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    if abs(level) < 1e-15:
        reps = 1
    idx = np.repeat(np.arange(len(base_params)), reps)
    p = base_params[idx]
    gc = base_clean[idx].astype(np.float32)
    if level > 0:
        gy, noise = add_rms_noise_numpy(gc, level, rng)
    else:
        gy = gc.copy()
        noise = np.zeros_like(gc,dtype=np.float32)
    ids = state_ids[idx].astype(np.int64)
    np.savez_compressed(
        path,
        gy=gy.astype(np.float32),
        g_clean=gc,
        noise=noise.astype(np.float32),
        parameters=p.astype(np.float32),
        a1=p[:,0].astype(np.float32),
        a2=p[:,1].astype(np.float32),
        a3=p[:,2].astype(np.float32),
        m=p[:,3].astype(np.float32),
        gamma=p[:,4].astype(np.float32),
        state_id=ids,
        noise_level=np.full(len(p),float(level),dtype=np.float32),
    )


def coverage_radius(candidate_norm: np.ndarray, selected_norm: np.ndarray, chunk=2048) -> float:
    worst = 0.0
    for s in range(0,len(candidate_norm),chunk):
        x = candidate_norm[s:s+chunk]
        d2 = np.sum((x[:,None,:]-selected_norm[None,:,:])**2,axis=2)
        worst = max(worst,float(np.sqrt(np.min(d2,axis=1)).max()))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp36_3p_config.json")
    ap.add_argument("--output-dir", default="data_exp36_3p")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-continuous-refine", action="store_true")
    args = ap.parse_args()

    cfg = load_json(Path(args.config))
    if args.quick:
        cfg = json.loads(json.dumps(cfg))
        cfg["candidate_count"] = 2500
        cfg["profile_grid"]["m_points"] = 31
        cfg["profile_grid"]["log_gamma_points"] = 31
        cfg["selected_count"] = 36
        cfg["split_counts"] = {"train":24,"val":6,"test":6}
        cfg["min_selected_g_separation_rms_snr"] = min(1.0,float(cfg["min_selected_g_separation_rms_snr"]))
        cfg["max_holdout_to_train_parameter_distance"] = max(0.80,float(cfg["max_holdout_to_train_parameter_distance"]))
        cfg["max_holdout_to_train_rms_snr_preferred"] = max(10.0,float(cfg["max_holdout_to_train_rms_snr_preferred"]))
        for k in cfg["noise_repetitions"]:
            cfg["noise_repetitions"][k] = {"train":4,"val":3,"test":3}

    out = Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    integ = int(cfg["physics_integration_points"])
    print("="*100)
    print("Exp36 recovery-aligned 3P data build")
    print("output:",out)
    print("candidate_count:",cfg["candidate_count"])
    print("recovery tolerances:",cfg["recovery_tolerances"])
    print("="*100)

    params = generate_candidates(cfg,int(cfg["candidate_count"]),int(cfg["candidate_seed"]))
    clean = forward_batched(params,integ)
    # Alias diagnostics use a float64 mirror of exactly the same physical formula.
    clean64 = forward64_batched(params,integ)
    pnorm = normalize_params(params[:,0],params[:,3],params[:,4],cfg)

    print("[1/6] profile bank")
    mgrid,lggrid,background,R,r2 = make_profile_bank(cfg,integ)
    np.savez_compressed(out/"profile_bank.npz",m_grid=mgrid,log_gamma_grid=lggrid,background=background,resonance=R)

    print("[2/6] recovery-aligned profile over all candidates")
    rec = profile_alias_margins(clean64,params,cfg,background,mgrid,lggrid,R,r2,cfg["recovery_tolerances"])
    print("[3/6] global/remote diagnostic profile")
    glob = profile_alias_margins(clean64,params,cfg,background,mgrid,lggrid,R,r2,cfg["global_alias_diagnostic"])

    selected_ids, grid_cutoff, scan = choose_alias_cutoff_and_select(pnorm,clean,rec["margin"],cfg)
    write_csv(out/"alias_threshold_scan.csv",scan)

    print("[4/6] continuous refinement of final selected states")
    if args.skip_continuous_refine:
        refine = [{
            "state_id":int(i),
            "recovery_alias_margin_grid_verified":float(rec["margin"][i]),
            "recovery_alias_margin_grid_formula":float(rec["margin_formula"][i]),
            "grid_alias_trigger_verified":str(rec["trigger"][i]),
            "grid_alias_a1_verified":float(rec["alias_a1"][i]),
            "grid_alias_m_verified":float(rec["alias_m"][i]),
            "grid_alias_gamma_verified":float(rec["alias_gamma"][i]),
            "recovery_alias_margin_refined":float(rec["margin"][i]),
            "refined_minus_grid_margin":0.0,
            "alias_source_refined":"verified_grid",
            "alias_trigger_refined":str(rec["trigger"][i]),
            "alias_a1_refined":float(rec["alias_a1"][i]),
            "alias_m_refined":float(rec["alias_m"][i]),
            "alias_gamma_refined":float(rec["alias_gamma"][i]),
            "alias_delta_a1_refined":abs(float(rec["alias_a1"][i]-params[i,0])),
            "alias_delta_m_refined":abs(float(rec["alias_m"][i]-params[i,3])),
            "alias_gamma_factor_refined":max(float(rec["alias_gamma"][i]/params[i,4]),float(params[i,4]/rec["alias_gamma"][i])),
        } for i in selected_ids]
    else:
        refine = refine_selected_continuous(
            selected_ids,params,clean64,cfg,background,mgrid,lggrid,R,r2,rec,integ
        )
    refine_by_id={int(r["state_id"]):r for r in refine}

    selected_params=params[selected_ids]
    selected_clean=clean[selected_ids]
    selected_norm=pnorm[selected_ids]

    print("[5/6] physical-state split")
    split=split_states(selected_ids,selected_norm,selected_clean,cfg,int(cfg["candidate_seed"])+77)
    local_by_split={
        "train":split["train_local"],
        "val":split["val_local"],
        "test":split["test_local"],
    }

    # Main CSVs.
    candidate_rows=[]
    for i in range(len(params)):
        candidate_rows.append({
            "state_id":i,"a1":params[i,0],"a2":params[i,1],"a3":params[i,2],"m":params[i,3],"gamma":params[i,4],
            "recovery_alias_margin_grid_verified":rec["margin"][i],
            "recovery_alias_margin_grid_formula":rec["margin_formula"][i],
            "grid_formula_minus_verified_margin":rec["formula_direct_margin_delta"][i],
            "recovery_alias_trigger_grid":rec["trigger"][i],
            "recovery_alias_a1_grid":rec["alias_a1"][i],
            "recovery_alias_m_grid":rec["alias_m"][i],
            "recovery_alias_gamma_grid":rec["alias_gamma"][i],
            "global_alias_margin_grid_verified":glob["margin"][i],
            "global_alias_margin_grid_formula":glob["margin_formula"][i],
            "global_alias_trigger_grid":glob["trigger"][i],
        })
    write_csv(out/"candidate_states.csv",candidate_rows)

    selected_rows=[]
    split_of={}
    for sp,locs in local_by_split.items():
        for loc in locs: split_of[int(selected_ids[loc])]=sp
    for rank,sid in enumerate(selected_ids):
        rr=refine_by_id[int(sid)]
        selected_rows.append({
            "selected_rank":rank,"state_id":int(sid),"split":split_of[int(sid)],
            "a1":params[sid,0],"a2":params[sid,1],"a3":params[sid,2],"m":params[sid,3],"gamma":params[sid,4],
            "recovery_alias_margin_grid_verified":rec["margin"][sid],
            "recovery_alias_margin_grid_formula":rec["margin_formula"][sid],
            "global_alias_margin_grid_verified":glob["margin"][sid],
            **{k:v for k,v in rr.items() if k!="state_id"},
        })
    write_csv(out/"selected_states.csv",selected_rows)
    write_csv(out/"profiled_alias_selected.csv",refine)

    split_rows=[]
    for sp,locs in local_by_split.items():
        for loc in locs:
            sid=int(selected_ids[loc])
            split_rows.append({"state_id":sid,"split":sp,"a1":params[sid,0],"a2":params[sid,1],"a3":params[sid,2],"m":params[sid,3],"gamma":params[sid,4]})
    write_csv(out/"split_manifest.csv",split_rows)

    # Separation/coverage.
    D=split["distance_matrix"]
    tri=D[np.triu_indices(len(selected_ids),1)]
    train=split["train_local"]; val=split["val_local"]; test=split["test_local"]
    h=np.concatenate([val,test])
    hold_to_train=np.min(D[np.ix_(h,train)],axis=1)
    param_hold_to_train=float(split["max_holdout_to_train_parameter_distance"])
    sep_rows=[
        {"metric":"min_selected_g_separation_rms_snr","value":float(np.min(tri))},
        {"metric":"max_holdout_to_train_rms_snr","value":float(np.max(hold_to_train))},
        {"metric":"max_holdout_to_train_parameter_distance","value":param_hold_to_train},
        {"metric":"min_train_val_rms_snr","value":float(np.min(D[np.ix_(train,val)]))},
        {"metric":"min_train_test_rms_snr","value":float(np.min(D[np.ix_(train,test)]))},
        {"metric":"min_val_test_rms_snr","value":float(np.min(D[np.ix_(val,test)]))},
    ]
    write_csv(out/"split_separation_summary.csv",sep_rows)

    cover_radius=coverage_radius(pnorm,selected_norm)
    ranges={}
    for name,col in [("a1",0),("m",3),("gamma",4)]:
        vals=selected_params[:,col]
        ranges[name]={"min":float(vals.min()),"max":float(vals.max())}
    coverage_rows=[
        {"metric":"candidate_count","value":len(params)},
        {"metric":"selected_count","value":len(selected_ids)},
        {"metric":"parameter_cover_radius","value":cover_radius},
        {"metric":"val_a1_span_fraction","value":float(split["val_span_fraction"][0])},
        {"metric":"val_m_span_fraction","value":float(split["val_span_fraction"][1])},
        {"metric":"val_loggamma_span_fraction","value":float(split["val_span_fraction"][2])},
        {"metric":"test_a1_span_fraction","value":float(split["test_span_fraction"][0])},
        {"metric":"test_m_span_fraction","value":float(split["test_span_fraction"][1])},
        {"metric":"test_loggamma_span_fraction","value":float(split["test_span_fraction"][2])},
    ]
    write_csv(out/"coverage_summary.csv",coverage_rows)

    print("[6/6] noisy NPZ datasets + validation")
    for ni,level in enumerate(cfg["noise_levels"]):
        level=float(level)
        ndir=out/noise_tag(level)
        ndir.mkdir(parents=True,exist_ok=True)
        for si,(sp,locs) in enumerate(local_by_split.items()):
            reps=reps_for_level(cfg,level,sp)
            save_split_npz(
                ndir/f"{sp}.npz",
                selected_params[locs],selected_clean[locs],selected_ids[locs],
                level,reps,int(cfg["candidate_seed"])+10000*ni+100*si
            )

    # Direct-forward integrity.
    recalc=forward_batched(selected_params,integ)
    direct_rel=np.linalg.norm(recalc.astype(np.float64)-selected_clean.astype(np.float64))/max(np.linalg.norm(selected_clean.astype(np.float64)),1e-30)

    refined_margins=np.array([r["recovery_alias_margin_refined"] for r in refine],dtype=float)
    verified_grid_margins=np.array([r["recovery_alias_margin_grid_verified"] for r in refine],dtype=float)
    refined_scan=[]
    for th in (1.0,0.75,0.5,0.25,0.10,0.05,0.025,0.01,0.0):
        refined_scan.append({
            "refined_recovery_alias_cutoff":th,
            "selected_count_ge_cutoff":int(np.sum(refined_margins>=th)),
            "selected_fraction_ge_cutoff":float(np.mean(refined_margins>=th)),
        })
    write_csv(out/"refined_selected_margin_scan.csv",refined_scan)
    grid_refined_violation=refined_margins-verified_grid_margins
    # Float64 mirror must agree with the project's float32 forward after rounding.
    float64_to_project_rel=np.linalg.norm(clean64.astype(np.float32).astype(np.float64)-clean.astype(np.float64))/max(np.linalg.norm(clean.astype(np.float64)),1e-30)
    errors=[]; warnings=[]
    if len(selected_ids)!=int(cfg["selected_count"]): errors.append("selected_count mismatch")
    if float(np.min(tri))+1e-9 < float(cfg["min_selected_g_separation_rms_snr"]): errors.append("selected g separation below requirement")
    if param_hold_to_train-1e-9 > float(cfg["max_holdout_to_train_parameter_distance"]): errors.append("holdout->train normalized parameter cover exceeds requirement")
    preferred_g=float(cfg.get("max_holdout_to_train_rms_snr_preferred",math.inf))
    if float(np.max(hold_to_train)) > preferred_g: warnings.append("holdout->train g-space distance exceeds preferred diagnostic value; parameter-space coverage still passes")
    if float(np.min(split["test_span_fraction"])) < float(cfg["minimum_test_span_fraction"]): errors.append("test span fraction below requirement")
    if float(np.min(split["val_span_fraction"])) < float(cfg["minimum_val_span_fraction"]): warnings.append("val span fraction below preferred requirement")
    if direct_rel > 1e-10: errors.append("direct-forward integrity mismatch")
    if float64_to_project_rel > float(cfg.get("numerics",{}).get("forward64_roundtrip_relL2_max",1e-7)):
        errors.append("float64 diagnostic forward does not match project forward after float32 rounding")
    consistency_tol=float(cfg.get("numerics",{}).get("grid_refined_consistency_margin_tol",1e-9))
    if float(np.max(grid_refined_violation)) > consistency_tol:
        errors.append("refined alias margin exceeds verified grid margin; numerical consistency failure")
    if np.min(refined_margins) < 0.5*grid_cutoff and grid_cutoff>0:
        warnings.append("continuous refinement reduced some alias margins by >50% vs grid selection; grid cutoff is not a continuous-identifiability guarantee")
    if np.mean(refined_margins>=1.0)<0.5:
        warnings.append("fewer than 50% of selected states have recovery-aligned margin >= 1 reference-noise RMS; this is a physics/conditioning warning, not a data-integrity failure")

    val_rows=[
        {"check":"selected_count","status":"PASS" if len(selected_ids)==int(cfg["selected_count"]) else "FAIL","value":len(selected_ids)},
        {"check":"min_selected_g_separation","status":"PASS" if np.min(tri)>=float(cfg["min_selected_g_separation_rms_snr"])-1e-9 else "FAIL","value":float(np.min(tri))},
        {"check":"max_holdout_to_train_parameter_distance","status":"PASS" if param_hold_to_train<=float(cfg["max_holdout_to_train_parameter_distance"])+1e-9 else "FAIL","value":param_hold_to_train},
        {"check":"max_holdout_to_train_rms_snr","status":"INFO","value":float(np.max(hold_to_train))},
        {"check":"test_span_min","status":"PASS" if np.min(split["test_span_fraction"])>=float(cfg["minimum_test_span_fraction"]) else "FAIL","value":float(np.min(split["test_span_fraction"]))},
        {"check":"direct_forward_relL2","status":"PASS" if direct_rel<=1e-10 else "FAIL","value":direct_rel},
        {"check":"forward64_roundtrip_relL2","status":"PASS" if float64_to_project_rel<=float(cfg.get("numerics",{}).get("forward64_roundtrip_relL2_max",1e-7)) else "FAIL","value":float64_to_project_rel},
        {"check":"max_refined_minus_verified_grid_margin","status":"PASS" if np.max(grid_refined_violation)<=float(cfg.get("numerics",{}).get("grid_refined_consistency_margin_tol",1e-9)) else "FAIL","value":float(np.max(grid_refined_violation))},
        {"check":"recovery_margin_refined_min","status":"INFO","value":float(np.min(refined_margins))},
        {"check":"recovery_margin_refined_median","status":"INFO","value":float(np.median(refined_margins))},
        {"check":"fraction_refined_margin_ge_1","status":"INFO","value":float(np.mean(refined_margins>=1.0))},
    ]
    write_csv(out/"validation_summary.csv",val_rows)

    # Compact one-row summary for experiment bookkeeping.
    data_summary = [{
        "candidate_count": int(len(params)),
        "selected_count": int(len(selected_ids)),
        "train_state_count": int(len(train)),
        "val_state_count": int(len(val)),
        "test_state_count": int(len(test)),
        "grid_selection_cutoff_verified": float(grid_cutoff),
        "verified_grid_margin_min": float(np.min(verified_grid_margins)),
        "verified_grid_margin_median": float(np.median(verified_grid_margins)),
        "max_refined_minus_verified_grid_margin": float(np.max(grid_refined_violation)),
        "forward64_roundtrip_relL2": float(float64_to_project_rel),
        "refined_recovery_margin_min": float(np.min(refined_margins)),
        "refined_recovery_margin_median": float(np.median(refined_margins)),
        "refined_recovery_margin_p10": float(np.quantile(refined_margins,0.10)),
        "refined_recovery_margin_p90": float(np.quantile(refined_margins,0.90)),
        "fraction_refined_margin_ge_1": float(np.mean(refined_margins>=1.0)),
        "min_selected_g_separation_rms_snr": float(np.min(tri)),
        "max_holdout_to_train_parameter_distance": float(param_hold_to_train),
        "max_holdout_to_train_rms_snr": float(np.max(hold_to_train)),
        "parameter_cover_radius": float(cover_radius),
        "val_a1_span_fraction": float(split["val_span_fraction"][0]),
        "val_m_span_fraction": float(split["val_span_fraction"][1]),
        "val_loggamma_span_fraction": float(split["val_span_fraction"][2]),
        "test_a1_span_fraction": float(split["test_span_fraction"][0]),
        "test_m_span_fraction": float(split["test_span_fraction"][1]),
        "test_loggamma_span_fraction": float(split["test_span_fraction"][2]),
        "direct_forward_relL2": float(direct_rel),
        "validation_pass": int(len(errors)==0),
        "error_count": int(len(errors)),
        "warning_count": int(len(warnings)),
    }]
    write_csv(out/"exp36_data_summary.csv", data_summary)

    metadata={
        "experiment":cfg["experiment"],
        "mode":"a1mgamma",
        "free_params":["a1","m","gamma"],
        "fixed_parameters":cfg["fixed_parameters"],
        "parameter_ranges":cfg["parameter_ranges"],
        "recovery_tolerances":cfg["recovery_tolerances"],
        "global_alias_diagnostic":cfg["global_alias_diagnostic"],
        "reference_noise":cfg["reference_noise"],
        "candidate_count":len(params),
        "selected_count":len(selected_ids),
        "split_counts":{k:int(len(v)) for k,v in local_by_split.items()},
        "grid_selection_cutoff_verified":grid_cutoff,
        "alias_numerics_version":"exp36-v2-float64-direct-verified",
        "grid_cutoff_role":"preselection_only; continuous refined margins are the final identifiability audit",
        "numerics":cfg.get("numerics",{}),
        "forward64_roundtrip_relL2":float64_to_project_rel,
        "max_refined_minus_verified_grid_margin":float(np.max(grid_refined_violation)),
        "refined_recovery_margin":{"min":float(np.min(refined_margins)),"median":float(np.median(refined_margins)),"p10":float(np.quantile(refined_margins,.1)),"p90":float(np.quantile(refined_margins,.9)),"fraction_ge_1":float(np.mean(refined_margins>=1.0))},
        "selected_parameter_ranges":ranges,
        "parameter_cover_radius":cover_radius,
        "separation":{r["metric"]:r["value"] for r in sep_rows},
        "val_span_fraction":{"a1":float(split["val_span_fraction"][0]),"m":float(split["val_span_fraction"][1]),"loggamma":float(split["val_span_fraction"][2])},
        "test_span_fraction":{"a1":float(split["test_span_fraction"][0]),"m":float(split["test_span_fraction"][1]),"loggamma":float(split["test_span_fraction"][2])},
        "noise_levels":cfg["noise_levels"],
        "noise_repetitions":cfg["noise_repetitions"],
        "physics_integration_points":integ,
        "direct_forward_relL2":direct_rel,
        "validation":{"pass":len(errors)==0,"errors":errors,"warnings":warnings},
    }
    write_json(out/"metadata.json",metadata)

    # Hash all small audit artifacts and metadata.
    audit_files=["candidate_states.csv","selected_states.csv","profiled_alias_selected.csv","alias_threshold_scan.csv","refined_selected_margin_scan.csv","coverage_summary.csv","split_manifest.csv","split_separation_summary.csv","validation_summary.csv","exp36_data_summary.csv","metadata.json"]
    write_csv(out/"SHA256SUMS.csv",[{"file":f,"sha256":sha256(out/f)} for f in audit_files])

    print("="*100)
    print("DATA BUILD", "PASS" if not errors else "FAIL")
    print("selected:",len(selected_ids),"split:",{k:len(v) for k,v in local_by_split.items()})
    print("verified grid selection cutoff:",grid_cutoff)
    print("refined recovery margin min/median:",float(np.min(refined_margins)),float(np.median(refined_margins)))
    print("fraction refined margin >=1:",float(np.mean(refined_margins>=1.0)))
    print("min selected g separation:",float(np.min(tri)))
    print("max holdout->train parameter distance:",param_hold_to_train)
    print("max holdout->train g RMS-SNR (diagnostic):",float(np.max(hold_to_train)))
    print("direct forward relL2:",direct_rel)
    print("errors:",errors)
    print("warnings:",warnings)
    print("="*100)
    if errors:
        raise SystemExit(2)


if __name__=="__main__":
    main()
