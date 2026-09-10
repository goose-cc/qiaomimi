#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp37 data builder: continuous-identifiability-first 3P selection.

Pipeline:
1) 48k Sobol candidates and direct project/float64 forward checks.
2) Recovery-aligned grid alias profiling for every candidate.
3) Stratified prebank: keep the safest candidates in every parameter-space cell.
4) Continuous direct-verified alias refinement for the whole prebank.
5) Select train/val/test from *refined* margins under hard parameter coverage
   and holdout->train coverage constraints.
6) Generate 0%, 0.2%, 1% NPZ datasets after the physical-state split.

This is deliberately a data/physics experiment. It does not train a network.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

from mc_physics import add_rms_noise_numpy, scaled_forward_observation_numpy, valid_parameter_mask_numpy
from exp37_core import (
    forward64_batched,
    rms_rows,
    normalize_free_params,
    make_profile_bank,
    profile_alias_margins,
    refine_alias_ids,
    pairwise_g_distance,
    coverage_radius,
)


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
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
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


def generate_candidates(cfg: dict, count: int, seed: int) -> np.ndarray:
    pr = cfg["parameter_ranges"]
    fixed = cfg["fixed_parameters"]
    sampler = qmc.Sobol(d=3, scramble=True, seed=int(seed))
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

    # Reproducible anchors for full documented range.
    mid_a = 0.5 * (pr["a1"][0] + pr["a1"][1])
    mid_m = 0.5 * (pr["m"][0] + pr["m"][1])
    mid_g = math.sqrt(pr["gamma"][0] * pr["gamma"][1])
    anchors = np.array([
        [pr["a1"][0], fixed["a2"], fixed["a3"], mid_m, mid_g],
        [pr["a1"][1], fixed["a2"], fixed["a3"], mid_m, mid_g],
        [mid_a, fixed["a2"], fixed["a3"], pr["m"][0], mid_g],
        [mid_a, fixed["a2"], fixed["a3"], pr["m"][1], mid_g],
        [mid_a, fixed["a2"], fixed["a3"], mid_m, pr["gamma"][0]],
        [mid_a, fixed["a2"], fixed["a3"], mid_m, pr["gamma"][1]],
    ], dtype=np.float64)
    p[:len(anchors)] = anchors
    valid = valid_parameter_mask_numpy(p)
    if not bool(np.all(valid)):
        bad = int(np.sum(~valid))
        raise RuntimeError(f"{bad} generated candidates violate project physics validity")
    return p


def forward_project_batched(params: np.ndarray, integration_points: int, batch_size: int = 768):
    parts = []
    for start in range(0, len(params), int(batch_size)):
        parts.append(scaled_forward_observation_numpy(
            params[start:start+int(batch_size)], integration_points=integration_points
        ))
    return np.concatenate(parts, axis=0)


def stratified_prebank(params_norm: np.ndarray, grid_margin: np.ndarray, cfg: dict) -> np.ndarray:
    pcfg = cfg["continuous_prebank"]
    bins = np.asarray(pcfg["bins_per_axis"], dtype=int)
    per_cell = int(pcfg["top_per_cell"])
    max_states = int(pcfg["max_states"])
    if bins.shape != (3,) or np.any(bins < 1):
        raise ValueError("continuous_prebank.bins_per_axis must contain three positive integers")

    x = np.clip(np.asarray(params_norm, dtype=np.float64), 0.0, 1.0 - 1e-14)
    b = np.floor(x * bins[None, :]).astype(int)
    cell = (b[:, 0] * bins[1] + b[:, 1]) * bins[2] + b[:, 2]
    chosen = []
    for c in np.unique(cell):
        ids = np.flatnonzero(cell == c)
        order = ids[np.argsort(-np.nan_to_num(grid_margin[ids], nan=-np.inf))]
        chosen.extend(map(int, order[:per_cell]))
    chosen = list(dict.fromkeys(chosen))

    # If empty cells or invalid margins kept us below target, fill globally by
    # verified grid margin. This only supplements, never replaces stratification.
    if len(chosen) < max_states:
        mask = np.ones(len(params_norm), dtype=bool)
        mask[np.asarray(chosen, dtype=int)] = False
        rest = np.flatnonzero(mask)
        order = rest[np.argsort(-np.nan_to_num(grid_margin[rest], nan=-np.inf))]
        chosen.extend(map(int, order[:max_states-len(chosen)]))

    chosen = np.asarray(chosen[:max_states], dtype=np.int64)
    if len(chosen) < min(max_states, 64):
        raise RuntimeError("continuous prebank unexpectedly too small")
    return chosen


def axis_bin_counts(x: np.ndarray, bins: int) -> np.ndarray:
    xx = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0 - 1e-14)
    out = np.zeros((3, bins), dtype=int)
    for j in range(3):
        b = np.floor(xx[:, j] * bins).astype(int)
        out[j] = np.bincount(b, minlength=bins)[:bins]
    return out


def _threshold_candidates(margins: np.ndarray, cfg: dict, target: int) -> list[float]:
    finite = np.asarray(margins[np.isfinite(margins)], dtype=np.float64)
    vals = list(map(float, cfg["selection"].get("thresholds", [])))
    if len(finite):
        for factor in (2, 3, 4, 5, 6, 8, 10):
            keep = min(len(finite), target * factor)
            if keep >= target:
                vals.append(float(np.partition(finite, len(finite)-keep)[len(finite)-keep]))
    vals.append(0.0)
    return sorted(set(max(0.0, v) for v in vals), reverse=True)


def _select_train(eligible: np.ndarray, Dg: np.ndarray, Dp: np.ndarray,
                  margins: np.ndarray, params_norm: np.ndarray, n_train: int,
                  min_sep: float, max_hold_g: float, max_hold_p: float):
    ids = np.asarray(eligible, dtype=int)
    if len(ids) < n_train:
        return None

    pair_mask = (
        (Dg[np.ix_(ids, ids)] >= min_sep - 1e-9)
        & (Dg[np.ix_(ids, ids)] <= max_hold_g + 1e-9)
        & (Dp[np.ix_(ids, ids)] <= max_hold_p + 1e-9)
    )
    np.fill_diagonal(pair_mask, False)
    degree = pair_mask.sum(axis=1).astype(np.float64)
    degree /= max(float(degree.max()), 1.0)
    mm = np.asarray(margins[ids], dtype=np.float64)
    mm = mm / max(float(np.nanmax(mm)), 1e-30)

    # Start near the center among candidates that have holdout-neighbor support.
    center_d = np.linalg.norm(params_norm[ids] - 0.5, axis=1)
    start_score = degree + 0.02 * mm - 0.02 * center_d
    first_loc = int(np.argmax(start_score))

    selected_local = [first_loc]
    selected_ids = [int(ids[first_loc])]
    available = np.ones(len(ids), dtype=bool)
    available[first_loc] = False
    min_param = Dp[ids, selected_ids[-1]].astype(np.float64)

    while len(selected_ids) < n_train:
        last = selected_ids[-1]
        available &= Dg[ids, last] >= min_sep - 1e-9
        if not np.any(available):
            return None
        min_param = np.minimum(min_param, Dp[ids, last])
        score = min_param + 0.025 * degree + 0.01 * mm
        score[~available] = -np.inf
        loc = int(np.argmax(score))
        if not np.isfinite(score[loc]):
            return None
        selected_local.append(loc)
        selected_ids.append(int(ids[loc]))
        available[loc] = False
    return np.asarray(selected_ids, dtype=int)


def _select_holdouts(eligible: np.ndarray, train: np.ndarray, Dg: np.ndarray, Dp: np.ndarray,
                     params_norm: np.ndarray, n_hold: int, min_sep: float,
                     max_hold_g: float, max_hold_p: float):
    train_set = set(map(int, train))
    pool = np.asarray([int(i) for i in eligible if int(i) not in train_set], dtype=int)
    if len(pool) < n_hold:
        return None

    gt = Dg[np.ix_(pool, train)]
    pt = Dp[np.ix_(pool, train)]
    # Same training anchor must be reasonably close in both spaces.
    paired = np.any(
        (gt >= min_sep - 1e-9) & (gt <= max_hold_g + 1e-9) & (pt <= max_hold_p + 1e-9),
        axis=1,
    )
    # Final bank must also respect min separation to every training state.
    separated = np.min(gt, axis=1) >= min_sep - 1e-9
    pool = pool[paired & separated]
    if len(pool) < n_hold:
        return None

    selected = []
    available = np.ones(len(pool), dtype=bool)
    center_d = np.linalg.norm(params_norm[pool] - 0.5, axis=1)
    first_loc = int(np.argmax(center_d))
    selected.append(int(pool[first_loc]))
    available[first_loc] = False
    min_param = Dp[pool, selected[-1]].astype(np.float64)

    while len(selected) < n_hold:
        last = selected[-1]
        available &= Dg[pool, last] >= min_sep - 1e-9
        if not np.any(available):
            return None
        min_param = np.minimum(min_param, Dp[pool, last])
        score = min_param.copy()
        score[~available] = -np.inf
        loc = int(np.argmax(score))
        if not np.isfinite(score[loc]):
            return None
        selected.append(int(pool[loc]))
        available[loc] = False
    return np.asarray(selected, dtype=int)


def _partition_holdouts(holdouts: np.ndarray, params_norm: np.ndarray, n_val: int, n_test: int,
                        min_val_span: float, min_test_span: float, seed: int):
    if n_val + n_test != len(holdouts):
        raise ValueError("holdout partition size mismatch")
    rng = np.random.default_rng(int(seed))
    full_span = np.ptp(params_norm, axis=0)
    full_span = np.where(full_span > 0, full_span, 1.0)
    best = None
    for _ in range(6000):
        perm = rng.permutation(holdouts)
        val = perm[:n_val]
        test = perm[n_val:]
        vs = np.ptp(params_norm[val], axis=0) / full_span
        ts = np.ptp(params_norm[test], axis=0) / full_span
        score = min(float(np.min(vs)), float(np.min(ts)))
        if best is None or score > best[0]:
            best = (score, val.copy(), test.copy(), vs.copy(), ts.copy())
    score, val, test, vs, ts = best
    if float(np.min(vs)) + 1e-12 < min_val_span or float(np.min(ts)) + 1e-12 < min_test_span:
        return None
    return val, test, vs, ts


def choose_continuous_bank(prebank_ids: np.ndarray, prebank_margins: np.ndarray,
                           params_norm_all: np.ndarray, clean64_all: np.ndarray, cfg: dict):
    scfg = cfg["selection"]
    counts = scfg["split_counts"]
    n_train, n_val, n_test = int(counts["train"]), int(counts["val"]), int(counts["test"])
    target = n_train + n_val + n_test
    min_sep = float(scfg["min_selected_g_separation_rms_snr"])
    max_cover = float(scfg["max_parameter_cover_radius"])
    max_hold_p = float(scfg["max_holdout_to_train_parameter_distance"])
    max_hold_g = float(scfg["max_holdout_to_train_rms_snr_hard"])
    bins = int(scfg["coverage_axis_bins"])
    min_bin_count = int(scfg["minimum_train_count_per_axis_bin"])

    prebank_ids = np.asarray(prebank_ids, dtype=int)
    pnorm = params_norm_all[prebank_ids]
    clean = clean64_all[prebank_ids]
    Dp = np.linalg.norm(pnorm[:, None, :] - pnorm[None, :, :], axis=2).astype(np.float32)
    np.fill_diagonal(Dp, np.inf)
    print("  precomputing direct pairwise g distances for refined prebank...")
    Dg = pairwise_g_distance(clean, float(cfg["reference_noise"]))

    thresholds = _threshold_candidates(prebank_margins, cfg, target)
    scan = []
    for thr in thresholds:
        eligible_local = np.flatnonzero(prebank_margins >= thr - 1e-15)
        row = {
            "continuous_recovery_alias_cutoff": float(thr),
            "eligible_refined_prebank_count": int(len(eligible_local)),
            "eligible_fraction_of_prebank": float(len(eligible_local) / len(prebank_ids)),
            "selection_success": 0,
        }
        if len(eligible_local) < target:
            scan.append(row)
            continue

        train_local = _select_train(
            eligible_local, Dg, Dp, prebank_margins, pnorm,
            n_train, min_sep, max_hold_g, max_hold_p
        )
        if train_local is None:
            row["failure_reason"] = "train_selection"
            scan.append(row)
            continue

        train_global = prebank_ids[train_local]
        cover = coverage_radius(params_norm_all, params_norm_all[train_global])
        row["train_parameter_cover_radius"] = float(cover)
        counts_axis = axis_bin_counts(params_norm_all[train_global], bins)
        min_axis = int(counts_axis.min())
        row["minimum_train_count_in_any_axis_bin"] = min_axis
        if cover > max_cover + 1e-12:
            row["failure_reason"] = "parameter_cover_radius"
            scan.append(row)
            continue
        if min_axis < min_bin_count:
            row["failure_reason"] = "axis_bin_coverage"
            scan.append(row)
            continue

        hold_local = _select_holdouts(
            eligible_local, train_local, Dg, Dp, pnorm,
            n_val+n_test, min_sep, max_hold_g, max_hold_p
        )
        if hold_local is None:
            row["failure_reason"] = "holdout_selection"
            scan.append(row)
            continue

        part = _partition_holdouts(
            hold_local, pnorm, n_val, n_test,
            float(scfg["minimum_val_span_fraction"]),
            float(scfg["minimum_test_span_fraction"]),
            int(cfg["candidate_seed"]) + 137,
        )
        if part is None:
            row["failure_reason"] = "holdout_span"
            scan.append(row)
            continue
        val_local, test_local, val_span, test_span = part

        gt = Dg[np.ix_(np.concatenate([val_local, test_local]), train_local)]
        pt = Dp[np.ix_(np.concatenate([val_local, test_local]), train_local)]
        max_g = float(np.max(np.min(gt, axis=1)))
        max_p = float(np.max(np.min(pt, axis=1)))
        selected_local = np.concatenate([train_local, val_local, test_local])
        sub = Dg[np.ix_(selected_local, selected_local)].copy()
        np.fill_diagonal(sub, np.inf)
        min_final_g = float(np.min(sub))
        final_global = prebank_ids[selected_local]
        final_cover = coverage_radius(params_norm_all, params_norm_all[final_global])

        row.update({
            "selection_success": 1,
            "train_count": n_train, "val_count": n_val, "test_count": n_test,
            "final_parameter_cover_radius": float(final_cover),
            "max_holdout_to_train_parameter_distance": max_p,
            "max_holdout_to_train_rms_snr": max_g,
            "min_selected_g_separation_rms_snr": min_final_g,
            "val_span_min_fraction": float(np.min(val_span)),
            "test_span_min_fraction": float(np.min(test_span)),
        })
        scan.append(row)
        return {
            "train_local": train_local,
            "val_local": val_local,
            "test_local": test_local,
            "selected_local": selected_local,
            "selected_global": final_global,
            "train_global": prebank_ids[train_local],
            "val_global": prebank_ids[val_local],
            "test_global": prebank_ids[test_local],
            "Dg_prebank": Dg,
            "Dp_prebank": Dp,
            "chosen_cutoff": float(thr),
            "parameter_cover_radius": float(final_cover),
            "train_parameter_cover_radius": float(cover),
            "max_holdout_to_train_parameter_distance": max_p,
            "max_holdout_to_train_rms_snr": max_g,
            "min_selected_g_separation_rms_snr": min_final_g,
            "val_span_fraction": val_span,
            "test_span_fraction": test_span,
            "axis_bin_counts_train": counts_axis,
            "scan": scan,
        }

    return {"scan": scan, "failure": True}


def noise_tag(level: float) -> str:
    if abs(level) < 1e-15:
        return "noise_0pct"
    pct = 100.0 * level
    s = f"{pct:g}".replace(".", "p")
    return f"noise_{s}pct"


def reps_for_level(cfg: dict, level: float, split: str) -> int:
    table = cfg["noise_repetitions"]
    key = min(table.keys(), key=lambda k: abs(float(k) - level))
    return int(table[key][split])


def save_split_npz(path: Path, params: np.ndarray, clean: np.ndarray, state_ids: np.ndarray,
                   level: float, reps: int, seed: int):
    rng = np.random.default_rng(int(seed))
    if abs(level) < 1e-15:
        reps = 1
    idx = np.repeat(np.arange(len(params)), int(reps))
    p = params[idx]
    gc = clean[idx].astype(np.float32)
    if level > 0:
        gy, noise = add_rms_noise_numpy(gc, float(level), rng)
    else:
        gy = gc.copy()
        noise = np.zeros_like(gc, dtype=np.float32)
    ids = state_ids[idx].astype(np.int64)
    np.savez_compressed(
        path,
        gy=gy.astype(np.float32), g_clean=gc, noise=noise.astype(np.float32),
        parameters=p.astype(np.float32),
        a1=p[:,0].astype(np.float32), a2=p[:,1].astype(np.float32),
        a3=p[:,2].astype(np.float32), m=p[:,3].astype(np.float32),
        gamma=p[:,4].astype(np.float32), state_id=ids,
        noise_level=np.full(len(p), float(level), dtype=np.float32),
    )


def quick_cfg(cfg: dict) -> dict:
    q = json.loads(json.dumps(cfg))
    q["candidate_count"] = 2500
    q["profile_grid"] = {"m_points": 31, "log_gamma_points": 31}
    q["continuous_prebank"] = {
        "bins_per_axis": [4,4,4],
        "top_per_cell": 2,
        "max_states": 128,
        "continuous_maxiter": 24,
        "continuous_multistart": 2,
    }
    q["selection"]["selected_count"] = 36
    q["selection"]["split_counts"] = {"train":24,"val":6,"test":6}
    q["selection"]["min_selected_g_separation_rms_snr"] = 0.7
    q["selection"]["max_parameter_cover_radius"] = 0.9
    q["selection"]["max_holdout_to_train_parameter_distance"] = 0.9
    q["selection"]["max_holdout_to_train_rms_snr_hard"] = 20.0
    q["selection"]["max_holdout_to_train_rms_snr_preferred"] = 20.0
    q["selection"]["coverage_axis_bins"] = 4
    q["selection"]["minimum_train_count_per_axis_bin"] = 1
    q["selection"]["minimum_test_span_fraction"] = 0.35
    q["selection"]["minimum_val_span_fraction"] = 0.35
    for k in q["noise_repetitions"]:
        q["noise_repetitions"][k] = {"train":3,"val":2,"test":3}
    q["varpro"]["max_eval_repetitions_per_state"] = {"0":1,"0.002":2,"0.01":2}
    q["varpro"]["m_points"] = 61
    q["varpro"]["log_gamma_points"] = 61
    q["varpro"]["grid_seed_count_clean"] = 12
    q["varpro"]["grid_seed_count_noisy"] = 5
    q["varpro"]["continuous_maxiter"] = 20
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp37_3p_config.json")
    ap.add_argument("--output-dir", default="data_exp37_3p")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = load_json(Path(args.config))
    if args.quick:
        cfg = quick_cfg(cfg)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    integ = int(cfg["physics_integration_points"])
    print("="*100)
    print("EXP37 CONTINUOUS-IDENTIFIABILITY-FIRST 3P DATA BUILD")
    print("candidate_count:", cfg["candidate_count"])
    print("recovery tolerances:", cfg["recovery_tolerances"])
    print("="*100)

    print("[1/7] candidates + forward")
    params = generate_candidates(cfg, int(cfg["candidate_count"]), int(cfg["candidate_seed"]))
    clean_project = forward_project_batched(params, integ)
    clean64 = forward64_batched(params, integ)
    roundtrip = float(
        np.linalg.norm(clean64.astype(np.float32).astype(np.float64)-clean_project.astype(np.float64))
        / max(np.linalg.norm(clean_project.astype(np.float64)), 1e-30)
    )
    pnorm = normalize_free_params(params, cfg)

    print("[2/7] recovery-aligned verified grid profile on all candidates")
    mgrid, lggrid, background, R, r2 = make_profile_bank(cfg, integ)
    np.savez_compressed(out/"profile_bank.npz",
                        m_grid=mgrid, log_gamma_grid=lggrid,
                        background=background, resonance=R, r2=r2)
    grid = profile_alias_margins(
        clean64, params, cfg, background, mgrid, lggrid, R, r2,
        cfg["recovery_tolerances"],
    )
    write_csv(out/"candidate_states.csv", [
        {
            "state_id":int(i),
            "a1":params[i,0],"a2":params[i,1],"a3":params[i,2],
            "m":params[i,3],"gamma":params[i,4],
            "recovery_alias_margin_grid_verified":float(grid["margin"][i]),
            "recovery_alias_trigger_grid":str(grid["trigger"][i]),
            "recovery_alias_a1_grid":float(grid["alias_a1"][i]),
            "recovery_alias_m_grid":float(grid["alias_m"][i]),
            "recovery_alias_gamma_grid":float(grid["alias_gamma"][i]),
        }
        for i in range(len(params))
    ])

    print("[3/7] stratified continuous-refinement prebank")
    prebank_ids = stratified_prebank(pnorm, grid["margin"], cfg)
    print("  prebank states:", len(prebank_ids))
    print("[4/7] direct-verified continuous alias refinement of prebank")
    ccfg = cfg["continuous_prebank"]
    refined_rows = refine_alias_ids(
        prebank_ids, params, clean64, cfg, background, mgrid, lggrid, R, r2, grid, integ,
        maxiter=int(ccfg["continuous_maxiter"]),
        multistart=int(ccfg["continuous_multistart"]),
        progress_every=100 if not args.quick else 32,
    )
    refined_by_id = {int(r["state_id"]): r for r in refined_rows}
    refined_margins = np.array(
        [float(refined_by_id[int(i)]["recovery_alias_margin_refined"]) for i in prebank_ids],
        dtype=np.float64,
    )

    prebank_rows = []
    for rank, sid in enumerate(prebank_ids):
        rr = refined_by_id[int(sid)]
        prebank_rows.append({
            "prebank_rank": rank, "state_id": int(sid),
            "a1": params[sid,0], "a2": params[sid,1], "a3": params[sid,2],
            "m": params[sid,3], "gamma": params[sid,4],
            "recovery_alias_margin_grid_verified": float(grid["margin"][sid]),
            **{k:v for k,v in rr.items() if k!="state_id"},
        })
    write_csv(out/"continuous_prebank.csv", prebank_rows)

    print("[5/7] refined-margin selection with hard coverage constraints")
    selected = choose_continuous_bank(prebank_ids, refined_margins, pnorm, clean64, cfg)
    write_csv(out/"continuous_threshold_scan.csv", selected["scan"])
    if selected.get("failure"):
        write_json(out/"selection_failure.json", {
            "message": "No 260-state bank satisfied all hard continuous-margin/coverage constraints.",
            "suggestion": "Do not train. Inspect continuous_threshold_scan.csv; enlarge the prebank/candidate pool or redesign g only after diagnosing which hard constraint failed.",
        })
        raise RuntimeError(
            "Exp37 could not construct a coverage-controlled bank from continuously refined states. "
            "See continuous_threshold_scan.csv and selection_failure.json."
        )

    selected_ids = np.asarray(selected["selected_global"], dtype=int)
    train_ids = np.asarray(selected["train_global"], dtype=int)
    val_ids = np.asarray(selected["val_global"], dtype=int)
    test_ids = np.asarray(selected["test_global"], dtype=int)
    split_of = {int(i):"train" for i in train_ids}
    split_of.update({int(i):"val" for i in val_ids})
    split_of.update({int(i):"test" for i in test_ids})

    selected_rows = []
    for rank, sid in enumerate(selected_ids):
        rr = refined_by_id[int(sid)]
        selected_rows.append({
            "selected_rank": rank, "state_id": int(sid), "split": split_of[int(sid)],
            "a1": params[sid,0], "a2": params[sid,1], "a3": params[sid,2],
            "m": params[sid,3], "gamma": params[sid,4],
            "recovery_alias_margin_grid_verified": float(grid["margin"][sid]),
            **{k:v for k,v in rr.items() if k!="state_id"},
        })
    write_csv(out/"selected_states.csv", selected_rows)
    write_csv(out/"profiled_alias_selected.csv", [
        {"state_id": int(sid), **{k:v for k,v in refined_by_id[int(sid)].items() if k!="state_id"}}
        for sid in selected_ids
    ])

    split_rows = []
    for sid in selected_ids:
        split_rows.append({
            "state_id": int(sid), "split": split_of[int(sid)],
            "a1": params[sid,0], "a2": params[sid,1], "a3": params[sid,2],
            "m": params[sid,3], "gamma": params[sid,4],
        })
    write_csv(out/"split_manifest.csv", split_rows)

    print("[6/7] noise datasets after frozen physical-state split")
    clean32 = clean_project
    for li, level in enumerate(map(float, cfg["noise_levels"])):
        ndir = out/noise_tag(level)
        ndir.mkdir(exist_ok=True)
        for si, (sp, ids) in enumerate([("train",train_ids),("val",val_ids),("test",test_ids)]):
            save_split_npz(
                ndir/f"{sp}.npz", params[ids], clean32[ids], ids,
                level, reps_for_level(cfg, level, sp),
                int(cfg["candidate_seed"]) + 10000*li + 100*si + 17,
            )

    print("[7/7] summaries + integrity")
    final_margins = np.array(
        [float(refined_by_id[int(i)]["recovery_alias_margin_refined"]) for i in selected_ids],
        dtype=np.float64,
    )
    preferred = float(cfg["selection"]["max_holdout_to_train_rms_snr_preferred"])
    warnings = []
    if selected["max_holdout_to_train_rms_snr"] > preferred:
        warnings.append("holdout->train g coverage passes hard limit but exceeds preferred value")
    if np.median(final_margins) < 0.01:
        warnings.append("continuous recovery-aligned margin remains very small; inspect Jacobian and VarPro before any network training")

    cover_rows = [
        {"metric":"parameter_cover_radius", "value":selected["parameter_cover_radius"]},
        {"metric":"train_parameter_cover_radius", "value":selected["train_parameter_cover_radius"]},
        {"metric":"max_holdout_to_train_parameter_distance", "value":selected["max_holdout_to_train_parameter_distance"]},
        {"metric":"max_holdout_to_train_rms_snr", "value":selected["max_holdout_to_train_rms_snr"]},
        {"metric":"min_selected_g_separation_rms_snr", "value":selected["min_selected_g_separation_rms_snr"]},
        {"metric":"val_a1_span_fraction", "value":float(selected["val_span_fraction"][0])},
        {"metric":"val_m_span_fraction", "value":float(selected["val_span_fraction"][1])},
        {"metric":"val_loggamma_span_fraction", "value":float(selected["val_span_fraction"][2])},
        {"metric":"test_a1_span_fraction", "value":float(selected["test_span_fraction"][0])},
        {"metric":"test_m_span_fraction", "value":float(selected["test_span_fraction"][1])},
        {"metric":"test_loggamma_span_fraction", "value":float(selected["test_span_fraction"][2])},
    ]
    write_csv(out/"coverage_summary.csv", cover_rows)

    scan_final = []
    for t in [0.1,0.05,0.025,0.01,0.0075,0.005,0.003,0.002,0.001,0.0]:
        scan_final.append({
            "refined_margin_cutoff":t,
            "selected_count_ge_cutoff":int(np.sum(final_margins >= t)),
            "selected_fraction_ge_cutoff":float(np.mean(final_margins >= t)),
        })
    write_csv(out/"selected_refined_margin_scan.csv", scan_final)

    summary = [{
        "candidate_count":len(params),
        "prebank_count":len(prebank_ids),
        "selected_count":len(selected_ids),
        "train_state_count":len(train_ids),
        "val_state_count":len(val_ids),
        "test_state_count":len(test_ids),
        "chosen_continuous_recovery_alias_cutoff":selected["chosen_cutoff"],
        "refined_margin_min":float(np.min(final_margins)),
        "refined_margin_p10":float(np.quantile(final_margins,0.1)),
        "refined_margin_median":float(np.median(final_margins)),
        "refined_margin_p90":float(np.quantile(final_margins,0.9)),
        "refined_margin_max":float(np.max(final_margins)),
        "fraction_refined_margin_ge_0p01":float(np.mean(final_margins>=0.01)),
        "fraction_refined_margin_ge_0p1":float(np.mean(final_margins>=0.1)),
        "fraction_refined_margin_ge_1":float(np.mean(final_margins>=1.0)),
        "parameter_cover_radius":selected["parameter_cover_radius"],
        "train_parameter_cover_radius":selected["train_parameter_cover_radius"],
        "max_holdout_to_train_parameter_distance":selected["max_holdout_to_train_parameter_distance"],
        "max_holdout_to_train_rms_snr":selected["max_holdout_to_train_rms_snr"],
        "min_selected_g_separation_rms_snr":selected["min_selected_g_separation_rms_snr"],
        "forward64_roundtrip_relL2":roundtrip,
        "warning_count":len(warnings),
    }]
    write_csv(out/"exp37_data_summary.csv", summary)

    metadata = {
        "experiment":cfg["experiment"],
        "dataset_version":cfg["dataset_version"],
        "parameter_ranges":cfg["parameter_ranges"],
        "fixed_parameters":cfg["fixed_parameters"],
        "recovery_tolerances":cfg["recovery_tolerances"],
        "reference_noise":cfg["reference_noise"],
        "physics_integration_points":integ,
        "candidate_count":len(params),
        "prebank_count":len(prebank_ids),
        "selected_count":len(selected_ids),
        "split_counts":{"train":len(train_ids),"val":len(val_ids),"test":len(test_ids)},
        "chosen_continuous_recovery_alias_cutoff":selected["chosen_cutoff"],
        "coverage":{
            "parameter_cover_radius":selected["parameter_cover_radius"],
            "train_parameter_cover_radius":selected["train_parameter_cover_radius"],
            "max_holdout_to_train_parameter_distance":selected["max_holdout_to_train_parameter_distance"],
            "max_holdout_to_train_rms_snr":selected["max_holdout_to_train_rms_snr"],
            "min_selected_g_separation_rms_snr":selected["min_selected_g_separation_rms_snr"],
        },
        "forward64_roundtrip_relL2":roundtrip,
        "warnings":warnings,
        "config_snapshot":cfg,
    }
    write_json(out/"metadata.json", metadata)

    # Hash the small audit files (not the large NPZs/profile bank).
    audit = [
        "candidate_states.csv","continuous_prebank.csv","continuous_threshold_scan.csv","selected_states.csv",
        "profiled_alias_selected.csv","split_manifest.csv","coverage_summary.csv",
        "selected_refined_margin_scan.csv","exp37_data_summary.csv","metadata.json",
    ]
    write_csv(out/"SHA256SUMS.csv", [
        {"file":name,"sha256":sha256(out/name)} for name in audit if (out/name).exists()
    ])

    print("="*100)
    print("EXP37 DATA BUILD PASS")
    print("selected:",len(selected_ids),"split:",{"train":len(train_ids),"val":len(val_ids),"test":len(test_ids)})
    print("chosen CONTINUOUS cutoff:",selected["chosen_cutoff"])
    print("refined margin min/median:",float(np.min(final_margins)),float(np.median(final_margins)))
    print("parameter cover radius:",selected["parameter_cover_radius"])
    print("max holdout->train parameter distance:",selected["max_holdout_to_train_parameter_distance"])
    print("max holdout->train g RMS-SNR:",selected["max_holdout_to_train_rms_snr"])
    print("min selected g separation:",selected["min_selected_g_separation_rms_snr"])
    print("forward64 roundtrip relL2:",roundtrip)
    print("warnings:",warnings)
    print("="*100)


if __name__ == "__main__":
    main()
