#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the formal CP02 v1 machine-learning physical-state dataset.

Frozen scientific definition:
  * corrected m^2 resonance physics (rho-m2-center-v1)
  * q^2 design = hybrid_ultranear_240
  * identifiable iff D_M^continuous >= 1.5

The large candidate pool is screened with a cheap coarse alias grid.  That grid
is used only to reject impossible states and prioritize continuous refinement;
it is never used as the final acceptance definition.  Every newly accepted
physical state receives a continuous margin.  Existing CP02 continuously
refined prebank states can be reused as verified seeds when present.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

import cp02_core as core
from cp02_observation import design_by_id, load_config, make_q2


CACHE_FIELDS = [
    "refinement_signature", "physical_state_id", "state_id",
    "a1", "a2", "a3", "m", "gamma", "screen_grid_mahalanobis",
    "continuous_mahalanobis", "continuous_rms_snr", "trigger",
    "alias_a1", "alias_m", "alias_gamma",
]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def append_csv_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def stable_state_identity(params: np.ndarray, formula_version: str, design_id: str) -> tuple[str, int]:
    p = np.asarray(params, dtype="<f8").reshape(5)
    h = hashlib.blake2b(digest_size=16)
    h.update(formula_version.encode("utf-8"))
    h.update(b"\0")
    h.update(design_id.encode("utf-8"))
    h.update(b"\0")
    h.update(p.tobytes())
    digest = h.digest()
    physical_state_id = "cp02v1_" + digest.hex()
    state_id = int.from_bytes(digest[:8], "little", signed=False) & ((1 << 63) - 1)
    return physical_state_id, state_id


def frozen_definition_check(physics_cfg: dict, dataset_cfg: dict) -> tuple[str, float]:
    frozen = dataset_cfg["frozen_definition"]
    formula = str(frozen["physics_formula_version"])
    design_id = str(frozen["design_id"])
    threshold = float(frozen["continuous_threshold_mahalanobis"])
    if physics_cfg.get("physics_formula_version") != formula:
        raise RuntimeError(
            f"physics formula drift: expected {formula!r}, got {physics_cfg.get('physics_formula_version')!r}"
        )
    if abs(float(physics_cfg["reference_noise"]) - float(frozen["reference_noise"])) > 1e-15:
        raise RuntimeError("reference_noise drift from the frozen CP02 definition")
    tol = physics_cfg["recovery_tolerances"]
    ftol = frozen["recovery_tolerances"]
    for k in ("a1_abs", "m_abs", "gamma_factor"):
        if abs(float(tol[k]) - float(ftol[k])) > 1e-15:
            raise RuntimeError(f"recovery tolerance drift for {k}")
    design_by_id(physics_cfg, design_id)  # uniqueness/existence check
    if threshold != 1.5:
        raise RuntimeError("CP02 ML v1 is frozen at D_M^continuous >= 1.5")
    return design_id, threshold


def runtime_settings(physics_cfg: dict, dataset_cfg: dict, quick: bool, workers_override: int | None) -> dict[str, Any]:
    off = physics_cfg["official_bank"]
    r = dataset_cfg["refinement"]
    a = dataset_cfg["continuous_audit"]
    q = dataset_cfg.get("quick", {})
    if quick:
        return {
            "target": int(q["target_physical_states"]),
            "candidate_count": int(q["candidate_count"]),
            "candidate_forward_batch_size": int(q["candidate_forward_batch_size"]),
            "coarse_m_points": int(q["coarse_m_points"]),
            "coarse_g_points": int(q["coarse_log_gamma_points"]),
            "coarse_state_batch_size": int(q["coarse_state_batch_size"]),
            "official_m_points": int(q["official_m_points"]),
            "official_g_points": int(q["official_log_gamma_points"]),
            "top_regions": int(q["continuous_top_regions"]),
            "maxiter": int(q["continuous_maxiter"]),
            "multistart": int(q["continuous_multistart"]),
            "workers": int(workers_override if workers_override is not None else q["workers"]),
            "checkpoint_batch_size": int(q["checkpoint_batch_size"]),
            "audit_random_count": int(q["audit_random_count"]),
            "audit_boundary_count": int(q["audit_boundary_count"]),
            "audit_m_points": int(q["audit_m_points"]),
            "audit_g_points": int(q["audit_log_gamma_points"]),
            "audit_top_regions": int(q["audit_top_regions"]),
            "audit_maxiter": int(q["audit_maxiter"]),
            "audit_multistart": int(q["audit_multistart"]),
            "train_replicas": int(q["train_replicas"]),
        }
    return {
        "target": int(dataset_cfg["target_physical_states"]),
        "candidate_count": int(dataset_cfg["candidate_count"]),
        "candidate_forward_batch_size": int(dataset_cfg["candidate_forward_batch_size"]),
        "coarse_m_points": int(dataset_cfg["coarse_screen"]["m_points"]),
        "coarse_g_points": int(dataset_cfg["coarse_screen"]["log_gamma_points"]),
        "coarse_state_batch_size": int(dataset_cfg["coarse_screen"]["state_batch_size"]),
        "official_m_points": int(off["profile_m_points"]),
        "official_g_points": int(off["profile_log_gamma_points"]),
        "top_regions": int(off["continuous_top_regions"]),
        "maxiter": int(off["continuous_maxiter"]),
        "multistart": int(off["continuous_multistart"]),
        "workers": int(workers_override if workers_override is not None else r["workers"]),
        "checkpoint_batch_size": int(r["checkpoint_batch_size"]),
        "audit_random_count": int(a["random_count"]),
        "audit_boundary_count": int(a["boundary_count"]),
        "audit_m_points": int(a["m_points"]),
        "audit_g_points": int(a["log_gamma_points"]),
        "audit_top_regions": int(a["top_regions"]),
        "audit_maxiter": int(a["maxiter"]),
        "audit_multistart": int(a["multistart"]),
        "train_replicas": int(dataset_cfg["noise"]["train_replicas"]),
    }


def forward_batched(params: np.ndarray, q2: np.ndarray, physics_cfg: dict, batch_size: int) -> np.ndarray:
    parts = []
    for start in range(0, len(params), int(batch_size)):
        parts.append(core.forward(params[start:start + int(batch_size)], q2, physics_cfg["integration_points"]))
    if not parts:
        return np.empty((0, len(q2)), dtype=np.float64)
    return np.concatenate(parts, axis=0).astype(np.float64, copy=False)


def grid_margin_batch_fast(params: np.ndarray, gclean: np.ndarray, bank: dict, physics_cfg: dict) -> np.ndarray:
    """Exact CP02 grid margin, vectorized over a small state batch.

    The result matches grid_alias_one/grid_alias_batch up to floating roundoff,
    but computes the expensive R@y product once per state rather than once per
    unacceptable region.
    """
    p = np.asarray(params, dtype=np.float64)
    G = np.asarray(gclean, dtype=np.float64)
    y = G - bank["background"][None, :]
    R = bank["R"]
    r2 = bank["r2"]
    dot = y @ R.T
    y2 = np.einsum("ij,ij->i", y, y)[:, None]
    d = physics_cfg["regular_numerical_domain"]
    tol = physics_cfg["recovery_tolerances"]
    amin, amax = map(float, d["a1"])
    free = np.clip(dot / r2[None, :], amin, amax)
    base_sse = np.maximum(y2 + free * free * r2[None, :] - 2.0 * free * dot, 0.0)
    best = np.full(len(p), np.inf, dtype=np.float64)

    hi = p[:, 0] - float(tol["a1_abs"])
    valid = hi >= amin
    if np.any(valid):
        aa = np.minimum(np.maximum(dot[valid] / r2[None, :], amin), np.minimum(hi[valid, None], amax))
        sse = np.maximum(y2[valid] + aa * aa * r2[None, :] - 2.0 * aa * dot[valid], 0.0)
        best[valid] = np.minimum(best[valid], np.min(sse, axis=1))

    lo = p[:, 0] + float(tol["a1_abs"])
    valid = lo <= amax
    if np.any(valid):
        aa = np.minimum(np.maximum(dot[valid] / r2[None, :], np.maximum(lo[valid, None], amin)), amax)
        sse = np.maximum(y2[valid] + aa * aa * r2[None, :] - 2.0 * aa * dot[valid], 0.0)
        best[valid] = np.minimum(best[valid], np.min(sse, axis=1))

    masks = (
        bank["mm"][None, :] <= p[:, 3, None] - float(tol["m_abs"]) + 1e-12,
        bank["mm"][None, :] >= p[:, 3, None] + float(tol["m_abs"]) - 1e-12,
        bank["gg"][None, :] <= p[:, 4, None] / float(tol["gamma_factor"]) + 1e-12,
        bank["gg"][None, :] >= p[:, 4, None] * float(tol["gamma_factor"]) - 1e-12,
    )
    for mask in masks:
        region_sse = np.where(mask, base_sse, np.inf)
        best = np.minimum(best, np.min(region_sse, axis=1))

    sigma = float(physics_cfg["reference_noise"]) * np.maximum(core.rms_rows(G), 1e-30)
    return np.sqrt(np.maximum(best, 0.0)) / sigma


def screen_candidates(params: np.ndarray, q2: np.ndarray, physics_cfg: dict, bank: dict,
                      forward_batch: int, state_batch: int) -> np.ndarray:
    margins = np.empty(len(params), dtype=np.float64)
    for start in range(0, len(params), int(forward_batch)):
        stop = min(start + int(forward_batch), len(params))
        gc = core.forward(params[start:stop], q2, physics_cfg["integration_points"])
        for s2 in range(0, len(gc), int(state_batch)):
            e2 = min(s2 + int(state_batch), len(gc))
            margins[start + s2:start + e2] = grid_margin_batch_fast(
                params[start + s2:start + e2], gc[s2:e2], bank, physics_cfg
            )
        if stop == len(params) or stop % max(int(forward_batch) * 10, 1) == 0:
            print(f"  coarse grid screen: {stop}/{len(params)}")
    return margins


def grid_alias_winners_one_fast(p: np.ndarray, gclean: np.ndarray, bank: dict, physics_cfg: dict) -> list[dict[str, Any]]:
    tol = physics_cfg["recovery_tolerances"]
    d = physics_cfg["regular_numerical_domain"]
    y = np.asarray(gclean, dtype=np.float64) - bank["background"]
    R, r2 = bank["R"], bank["r2"]
    dot = R @ y
    y2 = float(y @ y)
    amin, amax = map(float, d["a1"])
    free = np.clip(dot / r2, amin, amax)
    base = np.maximum(y2 + free * free * r2 - 2.0 * free * dot, 0.0)
    wins: list[dict[str, Any]] = []

    for region in core.REGIONS:
        if region == "a1_low":
            hi = float(p[0]) - float(tol["a1_abs"])
            if hi < amin:
                continue
            aa = np.clip(dot / r2, amin, min(hi, amax))
            sse = np.maximum(y2 + aa * aa * r2 - 2.0 * aa * dot, 0.0)
            j = int(np.argmin(sse))
        elif region == "a1_high":
            lo = float(p[0]) + float(tol["a1_abs"])
            if lo > amax:
                continue
            aa = np.clip(dot / r2, max(lo, amin), amax)
            sse = np.maximum(y2 + aa * aa * r2 - 2.0 * aa * dot, 0.0)
            j = int(np.argmin(sse))
        else:
            aa = free
            sse = base
            if region == "m_low":
                mask = bank["mm"] <= float(p[3]) - float(tol["m_abs"]) + 1e-12
            elif region == "m_high":
                mask = bank["mm"] >= float(p[3]) + float(tol["m_abs"]) - 1e-12
            elif region == "gamma_low":
                mask = bank["gg"] <= float(p[4]) / float(tol["gamma_factor"]) + 1e-12
            else:
                mask = bank["gg"] >= float(p[4]) * float(tol["gamma_factor"]) - 1e-12
            if not np.any(mask):
                continue
            ids = np.flatnonzero(mask)
            j = int(ids[np.argmin(sse[ids])])
        resid = y - float(aa[j]) * R[j]
        sd = float(resid @ resid)
        wins.append({
            "region": region, "sse": sd, "a1": float(aa[j]),
            "m": float(bank["mm"][j]), "gamma": float(bank["gg"][j]), "idx": j,
        })

    wins.sort(key=lambda z: z["sse"])
    sigma = float(physics_cfg["reference_noise"]) * max(core.rms(gclean), 1e-30)
    nq = len(gclean)
    for w in wins:
        w["rms_snr"] = math.sqrt(max(float(w["sse"]), 0.0) / nq) / sigma
        w["mahalanobis"] = math.sqrt(max(float(w["sse"]), 0.0)) / sigma
    return wins


def refine_region_fast(p: np.ndarray, gclean: np.ndarray, gridwin: dict[str, Any], region: str,
                       physics_cfg: dict, q2: np.ndarray, bank: dict,
                       maxiter: int, multistart: int) -> dict[str, Any] | None:
    bounds = core._region_bounds(p, region, physics_cfg)
    if bounds is None:
        return None
    ab, mb, lb = bounds
    fixed = physics_cfg["fixed_parameters"]
    target = np.asarray(gclean, dtype=np.float64)
    bg = np.asarray(bank["background"], dtype=np.float64)
    y = target - bg

    def evalx(x: np.ndarray) -> tuple[float, float, float]:
        mv, lg = float(x[0]), float(x[1])
        gv = math.exp(lg)
        unit = np.array([[1.0, fixed["a2"], fixed["a3"], mv, gv]], dtype=np.float64)
        R = core.forward(unit, q2, physics_cfg["integration_points"])[0] - bg
        r2 = float(R @ R)
        aa = float(np.clip((y @ R) / max(r2, 1e-300), ab[0], ab[1]))
        resid = y - aa * R
        return float(resid @ resid), aa, gv

    starts = [
        [gridwin["m"], math.log(gridwin["gamma"])],
        [np.clip(p[3], *mb), np.clip(math.log(p[4]), *lb)],
        [mb[0], lb[0]], [mb[1], lb[1]], [mb[0], lb[1]], [mb[1], lb[0]],
    ]
    best = None
    for st in starts[:max(1, int(multistart))]:
        opt = minimize(
            lambda x: evalx(x)[0], np.asarray(st, dtype=np.float64),
            bounds=[tuple(mb), tuple(lb)], method="L-BFGS-B",
            options={"maxiter": int(maxiter), "ftol": 1e-15},
        )
        sd, aa, gv = evalx(opt.x)
        rec = {"region": region, "sse": sd, "a1": aa, "m": float(opt.x[0]), "gamma": gv}
        if best is None or sd < best["sse"]:
            best = rec
    return best


def continuous_alias_one_fast(p: np.ndarray, gclean: np.ndarray, bank: dict, physics_cfg: dict,
                              q2: np.ndarray, top_regions: int, maxiter: int,
                              multistart: int) -> dict[str, Any]:
    wins = grid_alias_winners_one_fast(p, gclean, bank, physics_cfg)
    if not wins:
        raise RuntimeError("no valid unacceptable alias region for continuous refinement")
    candidates: list[dict[str, Any]] = []
    for w in wins[:int(top_regions)]:
        refined = refine_region_fast(
            p, gclean, w, w["region"], physics_cfg, q2, bank,
            maxiter=int(maxiter), multistart=int(multistart),
        )
        if refined is not None:
            candidates.append(refined)
        candidates.append({k: w[k] for k in ("region", "sse", "a1", "m", "gamma")})
    best = min(candidates, key=lambda z: z["sse"])
    sigma = float(physics_cfg["reference_noise"]) * max(core.rms(gclean), 1e-30)
    best["rms_snr"] = math.sqrt(float(best["sse"]) / len(gclean)) / sigma
    best["mahalanobis"] = math.sqrt(float(best["sse"])) / sigma
    return best


def bin_indices_3d(params: np.ndarray, bins: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = [np.asarray(bins[k], dtype=np.float64) for k in ("a1", "m", "gamma")]
    cols = [0, 3, 4]
    ids = []
    for e, col in zip(edges, cols):
        b = np.searchsorted(e, params[:, col], side="right") - 1
        b = np.clip(b, 0, len(e) - 2)
        ids.append(b.astype(np.int32))
    ia, im, ig = ids
    cell = ia * ((len(edges[1]) - 1) * (len(edges[2]) - 1)) + im * (len(edges[2]) - 1) + ig
    return ia, im, ig, cell.astype(np.int32)


def interleaved_queue(indices: np.ndarray, params: np.ndarray, margins: np.ndarray,
                      bins: dict[str, Any], initial_cell_counts: dict[int, int] | None = None) -> np.ndarray:
    if len(indices) == 0:
        return np.empty(0, dtype=np.int64)
    _, _, _, cells = bin_indices_3d(params[indices], bins)
    per_cell: dict[int, list[int]] = {}
    for global_idx, cell in zip(indices.tolist(), cells.tolist()):
        per_cell.setdefault(int(cell), []).append(int(global_idx))
    for cell, ids in per_cell.items():
        ids.sort(key=lambda i: (-float(margins[i]), i))

    initial = initial_cell_counts or {}
    heap: list[tuple[int, float, int, int]] = []
    for cell in sorted(per_cell):
        i0 = per_cell[cell][0]
        heapq.heappush(heap, (int(initial.get(cell, 0)), -float(margins[i0]), cell, 0))
    order: list[int] = []
    while heap:
        level, neg_margin, cell, pos = heapq.heappop(heap)
        idx = per_cell[cell][pos]
        order.append(idx)
        nxt = pos + 1
        if nxt < len(per_cell[cell]):
            j = per_cell[cell][nxt]
            heapq.heappush(heap, (level + 1, -float(margins[j]), cell, nxt))
    return np.asarray(order, dtype=np.int64)


def load_verified_seed_states(dataset_cfg: dict, physics_cfg: dict, q2: np.ndarray, threshold: float,
                              formula_version: str, design_id: str, disabled: bool) -> list[dict[str, Any]]:
    vcfg = dataset_cfg.get("verified_seed", {})
    if disabled or not bool(vcfg.get("use_if_present", True)):
        return []
    csv_path = Path(vcfg["refined_prebank_csv"])
    q2_path = Path(vcfg["q2_npy"])
    if not csv_path.exists() and not q2_path.exists():
        print("No prior CP02 refined prebank found; building the ML dataset from new candidates only.")
        return []
    if not csv_path.exists() or not q2_path.exists():
        raise RuntimeError("verified_seed requires both refined_prebank_csv and q2_npy when either is present")
    prior_q2 = np.load(q2_path).astype(np.float64)
    if prior_q2.shape != q2.shape or not np.allclose(prior_q2, q2, rtol=0.0, atol=1e-12):
        raise RuntimeError("verified CP02 prebank q2 does not match frozen hybrid_ultranear_240")

    fixed = physics_cfg["fixed_parameters"]
    out: list[dict[str, Any]] = []
    for row in read_csv(csv_path):
        margin = float(row["continuous_mahalanobis"])
        if margin < threshold - 1e-12:
            continue
        p = np.array([
            float(row["a1"]), float(fixed["a2"]), float(fixed["a3"]),
            float(row["m"]), float(row["gamma"]),
        ], dtype=np.float64)
        pid, sid = stable_state_identity(p, formula_version, design_id)
        out.append({
            "physical_state_id": pid, "state_id": sid, "params": p,
            "screen_grid_mahalanobis": float(row.get("grid_mahalanobis", "nan")),
            "continuous_mahalanobis": margin,
            "continuous_rms_snr": float(row.get("continuous_rms_snr", "nan")),
            "trigger": row.get("trigger", ""),
            "alias_a1": float(row.get("alias_a1", "nan")),
            "alias_m": float(row.get("alias_m", "nan")),
            "alias_gamma": float(row.get("alias_gamma", "nan")),
            "source": "verified_cp02_prebank",
        })
    print(f"Loaded {len(out)} verified seed states with D_M^continuous >= {threshold:g}.")
    return out


def refinement_signature(physics_cfg: dict, design_id: str, q2: np.ndarray, settings: dict[str, Any]) -> str:
    payload = {
        "physics_formula_version": physics_cfg["physics_formula_version"],
        "design_id": design_id,
        "q2_sha256": hashlib.sha256(np.asarray(q2, dtype="<f8").tobytes()).hexdigest(),
        "recovery_tolerances": physics_cfg["recovery_tolerances"],
        "reference_noise": physics_cfg["reference_noise"],
        "integration_points": physics_cfg["integration_points"],
        "m_points": settings["official_m_points"],
        "log_gamma_points": settings["official_g_points"],
        "top_regions": settings["top_regions"],
        "maxiter": settings["maxiter"],
        "multistart": settings["multistart"],
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def load_refinement_cache(path: Path, signature: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    cache = {}
    for row in read_csv(path):
        if row.get("refinement_signature") == signature:
            cache[row["physical_state_id"]] = row
    return cache


def cache_row_to_record(row: dict[str, str]) -> dict[str, Any]:
    p = np.array([float(row["a1"]), float(row["a2"]), float(row["a3"]), float(row["m"]), float(row["gamma"])], dtype=np.float64)
    return {
        "physical_state_id": row["physical_state_id"], "state_id": int(row["state_id"]), "params": p,
        "screen_grid_mahalanobis": float(row["screen_grid_mahalanobis"]),
        "continuous_mahalanobis": float(row["continuous_mahalanobis"]),
        "continuous_rms_snr": float(row["continuous_rms_snr"]), "trigger": row["trigger"],
        "alias_a1": float(row["alias_a1"]), "alias_m": float(row["alias_m"]),
        "alias_gamma": float(row["alias_gamma"]), "source": "new_continuous_refinement",
    }


def record_to_cache_row(record: dict[str, Any], signature: str) -> dict[str, Any]:
    p = record["params"]
    return {
        "refinement_signature": signature, "physical_state_id": record["physical_state_id"],
        "state_id": record["state_id"], "a1": p[0], "a2": p[1], "a3": p[2], "m": p[3], "gamma": p[4],
        "screen_grid_mahalanobis": record["screen_grid_mahalanobis"],
        "continuous_mahalanobis": record["continuous_mahalanobis"],
        "continuous_rms_snr": record["continuous_rms_snr"], "trigger": record["trigger"],
        "alias_a1": record["alias_a1"], "alias_m": record["alias_m"], "alias_gamma": record["alias_gamma"],
    }


def refine_new_states(queue: np.ndarray, params: np.ndarray, screen_margin: np.ndarray,
                      accepted: list[dict[str, Any]], target: int, threshold: float,
                      physics_cfg: dict, q2: np.ndarray, bank: dict, settings: dict[str, Any],
                      formula_version: str, design_id: str, cache_path: Path, signature: str) -> tuple[list[dict[str, Any]], int]:
    cache = load_refinement_cache(cache_path, signature)
    print(f"Refinement cache: {len(cache)} matching rows")
    accepted_ids = {r["physical_state_id"] for r in accepted}
    refined_new = 0
    pos = 0
    batch_size = max(1, int(settings["checkpoint_batch_size"]))
    workers = max(1, int(settings["workers"]))

    while len(accepted) < target and pos < len(queue):
        ids = queue[pos:pos + batch_size]
        pos += len(ids)
        todo = []
        for idx in ids:
            p = params[int(idx)]
            pid, sid = stable_state_identity(p, formula_version, design_id)
            if pid in accepted_ids:
                continue
            cached = cache.get(pid)
            if cached is not None:
                rec = cache_row_to_record(cached)
                if not np.allclose(rec["params"], p, rtol=0.0, atol=1e-14):
                    raise RuntimeError(f"cache identity mismatch for {pid}")
                if rec["continuous_mahalanobis"] >= threshold - 1e-12:
                    accepted.append(rec)
                    accepted_ids.add(pid)
                continue
            todo.append((int(idx), pid, sid))

        if todo:
            batch_params = np.asarray([params[i] for i, _, _ in todo], dtype=np.float64)
            batch_clean = core.forward(batch_params, q2, physics_cfg["integration_points"])

            def work(j: int) -> dict[str, Any]:
                idx, pid, sid = todo[j]
                p = batch_params[j]
                gc = batch_clean[j]
                w = continuous_alias_one_fast(
                    p, gc, bank, physics_cfg, q2,
                    top_regions=settings["top_regions"], maxiter=settings["maxiter"],
                    multistart=settings["multistart"],
                )
                return {
                    "physical_state_id": pid, "state_id": sid, "params": p.copy(),
                    "screen_grid_mahalanobis": float(screen_margin[idx]),
                    "continuous_mahalanobis": float(w["mahalanobis"]),
                    "continuous_rms_snr": float(w["rms_snr"]), "trigger": str(w["region"]),
                    "alias_a1": float(w["a1"]), "alias_m": float(w["m"]), "alias_gamma": float(w["gamma"]),
                    "source": "new_continuous_refinement",
                }

            if workers == 1:
                rows = [work(j) for j in range(len(todo))]
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    rows = list(ex.map(work, range(len(todo))))
            refined_new += len(rows)
            append_csv_rows(cache_path, [record_to_cache_row(r, signature) for r in rows], CACHE_FIELDS)
            for rec in rows:
                cache[rec["physical_state_id"]] = {k: str(v) for k, v in record_to_cache_row(rec, signature).items()}
                if rec["continuous_mahalanobis"] >= threshold - 1e-12 and rec["physical_state_id"] not in accepted_ids:
                    accepted.append(rec)
                    accepted_ids.add(rec["physical_state_id"])

        print(
            f"  continuous refinement queue: attempted={pos}/{len(queue)} "
            f"newly_computed={refined_new} accepted={len(accepted)}/{target}"
        )

    if len(accepted) < target:
        raise RuntimeError(
            f"Only {len(accepted)} accepted physical states were found after exhausting {len(queue)} "
            "coarse-grid survivors. Increase candidate_count; do not lower the frozen D_M threshold."
        )
    return accepted[:target], refined_new


def choose_seed_subset(records: list[dict[str, Any]], target: int, bins: dict[str, Any]) -> list[dict[str, Any]]:
    if len(records) <= target:
        return records
    p = np.asarray([r["params"] for r in records], dtype=np.float64)
    m = np.asarray([r["continuous_mahalanobis"] for r in records], dtype=np.float64)
    order = interleaved_queue(np.arange(len(records), dtype=np.int64), p, m, bins)
    return [records[int(i)] for i in order[:target]]


def audit_continuous(records: list[dict[str, Any]], clean: np.ndarray, physics_cfg: dict, q2: np.ndarray,
                     dataset_cfg: dict, settings: dict[str, Any], threshold: float, workers: int,
                     out_path: Path) -> dict[str, Any]:
    n = len(records)
    boundary_n = min(int(settings["audit_boundary_count"]), n)
    random_n = min(int(settings["audit_random_count"]), max(0, n - boundary_n))
    margins = np.asarray([r["continuous_mahalanobis"] for r in records], dtype=np.float64)
    boundary = np.argsort(margins)[:boundary_n]
    remaining = np.setdiff1d(np.arange(n, dtype=np.int64), boundary, assume_unique=False)
    rng = np.random.default_rng(int(dataset_cfg["continuous_audit"]["seed"]))
    random_ids = rng.choice(remaining, size=random_n, replace=False) if random_n else np.empty(0, dtype=np.int64)
    audit_ids = np.concatenate([boundary.astype(np.int64), np.asarray(random_ids, dtype=np.int64)])
    kind = {int(i): "boundary" for i in boundary}
    for i in random_ids:
        kind[int(i)] = "random"
    if len(audit_ids) == 0:
        core.write_csv(out_path, [])
        return {"count": 0, "fail_count": 0, "min_audit_margin": math.nan}

    print(f"Building frozen-definition continuous re-audit bank for {len(audit_ids)} accepted states...")
    bank = core.build_profile_bank(
        physics_cfg, q2, physics_cfg["integration_points"],
        settings["audit_m_points"], settings["audit_g_points"],
    )

    def work(local_idx: int) -> dict[str, Any]:
        rec = records[local_idx]
        w = continuous_alias_one_fast(
            rec["params"], clean[local_idx], bank, physics_cfg, q2,
            top_regions=settings["audit_top_regions"], maxiter=settings["audit_maxiter"],
            multistart=settings["audit_multistart"],
        )
        return {
            "physical_state_id": rec["physical_state_id"], "state_id": rec["state_id"],
            "audit_kind": kind[int(local_idx)],
            "official_continuous_mahalanobis": rec["continuous_mahalanobis"],
            "audit_continuous_mahalanobis": float(w["mahalanobis"]),
            "audit_trigger": w["region"], "audit_alias_a1": w["a1"],
            "audit_alias_m": w["m"], "audit_alias_gamma": w["gamma"],
            "threshold": threshold,
            "audit_pass": int(float(w["mahalanobis"]) >= threshold - 1e-12),
        }

    if workers <= 1:
        rows = [work(int(i)) for i in audit_ids]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            rows = list(ex.map(lambda i: work(int(i)), audit_ids.tolist()))
    core.write_csv(out_path, rows)
    fails = [r for r in rows if not int(r["audit_pass"])]
    return {
        "count": len(rows), "fail_count": len(fails),
        "min_audit_margin": float(min(float(r["audit_continuous_mahalanobis"]) for r in rows)),
        "boundary_count": boundary_n, "random_count": random_n,
        "failed_physical_state_ids": [str(r["physical_state_id"]) for r in fails],
    }


def allocate_counts(capacity: np.ndarray, target: int) -> np.ndarray:
    cap = np.asarray(capacity, dtype=np.int64)
    if target < 0 or target > int(cap.sum()):
        raise ValueError("invalid stratified allocation target")
    if target == 0:
        return np.zeros_like(cap)
    ideal = cap.astype(np.float64) * (float(target) / float(cap.sum()))
    out = np.minimum(np.floor(ideal).astype(np.int64), cap)
    remain = int(target - out.sum())
    frac = ideal - np.floor(ideal)
    order = np.argsort(-frac, kind="stable")
    while remain > 0:
        progressed = False
        for j in order:
            if out[j] < cap[j]:
                out[j] += 1
                remain -= 1
                progressed = True
                if remain == 0:
                    break
        if not progressed:
            raise RuntimeError("could not complete stratified allocation")
    return out


def stratified_physical_split(params: np.ndarray, bins: dict[str, Any], split_cfg: dict[str, Any]) -> np.ndarray:
    n = len(params)
    train_f = float(split_cfg["train_fraction"])
    val_f = float(split_cfg["val_fraction"])
    test_f = float(split_cfg["test_fraction"])
    if abs(train_f + val_f + test_f - 1.0) > 1e-12:
        raise ValueError("split fractions must sum to 1")
    n_val = int(round(n * val_f))
    n_test = int(round(n * test_f))
    n_train = n - n_val - n_test
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("all physical splits must be non-empty")

    _, _, _, cell = bin_indices_3d(params, bins)
    unique = np.unique(cell)
    counts = np.asarray([np.sum(cell == c) for c in unique], dtype=np.int64)
    val_counts = allocate_counts(counts, n_val)
    remain_cap = counts - val_counts
    test_counts = allocate_counts(remain_cap, n_test)
    rng = np.random.default_rng(int(split_cfg["seed"]))
    split = np.empty(n, dtype="<U5")
    for c, nv, nt in zip(unique, val_counts, test_counts):
        ids = np.flatnonzero(cell == c)
        ids = rng.permutation(ids)
        nv, nt = int(nv), int(nt)
        split[ids[:nv]] = "val"
        split[ids[nv:nv + nt]] = "test"
        split[ids[nv + nt:]] = "train"
    assert int(np.sum(split == "train")) == n_train
    assert int(np.sum(split == "val")) == n_val
    assert int(np.sum(split == "test")) == n_test
    return split


def noise_tag(level: float) -> str:
    if abs(level) < 1e-15:
        return "noise_0pct"
    pct = 100.0 * float(level)
    return "noise_" + ("%g" % pct).replace(".", "p") + "pct"


def save_sample_npz(path: Path, params: np.ndarray, clean: np.ndarray, physical_ids: np.ndarray,
                    state_ids: np.ndarray, margins: np.ndarray, split_name: str, q2: np.ndarray,
                    level: float, replicas: int, seed: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    replicas = 1 if abs(level) < 1e-15 else int(replicas)
    base_idx = np.repeat(np.arange(len(params), dtype=np.int64), replicas)
    replica_id = np.tile(np.arange(replicas, dtype=np.int16), len(params))
    p = params[base_idx]
    gc64 = clean[base_idx]
    rng = np.random.default_rng(int(seed))
    if level > 0:
        sigma = float(level) * np.sqrt(np.mean(gc64 * gc64, axis=1, keepdims=True))
        noise64 = sigma * rng.standard_normal(gc64.shape)
        gy64 = gc64 + noise64
    else:
        noise64 = np.zeros_like(gc64)
        gy64 = gc64.copy()
    pids = physical_ids[base_idx]
    sids = state_ids[base_idx]
    sample_ids = np.asarray(
        [f"{pid}:{split_name}:{noise_tag(level)}:r{int(rep)}" for pid, rep in zip(pids.tolist(), replica_id.tolist())],
        dtype="<U96",
    )
    np.savez_compressed(
        path,
        q2=np.asarray(q2, dtype=np.float64),
        gy=gy64.astype(np.float32),
        g_clean=gc64.astype(np.float32),
        noise=noise64.astype(np.float32),
        params=p.astype(np.float32),
        parameters=p.astype(np.float32),
        params3=p[:, [0, 3, 4]].astype(np.float32),
        a1=p[:, 0].astype(np.float32), m=p[:, 3].astype(np.float32), gamma=p[:, 4].astype(np.float32),
        physical_state_id=pids, state_id=sids.astype(np.int64), sample_id=sample_ids,
        split=np.full(len(p), split_name, dtype="<U5"),
        noise_level=np.full(len(p), float(level), dtype=np.float32),
        noise_replica_id=replica_id,
        continuous_margin=margins[base_idx].astype(np.float32),
    )
    return len(p)


def coverage_rows(candidate_params: np.ndarray, survivor_mask: np.ndarray, seed_params: np.ndarray,
                  accepted_params: np.ndarray, split: np.ndarray, bins: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    specs = (("a1", 0), ("m", 3), ("gamma", 4))
    for name, col in specs:
        edges = np.asarray(bins[name], dtype=np.float64)
        for i in range(len(edges) - 1):
            lo, hi = float(edges[i]), float(edges[i + 1])
            def mask(x: np.ndarray) -> np.ndarray:
                if len(x) == 0:
                    return np.zeros(0, dtype=bool)
                upper = hi + (1e-12 if i == len(edges) - 2 else 0.0)
                return (x[:, col] >= lo) & (x[:, col] < upper)
            cm = mask(candidate_params)
            sm = mask(seed_params)
            am = mask(accepted_params)
            rows.append({
                "axis": name, "bin_index": i, "lo": lo, "hi": hi,
                "candidate_count": int(cm.sum()),
                "grid_survivor_count": int(np.sum(cm & survivor_mask)),
                "verified_seed_eligible_count": int(sm.sum()),
                "accepted_count": int(am.sum()),
                "train_count": int(np.sum(am & (split == "train"))),
                "val_count": int(np.sum(am & (split == "val"))),
                "test_count": int(np.sum(am & (split == "test"))),
            })
    return rows


def coverage_cell_rows(candidate_params: np.ndarray, survivor_mask: np.ndarray, seed_params: np.ndarray,
                       accepted_params: np.ndarray, split: np.ndarray, bins: dict[str, Any]) -> list[dict[str, Any]]:
    ca, cm, cg, cc = bin_indices_3d(candidate_params, bins)
    if len(seed_params):
        sa, sm, sg, sc = bin_indices_3d(seed_params, bins)
    else:
        sa = sm = sg = sc = np.empty(0, dtype=np.int32)
    aa, am, ag, ac = bin_indices_3d(accepted_params, bins)
    cells = sorted(set(cc.tolist()) | set(sc.tolist()) | set(ac.tolist()))
    rows = []
    for cell in cells:
        c_mask = cc == cell
        s_mask = sc == cell
        a_mask = ac == cell
        if np.any(a_mask):
            first = int(np.flatnonzero(a_mask)[0])
            ia, im, ig = int(aa[first]), int(am[first]), int(ag[first])
        elif np.any(c_mask):
            first = int(np.flatnonzero(c_mask)[0])
            ia, im, ig = int(ca[first]), int(cm[first]), int(cg[first])
        else:
            first = int(np.flatnonzero(s_mask)[0])
            ia, im, ig = int(sa[first]), int(sm[first]), int(sg[first])
        rows.append({
            "cell_id": int(cell), "a1_bin": ia, "m_bin": im, "gamma_bin": ig,
            "candidate_count": int(c_mask.sum()),
            "grid_survivor_count": int(np.sum(c_mask & survivor_mask)),
            "verified_seed_eligible_count": int(s_mask.sum()),
            "accepted_count": int(a_mask.sum()),
            "train_count": int(np.sum(a_mask & (split == "train"))),
            "val_count": int(np.sum(a_mask & (split == "val"))),
            "test_count": int(np.sum(a_mask & (split == "test"))),
        })
    return rows


def parameter_range_summary(params: np.ndarray, split: np.ndarray, margins: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for name in ("all", "train", "val", "test"):
        mask = np.ones(len(params), dtype=bool) if name == "all" else (split == name)
        p = params[mask]
        m = margins[mask]
        rows.append({
            "split": name, "physical_states": int(mask.sum()),
            "a1_min": float(np.min(p[:, 0])), "a1_max": float(np.max(p[:, 0])),
            "m_min": float(np.min(p[:, 3])), "m_max": float(np.max(p[:, 3])),
            "gamma_min": float(np.min(p[:, 4])), "gamma_max": float(np.max(p[:, 4])),
            "continuous_margin_min": float(np.min(m)),
            "continuous_margin_median": float(np.median(m)),
            "continuous_margin_p10": float(np.quantile(m, 0.10)),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics-config", default="cp02_corrected_3p_config.json")
    ap.add_argument("--dataset-config", default="cp02_ml_dataset_v1_config.json")
    ap.add_argument("--output-dir", default="data_cp02_ml_v1")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-verified-seed", action="store_true")
    args = ap.parse_args()

    physics_cfg = load_config(args.physics_config)
    dataset_cfg = load_json(args.dataset_config)
    design_id, threshold = frozen_definition_check(physics_cfg, dataset_cfg)
    settings = runtime_settings(physics_cfg, dataset_cfg, args.quick, args.workers)
    formula_version = str(physics_cfg["physics_formula_version"])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    generated_q2 = make_q2(design_by_id(physics_cfg, design_id))
    prior_q2_path = Path(dataset_cfg.get("verified_seed", {}).get("q2_npy", ""))
    if prior_q2_path.exists() and not args.quick:
        q2 = np.load(prior_q2_path).astype(np.float64)
        q2_source = str(prior_q2_path)
        if q2.ndim != 1 or len(q2) < 8 or not np.isfinite(q2).all() or np.any(q2 >= 0.0):
            raise RuntimeError("saved frozen q2 array is invalid")
        if q2.shape != generated_q2.shape or not np.allclose(q2, generated_q2, rtol=0.0, atol=1e-12):
            print("WARNING: saved CP02 q2 differs from the current design generator; using the saved q2 because the threshold was calibrated on that artifact.")
    else:
        q2 = generated_q2
        q2_source = "cp02_observation.make_q2(current config)"
    np.save(out / "q2.npy", q2.astype(np.float64))
    nominal_points = int(dataset_cfg["frozen_definition"].get("nominal_design_points", len(q2)))
    if len(q2) != nominal_points:
        print(
            f"NOTE: design label is nominally {nominal_points} points, but the frozen unique q2 array has {len(q2)} points. "
            "This builder preserves the exact calibrated q2 instead of adding/removing a point after threshold calibration."
        )

    print("[1/7] Load verified seeds and generate a larger full-domain candidate pool")
    seeds = load_verified_seed_states(
        dataset_cfg, physics_cfg, q2, threshold, formula_version, design_id,
        disabled=args.no_verified_seed or args.quick,
    )
    seeds = choose_seed_subset(seeds, settings["target"], dataset_cfg["coverage_bins"])
    seed_pids = {r["physical_state_id"] for r in seeds}
    candidate_params = core.sample_full_domain(
        physics_cfg, settings["candidate_count"], int(dataset_cfg["candidate_seed"])
    )

    print("[2/7] Cheap coarse-grid alias screen over all new candidates")
    coarse_bank = core.build_profile_bank(
        physics_cfg, q2, physics_cfg["integration_points"],
        settings["coarse_m_points"], settings["coarse_g_points"],
    )
    screen_margin = screen_candidates(
        candidate_params, q2, physics_cfg, coarse_bank,
        settings["candidate_forward_batch_size"], settings["coarse_state_batch_size"],
    )
    survivor_mask = screen_margin >= threshold - 1e-12
    survivor_ids = np.flatnonzero(survivor_mask)
    print(f"Coarse survivors: {len(survivor_ids)}/{len(candidate_params)}. Grid margin is NOT the final label.")
    np.savez_compressed(
        out / "candidate_screen.npz",
        params=candidate_params.astype(np.float64),
        coarse_grid_mahalanobis=screen_margin.astype(np.float64),
        coarse_grid_survivor=survivor_mask.astype(np.uint8),
    )

    # Exclude exact states already present in the verified seed bank.
    keep = []
    for idx in survivor_ids:
        pid, _ = stable_state_identity(candidate_params[int(idx)], formula_version, design_id)
        if pid not in seed_pids:
            keep.append(int(idx))
    survivor_ids = np.asarray(keep, dtype=np.int64)

    seed_params = np.asarray([r["params"] for r in seeds], dtype=np.float64) if seeds else np.empty((0, 5), dtype=np.float64)
    seed_cell_counts: dict[int, int] = {}
    if len(seed_params):
        _, _, _, scell = bin_indices_3d(seed_params, dataset_cfg["coverage_bins"])
        for c in scell:
            seed_cell_counts[int(c)] = seed_cell_counts.get(int(c), 0) + 1
    queue = interleaved_queue(
        survivor_ids, candidate_params, screen_margin,
        dataset_cfg["coverage_bins"], initial_cell_counts=seed_cell_counts,
    )

    print("[3/7] Continuous refinement only for the coverage-balanced survivor queue")
    official_bank = core.build_profile_bank(
        physics_cfg, q2, physics_cfg["integration_points"],
        settings["official_m_points"], settings["official_g_points"],
    )
    signature = refinement_signature(physics_cfg, design_id, q2, settings)
    accepted = list(seeds)
    # Keep a small backfill buffer so that rare near-threshold audit failures can be
    # quarantined without lowering the frozen scientific threshold. The final published
    # dataset is still exactly settings["target"] physical states.
    backfill_extra = max(
        int(dataset_cfg.get("audit_backfill_extra", 0)),
        int(settings["audit_boundary_count"]) + int(settings["audit_random_count"]),
        max(64, int(math.ceil(0.02 * float(settings["target"])))),
    )
    refinement_target = int(settings["target"]) + int(backfill_extra)
    accepted, refined_new = refine_new_states(
        queue, candidate_params, screen_margin, accepted, refinement_target, threshold,
        physics_cfg, q2, official_bank, settings, formula_version, design_id,
        out / "continuous_refinement_cache.csv", signature,
    )

    # Stable, deterministic ordering independent of thread completion. Audit may
    # discover a smaller alias for a few borderline states; those states are quarantined
    # and replaced from the backfill buffer, never silently published.
    accepted.sort(key=lambda r: r["physical_state_id"])
    pids_all = [r["physical_state_id"] for r in accepted]
    sids_all = [int(r["state_id"]) for r in accepted]
    if len(set(pids_all)) != len(pids_all) or len(set(sids_all)) != len(sids_all):
        raise RuntimeError("physical_state_id/state_id collision detected")

    quarantined: set[str] = set()
    audit_history: list[dict[str, Any]] = []
    max_audit_rounds = 6
    for audit_round in range(1, max_audit_rounds + 1):
        trial_records = [r for r in accepted if r["physical_state_id"] not in quarantined]
        if len(trial_records) < int(settings["target"]):
            write_json(out / "continuous_audit_failure.json", {
                "status": "FAIL",
                "reason": "not enough accepted states remain after audit quarantine",
                "target": int(settings["target"]),
                "available_after_quarantine": len(trial_records),
                "quarantined_physical_state_ids": sorted(quarantined),
                "audit_history": audit_history,
            })
            raise RuntimeError(
                "continuous audit quarantine exhausted the backfill buffer. "
                "Increase candidate_count or audit_backfill_extra; do not lower the threshold."
            )

        trial_records = trial_records[: int(settings["target"])]
        trial_params = np.asarray([r["params"] for r in trial_records], dtype=np.float64)
        trial_margins = np.asarray([r["continuous_mahalanobis"] for r in trial_records], dtype=np.float64)
        if np.any(trial_margins < threshold - 1e-12):
            raise RuntimeError("internal error: accepted state below continuous threshold")

        print("Computing float64 clean forward curves for the final physical-state bank...")
        trial_clean = forward_batched(trial_params, q2, physics_cfg, settings["candidate_forward_batch_size"])

        print("[4/7] Frozen-definition continuous re-audit on random + near-threshold accepted states")
        audit_path = out / ("continuous_audit.csv" if audit_round == 1 else f"continuous_audit_round{audit_round}.csv")
        audit = audit_continuous(
            trial_records, trial_clean, physics_cfg, q2, dataset_cfg, settings, threshold,
            settings["workers"], audit_path,
        )
        audit_history.append({
            "round": audit_round,
            "fail_count": int(audit["fail_count"]),
            "min_audit_margin": float(audit["min_audit_margin"]),
            "failed_physical_state_ids": list(audit.get("failed_physical_state_ids", [])),
        })
        if not audit["fail_count"]:
            accepted = trial_records
            params = trial_params
            clean = trial_clean
            margins = trial_margins
            pids = np.asarray([r["physical_state_id"] for r in accepted], dtype="<U40")
            sids = np.asarray([r["state_id"] for r in accepted], dtype=np.int64)
            if audit_round > 1 or quarantined:
                write_json(out / "continuous_audit_quarantine.json", {
                    "status": "PASS_AFTER_QUARANTINE",
                    "quarantined_count": len(quarantined),
                    "quarantined_physical_state_ids": sorted(quarantined),
                    "audit_history": audit_history,
                })
            break

        failed_ids = set(map(str, audit.get("failed_physical_state_ids", [])))
        if not failed_ids:
            write_json(out / "continuous_audit_failure.json", {
                "status": "FAIL",
                "reason": "audit reported failures but did not return failed IDs",
                "audit_history": audit_history,
            })
            raise RuntimeError("continuous audit failed but no failed IDs were returned")
        quarantined.update(failed_ids)
        print(
            f"  audit round {audit_round}: quarantined {len(failed_ids)} failed state(s); "
            f"total_quarantined={len(quarantined)}. Backfilling from extra accepted states."
        )
    else:
        write_json(out / "continuous_audit_failure.json", {
            "status": "FAIL",
            "reason": "continuous audit still found failures after quarantine rounds",
            "quarantined_physical_state_ids": sorted(quarantined),
            "audit_history": audit_history,
        })
        raise RuntimeError(
            "continuous audit still failed after quarantine/backfill rounds. "
            "Increase candidate_count or audit_backfill_extra; do not lower the threshold."
        )

    print("[5/7] Split by physical state before any noise is generated")
    split = stratified_physical_split(params, dataset_cfg["coverage_bins"], dataset_cfg["split"])
    split_counts = {k: int(np.sum(split == k)) for k in ("train", "val", "test")}
    print("Physical split:", split_counts)

    physical_rows = []
    for i, rec in enumerate(accepted):
        p = params[i]
        physical_rows.append({
            "physical_state_id": rec["physical_state_id"], "state_id": rec["state_id"],
            "split": split[i], "a1": p[0], "a2": p[1], "a3": p[2], "m": p[3], "gamma": p[4],
            "continuous_margin": rec["continuous_mahalanobis"],
            "continuous_rms_snr": rec["continuous_rms_snr"],
            "screen_grid_mahalanobis": rec["screen_grid_mahalanobis"],
            "source": rec["source"], "alias_trigger": rec["trigger"],
            "alias_a1": rec["alias_a1"], "alias_m": rec["alias_m"], "alias_gamma": rec["alias_gamma"],
        })
    core.write_csv(out / "physical_states.csv", physical_rows)
    np.savez_compressed(
        out / "physical_states.npz",
        q2=q2.astype(np.float64), params=params.astype(np.float64), params3=params[:, [0, 3, 4]].astype(np.float64),
        g_clean=clean.astype(np.float64), physical_state_id=pids, state_id=sids,
        split=split, continuous_margin=margins.astype(np.float64),
        source=np.asarray([r["source"] for r in accepted], dtype="<U32"),
    )

    core.write_csv(
        out / "coverage_by_axis.csv",
        coverage_rows(candidate_params, survivor_mask, seed_params, params, split, dataset_cfg["coverage_bins"]),
    )
    core.write_csv(
        out / "coverage_3d_cells.csv",
        coverage_cell_rows(candidate_params, survivor_mask, seed_params, params, split, dataset_cfg["coverage_bins"]),
    )
    core.write_csv(out / "dataset_summary.csv", parameter_range_summary(params, split, margins))

    print("[6/7] Generate noise only after the physical-state split")
    noise_cfg = dataset_cfg["noise"]
    sample_counts: dict[str, int] = {}
    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "val")
    test_idx = np.flatnonzero(split == "test")
    train_level = float(noise_cfg["train_level"])
    val_level = float(noise_cfg["val_level"])
    base_seed = int(noise_cfg["seed"])

    train_dir = out / noise_tag(train_level)
    sample_counts["train"] = save_sample_npz(
        train_dir / "train.npz", params[train_idx], clean[train_idx], pids[train_idx], sids[train_idx],
        margins[train_idx], "train", q2, train_level, settings["train_replicas"], base_seed + 101,
    )
    sample_counts["val"] = save_sample_npz(
        train_dir / "val.npz", params[val_idx], clean[val_idx], pids[val_idx], sids[val_idx],
        margins[val_idx], "val", q2, val_level, int(noise_cfg["val_replicas"]), base_seed + 201,
    )
    for j, level in enumerate(noise_cfg["test_levels"]):
        level = float(level)
        key = "test_" + noise_tag(level)
        sample_counts[key] = save_sample_npz(
            out / noise_tag(level) / "test.npz",
            params[test_idx], clean[test_idx], pids[test_idx], sids[test_idx], margins[test_idx],
            "test", q2, level, int(noise_cfg["test_replicas"]), base_seed + 1001 + 97 * j,
        )

    # Train-only normalization statistics.  No validation/test samples contribute.
    with np.load(train_dir / "train.npz", allow_pickle=False) as z:
        gy_train = z["gy"].astype(np.float64)
    input_mean = gy_train.mean(axis=0)
    input_std = gy_train.std(axis=0)
    floor = 1e-12 * np.maximum(np.abs(input_mean), 1.0)
    input_std = np.maximum(input_std, floor)
    target_train = np.column_stack([
        params[train_idx, 0], params[train_idx, 3], np.log(params[train_idx, 4])
    ]).astype(np.float64)
    target_mean = target_train.mean(axis=0)
    target_std = target_train.std(axis=0)
    if np.any(target_std <= 0):
        raise RuntimeError("degenerate train-only target normalization")
    np.savez_compressed(
        out / "normalization_train_only.npz",
        input_mean=input_mean, input_std=input_std,
        target_names=np.asarray(["a1", "m", "log_gamma"], dtype="<U16"),
        target_mean=target_mean, target_std=target_std,
        target_transform=np.asarray(["identity", "identity", "natural_log"], dtype="<U16"),
        training_noise=np.asarray(train_level, dtype=np.float64),
    )
    write_json(out / "normalization_train_only.json", {
        "source_split": "train only", "training_noise": train_level,
        "target_names": ["a1", "m", "log_gamma"],
        "target_transform": ["identity", "identity", "natural_log"],
        "target_mean": target_mean.tolist(), "target_std": target_std.tolist(),
        "input_points": len(input_mean),
    })

    print("[7/7] Write frozen metadata and integrity summary")
    metadata = {
        "dataset_id": dataset_cfg["dataset_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "quick_mode": bool(args.quick),
        "frozen_definition": {
            **dataset_cfg["frozen_definition"],
            "q2_min": float(q2.min()), "q2_max": float(q2.max()), "q2_points": int(len(q2)),
            "q2_nominal_design_points": nominal_points, "q2_source": q2_source,
            "q2_sha256": hashlib.sha256(np.asarray(q2, dtype="<f8").tobytes()).hexdigest(),
            "formula": physics_cfg["formula"],
        },
        "candidate_sampling": {
            "domain": "cp02 regular_numerical_domain (empirical identifiable ranges are coverage references, not hard bounds)",
            "regular_numerical_domain": physics_cfg["regular_numerical_domain"],
            "candidate_count": int(len(candidate_params)), "candidate_seed": int(dataset_cfg["candidate_seed"]),
            "coarse_grid": {"m_points": settings["coarse_m_points"], "log_gamma_points": settings["coarse_g_points"]},
            "coarse_survivors": int(survivor_mask.sum()),
        },
        "continuous_refinement": {
            "refinement_signature": signature,
            "official_grid": {"m_points": settings["official_m_points"], "log_gamma_points": settings["official_g_points"]},
            "top_regions": settings["top_regions"], "maxiter": settings["maxiter"], "multistart": settings["multistart"],
            "newly_computed_this_run": int(refined_new),
            "verified_seed_accepted": int(sum(r["source"] == "verified_cp02_prebank" for r in accepted)),
            "scientific_acceptance_rule": "continuous_mahalanobis >= 1.5; coarse grid is screening/prioritization only",
        },
        "physical_states": {
            "count": int(len(params)), "split_counts": split_counts,
            "continuous_margin_min": float(margins.min()), "continuous_margin_median": float(np.median(margins)),
        },
        "noise": {
            **noise_cfg,
            "train_replicas_effective": int(settings["train_replicas"]),
            "model": "g_noisy = g_clean + noise_level * RMS(g_clean) * N(0,1)",
            "split_before_noise": True,
            "test_uses_same_physical_states_at_all_noise_levels": True,
            "sample_counts": sample_counts,
        },
        "normalization": {
            "statistics_source": "train split only",
            "gamma_target": "natural_log(gamma) before standardization",
        },
        "coverage_bins": dataset_cfg["coverage_bins"],
        "continuous_audit": audit,
        "validation": {
            "pass": True,
            "builder_checks": [
                "frozen formula/design/threshold", "all accepted margins >= 1.5",
                "unique physical_state_id/state_id", "continuous audit pass", "split before noise",
            ],
            "external_validator": "run validate_cp02_ml_dataset_v1.py before training",
        },
    }
    write_json(out / "metadata.json", metadata)
    print(json.dumps({
        "status": "PASS", "output_dir": str(out), "physical_states": len(params),
        "split_counts": split_counts, "sample_counts": sample_counts,
        "min_continuous_margin": float(margins.min()), "audit": audit,
    }, indent=2))


if __name__ == "__main__":
    main()
