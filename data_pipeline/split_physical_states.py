from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from pipeline_core import (
    default_output_dir,
    load_config,
    master_rms,
    nearest_to_train_scores,
    normalized_parameter_matrix,
    pairwise_min_snr_between,
    parameter_matrix_from_rows,
    read_csv_rows,
    update_metadata,
    write_csv_rows,
)

PARAMETER_NAMES = ("a1", "a2", "a3", "m", "gamma")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split selected physical states before noise expansion")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--smoke-permissive", action="store_true")
    return p.parse_args()


def _target_counts(n: int, cfg: dict) -> tuple[int, int, int]:
    sp = cfg["split"]
    train_f = float(sp.get("train_fraction", 0.70))
    val_f = float(sp.get("val_fraction", 0.15))
    test_f = float(sp.get("test_fraction", 0.15))
    total = train_f + val_f + test_f
    if total <= 0.0:
        raise RuntimeError("split fractions must sum to a positive value")
    train_f, val_f, test_f = train_f / total, val_f / total, test_f / total

    min_train = int(sp.get("min_train_states", 1))
    min_val = int(sp.get("min_val_states", 5))
    min_test = int(sp.get("min_test_states", 5))
    hard_min_total = min_train + min_val + min_test
    if n < hard_min_total:
        raise RuntimeError(
            "selected bank is too small for requested split minima: "
            f"selected={n}, required>={hard_min_total} "
            f"(train>={min_train}, val>={min_val}, test>={min_test})"
        )

    n_val = max(min_val, int(round(n * val_f)))
    n_test = max(min_test, int(round(n * test_f)))
    n_train = n - n_val - n_test

    # If the nominal fractions would violate the train hard minimum, shrink only
    # the holdout allocation while preserving val:test ratio and both holdout minima.
    if n_train < min_train:
        holdout_total = n - min_train
        holdout_weight = val_f + test_f
        frac_val = 0.5 if holdout_weight <= 0.0 else val_f / holdout_weight
        n_val = int(round(holdout_total * frac_val))
        n_val = max(min_val, min(n_val, holdout_total - min_test))
        n_test = holdout_total - n_val
        if n_test < min_test:
            n_test = min_test
            n_val = holdout_total - n_test
        n_train = n - n_val - n_test

    if n_train < min_train or n_val < min_val or n_test < min_test:
        raise RuntimeError(
            "unable to satisfy configured split minima after count allocation: "
            f"got train/val/test={n_train}/{n_val}/{n_test}, "
            f"required>={min_train}/{min_val}/{min_test}"
        )
    return int(n_train), int(n_val), int(n_test)


def _pairwise_snr_matrix(g: np.ndarray, rms: np.ndarray, reference_noise: float) -> np.ndarray:
    """Exact selected-bank pairwise RMS-SNR matrix. Selected banks are intentionally small."""
    g64 = np.asarray(g, dtype=np.float64)
    n, d = g64.shape
    norm = np.sum(g64 * g64, axis=1)
    dist2 = np.maximum(norm[:, None] + norm[None, :] - 2.0 * (g64 @ g64.T), 0.0)
    rmsd = np.sqrt(dist2 / float(d))
    sigma = float(reference_noise) * 0.5 * (rms[:, None] + rms[None, :])
    out = rmsd / np.maximum(sigma, 1e-30)
    np.fill_diagonal(out, 0.0)
    return out



def _try_coverable_holdout(
    adj: np.ndarray,
    z: np.ndarray,
    target_holdout: int,
    seed: int,
    trials: int = 4000,
    stop_on_first: bool = False,
) -> tuple[np.ndarray | None, dict]:
    """Find a holdout subset whose members all retain a train neighbor.

    ``adj[i, j]`` means states i and j are within the configured
    holdout->train RMS-SNR cover radius.  Train is the complement of the
    returned holdout.  The removal invariant guarantees that every state
    already placed in holdout keeps at least one adjacent state in train.

    This helper is shared with selection so split feasibility is checked
    before a selected bank is frozen.
    """
    adj = np.asarray(adj, dtype=bool)
    z = np.asarray(z, dtype=np.float64)
    n = int(adj.shape[0])
    target_holdout = int(target_holdout)
    trials = max(1, int(trials))

    if adj.shape != (n, n):
        raise ValueError("adj must be a square matrix")
    if len(z) != n:
        raise ValueError("z row count must match adj")
    if target_holdout < 0:
        raise ValueError("target_holdout must be non-negative")
    if target_holdout == 0:
        return np.empty(0, dtype=np.int64), {
            "requested_holdout_count": 0,
            "best_holdout_found": 0,
            "search_trials": trials,
            "cover_graph_nonisolated_count": int(np.sum(np.sum(adj, axis=1) > 0)),
            "cover_graph_isolated_count": int(np.sum(np.sum(adj, axis=1) == 0)),
            "cover_graph_degree_min": int(np.min(np.sum(adj, axis=1))) if n else 0,
            "cover_graph_degree_median": float(np.median(np.sum(adj, axis=1))) if n else 0.0,
            "cover_graph_degree_max": int(np.max(np.sum(adj, axis=1))) if n else 0,
        }

    # Self-edges are never valid cover neighbors.
    adj = adj.copy()
    np.fill_diagonal(adj, False)
    degree = np.sum(adj, axis=1).astype(np.int64)
    nonisolated = np.where(degree > 0)[0]

    meta = {
        "requested_holdout_count": target_holdout,
        "best_holdout_found": 0,
        "search_trials": trials,
        "cover_graph_nonisolated_count": int(len(nonisolated)),
        "cover_graph_isolated_count": int(n - len(nonisolated)),
        "cover_graph_degree_min": int(np.min(degree)) if n else 0,
        "cover_graph_degree_median": float(np.median(degree)) if n else 0.0,
        "cover_graph_degree_max": int(np.max(degree)) if n else 0,
    }

    if target_holdout > n or len(nonisolated) < target_holdout:
        return None, meta

    rng = np.random.default_rng(int(seed))
    center = np.full(z.shape[1], 0.5, dtype=np.float64) if z.ndim == 2 else None

    best_full = None
    best_full_key = None
    best_partial = 0

    for _ in range(trials):
        train_mask = np.ones(n, dtype=bool)
        hold_mask = np.zeros(n, dtype=bool)
        # Number of currently remaining train neighbors for each state.
        train_neighbor_count = degree.copy()
        chosen: list[int] = []
        min_param = np.full(n, np.inf, dtype=np.float64)

        for step in range(target_holdout):
            # Vectorized form of the original eligibility test.
            #
            # A previously chosen holdout with <2 remaining train neighbors is
            # "fragile": removing any of its current train neighbors would
            # strand it.  Therefore every vertex adjacent to a fragile holdout
            # is blocked from being moved to holdout on this step.
            fragile_holdout = hold_mask & (train_neighbor_count < 2)
            if np.any(fragile_holdout):
                blocked = np.any(adj[fragile_holdout], axis=0)
            else:
                blocked = np.zeros(n, dtype=bool)

            eligible_mask = (
                train_mask
                & (degree > 0)
                & (train_neighbor_count >= 1)
                & (~blocked)
            )
            eligible_arr = np.flatnonzero(eligible_mask)
            if len(eligible_arr) == 0:
                break
            if step == 0:
                if z.ndim == 2 and z.shape[1] > 0:
                    coverage_score = np.linalg.norm(z[eligible_arr] - center, axis=1)
                else:
                    coverage_score = np.zeros(len(eligible_arr), dtype=np.float64)
            else:
                coverage_score = min_param[eligible_arr]

            # Prefer candidates that have more alternative train neighbors,
            # with deterministic random jitter only for tie diversification.
            degree_scale = max(float(np.max(degree)), 1.0)
            degree_score = degree[eligible_arr] / degree_scale
            jitter = rng.random(len(eligible_arr))
            score = coverage_score + 0.03 * degree_score + 0.005 * jitter
            j = int(eligible_arr[int(np.argmax(score))])

            train_mask[j] = False
            hold_mask[j] = True
            chosen.append(j)
            train_neighbor_count = train_neighbor_count - adj[:, j].astype(np.int64)

            if z.ndim == 2 and z.shape[1] > 0:
                d = np.linalg.norm(z - z[j], axis=1)
                min_param = np.minimum(min_param, d)

        best_partial = max(best_partial, len(chosen))
        if len(chosen) != target_holdout:
            continue

        hold = np.asarray(chosen, dtype=np.int64)

        # Selection only needs an existence proof.  Returning immediately on
        # the first complete holdout avoids spending thousands of extra trials
        # optimizing a span score that selection never uses.
        if stop_on_first:
            meta["best_holdout_found"] = int(target_holdout)
            if z.ndim == 2 and z.shape[1] > 0:
                spans = np.max(z[hold], axis=0) - np.min(z[hold], axis=0)
                meta["holdout_min_normalized_span"] = float(np.min(spans))
                meta["holdout_total_normalized_span"] = float(np.sum(spans))
            return hold, meta

        if z.ndim == 2 and z.shape[1] > 0:
            spans = np.max(z[hold], axis=0) - np.min(z[hold], axis=0)
            key = (float(np.min(spans)), float(np.sum(spans)))
        else:
            key = (0.0, 0.0)

        if best_full_key is None or key > best_full_key:
            best_full_key = key
            best_full = hold

    meta["best_holdout_found"] = int(target_holdout if best_full is not None else best_partial)
    if best_full is not None and best_full_key is not None:
        meta["holdout_min_normalized_span"] = float(best_full_key[0])
        meta["holdout_total_normalized_span"] = float(best_full_key[1])

    return best_full, meta


def _choose_train_cover(
    dmat: np.ndarray,
    alias_score: np.ndarray,
    target_train: int,
    max_cover: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Farthest-first train cover, matching the successful Exp35 split philosophy.

    Start from the safest selected state, build the requested train size by repeatedly
    adding the currently worst-covered state, then promote extra states into train
    only while needed to satisfy the hard holdout->train coverage threshold.
    """
    n = len(dmat)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), 0
    first = int(np.nanargmax(np.where(np.isfinite(alias_score), alias_score, -np.inf)))
    train = [first]
    remaining = set(range(n)) - {first}
    nearest = np.asarray(dmat[:, first], dtype=np.float64).copy()
    nearest[first] = 0.0

    while len(train) < int(target_train) and remaining:
        j = max(remaining, key=lambda x: float(nearest[x]))
        train.append(int(j))
        remaining.remove(j)
        nearest = np.minimum(nearest, dmat[:, j])
        nearest[np.asarray(train, dtype=np.int64)] = 0.0

    promoted = 0
    while remaining:
        j = max(remaining, key=lambda x: float(nearest[x]))
        if float(nearest[j]) <= float(max_cover) + 1e-12:
            break
        train.append(int(j))
        remaining.remove(j)
        promoted += 1
        nearest = np.minimum(nearest, dmat[:, j])
        nearest[np.asarray(train, dtype=np.int64)] = 0.0

    return np.asarray(sorted(train), dtype=np.int64), nearest, promoted


def _adjust_holdout_counts(remaining_count: int, cfg: dict, nominal_val: int, nominal_test: int) -> tuple[int, int]:
    sp = cfg["split"]
    min_val = int(sp.get("min_val_states", 5))
    min_test = int(sp.get("min_test_states", 5))
    if remaining_count < min_val + min_test:
        raise RuntimeError("train coverage leaves too few states for val/test minima")
    if remaining_count == nominal_val + nominal_test:
        return nominal_val, nominal_test

    # Preserve the intended val:test ratio as closely as possible after train promotions.
    val_weight = float(sp.get("val_fraction", 0.15))
    test_weight = float(sp.get("test_fraction", 0.15))
    frac_val = val_weight / max(val_weight + test_weight, 1e-30)
    n_val = int(round(remaining_count * frac_val))
    n_val = max(min_val, min(n_val, remaining_count - min_test))
    n_test = remaining_count - n_val
    if n_test < min_test:
        n_test = min_test
        n_val = remaining_count - n_test
    return int(n_val), int(n_test)


def _span_score(z: np.ndarray) -> tuple[float, float]:
    if len(z) == 0 or z.shape[1] == 0:
        return 0.0, 0.0
    span = np.max(z, axis=0) - np.min(z, axis=0)
    return float(np.min(span)), float(np.sum(span))


def _balanced_val_test(
    remaining: np.ndarray,
    z: np.ndarray,
    n_val: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split already-covered holdouts while preserving parameter-space coverage.

    For the small formal selected banks we can enumerate the smaller side exactly.
    The objective maximizes the worst normalized parameter span across val and test,
    then total span. For larger combinatorial cases a deterministic farthest fallback
    is used instead of constructing any N x N candidate-pool matrix.
    """
    remaining = np.asarray(remaining, dtype=np.int64)
    if len(remaining) != int(n_val) + int(n_test):
        raise ValueError("remaining count mismatch")

    choose_test = n_test <= n_val
    k = int(n_test if choose_test else n_val)
    comb_count = math.comb(len(remaining), k)
    best = None
    best_key = None

    if comb_count <= 200000:
        all_set = set(int(x) for x in remaining)
        for combo in itertools.combinations([int(x) for x in remaining], k):
            a = np.asarray(combo, dtype=np.int64)
            b = np.asarray(sorted(all_set - set(combo)), dtype=np.int64)
            if choose_test:
                test, val = a, b
            else:
                val, test = a, b
            vmin, vsum = _span_score(z[val])
            tmin, tsum = _span_score(z[test])
            # First protect the weaker split, then reward total multidimensional span.
            key = (min(vmin, tmin), vmin + tmin, vsum + tsum)
            if best_key is None or key > best_key:
                best_key = key
                best = (val, test)
    else:
        rng = np.random.default_rng(int(seed))
        first = int(rng.choice(remaining))
        order = [first]
        active = set(int(x) for x in remaining) - {first}
        min_d = {int(x): float(np.linalg.norm(z[int(x)] - z[first])) for x in active}
        while active:
            j = max(active, key=lambda x: min_d[x])
            order.append(j)
            active.remove(j)
            for x in active:
                min_d[x] = min(min_d[x], float(np.linalg.norm(z[x] - z[j])))
        val, test = [], []
        for idx in order:
            if len(val) >= n_val:
                test.append(idx)
            elif len(test) >= n_test:
                val.append(idx)
            elif len(val) / max(n_val, 1) <= len(test) / max(n_test, 1):
                val.append(idx)
            else:
                test.append(idx)
        best = (np.asarray(sorted(val), dtype=np.int64), np.asarray(sorted(test), dtype=np.int64))
        vmin, vsum = _span_score(z[best[0]])
        tmin, tsum = _span_score(z[best[1]])
        best_key = (min(vmin, tmin), vmin + tmin, vsum + tsum)

    assert best is not None
    return best[0], best[1], {
        "holdout_assignment_exact": bool(comb_count <= 200000),
        "holdout_assignment_combinations": int(comb_count),
        "val_test_min_normalized_span": float(best_key[0]),
        "val_test_sum_min_span": float(best_key[1]),
        "val_test_total_span": float(best_key[2]),
    }


def run(config_path: str, output_dir: str | None = None, seed: int | None = None,
        smoke_permissive: bool = False) -> Path:
    cfg = load_config(config_path)
    out = Path(output_dir) if output_dir else default_output_dir(cfg)
    sel_rows = read_csv_rows(out / "selected_states.csv")
    params = parameter_matrix_from_rows(sel_rows)
    ids = np.asarray([r["state_id"] for r in sel_rows])
    alias_score = np.asarray([float(r.get("alias_score", 0.0)) for r in sel_rows], dtype=np.float64)
    cand_rows = read_csv_rows(out / "candidate_states.csv")
    cand_index = {r["state_id"]: int(r["state_index"]) for r in cand_rows}
    selected_candidate_idx = np.asarray([cand_index[s] for s in ids], dtype=np.int64)
    bank = np.load(out / "candidate_g_clean.npz", allow_pickle=False)
    g_all = np.asarray(bank["g_clean"], dtype=np.float32)
    obs_idx = np.asarray(bank["observation_indices"], dtype=np.int64)
    g = g_all[selected_candidate_idx][:, obs_idx]
    rms = master_rms(g_all[selected_candidate_idx])
    reference_noise = float(cfg["identifiability"]["reference_noise"])

    split_cfg = cfg["split"]
    seed = int(seed if seed is not None else split_cfg.get("seed", cfg["sampling"]["seed"] + 1000))
    target_train, nominal_val, nominal_test = _target_counts(len(params), cfg)
    max_cover = float(split_cfg.get("max_train_cover_rms_snr", 2.0))
    min_val = int(split_cfg.get("min_val_states", 5))
    min_test = int(split_cfg.get("min_test_states", 5))

    dmat = _pairwise_snr_matrix(g, rms, reference_noise)
    train, nearest_after_cover, promoted = _choose_train_cover(
        dmat, alias_score, target_train=target_train, max_cover=max_cover
    )
    remaining = np.asarray(sorted(set(range(len(params))) - set(int(x) for x in train)), dtype=np.int64)

    if len(remaining) < min_val + min_test:
        support = {
            "reason": "train coverage requirement leaves too few independent val/test states",
            "max_train_cover_rms_snr": max_cover,
            "selected_state_count": int(len(params)),
            "nominal_train_state_count": int(target_train),
            "promoted_extra_train_states": int(promoted),
            "final_train_state_count": int(len(train)),
            "remaining_holdout_state_count": int(len(remaining)),
            "min_val_states": int(min_val),
            "min_test_states": int(min_test),
            "max_nearest_after_train_cover_rms_snr": float(np.max(nearest_after_cover[remaining])) if len(remaining) else float("inf"),
        }
        (out / "split_insufficient_support.json").write_text(json.dumps(support, indent=2), encoding="utf-8")
        if not smoke_permissive:
            raise RuntimeError("physical-state split coverage failed; see split_insufficient_support.json")

    n_val, n_test = _adjust_holdout_counts(len(remaining), cfg, nominal_val, nominal_test)
    z = normalized_parameter_matrix(params, cfg)
    val, test, assign_meta = _balanced_val_test(remaining, z, n_val, n_test, seed)

    hold = np.concatenate([val, test])
    cover_scores, cover_neighbor = nearest_to_train_scores(g, rms, train, hold, reference_noise)
    cover_ok = bool(len(cover_scores) == 0 or np.max(cover_scores) <= max_cover + 1e-8)
    if not smoke_permissive and not cover_ok:
        support = {
            "reason": "train-cover construction did not satisfy max_train_cover_rms_snr",
            "max_train_cover_rms_snr": max_cover,
            "max_actual_holdout_to_train_rms_snr": float(np.max(cover_scores)),
            "train_state_count": int(len(train)),
            "val_state_count": int(len(val)),
            "test_state_count": int(len(test)),
        }
        (out / "split_insufficient_support.json").write_text(json.dumps(support, indent=2), encoding="utf-8")
        raise RuntimeError("physical-state split coverage failed; see split_insufficient_support.json")

    labels = np.full(len(params), "", dtype=object)
    labels[train] = "train"
    labels[val] = "val"
    labels[test] = "test"
    manifest = []
    for i, row in enumerate(sel_rows):
        r = {"state_id": row["state_id"], "selected_index": i, "candidate_state_index": int(selected_candidate_idx[i]), "split": str(labels[i])}
        for name in PARAMETER_NAMES:
            r[name] = float(row[name])
        manifest.append(r)
    write_csv_rows(out / "split_manifest.csv", manifest)

    coverage_rows = []
    for qpos, idx in enumerate(hold):
        nidx = int(cover_neighbor[qpos])
        coverage_rows.append({
            "state_id": str(ids[idx]),
            "split": str(labels[idx]),
            "nearest_train_state_id": str(ids[nidx]) if nidx >= 0 else "",
            "nearest_train_rms_snr": float(cover_scores[qpos]),
        })
    write_csv_rows(out / "holdout_to_train_coverage.csv", coverage_rows)

    pair_rows = []
    split_indices = {"train": train, "val": val, "test": test}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        li, ri = split_indices[left], split_indices[right]
        met = pairwise_min_snr_between(g[li], rms[li], g[ri], rms[ri], reference_noise)
        pair_rows.append({"split_pair": f"{left}-{right}", "left_count": len(li), "right_count": len(ri), **met})
    write_csv_rows(out / "split_separation_summary.csv", pair_rows)

    max_actual_cover = float(np.max(cover_scores)) if len(cover_scores) else 0.0
    update_metadata(out, {
        "status": "PHYSICAL_SPLIT_READY",
        "split": {
            **cfg["split"],
            "seed": seed,
            "state_counts": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
            "nominal_state_counts": {"train": int(target_train), "val": int(nominal_val), "test": int(nominal_test)},
            "coverage_promoted_extra_train_states": int(promoted),
            "max_actual_holdout_to_train_rms_snr": max_actual_cover,
            "coverage_pass": cover_ok,
            "split_strategy": "train_cover_then_balanced_holdout",
            **assign_meta,
            "smoke_permissive": bool(smoke_permissive),
            "cross_split_g_separation": pair_rows,
        },
    })
    print("=" * 100)
    print(f"physical-state split : train={len(train)} val={len(val)} test={len(test)}")
    print(f"train cover promoted : {promoted}")
    print(f"max holdout->train   : {max_actual_cover:.6g} RMS-SNR")
    for r in pair_rows:
        print(f"{r['split_pair']:>10s} min g separation: {r['min_rms_snr']:.6g} RMS-SNR")
    print(f"[done] {out / 'split_manifest.csv'}")
    print(f"[done] {out / 'split_separation_summary.csv'}")
    print("=" * 100)
    return out


def main() -> None:
    a = parse_args()
    run(a.config, a.output_dir, a.seed, a.smoke_permissive)


if __name__ == "__main__":
    main()
