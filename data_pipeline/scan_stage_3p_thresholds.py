from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from pipeline_core import (
    PARAMETER_NAMES,
    load_config,
    master_rms,
    normalized_parameter_matrix,
    parameter_matrix_from_rows,
    read_csv_rows,
    symmetric_rms_snr,
    write_csv_rows,
)


def _float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reuse the formal 3P candidate/identifiability bank and scan alias cutoffs without noise expansion."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--source-output", default="data_pipeline_outputs/stage_3p")
    p.add_argument("--output-dir", default="data_pipeline_outputs/stage_3p_threshold_scan")
    p.add_argument("--thresholds", type=_float_list, default=_float_list("0.10,0.15,0.20,0.25,0.30,0.35,0.40"))
    p.add_argument("--min-train-states", type=int, default=200)
    p.add_argument("--min-val-states", type=int, default=30)
    p.add_argument("--min-test-states", type=int, default=30)
    p.add_argument("--max-selected-states", type=int, default=500)
    p.add_argument("--min-separation-rms-snr", type=float, default=1.3)
    p.add_argument("--max-parameter-cover-radius", type=float, default=0.35)
    p.add_argument("--max-train-cover-rms-snr", type=float, default=2.0)
    p.add_argument("--min-test-parameter-span", type=float, default=0.5)
    p.add_argument("--holdout-search-trials", type=int, default=500)
    p.add_argument("--assignment-trials", type=int, default=4000)
    p.add_argument("--seed", type=int, default=20260842)
    return p.parse_args()


def _pairwise_snr_matrix(g: np.ndarray, rms: np.ndarray, reference_noise: float) -> np.ndarray:
    """Exact selected-bank pairwise RMS-SNR matrix; selected bank is capped at <=500."""
    g64 = np.asarray(g, dtype=np.float64)
    n, d = g64.shape
    norm = np.sum(g64 * g64, axis=1)
    dist2 = np.maximum(norm[:, None] + norm[None, :] - 2.0 * (g64 @ g64.T), 0.0)
    rmsd = np.sqrt(dist2 / float(d))
    sigma = float(reference_noise) * 0.5 * (rms[:, None] + rms[None, :])
    out = rmsd / np.maximum(sigma, 1e-30)
    np.fill_diagonal(out, 0.0)
    return out


def _greedy_separated_bank(
    params: np.ndarray,
    g_obs: np.ndarray,
    rms: np.ndarray,
    alias: np.ndarray,
    safe_idx: np.ndarray,
    cfg: dict,
    min_sep: float,
    max_selected: int,
) -> tuple[np.ndarray, float]:
    """Coverage-first greedy independent bank.

    It intentionally keeps selecting until max_selected or until no additional state
    can satisfy the hard selected-bank g-separation requirement.  This makes the scan
    answer "how many usable physical states can this cutoff support?" rather than
    stopping as soon as parameter coverage is already good.
    """
    safe_idx = np.asarray(safe_idx, dtype=np.int64)
    if len(safe_idx) == 0:
        return np.empty(0, dtype=np.int64), float("inf")

    z = normalized_parameter_matrix(params, cfg)
    center_dist = np.linalg.norm(z[safe_idx] - 0.5, axis=1)
    first_local = int(np.argmax(alias[safe_idx] + 1.0e-9 * center_dist))
    first = int(safe_idx[first_local])

    selected = [first]
    chosen = np.zeros(len(safe_idx), dtype=bool)
    chosen[first_local] = True

    min_param = np.linalg.norm(z[safe_idx] - z[first], axis=1)
    min_g = symmetric_rms_snr(
        g_obs[safe_idx],
        rms[safe_idx],
        g_obs[first][None, :],
        np.full(len(safe_idx), rms[first], dtype=np.float64),
        float(cfg["identifiability"]["reference_noise"]),
    )
    min_g[first_local] = 0.0

    limit = min(int(max_selected), len(safe_idx))
    while len(selected) < limit:
        eligible = (~chosen) & (min_g >= float(min_sep))
        if not np.any(eligible):
            break
        # First preserve normalized parameter-space coverage; use alias margin only as a tiny tie-break.
        score = np.where(eligible, min_param + 1.0e-6 * alias[safe_idx], -np.inf)
        pick_local = int(np.argmax(score))
        pick = int(safe_idx[pick_local])
        selected.append(pick)
        chosen[pick_local] = True

        dparam = np.linalg.norm(z[safe_idx] - z[pick], axis=1)
        min_param = np.minimum(min_param, dparam)
        dg = symmetric_rms_snr(
            g_obs[safe_idx],
            rms[safe_idx],
            g_obs[pick][None, :],
            np.full(len(safe_idx), rms[pick], dtype=np.float64),
            float(cfg["identifiability"]["reference_noise"]),
        )
        min_g = np.minimum(min_g, dg)
        min_g[chosen] = 0.0

    return np.asarray(selected, dtype=np.int64), float(np.max(min_param))


def _normalized_spans(z: np.ndarray, free_params: list[str]) -> dict[str, float]:
    if len(z) == 0:
        return {name: 0.0 for name in free_params}
    span = np.max(z, axis=0) - np.min(z, axis=0)
    return {name: float(span[i]) for i, name in enumerate(free_params)}


def _can_remove_to_holdout(
    j: int,
    hold_mask: np.ndarray,
    train_neighbor_count: np.ndarray,
    adj: np.ndarray,
) -> bool:
    # j itself must retain a train neighbor after it becomes holdout.
    if int(train_neighbor_count[j]) < 1:
        return False
    affected = hold_mask & adj[:, j]
    if np.any(train_neighbor_count[affected] < 2):
        return False
    return True


def _build_coverable_holdout(
    adj: np.ndarray,
    z: np.ndarray,
    holdout_count: int,
    trials: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, dict]:
    """Find a holdout subset H such that every h in H has a <=cover neighbor in train.

    Train is the complement of H.  The greedy removal invariant guarantees that each
    already-selected holdout keeps at least one train neighbor after every removal.
    Multiple randomized coverage-first trials are used because the cover graph can be sparse.
    """
    n = len(adj)
    degree = np.sum(adj, axis=1).astype(np.int64)
    nonisolated = np.where(degree > 0)[0]
    base_meta = {
        "cover_graph_nonisolated_count": int(len(nonisolated)),
        "cover_graph_isolated_count": int(n - len(nonisolated)),
        "cover_graph_degree_min": int(np.min(degree)) if n else 0,
        "cover_graph_degree_median": float(np.median(degree)) if n else 0.0,
        "cover_graph_degree_max": int(np.max(degree)) if n else 0,
    }
    if len(nonisolated) < int(holdout_count):
        return None, base_meta

    best = None
    best_key = None
    center = np.full(z.shape[1], 0.5, dtype=np.float64)

    for trial in range(max(1, int(trials))):
        train_mask = np.ones(n, dtype=bool)
        hold_mask = np.zeros(n, dtype=bool)
        train_neighbor_count = degree.copy()
        min_param = np.full(n, np.inf, dtype=np.float64)
        chosen: list[int] = []

        for step in range(int(holdout_count)):
            candidates = np.where(train_mask & (degree > 0))[0]
            if len(candidates) == 0:
                break
            eligible = []
            for j in candidates:
                if _can_remove_to_holdout(int(j), hold_mask, train_neighbor_count, adj):
                    eligible.append(int(j))
            if not eligible:
                break
            eligible = np.asarray(eligible, dtype=np.int64)

            if step == 0:
                coverage_score = np.linalg.norm(z[eligible] - center, axis=1)
            else:
                coverage_score = min_param[eligible]
            # Mildly prefer nodes with more alternative train neighbors.
            degree_score = degree[eligible] / max(float(np.max(degree)), 1.0)
            jitter = rng.random(len(eligible))
            score = coverage_score + 0.03 * degree_score + 0.005 * jitter
            j = int(eligible[int(np.argmax(score))])

            train_mask[j] = False
            hold_mask[j] = True
            chosen.append(j)
            train_neighbor_count = train_neighbor_count - adj[:, j].astype(np.int64)
            d = np.linalg.norm(z - z[j], axis=1)
            min_param = np.minimum(min_param, d)

        if len(chosen) != int(holdout_count):
            continue
        h = np.asarray(chosen, dtype=np.int64)
        spans = np.max(z[h], axis=0) - np.min(z[h], axis=0)
        key = (float(np.min(spans)), float(np.sum(spans)))
        if best_key is None or key > best_key:
            best_key = key
            best = h

    if best is not None:
        base_meta["holdout_min_normalized_span"] = float(best_key[0])
        base_meta["holdout_total_normalized_span"] = float(best_key[1])
    return best, base_meta


def _assign_val_test(
    holdout: np.ndarray,
    z: np.ndarray,
    n_val: int,
    n_test: int,
    trials: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    holdout = np.asarray(holdout, dtype=np.int64)
    if len(holdout) != int(n_val) + int(n_test):
        raise ValueError("holdout size mismatch")
    best = None
    best_key = None

    # Deterministic seed assignment plus randomized assignments.
    order0 = np.argsort(np.linalg.norm(z[holdout] - 0.5, axis=1))[::-1]
    deterministic = holdout[order0]
    assignments = [(deterministic[:n_test], deterministic[n_test:])]
    for _ in range(max(1, int(trials))):
        p = rng.permutation(holdout)
        assignments.append((p[:n_test], p[n_test:]))

    for test, val in assignments:
        test_span = np.max(z[test], axis=0) - np.min(z[test], axis=0)
        val_span = np.max(z[val], axis=0) - np.min(z[val], axis=0)
        key = (
            float(min(np.min(test_span), np.min(val_span))),
            float(np.min(test_span)),
            float(np.min(val_span)),
            float(np.sum(test_span) + np.sum(val_span)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = (np.asarray(val, dtype=np.int64), np.asarray(test, dtype=np.int64))

    assert best is not None
    return best[0], best[1], {
        "val_test_weakest_span": float(best_key[0]),
        "test_min_normalized_span": float(best_key[1]),
        "val_min_normalized_span": float(best_key[2]),
    }


def _write_selected_preview(
    path: Path,
    candidate_rows: list[dict],
    selected_idx: np.ndarray,
    alias: np.ndarray,
) -> None:
    rows = []
    for rank, idx in enumerate(selected_idx):
        r = {
            "selected_rank": int(rank),
            "state_id": candidate_rows[int(idx)]["state_id"],
            "state_index": int(idx),
            "alias_score": float(alias[int(idx)]),
        }
        for name in PARAMETER_NAMES:
            r[name] = float(candidate_rows[int(idx)][name])
        rows.append(r)
    write_csv_rows(path, rows)


def _write_split_preview(
    path: Path,
    candidate_rows: list[dict],
    selected_idx: np.ndarray,
    train_local: np.ndarray,
    val_local: np.ndarray,
    test_local: np.ndarray,
) -> None:
    labels = np.full(len(selected_idx), "", dtype=object)
    labels[train_local] = "train"
    labels[val_local] = "val"
    labels[test_local] = "test"
    rows = []
    for local_i, global_i in enumerate(selected_idx):
        r = {
            "selected_index": int(local_i),
            "state_id": candidate_rows[int(global_i)]["state_id"],
            "state_index": int(global_i),
            "split": str(labels[local_i]),
        }
        for name in PARAMETER_NAMES:
            r[name] = float(candidate_rows[int(global_i)][name])
        rows.append(r)
    write_csv_rows(path, rows)


def run(a: argparse.Namespace) -> Path:
    cfg = load_config(a.config)
    if str(cfg.get("stage", "")).upper() != "3P":
        raise RuntimeError("this scanner is intentionally scoped to the formal 3P stage")

    source = Path(a.source_output)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    required = [
        source / "candidate_states.csv",
        source / "candidate_g_clean.npz",
        source / "identifiability_results.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "formal 3P candidate/identifiability outputs are missing; run .\\run_stage_3p.ps1 -Overwrite once first:\n"
            + "\n".join(missing)
        )

    candidate_rows = read_csv_rows(source / "candidate_states.csv")
    ident_rows = read_csv_rows(source / "identifiability_results.csv")
    if len(candidate_rows) != len(ident_rows):
        raise RuntimeError("candidate and identifiability row counts differ")

    params = parameter_matrix_from_rows(candidate_rows)
    bank = np.load(source / "candidate_g_clean.npz", allow_pickle=False)
    g_master = np.asarray(bank["g_clean"], dtype=np.float32)
    obs_idx = np.asarray(bank["observation_indices"], dtype=np.int64)
    g_obs = g_master[:, obs_idx]
    rms = master_rms(g_master)
    alias = np.asarray([float(r["alias_score"]) for r in ident_rows], dtype=np.float64)
    finite = np.isfinite(alias)
    z_all = normalized_parameter_matrix(params, cfg)
    free_params = list(cfg["free_params"])
    ref_noise = float(cfg["identifiability"]["reference_noise"])

    min_selected_required = int(a.min_train_states + a.min_val_states + a.min_test_states)
    if a.max_selected_states < min_selected_required:
        raise ValueError(
            f"max_selected_states={a.max_selected_states} < hard split minimum {min_selected_required}"
        )

    print("=" * 110)
    print("3P threshold scan — REUSES existing formal candidate/identifiability results")
    print(f"source                     : {source}")
    print(f"thresholds                 : {a.thresholds}")
    print(f"hard split minima          : train>={a.min_train_states}, val>={a.min_val_states}, test>={a.min_test_states}")
    print(f"selected states hard floor : {min_selected_required}")
    print(f"selected bank cap          : {a.max_selected_states}")
    print(f"selected min separation    : {a.min_separation_rms_snr} RMS-SNR")
    print(f"holdout->train max         : {a.max_train_cover_rms_snr} RMS-SNR")
    print(f"test normalized span min   : {a.min_test_parameter_span}")
    print("NO noise realization is generated.")
    print("=" * 110)

    rows: list[dict] = []
    rng_master = np.random.default_rng(int(a.seed))

    for threshold in a.thresholds:
        safe_idx = np.where(finite & (alias >= float(threshold)))[0]
        row: dict[str, object] = {
            "threshold": float(threshold),
            "alias_safe_candidates": int(len(safe_idx)),
            "hard_min_train": int(a.min_train_states),
            "hard_min_val": int(a.min_val_states),
            "hard_min_test": int(a.min_test_states),
            "hard_min_selected": int(min_selected_required),
            "max_selected_cap": int(a.max_selected_states),
            "min_selected_g_separation_required": float(a.min_separation_rms_snr),
            "max_holdout_to_train_required": float(a.max_train_cover_rms_snr),
            "min_test_parameter_span_required": float(a.min_test_parameter_span),
        }

        if len(safe_idx) == 0:
            row.update({
                "greedy_selected_states": 0,
                "selected_count_pass": False,
                "split_feasible": False,
                "test_coverage_pass": False,
                "formal_feasible": False,
                "failure_reason": "no alias-safe candidates",
            })
            rows.append(row)
            continue

        selected_idx, safe_cover_radius = _greedy_separated_bank(
            params=params,
            g_obs=g_obs,
            rms=rms,
            alias=alias,
            safe_idx=safe_idx,
            cfg=cfg,
            min_sep=float(a.min_separation_rms_snr),
            max_selected=int(a.max_selected_states),
        )
        n_sel = int(len(selected_idx))
        row["greedy_selected_states"] = n_sel
        row["selected_count_pass"] = bool(n_sel >= min_selected_required)
        row["selected_parameter_cover_radius"] = float(safe_cover_radius)

        z_sel = z_all[selected_idx]
        selected_spans = _normalized_spans(z_sel, free_params)
        for name, value in selected_spans.items():
            row[f"selected_span_{name}"] = value

        if n_sel > 1:
            g_sel = g_obs[selected_idx]
            rms_sel = rms[selected_idx]
            dmat = _pairwise_snr_matrix(g_sel, rms_sel, ref_noise)
            tmp = dmat.copy()
            np.fill_diagonal(tmp, np.inf)
            row["selected_min_g_separation"] = float(np.min(tmp))
        else:
            g_sel = g_obs[selected_idx]
            rms_sel = rms[selected_idx]
            dmat = np.zeros((n_sel, n_sel), dtype=np.float64)
            row["selected_min_g_separation"] = float("inf")

        if n_sel < min_selected_required:
            row.update({
                "split_feasible": False,
                "train_states": max(0, n_sel - a.min_val_states - a.min_test_states),
                "val_states": 0,
                "test_states": 0,
                "test_coverage_pass": False,
                "formal_feasible": False,
                "failure_reason": "not enough mutually separated alias-safe states for 200/30/30",
            })
            rows.append(row)
            print(
                f"[threshold {threshold:.2f}] safe={len(safe_idx):4d} "
                f"selected={n_sel:4d} -> FAIL selected<{min_selected_required}"
            )
            continue

        # Use exactly the hard-minimum holdout count.  All additional selected states stay in train,
        # which gives the strongest chance of satisfying holdout->train coverage while enforcing 30/30.
        holdout_count = int(a.min_val_states + a.min_test_states)
        adj = (dmat <= float(a.max_train_cover_rms_snr) + 1.0e-12)
        np.fill_diagonal(adj, False)

        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        holdout, cover_meta = _build_coverable_holdout(
            adj=adj,
            z=z_sel,
            holdout_count=holdout_count,
            trials=int(a.holdout_search_trials),
            rng=rng,
        )
        row.update(cover_meta)

        if holdout is None:
            row.update({
                "split_feasible": False,
                "train_states": int(n_sel - holdout_count),
                "val_states": int(a.min_val_states),
                "test_states": int(a.min_test_states),
                "max_holdout_to_train_rms_snr": float("inf"),
                "test_coverage_pass": False,
                "formal_feasible": False,
                "failure_reason": "no coverable 30+30 holdout subset under RMS-SNR<=2.0",
            })
            rows.append(row)
            print(
                f"[threshold {threshold:.2f}] safe={len(safe_idx):4d} selected={n_sel:4d} "
                f"-> FAIL no coverable 60-state holdout"
            )
            continue

        holdout_mask = np.zeros(n_sel, dtype=bool)
        holdout_mask[holdout] = True
        train = np.where(~holdout_mask)[0]
        val, test, assign_meta = _assign_val_test(
            holdout=holdout,
            z=z_sel,
            n_val=int(a.min_val_states),
            n_test=int(a.min_test_states),
            trials=int(a.assignment_trials),
            rng=rng,
        )
        row.update(assign_meta)

        max_cover = float(np.max(np.min(dmat[np.ix_(holdout, train)], axis=1)))
        row["max_holdout_to_train_rms_snr"] = max_cover
        row["train_states"] = int(len(train))
        row["val_states"] = int(len(val))
        row["test_states"] = int(len(test))

        val_spans = _normalized_spans(z_sel[val], free_params)
        test_spans = _normalized_spans(z_sel[test], free_params)
        for name in free_params:
            row[f"val_span_{name}"] = float(val_spans[name])
            row[f"test_span_{name}"] = float(test_spans[name])

        test_coverage_pass = all(float(test_spans[name]) + 1.0e-12 >= float(a.min_test_parameter_span) for name in free_params)
        split_feasible = (
            len(train) >= int(a.min_train_states)
            and len(val) >= int(a.min_val_states)
            and len(test) >= int(a.min_test_states)
            and max_cover <= float(a.max_train_cover_rms_snr) + 1.0e-8
        )
        formal_feasible = bool(
            row["selected_count_pass"]
            and split_feasible
            and test_coverage_pass
            and float(row["selected_min_g_separation"]) + 1.0e-8 >= float(a.min_separation_rms_snr)
            and float(row["selected_parameter_cover_radius"]) <= float(a.max_parameter_cover_radius) + 1.0e-12
        )
        row["split_feasible"] = bool(split_feasible)
        row["test_coverage_pass"] = bool(test_coverage_pass)
        row["formal_feasible"] = formal_feasible
        row["failure_reason"] = "" if formal_feasible else (
            "test parameter span below requirement" if not test_coverage_pass
            else "selected parameter cover radius above requirement" if float(row["selected_parameter_cover_radius"]) > float(a.max_parameter_cover_radius)
            else "hard split requirement failed"
        )

        tag = f"{threshold:.2f}".replace(".", "p")
        _write_selected_preview(out / f"selected_preview_t{tag}.csv", candidate_rows, selected_idx, alias)
        _write_split_preview(out / f"split_preview_t{tag}.csv", candidate_rows, selected_idx, train, val, test)

        rows.append(row)
        print(
            f"[threshold {threshold:.2f}] safe={len(safe_idx):4d} selected={n_sel:4d} "
            f"split={len(train)}/{len(val)}/{len(test)} cover={max_cover:.4g} "
            f"testSpanMin={min(test_spans.values()):.3f} "
            f"{'PASS' if formal_feasible else 'FAIL'}"
        )

    summary_csv = out / "threshold_scan_summary.csv"
    write_csv_rows(summary_csv, rows)

    feasible = [r for r in rows if bool(r.get("formal_feasible", False))]
    recommended = max((float(r["threshold"]) for r in feasible), default=None)
    summary = {
        "status": "PASS" if feasible else "NO_FEASIBLE_THRESHOLD_IN_SCAN",
        "source_output": str(source),
        "thresholds": [float(x) for x in a.thresholds],
        "hard_requirements": {
            "min_train_states": int(a.min_train_states),
            "min_val_states": int(a.min_val_states),
            "min_test_states": int(a.min_test_states),
            "min_selected_states": int(min_selected_required),
            "min_selected_g_separation_rms_snr": float(a.min_separation_rms_snr),
            "max_parameter_cover_radius": float(a.max_parameter_cover_radius),
            "max_holdout_to_train_rms_snr": float(a.max_train_cover_rms_snr),
            "min_test_parameter_span_fraction": float(a.min_test_parameter_span),
        },
        "recommended_highest_feasible_threshold": recommended,
        "note": (
            "The recommendation is the highest scanned cutoff satisfying all data-side hard requirements. "
            "It does not freeze a training dataset; run the formal 3P pipeline only after team review."
        ),
        "rows": rows,
    }
    (out / "threshold_scan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 110)
    print(f"[done] {summary_csv}")
    print(f"[done] {out / 'threshold_scan_summary.json'}")
    if recommended is None:
        print("RESULT: no scanned threshold satisfies all hard requirements.")
        print("Do NOT lower other hard constraints automatically; review the summary first.")
    else:
        print(f"RESULT: highest scanned data-side feasible threshold = {recommended:.2f}")
        print("This is a scan result only; no noisy training dataset was generated.")
    print("=" * 110)
    return out


def main() -> None:
    a = parse_args()
    run(a)


if __name__ == "__main__":
    main()
