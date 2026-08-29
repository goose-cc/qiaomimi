from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from split_physical_states import _try_coverable_holdout

from pipeline_core import (
    PARAMETER_NAMES,
    default_output_dir,
    load_config,
    master_rms,
    normalized_parameter_matrix,
    parameter_matrix_from_rows,
    read_csv_rows,
    symmetric_rms_snr,
    update_metadata,
    write_csv_rows,
)


def parse_float_list(text: str | None) -> list[float]:
    if text is None:
        return []
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter generalized alias-safe states while preserving parameter coverage")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--alias-threshold", type=float, required=True)
    p.add_argument("--scan-thresholds", type=parse_float_list)
    p.add_argument("--min-separation-rms-snr", type=float)
    p.add_argument("--max-parameter-cover-radius", type=float)
    p.add_argument("--max-selected-states", type=int)
    p.add_argument("--min-selected-states", type=int)
    p.add_argument("--smoke-permissive", action="store_true")
    return p.parse_args()


def _safe_pairwise_snr(g_safe: np.ndarray, rms_safe: np.ndarray, reference_noise: float) -> np.ndarray:
    """Exact pairwise RMS-SNR on the alias-safe pool.

    The alias-safe pool is small enough that this exact matrix is cheap.  It lets
    selection reason about later split feasibility without ever constructing an
    N x N matrix on the full candidate pool.
    """
    n = len(g_safe)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if i + 1 >= n:
            break
        d = symmetric_rms_snr(
            np.repeat(g_safe[i][None, :], n - i - 1, axis=0),
            np.full(n - i - 1, rms_safe[i]),
            g_safe[i + 1 :], rms_safe[i + 1 :],
            float(reference_noise),
        )
        out[i, i + 1 :] = d
        out[i + 1 :, i] = d
    return out


def _selection_split_feasibility(
    selected_local: list[int] | np.ndarray,
    safe_pair_snr: np.ndarray,
    z_safe: np.ndarray,
    split_cover_max: float,
    min_holdout: int,
    split_seed: int,
) -> tuple[bool, dict]:
    sel = np.asarray(selected_local, dtype=np.int64)
    if len(sel) <= min_holdout:
        return False, {
            "requested_holdout_count": int(min_holdout),
            "best_holdout_found": 0,
            "cover_graph_nonisolated_count": 0,
            "cover_graph_isolated_count": int(len(sel)),
            "selection_split_feasible": False,
        }

    d = safe_pair_snr[np.ix_(sel, sel)]
    adj = (d <= float(split_cover_max) + 1e-12) & (~np.eye(len(sel), dtype=bool))
    hold, meta = _try_coverable_holdout(
        adj,
        z_safe[sel],
        target_holdout=int(min_holdout),
        seed=int(split_seed) + 104729 * len(sel),
        # This call is only a feasibility check, not final holdout
        # optimization.  The helper returns as soon as it finds one valid
        # 60-state holdout.  512 deterministic seeded restarts are a fallback
        # for sparse cover graphs.
        trials=512,
        stop_on_first=True,
    )
    meta = dict(meta)
    meta["selection_split_feasible"] = bool(hold is not None)
    return bool(hold is not None), meta


def _greedy_select(params: np.ndarray, g_obs: np.ndarray, rms: np.ndarray, alias: np.ndarray,
                   safe_idx: np.ndarray, cfg: dict, min_sep: float, max_cover: float,
                   max_selected: int, min_selected: int) -> tuple[np.ndarray, dict]:
    z = normalized_parameter_matrix(params, cfg)
    safe_idx = np.asarray(safe_idx, dtype=np.int64)

    split_cfg = cfg.get("split", {})
    split_cover_max = float(split_cfg.get("max_train_cover_rms_snr", 2.0))
    min_holdout = int(split_cfg.get("min_val_states", 5)) + int(split_cfg.get("min_test_states", 5))
    split_seed = int(split_cfg.get("seed", cfg.get("sampling", {}).get("seed", 0) + 1000))
    reference_noise = float(cfg["identifiability"]["reference_noise"])

    if len(safe_idx) == 0:
        return np.empty(0, dtype=np.int64), {
            "actual_parameter_cover_radius": float("inf"),
            "actual_min_selected_g_separation": float("nan"),
            "hit_max_selected_states": False,
            "required_min_selected_states": int(min_selected),
            "selection_count_requirement_met": False,
            "stopped_by_no_eligible": False,
            "required_coverable_holdout_states": int(min_holdout),
            "selection_split_feasible": False,
            "best_holdout_found": 0,
        }

    z_safe = z[safe_idx]
    g_safe = g_obs[safe_idx]
    rms_safe = rms[safe_idx]
    safe_pair_snr = _safe_pairwise_snr(g_safe, rms_safe, reference_noise)

    center_dist = np.linalg.norm(z_safe - 0.5, axis=1)
    order_key = alias[safe_idx] + 1e-9 * center_dist
    first_local = int(np.argmax(order_key))
    first = int(safe_idx[first_local])

    selected = [first]
    selected_local = [first_local]
    min_param = np.linalg.norm(z_safe - z_safe[first_local], axis=1)
    min_g = safe_pair_snr[:, first_local].copy()
    min_g[first_local] = 0.0

    cover_potential = np.sum(
        (safe_pair_snr <= split_cover_max + 1e-12)
        & (safe_pair_snr >= float(min_sep) - 1e-12)
        & (~np.eye(len(safe_idx), dtype=bool)),
        axis=1,
    ).astype(np.float64)

    stopped_by_no_eligible = False
    split_ok = False
    split_meta: dict = {
        "requested_holdout_count": int(min_holdout),
        "best_holdout_found": 0,
        "selection_split_feasible": False,
    }

    while len(selected) < int(max_selected):
        cover = float(np.max(min_param))
        base_ready = cover <= float(max_cover) and len(selected) >= int(min_selected)

        if base_ready:
            split_ok, split_meta = _selection_split_feasibility(
                selected_local,
                safe_pair_snr,
                z_safe,
                split_cover_max,
                min_holdout,
                split_seed,
            )
            if split_ok:
                break

        selected_mask = np.zeros(len(safe_idx), dtype=bool)
        selected_mask[np.asarray(selected_local, dtype=np.int64)] = True
        eligible = (min_g >= float(min_sep) - 1e-12) & (~selected_mask)

        if not np.any(eligible):
            stopped_by_no_eligible = True
            break

        if base_ready and not split_ok:
            # Once the count/coverage requirements are already satisfied, the only
            # remaining hard requirement is split feasibility. Prefer companions
            # that connect currently isolated selected states under the formal
            # holdout->train threshold, while still obeying min_sep.
            sel_loc = np.asarray(selected_local, dtype=np.int64)
            sub = safe_pair_snr[np.ix_(sel_loc, sel_loc)]
            adj = (sub <= split_cover_max + 1e-12) & (~np.eye(len(sel_loc), dtype=bool))
            isolated = np.sum(adj, axis=1) == 0

            d_to_sel = safe_pair_snr[:, sel_loc]
            edge_mask = d_to_sel <= split_cover_max + 1e-12
            edge_count = np.sum(edge_mask, axis=1).astype(np.float64)
            if np.any(isolated):
                isolated_connections = np.sum(edge_mask[:, isolated], axis=1).astype(np.float64)
            else:
                isolated_connections = np.zeros(len(safe_idx), dtype=np.float64)

            score = (
                1.0e6 * isolated_connections
                + 1.0e4 * edge_count
                + 1.0e2 * cover_potential
                + min_param
                + 1.0e-6 * alias[safe_idx]
            )
            score = np.where(eligible, score, -np.inf)
        else:
            # Preserve the original coverage-first behavior, but mildly prefer
            # candidates with future split support.
            score = np.where(
                eligible,
                min_param + 1.0e-3 * cover_potential + 1.0e-6 * alias[safe_idx],
                -np.inf,
            )

        pick_local = int(np.argmax(score))
        pick = int(safe_idx[pick_local])
        selected.append(pick)
        selected_local.append(pick_local)

        dparam = np.linalg.norm(z_safe - z_safe[pick_local], axis=1)
        min_param = np.minimum(min_param, dparam)
        min_g = np.minimum(min_g, safe_pair_snr[:, pick_local])
        min_g[np.asarray(selected_local, dtype=np.int64)] = 0.0

    sel = np.asarray(selected, dtype=np.int64)
    actual_cover = float(
        np.max(
            np.min(
                np.linalg.norm(z_safe[:, None, :] - z[sel][None, :, :], axis=2),
                axis=1,
            )
        )
    )

    if len(sel) > 1:
        sl = np.asarray(selected_local, dtype=np.int64)
        sub = safe_pair_snr[np.ix_(sl, sl)].copy()
        np.fill_diagonal(sub, np.inf)
        actual_min_sep = float(np.min(sub))
    else:
        actual_min_sep = float("inf")

    split_ok, split_meta = _selection_split_feasibility(
        selected_local,
        safe_pair_snr,
        z_safe,
        split_cover_max,
        min_holdout,
        split_seed,
    )

    return sel, {
        "actual_parameter_cover_radius": actual_cover,
        "actual_min_selected_g_separation": actual_min_sep,
        "hit_max_selected_states": bool(len(sel) >= int(max_selected)),
        "required_min_selected_states": int(min_selected),
        "selection_count_requirement_met": bool(len(sel) >= int(min_selected)),
        "stopped_by_no_eligible": bool(stopped_by_no_eligible),
        "required_coverable_holdout_states": int(min_holdout),
        "split_cover_max_rms_snr": float(split_cover_max),
        **split_meta,
    }


def run(config_path: str, output_dir: str | None, alias_threshold: float,
        min_separation_rms_snr: float | None = None,
        max_parameter_cover_radius: float | None = None,
        max_selected_states: int | None = None,
        min_selected_states: int | None = None,
        scan_thresholds: list[float] | None = None,
        smoke_permissive: bool = False) -> Path:
    cfg = load_config(config_path)
    out = Path(output_dir) if output_dir else default_output_dir(cfg)
    cand_rows = read_csv_rows(out / "candidate_states.csv")
    ident_rows = read_csv_rows(out / "identifiability_results.csv")
    if len(cand_rows) != len(ident_rows):
        raise RuntimeError("candidate and identifiability row counts differ")
    params = parameter_matrix_from_rows(cand_rows)
    bank = np.load(out / "candidate_g_clean.npz", allow_pickle=False)
    g = np.asarray(bank["g_clean"], dtype=np.float32)
    obs_idx = np.asarray(bank["observation_indices"], dtype=np.int64)
    g_obs = g[:, obs_idx]
    rms = master_rms(g)
    alias = np.asarray([float(r["alias_score"]) for r in ident_rows], dtype=np.float64)

    sel_cfg = cfg.get("selection", {})
    min_sep = float(min_separation_rms_snr if min_separation_rms_snr is not None else sel_cfg.get("min_separation_rms_snr", 1.3))
    max_cover = float(max_parameter_cover_radius if max_parameter_cover_radius is not None else sel_cfg.get("max_parameter_cover_radius", 0.35))
    max_sel = int(max_selected_states if max_selected_states is not None else sel_cfg.get("max_selected_states", 260))
    split_cfg = cfg.get("split", {})
    split_minimum = int(split_cfg.get("min_val_states", 5)) + int(split_cfg.get("min_test_states", 5)) + 1
    min_sel = int(min_selected_states if min_selected_states is not None else sel_cfg.get("min_selected_states", split_minimum))
    if min_sel < split_minimum:
        min_sel = split_minimum
    if max_sel < min_sel:
        raise RuntimeError(f"max_selected_states={max_sel} is smaller than min_selected_states={min_sel}")

    finite = np.isfinite(alias)
    safe_idx = np.where(finite & (alias >= float(alias_threshold)))[0]
    if len(safe_idx) == 0:
        raise RuntimeError(f"no states satisfy alias_threshold={alias_threshold}")
    selected_idx, metrics = _greedy_select(params, g_obs, rms, alias, safe_idx, cfg, min_sep, max_cover, max_sel, min_sel)
    if len(selected_idx) == 0:
        raise RuntimeError("selection produced zero states")
    if len(selected_idx) < min_sel and not smoke_permissive:
        import json
        support = {
            "reason": "not enough mutually separated alias-safe states for requested selected-bank minimum",
            "alias_threshold": float(alias_threshold),
            "alias_safe_candidate_count": int(len(safe_idx)),
            "selected_state_count": int(len(selected_idx)),
            "required_min_selected_states": int(min_sel),
            "min_separation_rms_snr": float(min_sep),
            "max_parameter_cover_radius": float(max_cover),
            "stopped_by_no_eligible": bool(metrics.get("stopped_by_no_eligible", False)),
        }
        (out / "selection_insufficient_support.json").write_text(json.dumps(support, indent=2), encoding="utf-8")
        raise RuntimeError("selected bank is too small; see selection_insufficient_support.json")
    if not smoke_permissive and not bool(metrics.get("selection_split_feasible", False)):
        import json
        support = {
            "reason": "alias-safe separated bank cannot provide the required coverable val/test holdout",
            "alias_threshold": float(alias_threshold),
            "alias_safe_candidate_count": int(len(safe_idx)),
            "selected_state_count": int(len(selected_idx)),
            "required_min_selected_states": int(min_sel),
            "required_coverable_holdout_states": int(metrics.get("required_coverable_holdout_states", 0)),
            "max_train_cover_rms_snr": float(metrics.get("split_cover_max_rms_snr", split_cfg.get("max_train_cover_rms_snr", 2.0))),
            "best_holdout_found": int(metrics.get("best_holdout_found", 0)),
            "cover_graph_nonisolated_count": int(metrics.get("cover_graph_nonisolated_count", 0)),
            "cover_graph_isolated_count": int(metrics.get("cover_graph_isolated_count", 0)),
            "min_separation_rms_snr": float(min_sep),
            "max_parameter_cover_radius": float(max_cover),
            "stopped_by_no_eligible": bool(metrics.get("stopped_by_no_eligible", False)),
            "hit_max_selected_states": bool(metrics.get("hit_max_selected_states", False)),
        }
        (out / "selection_insufficient_support.json").write_text(json.dumps(support, indent=2), encoding="utf-8")
        raise RuntimeError("selected bank is not split-feasible; see selection_insufficient_support.json")

    if not smoke_permissive:
        if metrics["actual_parameter_cover_radius"] > max_cover + 1e-12:
            raise RuntimeError(
                f"parameter coverage failed: actual radius {metrics['actual_parameter_cover_radius']:.6g} > {max_cover:.6g}"
            )
        if len(selected_idx) > 1 and metrics["actual_min_selected_g_separation"] + 1e-8 < min_sep:
            raise RuntimeError("selected-bank g separation guarantee failed")

    selected_rows = []
    selected_set = set(int(x) for x in selected_idx)
    z = normalized_parameter_matrix(params, cfg)
    for rank, idx in enumerate(selected_idx):
        r = cand_rows[int(idx)].copy()
        r["selected_rank"] = rank
        r["alias_score"] = float(alias[idx])
        r["nearest_alias_state_id"] = ident_rows[int(idx)]["nearest_alias_state_id"]
        if len(selected_idx) > 1:
            others = np.asarray([j for j in selected_idx if int(j) != int(idx)], dtype=np.int64)
            s = symmetric_rms_snr(
                np.repeat(g_obs[idx][None, :], len(others), axis=0),
                np.full(len(others), rms[idx]), g_obs[others], rms[others],
                float(cfg["identifiability"]["reference_noise"]),
            )
            r["nearest_selected_rms_snr"] = float(np.min(s))
            r["nearest_selected_state_id"] = cand_rows[int(others[int(np.argmin(s))])]["state_id"]
        else:
            r["nearest_selected_rms_snr"] = float("inf")
            r["nearest_selected_state_id"] = ""
        selected_rows.append(r)
    write_csv_rows(out / "selected_states.csv", selected_rows)

    scan = scan_thresholds or [float(x) for x in cfg["identifiability"].get("threshold_scan", [])]
    scan_rows = []
    for t in scan:
        idx = np.where(finite & (alias >= float(t)))[0]
        scan_rows.append({
            "alias_threshold": float(t),
            "alias_safe_count": int(len(idx)),
            "alias_safe_fraction": float(len(idx) / max(len(alias), 1)),
        })
    write_csv_rows(out / "selection_threshold_scan.csv", scan_rows)

    update_metadata(out, {
        "status": "SELECTED_STATES_READY",
        "selection": {
            "alias_threshold": float(alias_threshold),
            "alias_safe_candidate_count": int(len(safe_idx)),
            "selected_state_count": int(len(selected_idx)),
            "retention_from_candidate": float(len(selected_idx) / len(params)),
            "retention_from_alias_safe": float(len(selected_idx) / len(safe_idx)),
            "min_separation_rms_snr": min_sep,
            "max_parameter_cover_radius": max_cover,
            **metrics,
            "smoke_permissive": bool(smoke_permissive),
        },
    })
    print("=" * 100)
    print(f"alias-safe candidates       : {len(safe_idx)} / {len(params)}")
    print(f"selected physical states    : {len(selected_idx)} (minimum requested {min_sel})")
    print(f"actual min g separation     : {metrics['actual_min_selected_g_separation']:.6g}")
    print(f"actual parameter cover rad. : {metrics['actual_parameter_cover_radius']:.6g}")
    print(f"coverable holdout target    : {metrics.get('required_coverable_holdout_states', 0)}")
    print(f"best coverable holdout      : {metrics.get('best_holdout_found', 0)}")
    print(f"selection split-feasible    : {bool(metrics.get('selection_split_feasible', False))}")
    print(f"[done] {out / 'selected_states.csv'}")
    print("=" * 100)
    return out


def main() -> None:
    a = parse_args()
    run(a.config, a.output_dir, a.alias_threshold, a.min_separation_rms_snr,
        a.max_parameter_cover_radius, a.max_selected_states, a.min_selected_states,
        a.scan_thresholds, a.smoke_permissive)


if __name__ == "__main__":
    main()
