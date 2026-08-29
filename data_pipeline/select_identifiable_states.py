from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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
    p.add_argument("--smoke-permissive", action="store_true")
    return p.parse_args()


def _greedy_select(params: np.ndarray, g_obs: np.ndarray, rms: np.ndarray, alias: np.ndarray,
                   safe_idx: np.ndarray, cfg: dict, min_sep: float, max_cover: float,
                   max_selected: int) -> tuple[np.ndarray, dict]:
    z = normalized_parameter_matrix(params, cfg)
    safe_idx = np.asarray(safe_idx, dtype=np.int64)
    if len(safe_idx) == 0:
        return np.empty(0, dtype=np.int64), {
            "actual_parameter_cover_radius": float("inf"),
            "actual_min_selected_g_separation": float("nan"),
            "hit_max_selected_states": False,
        }
    # Start at the safest candidate, breaking ties toward a parameter-space edge.
    center_dist = np.linalg.norm(z[safe_idx] - 0.5, axis=1)
    order_key = alias[safe_idx] + 1e-9 * center_dist
    first = int(safe_idx[int(np.argmax(order_key))])
    selected = [first]
    min_param = np.linalg.norm(z[safe_idx] - z[first], axis=1)
    min_g = symmetric_rms_snr(
        g_obs[safe_idx], rms[safe_idx], g_obs[first][None, :], np.full(len(safe_idx), rms[first]),
        float(cfg["identifiability"]["reference_noise"]),
    )
    min_g[safe_idx == first] = 0.0

    while len(selected) < int(max_selected):
        cover = float(np.max(min_param))
        if cover <= float(max_cover):
            break
        eligible = (min_g >= float(min_sep)) & (~np.isin(safe_idx, np.asarray(selected, dtype=np.int64)))
        if not np.any(eligible):
            break
        # Coverage first, alias safety second.
        score = np.where(eligible, min_param + 1e-6 * alias[safe_idx], -np.inf)
        pick_local = int(np.argmax(score))
        pick = int(safe_idx[pick_local])
        selected.append(pick)
        dparam = np.linalg.norm(z[safe_idx] - z[pick], axis=1)
        min_param = np.minimum(min_param, dparam)
        dg = symmetric_rms_snr(
            g_obs[safe_idx], rms[safe_idx], g_obs[pick][None, :], np.full(len(safe_idx), rms[pick]),
            float(cfg["identifiability"]["reference_noise"]),
        )
        min_g = np.minimum(min_g, dg)
        min_g[np.isin(safe_idx, np.asarray(selected, dtype=np.int64))] = 0.0

    sel = np.asarray(selected, dtype=np.int64)
    actual_cover = float(np.max(np.min(np.linalg.norm(z[safe_idx, None, :] - z[sel][None, :, :], axis=2), axis=1)))
    if len(sel) > 1:
        nearest_sep = []
        for i in range(len(sel)):
            others = np.delete(sel, i)
            s = symmetric_rms_snr(
                np.repeat(g_obs[sel[i]][None, :], len(others), axis=0),
                np.full(len(others), rms[sel[i]]),
                g_obs[others], rms[others],
                float(cfg["identifiability"]["reference_noise"]),
            )
            nearest_sep.append(float(np.min(s)))
        actual_min_sep = float(np.min(nearest_sep))
    else:
        actual_min_sep = float("inf")
    return sel, {
        "actual_parameter_cover_radius": actual_cover,
        "actual_min_selected_g_separation": actual_min_sep,
        "hit_max_selected_states": bool(len(sel) >= int(max_selected)),
    }


def run(config_path: str, output_dir: str | None, alias_threshold: float,
        min_separation_rms_snr: float | None = None,
        max_parameter_cover_radius: float | None = None,
        max_selected_states: int | None = None,
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

    finite = np.isfinite(alias)
    safe_idx = np.where(finite & (alias >= float(alias_threshold)))[0]
    if len(safe_idx) == 0:
        raise RuntimeError(f"no states satisfy alias_threshold={alias_threshold}")
    selected_idx, metrics = _greedy_select(params, g_obs, rms, alias, safe_idx, cfg, min_sep, max_cover, max_sel)
    if len(selected_idx) == 0:
        raise RuntimeError("selection produced zero states")
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
    print(f"selected physical states    : {len(selected_idx)}")
    print(f"actual min g separation     : {metrics['actual_min_selected_g_separation']:.6g}")
    print(f"actual parameter cover rad. : {metrics['actual_parameter_cover_radius']:.6g}")
    print(f"[done] {out / 'selected_states.csv'}")
    print("=" * 100)
    return out


def main() -> None:
    a = parse_args()
    run(a.config, a.output_dir, a.alias_threshold, a.min_separation_rms_snr,
        a.max_parameter_cover_radius, a.max_selected_states, a.scan_thresholds, a.smoke_permissive)


if __name__ == "__main__":
    main()
