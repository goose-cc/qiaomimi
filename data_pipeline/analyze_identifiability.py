from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline_core import (
    PARAMETER_NAMES,
    default_output_dir,
    find_candidate_aliases,
    continuous_profile_aliases,
    load_config,
    parameter_matrix_from_rows,
    quantiles,
    read_csv_rows,
    update_metadata,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generalized candidate-pool identifiability analysis")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--backend", choices=("auto", "exact_kdtree", "projected_kdtree"), default="auto")
    return p.parse_args()


def run(config_path: str, output_dir: str | None = None, backend: str = "auto") -> Path:
    cfg = load_config(config_path)
    out = Path(output_dir) if output_dir else default_output_dir(cfg)
    rows = read_csv_rows(out / "candidate_states.csv")
    params = parameter_matrix_from_rows(rows)
    state_id = np.asarray([r["state_id"] for r in rows])
    bank = np.load(out / "candidate_g_clean.npz", allow_pickle=False)
    g = np.asarray(bank["g_clean"], dtype=np.float32)
    obs_idx = np.asarray(bank["observation_indices"], dtype=np.int32)
    if len(g) != len(params):
        raise RuntimeError("candidate state table and g bank row counts differ")

    alias_rows, search_meta = find_candidate_aliases(
        params=params, g_master=g, obs_idx=obs_idx, state_id=state_id, cfg=cfg, backend=backend
    )
    profiled_rows, profile_meta = continuous_profile_aliases(
        params=params, g_master=g, obs_idx=obs_idx, state_id=state_id, cfg=cfg
    )
    if profiled_rows:
        for row, prow in zip(alias_rows, profiled_rows):
            candidate_score = float(row["alias_score"])
            profiled_score = float(prow["profiled_alias_rms_snr"])
            row["candidate_alias_score"] = candidate_score
            row["nearest_candidate_alias_state_id"] = row["nearest_alias_state_id"]
            row.update(prow)
            if profiled_score < candidate_score:
                row["alias_score"] = profiled_score
                row["nearest_alias_rms_snr"] = profiled_score
                row["g_rms_distance"] = float(prow["profiled_g_rms_distance"])
                row["nearest_alias_state_id"] = "PROFILED_CONTINUOUS"
                row["alias_source"] = "continuous_profile"
            else:
                row["alias_source"] = "candidate_pool"
    else:
        for row in alias_rows:
            row["candidate_alias_score"] = float(row["alias_score"])
            row["nearest_candidate_alias_state_id"] = row["nearest_alias_state_id"]
            row["alias_source"] = "candidate_pool"
    write_csv_rows(out / "identifiability_results.csv", alias_rows)

    score = np.asarray([float(r["alias_score"]) for r in alias_rows], dtype=np.float64)
    finite = score[np.isfinite(score)]
    stats = quantiles(finite)
    scan = [float(x) for x in cfg["identifiability"].get("threshold_scan", [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])]
    summary_rows = []
    for threshold in scan:
        safe = np.isfinite(score) & (score >= threshold)
        summary_rows.append({
            "alias_threshold": threshold,
            "candidate_state_count": int(len(score)),
            "resolved_state_count": int(np.sum(np.isfinite(score))),
            "alias_safe_count": int(np.sum(safe)),
            "alias_safe_fraction": float(np.mean(safe)),
            **{f"alias_score_{k}": v for k, v in stats.items()},
            "reference_noise": float(cfg["identifiability"]["reference_noise"]),
            "search_backend": search_meta["backend"],
            "search_exact": bool(search_meta["search_exact"]),
            "continuous_profile_enabled": bool(profile_meta.get("enabled", False)),
        })
    write_csv_rows(out / "alias_summary.csv", summary_rows)

    hard_count = min(len(alias_rows), int(cfg["identifiability"].get("hard_state_count", 200)))
    order = np.argsort(np.where(np.isfinite(score), score, np.inf))[:hard_count]
    hard_rows = []
    for rank, idx in enumerate(order, 1):
        src = alias_rows[int(idx)]
        row = {
            "hard_rank": rank,
            "state_id": src["state_id"],
            "nearest_alias_state_id": src["nearest_alias_state_id"],
            "alias_score": src["alias_score"],
            "alias_source": src.get("alias_source", "candidate_pool"),
            "profiled_alias_physical": src.get("profiled_alias_physical", ""),
            "parameter_distance": src["normalized_parameter_distance"],
            "raw_parameter_distance": src["raw_parameter_distance"],
            "g_distance": src["g_rms_distance"],
            "g_rms_snr": src["nearest_alias_rms_snr"],
        }
        for name in PARAMETER_NAMES:
            row[name] = src[name]
            row[f"alias_{name}"] = src[f"alias_{name}"]
            row[f"delta_{name}"] = src[f"delta_{name}"]
        row["gamma_factor"] = src["gamma_factor"]
        hard_rows.append(row)
    write_csv_rows(out / "hard_states.csv", hard_rows)

    update_metadata(out, {
        "status": "IDENTIFIABILITY_READY",
        "identifiability": {
            "reference_noise": float(cfg["identifiability"]["reference_noise"]),
            "unacceptable_difference": dict(cfg["identifiability"]["unacceptable_difference"]),
            "search": search_meta,
            "continuous_profile": profile_meta,
            "alias_score_summary": stats,
            "threshold_scan": scan,
        },
    })
    print("=" * 100)
    print(f"stage                 : {cfg['stage']}")
    print(f"candidate states      : {len(score)}")
    print(f"search backend        : {search_meta['backend']} (exact={search_meta['search_exact']})")
    print(f"unresolved aliases    : {search_meta['unresolved_count']}")
    print(f"continuous profile    : {profile_meta.get('enabled', False)}")
    print(f"alias score median    : {stats['p50']:.6g}")
    for row in summary_rows:
        print(f"alias >= {row['alias_threshold']:.3g}: {row['alias_safe_count']} / {len(score)}")
    print("=" * 100)
    print(f"[done] {out / 'identifiability_results.csv'}")
    print(f"[done] {out / 'alias_summary.csv'}")
    print(f"[done] {out / 'hard_states.csv'}")
    return out


def main() -> None:
    a = parse_args()
    run(a.config, a.output_dir, a.backend)


if __name__ == "__main__":
    main()
