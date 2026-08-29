from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from pipeline_core import (
    default_output_dir,
    load_config,
    normalized_parameter_matrix,
    parameter_matrix_from_rows,
    quantiles,
    read_csv_rows,
    update_metadata,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze parameter-space coverage before/after identifiability selection")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--bins", type=int)
    return p.parse_args()


def _stats_rows(label: str, params: np.ndarray, cfg: dict) -> list[dict]:
    rows = []
    for name in cfg["free_params"]:
        x = params[:, ("a1", "a2", "a3", "m", "gamma").index(name)].astype(np.float64)
        q = quantiles(x)
        rows.append({
            "dataset": label,
            "parameter": name,
            "count": int(len(x)),
            "min": q["min"], "max": q["max"], "mean": q["mean"],
            "std": float(np.std(x)), "p05": q["p05"], "p25": q["p25"],
            "p50": q["p50"], "p75": q["p75"], "p95": q["p95"],
        })
    return rows


def _projection_rows(params_all: np.ndarray, params_sel: np.ndarray, cfg: dict, bins: int) -> list[dict]:
    free = list(cfg["free_params"])
    z_all = normalized_parameter_matrix(params_all, cfg, free)
    z_sel = normalized_parameter_matrix(params_sel, cfg, free)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows = []
    for i, j in itertools.combinations(range(len(free)), 2):
        ha, _, _ = np.histogram2d(z_all[:, i], z_all[:, j], bins=(edges, edges))
        hs, _, _ = np.histogram2d(z_sel[:, i], z_sel[:, j], bins=(edges, edges))
        cand_occ = ha > 0
        sel_occ = hs > 0
        relevant = int(np.sum(cand_occ))
        retained = int(np.sum(cand_occ & sel_occ))
        rows.append({
            "parameter_x": free[i],
            "parameter_y": free[j],
            "bins_per_axis": int(bins),
            "candidate_occupied_bins": relevant,
            "selected_occupied_bins": int(np.sum(sel_occ)),
            "candidate_bins_retained": retained,
            "projection_coverage_fraction": float(retained / max(relevant, 1)),
        })
    return rows


def run(config_path: str, output_dir: str | None = None, bins: int | None = None) -> Path:
    cfg = load_config(config_path)
    out = Path(output_dir) if output_dir else default_output_dir(cfg)
    cand_rows = read_csv_rows(out / "candidate_states.csv")
    sel_rows = read_csv_rows(out / "selected_states.csv")
    cand = parameter_matrix_from_rows(cand_rows)
    sel = parameter_matrix_from_rows(sel_rows)
    bins = int(bins or cfg.get("coverage", {}).get("projection_bins", 10))

    stats_rows = _stats_rows("candidate", cand, cfg) + _stats_rows("selected", sel, cfg)
    write_csv_rows(out / "parameter_coverage_statistics.csv", stats_rows)
    projection_rows = _projection_rows(cand, sel, cfg, bins)
    write_csv_rows(out / "parameter_projection_coverage.csv", projection_rows)

    meta = __import__("json").loads((out / "metadata.json").read_text(encoding="utf-8"))
    selection = meta.get("selection", {})
    summary = {
        "stage": cfg["stage"],
        "candidate_state_count": int(len(cand)),
        "selected_state_count": int(len(sel)),
        "retention_rate": float(len(sel) / max(len(cand), 1)),
        "alias_threshold": selection.get("alias_threshold"),
        "min_separation_rms_snr": selection.get("min_separation_rms_snr"),
        "actual_min_selected_g_separation": selection.get("actual_min_selected_g_separation"),
        "max_parameter_cover_radius": selection.get("max_parameter_cover_radius"),
        "actual_parameter_cover_radius": selection.get("actual_parameter_cover_radius"),
        "noise_scale_used_for_alias_score": cfg["identifiability"]["reference_noise"],
    }
    for name in cfg["free_params"]:
        k = ("a1", "a2", "a3", "m", "gamma").index(name)
        summary[f"candidate_{name}_min"] = float(np.min(cand[:, k]))
        summary[f"candidate_{name}_max"] = float(np.max(cand[:, k]))
        summary[f"selected_{name}_min"] = float(np.min(sel[:, k]))
        summary[f"selected_{name}_max"] = float(np.max(sel[:, k]))
    write_csv_rows(out / "coverage_summary.csv", [summary])

    update_metadata(out, {
        "status": "COVERAGE_READY",
        "coverage": {
            "projection_bins": bins,
            "candidate_state_count": int(len(cand)),
            "selected_state_count": int(len(sel)),
            "retention_rate": float(len(sel) / max(len(cand), 1)),
            "minimum_projection_coverage_fraction": float(min((r["projection_coverage_fraction"] for r in projection_rows), default=1.0)),
        },
    })
    print(f"[done] {out / 'coverage_summary.csv'}")
    print(f"[done] {out / 'parameter_coverage_statistics.csv'}")
    print(f"[done] {out / 'parameter_projection_coverage.csv'}")
    return out


def main() -> None:
    a = parse_args()
    run(a.config, a.output_dir, a.bins)


if __name__ == "__main__":
    main()
