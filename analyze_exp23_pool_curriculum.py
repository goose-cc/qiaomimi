#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate Exp23 formal pool-backed curriculum results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate Exp23 formal pool-backed results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", default="model/exp23")
    p.add_argument("--output", default="validation_results/exp23_pool_curriculum_summary.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output)
    if not root.exists():
        raise FileNotFoundError(root)

    rows: list[dict[str, Any]] = []
    for best_path in sorted(root.rglob("validation_best.json")):
        run_dir = best_path.parent
        summary_path = run_dir / "training_summary.json"
        if not summary_path.exists():
            continue
        best = json.loads(best_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        clean = best.get("validation_0pct", {})
        noisy = best.get("validation_train_noise", {})

        rows.append({
            "run": str(run_dir.relative_to(root)),
            "stage": summary.get("stage"),
            "model_type": summary.get("model_type", summary.get("model")),
            "selection_strategy": summary.get("selection_strategy"),
            "free_parameters": ";".join(summary.get("free_parameters", [])),
            "pool_usable_rows": summary.get("pool_usable_rows"),
            "val_pool_usable_rows": summary.get("val_pool_usable_rows"),
            "coverage_bins": summary.get("coverage_bins"),
            "candidate_multiplier": summary.get("candidate_multiplier"),
            "soft_target_g_gap_percent": summary.get("soft_target_g_gap_percent"),
            "training_noise_percent": 100.0 * float(summary.get("training_noise_level", 0.0)),
            "selected_training_samples": summary.get("selected_training_samples"),
            "pool_rows_read_this_run": summary.get("pool_rows_read_this_run"),
            "mean_batch_min_g_gap_percent": summary.get("mean_batch_min_g_gap_percent"),
            "median_batch_min_g_gap_percent": summary.get("median_batch_min_g_gap_percent"),
            "mean_batch_median_nearest_g_gap_percent": summary.get("mean_batch_median_nearest_g_gap_percent"),
            "g_gap_target_met_fraction": summary.get("target_met_fraction"),
            "parameter_bin_coverage_fraction": summary.get("mean_parameter_bin_coverage_fraction"),
            "mean_unique_joint_fraction_of_batch": summary.get("mean_unique_joint_fraction_of_batch"),
            "final_global_joint_bins_coverage_fraction": summary.get("final_global_joint_bins_coverage_fraction"),
            "best_step": best.get("step"),
            "val_0pct_f_relative_l2": clean.get("f_relative_l2_mean"),
            "val_train_noise_f_relative_l2": noisy.get("f_relative_l2_mean"),
            "val_0pct_resonance_window_relative_l2": clean.get("resonance_window_relative_l2_mean"),
            "val_train_noise_resonance_window_relative_l2": noisy.get("resonance_window_relative_l2_mean"),
            "val_0pct_peak_height_relative_error": clean.get("resonance_peak_height_relative_error_mean"),
            "val_train_noise_peak_height_relative_error": noisy.get("resonance_peak_height_relative_error_mean"),
            "val_0pct_physics_g_relative_l2": clean.get("physics_g_relative_l2_mean"),
            "val_train_noise_physics_g_relative_l2": noisy.get("physics_g_relative_l2_mean"),
            # If a parameter-regression diagnostic is later run, keep these columns compatible too.
            "val_0pct_gamma_median_factor": clean.get("gamma_median_factor"),
            "val_train_noise_gamma_median_factor": noisy.get("gamma_median_factor"),
            "val_0pct_gamma_within_x1p2": clean.get("gamma_within_x1p2"),
            "val_train_noise_gamma_within_x1p2": noisy.get("gamma_within_x1p2"),
            "val_0pct_free_mae": clean.get("free_parameter_mae_mean"),
            "val_train_noise_free_mae": noisy.get("free_parameter_mae_mean"),
        })

    if not rows:
        raise RuntimeError(f"No completed Exp23 runs found under {root}")

    fields = list(rows[0].keys())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("=" * 132)
    print("Exp23 formal summary")
    print(
        f"{'run':38s} | {'g-min%':>8s} | {'marg cov':>8s} | {'joint':>7s} | "
        f"{'f-rel':>9s} | {'res-window':>10s} | {'peak err':>9s} | {'g-rel':>9s}"
    )
    print("-" * 132)

    def fmt(v, scale=1.0, digits=4):
        try:
            return f"{float(v)*scale:.{digits}g}"
        except (TypeError, ValueError):
            return "nan"

    for r in rows:
        print(
            f"{r['run'][:38]:38s} | "
            f"{fmt(r['mean_batch_min_g_gap_percent']):>8s} | "
            f"{fmt(r['parameter_bin_coverage_fraction'],100)+'%':>8s} | "
            f"{fmt(r['mean_unique_joint_fraction_of_batch'],100)+'%':>7s} | "
            f"{fmt(r['val_train_noise_f_relative_l2']):>9s} | "
            f"{fmt(r['val_train_noise_resonance_window_relative_l2']):>10s} | "
            f"{fmt(r['val_train_noise_peak_height_relative_error']):>9s} | "
            f"{fmt(r['val_train_noise_physics_g_relative_l2']):>9s}"
        )
    print("-" * 132)
    print(f"CSV: {output}")
    print("For the A/B test: coverage should stay similar; g-diverse should increase g-min; then compare unseen validation f/resonance metrics.")
    print("=" * 132)


if __name__ == "__main__":
    main()
