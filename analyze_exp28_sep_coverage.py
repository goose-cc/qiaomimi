#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp28 aggregate diagnostics and stage-gate report.

Exp27 threshold=1.5 and Exp28 use the same globally separated physical state
bank but different split policies.  Their native test-state identities are not
identical, so the baseline-vs-Exp28 table is an aggregate split-policy
comparison, not a paired per-state test.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze Exp28 separation + coverage experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--result-dir", required=True)
    p.add_argument("--result-prefix", default="exp28")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--noise-dir", default="noise_0p2pct")
    p.add_argument(
        "--exp27-scan-summary",
        default="validation_results/exp27_gsep_threshold_scan/exp27_threshold_scan_summary.csv",
    )
    p.add_argument("--baseline-threshold", type=float, default=1.5)
    p.add_argument("--target-gamma-within-x1p2", type=float, default=0.95)
    p.add_argument("--target-catastrophic-ge5", type=float, default=0.0)
    p.add_argument("--target-median-f-rel-l2", type=float, default=0.05)
    return p.parse_args()


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    result_dir = Path(args.result_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    coverage = pd.read_csv(data_dir / "coverage_summary.csv")
    result = pd.read_csv(result_dir / (args.result_prefix + "_summary.csv"))
    samples = pd.read_csv(result_dir / (args.result_prefix + "_samples.csv"))

    row = result[result["noise_dir"] == args.noise_dir]
    if row.empty:
        raise ValueError("no %s row in Exp28 summary" % args.noise_dir)
    row = row.iloc[0]

    ss = samples[samples["noise_dir"] == args.noise_dir].copy()
    if ss.empty:
        raise ValueError("no %s samples in Exp28 results" % args.noise_dir)

    gamma_factor = ss["gamma_factor_error"].to_numpy(float)
    f_rel = ss["f_relative_l2"].to_numpy(float)

    exp28 = {
        "experiment": "Exp28 sep+coverage",
        "min_rms_snr": float(meta["global_min_rms_snr"]),
        "max_holdout_to_train_rms_snr": float(
            coverage[coverage["split"] == "holdout_all"][
                "max_nearest_train_rms_snr"
            ].iloc[0]
        ),
        "physical_state_count": int(meta["selected_state_count"]),
        "train_state_count": int(meta["selected_state_counts"]["train"]),
        "val_state_count": int(meta["selected_state_counts"]["val"]),
        "test_state_count": int(meta["selected_state_counts"]["test"]),
        "gamma_median_factor": float(row["gamma_median_factor"]),
        "gamma_p90_factor": float(row["gamma_p90_factor"]),
        "gamma_p99_factor": float(np.quantile(gamma_factor, 0.99)),
        "gamma_max_factor": float(np.max(gamma_factor)),
        "gamma_within_x1p2": float(row["gamma_within_x1p2"]),
        "gamma_within_x1p5": float(row["gamma_within_x1p5"]),
        "gamma_within_x2": float(row["gamma_within_x2"]),
        "a1_median_norm_error": float(row["second_median_norm_error"]),
        "a1_p90_norm_error": float(row["second_p90_norm_error"]),
        "median_f_relative_l2": float(row["median_f_relative_l2"]),
        "p90_f_relative_l2": float(row["p90_f_relative_l2"]),
        "p99_f_relative_l2": float(np.quantile(f_rel, 0.99)),
        "median_g_clean_relative_l2": float(row["median_g_clean_relative_l2"]),
        "catastrophic_gamma_factor_ge5": float(row["catastrophic_gamma_factor_ge5"]),
        "catastrophic_gamma_factor_ge10": float(row["catastrophic_gamma_factor_ge10"]),
    }

    comparison = [exp28]
    baseline_path = Path(args.exp27_scan_summary)
    if baseline_path.exists():
        bdf = pd.read_csv(baseline_path)
        idx = np.argmin(np.abs(
            bdf["requested_min_rms_snr"].to_numpy(float) - float(args.baseline_threshold)
        ))
        b = bdf.iloc[int(idx)]
        comparison.insert(0, {
            "experiment": "Exp27 min-sep-only aggregate",
            "min_rms_snr": float(b["actual_global_min_rms_snr"]),
            "max_holdout_to_train_rms_snr": math.nan,
            "physical_state_count": int(b["selected_state_count"]),
            "train_state_count": int(b["train_state_count"]),
            "val_state_count": int(b["val_state_count"]),
            "test_state_count": int(b["test_state_count"]),
            "gamma_median_factor": float(b["gamma_median_factor"]),
            "gamma_p90_factor": float(b["gamma_p90_factor"]),
            "gamma_p99_factor": float(b["gamma_p99_factor"]),
            "gamma_max_factor": float(b["gamma_max_factor"]),
            "gamma_within_x1p2": float(b["gamma_within_x1p2"]),
            "gamma_within_x1p5": float(b["gamma_within_x1p5"]),
            "gamma_within_x2": float(b["gamma_within_x2"]),
            "a1_median_norm_error": math.nan,
            "a1_p90_norm_error": math.nan,
            "median_f_relative_l2": float(b["median_f_relative_l2"]),
            "p90_f_relative_l2": float(b["p90_f_relative_l2"]),
            "p99_f_relative_l2": float(b["p99_f_relative_l2"]),
            "median_g_clean_relative_l2": float(b["median_g_clean_relative_l2"]),
            "catastrophic_gamma_factor_ge5": float(b["catastrophic_gamma_factor_ge5"]),
            "catastrophic_gamma_factor_ge10": float(b["catastrophic_gamma_factor_ge10"]),
        })

    write_csv(out / "exp28_comparison_summary.csv", comparison)

    stage_gate = [
        {
            "criterion": "gamma within x1.2",
            "target": float(args.target_gamma_within_x1p2),
            "actual": exp28["gamma_within_x1p2"],
            "pass": int(exp28["gamma_within_x1p2"] >= args.target_gamma_within_x1p2),
            "meaning": "provisional continuous-parameter accuracy gate",
        },
        {
            "criterion": "catastrophic gamma factor >=5",
            "target": float(args.target_catastrophic_ge5),
            "actual": exp28["catastrophic_gamma_factor_ge5"],
            "pass": int(exp28["catastrophic_gamma_factor_ge5"] <= args.target_catastrophic_ge5 + 1e-12),
            "meaning": "no severe branch-jump failures",
        },
        {
            "criterion": "median f relative L2",
            "target": float(args.target_median_f_rel_l2),
            "actual": exp28["median_f_relative_l2"],
            "pass": int(exp28["median_f_relative_l2"] <= args.target_median_f_rel_l2),
            "meaning": "typical reconstructed spectral-shape quality",
        },
    ]
    write_csv(out / "exp28_stage_gate.csv", stage_gate)

    # Worst cases are still important because median can hide narrow-peak failures.
    worst = ss.sort_values("gamma_factor_error", ascending=False).head(20).copy()
    keep = [
        "true_a1", "pred_a1", "true_gamma", "pred_gamma",
        "gamma_factor_error", "f_relative_l2", "g_clean_relative_l2",
    ]
    worst[keep].to_csv(
        out / "exp28_top20_worst_gamma_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Coverage distribution.
        c = pd.read_csv(data_dir / "holdout_to_train_coverage.csv")
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in ("val", "test"):
            x = c[c["split"] == name]["nearest_train_rms_snr"]
            ax.hist(x, bins=15, alpha=0.55, label=name)
        ax.axvline(meta["global_min_rms_snr"], linestyle="--", label="min separation")
        ax.axvline(meta["max_train_cover_rms_snr"], linestyle=":", label="max coverage")
        ax.set_xlabel("holdout -> nearest train RMS-SNR")
        ax.set_ylabel("physical state count")
        ax.set_title("Exp28: holdouts are distinguishable but still covered")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "exp28_holdout_coverage.png", dpi=170)
        plt.close(fig)

        if len(comparison) == 2:
            cdf = pd.DataFrame(comparison)
            labels = ["Exp27 sep-only", "Exp28 sep+coverage"]
            x = np.arange(2)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - 0.18, cdf["gamma_within_x1p2"], width=0.36, label="gamma within x1.2")
            ax.bar(x + 0.18, cdf["catastrophic_gamma_factor_ge5"], width=0.36, label="factor >=5")
            ax.set_xticks(x, labels)
            ax.set_ylabel("fraction")
            ax.set_title("Exp28: does coverage help at the same ~1.5 separation?")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / "exp28_gamma_comparison.png", dpi=170)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - 0.18, cdf["median_f_relative_l2"], width=0.36, label="median")
            ax.bar(x + 0.18, cdf["p90_f_relative_l2"], width=0.36, label="p90")
            ax.set_xticks(x, labels)
            ax.set_ylabel("f relative L2")
            ax.set_title("Exp28 f reconstruction: separation-only vs separation+coverage")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / "exp28_f_comparison.png", dpi=170)
            plt.close(fig)
    except Exception as exc:
        print("warning: plotting skipped: %s" % exc)

    print("=" * 100)
    print("Exp28 analysis ready")
    print("Provisional stage gate: continuous gamma within x1.2 >= %.1f%%"
          % (100 * args.target_gamma_within_x1p2))
    print("This is a tolerance-based engineering milestone, not literal classification accuracy.")
    print("Read first:")
    print("  %s" % (out / "exp28_comparison_summary.csv"))
    print("  %s" % (out / "exp28_stage_gate.csv"))
    print("  %s" % (out / "exp28_top20_worst_gamma_cases.csv"))
    print("=" * 100)


if __name__ == "__main__":
    main()
