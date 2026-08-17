#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate Exp26-style runs across several minimum g-separation thresholds.

The physics, network and loss stay unchanged.  This script answers the next
question directly:

    If we require the retained physical states to be farther apart in observed
    g-space, do gamma recovery, f recovery and catastrophic failures improve?

Important interpretation
------------------------
Each threshold defines a different identifiable subproblem, so the native test
physical states are not identical across thresholds.  The output therefore
reports both reconstruction quality and the number/support of retained states.
A better score at a larger threshold means the more-separated subproblem is
easier/stabler; it does NOT by itself prove the original full parameter space
has been solved.
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
        description="Analyze Exp27 g-separation threshold scan",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--noise-dir", default="noise_0p2pct")
    return p.parse_args()


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def safe_quantile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else math.nan


def safe_max(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.max(x)) if len(x) else math.nan


def pair_min(pair_df, a, b):
    m = (
        ((pair_df["split_1"] == a) & (pair_df["split_2"] == b))
        | ((pair_df["split_1"] == b) & (pair_df["split_2"] == a))
    )
    if not np.any(m):
        return math.nan
    return float(pair_df.loc[m, "min_rms_snr"].iloc[0])


def plot_lines(path, df, columns, ylabel, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("warning: plotting skipped: %s" % exc)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label in columns:
        ax.plot(df["requested_min_rms_snr"], df[col], marker="o", label=label)
    ax.set_xlabel("required minimum nearest-state RMS-SNR")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(columns) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    required = {"threshold", "data_dir", "result_dir", "analysis_dir", "result_prefix"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError("manifest is missing columns: %s" % sorted(missing))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    worst_rows = []
    state_rows = []

    for _, entry in manifest.sort_values("threshold").iterrows():
        threshold = float(entry["threshold"])
        data_dir = Path(str(entry["data_dir"]))
        result_dir = Path(str(entry["result_dir"]))
        analysis_dir = Path(str(entry["analysis_dir"]))
        prefix = str(entry["result_prefix"])

        meta_path = data_dir / "metadata.json"
        summary_path = result_dir / (prefix + "_summary.csv")
        samples_path = result_dir / (prefix + "_samples.csv")
        pair_path = data_dir / "split_pair_gseparation_summary.csv"
        selected_path = data_dir / "selected_states.csv"

        for path in (meta_path, summary_path, samples_path, pair_path, selected_path):
            if not path.exists():
                raise FileNotFoundError(path)

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        summary = pd.read_csv(summary_path)
        samples = pd.read_csv(samples_path)
        pair_df = pd.read_csv(pair_path)
        selected = pd.read_csv(selected_path)

        srow = summary[summary["noise_dir"] == args.noise_dir]
        if srow.empty:
            raise ValueError("%s has no %s row" % (summary_path, args.noise_dir))
        srow = srow.iloc[0]

        ss = samples[samples["noise_dir"] == args.noise_dir].copy()
        if ss.empty:
            raise ValueError("%s has no %s samples" % (samples_path, args.noise_dir))

        gamma_factor = ss["gamma_factor_error"].to_numpy(float)
        f_rel = ss["f_relative_l2"].to_numpy(float)

        counts = meta["selected_state_counts"]
        per_state = meta.get("per_state_sample_counts", {})

        rows.append({
            "requested_min_rms_snr": threshold,
            "actual_global_min_rms_snr": float(meta["global_min_rms_snr"]),
            "selected_state_count": int(meta["selected_state_count"]),
            "train_state_count": int(counts["train"]),
            "val_state_count": int(counts["val"]),
            "test_state_count": int(counts["test"]),
            "train_samples_per_state": int(per_state.get("train", 0)),
            "val_samples_per_state": int(per_state.get("val", 0)),
            "test_samples_per_state": int(per_state.get("test", 0)),
            "train_test_min_rms_snr": pair_min(pair_df, "train", "test"),
            "train_val_min_rms_snr": pair_min(pair_df, "train", "val"),
            "val_test_min_rms_snr": pair_min(pair_df, "val", "test"),
            "gamma_median_factor": float(srow["gamma_median_factor"]),
            "gamma_p90_factor": float(srow["gamma_p90_factor"]),
            "gamma_p99_factor": safe_quantile(gamma_factor, 0.99),
            "gamma_max_factor": safe_max(gamma_factor),
            "gamma_within_x1p2": float(srow["gamma_within_x1p2"]),
            "gamma_within_x1p5": float(srow["gamma_within_x1p5"]),
            "gamma_within_x2": float(srow["gamma_within_x2"]),
            "median_f_relative_l2": float(srow["median_f_relative_l2"]),
            "p90_f_relative_l2": float(srow["p90_f_relative_l2"]),
            "p99_f_relative_l2": safe_quantile(f_rel, 0.99),
            "max_f_relative_l2": safe_max(f_rel),
            "median_g_clean_relative_l2": float(srow["median_g_clean_relative_l2"]),
            "catastrophic_gamma_factor_ge5": float(srow["catastrophic_gamma_factor_ge5"]),
            "catastrophic_gamma_factor_ge10": float(srow["catastrophic_gamma_factor_ge10"]),
        })

        # Preserve the physical support for later interpretation.
        for _, st in selected.iterrows():
            state_rows.append({
                "requested_min_rms_snr": threshold,
                "split": st["split"],
                "a1": float(st["a1"]),
                "gamma": float(st["gamma"]),
                "nearest_state_rms_snr": float(st["snr_sep_rms"]),
            })

        ranked = ss.sort_values("gamma_factor_error", ascending=False).reset_index(drop=True)
        for rank_idx in range(min(10, len(ranked))):
            r = ranked.iloc[rank_idx]
            worst_rows.append({
                "requested_min_rms_snr": threshold,
                "rank": rank_idx + 1,
                "true_a1": float(r["true_a1"]),
                "pred_a1": float(r["pred_a1"]),
                "true_gamma": float(r["true_gamma"]),
                "pred_gamma": float(r["pred_gamma"]),
                "gamma_factor_error": float(r["gamma_factor_error"]),
                "f_relative_l2": float(r["f_relative_l2"]),
                "g_clean_relative_l2": float(r["g_clean_relative_l2"]),
            })

    scan = pd.DataFrame(rows).sort_values("requested_min_rms_snr").reset_index(drop=True)
    scan.to_csv(
        output_dir / "exp27_threshold_scan_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_csv(output_dir / "exp27_top10_worst_gamma_cases.csv", worst_rows)
    write_csv(output_dir / "exp27_selected_state_support.csv", state_rows)

    plot_lines(
        output_dir / "gamma_accuracy_vs_min_gseparation.png",
        scan,
        [
            ("gamma_median_factor", "median factor"),
            ("gamma_p90_factor", "p90 factor"),
            ("gamma_p99_factor", "p99 factor"),
        ],
        "gamma factor error",
        "Exp27 gamma recovery vs required g-separation",
    )
    plot_lines(
        output_dir / "gamma_within_x1p2_vs_min_gseparation.png",
        scan,
        [("gamma_within_x1p2", "within x1.2")],
        "fraction",
        "Exp27 continuous gamma accuracy vs required g-separation",
    )
    plot_lines(
        output_dir / "f_error_vs_min_gseparation.png",
        scan,
        [
            ("median_f_relative_l2", "median f relL2"),
            ("p90_f_relative_l2", "p90 f relL2"),
        ],
        "f relative L2",
        "Exp27 f reconstruction vs required g-separation",
    )
    plot_lines(
        output_dir / "catastrophic_error_vs_min_gseparation.png",
        scan,
        [
            ("catastrophic_gamma_factor_ge5", "gamma factor >=5"),
            ("catastrophic_gamma_factor_ge10", "gamma factor >=10"),
        ],
        "fraction",
        "Exp27 catastrophic gamma failures",
    )
    plot_lines(
        output_dir / "physical_state_count_vs_min_gseparation.png",
        scan,
        [
            ("selected_state_count", "all states"),
            ("train_state_count", "train"),
            ("val_state_count", "val"),
            ("test_state_count", "test"),
        ],
        "physical state count",
        "Exp27 price of stronger g-separation",
    )

    # One compact trade-off plot: reconstruction quality versus retained support.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(
            scan["selected_state_count"],
            scan["gamma_median_factor"],
            c=scan["requested_min_rms_snr"],
            s=90,
        )
        for _, r in scan.iterrows():
            ax.annotate(
                "SNR>=%.1f" % r["requested_min_rms_snr"],
                (r["selected_state_count"], r["gamma_median_factor"]),
                xytext=(5, 5),
                textcoords="offset points",
            )
        ax.set_xlabel("retained physical state count")
        ax.set_ylabel("median gamma factor error")
        ax.set_title("Exp27 identifiability trade-off: support size vs inversion accuracy")
        fig.colorbar(sc, ax=ax, label="required minimum RMS-SNR")
        fig.tight_layout()
        fig.savefig(output_dir / "support_vs_accuracy_tradeoff.png", dpi=170)
        plt.close(fig)
    except Exception as exc:
        print("warning: tradeoff plot skipped: %s" % exc)

    print("=" * 100)
    print("Exp27 threshold scan analysis ready")
    print("Interpretation reminder:")
    print("  Higher thresholds remove more near-degenerate physical states.")
    print("  Therefore improvement means the more-identifiable SUBPROBLEM is easier,")
    print("  while the shrinking selected-state count quantifies the price paid.")
    print("Read first:")
    print("  %s" % (output_dir / "exp27_threshold_scan_summary.csv"))
    print("  %s" % (output_dir / "exp27_top10_worst_gamma_cases.csv"))
    print("  %s" % (output_dir / "support_vs_accuracy_tradeoff.png"))
    print("=" * 100)


if __name__ == "__main__":
    main()
