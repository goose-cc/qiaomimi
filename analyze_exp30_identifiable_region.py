#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate Exp30 alias-exclusion sweep and report sample/state-level accuracy."""
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
        description="Analyze Exp30 identifiable-region sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--noise-dir", default="noise_0p2pct")
    p.add_argument("--target-within-x1p2", type=float, default=0.95)
    return p.parse_args()


def factor(pred, true):
    pred = np.clip(np.asarray(pred, dtype=float), 1e-30, None)
    true = np.clip(np.asarray(true, dtype=float), 1e-30, None)
    return np.maximum(pred / true, true / pred)


def state_level_metrics(samples):
    rows = []
    # true a1/gamma repeat exactly within a physical state.
    for (a1, gamma), g in samples.groupby(["true_a1", "true_gamma"], sort=False):
        pred_gamma = float(np.median(g["pred_gamma"].to_numpy(float)))
        pred_a1 = float(np.median(g["pred_a1"].to_numpy(float)))
        gf = float(max(pred_gamma / float(gamma), float(gamma) / pred_gamma))
        rows.append({
            "true_a1": float(a1),
            "true_gamma": float(gamma),
            "median_pred_a1": pred_a1,
            "median_pred_gamma": pred_gamma,
            "gamma_factor_error": gf,
            "median_f_relative_l2": float(np.nanmedian(g["f_relative_l2"])),
            "median_g_clean_relative_l2": float(np.nanmedian(g["g_clean_relative_l2"])),
            "noise_realization_count": int(len(g)),
        })
    df = pd.DataFrame(rows)
    return {
        "physical_state_count": int(len(df)),
        "state_gamma_median_factor": float(np.median(df["gamma_factor_error"])),
        "state_gamma_p90_factor": float(np.quantile(df["gamma_factor_error"], 0.9)),
        "state_gamma_max_factor": float(np.max(df["gamma_factor_error"])),
        "state_gamma_within_x1p2": float(np.mean(df["gamma_factor_error"] <= 1.2)),
        "state_gamma_within_x1p5": float(np.mean(df["gamma_factor_error"] <= 1.5)),
        "state_gamma_within_x2": float(np.mean(df["gamma_factor_error"] <= 2.0)),
        "state_median_f_relative_l2": float(np.median(df["median_f_relative_l2"])),
    }, df


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
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    worst_rows = []
    all_state_rows = []

    for _, e in manifest.sort_values("alias_min_rms_snr").iterrows():
        alias_min = float(e["alias_min_rms_snr"])
        data_dir = Path(str(e["data_dir"]))
        result_dir = Path(str(e["result_dir"]))
        prefix = str(e["result_prefix"])

        meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
        summary = pd.read_csv(result_dir / (prefix + "_summary.csv"))
        samples = pd.read_csv(result_dir / (prefix + "_samples.csv"))

        sr = summary[summary["noise_dir"] == args.noise_dir]
        if sr.empty:
            raise ValueError("missing %s summary row" % args.noise_dir)
        sr = sr.iloc[0]
        ss = samples[samples["noise_dir"] == args.noise_dir].copy()
        if ss.empty:
            raise ValueError("missing %s samples" % args.noise_dir)

        gf = ss["gamma_factor_error"].to_numpy(float)
        f = ss["f_relative_l2"].to_numpy(float)
        state_metrics, state_df = state_level_metrics(ss)

        cov = meta["coverage_summary"]
        hold = [r for r in cov if r["split"] == "holdout_all"][0]

        rows.append({
            "alias_min_rms_snr": alias_min,
            "requested_min_rms_snr": float(meta["requested_min_rms_snr"]),
            "actual_global_min_rms_snr": float(meta["global_min_rms_snr"]),
            "max_holdout_to_train_rms_snr": float(hold["max_nearest_train_rms_snr"]),
            "alias_safe_candidate_count": int(meta["alias_safe_candidate_count"]),
            "selected_state_count": int(meta["selected_state_count"]),
            "train_state_count": int(meta["selected_state_counts"]["train"]),
            "val_state_count": int(meta["selected_state_counts"]["val"]),
            "test_state_count": int(meta["selected_state_counts"]["test"]),
            "selected_alias_score_min": float(meta["selected_alias_score_min"]),
            "selected_alias_score_median": float(meta["selected_alias_score_median"]),
            "gamma_median_factor": float(sr["gamma_median_factor"]),
            "gamma_p90_factor": float(sr["gamma_p90_factor"]),
            "gamma_p99_factor": float(np.quantile(gf, 0.99)),
            "gamma_max_factor": float(np.max(gf)),
            "gamma_within_x1p2": float(sr["gamma_within_x1p2"]),
            "gamma_within_x1p5": float(sr["gamma_within_x1p5"]),
            "gamma_within_x2": float(sr["gamma_within_x2"]),
            "median_f_relative_l2": float(sr["median_f_relative_l2"]),
            "p90_f_relative_l2": float(sr["p90_f_relative_l2"]),
            "p99_f_relative_l2": float(np.quantile(f, 0.99)),
            "catastrophic_gamma_factor_ge5": float(sr["catastrophic_gamma_factor_ge5"]),
            "catastrophic_gamma_factor_ge10": float(sr["catastrophic_gamma_factor_ge10"]),
            **state_metrics,
        })

        state_df["alias_min_rms_snr"] = alias_min
        all_state_rows.append(state_df)

        w = ss.sort_values("gamma_factor_error", ascending=False).head(15)
        for rank, (_, r) in enumerate(w.iterrows(), start=1):
            worst_rows.append({
                "alias_min_rms_snr": alias_min,
                "rank": rank,
                "true_a1": float(r["true_a1"]),
                "pred_a1": float(r["pred_a1"]),
                "true_gamma": float(r["true_gamma"]),
                "pred_gamma": float(r["pred_gamma"]),
                "gamma_factor_error": float(r["gamma_factor_error"]),
                "f_relative_l2": float(r["f_relative_l2"]),
                "g_clean_relative_l2": float(r["g_clean_relative_l2"]),
            })

    scan = pd.DataFrame(rows).sort_values("alias_min_rms_snr")
    scan.to_csv(out / "exp30_identifiable_region_summary.csv", index=False, encoding="utf-8-sig")
    write_csv(out / "exp30_top15_worst_cases.csv", worst_rows)
    if all_state_rows:
        pd.concat(all_state_rows, ignore_index=True).to_csv(
            out / "exp30_physical_state_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # Engineering gate: both noisy-sample and physical-state accuracy matter.
    gates = []
    for _, r in scan.iterrows():
        gates.append({
            "alias_min_rms_snr": float(r["alias_min_rms_snr"]),
            "sample_within_x1p2_target": float(args.target_within_x1p2),
            "sample_within_x1p2_actual": float(r["gamma_within_x1p2"]),
            "sample_pass": int(r["gamma_within_x1p2"] >= args.target_within_x1p2),
            "state_within_x1p2_target": float(args.target_within_x1p2),
            "state_within_x1p2_actual": float(r["state_gamma_within_x1p2"]),
            "state_pass": int(r["state_gamma_within_x1p2"] >= args.target_within_x1p2),
            "catastrophic_factor_ge5": float(r["catastrophic_gamma_factor_ge5"]),
            "selected_state_count": int(r["selected_state_count"]),
        })
    write_csv(out / "exp30_stage_gate.csv", gates)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(scan["alias_min_rms_snr"], scan["gamma_within_x1p2"],
                marker="o", label="noisy samples")
        ax.plot(scan["alias_min_rms_snr"], scan["state_gamma_within_x1p2"],
                marker="o", label="physical states")
        ax.axhline(args.target_within_x1p2, linestyle="--", label="95% stage target")
        ax.set_xlabel("minimum continuous remote-alias RMS-SNR")
        ax.set_ylabel("fraction within gamma x1.2")
        ax.set_title("Exp30: does continuous alias exclusion improve inversion?")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "gamma_accuracy_vs_alias_exclusion.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(scan["alias_min_rms_snr"], scan["gamma_p90_factor"],
                marker="o", label="gamma p90 factor")
        ax.plot(scan["alias_min_rms_snr"], scan["gamma_p99_factor"],
                marker="o", label="gamma p99 factor")
        ax.set_xlabel("minimum continuous remote-alias RMS-SNR")
        ax.set_ylabel("gamma factor error")
        ax.set_title("Exp30 tail errors after remote-alias exclusion")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "gamma_tail_vs_alias_exclusion.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(scan["alias_min_rms_snr"], scan["median_f_relative_l2"],
                marker="o", label="median f relL2")
        ax.plot(scan["alias_min_rms_snr"], scan["p90_f_relative_l2"],
                marker="o", label="p90 f relL2")
        ax.set_xlabel("minimum continuous remote-alias RMS-SNR")
        ax.set_ylabel("f relative L2")
        ax.set_title("Exp30 f reconstruction vs alias exclusion")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "f_error_vs_alias_exclusion.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(scan["alias_min_rms_snr"], scan["selected_state_count"],
                marker="o", label="selected physical states")
        ax.plot(scan["alias_min_rms_snr"], scan["alias_safe_candidate_count"],
                marker="o", label="alias-safe candidates")
        ax.set_xlabel("minimum continuous remote-alias RMS-SNR")
        ax.set_ylabel("count")
        ax.set_title("Exp30 price of demanding a more identifiable region")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "support_size_vs_alias_exclusion.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(
            scan["selected_state_count"],
            scan["state_gamma_within_x1p2"],
            c=scan["alias_min_rms_snr"],
            s=100,
        )
        for _, r in scan.iterrows():
            ax.annotate(
                "alias>=%.2f" % r["alias_min_rms_snr"],
                (r["selected_state_count"], r["state_gamma_within_x1p2"]),
                xytext=(5, 5), textcoords="offset points",
            )
        ax.set_xlabel("retained physical state count")
        ax.set_ylabel("physical-state gamma within x1.2")
        ax.set_title("Exp30 identifiability trade-off")
        fig.colorbar(sc, ax=ax, label="alias exclusion RMS-SNR")
        fig.tight_layout()
        fig.savefig(out / "support_vs_state_accuracy.png", dpi=170)
        plt.close(fig)
    except Exception as exc:
        print("warning: plotting skipped: %s" % exc)

    print("=" * 100)
    print("Exp30 aggregate analysis ready")
    print("Read first:")
    print("  %s" % (out / "exp30_identifiable_region_summary.csv"))
    print("  %s" % (out / "exp30_stage_gate.csv"))
    print("  %s" % (out / "exp30_top15_worst_cases.csv"))
    print("=" * 100)


if __name__ == "__main__":
    main()
