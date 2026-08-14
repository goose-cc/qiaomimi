#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare Exp24A direct regression with Exp24B coarse-to-fine regression."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze Exp24 direct vs coarse-to-fine gamma recovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--baseline-dir", default="validation_results/exp24a_direct_dense")
    p.add_argument("--baseline-prefix", default="exp24a")
    p.add_argument("--coarse-dir", default="validation_results/exp24b_coarse_to_fine")
    p.add_argument("--coarse-prefix", default="exp24b")
    p.add_argument("--output-dir", default="validation_results/exp24_comparison")
    p.add_argument("--noise-dir", default="noise_0p2pct")
    p.add_argument(
        "--gamma-edges",
        type=parse_float_list,
        default=[0.01, 0.0316227766, 0.1, 0.316227766, 1.0],
    )
    return p.parse_args()


def factor(pred, true):
    pred = np.clip(np.asarray(pred, dtype=float), 1e-30, None)
    true = np.clip(np.asarray(true, dtype=float), 1e-30, None)
    return np.maximum(pred / true, true / pred)


def gamma_class(values, edges):
    x = np.asarray(values, dtype=float)
    return np.sum(x[:, None] >= np.asarray(edges[1:-1], dtype=float)[None, :], axis=1)


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k); fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def summarize(model_name, split, df, edges, has_oracle):
    ferr = factor(df["pred_gamma"], df["true_gamma"])
    tc = gamma_class(df["true_gamma"], edges)
    pc = gamma_class(df["pred_gamma"], edges)
    jump = np.abs(tc - pc)
    row = {
        "model": model_name,
        "split": split,
        "count": int(len(df)),
        "gamma_median_factor": float(np.median(ferr)),
        "gamma_p90_factor": float(np.quantile(ferr, 0.9)),
        "gamma_within_x1p2": float(np.mean(ferr <= 1.2)),
        "coarse_class_accuracy": float(np.mean(tc == pc)),
        "coarse_jump_ge2": float(np.mean(jump >= 2)),
        "median_f_relative_l2": float(np.nanmedian(df["f_relative_l2"])),
        "median_g_clean_relative_l2": float(np.nanmedian(df["g_clean_relative_l2"])),
    }
    if has_oracle and "oracle_gamma" in df:
        of = factor(df["oracle_gamma"], df["true_gamma"])
        row["oracle_bin_gamma_median_factor"] = float(np.median(of))
        row["oracle_bin_gamma_within_x1p2"] = float(np.mean(of <= 1.2))
    else:
        row["oracle_bin_gamma_median_factor"] = math.nan
        row["oracle_bin_gamma_within_x1p2"] = math.nan
    return row


def plot_metric(path, per_gamma, metric, ylabel):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in ("direct_regression", "coarse_to_fine"):
        sub = per_gamma[per_gamma["model"] == model].sort_values("true_gamma")
        ax.plot(sub["true_gamma"], sub[metric], marker="o", label=model)
    ax.set_xscale("log")
    ax.set_xlabel("true gamma")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=165)
    plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(Path(args.baseline_dir) / (args.baseline_prefix + "_samples.csv"))
    coarse = pd.read_csv(Path(args.coarse_dir) / (args.coarse_prefix + "_samples.csv"))
    base = base[base["noise_dir"] == args.noise_dir].copy()
    coarse = coarse[coarse["noise_dir"] == args.noise_dir].copy()

    # Standardize baseline a1 column names.
    if "true_a1" not in base.columns and "true_second" in base.columns:
        base["true_a1"] = base["true_second"]
        base["pred_a1"] = base["pred_second"]

    rows = []
    per_gamma_rows = []
    for name, df, has_oracle in (
        ("direct_regression", base, False),
        ("coarse_to_fine", coarse, True),
    ):
        for split, sdf in df.groupby("split", sort=True):
            rows.append(summarize(name, split, sdf, args.gamma_edges, has_oracle))
            for gamma, gdf in sdf.groupby("true_gamma", sort=True):
                r = summarize(name, split, gdf, args.gamma_edges, has_oracle)
                r["true_gamma"] = float(gamma)
                per_gamma_rows.append(r)

    write_csv(out / "exp24_comparison_summary.csv", rows)
    write_csv(out / "exp24_per_gamma_summary.csv", per_gamma_rows)

    # Explicitly list catastrophic branch jumps.
    catastrophic = []
    for name, df in (("direct_regression", base), ("coarse_to_fine", coarse)):
        tc = gamma_class(df["true_gamma"], args.gamma_edges)
        pc = gamma_class(df["pred_gamma"], args.gamma_edges)
        jump = np.abs(tc - pc)
        ferr = factor(df["pred_gamma"], df["true_gamma"])
        idx = np.where((jump >= 2) | (ferr >= 5.0))[0]
        for i in idx:
            row = df.iloc[int(i)]
            catastrophic.append({
                "model": name,
                "split": row["split"],
                "true_a1": float(row["true_a1"]),
                "pred_a1": float(row["pred_a1"]),
                "true_gamma": float(row["true_gamma"]),
                "pred_gamma": float(row["pred_gamma"]),
                "true_class": int(tc[i]),
                "pred_class": int(pc[i]),
                "class_jump": int(jump[i]),
                "gamma_factor_error": float(ferr[i]),
                "f_relative_l2": float(row["f_relative_l2"]),
                "g_clean_relative_l2": float(row["g_clean_relative_l2"]),
            })
    write_csv(out / "exp24_catastrophic_branch_jumps.csv", catastrophic)

    pg = pd.DataFrame(per_gamma_rows)
    for split in ("test_interp_gamma", "test_interp_both"):
        s = pg[pg["split"] == split]
        if s.empty:
            continue
        plot_metric(
            out / ("gamma_factor_vs_true_gamma_%s.png" % split),
            s, "gamma_median_factor", "median gamma factor error",
        )
        plot_metric(
            out / ("gamma_within_x1p2_vs_true_gamma_%s.png" % split),
            s, "gamma_within_x1p2", "fraction within x1.2",
        )
        plot_metric(
            out / ("f_l2_vs_true_gamma_%s.png" % split),
            s, "median_f_relative_l2", "median f relative L2",
        )
        if np.any(np.isfinite(s["oracle_bin_gamma_median_factor"])):
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                c = s[s["model"] == "coarse_to_fine"].sort_values("true_gamma")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(c["true_gamma"], c["gamma_median_factor"], marker="o", label="predicted bin")
                ax.plot(c["true_gamma"], c["oracle_bin_gamma_median_factor"], marker="o", label="oracle true bin")
                ax.set_xscale("log")
                ax.set_xlabel("true gamma")
                ax.set_ylabel("median gamma factor error")
                ax.set_title("Exp24 coarse-to-fine: classification vs within-bin bottleneck")
                ax.legend()
                fig.tight_layout()
                fig.savefig(out / ("predicted_vs_oracle_bin_%s.png" % split), dpi=165)
                plt.close(fig)
            except Exception:
                pass

    print("=" * 100)
    print("Exp24 comparison ready")
    print("  %s" % (out / "exp24_comparison_summary.csv"))
    print("  %s" % (out / "exp24_per_gamma_summary.csv"))
    print("  %s" % (out / "exp24_catastrophic_branch_jumps.csv"))
    print("=" * 100)


if __name__ == "__main__":
    main()
