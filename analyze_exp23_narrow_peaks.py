#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare Exp23 baseline vs narrow-weighted training, with narrow-peak diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze Exp23 narrow-gamma baseline and weighted models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="data_exp23_narrow_gamma_q500")
    p.add_argument("--baseline-dir", default="validation_results/exp23a_dense_gamma_baseline")
    p.add_argument("--baseline-prefix", default="exp23a")
    p.add_argument("--weighted-dir", default="validation_results/exp23b_narrow_weighted")
    p.add_argument("--weighted-prefix", default="exp23b")
    p.add_argument("--output-dir", default="validation_results/exp23_narrow_comparison")
    p.add_argument("--noise-dir", default="noise_0p2pct")
    p.add_argument("--narrow-threshold", type=float, default=0.06)
    p.add_argument("--integration-points", type=int, default=512)
    return p.parse_args()


def factor_error(pred, true):
    pred = np.clip(np.asarray(pred, dtype=float), 1e-30, None)
    true = np.clip(np.asarray(true, dtype=float), 1e-30, None)
    return np.maximum(pred / true, true / pred)


def band_name(gamma, threshold):
    g = float(gamma)
    if g <= 0.02 + 1e-12:
        return "ultra_narrow_<=0.02"
    if g <= threshold + 1e-12:
        return "narrow_0.02_to_%.3g" % threshold
    if g <= 0.2 + 1e-12:
        return "moderate_%.3g_to_0.2" % threshold
    return "broad_>0.2"


def summarize_group(label, split, band, group):
    ferr = factor_error(group["pred_gamma"], group["true_gamma"])
    return {
        "model": label,
        "split": split,
        "gamma_band": band,
        "count": int(len(group)),
        "gamma_median_factor": float(np.median(ferr)),
        "gamma_p90_factor": float(np.quantile(ferr, 0.90)),
        "gamma_within_x1p2": float(np.mean(ferr <= 1.2)),
        "median_a1_abs_error": float(np.median(np.abs(group["pred_a1"] - group["true_a1"]))),
        "median_f_relative_l2": float(np.nanmedian(group["f_relative_l2"])),
        "median_g_clean_relative_l2": float(np.nanmedian(group["g_clean_relative_l2"])),
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError("no rows for %s" % path)
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_line_plot(path, xs, series, xlabel, ylabel, title, logx=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("warning: plot skipped: %s" % exc)
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, ys in series:
        ax.plot(xs, ys, marker="o", label=label)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_narrow_curve_bundle(
    *,
    label,
    result_dir,
    prefix,
    samples,
    output_dir,
    narrow_threshold,
    integration_points,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mc_pool_config import DEFAULT_PHYSICS
        from mc_physics import scaled_curves_numpy
    except Exception as exc:
        print("warning: narrow curve plots skipped: %s" % exc)
        return []

    selected_records = []
    for split in ("test_interp_gamma", "test_interp_both"):
        sub = samples[
            (samples["split"] == split)
            & (samples["true_gamma"] <= float(narrow_threshold) + 1e-12)
        ].copy()
        if sub.empty:
            continue
        sub["factor"] = factor_error(sub["pred_gamma"], sub["true_gamma"])
        sub = sub.sort_values("factor", kind="mergesort").reset_index(drop=True)
        picks = [
            ("best", 0),
            ("median", len(sub) // 2),
            ("worst", len(sub) - 1),
        ]
        for rank, idx in picks:
            row = sub.iloc[int(idx)]
            p_true = np.array([[
                row["true_a1"], 0.025, 0.0, 0.8, row["true_gamma"]
            ]], dtype=np.float64)
            p_pred = np.array([[
                row["pred_a1"], 0.025, 0.0, 0.8, row["pred_gamma"]
            ]], dtype=np.float64)
            physics = replace(DEFAULT_PHYSICS, q2_points=1000)
            f_true, g_true = scaled_curves_numpy(
                p_true, integration_points=int(integration_points), config=physics
            )
            f_pred, g_pred = scaled_curves_numpy(
                p_pred, integration_points=int(integration_points), config=physics
            )
            f_true = f_true[0]; f_pred = f_pred[0]
            g_true = g_true[0]; g_pred = g_pred[0]
            x = np.linspace(physics.s_min, physics.s_max, physics.output_points)
            q = np.linspace(physics.q2_min, physics.q2_max, physics.q2_points)

            d = output_dir / "narrow_curve_fits" / label / split
            d.mkdir(parents=True, exist_ok=True)
            stem = "%s_gamma%.5g" % (rank, float(row["true_gamma"]))

            fig, ax = plt.subplots(figsize=(7.4, 4.5))
            ax.plot(x, f_true, label="true f")
            ax.plot(x, f_pred, "--", label="predicted f")
            ax.set_xlabel("s")
            ax.set_ylabel("f(s)")
            ax.set_title(
                "%s | %s | %s\ntrue a1=%.5g pred=%.5g; true gamma=%.5g pred=%.5g\n"
                "gamma factor=%.4g, f relL2=%.4g"
                % (
                    label, split, rank,
                    row["true_a1"], row["pred_a1"],
                    row["true_gamma"], row["pred_gamma"],
                    row["factor"], row["f_relative_l2"],
                )
            )
            ax.legend()
            fig.tight_layout()
            fpath = d / ("f_" + stem + ".png")
            fig.savefig(fpath, dpi=170)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7.4, 4.5))
            ax.plot(q, g_true, label="true g")
            ax.plot(q, g_pred, "--", label="predicted-parameter g")
            ax.set_xlabel(r"$q^2$")
            ax.set_ylabel("g")
            ax.set_title("%s | %s | %s | narrow sample" % (label, split, rank))
            ax.legend()
            fig.tight_layout()
            gpath = d / ("g_" + stem + ".png")
            fig.savefig(gpath, dpi=170)
            plt.close(fig)

            selected_records.append({
                "model": label,
                "split": split,
                "rank": rank,
                "true_a1": float(row["true_a1"]),
                "pred_a1": float(row["pred_a1"]),
                "true_gamma": float(row["true_gamma"]),
                "pred_gamma": float(row["pred_gamma"]),
                "gamma_factor_error": float(row["factor"]),
                "f_relative_l2": float(row["f_relative_l2"]),
                "g_clean_relative_l2": float(row["g_clean_relative_l2"]),
                "f_plot": str(fpath),
                "g_plot": str(gpath),
            })
    return selected_records


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    gamma_train = np.asarray(metadata["gamma_train_values"], dtype=float)
    gamma_interp = np.asarray(metadata["gamma_interp_values"], dtype=float)

    configs = [
        ("baseline", Path(args.baseline_dir), args.baseline_prefix),
        ("narrow_weighted", Path(args.weighted_dir), args.weighted_prefix),
    ]

    rows = []
    per_gamma_rows = []
    curve_rows = []

    for label, result_dir, prefix in configs:
        sample_path = result_dir / (prefix + "_samples.csv")
        if not sample_path.exists():
            raise FileNotFoundError(sample_path)
        samples = pd.read_csv(sample_path)
        samples = samples[samples["noise_dir"] == args.noise_dir].copy()
        if samples.empty:
            raise ValueError("no %s rows in %s" % (args.noise_dir, sample_path))

        samples["gamma_band"] = [
            band_name(x, args.narrow_threshold) for x in samples["true_gamma"]
        ]

        for split, split_df in samples.groupby("split", sort=True):
            rows.append(summarize_group(label, split, "ALL", split_df))
            for band, group in split_df.groupby("gamma_band", sort=True):
                rows.append(summarize_group(label, split, band, group))
            for gamma, group in split_df.groupby("true_gamma", sort=True):
                r = summarize_group(label, split, "gamma=%.8g" % gamma, group)
                r["true_gamma"] = float(gamma)
                per_gamma_rows.append(r)

        curve_rows.extend(
            save_narrow_curve_bundle(
                label=label,
                result_dir=result_dir,
                prefix=prefix,
                samples=samples,
                output_dir=out,
                narrow_threshold=args.narrow_threshold,
                integration_points=args.integration_points,
            )
        )

    write_csv(out / "exp23_narrow_band_summary.csv", rows)
    if per_gamma_rows:
        write_csv(out / "exp23_per_gamma_summary.csv", per_gamma_rows)
    if curve_rows:
        write_csv(out / "exp23_narrow_curve_samples.csv", curve_rows)

    # Plot interpolation performance vs true gamma.
    for split in ("test_interp_gamma", "test_interp_both"):
        gamma_union = sorted({
            float(r.get("true_gamma")) for r in per_gamma_rows
            if r["split"] == split and "true_gamma" in r
        })
        if not gamma_union:
            continue
        for metric, ylabel, filename in (
            ("gamma_median_factor", "median gamma factor error", "gamma_factor_vs_true_gamma_%s.png" % split),
            ("gamma_within_x1p2", "fraction within x1.2", "gamma_within_x1p2_vs_true_gamma_%s.png" % split),
            ("median_f_relative_l2", "median f relative L2", "f_l2_vs_true_gamma_%s.png" % split),
        ):
            series = []
            for label, _, _ in configs:
                vals = []
                for g in gamma_union:
                    match = [
                        r for r in per_gamma_rows
                        if r["model"] == label and r["split"] == split
                        and "true_gamma" in r and np.isclose(r["true_gamma"], g, atol=1e-10)
                    ]
                    vals.append(float(match[0][metric]) if match else math.nan)
                series.append((label, vals))
            save_line_plot(
                out / filename,
                gamma_union,
                series,
                "true gamma",
                ylabel,
                "Exp23 %s | %s" % (metric, split),
                logx=True,
            )

    print("=" * 100)
    print("Exp23 narrow-peak comparison")
    print("gamma train anchors : %s" % [float(x) for x in gamma_train])
    print("gamma interp values : %s" % [float(x) for x in gamma_interp])
    print("narrow threshold    : %.4g" % args.narrow_threshold)
    print("summary             : %s" % (out / "exp23_narrow_band_summary.csv"))
    if curve_rows:
        print("narrow curves       : %s" % (out / "narrow_curve_fits"))
    print("=" * 100)


if __name__ == "__main__":
    main()
