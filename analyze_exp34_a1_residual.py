#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze Exp34: Exp33 TCN baseline vs frozen-TCN+a1-residual.

The key checks are:
1. Does a1 recovery improve?
2. Does joint recovery improve?
3. Are gamma predictions unchanged as intended?
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze Exp34 a1 residual experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--noise-dir", default="noise_0p2pct")
    p.add_argument("--a1-success-tol", type=float, default=0.005)
    p.add_argument("--gamma-success-factor", type=float, default=1.20)
    p.add_argument("--target-rate", type=float, default=0.98)
    p.add_argument("--fallback-rate", type=float, default=0.95)
    return p.parse_args()


def q(x, level):
    x = np.asarray(x, dtype=np.float64)
    return float(np.quantile(x, level)) if len(x) else math.nan


def load_samples(row, noise_dir):
    path = Path(str(row["result_dir"])) / (str(row["result_prefix"]) + "_samples.csv")
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "noise_dir" in df.columns:
        df = df[df["noise_dir"].astype(str) == str(noise_dir)].copy()
    if df.empty:
        raise ValueError("%s has no samples for %s" % (path, noise_dir))
    return df.reset_index(drop=True)


def metrics_from_samples(df, a1_tol, gamma_tol):
    ta = df["true_a1"].to_numpy(np.float64)
    pa = df["pred_a1"].to_numpy(np.float64)
    tg = np.clip(df["true_gamma"].to_numpy(np.float64), 1e-30, None)
    pg = np.clip(df["pred_gamma"].to_numpy(np.float64), 1e-30, None)

    a1_abs = np.abs(pa - ta)
    gamma_log = np.abs(np.log10(pg) - np.log10(tg))
    gamma_factor = np.exp(np.abs(np.log(pg) - np.log(tg)))
    a1_ok = a1_abs <= float(a1_tol)
    gamma_ok = gamma_factor <= float(gamma_tol)
    joint_ok = a1_ok & gamma_ok

    out = {
        "count": int(len(df)),
        "a1_mae": float(np.mean(a1_abs)),
        "a1_rmse": float(np.sqrt(np.mean((pa - ta) ** 2))),
        "a1_median_abs_error": float(np.median(a1_abs)),
        "a1_p90_abs_error": q(a1_abs, 0.90),
        "a1_p95_abs_error": q(a1_abs, 0.95),
        "a1_recovery_rate": float(np.mean(a1_ok)),
        "gamma_log10_mae": float(np.mean(gamma_log)),
        "gamma_p90_factor": q(gamma_factor, 0.90),
        "gamma_p95_factor": q(gamma_factor, 0.95),
        "gamma_recovery_rate": float(np.mean(gamma_ok)),
        "joint_recovery_rate": float(np.mean(joint_ok)),
    }
    for col in ("f_relative_l2", "g_clean_relative_l2"):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)
            if np.any(np.isfinite(v)):
                out["median_" + col] = float(np.nanmedian(v))
                out["p90_" + col] = float(np.nanquantile(v, 0.90))
                out["p95_" + col] = float(np.nanquantile(v, 0.95))
    return out


def summarize_runs(g, model):
    row = {"model": model, "run_count": int(len(g))}
    exclude = {"data_seed", "train_seed", "count"}
    for c in g.columns:
        if c in exclude or not pd.api.types.is_numeric_dtype(g[c]):
            continue
        vals = pd.to_numeric(g[c], errors="coerce").to_numpy(np.float64)
        if np.any(np.isfinite(vals)):
            row[c + "_mean"] = float(np.nanmean(vals))
            row[c + "_std"] = float(np.nanstd(vals))
            row[c + "_min"] = float(np.nanmin(vals))
            row[c + "_max"] = float(np.nanmax(vals))
    return row


def stage_gate(per_run, target, fallback):
    rows = []
    for model, g in per_run.groupby("model"):
        rates = g["joint_recovery_rate"].to_numpy(np.float64)
        rows.append(
            {
                "model": model,
                "run_count": int(len(rates)),
                "joint_recovery_mean": float(np.mean(rates)),
                "joint_recovery_std": float(np.std(rates)),
                "joint_recovery_min": float(np.min(rates)),
                "joint_recovery_max": float(np.max(rates)),
                "runs_ge_target": int(np.sum(rates >= target)),
                "target_rate": float(target),
                "mean_ge_target": bool(np.mean(rates) >= target),
                "all_runs_ge_target": bool(np.all(rates >= target)),
                "runs_ge_fallback": int(np.sum(rates >= fallback)),
                "fallback_rate": float(fallback),
                "mean_ge_fallback": bool(np.mean(rates) >= fallback),
                "all_runs_ge_fallback": bool(np.all(rates >= fallback)),
            }
        )
    return pd.DataFrame(rows)


def matched_comparison(per_run, sample_cache):
    rows = []
    tcn = per_run[per_run["model"] == "tcn"].set_index(["data_seed", "train_seed"])
    res = per_run[per_run["model"] == "a1res"].set_index(["data_seed", "train_seed"])
    shared = tcn.index.intersection(res.index)

    metrics = [
        "a1_mae",
        "a1_p90_abs_error",
        "a1_p95_abs_error",
        "a1_recovery_rate",
        "gamma_p90_factor",
        "gamma_recovery_rate",
        "joint_recovery_rate",
        "p90_f_relative_l2",
        "p90_g_clean_relative_l2",
    ]

    for idx in shared:
        b = tcn.loc[idx]
        r = res.loc[idx]
        if isinstance(b, pd.DataFrame):
            b = b.iloc[0]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]

        row = {"data_seed": int(idx[0]), "train_seed": int(idx[1])}
        for metric in metrics:
            if metric in per_run.columns:
                bv = float(b[metric])
                rv = float(r[metric])
                row["tcn_" + metric] = bv
                row["a1res_" + metric] = rv
                row["a1res_minus_tcn_" + metric] = rv - bv

        # Strong invariant: gamma branch should be identical sample-by-sample.
        bdf = sample_cache[("tcn", int(idx[0]), int(idx[1]))]
        rdf = sample_cache[("a1res", int(idx[0]), int(idx[1]))]
        n = min(len(bdf), len(rdf))
        gamma_delta = np.abs(
            bdf["pred_gamma"].to_numpy(np.float64)[:n]
            - rdf["pred_gamma"].to_numpy(np.float64)[:n]
        )
        row["max_abs_pred_gamma_change"] = float(np.max(gamma_delta))
        row["mean_abs_pred_gamma_change"] = float(np.mean(gamma_delta))
        rows.append(row)
    return pd.DataFrame(rows)


def make_plots(out, per_run):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    keys = sorted(set(zip(per_run["data_seed"], per_run["train_seed"])))
    x = np.arange(len(keys))
    labels = ["D%d/T%d" % k for k in keys]

    for metric, ylabel, filename, ylim in [
        ("a1_recovery_rate", "a1 recovery rate", "exp34_a1_recovery.png", (0.0, 1.01)),
        ("joint_recovery_rate", "joint recovery rate", "exp34_joint_recovery.png", (0.0, 1.01)),
        ("a1_p90_abs_error", "a1 P90 absolute error", "exp34_a1_p90.png", None),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for model in ("tcn", "a1res"):
            g = per_run[per_run["model"] == model].set_index(
                ["data_seed", "train_seed"]
            )
            vals = [
                float(g.loc[k, metric]) if k in g.index else np.nan
                for k in keys
            ]
            ax.plot(x, vals, marker="o", label=model.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title("Exp34 TCN vs A1-residual: " + ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / filename, dpi=170)
        plt.close(fig)


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    sample_cache = {}
    for _, r in manifest.sort_values(["model", "data_seed", "train_seed"]).iterrows():
        df = load_samples(r, args.noise_dir)
        model = str(r["model"]).lower()
        ds = int(r["data_seed"])
        ts = int(r["train_seed"])
        sample_cache[(model, ds, ts)] = df

        m = metrics_from_samples(
            df,
            a1_tol=args.a1_success_tol,
            gamma_tol=args.gamma_success_factor,
        )
        rows.append(
            {
                "model": model,
                "source": str(r.get("source", "")),
                "data_seed": ds,
                "train_seed": ts,
                "result_dir": str(r["result_dir"]),
                **m,
            }
        )

    per_run = pd.DataFrame(rows).sort_values(["model", "data_seed", "train_seed"])
    per_run.to_csv(out / "exp34_per_run_metrics.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [summarize_runs(g, model) for model, g in per_run.groupby("model")]
    )
    summary.to_csv(out / "exp34_model_summary.csv", index=False, encoding="utf-8-sig")

    matched = matched_comparison(per_run, sample_cache)
    matched.to_csv(
        out / "exp34_matched_tcn_a1res_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gate = stage_gate(per_run, args.target_rate, args.fallback_rate)
    gate.to_csv(out / "exp34_stage_gate.csv", index=False, encoding="utf-8-sig")

    gamma_check = matched[
        [
            "data_seed",
            "train_seed",
            "max_abs_pred_gamma_change",
            "mean_abs_pred_gamma_change",
        ]
    ].copy()
    gamma_check["gamma_unchanged_1e-6"] = (
        gamma_check["max_abs_pred_gamma_change"] <= 1e-6
    )
    gamma_check.to_csv(
        out / "exp34_gamma_invariance_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    make_plots(out, per_run)

    print("=" * 100)
    print("Exp34 analysis complete")
    for _, r in summary.iterrows():
        print(
            "%-6s | a1 rec %.2f%% | a1 P90 %.6g | gamma rec %.2f%% | "
            "joint %.2f%% (min %.2f%%)"
            % (
                str(r["model"]).upper(),
                100.0 * float(r["a1_recovery_rate_mean"]),
                float(r["a1_p90_abs_error_mean"]),
                100.0 * float(r["gamma_recovery_rate_mean"]),
                100.0 * float(r["joint_recovery_rate_mean"]),
                100.0 * float(r["joint_recovery_rate_min"]),
            )
        )

    max_gamma_change = float(matched["max_abs_pred_gamma_change"].max())
    print("")
    print("Max sample-wise gamma prediction change: %.3e" % max_gamma_change)
    if max_gamma_change > 1e-6:
        print("WARNING: gamma changed more than expected; inspect checkpoint/data matching.")
    else:
        print("Gamma invariance check PASSED.")

    print("")
    print("Read first:")
    print("  %s" % (out / "exp34_model_summary.csv"))
    print("  %s" % (out / "exp34_matched_tcn_a1res_deltas.csv"))
    print("  %s" % (out / "exp34_stage_gate.csv"))
    print("  %s" % (out / "exp34_gamma_invariance_check.csv"))
    print("=" * 100)


if __name__ == "__main__":
    main()
