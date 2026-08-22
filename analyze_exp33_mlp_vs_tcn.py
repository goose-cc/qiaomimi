#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate Exp33 MLP-vs-TCN results under one common metric implementation.

The analyzer intentionally recomputes metrics from sample CSV files.  This lets
it compare:
- newly trained Exp33 MLP/TCN runs, and
- reused Exp32 MLP runs,
with exactly the same recovery thresholds and continuous-error definitions.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze Exp33 MLP vs TCN",
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


def load_samples(row, noise_dir: str) -> pd.DataFrame:
    result_dir = Path(str(row["result_dir"]))
    prefix = str(row["result_prefix"])
    path = result_dir / (prefix + "_samples.csv")
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "noise_dir" in df.columns:
        df = df[df["noise_dir"].astype(str) == str(noise_dir)].copy()
    if df.empty:
        raise ValueError("%s has no samples for noise_dir=%s" % (path, noise_dir))
    required = {"true_a1", "pred_a1", "true_gamma", "pred_gamma"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("%s missing columns %s" % (path, sorted(missing)))
    return df.reset_index(drop=True)


def metrics_from_samples(
    df: pd.DataFrame,
    *,
    a1_tol: float,
    gamma_factor_tol: float,
) -> dict[str, float]:
    ta = df["true_a1"].to_numpy(np.float64)
    pa = df["pred_a1"].to_numpy(np.float64)
    tg = np.clip(df["true_gamma"].to_numpy(np.float64), 1e-30, None)
    pg = np.clip(df["pred_gamma"].to_numpy(np.float64), 1e-30, None)

    a1_signed = pa - ta
    a1_abs = np.abs(a1_signed)
    gamma_log_signed = np.log10(pg) - np.log10(tg)
    gamma_log_abs = np.abs(gamma_log_signed)
    gamma_rel = np.abs(pg - tg) / tg
    gamma_factor = np.exp(np.abs(np.log(pg) - np.log(tg)))

    a1_ok = a1_abs <= float(a1_tol)
    gamma_ok = gamma_factor <= float(gamma_factor_tol)
    joint_ok = a1_ok & gamma_ok

    result = {
        "count": int(len(df)),
        "a1_mae": float(np.mean(a1_abs)),
        "a1_rmse": float(np.sqrt(np.mean(a1_signed**2))),
        "a1_median_abs_error": float(np.median(a1_abs)),
        "a1_p90_abs_error": q(a1_abs, 0.90),
        "a1_p95_abs_error": q(a1_abs, 0.95),
        "a1_max_abs_error": float(np.max(a1_abs)),
        "gamma_log10_mae": float(np.mean(gamma_log_abs)),
        "gamma_log10_rmse": float(np.sqrt(np.mean(gamma_log_signed**2))),
        "gamma_median_abs_log10_error": float(np.median(gamma_log_abs)),
        "gamma_p90_abs_log10_error": q(gamma_log_abs, 0.90),
        "gamma_p95_abs_log10_error": q(gamma_log_abs, 0.95),
        "gamma_relative_mae": float(np.mean(gamma_rel)),
        "gamma_median_relative_error": float(np.median(gamma_rel)),
        "gamma_p90_relative_error": q(gamma_rel, 0.90),
        "gamma_p95_relative_error": q(gamma_rel, 0.95),
        "gamma_median_factor": float(np.median(gamma_factor)),
        "gamma_p90_factor": q(gamma_factor, 0.90),
        "gamma_p95_factor": q(gamma_factor, 0.95),
        "gamma_max_factor": float(np.max(gamma_factor)),
        "a1_success_tol": float(a1_tol),
        "gamma_success_factor": float(gamma_factor_tol),
        "a1_recovery_rate": float(np.mean(a1_ok)),
        "gamma_recovery_rate": float(np.mean(gamma_ok)),
        "joint_recovery_rate": float(np.mean(joint_ok)),
    }

    for col, out_name in [
        ("f_relative_l2", "f_relative_l2"),
        ("g_clean_relative_l2", "g_clean_relative_l2"),
    ]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)
            if np.any(np.isfinite(v)):
                result["median_" + out_name] = float(np.nanmedian(v))
                result["p90_" + out_name] = float(np.nanquantile(v, 0.90))
                result["p95_" + out_name] = float(np.nanquantile(v, 0.95))
    return result


def numeric_metric_columns(df: pd.DataFrame):
    exclude = {
        "data_seed", "train_seed", "count",
        "a1_success_tol", "gamma_success_factor",
    }
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def summarize_runs(df: pd.DataFrame, label_cols: dict) -> dict:
    row = dict(label_cols)
    row["run_count"] = int(len(df))
    for c in numeric_metric_columns(df):
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)
        if not np.any(np.isfinite(vals)):
            continue
        row[c + "_mean"] = float(np.nanmean(vals))
        row[c + "_std"] = float(np.nanstd(vals, ddof=0))
        row[c + "_min"] = float(np.nanmin(vals))
        row[c + "_max"] = float(np.nanmax(vals))
    return row


def matched_deltas(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key_cols = ["data_seed", "train_seed"]
    mlp = per_run[per_run["model"] == "mlp"].set_index(key_cols)
    tcn = per_run[per_run["model"] == "tcn"].set_index(key_cols)
    shared = mlp.index.intersection(tcn.index)
    metrics = [
        "a1_mae",
        "a1_p90_abs_error",
        "gamma_log10_mae",
        "gamma_p90_factor",
        "gamma_p95_factor",
        "joint_recovery_rate",
        "p90_f_relative_l2",
    ]
    for idx in shared:
        m = mlp.loc[idx]
        t = tcn.loc[idx]
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        if isinstance(t, pd.DataFrame):
            t = t.iloc[0]
        row = {"data_seed": int(idx[0]), "train_seed": int(idx[1])}
        for metric in metrics:
            if metric not in per_run.columns:
                continue
            mv = float(m[metric])
            tv = float(t[metric])
            row["mlp_" + metric] = mv
            row["tcn_" + metric] = tv
            row["tcn_minus_mlp_" + metric] = tv - mv
        rows.append(row)
    return pd.DataFrame(rows)


def build_stage_gate(per_run: pd.DataFrame, target: float, fallback: float):
    rows = []
    for model, g in per_run.groupby("model", sort=True):
        rates = g["joint_recovery_rate"].to_numpy(np.float64)
        rows.append(
            {
                "model": model,
                "run_count": int(len(rates)),
                "joint_recovery_mean": float(np.mean(rates)),
                "joint_recovery_std": float(np.std(rates, ddof=0)),
                "joint_recovery_min": float(np.min(rates)),
                "joint_recovery_max": float(np.max(rates)),
                "runs_ge_target": int(np.sum(rates >= target)),
                "target_rate": float(target),
                "all_runs_ge_target": bool(np.all(rates >= target)),
                "mean_ge_target": bool(np.mean(rates) >= target),
                "runs_ge_fallback": int(np.sum(rates >= fallback)),
                "fallback_rate": float(fallback),
                "all_runs_ge_fallback": bool(np.all(rates >= fallback)),
                "mean_ge_fallback": bool(np.mean(rates) >= fallback),
            }
        )
    return pd.DataFrame(rows)


def make_plots(out: Path, per_run: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    order = per_run.sort_values(["data_seed", "train_seed", "model"]).copy()
    keys = sorted(set(zip(order["data_seed"], order["train_seed"])))
    x = np.arange(len(keys))
    labels = ["D%d/T%d" % k for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    for model in ("mlp", "tcn"):
        g = order[order["model"] == model].set_index(["data_seed", "train_seed"])
        vals = [float(g.loc[k, "joint_recovery_rate"]) if k in g.index else np.nan for k in keys]
        ax.plot(x, vals, marker="o", label=model.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Joint recovery rate")
    ax.set_ylim(0.0, 1.01)
    ax.set_title("Exp33 MLP vs TCN: joint recovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "exp33_joint_recovery_comparison.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for model in ("mlp", "tcn"):
        g = order[order["model"] == model].set_index(["data_seed", "train_seed"])
        vals = [float(g.loc[k, "gamma_p90_factor"]) if k in g.index else np.nan for k in keys]
        ax.plot(x, vals, marker="o", label=model.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Gamma P90 factor error")
    ax.set_title("Exp33 MLP vs TCN: gamma continuous error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "exp33_gamma_p90_comparison.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for model in ("mlp", "tcn"):
        g = order[order["model"] == model].set_index(["data_seed", "train_seed"])
        vals = [float(g.loc[k, "a1_p90_abs_error"]) if k in g.index else np.nan for k in keys]
        ax.plot(x, vals, marker="o", label=model.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("a1 P90 absolute error")
    ax.set_title("Exp33 MLP vs TCN: a1 continuous error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "exp33_a1_p90_comparison.png", dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    if args.a1_success_tol <= 0:
        raise ValueError("--a1-success-tol must be > 0")
    if args.gamma_success_factor <= 1:
        raise ValueError("--gamma-success-factor must be > 1")
    if not (0 < args.fallback_rate <= args.target_rate <= 1):
        raise ValueError("require 0 < fallback-rate <= target-rate <= 1")

    manifest = pd.read_csv(args.manifest)
    required = {
        "model", "data_seed", "train_seed",
        "result_dir", "result_prefix",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError("manifest missing columns: %s" % sorted(missing))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in manifest.sort_values(["model", "data_seed", "train_seed"]).iterrows():
        samples = load_samples(r, args.noise_dir)
        m = metrics_from_samples(
            samples,
            a1_tol=args.a1_success_tol,
            gamma_factor_tol=args.gamma_success_factor,
        )
        rows.append(
            {
                "model": str(r["model"]).lower(),
                "source": str(r.get("source", "exp33")),
                "data_seed": int(r["data_seed"]),
                "train_seed": int(r["train_seed"]),
                "result_dir": str(r["result_dir"]),
                **m,
            }
        )

    per_run = pd.DataFrame(rows).sort_values(["model", "data_seed", "train_seed"])
    per_run.to_csv(out / "exp33_per_run_metrics.csv", index=False, encoding="utf-8-sig")

    model_summary = pd.DataFrame(
        [
            summarize_runs(g, {"model": model})
            for model, g in per_run.groupby("model", sort=True)
        ]
    )
    model_summary.to_csv(
        out / "exp33_model_summary.csv", index=False, encoding="utf-8-sig"
    )

    by_data = pd.DataFrame(
        [
            summarize_runs(g, {"model": model, "data_seed": int(data_seed)})
            for (model, data_seed), g in per_run.groupby(["model", "data_seed"], sort=True)
        ]
    )
    by_data.to_csv(
        out / "exp33_by_data_seed.csv", index=False, encoding="utf-8-sig"
    )

    deltas = matched_deltas(per_run)
    deltas.to_csv(
        out / "exp33_matched_mlp_tcn_deltas.csv", index=False, encoding="utf-8-sig"
    )

    gate = build_stage_gate(per_run, args.target_rate, args.fallback_rate)
    gate.to_csv(out / "exp33_stage_gate.csv", index=False, encoding="utf-8-sig")

    make_plots(out, per_run)

    print("=" * 100)
    print("Exp33 analysis complete")
    print(
        "Recovery definition: |da1|<=%.6g AND gamma-factor<=%.4g"
        % (args.a1_success_tol, args.gamma_success_factor)
    )
    print("")
    print("MODEL SUMMARY")
    for _, r in model_summary.iterrows():
        print(
            "%-4s | joint %.2f%% +/- %.2f%% (min %.2f%%) | a1 MAE %.6g | "
            "gamma P90 %.4f"
            % (
                str(r["model"]).upper(),
                100.0 * float(r["joint_recovery_rate_mean"]),
                100.0 * float(r["joint_recovery_rate_std"]),
                100.0 * float(r["joint_recovery_rate_min"]),
                float(r["a1_mae_mean"]),
                float(r["gamma_p90_factor_mean"]),
            )
        )
    print("")
    print("STAGE GATE")
    for _, r in gate.iterrows():
        print(
            "%-4s | >=%.0f%%: %d/%d runs | >=%.0f%%: %d/%d runs"
            % (
                str(r["model"]).upper(),
                100 * args.target_rate,
                int(r["runs_ge_target"]),
                int(r["run_count"]),
                100 * args.fallback_rate,
                int(r["runs_ge_fallback"]),
                int(r["run_count"]),
            )
        )
    print("")
    print("Read first:")
    print("  %s" % (out / "exp33_model_summary.csv"))
    print("  %s" % (out / "exp33_per_run_metrics.csv"))
    print("  %s" % (out / "exp33_matched_mlp_tcn_deltas.csv"))
    print("  %s" % (out / "exp33_stage_gate.csv"))
    print("=" * 100)


if __name__ == "__main__":
    main()
