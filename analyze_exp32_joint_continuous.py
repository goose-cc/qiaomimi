#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate Exp32 data-seed x train-seed runs using continuous errors.

No gamma-within-x1.2 accuracy is used for ranking or stage-gating.  The report
focuses on a1 absolute error, gamma log/factor error, physics reconstruction,
and how those quantities vary with data construction and training seed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze Exp32 continuous a1+gamma validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--noise-dir", default="noise_0p2pct")
    return p.parse_args()


def q(x, value):
    return float(np.quantile(np.asarray(x, dtype=np.float64), value))


def state_metrics(samples: pd.DataFrame):
    rows = []
    for (a1, gamma), g in samples.groupby(["true_a1", "true_gamma"], sort=False):
        pa = float(np.median(g["pred_a1"].to_numpy(float)))
        # Median in log-space is the natural center for positive gamma.
        pg = float(np.exp(np.median(np.log(np.clip(g["pred_gamma"].to_numpy(float), 1e-30, None)))))
        aerr = abs(pa - float(a1))
        glog = abs(np.log10(max(pg, 1e-30)) - np.log10(max(float(gamma), 1e-30)))
        gfactor = float(np.exp(abs(np.log(max(pg, 1e-30)) - np.log(max(float(gamma), 1e-30)))))
        rows.append(
            {
                "true_a1": float(a1),
                "true_gamma": float(gamma),
                "median_pred_a1": pa,
                "median_pred_gamma": pg,
                "state_a1_abs_error": aerr,
                "state_gamma_abs_log10_error": glog,
                "state_gamma_factor_error": gfactor,
                "noise_realization_count": int(len(g)),
            }
        )
    sdf = pd.DataFrame(rows)
    if len(sdf) == 0:
        raise ValueError("no physical states found")
    metrics = {
        "state_count": int(len(sdf)),
        "state_a1_mae": float(sdf["state_a1_abs_error"].mean()),
        "state_a1_median_abs_error": float(sdf["state_a1_abs_error"].median()),
        "state_a1_p90_abs_error": q(sdf["state_a1_abs_error"], 0.90),
        "state_gamma_log10_mae": float(sdf["state_gamma_abs_log10_error"].mean()),
        "state_gamma_median_abs_log10_error": float(sdf["state_gamma_abs_log10_error"].median()),
        "state_gamma_p90_abs_log10_error": q(sdf["state_gamma_abs_log10_error"], 0.90),
        "state_gamma_median_factor": float(sdf["state_gamma_factor_error"].median()),
        "state_gamma_p90_factor": q(sdf["state_gamma_factor_error"], 0.90),
        "state_gamma_max_factor": float(sdf["state_gamma_factor_error"].max()),
    }
    return metrics, sdf


def load_run(row, noise_dir: str):
    result_dir = Path(str(row["result_dir"]))
    prefix = str(row["result_prefix"])
    summary_path = result_dir / (prefix + "_summary.csv")
    samples_path = result_dir / (prefix + "_samples.csv")
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)

    summary = pd.read_csv(summary_path)
    summary = summary[summary["noise_dir"].astype(str) == str(noise_dir)]
    if len(summary) != 1:
        raise ValueError("%s: expected exactly one row for %s" % (summary_path, noise_dir))

    samples = pd.read_csv(samples_path)
    samples = samples[samples["noise_dir"].astype(str) == str(noise_dir)].copy()
    if len(samples) == 0:
        raise ValueError("%s: no samples for %s" % (samples_path, noise_dir))

    sm, sdf = state_metrics(samples)
    base = summary.iloc[0].to_dict()
    base.update(sm)
    base.update(
        {
            "data_seed": int(row["data_seed"]),
            "train_seed": int(row["train_seed"]),
            "data_dir": str(row["data_dir"]),
            "result_dir": str(row["result_dir"]),
        }
    )
    return base, samples, sdf


def numeric_metric_columns(df: pd.DataFrame):
    excluded = {
        "data_seed",
        "train_seed",
        "noise_level",
        "count",
        "state_count",
    }
    return [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in excluded
    ]


def summarize_group(df: pd.DataFrame, group_name: str, group_value, metric_cols):
    row = {group_name: group_value, "run_count": int(len(df))}
    for c in metric_cols:
        row[c + "_mean"] = float(df[c].mean())
        row[c + "_std"] = float(df[c].std(ddof=0))
        row[c + "_min"] = float(df[c].min())
        row[c + "_max"] = float(df[c].max())
    return row


def build_state_difficulty(all_state_frames):
    merged = pd.concat(all_state_frames, ignore_index=True)
    rows = []
    for (a1, gamma), g in merged.groupby(["true_a1", "true_gamma"], sort=False):
        rows.append(
            {
                "true_a1": float(a1),
                "true_gamma": float(gamma),
                "run_occurrence_count": int(len(g)),
                "mean_state_a1_abs_error": float(g["state_a1_abs_error"].mean()),
                "p90_state_a1_abs_error": q(g["state_a1_abs_error"], 0.90),
                "mean_state_gamma_abs_log10_error": float(g["state_gamma_abs_log10_error"].mean()),
                "p90_state_gamma_abs_log10_error": q(g["state_gamma_abs_log10_error"], 0.90),
                "median_state_gamma_factor": float(g["state_gamma_factor_error"].median()),
                "p90_state_gamma_factor": q(g["state_gamma_factor_error"], 0.90),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # A continuous difficulty score for ordering only; it is not a pass/fail gate.
    # Standardize the two errors by their medians to avoid mixing physical units.
    a_scale = max(float(out["mean_state_a1_abs_error"].median()), 1e-12)
    g_scale = max(float(out["mean_state_gamma_abs_log10_error"].median()), 1e-12)
    out["relative_difficulty_score"] = (
        out["mean_state_a1_abs_error"] / a_scale
        + out["mean_state_gamma_abs_log10_error"] / g_scale
    )
    return out.sort_values("relative_difficulty_score", ascending=False)


def make_plots(out: Path, per_run: pd.DataFrame, all_samples: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    x = np.arange(len(per_run))
    labels = ["D%d/T%d" % (d, t) for d, t in zip(per_run["data_seed"], per_run["train_seed"])]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, per_run["a1_mae"], marker="o", label="a1 MAE")
    ax.plot(x, per_run["a1_p90_abs_error"], marker="s", label="a1 P90 abs error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("a1 absolute error")
    ax.set_title("Exp32 a1 continuous-error stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "exp32_a1_error_stability.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, per_run["gamma_log10_mae"], marker="o", label="gamma log10 MAE")
    ax.plot(x, per_run["gamma_p90_abs_log10_error"], marker="s", label="gamma P90 abs log10 error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|log10(gamma_hat/gamma)|")
    ax.set_title("Exp32 gamma continuous-error stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "exp32_gamma_error_stability.png", dpi=170)
    plt.close(fig)

    # Error coupling view: if a1 and gamma still compensate for one another, the
    # signed errors reveal it more clearly than an x1.2 accuracy statistic.
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(all_samples) > 30000:
        all_samples = all_samples.sample(30000, random_state=3201)
    ax.scatter(
        all_samples["a1_signed_error"],
        all_samples["gamma_signed_log10_error"],
        s=8,
        alpha=0.25,
    )
    ax.axhline(0.0, linewidth=1)
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("a1 signed error")
    ax.set_ylabel("log10(gamma_hat/gamma)")
    ax.set_title("Exp32 residual a1-gamma error coupling")
    fig.tight_layout()
    fig.savefig(out / "exp32_a1_gamma_error_coupling.png", dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    required = {"data_seed", "train_seed", "data_dir", "result_dir", "result_prefix"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError("manifest missing columns: %s" % sorted(missing))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_run_rows = []
    all_samples = []
    all_states = []
    for _, row in manifest.sort_values(["data_seed", "train_seed"]).iterrows():
        metrics, samples, states = load_run(row, args.noise_dir)
        per_run_rows.append(metrics)
        samples = samples.copy()
        samples["data_seed"] = int(row["data_seed"])
        samples["train_seed"] = int(row["train_seed"])
        states = states.copy()
        states["data_seed"] = int(row["data_seed"])
        states["train_seed"] = int(row["train_seed"])
        all_samples.append(samples)
        all_states.append(states)

    per_run = pd.DataFrame(per_run_rows).sort_values(["data_seed", "train_seed"])
    per_run.to_csv(out / "exp32_per_run_metrics.csv", index=False, encoding="utf-8-sig")

    metric_cols = numeric_metric_columns(per_run)
    overall = summarize_group(per_run, "scope", "all_runs", metric_cols)
    pd.DataFrame([overall]).to_csv(
        out / "exp32_overall_continuous_summary.csv", index=False, encoding="utf-8-sig"
    )

    data_rows = [
        summarize_group(g, "data_seed", int(seed), metric_cols)
        for seed, g in per_run.groupby("data_seed", sort=True)
    ]
    pd.DataFrame(data_rows).to_csv(
        out / "exp32_by_data_seed.csv", index=False, encoding="utf-8-sig"
    )

    train_rows = [
        summarize_group(g, "train_seed", int(seed), metric_cols)
        for seed, g in per_run.groupby("train_seed", sort=True)
    ]
    pd.DataFrame(train_rows).to_csv(
        out / "exp32_by_train_seed.csv", index=False, encoding="utf-8-sig"
    )

    all_samples_df = pd.concat(all_samples, ignore_index=True)
    all_samples_df.to_csv(
        out / "exp32_all_run_samples.csv", index=False, encoding="utf-8-sig"
    )

    difficult = build_state_difficulty(all_states)
    if len(difficult):
        difficult.to_csv(
            out / "exp32_state_difficulty.csv", index=False, encoding="utf-8-sig"
        )

    # Continuous signed-error correlation is a useful diagnostic of residual
    # a1-gamma compensation; it is not an accuracy threshold.
    corr = float(
        np.corrcoef(
            all_samples_df["a1_signed_error"].to_numpy(float),
            all_samples_df["gamma_signed_log10_error"].to_numpy(float),
        )[0, 1]
    )
    pd.DataFrame(
        [
            {
                "sample_count_across_runs": int(len(all_samples_df)),
                "a1_gamma_signed_error_correlation": corr,
                "interpretation": (
                    "large |correlation| means residual a1-gamma compensation remains; "
                    "near zero means the coupling is largely removed in prediction errors"
                ),
            }
        ]
    ).to_csv(out / "exp32_error_coupling_summary.csv", index=False, encoding="utf-8-sig")

    make_plots(out, per_run, all_samples_df)

    print("=" * 100)
    print("Exp32 continuous-error aggregation complete")
    print("Read first:")
    print("  %s" % (out / "exp32_overall_continuous_summary.csv"))
    print("  %s" % (out / "exp32_per_run_metrics.csv"))
    print("  %s" % (out / "exp32_by_data_seed.csv"))
    print("  %s" % (out / "exp32_error_coupling_summary.csv"))
    print("=" * 100)


if __name__ == "__main__":
    main()
