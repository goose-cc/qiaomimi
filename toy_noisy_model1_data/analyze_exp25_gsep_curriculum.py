#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze whether an easy->medium->hard g-space curriculum improves the SAME hard target task."""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from data_generate_exp20_pairwise import build_parameter_rows, forward_bank
from data_generate_exp22c_fixed_capacity import canonical_indices
from mc_pool_config import DEFAULT_PHYSICS


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare Exp25 full-from-start vs g-space curriculum",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--baseline-prefix", default="exp25a")
    p.add_argument("--curriculum-dir", required=True)
    p.add_argument("--curriculum-prefix", default="exp25b")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--noise-dir", default="noise_0p2pct")
    p.add_argument("--reference-noise", type=float, default=0.002)
    p.add_argument("--observation-points", type=int, default=500)
    p.add_argument("--model-input-points", type=int, default=1000)
    p.add_argument("--integration-points", type=int, default=128)
    return p.parse_args()


def factor(pred, true):
    pred = np.clip(np.asarray(pred, dtype=float), 1e-30, None)
    true = np.clip(np.asarray(true, dtype=float), 1e-30, None)
    return np.maximum(pred / true, true / pred)


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


def unique_states_from_data(data_dir, noise_dir):
    pairs = []
    for split in ("test_seen", "test_interp_gamma", "test_interp_second", "test_interp_both"):
        path = data_dir / noise_dir / (split + ".npz")
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as z:
            a1 = np.asarray(z["a1"], dtype=np.float64).reshape(-1)
            gamma = np.asarray(z["gamma"], dtype=np.float64).reshape(-1)
        for a, g in zip(a1, gamma):
            pairs.append((round(float(a), 10), round(float(g), 10)))
    pairs = sorted(set(pairs))
    return np.asarray(pairs, dtype=np.float64)


def local_g_separation(states, observation_points, model_input_points, integration_points, reference_noise):
    physics = replace(DEFAULT_PHYSICS, q2_points=int(model_input_points))
    obs_idx = canonical_indices(model_input_points, observation_points)
    params = np.zeros((len(states), 5), dtype=np.float64)
    params[:, 0] = states[:, 0]
    params[:, 1] = 0.025
    params[:, 2] = 0.0
    params[:, 3] = 0.8
    params[:, 4] = states[:, 1]
    _, g_master = forward_bank(params, physics=physics, integration_points=integration_points)
    g = np.asarray(g_master[:, obs_idx], dtype=np.float64)
    rms = np.sqrt(np.mean(g * g, axis=1))

    scores = np.empty(len(states), dtype=np.float64)
    nearest = np.empty(len(states), dtype=np.int32)
    for i in range(len(states)):
        d = g - g[i]
        rms_d = np.sqrt(np.mean(d * d, axis=1))
        sigma = reference_noise * 0.5 * (rms + rms[i])
        snr = rms_d / np.maximum(sigma, 1e-30)
        snr[i] = np.inf
        j = int(np.argmin(snr))
        scores[i] = float(snr[j])
        nearest[i] = j
    return scores, nearest


def assign_score(df, states, scores):
    keys = {(round(float(a), 10), round(float(g), 10)): float(s)
            for (a, g), s in zip(states, scores)}
    vals = []
    for a, g in zip(df["true_a1"], df["true_gamma"]):
        key = (round(float(a), 10), round(float(g), 10))
        if key not in keys:
            # float32 CSV round-off fallback
            da = (states[:, 0] - float(a)) / max(np.ptp(states[:, 0]), 1e-12)
            dg = (np.log10(states[:, 1]) - math.log10(float(g))) / max(np.ptp(np.log10(states[:, 1])), 1e-12)
            idx = int(np.argmin(da * da + dg * dg))
            vals.append(float(scores[idx]))
        else:
            vals.append(keys[key])
    return np.asarray(vals, dtype=np.float64)


def summary_row(model, split, df):
    ferr = factor(df["pred_gamma"], df["true_gamma"])
    return {
        "model": model,
        "split": split,
        "count": int(len(df)),
        "gamma_median_factor": float(np.median(ferr)),
        "gamma_p90_factor": float(np.quantile(ferr, 0.9)),
        "gamma_within_x1p2": float(np.mean(ferr <= 1.2)),
        "catastrophic_factor_ge5": float(np.mean(ferr >= 5.0)),
        "catastrophic_factor_ge10": float(np.mean(ferr >= 10.0)),
        "median_f_relative_l2": float(np.nanmedian(df["f_relative_l2"])),
        "median_g_clean_relative_l2": float(np.nanmedian(df["g_clean_relative_l2"])),
    }


def save_scatter(path, df, ycol, ylabel):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in ("full_from_start", "gsep_curriculum"):
        sub = df[df["model"] == model]
        ax.scatter(sub["local_gsep_rms_snr"], sub[ycol], s=9, alpha=0.25, label=model)
    ax.set_xlabel("local clean g separation (nearest-state RMS-SNR)")
    ax.set_ylabel(ylabel)
    if ycol == "gamma_factor_error":
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=165)
    plt.close(fig)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    states = unique_states_from_data(data_dir, args.noise_dir)
    scores, nearest = local_g_separation(
        states,
        args.observation_points,
        args.model_input_points,
        args.integration_points,
        args.reference_noise,
    )
    state_rows = []
    for i, ((a, g), s, j) in enumerate(zip(states, scores, nearest)):
        state_rows.append({
            "state_index": i,
            "a1": float(a),
            "gamma": float(g),
            "local_gsep_rms_snr": float(s),
            "nearest_state": int(j),
            "nearest_a1": float(states[j, 0]),
            "nearest_gamma": float(states[j, 1]),
        })
    write_csv(out / "evaluation_state_local_gseparation.csv", state_rows)

    configs = [
        ("full_from_start", Path(args.baseline_dir) / (args.baseline_prefix + "_samples.csv")),
        ("gsep_curriculum", Path(args.curriculum_dir) / (args.curriculum_prefix + "_samples.csv")),
    ]
    all_df = []
    summaries = []
    for model, path in configs:
        df = pd.read_csv(path)
        df = df[df["noise_dir"] == args.noise_dir].copy()
        df["model"] = model
        df["gamma_factor_error"] = factor(df["pred_gamma"], df["true_gamma"])
        df["local_gsep_rms_snr"] = assign_score(df, states, scores)
        all_df.append(df)
        for split, sdf in df.groupby("split", sort=True):
            summaries.append(summary_row(model, split, sdf))

    combined = pd.concat(all_df, ignore_index=True)
    write_csv(out / "exp25_comparison_summary.csv", summaries)

    # Separation-conditioned quartiles are computed globally on physical states,
    # then applied identically to both models.
    q = np.quantile(scores, [0.25, 0.5, 0.75])
    labels = [
        "Q1_hardest_low_gsep",
        "Q2",
        "Q3",
        "Q4_easiest_high_gsep",
    ]
    conditioned = []
    for model in ("full_from_start", "gsep_curriculum"):
        mdf = combined[combined["model"] == model]
        for split, sdf in mdf.groupby("split", sort=True):
            bins = np.digitize(sdf["local_gsep_rms_snr"].to_numpy(), q, right=True)
            for b, label in enumerate(labels):
                gdf = sdf[bins == b]
                if len(gdf) == 0:
                    continue
                r = summary_row(model, split, gdf)
                r["gsep_bin"] = label
                r["gsep_min"] = float(gdf["local_gsep_rms_snr"].min())
                r["gsep_median"] = float(gdf["local_gsep_rms_snr"].median())
                r["gsep_max"] = float(gdf["local_gsep_rms_snr"].max())
                conditioned.append(r)
    write_csv(out / "exp25_error_by_gseparation_quartile.csv", conditioned)

    # Explicit catastrophic cases.
    cats = combined[combined["gamma_factor_error"] >= 5.0].copy()
    if len(cats):
        cats = cats.sort_values(["model", "gamma_factor_error"], ascending=[True, False])
        cols = [
            "model", "split", "true_a1", "pred_a1", "true_gamma", "pred_gamma",
            "gamma_factor_error", "local_gsep_rms_snr",
            "f_relative_l2", "g_clean_relative_l2",
        ]
        cats[cols].to_csv(out / "exp25_catastrophic_gamma_cases.csv", index=False, encoding="utf-8-sig")

    save_scatter(
        out / "gamma_factor_error_vs_local_gseparation.png",
        combined,
        "gamma_factor_error",
        "gamma factor error",
    )
    save_scatter(
        out / "f_relative_l2_vs_local_gseparation.png",
        combined,
        "f_relative_l2",
        "f relative L2",
    )

    # Median trend by physical separation quartile.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cdf = pd.DataFrame(conditioned)
        for split in ("test_interp_gamma", "test_interp_both"):
            s = cdf[cdf["split"] == split]
            if s.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for model in ("full_from_start", "gsep_curriculum"):
                x = s[s["model"] == model].copy()
                order = {lab: i for i, lab in enumerate(labels)}
                x["order"] = x["gsep_bin"].map(order)
                x = x.sort_values("order")
                ax.plot(x["order"], x["gamma_median_factor"], marker="o", label=model)
            ax.set_xticks(range(4), ["hardest", "Q2", "Q3", "easiest"])
            ax.set_xlabel("local g-separation quartile")
            ax.set_ylabel("median gamma factor error")
            ax.set_title("Exp25 %s: does curriculum help the hard low-g-separation states?" % split)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / ("gamma_error_by_gsep_quartile_%s.png" % split), dpi=165)
            plt.close(fig)
    except Exception:
        pass

    print("=" * 100)
    print("Exp25 comparison ready")
    print("The key question is NOT whether high-g-separation states are easier.")
    print("The key question is whether easy->medium->full curriculum improves the SAME hard test set.")
    print("Read first:")
    print("  %s" % (out / "exp25_comparison_summary.csv"))
    print("  %s" % (out / "exp25_error_by_gseparation_quartile.csv"))
    print("  %s" % (out / "exp25_catastrophic_gamma_cases.csv"))
    print("=" * 100)


if __name__ == "__main__":
    main()
