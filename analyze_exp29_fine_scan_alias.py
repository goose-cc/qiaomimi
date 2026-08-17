#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp29: fine separation scan + continuous off-bank alias audit.

This experiment answers two questions left open by Exp27/Exp28:

1) The best tested minimum g-separation was ~1.5, but only 1.0/1.5/2.0 were
   tested.  Could the optimum lie between 1.0 and 1.5?
2) Why can a network prediction still produce a g curve very close to truth
   even though the retained physical state bank is globally separated?

The second point is subtle.  The Exp26/27 separation guarantee applies only to
the RETAINED TRUE STATES.  The neural network outputs continuous (a1, gamma),
so its prediction can land at an unselected point in the continuous parameter
domain.  A remote unselected point can still be a near-alias in g-space.

To diagnose that, this script creates one independent dense continuous probe
pool and, for every retained state, finds the closest g-space probe that is
PHYSICALLY REMOTE according to either:
    |delta a1| >= remote_a1_abs
or
    gamma factor >= remote_gamma_factor.

A small remote-alias RMS-SNR means the selected state is not globally unique
against the continuous domain, even if the selected state bank itself is well
separated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from data_generate_exp20_pairwise import DEFAULT_PHYSICS, FIXED_A2, FIXED_A3, FIXED_M, forward_bank
from data_generate_exp22c_fixed_capacity import canonical_indices


def parse_args():
    p = argparse.ArgumentParser(
        description="Exp29 fine g-separation scan + continuous alias audit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--noise-dir", default="noise_0p2pct")
    p.add_argument("--probe-count", type=int, default=12000)
    p.add_argument("--probe-seed", type=int, default=20260829)
    p.add_argument("--remote-a1-abs", type=float, default=0.03)
    p.add_argument("--remote-gamma-factor", type=float, default=2.0)
    p.add_argument("--audit-chunk", type=int, default=2048)
    return p.parse_args()


def latin_hypercube_1d(n, rng):
    x = (np.arange(n, dtype=np.float64) + rng.random(n)) / float(n)
    rng.shuffle(x)
    return x


def probe_parameters(n, seed, a1_min, a1_max, gamma_min, gamma_max):
    rng = np.random.default_rng(int(seed))
    u = latin_hypercube_1d(int(n), rng)
    v = latin_hypercube_1d(int(n), rng)
    a1 = float(a1_min) + (float(a1_max) - float(a1_min)) * u
    lg0 = math.log10(float(gamma_min))
    lg1 = math.log10(float(gamma_max))
    gamma = np.power(10.0, lg0 + (lg1 - lg0) * v)

    params = np.zeros((int(n), 5), dtype=np.float64)
    params[:, 0] = a1
    params[:, 1] = FIXED_A2
    params[:, 2] = FIXED_A3
    params[:, 3] = FIXED_M
    params[:, 4] = gamma
    return params


def selected_params(data_dir):
    states = pd.read_csv(Path(data_dir) / "selected_states.csv")
    if "state_index" in states.columns:
        states = states.sort_values("state_index").reset_index(drop=True)
    p = np.zeros((len(states), 5), dtype=np.float64)
    p[:, 0] = states["a1"].to_numpy(float)
    p[:, 1] = FIXED_A2
    p[:, 2] = FIXED_A3
    p[:, 3] = FIXED_M
    p[:, 4] = states["gamma"].to_numpy(float)
    return states, p


def g_distance_snr(g_a, rms_a, g_b, rms_b, reference_noise):
    d = np.asarray(g_b, dtype=np.float64) - np.asarray(g_a, dtype=np.float64)[None, :]
    rms_d = np.sqrt(np.mean(d * d, axis=1))
    sigma = float(reference_noise) * 0.5 * (
        np.asarray(rms_b, dtype=np.float64) + float(rms_a)
    )
    return rms_d / np.maximum(sigma, 1e-30)


def remote_alias_audit(
    selected,
    selected_g,
    selected_rms,
    probe_params,
    probe_g,
    probe_rms,
    reference_noise,
    remote_a1_abs,
    remote_gamma_factor,
    chunk,
):
    rows = []
    pa = probe_params[:, 0]
    pg = probe_params[:, 4]

    for i in range(len(selected)):
        a = float(selected[i, 0])
        g = float(selected[i, 4])
        gamma_factor = np.maximum(pg / g, g / pg)
        remote = (np.abs(pa - a) >= float(remote_a1_abs)) | (
            gamma_factor >= float(remote_gamma_factor)
        )
        remote_idx = np.where(remote)[0]
        if len(remote_idx) == 0:
            raise RuntimeError("no remote probes for selected state %d" % i)

        best_snr = math.inf
        best_j = -1
        for start in range(0, len(remote_idx), int(chunk)):
            idx = remote_idx[start:start + int(chunk)]
            snr = g_distance_snr(
                selected_g[i], selected_rms[i],
                probe_g[idx], probe_rms[idx],
                reference_noise,
            )
            k = int(np.argmin(snr))
            if float(snr[k]) < best_snr:
                best_snr = float(snr[k])
                best_j = int(idx[k])

        rows.append({
            "state_index": int(i),
            "a1": a,
            "gamma": g,
            "remote_alias_rms_snr": best_snr,
            "alias_a1": float(probe_params[best_j, 0]),
            "alias_gamma": float(probe_params[best_j, 4]),
            "alias_abs_delta_a1": float(abs(probe_params[best_j, 0] - a)),
            "alias_gamma_factor": float(max(
                probe_params[best_j, 4] / g,
                g / probe_params[best_j, 4],
            )),
        })
    return rows


def load_run_metrics(result_dir, prefix, noise_dir, reference_noise):
    summary = pd.read_csv(Path(result_dir) / (str(prefix) + "_summary.csv"))
    samples = pd.read_csv(Path(result_dir) / (str(prefix) + "_samples.csv"))

    row = summary[summary["noise_dir"] == noise_dir]
    if row.empty:
        raise ValueError("%s has no %s row" % (result_dir, noise_dir))
    row = row.iloc[0]

    ss = samples[samples["noise_dir"] == noise_dir].copy()
    if ss.empty:
        raise ValueError("%s has no %s samples" % (result_dir, noise_dir))

    factor = ss["gamma_factor_error"].to_numpy(float)
    f = ss["f_relative_l2"].to_numpy(float)
    g_rel = ss["g_clean_relative_l2"].to_numpy(float)

    # This ratio is only an intuitive approximation to the exact symmetric
    # RMS-SNR used in the data selector, but it is useful for interpreting why
    # a seemingly small relative g error need not violate a MinRMSSNR threshold.
    approx_g_snr = g_rel / float(reference_noise)

    return {
        "gamma_median_factor": float(row["gamma_median_factor"]),
        "gamma_p90_factor": float(row["gamma_p90_factor"]),
        "gamma_p99_factor": float(np.quantile(factor, 0.99)),
        "gamma_max_factor": float(np.max(factor)),
        "gamma_within_x1p2": float(row["gamma_within_x1p2"]),
        "gamma_within_x1p5": float(row["gamma_within_x1p5"]),
        "gamma_within_x2": float(row["gamma_within_x2"]),
        "median_f_relative_l2": float(row["median_f_relative_l2"]),
        "p90_f_relative_l2": float(row["p90_f_relative_l2"]),
        "p99_f_relative_l2": float(np.quantile(f, 0.99)),
        "catastrophic_gamma_factor_ge5": float(row["catastrophic_gamma_factor_ge5"]),
        "catastrophic_gamma_factor_ge10": float(row["catastrophic_gamma_factor_ge10"]),
        "median_g_clean_relative_l2": float(row["median_g_clean_relative_l2"]),
        "worst_sample_approx_g_snr": float(
            approx_g_snr[int(np.argmax(factor))]
        ),
    }, ss


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


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    required = {"threshold", "seed", "data_dir", "result_dir", "result_prefix"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError("manifest missing columns: %s" % sorted(missing))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # All scan points intentionally use the same physical model/domain.
    first_meta = json.loads(
        (Path(str(manifest.iloc[0]["data_dir"])) / "metadata.json").read_text(encoding="utf-8")
    )
    a1_min, a1_max = [float(x) for x in first_meta["a1_range"]]
    gamma_min, gamma_max = [float(x) for x in first_meta["gamma_range"]]
    reference_noise = float(first_meta["reference_noise"])
    obs_points = int(first_meta["physical_observation_points"])
    model_points = int(first_meta["model_input_points"])
    integration_points = int(first_meta.get("integration_points", 128))
    physics = replace(DEFAULT_PHYSICS, q2_points=model_points)
    obs_idx = canonical_indices(model_points, obs_points)

    print("=" * 100)
    print("Exp29 continuous alias audit")
    print("probe count              : %d" % args.probe_count)
    print("remote definition        : |delta a1| >= %.4g OR gamma factor >= %.4g"
          % (args.remote_a1_abs, args.remote_gamma_factor))
    print("physical observations    : %d" % obs_points)
    print("=" * 100)

    probes = probe_parameters(
        args.probe_count, args.probe_seed,
        a1_min, a1_max, gamma_min, gamma_max,
    )
    _, probe_g_master = forward_bank(
        probes, physics=physics, integration_points=integration_points
    )
    probe_g_master = np.asarray(probe_g_master, dtype=np.float32)
    probe_g = probe_g_master[:, obs_idx]
    probe_rms = np.sqrt(np.mean(probe_g_master.astype(np.float64) ** 2, axis=1))

    per_run_rows = []
    catastrophic_rows = []
    threshold_alias_rows = []

    # Audit a physical bank only once per threshold; training seeds share data.
    audited = {}

    for _, entry in manifest.sort_values(["threshold", "seed"]).iterrows():
        threshold = float(entry["threshold"])
        seed = int(entry["seed"])
        data_dir = Path(str(entry["data_dir"]))
        result_dir = Path(str(entry["result_dir"]))
        prefix = str(entry["result_prefix"])
        meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))

        metrics, samples = load_run_metrics(
            result_dir, prefix, args.noise_dir, float(meta["reference_noise"])
        )

        row = {
            "requested_min_rms_snr": threshold,
            "train_seed": seed,
            "actual_global_min_rms_snr": float(meta["global_min_rms_snr"]),
            "selected_state_count": int(meta["selected_state_count"]),
            "train_state_count": int(meta["selected_state_counts"]["train"]),
            "val_state_count": int(meta["selected_state_counts"]["val"]),
            "test_state_count": int(meta["selected_state_counts"]["test"]),
            **metrics,
        }
        per_run_rows.append(row)

        cats = samples[samples["gamma_factor_error"] >= 3.0].copy()
        if len(cats):
            cats = cats.sort_values("gamma_factor_error", ascending=False).head(20)
            for _, r in cats.iterrows():
                catastrophic_rows.append({
                    "requested_min_rms_snr": threshold,
                    "train_seed": seed,
                    "true_a1": float(r["true_a1"]),
                    "pred_a1": float(r["pred_a1"]),
                    "true_gamma": float(r["true_gamma"]),
                    "pred_gamma": float(r["pred_gamma"]),
                    "gamma_factor_error": float(r["gamma_factor_error"]),
                    "f_relative_l2": float(r["f_relative_l2"]),
                    "g_clean_relative_l2": float(r["g_clean_relative_l2"]),
                    "approx_g_rms_snr_from_relL2": float(
                        r["g_clean_relative_l2"] / float(meta["reference_noise"])
                    ),
                })

        if threshold not in audited:
            states_df, sp = selected_params(data_dir)
            _, selected_g_master = forward_bank(
                sp, physics=physics, integration_points=integration_points
            )
            selected_g_master = np.asarray(selected_g_master, dtype=np.float32)
            selected_g = selected_g_master[:, obs_idx]
            selected_rms = np.sqrt(
                np.mean(selected_g_master.astype(np.float64) ** 2, axis=1)
            )
            alias_rows = remote_alias_audit(
                sp, selected_g, selected_rms,
                probes, probe_g, probe_rms,
                reference_noise,
                args.remote_a1_abs,
                args.remote_gamma_factor,
                args.audit_chunk,
            )
            for ar in alias_rows:
                ar["requested_min_rms_snr"] = threshold
            write_csv(
                out / ("continuous_alias_states_snr%s.csv" % str(threshold).replace(".", "p")),
                alias_rows,
            )
            arr = np.asarray([r["remote_alias_rms_snr"] for r in alias_rows], dtype=float)
            threshold_alias_rows.append({
                "requested_min_rms_snr": threshold,
                "selected_state_count": len(arr),
                "remote_alias_median_rms_snr": float(np.median(arr)),
                "remote_alias_p10_rms_snr": float(np.quantile(arr, 0.10)),
                "remote_alias_min_rms_snr": float(np.min(arr)),
                "fraction_remote_alias_below_1p0": float(np.mean(arr < 1.0)),
                "fraction_remote_alias_below_1p5": float(np.mean(arr < 1.5)),
                "fraction_remote_alias_below_2p0": float(np.mean(arr < 2.0)),
                "fraction_remote_alias_below_own_threshold": float(np.mean(arr < threshold)),
            })
            audited[threshold] = True

    run_df = pd.DataFrame(per_run_rows)
    run_df.to_csv(out / "exp29_per_run_metrics.csv", index=False, encoding="utf-8-sig")
    write_csv(out / "exp29_catastrophic_prediction_cases.csv", catastrophic_rows)
    alias_df = pd.DataFrame(threshold_alias_rows).sort_values("requested_min_rms_snr")
    alias_df.to_csv(
        out / "exp29_continuous_alias_summary.csv",
        index=False, encoding="utf-8-sig",
    )

    # Aggregate multiple training seeds if supplied.
    numeric_cols = [
        "actual_global_min_rms_snr", "selected_state_count", "train_state_count",
        "val_state_count", "test_state_count", "gamma_median_factor", "gamma_p90_factor",
        "gamma_p99_factor", "gamma_max_factor", "gamma_within_x1p2",
        "gamma_within_x1p5", "gamma_within_x2", "median_f_relative_l2",
        "p90_f_relative_l2", "p99_f_relative_l2", "catastrophic_gamma_factor_ge5",
        "catastrophic_gamma_factor_ge10", "median_g_clean_relative_l2",
        "worst_sample_approx_g_snr",
    ]
    agg_rows = []
    for threshold, gdf in run_df.groupby("requested_min_rms_snr", sort=True):
        r = {
            "requested_min_rms_snr": float(threshold),
            "training_seed_count": int(len(gdf)),
        }
        for col in numeric_cols:
            r[col + "_mean"] = float(gdf[col].mean())
            r[col + "_std"] = float(gdf[col].std(ddof=0))
        alias_match = alias_df[
            np.isclose(alias_df["requested_min_rms_snr"], threshold)
        ]
        if len(alias_match):
            for col in alias_df.columns:
                if col != "requested_min_rms_snr":
                    r[col] = float(alias_match.iloc[0][col])
        agg_rows.append(r)

    agg = pd.DataFrame(agg_rows).sort_values("requested_min_rms_snr")
    agg.to_csv(
        out / "exp29_fine_scan_alias_summary.csv",
        index=False, encoding="utf-8-sig",
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            agg["requested_min_rms_snr"],
            agg["gamma_within_x1p2_mean"],
            yerr=agg["gamma_within_x1p2_std"],
            marker="o",
            capsize=3,
        )
        ax.set_xlabel("required minimum nearest-state RMS-SNR")
        ax.set_ylabel("gamma within x1.2")
        ax.set_title("Exp29 fine scan: locate the useful separation range")
        fig.tight_layout()
        fig.savefig(out / "gamma_within_x1p2_fine_scan.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            agg["requested_min_rms_snr"],
            agg["gamma_p90_factor_mean"],
            yerr=agg["gamma_p90_factor_std"],
            marker="o", label="gamma p90 factor",
        )
        ax.errorbar(
            agg["requested_min_rms_snr"],
            agg["gamma_median_factor_mean"],
            yerr=agg["gamma_median_factor_std"],
            marker="o", label="gamma median factor",
        )
        ax.set_xlabel("required minimum nearest-state RMS-SNR")
        ax.set_ylabel("gamma factor error")
        ax.set_title("Exp29 gamma error fine scan")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "gamma_error_fine_scan.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            agg["requested_min_rms_snr"],
            agg["median_f_relative_l2_mean"],
            marker="o", label="median f relL2",
        )
        ax.plot(
            agg["requested_min_rms_snr"],
            agg["p90_f_relative_l2_mean"],
            marker="o", label="p90 f relL2",
        )
        ax.set_xlabel("required minimum nearest-state RMS-SNR")
        ax.set_ylabel("f relative L2")
        ax.set_title("Exp29 f reconstruction fine scan")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "f_error_fine_scan.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            alias_df["requested_min_rms_snr"],
            alias_df["fraction_remote_alias_below_1p0"],
            marker="o", label="remote alias RMS-SNR < 1.0",
        )
        ax.plot(
            alias_df["requested_min_rms_snr"],
            alias_df["fraction_remote_alias_below_1p5"],
            marker="o", label="< 1.5",
        )
        ax.plot(
            alias_df["requested_min_rms_snr"],
            alias_df["fraction_remote_alias_below_2p0"],
            marker="o", label="< 2.0",
        )
        ax.set_xlabel("selected-bank minimum RMS-SNR")
        ax.set_ylabel("fraction of selected states")
        ax.set_title("Exp29 hidden continuous aliases outside the selected bank")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "continuous_remote_alias_rate.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            agg["requested_min_rms_snr"],
            agg["selected_state_count_mean"],
            marker="o", label="retained states",
        )
        ax.set_xlabel("required minimum nearest-state RMS-SNR")
        ax.set_ylabel("physical state count")
        ax.set_title("Exp29 coverage price of finer separation scan")
        fig.tight_layout()
        fig.savefig(out / "physical_state_count_fine_scan.png", dpi=170)
        plt.close(fig)
    except Exception as exc:
        print("warning: plotting skipped: %s" % exc)

    print("=" * 100)
    print("Exp29 analysis ready")
    print("Key interpretation:")
    print("  - selected-bank separation only constrains selected TRUE states")
    print("  - the network predicts continuous off-bank parameters")
    print("  - remote_alias_* quantifies near-degeneracy against an independent dense probe pool")
    print("Read first:")
    print("  %s" % (out / "exp29_fine_scan_alias_summary.csv"))
    print("  %s" % (out / "exp29_continuous_alias_summary.csv"))
    print("  %s" % (out / "exp29_catastrophic_prediction_cases.csv"))
    print("=" * 100)


if __name__ == "__main__":
    main()
