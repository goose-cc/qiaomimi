#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp26 post-analysis: verify split separation and show representative independent-test curves."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze Exp26 globally g-separated experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--result-dir", required=True)
    p.add_argument("--result-prefix", default="exp26")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--noise-dir", default="noise_0p2pct")
    p.add_argument("--integration-points", type=int, default=512)
    return p.parse_args()


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


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    selected = pd.read_csv(data_dir / "selected_states.csv")
    pair_summary = pd.read_csv(data_dir / "split_pair_gseparation_summary.csv")
    samples = pd.read_csv(result_dir / (args.result_prefix + "_samples.csv"))
    samples = samples[samples["noise_dir"] == args.noise_dir].copy()
    if samples.empty:
        raise ValueError("no %s test samples found" % args.noise_dir)

    # Map each test prediction back to the nearest selected TEST physical state
    # only for adding the known clean g-separation score to the prediction table.
    test_states = selected[selected["split"] == "test"].copy().reset_index(drop=True)
    state_a = test_states["a1"].to_numpy(float)
    state_g = test_states["gamma"].to_numpy(float)
    scores = test_states["snr_sep_rms"].to_numpy(float)
    a_span = max(float(state_a.max() - state_a.min()), 1e-12)
    lg = np.log10(state_g)
    lg_span = max(float(lg.max() - lg.min()), 1e-12)

    local_scores = []
    for a, g in zip(samples["true_a1"], samples["true_gamma"]):
        d = ((state_a - float(a)) / a_span) ** 2
        d += ((lg - np.log10(float(g))) / lg_span) ** 2
        local_scores.append(float(scores[int(np.argmin(d))]))
    samples["selected_state_nearest_rms_snr"] = local_scores
    samples.to_csv(
        output_dir / "exp26_test_samples_with_gseparation.csv",
        index=False, encoding="utf-8-sig"
    )

    # Per-test-physical-state aggregation.
    state_rows = []
    for (a, g), sdf in samples.groupby(["true_a1", "true_gamma"], sort=True):
        ferr = sdf["gamma_factor_error"].to_numpy(float)
        state_rows.append({
            "true_a1": float(a),
            "true_gamma": float(g),
            "count": int(len(sdf)),
            "nearest_selected_state_rms_snr": float(np.median(sdf["selected_state_nearest_rms_snr"])),
            "gamma_median_factor": float(np.median(ferr)),
            "gamma_within_x1p2": float(np.mean(ferr <= 1.2)),
            "median_f_relative_l2": float(np.nanmedian(sdf["f_relative_l2"])),
            "median_g_clean_relative_l2": float(np.nanmedian(sdf["g_clean_relative_l2"])),
        })
    write_csv(output_dir / "exp26_per_test_state_summary.csv", state_rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Error vs physical-state separation.
        st = pd.DataFrame(state_rows)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(st["nearest_selected_state_rms_snr"], st["gamma_median_factor"])
        ax.set_xlabel("test-state nearest global g-separation (RMS-SNR)")
        ax.set_ylabel("median gamma factor error")
        ax.set_title("Exp26 independent test states")
        fig.tight_layout()
        fig.savefig(output_dir / "gamma_error_vs_test_state_gseparation.png", dpi=165)
        plt.close(fig)

        # Parameter support plot comes from data generator; create a copy-style
        # performance plot with test states colored by error.
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(
            st["true_a1"], st["true_gamma"],
            c=st["gamma_median_factor"], s=55
        )
        ax.set_yscale("log")
        ax.set_xlabel("true a1")
        ax.set_ylabel("true gamma")
        ax.set_title("Exp26 independent test-state gamma error")
        fig.colorbar(sc, ax=ax, label="median gamma factor error")
        fig.tight_layout()
        fig.savefig(output_dir / "test_state_error_map.png", dpi=165)
        plt.close(fig)

        # Best / median / worst representative f/g curves, ranked by gamma factor.
        from mc_pool_config import DEFAULT_PHYSICS
        from mc_physics import scaled_curves_numpy
        ranked = samples.sort_values("gamma_factor_error", kind="mergesort").reset_index(drop=True)
        picks = [("best", 0), ("median", len(ranked)//2), ("worst", len(ranked)-1)]
        curve_rows = []
        physics = replace(DEFAULT_PHYSICS, q2_points=int(meta["model_input_points"]))
        s_grid = None
        try:
            from mc_physics import output_grids_numpy
            s_grid, q_grid = output_grids_numpy(physics)
        except Exception:
            s_grid = np.arange(DEFAULT_PHYSICS.output_points)
            q_grid = np.linspace(physics.q2_min, physics.q2_max, physics.q2_points)

        curve_dir = output_dir / "representative_curves"
        curve_dir.mkdir(parents=True, exist_ok=True)
        for rank, idx in picks:
            row = ranked.iloc[int(idx)]
            pt = np.array([[row["true_a1"], 0.025, 0.0, 0.8, row["true_gamma"]]], dtype=np.float64)
            pp = np.array([[row["pred_a1"], 0.025, 0.0, 0.8, row["pred_gamma"]]], dtype=np.float64)
            ft, gt = scaled_curves_numpy(pt, integration_points=args.integration_points, config=physics)
            fp, gp = scaled_curves_numpy(pp, integration_points=args.integration_points, config=physics)
            ft=ft[0];fp=fp[0];gt=gt[0];gp=gp[0]

            fig, ax = plt.subplots(figsize=(7.4, 4.5))
            ax.plot(s_grid, ft, label="true f")
            ax.plot(s_grid, fp, "--", label="predicted f")
            ax.set_xlabel("s"); ax.set_ylabel("f(s)")
            ax.set_title(
                "%s | true a1=%.5g pred=%.5g | true gamma=%.5g pred=%.5g\n"
                "gamma factor=%.4g | f relL2=%.4g"
                % (
                    rank, row["true_a1"], row["pred_a1"],
                    row["true_gamma"], row["pred_gamma"],
                    row["gamma_factor_error"], row["f_relative_l2"],
                )
            )
            ax.legend(); fig.tight_layout()
            fpath = curve_dir / ("f_%s.png" % rank)
            fig.savefig(fpath, dpi=170); plt.close(fig)

            fig, ax = plt.subplots(figsize=(7.4, 4.5))
            ax.plot(q_grid, gt, label="true g")
            ax.plot(q_grid, gp, "--", label="predicted-parameter g")
            ax.set_xlabel(r"$q^2$"); ax.set_ylabel("g")
            ax.set_title("%s independent test sample" % rank)
            ax.legend(); fig.tight_layout()
            gpath = curve_dir / ("g_%s.png" % rank)
            fig.savefig(gpath, dpi=170); plt.close(fig)
            curve_rows.append({
                "rank": rank,
                "true_a1": float(row["true_a1"]),
                "pred_a1": float(row["pred_a1"]),
                "true_gamma": float(row["true_gamma"]),
                "pred_gamma": float(row["pred_gamma"]),
                "gamma_factor_error": float(row["gamma_factor_error"]),
                "f_relative_l2": float(row["f_relative_l2"]),
                "g_clean_relative_l2": float(row["g_clean_relative_l2"]),
                "f_plot": str(fpath),
                "g_plot": str(gpath),
            })
        write_csv(output_dir / "representative_curve_samples.csv", curve_rows)
    except Exception as exc:
        print("warning: plot/curve generation skipped: %s" % exc)

    # Copy the small verification table into the analysis output.
    pair_summary.to_csv(
        output_dir / "split_pair_gseparation_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    print("=" * 100)
    print("Exp26 analysis ready")
    print("Global selected-state minimum RMS-SNR: %.6g" % meta["global_min_rms_snr"])
    print("Independent physical state counts    : %s" % meta["selected_state_counts"])
    print("Read first:")
    print("  %s" % (output_dir / "exp26_per_test_state_summary.csv"))
    print("  %s" % (output_dir / "split_pair_gseparation_summary.csv"))
    print("  %s" % (output_dir / "representative_curves"))
    print("=" * 100)


if __name__ == "__main__":
    main()
