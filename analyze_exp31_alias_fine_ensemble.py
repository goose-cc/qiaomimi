#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp31: fine alias-cutoff scan, multi-seed confirmation, and seed ensemble.

Purpose
-------
Exp30 found a strong improvement around continuous remote-alias cutoffs
0.25--0.50, with alias=0.25 giving about 90% gamma-within-x1.2 while keeping
good physical support.  Exp31 does NOT change the physics, network or loss.

It answers two remaining questions:
  1) Is the best cutoff actually around 0.20/0.25/0.30 rather than exactly 0.25?
  2) Are the remaining ~5--10% failures mostly training-seed instability?

For each cutoff we aggregate multiple independent training seeds.  When two or
more seeds exist for the exact same dataset, we also form an evaluation-only
ensemble by taking:
    a1_hat     = median(seed predictions)
    gamma_hat  = exp(median(log gamma_hat_seed))

The ensemble does not alter the training data or forward physics.  It only
reduces model-initialization variance.

The main stage-gate metric is physical-state gamma-within-x1.2, not only noisy
sample accuracy.
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
        description="Analyze Exp31 alias fine scan and seed ensemble",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--noise-dir", default="noise_0p2pct")
    p.add_argument("--target-within-x1p2", type=float, default=0.95)
    p.add_argument("--persistent-fail-factor", type=float, default=1.2)
    return p.parse_args()


def factor_error(pred, true):
    pred = np.clip(np.asarray(pred, dtype=float), 1e-30, None)
    true = np.clip(np.asarray(true, dtype=float), 1e-30, None)
    return np.maximum(pred / true, true / pred)


def state_metrics_from_samples(samples):
    rows = []
    for (a1, gamma), g in samples.groupby(["true_a1", "true_gamma"], sort=False):
        pg = float(np.median(g["pred_gamma"].to_numpy(float)))
        pa = float(np.median(g["pred_a1"].to_numpy(float)))
        gf = float(max(pg / float(gamma), float(gamma) / pg))
        rows.append({
            "true_a1": float(a1),
            "true_gamma": float(gamma),
            "median_pred_a1": pa,
            "median_pred_gamma": pg,
            "gamma_factor_error": gf,
            "noise_realization_count": int(len(g)),
        })
    sdf = pd.DataFrame(rows)
    return {
        "state_count": int(len(sdf)),
        "state_gamma_median_factor": float(np.median(sdf["gamma_factor_error"])),
        "state_gamma_p90_factor": float(np.quantile(sdf["gamma_factor_error"], 0.90)),
        "state_gamma_max_factor": float(np.max(sdf["gamma_factor_error"])),
        "state_gamma_within_x1p2": float(np.mean(sdf["gamma_factor_error"] <= 1.2)),
        "state_gamma_within_x1p5": float(np.mean(sdf["gamma_factor_error"] <= 1.5)),
        "state_gamma_within_x2": float(np.mean(sdf["gamma_factor_error"] <= 2.0)),
    }, sdf


def load_run(row, noise_dir):
    result_dir = Path(str(row["result_dir"]))
    prefix = str(row["result_prefix"])

    summary_path = result_dir / (prefix + "_summary.csv")
    samples_path = result_dir / (prefix + "_samples.csv")
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)

    summary = pd.read_csv(summary_path)
    samples = pd.read_csv(samples_path)

    sr = summary[summary["noise_dir"] == noise_dir]
    if sr.empty:
        raise ValueError("%s has no %s row" % (summary_path, noise_dir))
    sr = sr.iloc[0]

    ss = samples[samples["noise_dir"] == noise_dir].copy().reset_index(drop=True)
    if ss.empty:
        raise ValueError("%s has no %s samples" % (samples_path, noise_dir))

    sm, state_df = state_metrics_from_samples(ss)
    gf = ss["gamma_factor_error"].to_numpy(float)

    metrics = {
        "alias_min_rms_snr": float(row["alias_min_rms_snr"]),
        "train_seed": int(row["seed"]),
        "sample_count": int(len(ss)),
        "gamma_median_factor": float(sr["gamma_median_factor"]),
        "gamma_p90_factor": float(sr["gamma_p90_factor"]),
        "gamma_p99_factor": float(np.quantile(gf, 0.99)),
        "gamma_max_factor": float(np.max(gf)),
        "gamma_within_x1p2": float(sr["gamma_within_x1p2"]),
        "gamma_within_x1p5": float(sr["gamma_within_x1p5"]),
        "gamma_within_x2": float(sr["gamma_within_x2"]),
        "median_f_relative_l2": float(sr["median_f_relative_l2"]),
        "p90_f_relative_l2": float(sr["p90_f_relative_l2"]),
        "catastrophic_gamma_factor_ge5": float(sr["catastrophic_gamma_factor_ge5"]),
        "catastrophic_gamma_factor_ge10": float(sr["catastrophic_gamma_factor_ge10"]),
        **sm,
    }
    return metrics, ss, state_df


def check_row_alignment(sample_frames):
    """Require exact same test sample ordering before seed ensemble."""
    ref = sample_frames[0][["noise_dir", "true_a1", "true_gamma"]].copy()
    for i, frame in enumerate(sample_frames[1:], start=1):
        cur = frame[["noise_dir", "true_a1", "true_gamma"]]
        if len(cur) != len(ref):
            raise ValueError("seed sample row count mismatch")
        if not np.array_equal(cur["noise_dir"].to_numpy(), ref["noise_dir"].to_numpy()):
            raise ValueError("seed noise_dir ordering mismatch")
        if not np.allclose(cur["true_a1"].to_numpy(float), ref["true_a1"].to_numpy(float),
                           rtol=0, atol=1e-12):
            raise ValueError("seed true_a1 ordering mismatch")
        if not np.allclose(cur["true_gamma"].to_numpy(float), ref["true_gamma"].to_numpy(float),
                           rtol=0, atol=1e-12):
            raise ValueError("seed true_gamma ordering mismatch")


def ensemble_samples(sample_frames):
    check_row_alignment(sample_frames)
    ref = sample_frames[0].copy()
    a = np.stack([f["pred_a1"].to_numpy(float) for f in sample_frames], axis=0)
    g = np.stack([np.log(np.clip(f["pred_gamma"].to_numpy(float), 1e-30, None))
                  for f in sample_frames], axis=0)

    out = ref[["noise_dir", "true_a1", "true_gamma"]].copy()
    out["pred_a1"] = np.median(a, axis=0)
    out["pred_gamma"] = np.exp(np.median(g, axis=0))
    out["gamma_factor_error"] = factor_error(
        out["pred_gamma"].to_numpy(float),
        out["true_gamma"].to_numpy(float),
    )
    return out


def ensemble_metrics(es):
    gf = es["gamma_factor_error"].to_numpy(float)
    sm, state_df = state_metrics_from_samples(es)
    return {
        "ensemble_seed_count": np.nan,  # filled by caller
        "ensemble_gamma_median_factor": float(np.median(gf)),
        "ensemble_gamma_p90_factor": float(np.quantile(gf, 0.90)),
        "ensemble_gamma_p99_factor": float(np.quantile(gf, 0.99)),
        "ensemble_gamma_max_factor": float(np.max(gf)),
        "ensemble_gamma_within_x1p2": float(np.mean(gf <= 1.2)),
        "ensemble_gamma_within_x1p5": float(np.mean(gf <= 1.5)),
        "ensemble_gamma_within_x2": float(np.mean(gf <= 2.0)),
        "ensemble_catastrophic_factor_ge5": float(np.mean(gf >= 5.0)),
        "ensemble_state_count": int(sm["state_count"]),
        "ensemble_state_gamma_median_factor": float(sm["state_gamma_median_factor"]),
        "ensemble_state_gamma_p90_factor": float(sm["state_gamma_p90_factor"]),
        "ensemble_state_gamma_max_factor": float(sm["state_gamma_max_factor"]),
        "ensemble_state_gamma_within_x1p2": float(sm["state_gamma_within_x1p2"]),
        "ensemble_state_gamma_within_x1p5": float(sm["state_gamma_within_x1p5"]),
        "ensemble_state_gamma_within_x2": float(sm["state_gamma_within_x2"]),
    }, state_df


def write_csv(path, rows):
    if not rows:
        return
    fields, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    required = {"alias_min_rms_snr", "seed", "data_dir", "result_dir", "result_prefix"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError("manifest missing columns: %s" % sorted(missing))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_run_rows = []
    run_samples = {}
    run_states = {}

    for _, row in manifest.sort_values(["alias_min_rms_snr", "seed"]).iterrows():
        metrics, samples, state_df = load_run(row, args.noise_dir)
        per_run_rows.append(metrics)
        key = (float(row["alias_min_rms_snr"]), int(row["seed"]))
        run_samples[key] = samples
        run_states[key] = state_df

    per_run = pd.DataFrame(per_run_rows)
    per_run.to_csv(out / "exp31_per_run_metrics.csv", index=False, encoding="utf-8-sig")

    aggregate_rows = []
    ensemble_rows = []
    persistent_rows = []

    numeric_cols = [
        c for c in per_run.columns
        if c not in ("alias_min_rms_snr", "train_seed")
    ]

    for alias, gdf in per_run.groupby("alias_min_rms_snr", sort=True):
        row = {
            "alias_min_rms_snr": float(alias),
            "seed_count": int(len(gdf)),
        }
        for col in numeric_cols:
            row[col + "_mean"] = float(gdf[col].mean())
            row[col + "_std"] = float(gdf[col].std(ddof=0))

        seeds = sorted(int(x) for x in gdf["train_seed"].tolist())
        frames = [run_samples[(float(alias), s)] for s in seeds]

        if len(frames) >= 2:
            es = ensemble_samples(frames)
            em, est = ensemble_metrics(es)
            em["ensemble_seed_count"] = len(frames)
            row.update(em)

            est2 = est.copy()
            est2["alias_min_rms_snr"] = float(alias)
            est2["seed_count"] = len(frames)
            est2.to_csv(
                out / ("exp31_ensemble_state_metrics_alias_%s.csv"
                       % str(alias).replace(".", "p")),
                index=False,
                encoding="utf-8-sig",
            )

            # Persistent failure means the SAME physical truth fails in at least
            # half of independent seeds. This separates data/physics difficulty
            # from initialization noise.
            keys = None
            state_maps = []
            for s in seeds:
                sdf = run_states[(float(alias), s)].copy()
                sdf["key"] = list(zip(sdf["true_a1"], sdf["true_gamma"]))
                mp = {
                    k: float(v)
                    for k, v in zip(sdf["key"], sdf["gamma_factor_error"])
                }
                state_maps.append(mp)
                keys = set(mp) if keys is None else keys & set(mp)

            for key in sorted(keys):
                vals = np.asarray([mp[key] for mp in state_maps], dtype=float)
                fail_count = int(np.sum(vals > float(args.persistent_fail_factor)))
                if fail_count >= int(math.ceil(len(seeds) / 2.0)):
                    persistent_rows.append({
                        "alias_min_rms_snr": float(alias),
                        "true_a1": float(key[0]),
                        "true_gamma": float(key[1]),
                        "seed_count": len(seeds),
                        "fail_seed_count": fail_count,
                        "fail_seed_fraction": fail_count / float(len(seeds)),
                        "median_seed_gamma_factor": float(np.median(vals)),
                        "max_seed_gamma_factor": float(np.max(vals)),
                    })

            ensemble_rows.append({
                "alias_min_rms_snr": float(alias),
                **em,
            })

        aggregate_rows.append(row)

    agg = pd.DataFrame(aggregate_rows).sort_values("alias_min_rms_snr")
    agg.to_csv(out / "exp31_alias_fine_scan_summary.csv", index=False, encoding="utf-8-sig")
    write_csv(out / "exp31_ensemble_summary.csv", ensemble_rows)
    write_csv(out / "exp31_persistent_hard_states.csv", persistent_rows)

    # Ranking favors robust physical-state accuracy first, then sample accuracy,
    # then lower tail error. If an ensemble exists, rank by ensemble result.
    ranking = []
    for _, r in agg.iterrows():
        if np.isfinite(r.get("ensemble_state_gamma_within_x1p2", np.nan)):
            state_acc = float(r["ensemble_state_gamma_within_x1p2"])
            sample_acc = float(r["ensemble_gamma_within_x1p2"])
            tail = float(r["ensemble_gamma_p90_factor"])
            source = "ensemble"
        else:
            state_acc = float(r["state_gamma_within_x1p2_mean"])
            sample_acc = float(r["gamma_within_x1p2_mean"])
            tail = float(r["gamma_p90_factor_mean"])
            source = "seed_mean"
        ranking.append({
            "alias_min_rms_snr": float(r["alias_min_rms_snr"]),
            "ranking_source": source,
            "physical_state_within_x1p2": state_acc,
            "sample_within_x1p2": sample_acc,
            "gamma_p90_factor": tail,
            "passes_95pct_state_gate": int(state_acc >= args.target_within_x1p2),
        })

    rdf = pd.DataFrame(ranking).sort_values(
        ["physical_state_within_x1p2", "sample_within_x1p2", "gamma_p90_factor"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    rdf.insert(0, "rank", np.arange(1, len(rdf) + 1))
    rdf.to_csv(out / "exp31_threshold_ranking.csv", index=False, encoding="utf-8-sig")

    stage_rows = []
    for _, r in rdf.iterrows():
        stage_rows.append({
            "alias_min_rms_snr": float(r["alias_min_rms_snr"]),
            "state_target": float(args.target_within_x1p2),
            "state_actual": float(r["physical_state_within_x1p2"]),
            "state_pass": int(r["physical_state_within_x1p2"] >= args.target_within_x1p2),
            "sample_target": float(args.target_within_x1p2),
            "sample_actual": float(r["sample_within_x1p2"]),
            "sample_pass": int(r["sample_within_x1p2"] >= args.target_within_x1p2),
        })
    write_csv(out / "exp31_stage_gate.csv", stage_rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            agg["alias_min_rms_snr"],
            agg["state_gamma_within_x1p2_mean"],
            yerr=agg["state_gamma_within_x1p2_std"],
            marker="o",
            capsize=3,
            label="single-seed physical-state mean",
        )
        if "ensemble_state_gamma_within_x1p2" in agg.columns:
            m = np.isfinite(agg["ensemble_state_gamma_within_x1p2"])
            ax.plot(
                agg.loc[m, "alias_min_rms_snr"],
                agg.loc[m, "ensemble_state_gamma_within_x1p2"],
                marker="s",
                label="seed ensemble",
            )
        ax.axhline(args.target_within_x1p2, linestyle="--", label="95% target")
        ax.set_xlabel("minimum continuous remote-alias RMS-SNR")
        ax.set_ylabel("physical-state gamma within x1.2")
        ax.set_title("Exp31 fine scan + multi-seed stability")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "state_accuracy_fine_scan_ensemble.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            agg["alias_min_rms_snr"],
            agg["gamma_within_x1p2_mean"],
            yerr=agg["gamma_within_x1p2_std"],
            marker="o",
            capsize=3,
            label="single-seed sample mean",
        )
        if "ensemble_gamma_within_x1p2" in agg.columns:
            m = np.isfinite(agg["ensemble_gamma_within_x1p2"])
            ax.plot(
                agg.loc[m, "alias_min_rms_snr"],
                agg.loc[m, "ensemble_gamma_within_x1p2"],
                marker="s",
                label="seed ensemble",
            )
        ax.axhline(args.target_within_x1p2, linestyle="--", label="95% target")
        ax.set_xlabel("minimum continuous remote-alias RMS-SNR")
        ax.set_ylabel("noisy-sample gamma within x1.2")
        ax.set_title("Exp31 sample accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "sample_accuracy_fine_scan_ensemble.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            agg["alias_min_rms_snr"],
            agg["median_f_relative_l2_mean"],
            yerr=agg["median_f_relative_l2_std"],
            marker="o",
            capsize=3,
            label="median f relL2",
        )
        ax.errorbar(
            agg["alias_min_rms_snr"],
            agg["p90_f_relative_l2_mean"],
            yerr=agg["p90_f_relative_l2_std"],
            marker="o",
            capsize=3,
            label="p90 f relL2",
        )
        ax.set_xlabel("minimum continuous remote-alias RMS-SNR")
        ax.set_ylabel("f relative L2")
        ax.set_title("Exp31 f reconstruction across alias cutoffs")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "f_error_fine_scan_multiseed.png", dpi=170)
        plt.close(fig)
    except Exception as exc:
        print("warning: plotting skipped: %s" % exc)

    print("=" * 100)
    print("Exp31 analysis ready")
    print("Read first:")
    print("  %s" % (out / "exp31_alias_fine_scan_summary.csv"))
    print("  %s" % (out / "exp31_threshold_ranking.csv"))
    print("  %s" % (out / "exp31_stage_gate.csv"))
    print("  %s" % (out / "exp31_persistent_hard_states.csv"))
    print("=" * 100)


if __name__ == "__main__":
    main()
