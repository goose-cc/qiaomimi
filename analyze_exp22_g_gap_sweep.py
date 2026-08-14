#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate the completed Exp22 g-separation sweep.

This script intentionally reports continuous gamma-regression quality, not
classification-style snapping.  The main question is:

    If physically generated g curves are required to be farther apart,
    does gamma recovery improve, and how much parameter-grid resolution is
    lost to achieve that separation?

Expected directory layout is produced by run_exp22_complete_g_gap_sweep.ps1.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return vals


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate Exp22 physical g-gap sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gap-percents", type=parse_float_list, default=[0.4, 0.6, 1.0])
    p.add_argument("--data-root", default=".")
    p.add_argument("--result-root", default="validation_results")
    p.add_argument("--output-dir", default="validation_results/exp22_g_gap_sweep")
    p.add_argument("--result-prefix", default="exp22a")
    return p.parse_args()


def get_metric(df: pd.DataFrame, split: str, noise: float, column: str) -> float:
    rows = df[(df["split"] == split) & np.isclose(df["noise_level"], noise, atol=1e-7, rtol=1e-6)]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0][column])


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty csv: {path}")
    fields: list[str] = []
    seen: set[str] = set()
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


def save_plots(output_dir: Path, compact: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: plotting skipped: {exc}")
        return

    x = np.asarray([r["target_g_gap_percent"] for r in compact], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(x, [100 * r["seen_0pct_gamma_within_x1p2"] for r in compact], marker="o", label="seen, 0% noise")
    ax.plot(x, [100 * r["seen_0p2pct_gamma_within_x1p2"] for r in compact], marker="o", label="seen, 0.2% noise")
    ax.plot(x, [100 * r["interp_both_0pct_gamma_within_x1p2"] for r in compact], marker="o", label="unseen midpoints, 0% noise")
    ax.plot(x, [100 * r["interp_both_0p2pct_gamma_within_x1p2"] for r in compact], marker="o", label="unseen midpoints, 0.2% noise")
    ax.set_xlabel("minimum physical RMS gap between g curves (%)")
    ax.set_ylabel("samples with predicted gamma within factor 1.2 (%)")
    ax.set_title("Exp22: g separation vs gamma recovery")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "g_gap_vs_gamma_recovery.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(x, [r["training_state_count"] for r in compact], marker="o", label="training states")
    ax.plot(x, [r["a1_anchor_count"] for r in compact], marker="o", label="a1 anchors")
    ax.plot(x, [r["gamma_anchor_count"] for r in compact], marker="o", label="gamma anchors")
    ax.set_xlabel("minimum physical RMS gap between g curves (%)")
    ax.set_ylabel("count")
    ax.set_title("Exp22: g separation vs retained parameter resolution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "g_gap_vs_retained_resolution.png", dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compact: list[dict] = []
    long_rows: list[dict] = []

    for gap in sorted(args.gap_percents):
        t = tag(gap)
        data_dir = data_root / f"data_exp22a_g_gap_{t}pct"
        result_dir = result_root / f"exp22a_g_gap_{t}pct"
        metadata_path = data_dir / "metadata.json"
        summary_path = result_dir / f"{args.result_prefix}_summary.csv"

        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summary = pd.read_csv(summary_path)
        selected = metadata["selected_separation"]
        a1_values = metadata["second_train_values"]
        gamma_values = metadata["gamma_train_values"]

        for _, row in summary.iterrows():
            long_rows.append(
                {
                    "target_g_gap_percent": float(gap),
                    "actual_min_g_gap_percent": float(selected["min_rms_g_gap_percent"]),
                    "actual_min_rms_snr": float(selected["min_snr_rms"]),
                    "a1_anchor_count": len(a1_values),
                    "gamma_anchor_count": len(gamma_values),
                    "training_state_count": int(metadata["training_state_count"]),
                    **row.to_dict(),
                }
            )

        compact.append(
            {
                "target_g_gap_percent": float(gap),
                "actual_min_g_gap_percent": float(selected["min_rms_g_gap_percent"]),
                "actual_min_rms_snr": float(selected["min_snr_rms"]),
                "reference_noise_percent": 100.0 * float(metadata["reference_noise"]),
                "a1_anchor_count": len(a1_values),
                "gamma_anchor_count": len(gamma_values),
                "training_state_count": int(metadata["training_state_count"]),
                "a1_values": ";".join(f"{x:.8g}" for x in a1_values),
                "gamma_values": ";".join(f"{x:.8g}" for x in gamma_values),
                "seen_0pct_gamma_within_x1p2": get_metric(summary, "test_seen", 0.0, "gamma_within_x1p2"),
                "seen_0p2pct_gamma_within_x1p2": get_metric(summary, "test_seen", 0.002, "gamma_within_x1p2"),
                "interp_both_0pct_gamma_within_x1p2": get_metric(summary, "test_interp_both", 0.0, "gamma_within_x1p2"),
                "interp_both_0p2pct_gamma_within_x1p2": get_metric(summary, "test_interp_both", 0.002, "gamma_within_x1p2"),
                "seen_0pct_gamma_median_factor": get_metric(summary, "test_seen", 0.0, "gamma_median_factor"),
                "seen_0p2pct_gamma_median_factor": get_metric(summary, "test_seen", 0.002, "gamma_median_factor"),
                "interp_both_0pct_gamma_median_factor": get_metric(summary, "test_interp_both", 0.0, "gamma_median_factor"),
                "interp_both_0p2pct_gamma_median_factor": get_metric(summary, "test_interp_both", 0.002, "gamma_median_factor"),
            }
        )

    write_csv(output_dir / "exp22_g_gap_sweep_all_metrics.csv", long_rows)
    write_csv(output_dir / "exp22_g_gap_sweep_teacher_summary.csv", compact)
    save_plots(output_dir, compact)

    print("=" * 110)
    print("Exp22 g-gap sweep summary")
    print("A percentage below means: predicted gamma differs from true gamma by no more than a factor of 1.2.")
    print("No classification-style snapping is used here.")
    print("-" * 110)
    print(
        f"{'g gap':>8s} | {'actual':>8s} | {'states':>6s} | {'a1':>3s} | {'gamma':>5s} | "
        f"{'seen 0%':>8s} | {'seen .2%':>9s} | {'interp 0%':>9s} | {'interp .2%':>10s}"
    )
    for r in compact:
        def pct(v: float) -> str:
            return "nan" if not np.isfinite(v) else f"{100*v:7.1f}%"
        print(
            f"{r['target_g_gap_percent']:7.3g}% | {r['actual_min_g_gap_percent']:7.3g}% | "
            f"{r['training_state_count']:6d} | {r['a1_anchor_count']:3d} | {r['gamma_anchor_count']:5d} | "
            f"{pct(r['seen_0pct_gamma_within_x1p2']):>8s} | "
            f"{pct(r['seen_0p2pct_gamma_within_x1p2']):>9s} | "
            f"{pct(r['interp_both_0pct_gamma_within_x1p2']):>9s} | "
            f"{pct(r['interp_both_0p2pct_gamma_within_x1p2']):>10s}"
        )
    print("-" * 110)
    print(f"Read first: {output_dir / 'exp22_g_gap_sweep_teacher_summary.csv'}")
    print(f"Plot      : {output_dir / 'g_gap_vs_gamma_recovery.png'}")
    print(f"Trade-off : {output_dir / 'g_gap_vs_retained_resolution.png'}")
    print("=" * 110)


if __name__ == "__main__":
    main()
