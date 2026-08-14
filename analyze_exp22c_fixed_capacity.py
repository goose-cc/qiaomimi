#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate Exp22C fixed-capacity control results.

All Exp22C conditions use the same model input width, so the approximate
trainable parameter count must remain constant. This report compares the
physical observation count against classification, continuous gamma accuracy,
f reconstruction, and observed-point separation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_int_list(text):
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected non-empty comma-separated integers")
    return values


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate Exp22C fixed-capacity q2 control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--project-root", default=".")
    p.add_argument("--observation-points", type=parse_int_list, default=[100, 500, 1000])
    p.add_argument("--model-input-points", type=int, default=1000)
    p.add_argument(
        "--output-dir",
        default="validation_results/exp22c_fixed_capacity_comparison",
    )
    return p.parse_args()


def find_row(df, noise_level, split):
    mask = np.isclose(df["noise_level"].to_numpy(float), float(noise_level), atol=1e-10)
    mask &= df["split"].astype(str).to_numpy() == str(split)
    sub = df.loc[mask]
    if len(sub) != 1:
        raise ValueError("expected one row for noise=%s split=%s, found %d" %
                         (noise_level, split, len(sub)))
    return sub.iloc[0]


def find_class_row(df, noise_level):
    mask = np.isclose(df["noise_level"].to_numpy(float), float(noise_level), atol=1e-10)
    sub = df.loc[mask]
    if len(sub) != 1:
        raise ValueError("expected one classification row for noise=%s, found %d" %
                         (noise_level, len(sub)))
    return sub.iloc[0]


def parameter_count(input_points, hidden_sizes):
    total = 0
    prev = int(input_points)
    for width in hidden_sizes:
        width = int(width)
        total += prev * width + width
        prev = width
    # second head + gamma head, each Linear(prev, 1)
    total += 2 * (prev + 1)
    return int(total)


def save_plot(path, x, ys, labels, ylabel, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("warning: plotting skipped: %s" % exc)
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for y, label in zip(ys, labels):
        ax.plot(x, y, marker="o", label=label)
    ax.set_xlabel("physical q2 observation points")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(labels) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    root = Path(args.project_root)
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for obs in args.observation_points:
        data_dir = root / ("data_exp22c_qobs%d_fixed%d" % (obs, args.model_input_points))
        result_dir = root / ("validation_results/exp22c_qobs%d_fixed%d" %
                             (obs, args.model_input_points))
        prefix = "exp22c_qobs%d" % obs

        meta_path = data_dir / "metadata.json"
        train_meta_path = result_dir / (prefix + "_metadata.json")
        summary_path = result_dir / (prefix + "_summary.csv")
        class_path = result_dir / (prefix + "_classification_summary.csv")

        for path in (meta_path, train_meta_path, summary_path, class_path):
            if not path.exists():
                raise FileNotFoundError(path)

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        train_meta = json.loads(train_meta_path.read_text(encoding="utf-8"))
        summary = pd.read_csv(summary_path)
        classes = pd.read_csv(class_path)

        r = find_row(summary, 0.002, "test_seen")
        c = find_class_row(classes, 0.002)

        hidden = train_meta.get("training_args", {}).get("hidden_sizes", [256, 256, 128])
        if isinstance(hidden, str):
            hidden = [int(x.strip()) for x in hidden.strip("()[]").split(",") if x.strip()]
        model_input = int(meta["fixed_model_input_points"])
        approx_params = parameter_count(model_input, hidden)

        sep = meta["selected_separation"]
        rows.append({
            "physical_observation_points": int(meta["physical_observation_points"]),
            "model_input_points": model_input,
            "training_state_count": int(meta["training_state_count"]),
            "min_rms_snr_observed": float(sep["min_snr_rms"]),
            "median_nearest_rms_snr_observed": float(sep["median_nearest_snr_rms"]),
            "min_full_snr_observed": float(sep["min_snr"]),
            "median_nearest_full_snr_observed": float(sep["median_nearest_snr"]),
            "gamma_class_accuracy": float(c["gamma_anchor_accuracy"]),
            "a1_class_accuracy": float(c["a1_anchor_accuracy"]),
            "joint_state_accuracy": float(c["joint_state_accuracy"]),
            "gamma_median_factor": float(r["gamma_median_factor"]),
            "gamma_within_x1p2": float(r["gamma_within_x1p2"]),
            "a1_median_norm_error": float(r["second_median_norm_error"]),
            "median_f_relative_l2": float(r["median_f_relative_l2"]),
            "median_g_clean_relative_l2": float(r["median_g_clean_relative_l2"]),
            "approx_trainable_parameters": approx_params,
            "best_epoch": int(train_meta.get("best_epoch", -1)),
            "data_dir": str(data_dir),
            "result_dir": str(result_dir),
        })

    rows = sorted(rows, key=lambda r: r["physical_observation_points"])
    csv_path = out / "exp22c_fixed_capacity_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    x = np.asarray([r["physical_observation_points"] for r in rows], dtype=float)
    save_plot(
        out / "classification_accuracy_vs_observations.png",
        x,
        [
            [r["gamma_class_accuracy"] for r in rows],
            [r["joint_state_accuracy"] for r in rows],
        ],
        ["gamma_class_accuracy", "joint_state_accuracy"],
        "accuracy",
        "Exp22C fixed-capacity classification vs physical observations",
    )
    save_plot(
        out / "gamma_within_x1p2_vs_observations.png",
        x,
        [[r["gamma_within_x1p2"] for r in rows]],
        ["gamma_within_x1p2"],
        "fraction",
        "Exp22C continuous gamma within x1.2",
    )
    save_plot(
        out / "f_relative_l2_vs_observations.png",
        x,
        [[r["median_f_relative_l2"] for r in rows]],
        ["median_f_relative_l2"],
        "median f relative L2",
        "Exp22C f reconstruction vs physical observations",
    )
    save_plot(
        out / "separation_snr_vs_observations.png",
        x,
        [
            [r["min_rms_snr_observed"] for r in rows],
            [r["min_full_snr_observed"] for r in rows],
        ],
        ["min_rms_snr_observed", "min_full_snr_observed"],
        "SNR",
        "Exp22C observed-point separation vs physical observations",
    )
    save_plot(
        out / "model_parameter_count_vs_observations.png",
        x,
        [[r["approx_trainable_parameters"] for r in rows]],
        ["approx_trainable_parameters"],
        "trainable parameters",
        "Exp22C network capacity check (must be constant)",
    )

    param_counts = {int(r["approx_trainable_parameters"]) for r in rows}
    input_widths = {int(r["model_input_points"]) for r in rows}
    if len(param_counts) != 1 or len(input_widths) != 1:
        raise RuntimeError(
            "fixed-capacity control failed: parameter counts/input widths differ: %r / %r"
            % (param_counts, input_widths)
        )

    print("=" * 100)
    print("Exp22C fixed-capacity comparison")
    print("model input points        : %d for ALL conditions" % rows[0]["model_input_points"])
    print("trainable parameters      : %d for ALL conditions" %
          rows[0]["approx_trainable_parameters"])
    for r in rows:
        print(
            "Nobs=%4d | gamma class=%.3f | joint=%.3f | gamma<=x1.2=%.3f | "
            "gamma med=%.4f | f med L2=%.5f | min full-SNR=%.3f"
            % (
                r["physical_observation_points"],
                r["gamma_class_accuracy"],
                r["joint_state_accuracy"],
                r["gamma_within_x1p2"],
                r["gamma_median_factor"],
                r["median_f_relative_l2"],
                r["min_full_snr_observed"],
            )
        )
    print("summary: %s" % csv_path)
    print("=" * 100)


if __name__ == "__main__":
    main()
