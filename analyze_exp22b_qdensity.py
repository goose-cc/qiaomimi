#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate Exp22A (Nq=100) and Exp22B (denser Nq) into one comparison.

Primary question
----------------
With the SAME 27 a1-gamma classes, does recomputing g(q^2) at more observation
points improve:
  * class accuracy,
  * continuous gamma accuracy,
  * f reconstruction,
while RMS-SNR stays roughly unchanged and full-curve SNR accumulates?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_int_list(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated q2 point counts")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate Exp22A/Exp22B q2-density comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--project-root", default=".")
    p.add_argument("--q2-points", type=parse_int_list, default=[100, 500, 1000])
    p.add_argument("--output-dir", default="validation_results/exp22b_qdensity_comparison")
    p.add_argument("--train-noise-dir", default="noise_0p2pct")
    return p.parse_args()


def locations(root: Path, q: int) -> tuple[Path, Path, str]:
    if q == 100:
        return (
            root / "data_exp22a_rms_snr2",
            root / "validation_results" / "exp22a_rms_snr2",
            "exp22a",
        )
    return (
        root / f"data_exp22b_q{q}",
        root / "validation_results" / f"exp22b_q{q}",
        f"exp22b_q{q}",
    )


def get_sep(data_dir: Path, metadata: dict) -> dict:
    sep_path = data_dir / "sampling_separation_summary.json"
    if sep_path.exists():
        return json.loads(sep_path.read_text(encoding="utf-8"))
    sep = metadata.get("selected_separation")
    if sep is None:
        raise KeyError(f"{data_dir}/metadata.json has no selected_separation")
    return sep


def trainable_parameter_count(input_points: int, hidden=(256, 256, 128)) -> int:
    widths = [int(input_points), *map(int, hidden)]
    count = 0
    for i in range(len(widths) - 1):
        count += widths[i] * widths[i + 1] + widths[i + 1]  # weight + bias
    count += hidden[-1] * 1 + 1  # a1 head
    count += hidden[-1] * 1 + 1  # gamma head
    return int(count)


def load_row(root: Path, q: int, train_noise_dir: str) -> dict:
    data_dir, result_dir, prefix = locations(root, q)
    meta_path = data_dir / "metadata.json"
    summary_path = result_dir / f"{prefix}_summary.csv"
    class_path = result_dir / f"{prefix}_classification_summary.csv"

    for path in (meta_path, summary_path, class_path):
        if not path.exists():
            raise FileNotFoundError(path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sep = get_sep(data_dir, meta)
    summary = pd.read_csv(summary_path)
    class_df = pd.read_csv(class_path)

    srow = summary[
        (summary["noise_dir"] == train_noise_dir)
        & (summary["split"] == "test_seen")
    ]
    crow = class_df[class_df["noise_dir"] == train_noise_dir]
    if len(srow) != 1:
        raise ValueError(f"{summary_path}: expected one {train_noise_dir}/test_seen row")
    if len(crow) != 1:
        raise ValueError(f"{class_path}: expected one {train_noise_dir} row")

    s = srow.iloc[0]
    c = crow.iloc[0]
    q2_range = meta.get("q2_range", [-100.0, -6.0])
    q_span = abs(float(q2_range[1]) - float(q2_range[0]))
    q_step = q_span / max(int(q) - 1, 1)

    return {
        "q2_points": int(q),
        "q2_step": float(q_step),
        "training_state_count": int(meta.get("training_state_count", c["state_count"])),
        "min_rms_snr": float(sep["min_snr_rms"]),
        "median_nearest_rms_snr": float(sep["median_nearest_snr_rms"]),
        "min_full_snr": float(sep["min_snr"]),
        "median_nearest_full_snr": float(sep["median_nearest_snr"]),
        "gamma_class_accuracy": float(c["gamma_anchor_accuracy"]),
        "a1_class_accuracy": float(c["a1_anchor_accuracy"]),
        "joint_state_accuracy": float(c["joint_state_accuracy"]),
        "gamma_median_factor": float(s["gamma_median_factor"]),
        "gamma_within_x1p2": float(s["gamma_within_x1p2"]),
        "a1_median_norm_error": float(s["second_median_norm_error"]),
        "median_f_relative_l2": float(s["median_f_relative_l2"]),
        "median_g_clean_relative_l2": float(s["median_g_clean_relative_l2"]),
        "approx_trainable_parameters": trainable_parameter_count(int(q)),
        "data_dir": str(data_dir),
        "result_dir": str(result_dir),
    }


def plot_line(df: pd.DataFrame, columns: list[str], ylabel: str, title: str, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: plot skipped: {exc}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for col in columns:
        ax.plot(df["q2_points"], df[col], marker="o", label=col)
    ax.set_xlabel("q2 observation points")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(columns) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.project_root)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [load_row(root, q, args.train_noise_dir) for q in args.q2_points]
    df = pd.DataFrame(rows).sort_values("q2_points")
    csv_path = output_dir / "exp22b_qdensity_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_line(
        df,
        ["gamma_class_accuracy", "joint_state_accuracy"],
        "accuracy",
        "Exp22B classification accuracy vs q2 sampling density",
        output_dir / "classification_accuracy_vs_q2.png",
    )
    plot_line(
        df,
        ["gamma_within_x1p2"],
        "fraction",
        "Exp22B continuous gamma within x1.2 vs q2 sampling density",
        output_dir / "gamma_within_x1p2_vs_q2.png",
    )
    plot_line(
        df,
        ["min_rms_snr", "min_full_snr"],
        "SNR",
        "Exp22B nearest-state separation vs q2 sampling density",
        output_dir / "separation_snr_vs_q2.png",
    )
    plot_line(
        df,
        ["median_f_relative_l2"],
        "median f relative L2",
        "Exp22B f reconstruction vs q2 sampling density",
        output_dir / "f_relative_l2_vs_q2.png",
    )

    print("=" * 100)
    print("Exp22B q2-density comparison")
    print(df.to_string(index=False))
    print()
    print(f"saved: {csv_path}")
    print("Interpretation guardrail:")
    print("  - same physical parameter classes are used at every Nq")
    print("  - RMS-SNR is the pointwise-normalized separation")
    print("  - full-SNR can rise with sqrt(Nq) when more noisy observations add evidence")
    print("  - the MLP first layer also grows with Nq; approx_trainable_parameters is reported")
    print("=" * 100)


if __name__ == "__main__":
    main()
