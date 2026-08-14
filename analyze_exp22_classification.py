#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp22A post-training classification-style diagnostics.

The network is still trained as a continuous regressor.  For the "test_seen"
split only, this script additionally snaps each continuous prediction to the
nearest training anchor and reports classification accuracies:

- gamma anchor accuracy
- a1 anchor accuracy
- joint (a1, gamma) state accuracy

This does not change training.  It only makes the teacher's "classification /
one-to-one mapping" interpretation explicit.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze Exp22A test_seen predictions as discrete classes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--result-dir", required=True)
    p.add_argument("--result-prefix", default="exp22a")
    return p.parse_args()


def nearest_linear(values: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    anchors = np.asarray(anchors, dtype=np.float64).reshape(1, -1)
    return np.argmin(np.abs(values - anchors), axis=1)


def nearest_log(values: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-30, None).reshape(-1, 1)
    anchors = np.clip(np.asarray(anchors, dtype=np.float64), 1e-30, None).reshape(1, -1)
    return np.argmin(
        np.abs(np.log10(values) - np.log10(anchors)),
        axis=1,
    )


def factor_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = np.clip(np.asarray(pred, dtype=np.float64), 1e-30, None)
    true = np.clip(np.asarray(true, dtype=np.float64), 1e-30, None)
    return np.maximum(pred / true, true / pred)


def confusion(true_idx: np.ndarray, pred_idx: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(out, (true_idx, pred_idx), 1)
    return out


def save_confusion_plot(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: confusion plot skipped: {exc}")
        return

    denom = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    norm = matrix / denom

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(norm, vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, label="fraction within true class")
    ax.set_xlabel("predicted anchor")
    ax.set_ylabel("true anchor")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    if len(labels) <= 11:
        for i in range(len(labels)):
            for j in range(len(labels)):
                if matrix[i, j] > 0:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = data_dir / "metadata.json"
    sample_path = result_dir / f"{args.result_prefix}_samples.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    a1_anchors = np.asarray(metadata["second_train_values"], dtype=np.float64)
    gamma_anchors = np.asarray(metadata["gamma_train_values"], dtype=np.float64)

    samples = pd.read_csv(sample_path)
    seen = samples[samples["split"] == "test_seen"].copy()
    if seen.empty:
        raise ValueError("No test_seen rows found")

    rows: list[dict] = []
    output_dir = result_dir / "classification_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    for noise_dir, group in seen.groupby("noise_dir", sort=True):
        true_a1 = group["true_a1"].to_numpy(np.float64)
        pred_a1 = group["pred_a1"].to_numpy(np.float64)
        true_gamma = group["true_gamma"].to_numpy(np.float64)
        pred_gamma = group["pred_gamma"].to_numpy(np.float64)

        true_a1_idx = nearest_linear(true_a1, a1_anchors)
        pred_a1_idx = nearest_linear(pred_a1, a1_anchors)
        true_gamma_idx = nearest_log(true_gamma, gamma_anchors)
        pred_gamma_idx = nearest_log(pred_gamma, gamma_anchors)

        a1_ok = pred_a1_idx == true_a1_idx
        gamma_ok = pred_gamma_idx == true_gamma_idx
        joint_ok = a1_ok & gamma_ok
        ferr = factor_error(pred_gamma, true_gamma)

        rows.append(
            {
                "noise_dir": noise_dir,
                "noise_level": float(group["noise_level"].iloc[0]),
                "count": int(len(group)),
                "a1_anchor_accuracy": float(np.mean(a1_ok)),
                "gamma_anchor_accuracy": float(np.mean(gamma_ok)),
                "joint_state_accuracy": float(np.mean(joint_ok)),
                "gamma_median_factor": float(np.median(ferr)),
                "gamma_within_x1p2": float(np.mean(ferr <= 1.2)),
                "a1_anchor_count": int(len(a1_anchors)),
                "gamma_anchor_count": int(len(gamma_anchors)),
                "state_count": int(len(a1_anchors) * len(gamma_anchors)),
            }
        )

        gamma_matrix = confusion(true_gamma_idx, pred_gamma_idx, len(gamma_anchors))
        a1_matrix = confusion(true_a1_idx, pred_a1_idx, len(a1_anchors))

        gamma_labels = [f"{x:.4g}" for x in gamma_anchors]
        a1_labels = [f"{x:.4g}" for x in a1_anchors]
        save_confusion_plot(
            gamma_matrix,
            gamma_labels,
            f"Exp22A gamma-anchor confusion | {noise_dir}",
            output_dir / f"gamma_confusion_{noise_dir}.png",
        )
        save_confusion_plot(
            a1_matrix,
            a1_labels,
            f"Exp22A a1-anchor confusion | {noise_dir}",
            output_dir / f"a1_confusion_{noise_dir}.png",
        )

    summary_path = result_dir / f"{args.result_prefix}_classification_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 96)
    print("Exp22A classification-style diagnostics")
    print(f"a1 anchors    : {a1_anchors}")
    print(f"gamma anchors : {gamma_anchors}")
    print(f"summary       : {summary_path}")
    for row in rows:
        print(
            f"{row['noise_dir']:18s} | "
            f"gamma class={row['gamma_anchor_accuracy']:.3f} | "
            f"a1 class={row['a1_anchor_accuracy']:.3f} | "
            f"joint state={row['joint_state_accuracy']:.3f} | "
            f"gamma <=x1.2={row['gamma_within_x1p2']:.3f}"
        )
    print("=" * 96)


if __name__ == "__main__":
    main()
