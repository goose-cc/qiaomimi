#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp23: narrow-gamma curriculum dataset on the Exp22C fixed-capacity setup.

Controlled design
-----------------
- Unknowns remain a1 + gamma only.
- a2, a3, m stay fixed.
- Physical q2 observations are fixed at 500.
- Model input width is fixed at 1000.
- a1 anchors are reused from Exp22A.
- The controlled change is gamma coverage: default 5 anchors
      0.01, 0.0316228, 0.1, 0.316228, 1.0
  so the previous hard interpolation points become explicit training anchors.
- Total train/val/test sample budgets stay close to Exp22A/22C instead of
  keeping samples-per-state fixed.

The generated data are compatible with train_exp20_pairwise.py and
train_exp23_narrow_peak.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from data_generate_exp20_pairwise import (
    DEFAULT_PHYSICS,
    FIXED_A2,
    FIXED_A3,
    FIXED_M,
    arithmetic_midpoints,
    geometric_midpoints,
    make_split_arrays,
    noise_dir_name,
    save_npz,
    write_csv,
)
from data_generate_exp22c_fixed_capacity import (
    canonical_indices,
    separation_diagnostics,
    slim_for_training,
    transform_to_fixed_width,
)


def parse_float_list(text):
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate Exp23 narrow-gamma curriculum dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source-metadata", required=True,
                   help="Exp22A metadata.json; only a1 anchors are reused")
    p.add_argument("--output-dir", default="data_exp23_narrow_gamma_q500")
    p.add_argument(
        "--gamma-anchors",
        type=parse_float_list,
        default=[0.01, 0.0316227766, 0.1, 0.316227766, 1.0],
    )
    p.add_argument("--observation-points", type=int, default=500)
    p.add_argument("--model-input-points", type=int, default=1000)
    p.add_argument("--reference-noise", type=float, default=0.002)
    p.add_argument("--noise-levels", type=parse_float_list, default=[0.0, 0.002, 0.01])
    p.add_argument("--train-noise-level", type=float, default=0.002)
    p.add_argument("--integration-points", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260814)

    # Fixed total budgets: when gamma anchors increase, repetitions/state decrease.
    p.add_argument("--target-train-samples", type=int, default=20000)
    p.add_argument("--target-val-samples", type=int, default=2500)
    p.add_argument("--target-test-seen-samples", type=int, default=4400)
    p.add_argument("--target-test-interp-samples", type=int, default=2200)

    lean = p.add_mutually_exclusive_group()
    lean.add_argument("--lean-train-files", dest="lean_train_files", action="store_true")
    lean.add_argument("--no-lean-train-files", dest="lean_train_files", action="store_false")
    p.set_defaults(lean_train_files=True)

    p.add_argument("--compressed", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def count_per_state(target_total, n_states):
    if n_states <= 0:
        raise ValueError("n_states must be positive")
    return max(1, int(round(float(target_total) / float(n_states))))


def plot_gamma_layout(output_dir, a1_values, gamma_values, gamma_mid):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("warning: plotting skipped: %s" % exc)
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(gamma_values, np.ones(len(gamma_values)), s=65, label="train gamma anchors")
    ax.scatter(gamma_mid, np.zeros(len(gamma_mid)), s=55, marker="x", label="interp gamma")
    ax.set_xscale("log")
    ax.set_yticks([0, 1], labels=["interp", "train"])
    ax.set_xlabel("gamma")
    ax.set_title("Exp23 narrow-gamma curriculum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gamma_anchor_layout.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for gamma in gamma_values:
        ax.scatter(a1_values, np.full(len(a1_values), gamma), s=24)
    ax.set_yscale("log")
    ax.set_xlabel("a1")
    ax.set_ylabel("gamma")
    ax.set_title("Exp23 selected training parameter grid")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_parameter_grid.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    source_path = Path(args.source_metadata)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not (2 <= args.observation_points <= args.model_input_points):
        raise ValueError("require 2 <= observation_points <= model_input_points")
    if args.reference_noise <= 0:
        raise ValueError("--reference-noise must be positive")
    if not any(np.isclose(args.train_noise_level, x, atol=1e-12, rtol=0)
               for x in args.noise_levels):
        raise ValueError("--train-noise-level must be included in --noise-levels")

    source_meta = json.loads(source_path.read_text(encoding="utf-8"))
    if source_meta.get("mode") != "a1gamma":
        raise ValueError("source metadata must be a1gamma")

    a1_values = [float(x) for x in source_meta["second_train_values"]]
    gamma_values = sorted({float(x) for x in args.gamma_anchors})
    if len(gamma_values) < 2 or any(x <= 0 for x in gamma_values):
        raise ValueError("gamma anchors must contain >=2 positive unique values")
    if gamma_values[0] < 1e-3 or gamma_values[-1] > 1.0:
        raise ValueError("current model expects gamma anchors within [1e-3, 1]")

    a1_mid = arithmetic_midpoints(a1_values)
    gamma_mid = geometric_midpoints(gamma_values)

    split_states = {
        "train": len(a1_values) * len(gamma_values),
        "val": len(a1_values) * len(gamma_values),
        "test_seen": len(a1_values) * len(gamma_values),
        "test_interp_gamma": len(a1_values) * len(gamma_mid),
        "test_interp_second": len(a1_mid) * len(gamma_values),
        "test_interp_both": len(a1_mid) * len(gamma_mid),
    }
    counts = {
        "train": count_per_state(args.target_train_samples, split_states["train"]),
        "val": count_per_state(args.target_val_samples, split_states["val"]),
        "test_seen": count_per_state(args.target_test_seen_samples, split_states["test_seen"]),
        "test_interp_gamma": count_per_state(args.target_test_interp_samples, split_states["test_interp_gamma"]),
        "test_interp_second": count_per_state(args.target_test_interp_samples, split_states["test_interp_second"]),
        "test_interp_both": count_per_state(args.target_test_interp_samples, split_states["test_interp_both"]),
    }

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError("%s is not empty; use --overwrite" % output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    physics_master = replace(DEFAULT_PHYSICS, q2_points=int(args.model_input_points))
    obs_idx = canonical_indices(args.model_input_points, args.observation_points)

    pair_rows, nearest_rows, sep = separation_diagnostics(
        a1_values,
        gamma_values,
        physics_master,
        obs_idx,
        args.integration_points,
        args.reference_noise,
    )
    write_csv(output_dir / "state_pair_separation.csv", pair_rows)
    write_csv(output_dir / "state_nearest_neighbor_separation.csv", nearest_rows)
    plot_gamma_layout(output_dir, a1_values, gamma_values, gamma_mid)

    split_defs = {
        "test_seen": (a1_values, gamma_values, True, True),
        "test_interp_gamma": (a1_values, gamma_mid, True, False),
        "test_interp_second": (a1_mid, gamma_values, False, True),
        "test_interp_both": (a1_mid, gamma_mid, False, False),
    }

    print("=" * 100)
    print("Exp23 narrow-gamma curriculum")
    print("physical formula       : UNCHANGED")
    print("unknowns               : a1 + gamma")
    print("physical observations  : %d" % args.observation_points)
    print("fixed model input      : %d" % args.model_input_points)
    print("a1 anchors             : %d" % len(a1_values))
    print("gamma anchors          : %d -> %s" % (len(gamma_values), gamma_values))
    print("training states        : %d" % split_states["train"])
    print("train samples          : %d" % (counts["train"] * split_states["train"]))
    print("min observed RMS-SNR   : %.4g" % sep["min_rms_snr"])
    print("min observed full-SNR  : %.4g" % sep["min_full_snr"])
    print("=" * 100)

    for noise_level in args.noise_levels:
        ndir = output_dir / noise_dir_name(noise_level)
        ndir.mkdir(parents=True, exist_ok=True)

        if np.isclose(noise_level, args.train_noise_level, atol=1e-12, rtol=0):
            for split in ("train", "val"):
                arrays = make_split_arrays(
                    mode="a1gamma",
                    split=split,
                    second_values=a1_values,
                    gamma_values=gamma_values,
                    count_per_state=counts[split],
                    noise_level=noise_level,
                    physics=physics_master,
                    integration_points=args.integration_points,
                    base_seed=args.seed,
                    second_seen=True,
                    gamma_seen=True,
                )
                arrays = transform_to_fixed_width(arrays, obs_idx)
                if args.lean_train_files:
                    arrays = slim_for_training(arrays)
                save_npz(ndir / (split + ".npz"), arrays, args.compressed)
                print("saved %s samples=%d" % (ndir / (split + ".npz"), len(arrays["gamma"])))

        for split, (avals, gvals, a_seen, g_seen) in split_defs.items():
            arrays = make_split_arrays(
                mode="a1gamma",
                split=split,
                second_values=avals,
                gamma_values=gvals,
                count_per_state=counts[split],
                noise_level=noise_level,
                physics=physics_master,
                integration_points=args.integration_points,
                base_seed=args.seed,
                second_seen=a_seen,
                gamma_seen=g_seen,
            )
            arrays = transform_to_fixed_width(arrays, obs_idx)
            save_npz(ndir / (split + ".npz"), arrays, args.compressed)
            print("saved %s samples=%d" % (ndir / (split + ".npz"), len(arrays["gamma"])))

    metadata = {
        "experiment": "Exp23 narrow-gamma curriculum",
        "mode": "a1gamma",
        "second_parameter": "a1",
        "fixed_parameters": {"a2": FIXED_A2, "a3": FIXED_A3, "m": FIXED_M},
        "source_exp22a_metadata": str(source_path),
        "controlled_change": "denser gamma coverage with fixed q500 observations and fixed 1000-input capacity",
        "second_train_values": a1_values,
        "second_interp_values": a1_mid,
        "gamma_train_values": gamma_values,
        "gamma_interp_values": gamma_mid,
        "training_state_count": split_states["train"],
        "physical_observation_points": int(args.observation_points),
        "fixed_model_input_points": int(args.model_input_points),
        "observation_indices": [int(x) for x in obs_idx],
        "selected_separation": {
            "state_count": sep["state_count"],
            "min_snr_rms": sep["min_rms_snr"],
            "median_nearest_snr_rms": sep["median_nearest_rms_snr"],
            "min_snr": sep["min_full_snr"],
            "median_nearest_snr": sep["median_nearest_full_snr"],
            "min_relative_l2": sep["min_relative_l2"],
        },
        "noise_levels": [float(x) for x in args.noise_levels],
        "train_noise_level": float(args.train_noise_level),
        "reference_noise": float(args.reference_noise),
        "counts_per_state": counts,
        "split_state_counts": split_states,
        "actual_sample_counts": {k: int(counts[k] * split_states[k]) for k in counts},
        "q2_range": [float(physics_master.q2_min), float(physics_master.q2_max)],
        "q2_points": int(args.model_input_points),
        "output_points": int(physics_master.output_points),
        "integration_points": int(args.integration_points),
        "notes": [
            "The previous Exp22A gamma anchors were [0.01, 0.1, 1].",
            "Exp23 adds the geometric midpoints 0.0316228 and 0.316228 as train anchors.",
            "New interpolation gamma values are geometric midpoints between the five Exp23 anchors.",
            "Total sample budgets are held near the previous experiments; samples per state decrease as state count increases.",
            "Physical q2 observations remain 500 and the model input width remains 1000.",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "gamma_anchors.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["kind", "gamma"])
        writer.writeheader()
        for x in gamma_values:
            writer.writerow({"kind": "train", "gamma": float(x)})
        for x in gamma_mid:
            writer.writerow({"kind": "interp", "gamma": float(x)})

    print("=" * 100)
    print("Exp23 data ready")
    print("output: %s" % output_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()
