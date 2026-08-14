#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp22C: fixed-network-capacity control for q^2 observation density.

Scientific question
-------------------
Exp22B improved strongly when the raw input length increased from 100 to 500/1000,
but the first MLP layer also became larger. Exp22C removes that confound.

All conditions use the SAME model input width (default 1000) and therefore the
same trainable parameter count. The physical observation count is varied
(100/500/1000). A noisy curve observed at N_obs physical q^2 points is linearly
resampled onto the common 1000-point model grid before entering the network.

Important:
- Interpolation is only a fixed-width representation. It does NOT create new
  independent physical observations.
- The same Exp22A 9x3 a1-gamma classes are used in every condition.
- Physical formula, q^2 range, noise level, sample counts, MLP hidden widths,
  and training loss are unchanged.
- Noise is generated first on a common 1000-point master grid using the same
  deterministic seeds; lower-density conditions retain subsets of those noisy
  observations and interpolate them to the common model grid.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np

from data_generate_exp20_pairwise import (
    DEFAULT_PHYSICS,
    FIXED_A2,
    FIXED_A3,
    FIXED_M,
    arithmetic_midpoints,
    build_parameter_rows,
    forward_bank,
    geometric_midpoints,
    make_split_arrays,
    noise_dir_name,
    save_npz,
    write_csv,
)


def parse_float_list(text):
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def parse_args():
    p = argparse.ArgumentParser(
        description="Exp22C fixed-capacity q2 observation-density dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source-metadata", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--observation-points", type=int, required=True)
    p.add_argument("--model-input-points", type=int, default=1000)
    p.add_argument("--reference-noise", type=float, default=0.002)
    p.add_argument("--noise-levels", type=parse_float_list, default=[0.0, 0.002, 0.01])
    p.add_argument("--train-noise-level", type=float, default=0.002)
    p.add_argument("--integration-points", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260813)

    # 0 => reuse Exp22A counts_per_state.
    p.add_argument("--train-per-state", type=int, default=0)
    p.add_argument("--val-per-state", type=int, default=0)
    p.add_argument("--test-seen-per-state", type=int, default=0)
    p.add_argument("--test-interp-per-state", type=int, default=0)

    # Compatible with older Python versions (no argparse.BooleanOptionalAction).
    lean = p.add_mutually_exclusive_group()
    lean.add_argument("--lean-train-files", dest="lean_train_files", action="store_true")
    lean.add_argument("--no-lean-train-files", dest="lean_train_files", action="store_false")
    p.set_defaults(lean_train_files=True)

    p.add_argument("--compressed", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def source_count(meta, key):
    counts = meta.get("counts_per_state", {})
    if key not in counts:
        raise KeyError(
            "source metadata does not contain counts_per_state[%r]; "
            "use an explicit --*-per-state override" % key
        )
    return int(counts[key])


def resolve_counts(args, meta):
    interp_gamma = source_count(meta, "test_interp_gamma")
    out = {
        "train": args.train_per_state if args.train_per_state > 0 else source_count(meta, "train"),
        "val": args.val_per_state if args.val_per_state > 0 else source_count(meta, "val"),
        "test_seen": (
            args.test_seen_per_state if args.test_seen_per_state > 0
            else source_count(meta, "test_seen")
        ),
        "test_interp_gamma": (
            args.test_interp_per_state if args.test_interp_per_state > 0
            else interp_gamma
        ),
        "test_interp_second": (
            args.test_interp_per_state if args.test_interp_per_state > 0
            else source_count(meta, "test_interp_second")
        ),
        "test_interp_both": (
            args.test_interp_per_state if args.test_interp_per_state > 0
            else source_count(meta, "test_interp_both")
        ),
    }
    out = {k: int(v) for k, v in out.items()}
    if any(v <= 0 for v in out.values()):
        raise ValueError("all per-state counts must be positive, got %r" % out)
    return out


def canonical_indices(master_points, observation_points):
    """Return deterministic, endpoint-preserving observation indices.

    For the default 100/500/1000 experiment we make the sets nested:
    q100 subset q500 subset q1000. This makes the control more paired.
    """
    master_points = int(master_points)
    observation_points = int(observation_points)
    if not (2 <= observation_points <= master_points):
        raise ValueError("require 2 <= observation_points <= model_input_points")

    if observation_points == master_points:
        return np.arange(master_points, dtype=np.int64)

    def rounded_linspace(total, count):
        idx = np.rint(np.linspace(0, total - 1, count)).astype(np.int64)
        idx[0] = 0
        idx[-1] = total - 1
        if len(np.unique(idx)) != count:
            # Fallback that guarantees uniqueness.
            idx = np.floor(np.linspace(0, total, count, endpoint=False)).astype(np.int64)
            idx[-1] = total - 1
            idx = np.unique(idx)
            if len(idx) != count:
                raise RuntimeError("could not construct unique observation indices")
        return idx

    if master_points == 1000 and observation_points == 500:
        return rounded_linspace(master_points, 500)

    if master_points == 1000 and observation_points == 100:
        parent = rounded_linspace(master_points, 500)
        sub = rounded_linspace(len(parent), 100)
        idx = parent[sub]
        if len(np.unique(idx)) != 100:
            raise RuntimeError("nested q100 index construction failed")
        return idx.astype(np.int64)

    return rounded_linspace(master_points, observation_points)


def interpolation_plan(q_obs, q_model):
    q_obs = np.asarray(q_obs, dtype=np.float64)
    q_model = np.asarray(q_model, dtype=np.float64)
    if np.any(np.diff(q_obs) <= 0) or np.any(np.diff(q_model) <= 0):
        raise ValueError("q grids must be strictly increasing")
    if q_obs[0] > q_model[0] + 1e-12 or q_obs[-1] < q_model[-1] - 1e-12:
        raise ValueError("observation grid must cover model grid endpoints")

    hi = np.searchsorted(q_obs, q_model, side="left")
    hi = np.clip(hi, 1, len(q_obs) - 1)
    lo = hi - 1
    x0 = q_obs[lo]
    x1 = q_obs[hi]
    w = (q_model - x0) / np.maximum(x1 - x0, 1e-30)
    return lo.astype(np.int64), hi.astype(np.int64), w.astype(np.float32)


def interpolate_rows(values, lo, hi, w, chunk_rows=1024):
    values = np.asarray(values, dtype=np.float32)
    out = np.empty((values.shape[0], len(w)), dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)[None, :]
    for start in range(0, len(values), int(chunk_rows)):
        stop = min(start + int(chunk_rows), len(values))
        block = values[start:stop]
        left = block[:, lo]
        right = block[:, hi]
        out[start:stop] = left + (right - left) * w
    return out


def slim_for_training(arrays):
    keep = {"gy_noisy", "gamma", "a1", "noise_level", "q2", "y", "x"}
    return {k: v for k, v in arrays.items() if k in keep}


def controlled_pair_metrics(g_obs_1, g_obs_2, g_master_1, g_master_2, noise_level):
    a = np.asarray(g_obs_1, dtype=np.float64)
    b = np.asarray(g_obs_2, dtype=np.float64)
    ma = np.asarray(g_master_1, dtype=np.float64)
    mb = np.asarray(g_master_2, dtype=np.float64)
    d = a - b
    l2 = float(np.linalg.norm(d))
    rms_d = float(np.sqrt(np.mean(d * d)))
    denom = 0.5 * (np.linalg.norm(a) + np.linalg.norm(b))
    rel_l2 = l2 / denom if denom > 0 else math.inf
    mean_master_rms = 0.5 * (
        float(np.sqrt(np.mean(ma * ma))) + float(np.sqrt(np.mean(mb * mb)))
    )
    sigma = float(noise_level) * mean_master_rms
    return {
        "rms_g_difference": rms_d,
        "l2_g_difference": l2,
        "relative_l2_difference": float(rel_l2),
        "master_mean_rms_g": mean_master_rms,
        "sigma_reference": sigma,
        "snr_sep_rms": rms_d / sigma if sigma > 0 else math.inf,
        "snr_sep": l2 / sigma if sigma > 0 else math.inf,
    }


def separation_diagnostics(a1_values, gamma_values, physics, obs_idx, integration_points, reference_noise):
    params = build_parameter_rows("a1gamma", a1_values, gamma_values)
    _, g_master = forward_bank(params, physics=physics, integration_points=integration_points)
    g_obs = g_master[:, obs_idx]

    states = []
    n_gamma = len(gamma_values)
    for ia, a1 in enumerate(a1_values):
        for ig, gamma in enumerate(gamma_values):
            states.append({
                "state_index": ia * n_gamma + ig,
                "a1": float(a1),
                "gamma": float(gamma),
            })

    nearest = [None] * len(states)
    pair_rows = []
    for i, j in combinations(range(len(states)), 2):
        m = controlled_pair_metrics(
            g_obs[i], g_obs[j], g_master[i], g_master[j], reference_noise
        )
        pair_rows.append({
            "state_1": i,
            "state_2": j,
            "a1_1": states[i]["a1"],
            "a1_2": states[j]["a1"],
            "gamma_1": states[i]["gamma"],
            "gamma_2": states[j]["gamma"],
            "reference_noise_level": float(reference_noise),
            **m,
        })
        for state, other in ((i, j), (j, i)):
            current = nearest[state]
            if current is None or m["snr_sep_rms"] < current["snr_sep_rms"]:
                nearest[state] = {
                    "state": state,
                    "nearest_state": other,
                    "a1": states[state]["a1"],
                    "nearest_a1": states[other]["a1"],
                    "gamma": states[state]["gamma"],
                    "nearest_gamma": states[other]["gamma"],
                    **m,
                }

    nearest_rows = [r for r in nearest if r is not None]
    rms = np.asarray([r["snr_sep_rms"] for r in nearest_rows], dtype=np.float64)
    full = np.asarray([r["snr_sep"] for r in nearest_rows], dtype=np.float64)
    rel = np.asarray([r["relative_l2_difference"] for r in nearest_rows], dtype=np.float64)
    summary = {
        "state_count": int(len(states)),
        "min_rms_snr": float(np.min(rms)),
        "median_nearest_rms_snr": float(np.median(rms)),
        "min_full_snr": float(np.min(full)),
        "median_nearest_full_snr": float(np.median(full)),
        "min_relative_l2": float(np.min(rel)),
    }
    return pair_rows, nearest_rows, summary


def transform_to_fixed_width(arrays, obs_idx):
    q_model = np.asarray(arrays["q2"], dtype=np.float32)
    q_obs = q_model[obs_idx]
    noisy_observed = np.asarray(arrays["gy_noisy"], dtype=np.float32)[:, obs_idx]
    lo, hi, w = interpolation_plan(q_obs, q_model)
    arrays = dict(arrays)
    arrays["gy_noisy"] = interpolate_rows(noisy_observed, lo, hi, w)
    arrays["q2_observed"] = q_obs.astype(np.float32)
    arrays["observation_indices"] = np.asarray(obs_idx, dtype=np.int32)
    arrays["observation_points"] = np.asarray([len(obs_idx)], dtype=np.int32)
    arrays["model_input_points"] = np.asarray([len(q_model)], dtype=np.int32)
    return arrays


def plot_observation_layout(output_dir, q_model, obs_idx):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("warning: plotting skipped: %s" % exc)
        return
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.scatter(q_model[obs_idx], np.zeros(len(obs_idx)), s=12)
    ax.set_yticks([])
    ax.set_xlabel("q2")
    ax.set_title(
        "Exp22C physical observations: %d, fixed model input width: %d"
        % (len(obs_idx), len(q_model))
    )
    fig.tight_layout()
    fig.savefig(output_dir / "observation_layout.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    source_path = Path(args.source_metadata)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if args.model_input_points < 2:
        raise ValueError("--model-input-points must be >= 2")
    if not (2 <= args.observation_points <= args.model_input_points):
        raise ValueError("observation points must be between 2 and model input points")
    if args.reference_noise <= 0:
        raise ValueError("--reference-noise must be > 0")
    if not any(
        np.isclose(args.train_noise_level, x, atol=1e-12, rtol=0)
        for x in args.noise_levels
    ):
        raise ValueError("--train-noise-level must be included in --noise-levels")

    source_meta = json.loads(source_path.read_text(encoding="utf-8"))
    if source_meta.get("mode") != "a1gamma":
        raise ValueError("Exp22C expects Exp22A a1gamma metadata")

    a1_values = [float(x) for x in source_meta["second_train_values"]]
    gamma_values = [float(x) for x in source_meta["gamma_train_values"]]
    a1_mid = arithmetic_midpoints(a1_values)
    gamma_mid = geometric_midpoints(gamma_values)
    counts = resolve_counts(args, source_meta)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError("%s is not empty; use --overwrite" % output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Every condition is first generated on the same master grid. This ensures
    # identical input width and paired noise realizations before subsampling.
    physics_master = replace(DEFAULT_PHYSICS, q2_points=int(args.model_input_points))
    q_model = np.linspace(
        physics_master.q2_min,
        physics_master.q2_max,
        physics_master.q2_points,
        dtype=np.float64,
    )
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
    plot_observation_layout(output_dir, q_model, obs_idx)

    split_defs = {
        "test_seen": (a1_values, gamma_values, True, True),
        "test_interp_gamma": (a1_values, gamma_mid, True, False),
        "test_interp_second": (a1_mid, gamma_values, False, True),
        "test_interp_both": (a1_mid, gamma_mid, False, False),
    }

    print("=" * 100)
    print("Exp22C: fixed model capacity, variable physical q2 observations")
    print("Physical formula       : UNCHANGED")
    print("Parameter classes      : UNCHANGED from Exp22A")
    print("model input points     : %d (FIXED for every condition)" % args.model_input_points)
    print("physical observations  : %d" % args.observation_points)
    print("q2 range               : [%g, %g]" % (physics_master.q2_min, physics_master.q2_max))
    print("reference noise        : %.4g%%" % (100.0 * args.reference_noise))
    print("training states        : %d" % (len(a1_values) * len(gamma_values)))
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
        "experiment": "Exp22C fixed-network-capacity q2 observation-density control",
        "mode": "a1gamma",
        "second_parameter": "a1",
        "fixed_parameters": {"a2": FIXED_A2, "a3": FIXED_A3, "m": FIXED_M},
        "source_exp22a_metadata": str(source_path),
        "controlled_change": "physical_observation_points only",
        "fixed_model_input_points": int(args.model_input_points),
        "physical_observation_points": int(args.observation_points),
        "observation_indices": [int(x) for x in obs_idx],
        "observation_q2": [float(x) for x in q_model[obs_idx]],
        "representation": (
            "Noisy physical observations are linearly interpolated to the common "
            "%d-point model grid. Interpolation does not add independent observations."
            % args.model_input_points
        ),
        "paired_noise_design": (
            "All conditions generate the same seeded noisy 1000-point master curves first; "
            "lower-density conditions keep deterministic subsets before interpolation."
        ),
        "second_train_values": a1_values,
        "second_interp_values": a1_mid,
        "gamma_train_values": gamma_values,
        "gamma_interp_values": gamma_mid,
        "training_state_count": int(len(a1_values) * len(gamma_values)),
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
        "q2_range": [float(physics_master.q2_min), float(physics_master.q2_max)],
        "q2_points": int(args.model_input_points),
        "output_points": int(physics_master.output_points),
        "integration_points": int(args.integration_points),
        "counts_per_state": counts,
        "noise_definition": "master g_noisy = g_clean + noise_level * RMS(master g_clean) * N(0,1)",
        "notes": [
            "Every Exp22C condition presents exactly the same 1000 input features to the MLP.",
            "Therefore the first linear layer and total trainable parameter count are identical.",
            "Physical observation count changes before interpolation: 100, 500, or 1000.",
            "The common interpolation representation is intentionally information-neutral with respect to the number of independent observations.",
            "This experiment isolates the Exp22B observation-density gain from the larger first-layer parameter count.",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "observation_indices.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["slot", "master_index", "q2"])
        writer.writeheader()
        for slot, idx in enumerate(obs_idx):
            writer.writerow({"slot": slot, "master_index": int(idx), "q2": float(q_model[idx])})

    print("=" * 100)
    print("Exp22C data ready")
    print("output: %s" % output_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()
