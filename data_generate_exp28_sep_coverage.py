#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp28: separation + coverage constrained independent data.

Scientific purpose
------------------
Exp27 showed that increasing the *minimum* g-space separation from RMS-SNR 1.0
to about 1.5 reduced the tail of gamma failures, but pushing it to 2.0 removed
too many physical states and hurt continuous interpolation/generalization.

Exp28 therefore keeps the SAME globally separated physical state bank from the
Exp27 RMS-SNR=1.5 run and changes only the split policy:

    1) all retained physical truths already satisfy global pairwise
       g-separation >= source threshold (~1.5);
    2) choose TRAIN states by farthest-first coverage of that same state bank;
    3) require every VAL/TEST state to have a nearest TRAIN state with
       RMS-SNR <= max_train_cover_rms_snr (default 2.0);
    4) VAL/TEST remain disjoint physical truths and still inherit the global
       minimum g-separation.

So the intended hold-out relation is a bounded annulus in g-space:

    min separation  <=  d(g_holdout, g_train)  <=  max coverage distance

This does NOT change the forward physics or artificially modify g.  It changes
only which legitimate physical states are assigned to train/val/test.
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
import pandas as pd

from data_generate_exp20_pairwise import (
    DEFAULT_PHYSICS,
    FIXED_A2,
    FIXED_A3,
    FIXED_M,
    forward_bank,
    noise_dir_name,
    save_npz,
    write_csv,
)
from data_generate_exp22c_fixed_capacity import canonical_indices, slim_for_training
from data_generate_exp26_gseparated_independent import (
    count_per_state,
    make_split_arrays_from_bank,
    pairwise_split_summary,
    rms_snr_to_one,
    selected_nearest_rows,
)


def parse_float_list(text):
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return vals


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate Exp28 separation + coverage constrained data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source-data-dir",
        default="data_exp27_gsep_snr1p5",
        help="completed Exp27 threshold=1.5 data directory; selected physical bank is reused",
    )
    p.add_argument("--output-dir", default="data_exp28_sep1p5_cover2")
    p.add_argument("--max-train-cover-rms-snr", type=float, default=2.0)
    p.add_argument("--split-gamma-bins", type=int, default=8)
    p.add_argument("--reference-noise", type=float, default=-1.0,
                   help="<0 means reuse source metadata reference noise")
    p.add_argument("--integration-points", type=int, default=-1,
                   help="<0 means reuse source metadata integration points")
    p.add_argument("--noise-levels", type=parse_float_list, default=[0.0, 0.002, 0.01])
    p.add_argument("--train-noise-level", type=float, default=0.002)
    p.add_argument("--target-train-samples", type=int, default=20000)
    p.add_argument("--target-val-samples", type=int, default=3000)
    p.add_argument("--target-test-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--compressed", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_source_bank(source_dir):
    meta_path = source_dir / "metadata.json"
    states_path = source_dir / "selected_states.csv"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    if not states_path.exists():
        raise FileNotFoundError(states_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    states = pd.read_csv(states_path)
    if "state_index" in states.columns:
        states = states.sort_values("state_index").reset_index(drop=True)

    if "a1" not in states.columns or "gamma" not in states.columns:
        raise ValueError("source selected_states.csv must contain a1 and gamma")

    params = np.zeros((len(states), 5), dtype=np.float64)
    params[:, 0] = states["a1"].to_numpy(float)
    params[:, 1] = float(meta.get("fixed_parameters", {}).get("a2", FIXED_A2))
    params[:, 2] = float(meta.get("fixed_parameters", {}).get("a3", FIXED_A3))
    params[:, 3] = float(meta.get("fixed_parameters", {}).get("m", FIXED_M))
    params[:, 4] = states["gamma"].to_numpy(float)
    return meta, states, params


def pairwise_rms_snr_matrix(g_obs, master_rms, reference_noise):
    n = len(master_rms)
    dmat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        js = np.arange(n, dtype=np.int64)
        dmat[i] = rms_snr_to_one(
            g_obs, master_rms, reference_noise, int(i), js
        )
        dmat[i, i] = 0.0
    return dmat


def domain_corner_indices(params):
    a1_min = float(np.min(params[:, 0]))
    a1_max = float(np.max(params[:, 0]))
    g_min = float(np.min(params[:, 4]))
    g_max = float(np.max(params[:, 4]))
    a_span = max(a1_max - a1_min, 1e-12)
    lg = np.log10(params[:, 4])
    lg_min = math.log10(g_min)
    lg_max = math.log10(g_max)
    lg_span = max(lg_max - lg_min, 1e-12)

    corners = [
        (a1_min, lg_min),
        (a1_min, lg_max),
        (a1_max, lg_min),
        (a1_max, lg_max),
    ]
    idx = []
    for a, l in corners:
        dist2 = ((params[:, 0] - a) / a_span) ** 2
        dist2 += ((lg - l) / lg_span) ** 2
        idx.append(int(np.argmin(dist2)))
    return sorted(set(idx))


def target_split_counts(source_meta, n_states):
    source_counts = source_meta.get("selected_state_counts", {})
    if all(k in source_counts for k in ("train", "val", "test")):
        vals = np.asarray([
            int(source_counts["train"]),
            int(source_counts["val"]),
            int(source_counts["test"]),
        ], dtype=int)
        if int(np.sum(vals)) == int(n_states) and np.all(vals > 0):
            return {"train": int(vals[0]), "val": int(vals[1]), "test": int(vals[2])}

    # Fallback: 70/15/15 with largest-remainder allocation.
    raw = np.asarray([0.70, 0.15, 0.15], dtype=float) * int(n_states)
    counts = np.floor(raw).astype(int)
    rem = int(n_states) - int(np.sum(counts))
    frac = raw - counts
    for k in np.argsort(-frac)[:rem]:
        counts[k] += 1
    return {"train": int(counts[0]), "val": int(counts[1]), "test": int(counts[2])}


def choose_train_farthest_first(dmat, params, target_train, max_cover):
    """Choose a g-space covering TRAIN set; add extra centers only if required."""
    n = len(params)
    forced = domain_corner_indices(params)
    train = list(forced)
    remaining = set(range(n)) - set(train)

    nearest = np.min(dmat[:, train], axis=1)
    for i in train:
        nearest[i] = 0.0

    while len(train) < int(target_train):
        j = max(remaining, key=lambda x: float(nearest[x]))
        train.append(int(j))
        remaining.remove(j)
        nearest = np.minimum(nearest, dmat[:, j])
        nearest[train] = 0.0

    # Coverage constraint: if the nominal train count is insufficient, promote
    # the farthest holdout state to train until all remaining states are covered.
    promoted = 0
    while remaining:
        farthest = max(remaining, key=lambda x: float(nearest[x]))
        if float(nearest[farthest]) <= float(max_cover) + 1e-12:
            break
        train.append(int(farthest))
        remaining.remove(farthest)
        promoted += 1
        nearest = np.minimum(nearest, dmat[:, farthest])
        nearest[train] = 0.0

    return np.asarray(sorted(train), dtype=np.int32), nearest, promoted


def stratified_val_test(remaining, params, n_val, n_test, gamma_bins, seed):
    remaining = np.asarray(remaining, dtype=np.int32)
    if len(remaining) != int(n_val) + int(n_test):
        raise ValueError("remaining state count does not match val+test target")
    if n_val < 1 or n_test < 1:
        raise ValueError("val/test counts must both be positive")

    rng = np.random.default_rng(int(seed) + 2801)
    logg = np.log10(params[:, 4])
    edges = np.linspace(logg.min(), logg.max(), int(gamma_bins) + 1)
    bid = np.clip(
        np.digitize(logg, edges[1:-1], right=False),
        0,
        int(gamma_bins) - 1,
    )

    val = []
    test = []
    target_val_fraction = float(n_val) / float(n_val + n_test)

    for b in range(int(gamma_bins)):
        idx = remaining[bid[remaining] == b].copy()
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        raw_val = target_val_fraction * len(idx)
        k_val = int(round(raw_val))
        if len(idx) >= 2:
            k_val = min(max(k_val, 1), len(idx) - 1)
        else:
            k_val = 1 if len(val) < n_val else 0
        val.extend([int(x) for x in idx[:k_val]])
        test.extend([int(x) for x in idx[k_val:]])

    # Adjust exact totals while moving the states closest to the other split's
    # gamma support first; this keeps both supports broad.
    def move_one(src, dst):
        if not src:
            raise RuntimeError("cannot rebalance val/test")
        src_g = np.log10(params[np.asarray(src, dtype=int), 4])
        if dst:
            dst_g = np.log10(params[np.asarray(dst, dtype=int), 4])
            scores = np.min(np.abs(src_g[:, None] - dst_g[None, :]), axis=1)
            k = int(np.argmax(scores))  # move a state that improves destination coverage
        else:
            k = 0
        dst.append(src.pop(k))

    while len(val) > n_val:
        move_one(val, test)
    while len(val) < n_val:
        move_one(test, val)
    while len(test) > n_test:
        move_one(test, val)
    while len(test) < n_test:
        move_one(val, test)

    return np.asarray(sorted(val), dtype=np.int32), np.asarray(sorted(test), dtype=np.int32)


def coverage_rows(params, split_labels, dmat):
    train_idx = np.where(split_labels == "train")[0]
    rows = []
    for i in range(len(params)):
        if split_labels[i] == "train":
            continue
        scores = dmat[i, train_idx]
        k = int(np.argmin(scores))
        j = int(train_idx[k])
        rows.append({
            "state_index": int(i),
            "split": str(split_labels[i]),
            "a1": float(params[i, 0]),
            "gamma": float(params[i, 4]),
            "nearest_train_state": j,
            "nearest_train_a1": float(params[j, 0]),
            "nearest_train_gamma": float(params[j, 4]),
            "nearest_train_rms_snr": float(scores[k]),
        })
    return rows


def summarize_coverage(rows):
    out = []
    df = pd.DataFrame(rows)
    for name in ("val", "test", "holdout_all"):
        x = df if name == "holdout_all" else df[df["split"] == name]
        arr = x["nearest_train_rms_snr"].to_numpy(float)
        out.append({
            "split": name,
            "count": int(len(arr)),
            "min_nearest_train_rms_snr": float(np.min(arr)),
            "median_nearest_train_rms_snr": float(np.median(arr)),
            "p90_nearest_train_rms_snr": float(np.quantile(arr, 0.90)),
            "max_nearest_train_rms_snr": float(np.max(arr)),
        })
    return out


def plot_diagnostics(output_dir, params, split_labels, coverage, min_sep, max_cover):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("warning: plotting skipped: %s" % exc)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ("train", "val", "test"):
        idx = np.where(split_labels == name)[0]
        ax.scatter(params[idx, 0], params[idx, 4], s=35, alpha=0.85, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("a1")
    ax.set_ylabel("gamma")
    ax.set_title("Exp28 same separated state bank, coverage-aware split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_aware_split_parameter_support.png", dpi=170)
    plt.close(fig)

    cdf = pd.DataFrame(coverage)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ("val", "test"):
        arr = cdf[cdf["split"] == name]["nearest_train_rms_snr"].to_numpy(float)
        ax.hist(arr, bins=15, alpha=0.55, label=name)
    ax.axvline(float(min_sep), linestyle="--", label="global min separation")
    ax.axvline(float(max_cover), linestyle=":", label="max allowed train coverage")
    ax.set_xlabel("holdout -> nearest train RMS-SNR")
    ax.set_ylabel("state count")
    ax.set_title("Exp28 holdout coverage in observed g-space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "holdout_to_train_coverage_hist.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        cdf["a1"], cdf["gamma"],
        c=cdf["nearest_train_rms_snr"], s=60
    )
    tr = np.where(split_labels == "train")[0]
    ax.scatter(params[tr, 0], params[tr, 4], s=20, alpha=0.25, label="train")
    ax.set_yscale("log")
    ax.set_xlabel("a1")
    ax.set_ylabel("gamma")
    ax.set_title("Exp28 val/test distance to nearest training state")
    fig.colorbar(sc, ax=ax, label="nearest train RMS-SNR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "holdout_coverage_parameter_map.png", dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    source_dir = Path(args.source_data_dir)
    output_dir = Path(args.output_dir)

    source_meta, source_states, params = load_source_bank(source_dir)
    n = len(params)
    if n < 30:
        raise ValueError("source physical state bank is too small")

    source_min_sep = float(source_meta.get(
        "actual_global_min_rms_snr",
        source_meta.get("global_min_rms_snr", math.nan),
    ))
    if not np.isfinite(source_min_sep):
        raise ValueError("source metadata lacks global minimum RMS-SNR")

    reference_noise = (
        float(source_meta.get("reference_noise", 0.002))
        if args.reference_noise < 0 else float(args.reference_noise)
    )
    integration_points = (
        int(source_meta.get("integration_points", 128))
        if args.integration_points < 0 else int(args.integration_points)
    )
    obs_points = int(source_meta.get("physical_observation_points", 500))
    model_points = int(source_meta.get("model_input_points", 1000))

    if args.max_train_cover_rms_snr <= source_min_sep:
        raise ValueError(
            "max train coverage %.4g must exceed source global separation %.4g"
            % (args.max_train_cover_rms_snr, source_min_sep)
        )
    if not any(np.isclose(args.train_noise_level, x, atol=1e-12, rtol=0)
               for x in args.noise_levels):
        raise ValueError("train noise must be included in noise levels")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError("%s is not empty; use --overwrite" % output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    physics = replace(DEFAULT_PHYSICS, q2_points=model_points)
    obs_idx = canonical_indices(model_points, obs_points)

    # Same physical bank; only split policy changes.
    fx_bank, g_master = forward_bank(
        params, physics=physics, integration_points=integration_points
    )
    g_obs = np.asarray(g_master[:, obs_idx], dtype=np.float32)
    master_rms = np.sqrt(np.mean(np.asarray(g_master, dtype=np.float64) ** 2, axis=1))
    dmat = pairwise_rms_snr_matrix(g_obs, master_rms, reference_noise)

    offdiag = dmat[~np.eye(n, dtype=bool)]
    actual_global_min = float(np.min(offdiag))
    if actual_global_min + 1e-6 < source_min_sep:
        raise RuntimeError(
            "recomputed source bank min separation %.6g < source metadata %.6g"
            % (actual_global_min, source_min_sep)
        )

    targets = target_split_counts(source_meta, n)
    train_idx, nearest_to_train, promoted = choose_train_farthest_first(
        dmat,
        params,
        target_train=targets["train"],
        max_cover=args.max_train_cover_rms_snr,
    )
    remaining = np.asarray(
        sorted(set(range(n)) - set(int(x) for x in train_idx)),
        dtype=np.int32,
    )

    # If coverage forced extra train centers, preserve val:test ratio in what remains.
    if len(train_idx) != targets["train"]:
        holdout = len(remaining)
        source_holdout = max(targets["val"] + targets["test"], 1)
        val_ratio = float(targets["val"]) / float(source_holdout)
        n_val = int(round(val_ratio * holdout))
        n_val = min(max(n_val, 1), holdout - 1)
        n_test = holdout - n_val
    else:
        n_val = targets["val"]
        n_test = targets["test"]

    val_idx, test_idx = stratified_val_test(
        remaining,
        params,
        n_val=n_val,
        n_test=n_test,
        gamma_bins=args.split_gamma_bins,
        seed=args.seed,
    )

    split_labels = np.full(n, "", dtype=object)
    split_labels[train_idx] = "train"
    split_labels[val_idx] = "val"
    split_labels[test_idx] = "test"
    if np.any(split_labels == ""):
        raise RuntimeError("some physical states were not assigned")

    coverage = coverage_rows(params, split_labels, dmat)
    coverage_summary = summarize_coverage(coverage)
    max_holdout_cover = max(r["max_nearest_train_rms_snr"] for r in coverage_summary)
    if max_holdout_cover > args.max_train_cover_rms_snr + 1e-8:
        raise RuntimeError(
            "coverage guarantee failed: max holdout distance %.6g > %.6g"
            % (max_holdout_cover, args.max_train_cover_rms_snr)
        )

    nearest_rows, nearest_scores, nearest_idx = selected_nearest_rows(
        params, g_obs, master_rms, reference_noise
    )
    for i, row in enumerate(nearest_rows):
        row["split"] = str(split_labels[i])
        if "candidate_index" in source_states.columns:
            row["source_candidate_index"] = int(source_states.iloc[i]["candidate_index"])
    write_csv(output_dir / "selected_states.csv", nearest_rows)
    write_csv(
        output_dir / "split_pair_gseparation_summary.csv",
        pairwise_split_summary(
            params, g_obs, master_rms, split_labels, reference_noise
        ),
    )
    write_csv(output_dir / "holdout_to_train_coverage.csv", coverage)
    write_csv(output_dir / "coverage_summary.csv", coverage_summary)
    plot_diagnostics(
        output_dir, params, split_labels, coverage,
        min_sep=actual_global_min,
        max_cover=args.max_train_cover_rms_snr,
    )

    split_counts = {
        name: int(np.sum(split_labels == name))
        for name in ("train", "val", "test")
    }
    sample_targets = {
        "train": int(args.target_train_samples),
        "val": int(args.target_val_samples),
        "test": int(args.target_test_samples),
    }
    per_state = {
        name: count_per_state(sample_targets[name], split_counts[name])
        for name in ("train", "val", "test")
    }

    try:
        from mc_physics import output_grids_numpy
        s_grid, q_master = output_grids_numpy(physics)
    except Exception:
        q_master = np.linspace(
            physics.q2_min, physics.q2_max, physics.q2_points, dtype=np.float64
        )
        s_grid = np.arange(fx_bank.shape[1], dtype=np.float64)

    for noise_level in args.noise_levels:
        ndir = output_dir / noise_dir_name(noise_level)
        ndir.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "val", "test"):
            if split_name != "test" and not np.isclose(
                noise_level, args.train_noise_level, atol=1e-12, rtol=0
            ):
                continue
            idx = np.where(split_labels == split_name)[0]
            arrays = make_split_arrays_from_bank(
                split_name=split_name,
                params=params[idx],
                fx_bank=fx_bank[idx],
                g_master_bank=g_master[idx],
                count_per_state=per_state[split_name],
                noise_level=float(noise_level),
                q_master=np.asarray(q_master, dtype=np.float64),
                obs_idx=obs_idx,
                seed=args.seed + 28,
            )
            arrays["x"] = np.asarray(s_grid, dtype=np.float32)
            if split_name in ("train", "val"):
                arrays = slim_for_training(arrays)
            save_npz(ndir / (split_name + ".npz"), arrays, args.compressed)
            print(
                "saved %-5s noise=%7.4g | physical states=%3d | samples=%6d"
                % (split_name, noise_level, len(idx), len(arrays["gamma"]))
            )

    metadata = {
        "experiment": "Exp28 separation + coverage constrained independent splits",
        "mode": "a1gamma",
        "second_parameter": "a1",
        "fixed_parameters": {
            "a2": float(params[0, 1]),
            "a3": float(params[0, 2]),
            "m": float(params[0, 3]),
        },
        "source_data_dir": str(source_dir),
        "source_experiment": source_meta.get("experiment", "Exp27/Exp26 g-separated bank"),
        "source_requested_min_rms_snr": float(source_meta.get(
            "requested_min_rms_snr", source_min_sep
        )),
        "global_min_rms_snr": actual_global_min,
        "requested_min_rms_snr": float(source_meta.get(
            "requested_min_rms_snr", source_min_sep
        )),
        "max_train_cover_rms_snr": float(args.max_train_cover_rms_snr),
        "coverage_promoted_extra_train_states": int(promoted),
        "selected_state_count": int(n),
        "selected_state_counts": split_counts,
        "source_selected_state_counts": source_meta.get("selected_state_counts", {}),
        "a1_range": [
            float(np.min(params[:, 0])),
            float(np.max(params[:, 0])),
        ],
        "gamma_range": [
            float(np.min(params[:, 4])),
            float(np.max(params[:, 4])),
        ],
        "reference_noise": reference_noise,
        "physical_observation_points": obs_points,
        "model_input_points": model_points,
        "observation_indices": [int(x) for x in obs_idx],
        "noise_levels": [float(x) for x in args.noise_levels],
        "train_noise_level": float(args.train_noise_level),
        "per_state_sample_counts": per_state,
        "actual_sample_counts": {
            name: int(per_state[name] * split_counts[name])
            for name in ("train", "val", "test")
        },
        "coverage_summary": coverage_summary,
        "split_policy": (
            "Reuse the exact globally separated source state bank. "
            "Choose train states by farthest-first g-space coverage, then assign "
            "the remaining disjoint states to val/test with log-gamma stratification."
        ),
        "independence_guarantee": (
            "train/val/test physical parameter truths are disjoint. "
            "All states inherit the source global minimum g-separation."
        ),
        "coverage_guarantee": (
            "Every val/test physical truth has at least one train truth with "
            "RMS-SNR distance <= max_train_cover_rms_snr."
        ),
        "integration_points": integration_points,
        "seed": int(args.seed),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("Exp28 data ready")
    print("same physical state bank : %d states" % n)
    print("split counts             : %s" % split_counts)
    print("global minimum RMS-SNR   : %.6g" % actual_global_min)
    print("max holdout->train RMS-SNR: %.6g" % max_holdout_cover)
    print("coverage extra train states: %d" % promoted)
    print("Read first:")
    print("  %s" % (output_dir / "coverage_summary.csv"))
    print("  %s" % (output_dir / "holdout_to_train_coverage.csv"))
    print("  %s" % (output_dir / "coverage_aware_split_parameter_support.png"))
    print("=" * 100)


if __name__ == "__main__":
    main()
