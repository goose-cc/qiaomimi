#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp30 step 2: construct train/val/test inside an alias-filtered identifiable region.

Three simultaneous constraints are enforced:

1) CONTINUOUS ALIAS EXCLUSION
   candidate remote_alias_rms_snr >= alias_min_rms_snr

2) SELECTED-BANK SEPARATION
   every retained physical truth is at least min_rms_snr from the other retained
   truths in observed g-space

3) HOLDOUT COVERAGE
   every val/test truth has a TRAIN truth within max_train_cover_rms_snr

The forward formula, q2 observations, network and loss are not changed.
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
from data_generate_exp22c_fixed_capacity import slim_for_training
from data_generate_exp26_gseparated_independent import (
    count_per_state,
    make_split_arrays_from_bank,
    rms_snr_to_one,
    selected_nearest_rows,
    pairwise_split_summary,
)


def parse_float_list(text):
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return vals


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate Exp30 alias-filtered identifiable-region data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pool-dir", default="data_exp30_alias_pool")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--alias-min-rms-snr", type=float, required=True)
    p.add_argument("--min-rms-snr", type=float, default=1.3)
    p.add_argument("--max-train-cover-rms-snr", type=float, default=2.0)
    p.add_argument("--max-selected-states", type=int, default=260)
    p.add_argument("--min-required-states", type=int, default=30)
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--split-gamma-bins", type=int, default=8)
    p.add_argument("--noise-levels", type=parse_float_list, default=[0.0, 0.002, 0.01])
    p.add_argument("--train-noise-level", type=float, default=0.002)
    p.add_argument("--target-train-samples", type=int, default=20000)
    p.add_argument("--target-val-samples", type=int, default=3000)
    p.add_argument("--target-test-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260830)
    p.add_argument("--compressed", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def reconstruct_candidates(pool):
    a1 = np.asarray(pool["candidate_a1"], dtype=np.float64)
    gamma = np.asarray(pool["candidate_gamma"], dtype=np.float64)
    gi = np.asarray(pool["candidate_gamma_index"], dtype=np.int64)
    B = np.asarray(pool["B_master"], dtype=np.float64)
    R = np.asarray(pool["R_master"], dtype=np.float64)
    g_master = B[None, :] + a1[:, None] * R[gi]
    params = np.zeros((len(a1), 5), dtype=np.float64)
    params[:, 0] = a1
    params[:, 1] = FIXED_A2
    params[:, 2] = FIXED_A3
    params[:, 3] = FIXED_M
    params[:, 4] = gamma
    return params, g_master, gi


def greedy_pack_no_forced_corners(
    g_obs, master_rms, reference_noise, alias_score, min_rms_snr, max_selected
):
    n = len(master_rms)
    if n == 0:
        return np.empty(0, dtype=np.int32), [], False

    first = int(np.argmax(alias_score))
    selected = [first]
    active = np.ones(n, dtype=bool)
    active[first] = False
    min_sep = np.full(n, np.inf, dtype=np.float64)
    min_sep[first] = 0.0

    idx = np.where(active)[0]
    if len(idx):
        min_sep[idx] = rms_snr_to_one(
            g_obs, master_rms, reference_noise, first, idx
        )

    trace = [{
        "selection_order": 0,
        "candidate_local_index": first,
        "maximin_rms_snr_at_selection": math.inf,
        "alias_score": float(alias_score[first]),
    }]
    hit_cap = False

    while True:
        if len(selected) >= int(max_selected):
            hit_cap = True
            break
        masked = np.where(active, min_sep, -np.inf)
        j = int(np.argmax(masked))
        s = float(masked[j])
        if not np.isfinite(s) or s < float(min_rms_snr):
            break
        selected.append(j)
        active[j] = False
        min_sep[j] = 0.0
        trace.append({
            "selection_order": len(selected) - 1,
            "candidate_local_index": j,
            "maximin_rms_snr_at_selection": s,
            "alias_score": float(alias_score[j]),
        })
        idx = np.where(active)[0]
        if len(idx):
            d = rms_snr_to_one(
                g_obs, master_rms, reference_noise, j, idx
            )
            min_sep[idx] = np.minimum(min_sep[idx], d)

    return np.asarray(selected, dtype=np.int32), trace, hit_cap


def pairwise_snr_matrix(g_obs, master_rms, reference_noise):
    n = len(master_rms)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        d[i] = rms_snr_to_one(
            g_obs, master_rms, reference_noise, i, np.arange(n, dtype=np.int64)
        )
        d[i, i] = 0.0
    return d


def target_counts(n, train_fraction, val_fraction, test_fraction):
    f = np.asarray([train_fraction, val_fraction, test_fraction], dtype=np.float64)
    if np.any(f <= 0) or not np.isclose(np.sum(f), 1.0, atol=1e-8):
        raise ValueError("train/val/test fractions must be positive and sum to 1")
    raw = f * int(n)
    c = np.floor(raw).astype(int)
    rem = int(n) - int(np.sum(c))
    for k in np.argsort(-(raw - c))[:rem]:
        c[k] += 1
    return {"train": int(c[0]), "val": int(c[1]), "test": int(c[2])}


def choose_train_cover(dmat, alias_score, target_train, max_cover):
    n = len(dmat)
    first = int(np.argmax(alias_score))
    train = [first]
    remaining = set(range(n)) - {first}
    nearest = dmat[:, first].copy()
    nearest[first] = 0.0

    # Farthest-first gives train a broad g-space cover.
    while len(train) < int(target_train):
        j = max(remaining, key=lambda x: float(nearest[x]))
        train.append(int(j))
        remaining.remove(j)
        nearest = np.minimum(nearest, dmat[:, j])
        nearest[train] = 0.0

    promoted = 0
    while remaining:
        j = max(remaining, key=lambda x: float(nearest[x]))
        if float(nearest[j]) <= float(max_cover) + 1e-12:
            break
        train.append(int(j))
        remaining.remove(j)
        promoted += 1
        nearest = np.minimum(nearest, dmat[:, j])
        nearest[train] = 0.0

    return np.asarray(sorted(train), dtype=np.int32), nearest, promoted


def stratified_val_test(remaining, params, n_val, n_test, gamma_bins, seed):
    remaining = np.asarray(remaining, dtype=np.int32)
    if len(remaining) != int(n_val) + int(n_test):
        raise ValueError("remaining count mismatch")
    rng = np.random.default_rng(int(seed) + 3001)
    logg = np.log10(params[:, 4])
    edges = np.linspace(logg.min(), logg.max(), int(gamma_bins) + 1)
    bid = np.clip(
        np.digitize(logg, edges[1:-1], right=False),
        0,
        int(gamma_bins) - 1,
    )
    val, test = [], []
    frac_val = float(n_val) / float(max(n_val + n_test, 1))
    for b in range(int(gamma_bins)):
        idx = remaining[bid[remaining] == b].copy()
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        nv = int(round(frac_val * len(idx)))
        if len(idx) >= 2:
            nv = min(max(nv, 1), len(idx) - 1)
        else:
            nv = 1 if len(val) < n_val else 0
        val.extend(int(x) for x in idx[:nv])
        test.extend(int(x) for x in idx[nv:])

    def move_one(src, dst):
        if not src:
            raise RuntimeError("cannot rebalance holdouts")
        dst.append(src.pop())

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
        s = dmat[i, train_idx]
        k = int(np.argmin(s))
        j = int(train_idx[k])
        rows.append({
            "state_index": int(i),
            "split": str(split_labels[i]),
            "a1": float(params[i, 0]),
            "gamma": float(params[i, 4]),
            "nearest_train_state": j,
            "nearest_train_a1": float(params[j, 0]),
            "nearest_train_gamma": float(params[j, 4]),
            "nearest_train_rms_snr": float(s[k]),
        })
    return rows


def summarize_coverage(rows):
    df = pd.DataFrame(rows)
    out = []
    for name in ("val", "test", "holdout_all"):
        x = df if name == "holdout_all" else df[df["split"] == name]
        a = x["nearest_train_rms_snr"].to_numpy(float)
        out.append({
            "split": name,
            "count": int(len(a)),
            "min_nearest_train_rms_snr": float(np.min(a)),
            "median_nearest_train_rms_snr": float(np.median(a)),
            "p90_nearest_train_rms_snr": float(np.quantile(a, 0.9)),
            "max_nearest_train_rms_snr": float(np.max(a)),
        })
    return out


def plot_data(output_dir, params, split_labels, alias_scores, coverage, alias_min):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ("train", "val", "test"):
        idx = np.where(split_labels == name)[0]
        ax.scatter(params[idx, 0], params[idx, 4], s=35, alpha=0.8, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("a1")
    ax.set_ylabel("gamma")
    ax.set_title("Exp30 alias-filtered identifiable region")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "selected_parameter_support.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(alias_scores, bins=25)
    ax.axvline(float(alias_min), linestyle="--", label="required alias minimum")
    ax.set_xlabel("continuous remote-alias RMS-SNR")
    ax.set_ylabel("selected physical state count")
    ax.set_title("Exp30 selected-state continuous alias safety")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "selected_alias_score_hist.png", dpi=170)
    plt.close(fig)

    cdf = pd.DataFrame(coverage)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ("val", "test"):
        x = cdf[cdf["split"] == name]["nearest_train_rms_snr"]
        ax.hist(x, bins=15, alpha=0.55, label=name)
    ax.set_xlabel("holdout -> nearest train RMS-SNR")
    ax.set_ylabel("physical state count")
    ax.set_title("Exp30 coverage after alias exclusion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "holdout_coverage_hist.png", dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    pool_dir = Path(args.pool_dir)
    output_dir = Path(args.output_dir)

    pool_meta = json.loads((pool_dir / "metadata.json").read_text(encoding="utf-8"))
    pool = np.load(pool_dir / "alias_pool.npz", allow_pickle=False)
    params_all, g_master_all, _ = reconstruct_candidates(pool)
    alias_all = np.asarray(pool["remote_alias_rms_snr"], dtype=np.float64)
    alias_a1_all = np.asarray(pool["alias_a1"], dtype=np.float64)
    alias_gamma_all = np.asarray(pool["alias_gamma"], dtype=np.float64)
    obs_idx = np.asarray(pool["observation_indices"], dtype=np.int64)

    mask = alias_all >= float(args.alias_min_rms_snr)
    safe_idx = np.where(mask)[0]
    if len(safe_idx) < int(args.min_required_states):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "insufficient_support.json").write_text(
            json.dumps({
                "reason": "too few continuous-alias-safe candidates",
                "alias_min_rms_snr": float(args.alias_min_rms_snr),
                "safe_candidate_count": int(len(safe_idx)),
                "min_required_states": int(args.min_required_states),
            }, indent=2),
            encoding="utf-8",
        )
        print("insufficient alias-safe candidates: %d" % len(safe_idx))
        return

    params_safe = params_all[safe_idx]
    g_master_safe = g_master_all[safe_idx].astype(np.float32)
    alias_safe = alias_all[safe_idx]
    g_obs_safe = g_master_safe[:, obs_idx]
    rms_safe = np.sqrt(np.mean(g_master_safe.astype(np.float64) ** 2, axis=1))
    ref_noise = float(pool_meta["reference_noise"])

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError("%s is not empty; use --overwrite" % output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    packed_local, trace, hit_cap = greedy_pack_no_forced_corners(
        g_obs_safe, rms_safe, ref_noise, alias_safe,
        args.min_rms_snr, args.max_selected_states,
    )
    if len(packed_local) < int(args.min_required_states):
        (output_dir / "insufficient_support.json").write_text(
            json.dumps({
                "reason": "too few states after selected-bank g-separation packing",
                "alias_min_rms_snr": float(args.alias_min_rms_snr),
                "safe_candidate_count": int(len(safe_idx)),
                "selected_state_count": int(len(packed_local)),
                "min_required_states": int(args.min_required_states),
            }, indent=2),
            encoding="utf-8",
        )
        print("insufficient selected states after packing: %d" % len(packed_local))
        return

    selected_pool_idx = safe_idx[packed_local]
    params = params_all[selected_pool_idx]
    g_master = g_master_all[selected_pool_idx].astype(np.float32)
    alias_score = alias_all[selected_pool_idx]
    alias_a1 = alias_a1_all[selected_pool_idx]
    alias_gamma = alias_gamma_all[selected_pool_idx]
    g_obs = g_master[:, obs_idx]
    master_rms = np.sqrt(np.mean(g_master.astype(np.float64) ** 2, axis=1))

    nearest_rows, nearest_scores, _ = selected_nearest_rows(
        params, g_obs, master_rms, ref_noise
    )
    actual_min_sep = float(np.min(nearest_scores))

    dmat = pairwise_snr_matrix(g_obs, master_rms, ref_noise)
    tc = target_counts(
        len(params), args.train_fraction, args.val_fraction, args.test_fraction
    )
    train_idx, _, promoted = choose_train_cover(
        dmat, alias_score, tc["train"], args.max_train_cover_rms_snr
    )
    remaining = np.asarray(
        sorted(set(range(len(params))) - set(int(x) for x in train_idx)),
        dtype=np.int32,
    )

    if len(train_idx) != tc["train"]:
        hold = len(remaining)
        val_ratio = float(tc["val"]) / float(max(tc["val"] + tc["test"], 1))
        n_val = int(round(val_ratio * hold))
        n_val = min(max(n_val, 1), hold - 1)
        n_test = hold - n_val
    else:
        n_val, n_test = tc["val"], tc["test"]

    if n_val < 5 or n_test < 5:
        (output_dir / "insufficient_support.json").write_text(
            json.dumps({
                "reason": "coverage constraint leaves too few independent val/test states",
                "selected_state_count": int(len(params)),
                "train_state_count": int(len(train_idx)),
                "val_state_count": int(n_val),
                "test_state_count": int(n_test),
            }, indent=2),
            encoding="utf-8",
        )
        print("coverage leaves too few holdout states")
        return

    val_idx, test_idx = stratified_val_test(
        remaining, params, n_val, n_test, args.split_gamma_bins, args.seed
    )
    split_labels = np.full(len(params), "", dtype=object)
    split_labels[train_idx] = "train"
    split_labels[val_idx] = "val"
    split_labels[test_idx] = "test"

    coverage = coverage_rows(params, split_labels, dmat)
    coverage_summary = summarize_coverage(coverage)
    max_cover = max(
        r["max_nearest_train_rms_snr"] for r in coverage_summary
        if r["split"] == "holdout_all"
    )
    if max_cover > float(args.max_train_cover_rms_snr) + 1e-8:
        raise RuntimeError("coverage guarantee failed")

    for i, row in enumerate(nearest_rows):
        row["split"] = str(split_labels[i])
        row["pool_candidate_index"] = int(selected_pool_idx[i])
        row["remote_alias_rms_snr"] = float(alias_score[i])
        row["remote_alias_a1"] = float(alias_a1[i])
        row["remote_alias_gamma"] = float(alias_gamma[i])
    write_csv(output_dir / "selected_states.csv", nearest_rows)
    write_csv(
        output_dir / "split_pair_gseparation_summary.csv",
        pairwise_split_summary(
            params, g_obs, master_rms, split_labels, ref_noise
        ),
    )
    write_csv(output_dir / "holdout_to_train_coverage.csv", coverage)
    write_csv(output_dir / "coverage_summary.csv", coverage_summary)
    write_csv(output_dir / "selection_trace.csv", trace)
    plot_data(
        output_dir, params, split_labels, alias_score, coverage,
        args.alias_min_rms_snr,
    )

    physics = replace(
        DEFAULT_PHYSICS, q2_points=int(pool_meta["model_input_points"])
    )
    selected_fx, direct_g = forward_bank(
        params,
        physics=physics,
        integration_points=int(pool_meta["integration_points"]),
    )
    # Verify basis reconstruction against the unchanged physical forward.
    basis_rel = np.linalg.norm(
        np.asarray(direct_g, dtype=np.float64) - g_master.astype(np.float64),
        axis=1,
    ) / np.maximum(
        np.linalg.norm(np.asarray(direct_g, dtype=np.float64), axis=1),
        1e-30,
    )
    if float(np.max(basis_rel)) > 5e-5:
        raise RuntimeError("selected direct-forward consistency check failed")

    try:
        from mc_physics import output_grids_numpy
        s_grid, q_master = output_grids_numpy(physics)
    except Exception:
        q_master = np.linspace(
            physics.q2_min, physics.q2_max, physics.q2_points, dtype=np.float64
        )
        s_grid = np.arange(selected_fx.shape[1], dtype=np.float64)

    split_counts = {
        name: int(np.sum(split_labels == name))
        for name in ("train", "val", "test")
    }
    targets = {
        "train": int(args.target_train_samples),
        "val": int(args.target_val_samples),
        "test": int(args.target_test_samples),
    }
    per_state = {
        name: count_per_state(targets[name], split_counts[name])
        for name in ("train", "val", "test")
    }

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
                fx_bank=selected_fx[idx],
                g_master_bank=g_master[idx],
                count_per_state=per_state[split_name],
                noise_level=float(noise_level),
                q_master=np.asarray(q_master, dtype=np.float64),
                obs_idx=obs_idx,
                seed=args.seed,
            )
            arrays["x"] = np.asarray(s_grid, dtype=np.float32)
            if split_name in ("train", "val"):
                arrays = slim_for_training(arrays)
            save_npz(ndir / (split_name + ".npz"), arrays, args.compressed)

    metadata = {
        "experiment": "Exp30 continuous alias-filtered identifiable region",
        "mode": "a1gamma",
        "second_parameter": "a1",
        "fixed_parameters": {"a2": FIXED_A2, "a3": FIXED_A3, "m": FIXED_M},
        "pool_dir": str(pool_dir),
        "pool_candidate_count": int(len(params_all)),
        "alias_safe_candidate_count": int(len(safe_idx)),
        "alias_min_rms_snr": float(args.alias_min_rms_snr),
        "requested_min_rms_snr": float(args.min_rms_snr),
        "global_min_rms_snr": actual_min_sep,
        "max_train_cover_rms_snr": float(args.max_train_cover_rms_snr),
        "coverage_promoted_extra_train_states": int(promoted),
        "selected_state_count": int(len(params)),
        "selected_state_counts": split_counts,
        "selected_alias_score_min": float(np.min(alias_score)),
        "selected_alias_score_median": float(np.median(alias_score)),
        "selected_alias_score_max": float(np.max(alias_score)),
        "a1_range": [float(np.min(params[:, 0])), float(np.max(params[:, 0]))],
        "gamma_range": [float(np.min(params[:, 4])), float(np.max(params[:, 4]))],
        "original_a1_range": pool_meta["a1_range"],
        "original_gamma_range": pool_meta["gamma_range"],
        "reference_noise": ref_noise,
        "physical_observation_points": int(pool_meta["physical_observation_points"]),
        "model_input_points": int(pool_meta["model_input_points"]),
        "observation_indices": [int(x) for x in obs_idx],
        "noise_levels": [float(x) for x in args.noise_levels],
        "train_noise_level": float(args.train_noise_level),
        "per_state_sample_counts": per_state,
        "actual_sample_counts": {
            name: int(per_state[name] * split_counts[name])
            for name in ("train", "val", "test")
        },
        "coverage_summary": coverage_summary,
        "direct_forward_consistency_max_rel_l2": float(np.max(basis_rel)),
        "integration_points": int(pool_meta["integration_points"]),
        "seed": int(args.seed),
        "interpretation": (
            "Every retained truth passes continuous remote-alias exclusion, "
            "selected-bank pairwise g-separation, and train-coverage constraints."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("Exp30 identifiable-region data ready")
    print("alias-safe candidates    : %d / %d" % (len(safe_idx), len(params_all)))
    print("selected physical states : %d" % len(params))
    print("split counts             : %s" % split_counts)
    print("selected alias score min : %.5g" % np.min(alias_score))
    print("global min g RMS-SNR     : %.5g" % actual_min_sep)
    print("max holdout->train SNR   : %.5g" % max_cover)
    print("=" * 100)


if __name__ == "__main__":
    main()
