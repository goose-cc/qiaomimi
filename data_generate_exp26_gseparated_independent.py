#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp26: construct independent train/val/test states that are globally separated in g-space.

This experiment changes the DATA SUPPORT, not the physical formula and not the network.

Workflow
--------
1. Generate a large continuous candidate pool in (a1, log10(gamma)).
2. Compute the physical forward g(q^2) for every candidate.
3. Use farthest-point / max-min packing in observed g-space.
4. Keep only states whose pairwise RMS-SNR separation is at least MinRMSSNR.
5. Split the selected physical states into disjoint train / val / test sets.
6. Generate independent noisy realizations for each split.

Because the separation packing is GLOBAL before the split, the minimum
g-separation guarantee also holds across train-vs-val, train-vs-test, and
val-vs-test.  Therefore no two retained physical truths are near-duplicates in
g-space.

The selected states are intentionally irregular in parameter space; this is no
longer an a1 x gamma Cartesian rectangular grid.
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
    forward_bank,
    noise_dir_name,
    save_npz,
    write_csv,
)
from data_generate_exp22c_fixed_capacity import (
    canonical_indices,
    interpolation_plan,
    interpolate_rows,
    slim_for_training,
)


def parse_float_list(text):
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def parse_args():
    p = argparse.ArgumentParser(
        description="Exp26 globally g-separated independent train/val/test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default="data_exp26_gsep_snr1")
    p.add_argument("--candidate-count", type=int, default=6000)
    p.add_argument("--max-selected-states", type=int, default=260)
    p.add_argument("--min-rms-snr", type=float, default=1.0)
    p.add_argument("--reference-noise", type=float, default=0.002)
    p.add_argument("--a1-min", type=float, default=0.05)
    p.add_argument("--a1-max", type=float, default=0.20)
    p.add_argument("--gamma-min", type=float, default=0.01)
    p.add_argument("--gamma-max", type=float, default=1.0)
    p.add_argument("--observation-points", type=int, default=500)
    p.add_argument("--model-input-points", type=int, default=1000)
    p.add_argument("--integration-points", type=int, default=128)
    p.add_argument("--noise-levels", type=parse_float_list, default=[0.0, 0.002, 0.01])
    p.add_argument("--train-noise-level", type=float, default=0.002)

    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--split-gamma-bins", type=int, default=8)

    p.add_argument("--target-train-samples", type=int, default=20000)
    p.add_argument("--target-val-samples", type=int, default=3000)
    p.add_argument("--target-test-samples", type=int, default=5000)

    p.add_argument("--seed", type=int, default=20260815)
    p.add_argument("--compressed", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def latin_hypercube_1d(n, rng):
    x = (np.arange(n, dtype=np.float64) + rng.random(n)) / float(n)
    rng.shuffle(x)
    return x


def candidate_parameters(args):
    """Continuous LHS candidates plus four exact parameter-domain corners."""
    n = int(args.candidate_count)
    if n < 100:
        raise ValueError("--candidate-count should be >= 100")
    rng = np.random.default_rng(int(args.seed))
    u = latin_hypercube_1d(n, rng)
    v = latin_hypercube_1d(n, rng)
    a1 = args.a1_min + (args.a1_max - args.a1_min) * u
    lg0 = math.log10(args.gamma_min)
    lg1 = math.log10(args.gamma_max)
    gamma = np.power(10.0, lg0 + (lg1 - lg0) * v)

    # Exact corners ensure the candidate support spans the requested domain.
    a1[:4] = [args.a1_min, args.a1_min, args.a1_max, args.a1_max]
    gamma[:4] = [args.gamma_min, args.gamma_max, args.gamma_min, args.gamma_max]

    params = np.zeros((n, 5), dtype=np.float64)
    params[:, 0] = a1
    params[:, 1] = FIXED_A2
    params[:, 2] = FIXED_A3
    params[:, 3] = FIXED_M
    params[:, 4] = gamma
    return params


def rms_snr_to_one(g_obs, master_rms, reference_noise, selected_index, candidate_indices):
    idx = np.asarray(candidate_indices, dtype=np.int64)
    a = np.asarray(g_obs[idx], dtype=np.float64)
    b = np.asarray(g_obs[int(selected_index)], dtype=np.float64)
    d = a - b[None, :]
    rms_d = np.sqrt(np.mean(d * d, axis=1))
    sigma = float(reference_noise) * 0.5 * (
        np.asarray(master_rms[idx], dtype=np.float64) + float(master_rms[int(selected_index)])
    )
    return rms_d / np.maximum(sigma, 1e-30)


def greedy_pack(g_obs, master_rms, reference_noise, min_rms_snr, max_selected):
    n = len(master_rms)
    seed_states = [0, 1, 2, 3]

    # Corners must themselves satisfy the requested separation.
    for ii in range(len(seed_states)):
        for jj in range(ii + 1, len(seed_states)):
            score = rms_snr_to_one(
                g_obs, master_rms, reference_noise,
                seed_states[ii], [seed_states[jj]]
            )[0]
            if score < min_rms_snr:
                raise ValueError(
                    "requested MinRMSSNR %.4g is larger than a required corner-pair "
                    "separation %.4g; lower the threshold" % (min_rms_snr, score)
                )

    selected = list(seed_states)
    active = np.ones(n, dtype=bool)
    active[selected] = False
    min_sep = np.full(n, np.inf, dtype=np.float64)
    min_sep[selected] = 0.0

    for s in selected:
        idx = np.where(active)[0]
        if len(idx):
            scores = rms_snr_to_one(g_obs, master_rms, reference_noise, s, idx)
            min_sep[idx] = np.minimum(min_sep[idx], scores)

    trace = []
    for order, s in enumerate(selected):
        trace.append({
            "selection_order": order,
            "candidate_index": int(s),
            "maximin_rms_snr_at_selection": math.inf,
            "is_forced_corner": 1,
        })

    hit_cap = False
    while True:
        if len(selected) >= int(max_selected):
            hit_cap = True
            break
        masked = np.where(active, min_sep, -np.inf)
        j = int(np.argmax(masked))
        score = float(masked[j])
        if not np.isfinite(score) or score < float(min_rms_snr):
            break

        selected.append(j)
        active[j] = False
        min_sep[j] = 0.0
        trace.append({
            "selection_order": len(selected) - 1,
            "candidate_index": j,
            "maximin_rms_snr_at_selection": score,
            "is_forced_corner": 0,
        })

        idx = np.where(active)[0]
        if len(idx):
            scores = rms_snr_to_one(g_obs, master_rms, reference_noise, j, idx)
            min_sep[idx] = np.minimum(min_sep[idx], scores)

    return np.asarray(selected, dtype=np.int32), trace, hit_cap, min_sep


def selected_nearest_rows(params, g_obs, master_rms, reference_noise):
    n = len(params)
    rows = []
    all_scores = []
    nearest_idx = []
    for i in range(n):
        idx = np.asarray([j for j in range(n) if j != i], dtype=np.int64)
        scores = rms_snr_to_one(g_obs, master_rms, reference_noise, i, idx)
        k = int(np.argmin(scores))
        j = int(idx[k])
        s = float(scores[k])
        all_scores.append(s)
        nearest_idx.append(j)
        d = np.asarray(g_obs[i], dtype=np.float64) - np.asarray(g_obs[j], dtype=np.float64)
        rows.append({
            "state_index": i,
            "a1": float(params[i, 0]),
            "gamma": float(params[i, 4]),
            "nearest_state": j,
            "nearest_a1": float(params[j, 0]),
            "nearest_gamma": float(params[j, 4]),
            "snr_sep_rms": s,
            "rms_g_difference": float(np.sqrt(np.mean(d * d))),
        })
    return rows, np.asarray(all_scores, dtype=np.float64), np.asarray(nearest_idx, dtype=np.int32)


def stratified_partition(params, seed, train_fraction, val_fraction, test_fraction, gamma_bins):
    fractions = np.asarray([train_fraction, val_fraction, test_fraction], dtype=np.float64)
    if np.any(fractions <= 0) or not np.isclose(np.sum(fractions), 1.0, atol=1e-8):
        raise ValueError("train/val/test fractions must be positive and sum to 1")

    n = len(params)
    split = np.full(n, "", dtype=object)

    # Force the four physical-domain corners into train so train covers the
    # requested parameter boundaries. They are candidates 0..3 and remain the
    # first four selected states.
    split[:4] = "train"

    rest = np.arange(4, n, dtype=np.int64)
    logg = np.log10(params[:, 4])
    edges = np.linspace(logg.min(), logg.max(), int(gamma_bins) + 1)
    bin_id = np.clip(np.digitize(logg, edges[1:-1], right=False), 0, int(gamma_bins) - 1)

    rng = np.random.default_rng(int(seed) + 991)
    for b in range(int(gamma_bins)):
        idx = rest[bin_id[rest] == b]
        if len(idx) == 0:
            continue
        idx = idx.copy()
        rng.shuffle(idx)

        # Largest-remainder allocation, with val/test coverage whenever possible.
        raw = fractions * len(idx)
        counts = np.floor(raw).astype(int)
        remainder = len(idx) - int(np.sum(counts))
        if remainder > 0:
            order = np.argsort(-(raw - counts))
            for k in order[:remainder]:
                counts[k] += 1

        if len(idx) >= 3:
            for k in (1, 2):
                if counts[k] == 0:
                    donor = int(np.argmax(counts))
                    if counts[donor] > 1:
                        counts[donor] -= 1
                        counts[k] += 1

        a = int(counts[0])
        v = int(counts[1])
        split[idx[:a]] = "train"
        split[idx[a:a+v]] = "val"
        split[idx[a+v:]] = "test"

    if np.any(split == ""):
        raise RuntimeError("some selected states were not assigned to a split")
    return split


def split_pair_min(rows, split_labels, left, right):
    """Minimum g-separation between two split labels using the selected-state bank."""
    # rows only store nearest overall, so compute from the raw selected bank elsewhere.
    return None


def pairwise_split_summary(params, g_obs, master_rms, split_labels, reference_noise):
    names = ("train", "val", "test")
    rows = []
    for ia, a in enumerate(names):
        idx_a = np.where(split_labels == a)[0]
        for ib, b in enumerate(names):
            if ib < ia:
                continue
            idx_b = np.where(split_labels == b)[0]
            scores = []
            for i in idx_a:
                js = idx_b
                if a == b:
                    js = js[js != i]
                if len(js) == 0:
                    continue
                s = rms_snr_to_one(g_obs, master_rms, reference_noise, int(i), js)
                scores.extend(np.asarray(s, dtype=np.float64).tolist())
            arr = np.asarray(scores, dtype=np.float64)
            rows.append({
                "split_1": a,
                "split_2": b,
                "pair_count": int(len(arr)),
                "min_rms_snr": float(np.min(arr)) if len(arr) else math.nan,
                "median_rms_snr": float(np.median(arr)) if len(arr) else math.nan,
            })
    return rows


def count_per_state(target, n_states):
    return max(1, int(round(float(target) / float(n_states))))


def make_split_arrays_from_bank(
    *,
    split_name,
    params,
    fx_bank,
    g_master_bank,
    count_per_state,
    noise_level,
    q_master,
    obs_idx,
    seed,
):
    q_obs = np.asarray(q_master[obs_idx], dtype=np.float64)
    lo, hi, w = interpolation_plan(q_obs, np.asarray(q_master, dtype=np.float64))

    blocks = {
        "fx": [], "gy_clean": [], "gy_noisy": [], "parameters": [],
        "a1": [], "a2": [], "a3": [], "m": [], "gamma": [],
        "noise_level": [], "noise_sigma": [], "state_index": [],
    }
    split_code = {"train": 11, "val": 22, "test": 33}[str(split_name)]

    for i in range(len(params)):
        clean_master = np.asarray(g_master_bank[i], dtype=np.float32)
        clean_obs = clean_master[obs_idx]
        rms = float(np.sqrt(np.mean(clean_master.astype(np.float64) ** 2)))
        sigma = float(noise_level) * rms
        rng = np.random.default_rng(
            int(seed) + split_code * 1_000_003 + (i + 1) * 100_003
        )
        if noise_level > 0:
            z = rng.standard_normal((count_per_state, len(obs_idx))).astype(np.float32)
            noisy_obs = clean_obs[None, :] + np.float32(sigma) * z
        else:
            noisy_obs = np.repeat(clean_obs[None, :], count_per_state, axis=0)
        noisy_model = interpolate_rows(noisy_obs, lo, hi, w)

        p = np.asarray(params[i], dtype=np.float32)
        blocks["fx"].append(np.repeat(fx_bank[i][None, :], count_per_state, axis=0))
        blocks["gy_clean"].append(np.repeat(clean_master[None, :], count_per_state, axis=0))
        blocks["gy_noisy"].append(noisy_model.astype(np.float32))
        blocks["parameters"].append(np.repeat(p[None, :], count_per_state, axis=0))
        for key, value in (
            ("a1", p[0]), ("a2", p[1]), ("a3", p[2]), ("m", p[3]), ("gamma", p[4]),
            ("noise_level", noise_level), ("noise_sigma", sigma), ("state_index", i),
        ):
            dtype = np.int32 if key == "state_index" else np.float32
            blocks[key].append(np.full(count_per_state, value, dtype=dtype))

    out = {k: np.concatenate(v, axis=0) for k, v in blocks.items()}
    perm = np.random.default_rng(
        int(seed) + split_code * 9_999_991 + int(round(noise_level * 1e9))
    ).permutation(len(out["gamma"]))
    for k in list(out):
        out[k] = out[k][perm]
    out["q2"] = np.asarray(q_master, dtype=np.float32)
    out["y"] = np.asarray(q_master, dtype=np.float32)
    # fx grid is not needed by the trainer; preserve a simple index grid if x is absent.
    out["q2_observed"] = np.asarray(q_obs, dtype=np.float32)
    out["observation_indices"] = np.asarray(obs_idx, dtype=np.int32)
    out["observation_points"] = np.asarray([len(obs_idx)], dtype=np.int32)
    out["model_input_points"] = np.asarray([len(q_master)], dtype=np.int32)
    return out


def plot_selected_states(output_dir, params, split_labels, nearest_scores, min_rms_snr):
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
        ax.scatter(params[idx, 0], params[idx, 4], s=30, alpha=0.8, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("a1")
    ax.set_ylabel("gamma")
    ax.set_title("Exp26 irregular globally g-separated physical states")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "selected_states_by_split.png", dpi=165)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(nearest_scores, bins=24)
    ax.axvline(float(min_rms_snr), linestyle="--", label="required minimum")
    ax.set_xlabel("nearest-state RMS-SNR in g-space")
    ax.set_ylabel("state count")
    ax.set_title("Exp26 selected-state g-separation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "selected_state_gseparation_hist.png", dpi=165)
    plt.close(fig)


def main():
    args = parse_args()
    if not (0 < args.a1_min < args.a1_max):
        raise ValueError("require 0 < a1_min < a1_max")
    if not (0 < args.gamma_min < args.gamma_max):
        raise ValueError("require 0 < gamma_min < gamma_max")
    if args.min_rms_snr <= 0 or args.reference_noise <= 0:
        raise ValueError("separation/noise parameters must be positive")
    if not (2 <= args.observation_points <= args.model_input_points):
        raise ValueError("require 2 <= observation_points <= model_input_points")
    if not any(np.isclose(args.train_noise_level, x, atol=1e-12, rtol=0)
               for x in args.noise_levels):
        raise ValueError("train noise must be included in noise levels")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError("%s is not empty; use --overwrite" % output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    physics = replace(DEFAULT_PHYSICS, q2_points=int(args.model_input_points))
    obs_idx = canonical_indices(args.model_input_points, args.observation_points)

    print("=" * 100)
    print("Exp26 globally g-separated independent states")
    print("candidate states       : %d" % args.candidate_count)
    print("required min RMS-SNR   : %.4g" % args.min_rms_snr)
    print("reference noise        : %.4g (%.3f%%)" % (args.reference_noise, 100*args.reference_noise))
    print("physical q2 observations: %d" % args.observation_points)
    print("model input width      : %d" % args.model_input_points)
    print("=" * 100)

    candidates = candidate_parameters(args)
    _, g_master_candidates = forward_bank(
        candidates, physics=physics, integration_points=args.integration_points
    )
    g_obs_candidates = np.asarray(g_master_candidates[:, obs_idx], dtype=np.float32)
    candidate_rms = np.sqrt(
        np.mean(np.asarray(g_master_candidates, dtype=np.float64) ** 2, axis=1)
    )

    selected_idx, trace, hit_cap, _ = greedy_pack(
        g_obs_candidates,
        candidate_rms,
        args.reference_noise,
        args.min_rms_snr,
        args.max_selected_states,
    )
    if len(selected_idx) < 30:
        raise RuntimeError(
            "only %d globally separated states were found. Lower --min-rms-snr "
            "or increase the candidate pool." % len(selected_idx)
        )

    selected_params = np.asarray(candidates[selected_idx], dtype=np.float64)
    selected_g_master = np.asarray(g_master_candidates[selected_idx], dtype=np.float32)
    selected_g_obs = selected_g_master[:, obs_idx]
    selected_rms = candidate_rms[selected_idx]

    nearest_rows, nearest_scores, nearest_idx = selected_nearest_rows(
        selected_params, selected_g_obs, selected_rms, args.reference_noise
    )
    actual_min = float(np.min(nearest_scores))
    if actual_min + 1e-8 < args.min_rms_snr:
        raise RuntimeError(
            "internal selection error: actual min RMS-SNR %.6g < requested %.6g"
            % (actual_min, args.min_rms_snr)
        )

    split_labels = stratified_partition(
        selected_params,
        args.seed,
        args.train_fraction,
        args.val_fraction,
        args.test_fraction,
        args.split_gamma_bins,
    )
    split_summary_rows = pairwise_split_summary(
        selected_params, selected_g_obs, selected_rms, split_labels, args.reference_noise
    )

    for i, row in enumerate(nearest_rows):
        row["split"] = str(split_labels[i])
        row["candidate_index"] = int(selected_idx[i])
    write_csv(output_dir / "selected_states.csv", nearest_rows)
    write_csv(output_dir / "split_pair_gseparation_summary.csv", split_summary_rows)
    write_csv(output_dir / "selection_trace.csv", trace)
    plot_selected_states(
        output_dir, selected_params, split_labels, nearest_scores, args.min_rms_snr
    )

    split_counts = {name: int(np.sum(split_labels == name)) for name in ("train", "val", "test")}
    if split_counts["val"] < 5 or split_counts["test"] < 5:
        raise RuntimeError("validation/test physical state count is too small: %s" % split_counts)

    # Compute selected f only now; candidate f was intentionally discarded.
    selected_fx, _ = forward_bank(
        selected_params, physics=physics, integration_points=args.integration_points
    )
    try:
        from mc_physics import output_grids_numpy
        s_grid, q_master = output_grids_numpy(physics)
    except Exception:
        q_master = np.linspace(physics.q2_min, physics.q2_max, physics.q2_points, dtype=np.float64)
        s_grid = np.arange(selected_fx.shape[1], dtype=np.float64)

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
            # Train/val are only needed at the training noise.  Test is generated
            # at every requested noise for robustness checks.
            if split_name != "test" and not np.isclose(
                noise_level, args.train_noise_level, atol=1e-12, rtol=0
            ):
                continue
            idx = np.where(split_labels == split_name)[0]
            arrays = make_split_arrays_from_bank(
                split_name=split_name,
                params=selected_params[idx],
                fx_bank=selected_fx[idx],
                g_master_bank=selected_g_master[idx],
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
            print(
                "saved %-5s noise=%7.4g | physical states=%3d | samples=%6d"
                % (split_name, noise_level, len(idx), len(arrays["gamma"]))
            )

    metadata = {
        "experiment": "Exp26 globally g-separated independent train/val/test",
        "mode": "a1gamma",
        "second_parameter": "a1",
        "fixed_parameters": {"a2": FIXED_A2, "a3": FIXED_A3, "m": FIXED_M},
        "data_support": "irregular continuous candidate pool selected by global g-space max-min packing",
        "second_train_values": [float(args.a1_min), float(args.a1_max)],
        "a1_range": [float(args.a1_min), float(args.a1_max)],
        "gamma_range": [float(args.gamma_min), float(args.gamma_max)],
        "candidate_count": int(args.candidate_count),
        "selected_state_count": int(len(selected_idx)),
        "selected_state_counts": split_counts,
        "global_min_rms_snr": actual_min,
        "requested_min_rms_snr": float(args.min_rms_snr),
        "reference_noise": float(args.reference_noise),
        "selection_hit_max_state_cap": bool(hit_cap),
        "physical_observation_points": int(args.observation_points),
        "model_input_points": int(args.model_input_points),
        "observation_indices": [int(x) for x in obs_idx],
        "noise_levels": [float(x) for x in args.noise_levels],
        "train_noise_level": float(args.train_noise_level),
        "per_state_sample_counts": per_state,
        "actual_sample_counts": {
            name: int(per_state[name] * split_counts[name])
            for name in ("train", "val", "test")
        },
        "split_policy": (
            "global separation first; then disjoint train/val/test partition. "
            "All four parameter-domain corners are assigned to train; remaining "
            "states are stratified in log-gamma."
        ),
        "independence_guarantee": (
            "train/val/test use disjoint physical parameter states and distinct noise seeds. "
            "Because the physical states were globally packed before splitting, the requested "
            "g-separation also holds across split boundaries."
        ),
        "integration_points": int(args.integration_points),
        "seed": int(args.seed),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("Exp26 data ready")
    print("selected physical states: %d" % len(selected_idx))
    print("split counts            : %s" % split_counts)
    print("actual global min RMS-SNR: %.6g" % actual_min)
    print("hit max-selected cap    : %s" % hit_cap)
    print("Read first:")
    print("  %s" % (output_dir / "selected_states.csv"))
    print("  %s" % (output_dir / "split_pair_gseparation_summary.csv"))
    print("  %s" % (output_dir / "selected_states_by_split.png"))
    print("=" * 100)


if __name__ == "__main__":
    main()
