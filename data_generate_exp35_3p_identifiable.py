#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp35 step 2: build identifiable 3P train/val/test data from the alias pool.

Constraints:
1. candidate sampled remote-alias score >= alias threshold;
2. retained physical truths are globally separated in observed g-space;
3. every val/test physical truth has a train truth within the requested cover;
4. train/val/test are split by physical state before noise realizations.

The physical formula and g construction remain unchanged.
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
    forward_bank,
    noise_dir_name,
    save_npz,
)
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
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    return vals


def write_csv(path, rows):
    if not rows:
        return
    fields, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k); fields.append(k)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)



def slim_3p(arrays, keep_physics=False):
    keep = {"gy_noisy", "a1", "m", "gamma", "q2", "y", "q2_observed", "observation_indices", "observation_points", "model_input_points"}
    if keep_physics:
        keep |= {"parameters", "gy_clean", "fx"}
    return {k: v for k, v in arrays.items() if k in keep}


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate Exp35 identifiable a1+m+gamma data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pool-dir", default="data_exp35_3p_alias_pool")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--alias-min-rms-snr", type=float, default=0.40)
    p.add_argument("--min-rms-snr", type=float, default=1.30)
    p.add_argument("--max-train-cover-rms-snr", type=float, default=2.0)
    p.add_argument("--max-selected-states", type=int, default=420)
    p.add_argument("--min-required-states", type=int, default=60)

    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--split-m-bins", type=int, default=4)
    p.add_argument("--split-gamma-bins", type=int, default=6)

    p.add_argument("--noise-levels", type=parse_float_list, default=[0.0, 0.002, 0.01])
    p.add_argument("--train-noise-level", type=float, default=0.002)
    p.add_argument("--target-train-samples", type=int, default=30000)
    p.add_argument("--target-val-samples", type=int, default=5000)
    p.add_argument("--target-test-samples", type=int, default=8000)

    p.add_argument("--seed", type=int, default=20260870)
    p.add_argument("--compressed", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def greedy_pack(g_obs, master_rms, reference_noise, alias_score, min_rms_snr, max_selected):
    n = len(master_rms)
    if n == 0:
        return np.empty(0, np.int32), [], False
    first = int(np.argmax(alias_score))
    selected = [first]
    active = np.ones(n, dtype=bool); active[first] = False
    min_sep = np.full(n, np.inf, dtype=np.float64); min_sep[first] = 0
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
            hit_cap = True; break
        masked = np.where(active, min_sep, -np.inf)
        j = int(np.argmax(masked)); s = float(masked[j])
        if not np.isfinite(s) or s < float(min_rms_snr):
            break
        selected.append(j); active[j] = False; min_sep[j] = 0
        trace.append({
            "selection_order": len(selected)-1,
            "candidate_local_index": j,
            "maximin_rms_snr_at_selection": s,
            "alias_score": float(alias_score[j]),
        })
        idx = np.where(active)[0]
        if len(idx):
            ss = rms_snr_to_one(
                g_obs, master_rms, reference_noise, j, idx
            )
            min_sep[idx] = np.minimum(min_sep[idx], ss)
    return np.asarray(selected, dtype=np.int32), trace, hit_cap


def pairwise_snr_matrix(g_obs, master_rms, reference_noise):
    n = len(master_rms)
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        js = np.arange(i+1, n, dtype=np.int64)
        if len(js):
            s = rms_snr_to_one(g_obs, master_rms, reference_noise, i, js)
            out[i, js] = s.astype(np.float32)
            out[js, i] = s.astype(np.float32)
    np.fill_diagonal(out, 0)
    return out


def target_counts(n, train_fraction, val_fraction, test_fraction):
    f = np.asarray([train_fraction, val_fraction, test_fraction], float)
    if np.any(f <= 0) or not np.isclose(np.sum(f), 1.0):
        raise ValueError("split fractions must be positive and sum to 1")
    raw = f * int(n)
    c = np.floor(raw).astype(int)
    rem = int(n) - int(c.sum())
    order = np.argsort(-(raw-c))
    for k in order[:rem]:
        c[k] += 1
    return {"train": int(c[0]), "val": int(c[1]), "test": int(c[2])}


def choose_train_cover(dmat, alias_score, target_train, max_cover):
    n = len(dmat)
    # Seed train with safest states spread by farthest-to-train distance.
    first = int(np.argmax(alias_score))
    train = [first]
    remaining = set(range(n)); remaining.remove(first)
    nearest = np.asarray(dmat[:, first], dtype=np.float64)
    nearest[first] = 0.0

    while len(train) < int(target_train) and remaining:
        arr = np.asarray(sorted(remaining), dtype=np.int64)
        # Prefer states currently least covered by train, with alias score as a
        # very small deterministic tie-breaker.
        score = nearest[arr] + 1e-6 * alias_score[arr]
        j = int(arr[int(np.argmax(score))])
        train.append(j); remaining.remove(j)
        nearest = np.minimum(nearest, dmat[:, j])
        nearest[train] = 0.0

    promoted = 0
    while remaining:
        arr = np.asarray(sorted(remaining), dtype=np.int64)
        j = int(arr[int(np.argmax(nearest[arr]))])
        if float(nearest[j]) <= float(max_cover) + 1e-12:
            break
        train.append(j); remaining.remove(j); promoted += 1
        nearest = np.minimum(nearest, dmat[:, j])
        nearest[train] = 0.0
    return np.asarray(sorted(train), np.int32), nearest, promoted


def stratified_val_test(remaining, params, n_val, n_test, m_bins, gamma_bins, seed):
    remaining = np.asarray(remaining, dtype=np.int32)
    if len(remaining) != int(n_val) + int(n_test):
        raise ValueError("remaining count mismatch")
    rng = np.random.default_rng(int(seed) + 3501)
    mass = params[:, 3]
    logg = np.log10(params[:, 4])
    me = np.linspace(mass.min(), mass.max(), int(m_bins)+1)
    ge = np.linspace(logg.min(), logg.max(), int(gamma_bins)+1)
    mb = np.clip(np.digitize(mass, me[1:-1]), 0, int(m_bins)-1)
    gb = np.clip(np.digitize(logg, ge[1:-1]), 0, int(gamma_bins)-1)
    cell = mb * int(gamma_bins) + gb

    val, test = [], []
    frac_val = float(n_val) / max(float(n_val+n_test), 1.0)
    for c in range(int(m_bins)*int(gamma_bins)):
        idx = remaining[cell[remaining] == c].copy()
        if not len(idx):
            continue
        rng.shuffle(idx)
        nv = int(round(frac_val * len(idx)))
        if len(idx) >= 2:
            nv = min(max(nv, 1), len(idx)-1)
        else:
            nv = 1 if len(val) < n_val else 0
        val.extend(int(x) for x in idx[:nv])
        test.extend(int(x) for x in idx[nv:])

    def move(src, dst):
        if not src:
            raise RuntimeError("cannot rebalance val/test")
        dst.append(src.pop())
    while len(val) > n_val: move(val, test)
    while len(val) < n_val: move(test, val)
    while len(test) > n_test: move(test, val)
    while len(test) < n_test: move(val, test)
    return np.asarray(sorted(val), np.int32), np.asarray(sorted(test), np.int32)


def coverage_rows(params, split_labels, dmat):
    tr = np.where(split_labels == "train")[0]
    rows = []
    for i in range(len(params)):
        if split_labels[i] == "train":
            continue
        s = dmat[i, tr]
        k = int(np.argmin(s)); j = int(tr[k])
        rows.append({
            "state_index": int(i),
            "split": str(split_labels[i]),
            "a1": float(params[i,0]),
            "m": float(params[i,3]),
            "gamma": float(params[i,4]),
            "nearest_train_state": j,
            "nearest_train_a1": float(params[j,0]),
            "nearest_train_m": float(params[j,3]),
            "nearest_train_gamma": float(params[j,4]),
            "nearest_train_rms_snr": float(s[k]),
        })
    return rows


def summarize_coverage(rows):
    out = []
    for name in ("val", "test", "holdout_all"):
        rr = rows if name == "holdout_all" else [r for r in rows if r["split"] == name]
        a = np.asarray([r["nearest_train_rms_snr"] for r in rr], float)
        out.append({
            "split": name,
            "count": int(len(a)),
            "min_nearest_train_rms_snr": float(np.min(a)),
            "median_nearest_train_rms_snr": float(np.median(a)),
            "p90_nearest_train_rms_snr": float(np.quantile(a,0.90)),
            "max_nearest_train_rms_snr": float(np.max(a)),
        })
    return out


def parameter_ranges(params):
    return {
        "a1": [float(np.min(params[:,0])), float(np.max(params[:,0]))],
        "m": [float(np.min(params[:,3])), float(np.max(params[:,3]))],
        "gamma": [float(np.min(params[:,4])), float(np.max(params[:,4]))],
    }


def plot_selected(out, params, split_labels, alias_score):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    for xidx, xlabel, fname in [
        (0, "a1", "selected_a1_gamma.png"),
        (3, "m", "selected_m_gamma.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8,5))
        for name in ("train","val","test"):
            idx = np.where(split_labels == name)[0]
            ax.scatter(params[idx,xidx], params[idx,4], s=28, alpha=.8, label=name)
        ax.set_yscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel("gamma")
        ax.set_title("Exp35 selected 3P support")
        ax.legend(); fig.tight_layout(); fig.savefig(Path(out)/fname, dpi=170)
        plt.close(fig)


def main():
    args = parse_args()
    pool_dir, out = Path(args.pool_dir), Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise FileExistsError("%s is not empty; use --overwrite" % out)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((pool_dir/"metadata.json").read_text(encoding="utf-8"))
    with np.load(pool_dir/"alias_pool_3p.npz", allow_pickle=False) as p:
        params_all = np.asarray(p["candidate_parameters"], np.float64)
        g_all = np.asarray(p["g_master"], np.float32)
        alias_all = np.asarray(p["remote_alias_rms_snr"], np.float64)
        alias_idx_all = np.asarray(p["alias_candidate_index"], np.int64)
        obs_idx = np.asarray(p["observation_indices"], np.int64)

    ref_noise = float(meta["reference_noise"])
    safe = np.where(alias_all >= float(args.alias_min_rms_snr))[0]
    if len(safe) < int(args.min_required_states):
        raise RuntimeError(
            "only %d alias-safe candidates at threshold %.4g; lower threshold or "
            "increase candidate pool" % (len(safe), args.alias_min_rms_snr)
        )

    params_safe = params_all[safe]
    g_safe = g_all[safe]
    alias_safe = alias_all[safe]
    g_obs_safe = g_safe[:, obs_idx]
    rms_safe = np.sqrt(np.mean(np.asarray(g_safe,np.float64)**2, axis=1))

    packed_local, trace, hit_cap = greedy_pack(
        g_obs_safe, rms_safe, ref_noise, alias_safe,
        args.min_rms_snr, args.max_selected_states
    )
    if len(packed_local) < int(args.min_required_states):
        raise RuntimeError(
            "only %d states survive global g packing; lower min-rms-snr or "
            "increase candidate support" % len(packed_local)
        )

    selected_pool_idx = safe[packed_local]
    params = params_all[selected_pool_idx]
    g_master = g_all[selected_pool_idx]
    alias_score = alias_all[selected_pool_idx]
    alias_neighbor_pool = alias_idx_all[selected_pool_idx]

    g_obs = g_master[:, obs_idx]
    master_rms = np.sqrt(np.mean(np.asarray(g_master,np.float64)**2, axis=1))
    nearest_rows, nearest_scores, _ = selected_nearest_rows(
        params, g_obs, master_rms, ref_noise
    )
    dmat = pairwise_snr_matrix(g_obs, master_rms, ref_noise)

    tc = target_counts(
        len(params), args.train_fraction, args.val_fraction, args.test_fraction
    )
    train_idx, _, promoted = choose_train_cover(
        dmat, alias_score, tc["train"], args.max_train_cover_rms_snr
    )
    remaining = np.asarray(
        sorted(set(range(len(params))) - set(int(x) for x in train_idx)),
        np.int32
    )
    if len(train_idx) != tc["train"]:
        hold = len(remaining)
        val_ratio = tc["val"] / max(float(tc["val"]+tc["test"]),1.0)
        n_val = int(round(val_ratio*hold))
        n_val = min(max(n_val,1), hold-1)
        n_test = hold-n_val
    else:
        n_val,n_test = tc["val"],tc["test"]
    if n_val < 5 or n_test < 5:
        raise RuntimeError("coverage constraint leaves too few val/test states")

    val_idx,test_idx = stratified_val_test(
        remaining, params, n_val,n_test,
        args.split_m_bins,args.split_gamma_bins,args.seed
    )
    labels = np.full(len(params),"",dtype=object)
    labels[train_idx]="train"; labels[val_idx]="val"; labels[test_idx]="test"

    coverage = coverage_rows(params, labels, dmat)
    coverage_summary = summarize_coverage(coverage)
    max_cover = max(
        r["max_nearest_train_rms_snr"] for r in coverage_summary
        if r["split"]=="holdout_all"
    )
    if max_cover > args.max_train_cover_rms_snr + 1e-8:
        raise RuntimeError("holdout->train coverage guarantee failed")

    for i,row in enumerate(nearest_rows):
        pool_i = int(selected_pool_idx[i])
        alias_j = int(alias_neighbor_pool[i])
        nearest_j = int(row.get("nearest_state", -1))
        row.update({
            "split": str(labels[i]),
            "pool_candidate_index": pool_i,
            "a1": float(params[i,0]),
            "m": float(params[i,3]),
            "gamma": float(params[i,4]),
            "nearest_m": float(params[nearest_j,3]) if nearest_j >= 0 else math.nan,
            "remote_alias_rms_snr": float(alias_score[i]),
            "remote_alias_pool_index": alias_j,
            "remote_alias_a1": float(params_all[alias_j,0]),
            "remote_alias_m": float(params_all[alias_j,3]),
            "remote_alias_gamma": float(params_all[alias_j,4]),
        })
    write_csv(out/"selected_states.csv", nearest_rows)
    write_csv(out/"selection_trace.csv", trace)
    write_csv(out/"holdout_to_train_coverage.csv", coverage)
    write_csv(out/"coverage_summary.csv", coverage_summary)
    write_csv(
        out/"split_pair_gseparation_summary.csv",
        pairwise_split_summary(params,g_obs,master_rms,labels,ref_noise)
    )
    plot_selected(out,params,labels,alias_score)

    physics = replace(DEFAULT_PHYSICS, q2_points=int(meta["model_input_points"]))
    fx,direct_g = forward_bank(
        params, physics=physics, integration_points=int(meta["integration_points"])
    )
    rel = np.linalg.norm(
        np.asarray(direct_g,np.float64)-np.asarray(g_master,np.float64), axis=1
    ) / np.maximum(np.linalg.norm(np.asarray(direct_g,np.float64),axis=1),1e-30)
    if float(np.max(rel)) > 5e-5:
        raise RuntimeError("pool/direct forward consistency failed")

    try:
        from mc_physics import output_grids_numpy
        _,q_master = output_grids_numpy(physics)
    except Exception:
        q_master = np.linspace(
            physics.q2_min, physics.q2_max, physics.q2_points, dtype=np.float64
        )

    split_counts = {name:int(np.sum(labels==name)) for name in ("train","val","test")}
    targets = {
        "train":args.target_train_samples,
        "val":args.target_val_samples,
        "test":args.target_test_samples,
    }
    per_state = {
        name:count_per_state(targets[name],split_counts[name])
        for name in ("train","val","test")
    }

    for noise in args.noise_levels:
        ndir = out/noise_dir_name(noise); ndir.mkdir(parents=True,exist_ok=True)
        for name in ("train","val","test"):
            if name != "test" and not np.isclose(
                noise,args.train_noise_level,atol=1e-12,rtol=0
            ):
                continue
            idx = np.where(labels==name)[0]
            arrays = make_split_arrays_from_bank(
                split_name=name,
                params=params[idx],
                fx_bank=np.asarray(fx)[idx],
                g_master_bank=np.asarray(g_master)[idx],
                count_per_state=per_state[name],
                noise_level=float(noise),
                q_master=q_master,
                obs_idx=obs_idx,
                seed=int(args.seed)+{"train":11,"val":22,"test":33}[name]*100003,
            )
            arrays = slim_3p(arrays, keep_physics=(name == "test"))
            save_npz(ndir/(name+".npz"),arrays,compressed=bool(args.compressed))

    original_ranges = meta["parameter_ranges"]
    selected_ranges = parameter_ranges(params)
    metadata = {
        "experiment":"Exp35 identifiable 3P a1+m+gamma",
        "mode":"a1mgamma",
        "free_parameters":["a1","m","gamma"],
        "fixed_parameters":meta["fixed_parameters"],
        "original_parameter_ranges":original_ranges,
        "selected_parameter_ranges":selected_ranges,
        "reference_noise":ref_noise,
        "alias_min_rms_snr":float(args.alias_min_rms_snr),
        "alias_method":meta["alias_method"],
        "remote_criteria":meta["remote_criteria"],
        "min_rms_snr":float(args.min_rms_snr),
        "max_train_cover_rms_snr":float(args.max_train_cover_rms_snr),
        "alias_safe_candidate_count":int(len(safe)),
        "selected_state_count":int(len(params)),
        "selected_alias_score_min":float(np.min(alias_score)),
        "selected_alias_score_median":float(np.median(alias_score)),
        "global_selected_min_g_rms_snr":float(np.min(nearest_scores)),
        "max_holdout_to_train_rms_snr":float(max_cover),
        "selection_hit_cap":bool(hit_cap),
        "train_cover_promotions":int(promoted),
        "split_counts":split_counts,
        "samples_per_state":per_state,
        "noise_levels":[float(x) for x in args.noise_levels],
        "train_noise_level":float(args.train_noise_level),
        "physical_observation_points":int(meta["physical_observation_points"]),
        "model_input_points":int(meta["model_input_points"]),
        "observation_indices":[int(x) for x in obs_idx],
        "integration_points":int(meta["integration_points"]),
        "pool_seed":int(meta["seed"]),
        "data_seed":int(args.seed),
    }
    (out/"metadata.json").write_text(
        json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8"
    )

    write_csv(out/"dataset_gate.csv",[{
        "alias_threshold":float(args.alias_min_rms_snr),
        "alias_safe_candidates":int(len(safe)),
        "selected_states":int(len(params)),
        "selected_alias_min":float(np.min(alias_score)),
        "selected_g_separation_min":float(np.min(nearest_scores)),
        "max_holdout_train_cover":float(max_cover),
        "train_states":split_counts["train"],
        "val_states":split_counts["val"],
        "test_states":split_counts["test"],
        "pass_alias":bool(np.min(alias_score)>=args.alias_min_rms_snr-1e-8),
        "pass_separation":bool(np.min(nearest_scores)>=args.min_rms_snr-1e-8),
        "pass_coverage":bool(max_cover<=args.max_train_cover_rms_snr+1e-8),
    }])

    print("="*100)
    print("Exp35 3P identifiable data ready")
    print("alias-safe candidates    : %d" % len(safe))
    print("selected states          : %d" % len(params))
    print("split counts             : %s" % split_counts)
    print("selected alias min       : %.5g" % np.min(alias_score))
    print("global min g RMS-SNR     : %.5g" % np.min(nearest_scores))
    print("max holdout->train SNR   : %.5g" % max_cover)
    print("Read first:")
    print("  %s" % (out/"dataset_gate.csv"))
    print("  %s" % (out/"selected_states.csv"))
    print("="*100)


if __name__ == "__main__":
    main()
