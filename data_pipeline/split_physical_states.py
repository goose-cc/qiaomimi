from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline_core import (
    default_output_dir,
    load_config,
    master_rms,
    nearest_to_train_scores,
    normalized_parameter_matrix,
    pairwise_min_snr_between,
    parameter_matrix_from_rows,
    read_csv_rows,
    update_metadata,
    write_csv_rows,
)

PARAMETER_NAMES = ("a1", "a2", "a3", "m", "gamma")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split selected physical states before noise expansion")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--smoke-permissive", action="store_true")
    return p.parse_args()


def _target_counts(n: int, cfg: dict) -> tuple[int, int, int]:
    sp = cfg["split"]
    train_f = float(sp.get("train_fraction", 0.70))
    val_f = float(sp.get("val_fraction", 0.15))
    test_f = float(sp.get("test_fraction", 0.15))
    total = train_f + val_f + test_f
    train_f, val_f, test_f = train_f / total, val_f / total, test_f / total
    n_val = max(int(sp.get("min_val_states", 5)), int(round(n * val_f)))
    n_test = max(int(sp.get("min_test_states", 5)), int(round(n * test_f)))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise RuntimeError("selected bank is too small for requested split minima")
    return n_train, n_val, n_test


def _farthest_order(z: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n = len(z)
    first = int(rng.integers(0, n))
    order = [first]
    active = np.ones(n, dtype=bool)
    active[first] = False
    min_d = np.linalg.norm(z - z[first], axis=1)
    min_d[first] = 0.0
    while np.any(active):
        j = int(np.argmax(np.where(active, min_d, -np.inf)))
        order.append(j)
        active[j] = False
        d = np.linalg.norm(z - z[j], axis=1)
        min_d = np.minimum(min_d, d)
        min_d[~active] = 0.0
    return np.asarray(order, dtype=np.int64)


def run(config_path: str, output_dir: str | None = None, seed: int | None = None,
        smoke_permissive: bool = False) -> Path:
    cfg = load_config(config_path)
    out = Path(output_dir) if output_dir else default_output_dir(cfg)
    sel_rows = read_csv_rows(out / "selected_states.csv")
    params = parameter_matrix_from_rows(sel_rows)
    ids = np.asarray([r["state_id"] for r in sel_rows])
    cand_rows = read_csv_rows(out / "candidate_states.csv")
    cand_index = {r["state_id"]: int(r["state_index"]) for r in cand_rows}
    selected_candidate_idx = np.asarray([cand_index[s] for s in ids], dtype=np.int64)
    bank = np.load(out / "candidate_g_clean.npz", allow_pickle=False)
    g_all = np.asarray(bank["g_clean"], dtype=np.float32)
    obs_idx = np.asarray(bank["observation_indices"], dtype=np.int64)
    g = g_all[selected_candidate_idx][:, obs_idx]
    rms = master_rms(g_all[selected_candidate_idx])
    reference_noise = float(cfg["identifiability"]["reference_noise"])

    split_cfg = cfg["split"]
    seed = int(seed if seed is not None else split_cfg.get("seed", cfg["sampling"]["seed"] + 1000))
    n_train, n_val, n_test = _target_counts(len(params), cfg)
    z = normalized_parameter_matrix(params, cfg)
    order = _farthest_order(z, seed)
    holdout_count = n_val + n_test
    holdout = order[:holdout_count]
    train = np.asarray(sorted(set(range(len(params))) - set(int(x) for x in holdout)), dtype=np.int64)
    # Alternate farthest holdouts between val and test so both cover parameter-space extremes.
    val, test = [], []
    for idx in holdout:
        if len(val) >= n_val:
            test.append(int(idx))
        elif len(test) >= n_test:
            val.append(int(idx))
        elif len(val) / max(n_val, 1) <= len(test) / max(n_test, 1):
            val.append(int(idx))
        else:
            test.append(int(idx))
    val = np.asarray(val, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)

    max_cover = float(split_cfg.get("max_train_cover_rms_snr", 2.0))
    min_val = int(split_cfg.get("min_val_states", 5))
    min_test = int(split_cfg.get("min_test_states", 5))
    promoted = 0
    while True:
        hold = np.concatenate([val, test])
        scores, _ = nearest_to_train_scores(g, rms, train, hold, reference_noise)
        bad_local = np.where(scores > max_cover)[0]
        if len(bad_local) == 0:
            break
        # Promote the worst-covered holdout if doing so preserves split minima.
        worst_local = int(bad_local[np.argmax(scores[bad_local])])
        idx = int(hold[worst_local])
        if idx in set(val.tolist()) and len(val) > min_val:
            val = val[val != idx]
        elif idx in set(test.tolist()) and len(test) > min_test:
            test = test[test != idx]
        else:
            break
        train = np.sort(np.append(train, idx)).astype(np.int64)
        promoted += 1

    hold = np.concatenate([val, test])
    cover_scores, cover_neighbor = nearest_to_train_scores(g, rms, train, hold, reference_noise)
    cover_ok = bool(len(cover_scores) == 0 or np.max(cover_scores) <= max_cover + 1e-8)
    if not smoke_permissive and not cover_ok:
        support = {
            "reason": "coverage constraint leaves holdout states beyond max_train_cover_rms_snr",
            "max_train_cover_rms_snr": max_cover,
            "max_actual_holdout_to_train_rms_snr": float(np.max(cover_scores)),
            "train_state_count": int(len(train)),
            "val_state_count": int(len(val)),
            "test_state_count": int(len(test)),
        }
        (out / "split_insufficient_support.json").write_text(json.dumps(support, indent=2), encoding="utf-8")
        raise RuntimeError("physical-state split coverage failed; see split_insufficient_support.json")

    labels = np.full(len(params), "", dtype=object)
    labels[train] = "train"
    labels[val] = "val"
    labels[test] = "test"
    manifest = []
    for i, row in enumerate(sel_rows):
        r = {"state_id": row["state_id"], "selected_index": i, "candidate_state_index": int(selected_candidate_idx[i]), "split": str(labels[i])}
        for name in PARAMETER_NAMES:
            r[name] = float(row[name])
        manifest.append(r)
    write_csv_rows(out / "split_manifest.csv", manifest)

    coverage_rows = []
    for qpos, idx in enumerate(hold):
        nidx = int(cover_neighbor[qpos])
        coverage_rows.append({
            "state_id": str(ids[idx]),
            "split": str(labels[idx]),
            "nearest_train_state_id": str(ids[nidx]) if nidx >= 0 else "",
            "nearest_train_rms_snr": float(cover_scores[qpos]),
        })
    write_csv_rows(out / "holdout_to_train_coverage.csv", coverage_rows)

    pair_rows = []
    split_indices = {"train": train, "val": val, "test": test}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        li, ri = split_indices[left], split_indices[right]
        met = pairwise_min_snr_between(g[li], rms[li], g[ri], rms[ri], reference_noise)
        pair_rows.append({"split_pair": f"{left}-{right}", "left_count": len(li), "right_count": len(ri), **met})
    write_csv_rows(out / "split_separation_summary.csv", pair_rows)

    max_actual_cover = float(np.max(cover_scores)) if len(cover_scores) else 0.0
    update_metadata(out, {
        "status": "PHYSICAL_SPLIT_READY",
        "split": {
            **cfg["split"],
            "seed": seed,
            "state_counts": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
            "coverage_promoted_extra_train_states": int(promoted),
            "max_actual_holdout_to_train_rms_snr": max_actual_cover,
            "coverage_pass": cover_ok,
            "smoke_permissive": bool(smoke_permissive),
            "cross_split_g_separation": pair_rows,
        },
    })
    print("=" * 100)
    print(f"physical-state split : train={len(train)} val={len(val)} test={len(test)}")
    print(f"max holdout->train   : {max_actual_cover:.6g} RMS-SNR")
    for r in pair_rows:
        print(f"{r['split_pair']:>10s} min g separation: {r['min_rms_snr']:.6g} RMS-SNR")
    print(f"[done] {out / 'split_manifest.csv'}")
    print(f"[done] {out / 'split_separation_summary.csv'}")
    print("=" * 100)
    return out


def main() -> None:
    a = parse_args()
    run(a.config, a.output_dir, a.seed, a.smoke_permissive)


if __name__ == "__main__":
    main()
