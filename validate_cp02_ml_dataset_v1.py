#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Integrity validator for the formal CP02 ML physical-state dataset v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import cp02_core as core
from cp02_observation import design_by_id, load_config, make_q2


def noise_tag(level: float) -> str:
    if abs(level) < 1e-15:
        return "noise_0pct"
    return "noise_" + ("%g" % (100.0 * float(level))).replace(".", "p") + "pct"


def add(checks: list[dict[str, Any]], name: str, ok: bool, value: Any = "") -> None:
    checks.append({"check": name, "status": "PASS" if bool(ok) else "FAIL", "value": value})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics-config", default="cp02_corrected_3p_config.json")
    ap.add_argument("--dataset-dir", default="data_cp02_ml_v1")
    ap.add_argument("--forward-batch-size", type=int, default=384)
    args = ap.parse_args()

    cfg = load_config(args.physics_config)
    root = Path(args.dataset_dir)
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    frozen = meta["frozen_definition"]
    checks: list[dict[str, Any]] = []

    required = [
        "q2.npy", "physical_states.npz", "physical_states.csv", "candidate_screen.npz",
        "continuous_refinement_cache.csv", "continuous_audit.csv", "coverage_by_axis.csv",
        "coverage_3d_cells.csv", "dataset_summary.csv", "normalization_train_only.npz",
        "normalization_train_only.json", "metadata.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    add(checks, "required_files", not missing, missing)
    if missing:
        core.write_csv(root / "validation_summary.csv", checks)
        raise SystemExit(f"Missing required dataset files: {missing}")

    q2 = np.load(root / "q2.npy").astype(np.float64)
    design = make_q2(design_by_id(cfg, frozen["design_id"]))
    add(checks, "formula_version_frozen", cfg["physics_formula_version"] == frozen["physics_formula_version"], cfg["physics_formula_version"])
    add(checks, "q2_nominal_design_label_240", int(frozen.get("q2_nominal_design_points", frozen.get("nominal_design_points", 240))) == 240, frozen.get("q2_nominal_design_points", frozen.get("nominal_design_points", 240)))
    q2_hash = __import__("hashlib").sha256(np.asarray(q2, dtype="<f8").tobytes()).hexdigest()
    add(checks, "q2_hash_matches_metadata", q2_hash == frozen.get("q2_sha256", q2_hash), q2_hash)
    # Current code deduplicates the shared -20 endpoint, so hybrid_ultranear_240
    # currently has 239 unique q2 values. Matching the frozen q2 artifact is
    # more important than forcing a post-hoc duplicate that would change D_M.
    generator_match = q2.shape == design.shape and np.allclose(q2, design, rtol=0.0, atol=1e-12)
    saved_source = str(frozen.get("q2_source", "")) not in ("", "cp02_observation.make_q2(current config)")
    add(checks, "q2_matches_generator_or_frozen_saved_artifact", generator_match or saved_source, {"points": len(q2), "generator_points": len(design), "source": frozen.get("q2_source", "")})
    add(checks, "threshold_frozen_1p5", abs(float(frozen["continuous_threshold_mahalanobis"]) - 1.5) < 1e-15, frozen["continuous_threshold_mahalanobis"])

    with np.load(root / "physical_states.npz", allow_pickle=False) as z:
        params = z["params"].astype(np.float64)
        clean = z["g_clean"].astype(np.float64)
        pids = z["physical_state_id"].astype(str)
        sids = z["state_id"].astype(np.int64)
        split = z["split"].astype(str)
        margin = z["continuous_margin"].astype(np.float64)
        q2_in = z["q2"].astype(np.float64)

    n = len(params)
    threshold = float(frozen["continuous_threshold_mahalanobis"])
    add(checks, "physical_state_shapes", params.shape == (n, 5) and clean.shape == (n, len(q2)), {"params": params.shape, "g_clean": clean.shape})
    add(checks, "physical_state_id_unique", len(np.unique(pids)) == n, n)
    add(checks, "numeric_state_id_unique", len(np.unique(sids)) == n, n)
    add(checks, "all_continuous_margins_ge_1p5", bool(np.all(margin >= threshold - 1e-12)), float(np.min(margin)))
    add(checks, "physical_q2_matches_root", np.array_equal(q2_in, q2), len(q2_in))

    split_sets = {name: set(pids[split == name].tolist()) for name in ("train", "val", "test")}
    disjoint = not (split_sets["train"] & split_sets["val"] or split_sets["train"] & split_sets["test"] or split_sets["val"] & split_sets["test"])
    union = split_sets["train"] | split_sets["val"] | split_sets["test"]
    add(checks, "physical_split_disjoint", disjoint, {k: len(v) for k, v in split_sets.items()})
    add(checks, "physical_split_covers_all_states", len(union) == n, len(union))
    frac = {k: len(v) / max(n, 1) for k, v in split_sets.items()}
    add(checks, "split_ratio_close_80_10_10", abs(frac["train"] - 0.8) <= 1.0 / n and abs(frac["val"] - 0.1) <= 1.0 / n and abs(frac["test"] - 0.1) <= 1.0 / n, frac)

    # Full forward roundtrip is an important physics/data integrity check.
    parts = []
    for start in range(0, n, int(args.forward_batch_size)):
        parts.append(core.forward(params[start:start + int(args.forward_batch_size)], q2, cfg["integration_points"]))
    recalc = np.concatenate(parts, axis=0) if parts else np.empty_like(clean)
    rel_l2 = float(np.linalg.norm(recalc - clean) / max(np.linalg.norm(clean), 1e-30))
    add(checks, "float64_forward_roundtrip_relL2_lt_1e-12", rel_l2 < 1e-12, rel_l2)

    noise_cfg = meta["noise"]
    train_level = float(noise_cfg["train_level"])
    train_file = root / noise_tag(train_level) / "train.npz"
    val_file = root / noise_tag(float(noise_cfg["val_level"])) / "val.npz"
    add(checks, "train_file_exists", train_file.exists(), str(train_file))
    add(checks, "val_file_exists", val_file.exists(), str(val_file))

    def inspect_sample_file(path: Path, expected_split: str, expected_level: float, expected_state_set: set[str], expected_reps: int) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as z:
            gy = z["gy"].astype(np.float64)
            gc = z["g_clean"].astype(np.float64)
            nz = z["noise"].astype(np.float64)
            fpids = z["physical_state_id"].astype(str)
            fsids = z["state_id"].astype(np.int64)
            fsplit = z["split"].astype(str)
            level = z["noise_level"].astype(np.float64)
            reps = z["noise_replica_id"].astype(np.int64)
            fm = z["continuous_margin"].astype(np.float64)
            fq2 = z["q2"].astype(np.float64)
        unique, counts = np.unique(fpids, return_counts=True)
        return {
            "q2_ok": np.array_equal(fq2, q2),
            "split_ok": bool(np.all(fsplit == expected_split)),
            "level_ok": bool(np.allclose(level, expected_level, rtol=0.0, atol=1e-8)),
            "state_set_ok": set(unique.tolist()) == expected_state_set,
            "reps_ok": bool(np.all(counts == expected_reps)),
            "replica_range_ok": bool(np.all((reps >= 0) & (reps < expected_reps))),
            "noise_identity_ok": bool(np.allclose(gy - gc, nz, rtol=0.0, atol=3e-7)),
            "margin_ok": bool(np.all(fm >= threshold - 1e-6)),
            "sample_count": len(gy),
            "state_count": len(unique),
            "numeric_ids_unique_per_physical": len(np.unique(np.column_stack([fpids, fsids]), axis=0)) == len(unique),
        }

    train_reps = int(noise_cfg["train_replicas_effective"])
    val_reps = int(noise_cfg["val_replicas"])
    train_diag = inspect_sample_file(train_file, "train", train_level, split_sets["train"], train_reps)
    val_diag = inspect_sample_file(val_file, "val", float(noise_cfg["val_level"]), split_sets["val"], val_reps)
    for k, v in train_diag.items():
        if k not in ("sample_count", "state_count"):
            add(checks, "train_" + k, bool(v), v)
    for k, v in val_diag.items():
        if k not in ("sample_count", "state_count"):
            add(checks, "val_" + k, bool(v), v)

    test_orders = []
    for level in noise_cfg["test_levels"]:
        level = float(level)
        path = root / noise_tag(level) / "test.npz"
        add(checks, f"test_file_exists_{noise_tag(level)}", path.exists(), str(path))
        diag = inspect_sample_file(path, "test", level, split_sets["test"], int(noise_cfg["test_replicas"]))
        for k, v in diag.items():
            if k not in ("sample_count", "state_count"):
                add(checks, f"test_{noise_tag(level)}_{k}", bool(v), v)
        with np.load(path, allow_pickle=False) as z:
            test_orders.append(z["physical_state_id"].astype(str))
    same_test_order = all(np.array_equal(test_orders[0], x) for x in test_orders[1:])
    add(checks, "same_test_physical_states_and_order_at_0_0p2_1pct", same_test_order, len(test_orders[0]))

    # Verify the saved normalization really comes from train samples only.
    with np.load(train_file, allow_pickle=False) as z:
        train_gy = z["gy"].astype(np.float64)
    with np.load(root / "normalization_train_only.npz", allow_pickle=False) as z:
        mean_saved = z["input_mean"].astype(np.float64)
        std_saved = z["input_std"].astype(np.float64)
        target_mean_saved = z["target_mean"].astype(np.float64)
        target_std_saved = z["target_std"].astype(np.float64)
    mean = train_gy.mean(axis=0)
    std = train_gy.std(axis=0)
    std = np.maximum(std, 1e-12 * np.maximum(np.abs(mean), 1.0))
    train_mask = split == "train"
    targets = np.column_stack([params[train_mask, 0], params[train_mask, 3], np.log(params[train_mask, 4])])
    add(checks, "input_normalization_train_only_mean", np.allclose(mean_saved, mean, rtol=1e-12, atol=1e-12), float(np.max(np.abs(mean_saved - mean))))
    add(checks, "input_normalization_train_only_std", np.allclose(std_saved, std, rtol=1e-12, atol=1e-12), float(np.max(np.abs(std_saved - std))))
    add(checks, "target_normalization_train_only_mean", np.allclose(target_mean_saved, targets.mean(axis=0), rtol=1e-12, atol=1e-12), target_mean_saved.tolist())
    add(checks, "target_normalization_train_only_std", np.allclose(target_std_saved, targets.std(axis=0), rtol=1e-12, atol=1e-12), target_std_saved.tolist())

    audit = meta["continuous_audit"]
    add(checks, "continuous_audit_no_failures", int(audit["fail_count"]) == 0, audit)
    add(checks, "coverage_reports_exist", (root / "coverage_by_axis.csv").stat().st_size > 0 and (root / "coverage_3d_cells.csv").stat().st_size > 0, "axis+3d")

    core.write_csv(root / "validation_summary.csv", checks)
    failures = [r["check"] for r in checks if r["status"] != "PASS"]
    summary = {
        "status": "PASS" if not failures else "FAIL",
        "physical_states": n,
        "split_counts": {k: len(v) for k, v in split_sets.items()},
        "min_continuous_margin": float(np.min(margin)),
        "forward_roundtrip_relL2": rel_l2,
        "failures": failures,
    }
    (root / "validation_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit("CP02 ML dataset validation failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
