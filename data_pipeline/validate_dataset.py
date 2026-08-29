from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mc_physics import scaled_forward_observation_numpy, valid_parameter_mask_numpy
from pipeline_core import (
    PARAMETER_NAMES,
    default_output_dir,
    load_config,
    make_physics,
    noise_dir_name,
    parameter_matrix_from_rows,
    read_csv_rows,
    resolve_per_state_count,
    update_metadata,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the complete data-only pipeline output")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--smoke-permissive", action="store_true")
    return p.parse_args()


def _fail(errors: list[str], msg: str) -> None:
    errors.append(msg)


def _duplicate_row_pairs(x: np.ndarray, decimals: int = 7) -> int:
    x = np.ascontiguousarray(np.round(np.asarray(x, dtype=np.float64), decimals=decimals))
    if len(x) < 2:
        return 0
    view = x.view(np.dtype((np.void, x.dtype.itemsize * x.shape[1]))).ravel()
    _, counts = np.unique(view, return_counts=True)
    return int(np.sum(np.maximum(counts - 1, 0)))


def run(config_path: str, output_dir: str | None = None, smoke_permissive: bool = False) -> Path:
    cfg = load_config(config_path)
    out = Path(output_dir) if output_dir else default_output_dir(cfg)
    errors: list[str] = []
    warnings: list[str] = []

    required_files = [
        "candidate_states.csv", "candidate_g_clean.npz", "identifiability_results.csv",
        "alias_summary.csv", "hard_states.csv", "selected_states.csv", "coverage_summary.csv",
        "split_manifest.csv", "split_separation_summary.csv", "metadata.json",
    ]
    for name in required_files:
        if not (out / name).exists():
            _fail(errors, f"missing required file: {name}")
    if errors:
        report = {"status": "FAIL", "errors": errors, "warnings": warnings}
        (out / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError("dataset validation failed: missing required files")

    cand_rows = read_csv_rows(out / "candidate_states.csv")
    sel_rows = read_csv_rows(out / "selected_states.csv")
    split_rows = read_csv_rows(out / "split_manifest.csv")
    cand = parameter_matrix_from_rows(cand_rows)
    sel = parameter_matrix_from_rows(sel_rows)
    bank = np.load(out / "candidate_g_clean.npz", allow_pickle=False)
    g_all = np.asarray(bank["g_clean"], dtype=np.float32)
    q2 = np.asarray(bank["q2"], dtype=np.float32)

    if len(cand) != len(g_all):
        _fail(errors, "candidate_states.csv and candidate_g_clean.npz row counts differ")
    if not np.isfinite(cand).all() or not np.isfinite(g_all).all():
        _fail(errors, "candidate parameters or g_clean contain NaN/Inf")
    ids = [r["state_id"] for r in cand_rows]
    if len(ids) != len(set(ids)):
        _fail(errors, "candidate state_id values are not unique")
    pkey = np.round(cand, 12)
    if _duplicate_row_pairs(pkey, decimals=12) > 0:
        _fail(errors, "duplicate physical parameter rows exist in candidate pool")
    if _duplicate_row_pairs(g_all, decimals=12) > 0:
        _fail(errors, "duplicate clean g rows exist in candidate pool")

    # Config bounds + fixed parameter checks.
    for name in cfg["free_params"]:
        j = PARAMETER_NAMES.index(name)
        lo, hi = [float(v) for v in cfg["parameter_ranges"][name]]
        if np.any(cand[:, j] < lo - 1e-12) or np.any(cand[:, j] > hi + 1e-12):
            _fail(errors, f"candidate {name} values exceed configured range")
    for name, value in cfg["fixed_params"].items():
        j = PARAMETER_NAMES.index(name)
        if not np.allclose(cand[:, j], float(value), atol=1e-12, rtol=0):
            _fail(errors, f"fixed parameter {name} is not constant at {value}")
    if not np.all(valid_parameter_mask_numpy(cand)):
        _fail(errors, "candidate pool contains physically invalid parameter states")

    cand_index = {r["state_id"]: int(r["state_index"]) for r in cand_rows}
    selected_ids = [r["state_id"] for r in sel_rows]
    if len(selected_ids) != len(set(selected_ids)):
        _fail(errors, "selected state_id values are duplicated")
    if not set(selected_ids).issubset(set(ids)):
        _fail(errors, "selected_states.csv contains state_id not present in candidate pool")
    selected_candidate_idx = np.asarray([cand_index[s] for s in selected_ids], dtype=np.int64)
    selected_clean = g_all[selected_candidate_idx]
    if _duplicate_row_pairs(selected_clean, decimals=12) > 0:
        _fail(errors, "different selected physical states have duplicate g_clean rows")

    # Split state ownership must be exact and disjoint.
    split_ids = [r["state_id"] for r in split_rows]
    if len(split_ids) != len(set(split_ids)):
        _fail(errors, "same physical state appears more than once in split_manifest.csv")
    if set(split_ids) != set(selected_ids):
        _fail(errors, "split_manifest.csv does not cover selected_states.csv exactly")
    split_sets = {name: {r["state_id"] for r in split_rows if r["split"] == name} for name in ("train", "val", "test")}
    if (split_sets["train"] & split_sets["val"]) or (split_sets["train"] & split_sets["test"]) or (split_sets["val"] & split_sets["test"]):
        _fail(errors, "physical-state leakage exists across train/val/test")
    for name, minimum in (("val", int(cfg["split"].get("min_val_states", 5))), ("test", int(cfg["split"].get("min_test_states", 5)))):
        if len(split_sets[name]) < minimum and not smoke_permissive:
            _fail(errors, f"{name} has fewer than configured minimum physical states")

    # Coarse parameter coverage in test. This is intentionally a span check rather than an
    # unrealistic requirement that a tiny test split occupy every multidimensional bin.
    test_params = parameter_matrix_from_rows([r for r in split_rows if r["split"] == "test"])
    min_span = float(cfg.get("validation", {}).get("min_test_parameter_span_fraction", 0.50))
    if len(test_params):
        for name in cfg["free_params"]:
            j = PARAMETER_NAMES.index(name)
            full_span = float(np.ptp(sel[:, j]))
            test_span = float(np.ptp(test_params[:, j]))
            frac = 1.0 if full_span <= 1e-30 else test_span / full_span
            if frac + 1e-12 < min_span:
                msg = f"test coverage span for {name} is {frac:.3f}, below required {min_span:.3f}"
                if smoke_permissive:
                    warnings.append(msg)
                else:
                    _fail(errors, msg)

    # Direct-forward consistency on selected states.
    sample_n = min(len(sel), int(cfg.get("validation", {}).get("direct_forward_sample", 32)))
    if sample_n:
        sample = np.linspace(0, len(sel) - 1, sample_n, dtype=np.int64)
        physics = make_physics(cfg)
        direct = scaled_forward_observation_numpy(
            sel[sample], integration_points=int(cfg["forward"].get("integration_points", 128)), config=physics
        ).astype(np.float64)
        stored = selected_clean[sample].astype(np.float64)
        rel = np.linalg.norm(direct - stored, axis=1) / np.maximum(np.linalg.norm(direct, axis=1), 1e-30)
        max_rel = float(np.max(rel))
        tol = float(cfg.get("validation", {}).get("direct_forward_max_rel_l2", 5e-5))
        if max_rel > tol:
            _fail(errors, f"direct-forward consistency failed: max relL2 {max_rel:.6g} > {tol:.6g}")
    else:
        max_rel = float("nan")

    # Validate every generated noise dataset. Repeated gy for one state at noise=0 is expected;
    # physical-state leakage across split files is not.
    noise_cfg = cfg["noise"]
    expected_per_state = {
        name: resolve_per_state_count(noise_cfg, name, len(split_sets[name])) for name in ("train", "val", "test")
    }
    required_npz = {"gy", "gy_noisy", "gy_clean", "parameters", "a1", "a2", "a3", "m", "gamma", "state_id", "noise_level"}
    for level in [float(x) for x in noise_cfg["noise_levels"]]:
        ndir = out / noise_dir_name(level)
        for split in ("train", "val", "test"):
            path = ndir / f"{split}.npz"
            if not path.exists():
                _fail(errors, f"missing noisy split file: {path.relative_to(out)}")
                continue
            a = np.load(path, allow_pickle=False)
            missing = sorted(required_npz - set(a.files))
            if missing:
                _fail(errors, f"{path.name} missing keys: {missing}")
                continue
            if not np.isfinite(a["gy"]).all() or not np.isfinite(a["parameters"]).all():
                _fail(errors, f"{path.name} contains NaN/Inf")
            sids = np.asarray(a["state_id"]).astype(str)
            if not set(sids).issubset(split_sets[split]):
                _fail(errors, f"{path.name} contains state_id owned by another split")
            unique, counts = np.unique(sids, return_counts=True)
            if len(unique) != len(split_sets[split]):
                _fail(errors, f"{path.name} does not contain every {split} physical state")
            if len(counts) and np.any(counts != expected_per_state[split]):
                _fail(errors, f"{path.name} has unexpected noise-realization count per physical state")
            if not np.allclose(np.asarray(a["noise_level"], dtype=float), level, atol=1e-8, rtol=0):
                _fail(errors, f"{path.name} noise_level field does not match directory level")
            expected_row = {r["state_id"]: r for r in split_rows if r["split"] == split}
            pars = np.asarray(a["parameters"], dtype=np.float64)
            for sid in unique:
                mask = sids == sid
                row = expected_row[str(sid)]
                expected = np.asarray([float(row[name]) for name in PARAMETER_NAMES], dtype=np.float64)
                if not np.allclose(pars[mask], expected[None, :], atol=2e-7, rtol=0):
                    _fail(errors, f"{path.name} parameter matrix disagrees with split manifest for state {sid}")
                    break
                for j, pname in enumerate(PARAMETER_NAMES):
                    if not np.allclose(np.asarray(a[pname][mask], dtype=float), expected[j], atol=2e-7, rtol=0):
                        _fail(errors, f"{path.name} field {pname} disagrees with split manifest for state {sid}")
                        break
                cidx = cand_index[str(sid)]
                if not np.allclose(np.asarray(a["gy_clean"][mask], dtype=np.float64), g_all[cidx][None, :], atol=2e-7, rtol=0):
                    _fail(errors, f"{path.name} gy_clean disagrees with candidate bank for state {sid}")
                    break

    status = "PASS" if not errors else "FAIL"
    report = {
        "status": status,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "candidate_state_count": len(cand_rows),
        "selected_state_count": len(sel_rows),
        "split_state_counts": {k: len(v) for k, v in split_sets.items()},
        "direct_forward_max_rel_l2": max_rel,
        "q2_min": float(q2[0]),
        "q2_max": float(q2[-1]),
        "q2_domain_locked": True,
    }
    (out / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    update_metadata(out, {"validation": report, "status": "VALIDATED" if status == "PASS" else "VALIDATION_FAILED"})
    print("=" * 100)
    print(f"dataset validation: {status}")
    print(f"errors            : {len(errors)}")
    print(f"warnings          : {len(warnings)}")
    print(f"direct forward    : {max_rel:.6g} max relL2")
    print(f"[done] {out / 'validation_report.json'}")
    print("=" * 100)
    if errors:
        raise RuntimeError("dataset validation failed; inspect validation_report.json")
    return out


def main() -> None:
    a = parse_args()
    run(a.config, a.output_dir, a.smoke_permissive)


if __name__ == "__main__":
    main()
