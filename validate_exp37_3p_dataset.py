#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Strict post-build validation for Exp37 3P data.

This validator independently checks:
- physical-state split integrity and NPZ consistency;
- direct forward round-trip;
- selected-state g separation;
- hard parameter-space coverage;
- hard holdout->train parameter/g coverage;
- continuous recovery-aligned alias margins and unacceptable constraints;
- selected states are all above the chosen continuous-margin cutoff.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from exp37_core import (
    forward64_batched, normalize_free_params, coverage_radius,
    pairwise_g_distance, alias_margin_direct, unacceptable,
)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def row_param(r):
    return np.array([
        float(r["a1"]), float(r["a2"]), float(r["a3"]),
        float(r["m"]), float(r["gamma"])
    ], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp37_3p_config.json")
    ap.add_argument("--data-dir", default="data_exp37_3p")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.quick:
        # only validation thresholds that depend on selection config are read from metadata snapshot
        pass
    root = Path(args.data_dir)
    meta = json.loads((root/"metadata.json").read_text(encoding="utf-8"))
    ecfg = meta["config_snapshot"]
    scfg = ecfg["selection"]
    tol = ecfg["recovery_tolerances"]
    integ = int(ecfg["physics_integration_points"])
    chosen_cutoff = float(meta["chosen_continuous_recovery_alias_cutoff"])
    checks = []
    errors = []
    warnings = []

    selected_rows = read_csv(root/"selected_states.csv")
    candidate_rows = read_csv(root/"candidate_states.csv")
    split_rows = read_csv(root/"split_manifest.csv")
    alias_rows = {int(r["state_id"]):r for r in read_csv(root/"profiled_alias_selected.csv")}

    selected_ids = np.array([int(r["state_id"]) for r in selected_rows], dtype=int)
    selected_params = np.stack([row_param(r) for r in selected_rows])
    candidate_params = np.stack([row_param(r) for r in candidate_rows])
    split_of = {int(r["state_id"]):r["split"] for r in split_rows}

    # Counts and split leakage.
    expected = int(scfg["selected_count"])
    ok = len(selected_ids) == expected == len(set(map(int, selected_ids)))
    checks.append({"check":"selected_count_unique","status":"PASS" if ok else "FAIL","value":len(selected_ids)})
    if not ok:
        errors.append("selected count/uniqueness mismatch")

    sets = {sp:{int(r["state_id"]) for r in split_rows if r["split"]==sp} for sp in ("train","val","test")}
    for name, ov in (
        ("train_val", sets["train"] & sets["val"]),
        ("train_test", sets["train"] & sets["test"]),
        ("val_test", sets["val"] & sets["test"]),
    ):
        ok = len(ov)==0
        checks.append({"check":f"state_leakage_{name}","status":"PASS" if ok else "FAIL","value":len(ov)})
        if not ok:
            errors.append(f"physical-state leakage {name}")

    # Physical parameter consistency between selected and manifest.
    selected_map = {int(r["state_id"]):row_param(r) for r in selected_rows}
    manifest_ok = set(selected_map) == set(split_of)
    if manifest_ok:
        for r in split_rows:
            if not np.allclose(row_param(r), selected_map[int(r["state_id"])], rtol=0, atol=2e-12):
                manifest_ok = False
                break
    checks.append({"check":"manifest_matches_selected","status":"PASS" if manifest_ok else "FAIL","value":int(manifest_ok)})
    if not manifest_ok:
        errors.append("split manifest does not exactly match selected-state parameters")

    # Direct selected g and selected-state separation.
    selected_g = forward64_batched(selected_params, integ)
    Dg = pairwise_g_distance(selected_g, float(ecfg["reference_noise"]))
    min_sep = float(np.min(Dg))
    req_sep = float(scfg["min_selected_g_separation_rms_snr"])
    ok = min_sep + 1e-8 >= req_sep
    checks.append({"check":"min_selected_g_separation","status":"PASS" if ok else "FAIL","value":min_sep})
    if not ok:
        errors.append(f"selected g separation {min_sep:g} < {req_sep:g}")

    # Hard parameter coverage against the complete candidate pool.
    cand_norm = normalize_free_params(candidate_params, ecfg)
    sel_norm = normalize_free_params(selected_params, ecfg)
    cover = coverage_radius(cand_norm, sel_norm)
    req_cover = float(scfg["max_parameter_cover_radius"])
    ok = cover <= req_cover + 1e-10
    checks.append({"check":"parameter_cover_radius","status":"PASS" if ok else "FAIL","value":cover})
    if not ok:
        errors.append(f"parameter cover radius {cover:g} > {req_cover:g}")

    # Holdout -> train coverage, directly recomputed.
    train_idx = np.array([i for i,sid in enumerate(selected_ids) if split_of[int(sid)]=="train"], dtype=int)
    hold_idx = np.array([i for i,sid in enumerate(selected_ids) if split_of[int(sid)]!="train"], dtype=int)
    Dp = np.linalg.norm(sel_norm[:,None,:]-sel_norm[None,:,:], axis=2)
    if len(hold_idx) and len(train_idx):
        max_p = float(np.max(np.min(Dp[np.ix_(hold_idx,train_idx)], axis=1)))
        max_g = float(np.max(np.min(Dg[np.ix_(hold_idx,train_idx)], axis=1)))
    else:
        max_p = max_g = math.inf
    req_p = float(scfg["max_holdout_to_train_parameter_distance"])
    req_g = float(scfg["max_holdout_to_train_rms_snr_hard"])
    okp = max_p <= req_p + 1e-10
    okg = max_g <= req_g + 1e-8
    checks.append({"check":"max_holdout_to_train_parameter_distance","status":"PASS" if okp else "FAIL","value":max_p})
    checks.append({"check":"max_holdout_to_train_rms_snr_hard","status":"PASS" if okg else "FAIL","value":max_g})
    if not okp:
        errors.append(f"holdout->train parameter distance {max_p:g} > {req_p:g}")
    if not okg:
        errors.append(f"holdout->train g distance {max_g:g} > {req_g:g}")
    preferred = float(scfg["max_holdout_to_train_rms_snr_preferred"])
    if max_g > preferred:
        warnings.append(f"holdout->train g distance {max_g:g} exceeds preferred {preferred:g} but passes hard limit")

    # Axis bin coverage for train.
    bins = int(scfg["coverage_axis_bins"])
    minimum_bin = int(scfg["minimum_train_count_per_axis_bin"])
    trn = sel_norm[train_idx]
    min_axis_count = 10**9
    for j in range(3):
        b = np.floor(np.clip(trn[:,j],0,1-1e-14)*bins).astype(int)
        counts = np.bincount(b,minlength=bins)[:bins]
        min_axis_count = min(min_axis_count, int(counts.min()))
    ok = min_axis_count >= minimum_bin
    checks.append({"check":"minimum_train_count_in_any_axis_bin","status":"PASS" if ok else "FAIL","value":min_axis_count})
    if not ok:
        errors.append("train axis-bin coverage below hard minimum")

    # Continuous alias verification from direct float64 physics.
    vcfg = ecfg["validation"]
    rtol = float(vcfg["alias_margin_recompute_rtol"])
    atol = float(vcfg["alias_margin_recompute_atol"])
    max_margin_err = 0.0
    min_saved = math.inf
    bad_alias = 0
    below_cutoff = 0
    for sid, true_p in selected_map.items():
        r = alias_rows.get(sid)
        if r is None:
            errors.append(f"missing profiled alias for state {sid}")
            continue
        alias_p = true_p.copy()
        alias_p[0] = float(r["alias_a1_refined"])
        alias_p[3] = float(r["alias_m_refined"])
        alias_p[4] = float(r["alias_gamma_refined"])
        if not unacceptable(true_p, alias_p, tol):
            bad_alias += 1
        direct = alias_margin_direct(true_p, alias_p, ecfg, integ)
        saved = float(r["recovery_alias_margin_refined"])
        max_margin_err = max(max_margin_err, abs(direct-saved))
        min_saved = min(min_saved, saved)
        if not np.isclose(direct, saved, rtol=rtol, atol=atol):
            errors.append(f"state {sid}: refined alias direct margin mismatch {direct} vs {saved}")
        if saved + 1e-12 < chosen_cutoff:
            below_cutoff += 1
    oka = bad_alias == 0
    okc = below_cutoff == 0
    checks.append({"check":"all_refined_aliases_unacceptable","status":"PASS" if oka else "FAIL","value":bad_alias})
    checks.append({"check":"all_selected_above_continuous_cutoff","status":"PASS" if okc else "FAIL","value":below_cutoff})
    checks.append({"check":"max_refined_margin_direct_recompute_abs_error","status":"PASS" if max_margin_err <= atol + rtol*max(1.0,min_saved) else "INFO","value":max_margin_err})
    if not oka:
        errors.append(f"{bad_alias} refined aliases do not satisfy recovery-failure condition")
    if not okc:
        errors.append(f"{below_cutoff} selected states lie below the chosen continuous cutoff")

    # NPZ integrity for all three noise levels.
    required = {"gy","g_clean","a1","a2","a3","m","gamma","state_id","noise_level"}
    expected_by_split = {sp:set(ids) for sp,ids in sets.items()}
    forward_saved = {}
    for ndir in sorted(root.glob("noise_*pct")):
        for sp in ("train","val","test"):
            path = ndir/f"{sp}.npz"
            if not path.exists():
                errors.append(f"missing {path}")
                continue
            with np.load(path, allow_pickle=False) as z:
                missing = required-set(z.files)
                if missing:
                    errors.append(f"{path}: missing {sorted(missing)}")
                    continue
                n = len(z["state_id"])
                lengths_ok = all(len(z[k])==n for k in required)
                finite_ok = all(np.isfinite(z[k]).all() for k in ("gy","g_clean","a1","a2","a3","m","gamma","noise_level"))
                ids = z["state_id"].astype(int)
                ids_ok = set(map(int,np.unique(ids))) == expected_by_split[sp]
                param_ok = True
                clean_ok = True
                for sid in np.unique(ids):
                    mask = ids==sid
                    obs = np.array([z["a1"][mask][0],z["a2"][mask][0],z["a3"][mask][0],
                                    z["m"][mask][0],z["gamma"][mask][0]], dtype=np.float64)
                    exp = selected_map[int(sid)]
                    if not np.allclose(obs, exp, rtol=0, atol=2e-7):
                        param_ok=False; break
                    gc = z["g_clean"][mask]
                    if len(gc)>1 and not np.allclose(gc,gc[:1],rtol=0,atol=2e-7):
                        clean_ok=False; break
                    forward_saved.setdefault(int(sid), (exp, gc[0].astype(np.float64)))
                for suffix, ok, val in (
                    ("array_lengths",lengths_ok,n),("finite",finite_ok,int(finite_ok)),
                    ("state_ids_match_manifest",ids_ok,len(np.unique(ids))),
                    ("parameters_match_manifest",param_ok,int(param_ok)),
                    ("g_clean_constant_within_state",clean_ok,int(clean_ok)),
                ):
                    checks.append({"check":f"{ndir.name}_{sp}_{suffix}","status":"PASS" if ok else "FAIL","value":val})
                    if not ok:
                        errors.append(f"{ndir.name}/{sp}: {suffix} failed")

    # Saved g_clean vs direct float64/project-rounded forward.
    if forward_saved:
        items = sorted(forward_saved.items())
        p = np.stack([x[1][0] for x in items])
        saved = np.stack([x[1][1] for x in items])
        calc = forward64_batched(p, integ).astype(np.float32).astype(np.float64)
        rel = float(np.linalg.norm(calc-saved)/max(np.linalg.norm(saved),1e-30))
        req = float(vcfg["float64_roundtrip_relL2_max"])
        ok = rel <= req
        checks.append({"check":"saved_gclean_forward64_roundtrip_relL2","status":"PASS" if ok else "FAIL","value":rel})
        if not ok:
            errors.append(f"saved g_clean forward mismatch {rel:g} > {req:g}")

    payload = {"pass":len(errors)==0,"errors":errors,"warnings":warnings}
    (root/"postbuild_validation.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    write_csv(root/"postbuild_validation.csv", checks)

    print("="*100)
    print("EXP37 POST-BUILD VALIDATION:", "PASS" if not errors else "FAIL")
    print("errors:", errors)
    print("warnings:", warnings)
    print("parameter cover radius:", cover)
    print("max holdout->train param/g:", max_p, max_g)
    print("max alias direct-recompute abs error:", max_margin_err)
    print("="*100)
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
