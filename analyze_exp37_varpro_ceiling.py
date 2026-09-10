#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp37 3P variable-projection physical estimator diagnostic.

This is not used to train the neural network and does not remove a1 from the
final ML task. It is a physics-side estimator used to answer whether the same g
contains enough information for (a1,m,gamma) at 0%, 0.2%, and 1% noise.

For each observation:
- profile a1 analytically at fixed (m,gamma);
- search a dense (m,log-gamma) grid;
- continuously refine several distinct best grid basins;
- direct-verify the final fit with float64 forward physics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from exp37_core import varpro_predict_one, make_profile_bank


def write_csv(path, rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with Path(path).open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def level_key(level):
    return min(("0","0.002","0.01"), key=lambda k:abs(float(k)-float(level)))


def max_reps(cfg, level):
    table=cfg["varpro"]["max_eval_repetitions_per_state"]
    k=min(table.keys(),key=lambda x:abs(float(x)-float(level)))
    return int(table[k])


def select_eval_rows(z, max_per_state):
    ids=z["state_id"].astype(np.int64)
    keep=[]
    counts={}
    for i,sid in enumerate(ids):
        sid=int(sid)
        c=counts.get(sid,0)
        if c<max_per_state:
            keep.append(i); counts[sid]=c+1
    return np.asarray(keep,dtype=int)


def quant(x,q):
    return float(np.quantile(np.asarray(x,dtype=np.float64),q))


def summary_for(rows, level):
    rr=[r for r in rows if abs(float(r["noise_level"])-float(level))<1e-12]
    a1=np.array([r["a1_abs_error"] for r in rr],dtype=float)
    mm=np.array([r["m_abs_error"] for r in rr],dtype=float)
    gf=np.array([r["gamma_factor_error"] for r in rr],dtype=float)
    fit=np.array([r["fit_relL2"] for r in rr],dtype=float)
    aok=np.array([r["a1_recovered"] for r in rr],dtype=float)
    mok=np.array([r["m_recovered"] for r in rr],dtype=float)
    gok=np.array([r["gamma_recovered"] for r in rr],dtype=float)
    jok=np.array([r["joint_recovered"] for r in rr],dtype=float)

    by_state={}
    for r in rr:
        by_state.setdefault(int(r["state_id"]),[]).append(int(r["joint_recovered"]))
    state_rates=np.array([np.mean(v) for v in by_state.values()],dtype=float)

    return {
        "noise_level":float(level),"sample_count":len(rr),"physical_state_count":len(by_state),
        "a1_mae":float(np.mean(a1)),"a1_p90_abs":quant(a1,0.90),"a1_p95_abs":quant(a1,0.95),
        "m_mae":float(np.mean(mm)),"m_p90_abs":quant(mm,0.90),"m_p95_abs":quant(mm,0.95),
        "gamma_factor_median":quant(gf,0.50),"gamma_factor_p90":quant(gf,0.90),"gamma_factor_p95":quant(gf,0.95),
        "a1_recovery":float(np.mean(aok)),"m_recovery":float(np.mean(mok)),
        "gamma_recovery":float(np.mean(gok)),"joint_recovery":float(np.mean(jok)),
        "mean_state_joint_recovery":float(np.mean(state_rates)),
        "worst_state_joint_recovery":float(np.min(state_rates)),
        "fraction_states_joint_recovery_ge_0p9":float(np.mean(state_rates>=0.9)),
        "fit_relL2_median":quant(fit,0.50),"fit_relL2_p90":quant(fit,0.90),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="exp37_3p_config.json")
    ap.add_argument("--data-dir",default="data_exp37_3p")
    args=ap.parse_args()

    root=Path(args.data_dir)
    meta=json.loads((root/"metadata.json").read_text(encoding="utf-8"))
    cfg=meta["config_snapshot"]
    tol=cfg["recovery_tolerances"]
    integ=int(cfg["physics_integration_points"])
    vcfg=cfg["varpro"]

    # VarPro uses a denser independent search bank than the alias pre-screen.
    # Cache it in the data directory so reruns do not rebuild the bank.
    vpbank=root/"varpro_profile_bank.npz"
    if vpbank.exists():
        with np.load(vpbank,allow_pickle=False) as z:
            mgrid=z["m_grid"].astype(np.float64)
            lggrid=z["log_gamma_grid"].astype(np.float64)
            background=z["background"].astype(np.float64)
            R=z["resonance"].astype(np.float64)
            r2=z["r2"].astype(np.float64)
    else:
        print("building dense VarPro profile bank...")
        mgrid,lggrid,background,R,r2=make_profile_bank(
            cfg,integ,
            m_points=int(vcfg.get("m_points",161)),
            log_gamma_points=int(vcfg.get("log_gamma_points",161)),
        )
        np.savez_compressed(vpbank,m_grid=mgrid,log_gamma_grid=lggrid,
                            background=background,resonance=R,r2=r2)

    all_rows=[]
    for ndir in sorted(root.glob("noise_*pct")):
        path=ndir/"test.npz"
        with np.load(path,allow_pickle=False) as z:
            level=float(np.asarray(z["noise_level"])[0])
            keep=select_eval_rows(z,max_reps(cfg,level))
            print(f"noise={level:g}: evaluating {len(keep)} samples ({len(np.unique(z['state_id'][keep]))} states)")
            for rank,i in enumerate(keep):
                obs=z["gy"][i].astype(np.float64)
                pred=varpro_predict_one(
                    obs,cfg,background,mgrid,lggrid,R,r2,integ,
                    seed_count=int(vcfg["grid_seed_count_clean"] if abs(level)<1e-15 else vcfg["grid_seed_count_noisy"]),
                    maxiter=int(vcfg["continuous_maxiter"]),
                )
                ta=float(z["a1"][i]); tm=float(z["m"][i]); tg=float(z["gamma"][i])
                pa,pm,pg=pred["pred_a1"],pred["pred_m"],pred["pred_gamma"]
                ae=abs(pa-ta); me=abs(pm-tm)
                gf=max(pg/tg,tg/pg)
                aok=ae<=float(tol["a1_abs"])+1e-12
                mok=me<=float(tol["m_abs"])+1e-12
                gok=gf<=float(tol["gamma_factor"])+1e-12
                all_rows.append({
                    "noise_level":level,"sample_index":int(i),"state_id":int(z["state_id"][i]),
                    "true_a1":ta,"pred_a1":pa,"a1_abs_error":ae,"a1_recovered":int(aok),
                    "true_m":tm,"pred_m":pm,"m_abs_error":me,"m_recovered":int(mok),
                    "true_gamma":tg,"pred_gamma":pg,"gamma_factor_error":gf,"gamma_recovered":int(gok),
                    "joint_recovered":int(aok and mok and gok),
                    "fit_relL2":pred["fit_relL2"],"fit_sse":pred["fit_sse"],"optimizer_source":pred["source"],
                })
                if (rank+1)%50==0 or rank+1==len(keep):
                    print(f"  VarPro: {rank+1}/{len(keep)}")

    write_csv(root/"varpro_predictions.csv",all_rows)
    levels=sorted(set(float(r["noise_level"]) for r in all_rows))
    summary=[summary_for(all_rows,l) for l in levels]
    write_csv(root/"varpro_summary.csv",summary)

    print("="*100)
    print("EXP37 VARPRO PHYSICS DIAGNOSTIC COMPLETE")
    for r in summary:
        print(
            f"noise={r['noise_level']:g} joint={r['joint_recovery']:.4f} "
            f"a1={r['a1_recovery']:.4f} m={r['m_recovery']:.4f} "
            f"gamma={r['gamma_recovery']:.4f} "
            f"gammaP90={r['gamma_factor_p90']:.4f} fitP90={r['fit_relL2_p90']:.4g}"
        )
    print("="*100)


if __name__=="__main__":
    main()
