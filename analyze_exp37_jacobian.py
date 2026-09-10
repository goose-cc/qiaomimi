#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp37 Jacobian conditioning diagnostic for selected 3P physical states.

Columns are scaled to the actual recovery tolerances:
  z1 = a1 / tol_a1
  z2 = m / tol_m
  z3 = ln(gamma) / ln(gamma_factor_tol)

The Jacobian is additionally normalized by reference-noise RMS and sqrt(Nq), so
its singular values are in approximate RMS-SNR per one tolerance-unit motion.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from exp37_core import forward64, rms_rows


def read_csv(path):
    with Path(path).open(newline="",encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with Path(path).open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def row_param(r):
    return np.array([float(r["a1"]),float(r["a2"]),float(r["a3"]),
                     float(r["m"]),float(r["gamma"])],dtype=np.float64)


def finite_diff_scaled_jacobian(p, cfg, integ):
    tol=cfg["recovery_tolerances"]
    pr=cfg["parameter_ranges"]
    h=float(cfg["jacobian"]["step_in_tolerance_units"])
    scales=np.array([float(tol["a1_abs"]),float(tol["m_abs"]),math.log(float(tol["gamma_factor"]))],dtype=np.float64)

    g0=forward64(p[None,:],integration_points=integ)[0]
    q=len(g0)
    J=np.empty((q,3),dtype=np.float64)

    # Each derivative is with respect to one *tolerance unit*.
    # Use centered finite differences when possible, otherwise one-sided.
    for j in range(3):
        pp=p.copy(); pm=p.copy()
        if j==0:
            delta=h*scales[j]
            lo,hi=map(float,pr["a1"])
            can_p=p[0]+delta<=hi
            can_m=p[0]-delta>=lo
            if can_p and can_m:
                pp[0]+=delta; pm[0]-=delta
                gp,gm=forward64(np.vstack([pp,pm]),integration_points=integ)
                J[:,j]=(gp-gm)/(2*h)
            elif can_p:
                pp[0]+=delta
                gp=forward64(pp[None,:],integration_points=integ)[0]
                J[:,j]=(gp-g0)/h
            else:
                pm[0]-=delta
                gm=forward64(pm[None,:],integration_points=integ)[0]
                J[:,j]=(g0-gm)/h
        elif j==1:
            delta=h*scales[j]
            lo,hi=map(float,pr["m"])
            can_p=p[3]+delta<=hi
            can_m=p[3]-delta>=lo
            if can_p and can_m:
                pp[3]+=delta; pm[3]-=delta
                gp,gm=forward64(np.vstack([pp,pm]),integration_points=integ)
                J[:,j]=(gp-gm)/(2*h)
            elif can_p:
                pp[3]+=delta
                gp=forward64(pp[None,:],integration_points=integ)[0]
                J[:,j]=(gp-g0)/h
            else:
                pm[3]-=delta
                gm=forward64(pm[None,:],integration_points=integ)[0]
                J[:,j]=(g0-gm)/h
        else:
            dlog=h*scales[j]
            lg=math.log(float(p[4]))
            lo,hi=math.log(float(pr["gamma"][0])),math.log(float(pr["gamma"][1]))
            can_p=lg+dlog<=hi
            can_m=lg-dlog>=lo
            if can_p and can_m:
                pp[4]=math.exp(lg+dlog); pm[4]=math.exp(lg-dlog)
                gp,gm=forward64(np.vstack([pp,pm]),integration_points=integ)
                J[:,j]=(gp-gm)/(2*h)
            elif can_p:
                pp[4]=math.exp(lg+dlog)
                gp=forward64(pp[None,:],integration_points=integ)[0]
                J[:,j]=(gp-g0)/h
            else:
                pm[4]=math.exp(lg-dlog)
                gm=forward64(pm[None,:],integration_points=integ)[0]
                J[:,j]=(g0-gm)/h

    den=max(float(cfg["reference_noise"])*float(rms_rows(g0[None,:])[0])*math.sqrt(q),1e-30)
    Jn=J/den
    return g0,Jn


def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den>0 else math.nan


def summarize(rows, split):
    subset=[r for r in rows if split=="all" or r["split"]==split]
    def vals(k):
        return np.array([float(r[k]) for r in subset],dtype=np.float64)
    out={"split":split,"state_count":len(subset)}
    for k in ("sigma_min_3p","condition_3p","sigma_min_2p_a1_gamma","condition_2p_a1_gamma",
              "cos_a1_m","cos_a1_loggamma","cos_m_loggamma"):
        x=vals(k)
        out[k+"_median"]=float(np.median(x))
        out[k+"_p10"]=float(np.quantile(x,0.10))
        out[k+"_p90"]=float(np.quantile(x,0.90))
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="exp37_3p_config.json")
    ap.add_argument("--data-dir",default="data_exp37_3p")
    args=ap.parse_args()

    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    meta=json.loads((Path(args.data_dir)/"metadata.json").read_text(encoding="utf-8"))
    cfg=meta["config_snapshot"]
    integ=int(cfg["physics_integration_points"])
    rows_in=read_csv(Path(args.data_dir)/"selected_states.csv")
    tol=cfg["recovery_tolerances"]
    scale=np.array([float(tol["a1_abs"]),float(tol["m_abs"]),math.log(float(tol["gamma_factor"]))],dtype=np.float64)
    eps=float(cfg["jacobian"]["condition_eps"])

    out=[]
    for i,r in enumerate(rows_in):
        p=row_param(r)
        _,J=finite_diff_scaled_jacobian(p,cfg,integ)
        U,S,Vt=np.linalg.svd(J,full_matrices=False)
        smax=float(S[0]); smin=float(S[-1])
        cond=smax/max(smin,eps)
        J2=J[:,[0,2]]
        S2=np.linalg.svd(J2,compute_uv=False)
        cond2=float(S2[0]/max(float(S2[-1]),eps))
        v=Vt[-1].copy()
        # Stable sign convention: gamma component positive; if zero, a1 positive.
        if v[2] < 0 or (abs(v[2])<1e-15 and v[0]<0):
            v=-v
        out.append({
            "state_id":int(r["state_id"]),"split":r["split"],
            "a1":p[0],"m":p[3],"gamma":p[4],
            "sigma_max_3p":smax,"sigma_mid_3p":float(S[1]),"sigma_min_3p":smin,
            "condition_3p":cond,
            "sigma_min_2p_a1_gamma":float(S2[-1]),"condition_2p_a1_gamma":cond2,
            "cos_a1_m":cosine(J[:,0],J[:,1]),
            "cos_a1_loggamma":cosine(J[:,0],J[:,2]),
            "cos_m_loggamma":cosine(J[:,1],J[:,2]),
            "null_a1_tol_units":float(v[0]),
            "null_m_tol_units":float(v[1]),
            "null_loggamma_tol_units":float(v[2]),
            "null_da1_actual":float(v[0]*scale[0]),
            "null_dm_actual":float(v[1]*scale[1]),
            "null_dloggamma_actual":float(v[2]*scale[2]),
            "null_gamma_factor_signed":float(math.exp(v[2]*scale[2])),
        })
        if (i+1)%50==0 or i+1==len(rows_in):
            print(f"  Jacobian states: {i+1}/{len(rows_in)}")

    root=Path(args.data_dir)
    write_csv(root/"jacobian_selected.csv",out)
    summary=[summarize(out,s) for s in ("all","train","val","test")]
    write_csv(root/"jacobian_summary.csv",summary)

    print("="*100)
    print("EXP37 JACOBIAN DIAGNOSTIC COMPLETE")
    allrow=summary[0]
    print("median sigma_min 3P:",allrow["sigma_min_3p_median"])
    print("median condition 3P:",allrow["condition_3p_median"])
    print("median condition 2P(a1,gamma):",allrow["condition_2p_a1_gamma_median"])
    print("median cos(a1,loggamma):",allrow["cos_a1_loggamma_median"])
    print("median cos(m,loggamma):",allrow["cos_m_loggamma_median"])
    print("="*100)


if __name__=="__main__":
    main()
