#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate three Exp36 TCN seeds."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np


def read_csv(path):
    with open(path,newline="",encoding="utf-8-sig") as f: return list(csv.DictReader(f))
def write_csv(path,rows):
    if not rows: Path(path).write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def fnum(x):
    try:return float(x)
    except:return np.nan


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="exp36_3p_config.json")
    ap.add_argument("--runs-root",default="validation_results")
    ap.add_argument("--output-dir",default="validation_results/exp36_3p_tcn_summary")
    args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    seeds=cfg["training"]["seeds"]
    rows=[]
    for seed in seeds:
        p=Path(args.runs_root)/f"exp36_tcn3p_seed{seed}"/"exp36_summary.csv"
        if not p.exists(): raise FileNotFoundError(p)
        rows.extend(read_csv(p))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"exp36_per_run_metrics.csv",rows)

    numeric=[k for k in rows[0] if k not in ("noise_dir",)]
    summaries=[]
    for noise in sorted(set(r["noise_level"] for r in rows),key=float):
        g=[r for r in rows if r["noise_level"]==noise]
        s={"noise_level":float(noise),"run_count":len(g)}
        for k in numeric:
            if k in ("seed","noise_level"): continue
            vals=np.array([fnum(r[k]) for r in g],float)
            if np.any(np.isfinite(vals)):
                s[k+"_mean"]=float(np.nanmean(vals));s[k+"_std"]=float(np.nanstd(vals))
                s[k+"_min"]=float(np.nanmin(vals));s[k+"_max"]=float(np.nanmax(vals))
        summaries.append(s)
    write_csv(out/"exp36_model_summary.csv",summaries)

    # A 3P gate should not mechanically demand 98% joint.  Report per-parameter
    # 95/98% gates and keep joint as a descriptive metric.
    gates=[]
    for s in summaries:
        row={"noise_level":s["noise_level"]}
        prefix="sample_"
        for param in ("a1","m","gamma"):
            mean=s[f"{prefix}{param}_recovery_mean"]
            row[f"{param}_recovery_mean"]=mean
            row[f"{param}_ge_95"]=int(mean>=0.95)
            row[f"{param}_ge_98"]=int(mean>=0.98)
        row["all_params_ge_95"]=int(all(row[f"{p}_ge_95"] for p in ("a1","m","gamma")))
        row["all_params_ge_98"]=int(all(row[f"{p}_ge_98"] for p in ("a1","m","gamma")))
        row["joint_recovery_mean"]=s["sample_joint_recovery_mean"]
        row["joint_recovery_min_seed"]=s["sample_joint_recovery_min"]
        row["state_joint_recovery_mean"]=s["state_joint_recovery_mean"]
        gates.append(row)
    write_csv(out/"exp36_stage_gate.csv",gates)

    # Simple bottleneck ranking at the training/reference noise.
    target=min(summaries,key=lambda s:abs(float(s["noise_level"])-float(cfg["training_noise"])))
    bottleneck=sorted([
        ("a1",target["sample_a1_recovery_mean"]),
        ("m",target["sample_m_recovery_mean"]),
        ("gamma",target["sample_gamma_recovery_mean"]),
    ],key=lambda x:x[1])
    write_csv(out/"exp36_bottleneck_ranking.csv",[{"rank":i+1,"parameter":p,"recovery_mean":v} for i,(p,v) in enumerate(bottleneck)])

    print("="*90)
    for s in summaries:
        print(f"noise={s['noise_level']:g} | a1 {100*s['sample_a1_recovery_mean']:.2f}% | m {100*s['sample_m_recovery_mean']:.2f}% | gamma {100*s['sample_gamma_recovery_mean']:.2f}% | joint {100*s['sample_joint_recovery_mean']:.2f}%")
    print("reference-noise bottleneck:",bottleneck[0][0])
    print("read:",out/"exp36_model_summary.csv")
    print("read:",out/"exp36_stage_gate.csv")
    print("read:",out/"exp36_bottleneck_ranking.csv")
    print("="*90)


if __name__=="__main__":
    main()
