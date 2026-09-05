#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate Exp35 MLP / TCN / parameter-specific 3P baselines."""
from __future__ import annotations
import argparse, math, json
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--noise-dir",default="noise_0p2pct")
    p.add_argument("--target-rate",type=float,default=.98)
    p.add_argument("--fallback-rate",type=float,default=.95)
    return p.parse_args()


def summarize(g,model):
    row={"model":model,"run_count":len(g)}
    for c in g.columns:
        if c in ("data_seed","train_seed") or not pd.api.types.is_numeric_dtype(g[c]):continue
        v=pd.to_numeric(g[c],errors="coerce").to_numpy(float)
        if np.any(np.isfinite(v)):
            row[c+"_mean"]=float(np.nanmean(v));row[c+"_std"]=float(np.nanstd(v))
            row[c+"_min"]=float(np.nanmin(v));row[c+"_max"]=float(np.nanmax(v))
    return row


def main():
    args=parse_args();man=pd.read_csv(args.manifest);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    data_rows=[]
    seen_data=set()
    for _,r in man.iterrows():
        result=Path(str(r["result_dir"]));prefix=str(r["result_prefix"])
        s=pd.read_csv(result/(prefix+"_summary.csv"))
        s=s[s["noise_dir"].astype(str)==args.noise_dir]
        if len(s)!=1:raise ValueError("expected one %s row in %s"%(args.noise_dir,result))
        row=s.iloc[0].to_dict()
        rows.append({"model":str(r["model"]),"data_seed":int(r["data_seed"]),
                     "train_seed":int(r["train_seed"]),"result_dir":str(result),**row})
        dkey=(int(r["data_seed"]),str(r["data_dir"]))
        if dkey not in seen_data:
            seen_data.add(dkey)
            meta=json.loads((Path(dkey[1])/"metadata.json").read_text(encoding="utf-8"))
            data_rows.append({
                "data_seed":dkey[0],"data_dir":dkey[1],
                "alias_safe_candidate_count":meta["alias_safe_candidate_count"],
                "selected_state_count":meta["selected_state_count"],
                "selected_alias_score_min":meta["selected_alias_score_min"],
                "global_selected_min_g_rms_snr":meta["global_selected_min_g_rms_snr"],
                "max_holdout_to_train_rms_snr":meta["max_holdout_to_train_rms_snr"],
                "train_states":meta["split_counts"]["train"],
                "val_states":meta["split_counts"]["val"],
                "test_states":meta["split_counts"]["test"],
            })
    per=pd.DataFrame(rows).sort_values(["model","data_seed","train_seed"])
    per.to_csv(out/"exp35_per_run_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(data_rows).sort_values("data_seed").to_csv(
        out/"exp35_data_summary.csv",index=False,encoding="utf-8-sig"
    )

    summ=pd.DataFrame([summarize(g,m) for m,g in per.groupby("model")])
    summ.to_csv(out/"exp35_model_summary.csv",index=False,encoding="utf-8-sig")

    gates=[]
    for m,g in per.groupby("model"):
        v=g["joint_recovery_rate"].to_numpy(float)
        gates.append({
            "model":m,"run_count":len(v),
            "joint_mean":float(v.mean()),"joint_std":float(v.std()),
            "joint_min":float(v.min()),"joint_max":float(v.max()),
            "runs_ge_98":int(np.sum(v>=args.target_rate)),
            "mean_ge_98":bool(v.mean()>=args.target_rate),
            "all_ge_95":bool(np.all(v>=args.fallback_rate)),
            "runs_ge_95":int(np.sum(v>=args.fallback_rate)),
        })
    pd.DataFrame(gates).to_csv(out/"exp35_stage_gate.csv",index=False,encoding="utf-8-sig")

    # Matched model deltas by identical data/train seed.
    wide=per.pivot_table(
        index=["data_seed","train_seed"],columns="model",
        values=["a1_recovery_rate","m_recovery_rate","gamma_recovery_rate",
                "joint_recovery_rate","a1_p90_abs_error","m_p90_abs_error",
                "gamma_p90_factor"],
        aggfunc="first"
    )
    rows2=[]
    for idx,row in wide.iterrows():
        rr={"data_seed":idx[0],"train_seed":idx[1]}
        for metric in ["a1_recovery_rate","m_recovery_rate","gamma_recovery_rate",
                       "joint_recovery_rate","a1_p90_abs_error","m_p90_abs_error",
                       "gamma_p90_factor"]:
            for model in ("mlp","tcn","pspec"):
                key=(metric,model)
                if key in row.index:rr[model+"_"+metric]=float(row[key])
            if (metric,"mlp") in row.index and (metric,"tcn") in row.index:
                rr["tcn_minus_mlp_"+metric]=float(row[(metric,"tcn")]-row[(metric,"mlp")])
            if (metric,"tcn") in row.index and (metric,"pspec") in row.index:
                rr["pspec_minus_tcn_"+metric]=float(row[(metric,"pspec")]-row[(metric,"tcn")])
        rows2.append(rr)
    pd.DataFrame(rows2).to_csv(out/"exp35_matched_model_deltas.csv",index=False,encoding="utf-8-sig")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys=sorted(set(zip(per.data_seed,per.train_seed)));x=np.arange(len(keys))
        labels=["D%d/T%d"%k for k in keys]
        for metric,ylabel,fname in [
            ("joint_recovery_rate","joint recovery","exp35_joint_recovery.png"),
            ("a1_recovery_rate","a1 recovery","exp35_a1_recovery.png"),
            ("m_recovery_rate","m recovery","exp35_m_recovery.png"),
            ("gamma_recovery_rate","gamma recovery","exp35_gamma_recovery.png"),
        ]:
            fig,ax=plt.subplots(figsize=(10,5))
            for model in ("mlp","tcn","pspec"):
                gg=per[per.model==model].set_index(["data_seed","train_seed"])
                vals=[float(gg.loc[k,metric]) if k in gg.index else np.nan for k in keys]
                ax.plot(x,vals,marker="o",label=model.upper())
            ax.set_xticks(x);ax.set_xticklabels(labels,rotation=45,ha="right",fontsize=8)
            ax.set_ylim(0,1.01);ax.set_ylabel(ylabel);ax.legend();fig.tight_layout()
            fig.savefig(out/fname,dpi=170);plt.close(fig)
    except Exception as e:
        print("plot skipped:",e)

    print("="*100)
    for _,r in summ.iterrows():
        print("%-6s | a1 %.2f%% | m %.2f%% | gamma %.2f%% | joint %.2f%% (min %.2f%%)"%(
            str(r["model"]).upper(),
            100*float(r["a1_recovery_rate_mean"]),
            100*float(r["m_recovery_rate_mean"]),
            100*float(r["gamma_recovery_rate_mean"]),
            100*float(r["joint_recovery_rate_mean"]),
            100*float(r["joint_recovery_rate_min"]),
        ))
    print("Read first:")
    print(" ",out/"exp35_data_summary.csv")
    print(" ",out/"exp35_model_summary.csv")
    print(" ",out/"exp35_stage_gate.csv")
    print(" ",out/"exp35_matched_model_deltas.csv")
    print("="*100)


if __name__=="__main__":
    main()
