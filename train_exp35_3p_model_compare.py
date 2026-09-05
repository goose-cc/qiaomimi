#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp35 train one of three direct 3P inverse-regression baselines.

Input:
    g(q^2)
Output:
    a1, m, gamma

Models:
    mlp, tcn, pspec

All models use exactly the same dataset, normalization, loss, optimizer,
checkpoint protocol and evaluation tolerances.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from TripleInverseModels import (
    TripleInverseMLP,
    TripleInverseTCN,
    TripleInverseParameterSpecific,
)
from mc_physics import scaled_curves_numpy
from mc_pool_config import DEFAULT_PHYSICS


class TripleNpzDataset(Dataset):
    def __init__(self,path,max_samples=0):
        self.path=Path(path)
        with np.load(self.path,allow_pickle=False) as d:
            req=("gy_noisy","a1","m","gamma")
            miss=[k for k in req if k not in d]
            if miss: raise KeyError("%s missing %s"%(path,miss))
            self.gy=np.asarray(d["gy_noisy"],np.float32)
            self.a1=np.asarray(d["a1"],np.float32).reshape(-1)
            self.m=np.asarray(d["m"],np.float32).reshape(-1)
            self.gamma=np.asarray(d["gamma"],np.float32).reshape(-1)
            self.parameters=np.asarray(d["parameters"],np.float32) if "parameters" in d else None
            self.gy_clean=np.asarray(d["gy_clean"],np.float32) if "gy_clean" in d else None
            self.fx=np.asarray(d["fx"],np.float32) if "fx" in d else None
        n=len(self.a1)
        if self.gy.ndim!=2 or self.gy.shape[0]!=n: raise ValueError("gy shape mismatch")
        if not (len(self.m)==len(self.gamma)==n): raise ValueError("target length mismatch")
        if np.any(~np.isfinite(self.gy)) or np.any(self.gamma<=0): raise ValueError("invalid data")
        if max_samples and n>int(max_samples):
            idx=np.arange(int(max_samples))
            for k in ("gy","a1","m","gamma","parameters","gy_clean","fx"):
                v=getattr(self,k)
                if v is not None: setattr(self,k,v[idx])
    def __len__(self): return len(self.a1)
    def __getitem__(self,i):
        return {
            "gy":torch.from_numpy(self.gy[i]),
            "a1":torch.tensor(self.a1[i],dtype=torch.float32),
            "m":torch.tensor(self.m[i],dtype=torch.float32),
            "gamma":torch.tensor(self.gamma[i],dtype=torch.float32),
        }


def write_csv(path,rows):
    if not rows:return
    fields=[];seen=set()
    for r in rows:
        for k in r:
            if k not in seen:seen.add(k);fields.append(k)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def resolve_device(text):
    if str(text).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(text)


def parse_int_tuple(text):
    vals=tuple(int(x.strip()) for x in str(text).split(",") if x.strip())
    if not vals:return ()
    return vals


def parse_args():
    p=argparse.ArgumentParser(
        description="Exp35 3P model comparison trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",choices=("mlp","tcn","pspec"),required=True)
    p.add_argument("--data-dir",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--result-prefix",default="exp35")
    p.add_argument("--train-noise-dir",default="noise_0p2pct")
    p.add_argument("--device",default="cuda")
    p.add_argument("--seed",type=int,default=20260880)
    p.add_argument("--epochs",type=int,default=80)
    p.add_argument("--batch-size",type=int,default=256)
    p.add_argument("--eval-batch-size",type=int,default=512)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--dropout",type=float,default=.05)
    p.add_argument("--patience",type=int,default=15)
    p.add_argument("--num-workers",type=int,default=0)
    p.add_argument("--hidden-sizes",default="256,256,128")
    p.add_argument("--channels",type=int,default=64)
    p.add_argument("--dilations",default="1,2,4,8,16")
    p.add_argument("--head-hidden",type=int,default=128)
    p.add_argument("--amplitude-bins",type=int,default=16)
    p.add_argument("--amplitude-hidden",type=int,default=96)
    p.add_argument("--parameter-bounds-source",choices=("selected","original"),default="selected")

    p.add_argument("--a1-loss-weight",type=float,default=1.0)
    p.add_argument("--m-loss-weight",type=float,default=1.0)
    p.add_argument("--gamma-loss-weight",type=float,default=1.0)
    p.add_argument("--a1-success-tol",type=float,default=.005)
    p.add_argument("--m-success-tol",type=float,default=.03)
    p.add_argument("--gamma-success-factor",type=float,default=1.20)

    p.add_argument("--max-train-samples",type=int,default=0)
    p.add_argument("--max-val-samples",type=int,default=0)
    p.add_argument("--max-test-samples",type=int,default=0)
    p.add_argument("--skip-physics-metrics",action="store_true")
    p.add_argument("--physics-integration-points",type=int,default=512)
    return p.parse_args()


def input_norm(x):
    x=np.asarray(x,np.float64)
    mean=x.mean(0);scale=x.std(0)
    gr=float(np.sqrt(np.mean(x*x)));floor=max(gr*1e-6,1e-12)
    scale=np.maximum(scale,floor)
    return mean.astype(np.float32),scale.astype(np.float32),gr


def ranges_from_meta(meta,source):
    key="selected_parameter_ranges" if source=="selected" else "original_parameter_ranges"
    r=meta[key]
    return (
        tuple(float(x) for x in r["a1"]),
        tuple(float(x) for x in r["m"]),
        tuple(float(x) for x in r["gamma"]),
    )


def make_model(args,n,ranges):
    a1r,mr,gr=ranges
    common=dict(
        input_points=n,dropout=args.dropout,
        a1_min=a1r[0],a1_max=a1r[1],
        m_min=mr[0],m_max=mr[1],
        gamma_min=gr[0],gamma_max=gr[1],
    )
    if args.model=="mlp":
        return TripleInverseMLP(hidden_sizes=parse_int_tuple(args.hidden_sizes),**common)
    tcommon=dict(
        channels=args.channels,dilations=parse_int_tuple(args.dilations),
        head_hidden=args.head_hidden,use_coordinate_channel=True,**common
    )
    if args.model=="tcn":
        return TripleInverseTCN(**tcommon)
    return TripleInverseParameterSpecific(
        amplitude_bins=args.amplitude_bins,amplitude_hidden=args.amplitude_hidden,**tcommon
    )


def loss_parts(pred,a1,m,gamma,spans,weights):
    pa,pm,plg=pred
    a1span,mspan,lgspan=spans
    la=torch.mean(((pa-a1)/a1span)**2)
    lm=torch.mean(((pm-m)/mspan)**2)
    lg=torch.mean(((plg-torch.log10(gamma))/lgspan)**2)
    total=weights[0]*la+weights[1]*lm+weights[2]*lg
    return total,la,lm,lg


def gamma_factor(pg,tg):
    pg=np.clip(np.asarray(pg,np.float64),1e-30,None)
    tg=np.clip(np.asarray(tg,np.float64),1e-30,None)
    return np.maximum(pg/tg,tg/pg)


def metric_arrays(pa,ta,pm,tm,pg,tg,a1tol,mtol,gfact):
    ea=np.abs(pa-ta); em=np.abs(pm-tm)
    gf=gamma_factor(pg,tg)
    gl=np.abs(np.log10(np.clip(pg,1e-30,None))-np.log10(np.clip(tg,1e-30,None)))
    ar=ea<=a1tol; mr=em<=mtol; gr=gf<=gfact; jr=ar&mr&gr
    return ea,em,gf,gl,ar,mr,gr,jr


def metrics(pa,ta,pm,tm,pg,tg,a1tol,mtol,gfact):
    ea,em,gf,gl,ar,mr,gr,jr=metric_arrays(
        pa,ta,pm,tm,pg,tg,a1tol,mtol,gfact
    )
    return {
        "count":int(len(ea)),
        "a1_mae":float(np.mean(ea)),
        "a1_rmse":float(np.sqrt(np.mean((pa-ta)**2))),
        "a1_p90_abs_error":float(np.quantile(ea,.90)),
        "a1_p95_abs_error":float(np.quantile(ea,.95)),
        "a1_recovery_rate":float(np.mean(ar)),
        "m_mae":float(np.mean(em)),
        "m_rmse":float(np.sqrt(np.mean((pm-tm)**2))),
        "m_p90_abs_error":float(np.quantile(em,.90)),
        "m_p95_abs_error":float(np.quantile(em,.95)),
        "m_recovery_rate":float(np.mean(mr)),
        "gamma_log10_mae":float(np.mean(gl)),
        "gamma_p90_factor":float(np.quantile(gf,.90)),
        "gamma_p95_factor":float(np.quantile(gf,.95)),
        "gamma_recovery_rate":float(np.mean(gr)),
        "joint_recovery_rate":float(np.mean(jr)),
    }


def eval_loader(model,loader,device,spans,weights,args):
    model.eval();total=0;n=0
    PA=[];TA=[];PM=[];TM=[];PG=[];TG=[]
    with torch.no_grad():
        for b in loader:
            gy=b["gy"].to(device);a=b["a1"].to(device);m=b["m"].to(device);g=b["gamma"].to(device)
            pred=model(gy);loss,*_=loss_parts(pred,a,m,g,spans,weights)
            total+=float(loss.item())*len(a);n+=len(a)
            PA.append(pred[0].cpu().numpy());TA.append(a.cpu().numpy())
            PM.append(pred[1].cpu().numpy());TM.append(m.cpu().numpy())
            PG.append(torch.pow(10.0,pred[2]).cpu().numpy());TG.append(g.cpu().numpy())
    pa=np.concatenate(PA).astype(np.float64);ta=np.concatenate(TA).astype(np.float64)
    pm=np.concatenate(PM).astype(np.float64);tm=np.concatenate(TM).astype(np.float64)
    pg=np.concatenate(PG).astype(np.float64);tg=np.concatenate(TG).astype(np.float64)
    return total/max(n,1),metrics(
        pa,ta,pm,tm,pg,tg,args.a1_success_tol,args.m_success_tol,args.gamma_success_factor
    )


def physics_metrics(pred_params,true_fx,true_g,points):
    from dataclasses import replace
    cfg=replace(DEFAULT_PHYSICS,q2_points=true_g.shape[1])
    fxp,gp=scaled_curves_numpy(pred_params,integration_points=int(points),config=cfg)
    frel=np.linalg.norm(np.asarray(fxp,np.float64)-np.asarray(true_fx,np.float64),axis=1)/np.maximum(
        np.linalg.norm(np.asarray(true_fx,np.float64),axis=1),1e-30
    )
    grel=np.linalg.norm(np.asarray(gp,np.float64)-np.asarray(true_g,np.float64),axis=1)/np.maximum(
        np.linalg.norm(np.asarray(true_g,np.float64),axis=1),1e-30
    )
    return frel,grel


def predict_dataset(model,ds,device,batch,args):
    loader=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=args.num_workers)
    model.eval(); PA=[];PM=[];PG=[]
    with torch.no_grad():
        for b in loader:
            pred=model(b["gy"].to(device))
            PA.append(pred[0].cpu().numpy());PM.append(pred[1].cpu().numpy())
            PG.append(torch.pow(10.0,pred[2]).cpu().numpy())
    pa=np.concatenate(PA).astype(np.float64)
    pm=np.concatenate(PM).astype(np.float64)
    pg=np.concatenate(PG).astype(np.float64)
    ta=ds.a1.astype(np.float64);tm=ds.m.astype(np.float64);tg=ds.gamma.astype(np.float64)
    met=metrics(pa,ta,pm,tm,pg,tg,args.a1_success_tol,args.m_success_tol,args.gamma_success_factor)
    frel=np.full(len(pa),np.nan);grel=np.full(len(pa),np.nan)
    if not args.skip_physics_metrics and ds.parameters is not None and ds.fx is not None and ds.gy_clean is not None:
        pp=np.asarray(ds.parameters,np.float64).copy()
        pp[:,0]=pa;pp[:,3]=pm;pp[:,4]=pg
        # Physics calculation in chunks to control RAM.
        fparts=[];gparts=[]
        for s in range(0,len(pp),512):
            f,g=physics_metrics(
                pp[s:s+512],ds.fx[s:s+512],ds.gy_clean[s:s+512],
                args.physics_integration_points
            )
            fparts.append(f);gparts.append(g)
        frel=np.concatenate(fparts);grel=np.concatenate(gparts)
        met.update({
            "median_f_relative_l2":float(np.median(frel)),
            "p90_f_relative_l2":float(np.quantile(frel,.90)),
            "p95_f_relative_l2":float(np.quantile(frel,.95)),
            "median_g_clean_relative_l2":float(np.median(grel)),
            "p90_g_clean_relative_l2":float(np.quantile(grel,.90)),
            "p95_g_clean_relative_l2":float(np.quantile(grel,.95)),
        })
    return pa,pm,pg,met,frel,grel


def noise_dirs(data_dir):
    out=[]
    for p in Path(data_dir).glob("noise_*pct"):
        if (p/"test.npz").exists():
            # parse percentage label for sorting only
            s=p.name[len("noise_"):-len("pct")].replace("p",".")
            try: val=float(s)/100.0
            except: val=math.nan
            out.append((val,p))
    return sorted(out,key=lambda x:(math.inf if not np.isfinite(x[0]) else x[0]))


def main():
    args=parse_args();set_seed(args.seed);device=resolve_device(args.device)
    data_dir=Path(args.data_dir);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((data_dir/"metadata.json").read_text(encoding="utf-8"))
    if meta.get("mode")!="a1mgamma":raise ValueError("Exp35 requires a1mgamma data")

    tr=TripleNpzDataset(data_dir/args.train_noise_dir/"train.npz",args.max_train_samples)
    va=TripleNpzDataset(data_dir/args.train_noise_dir/"val.npz",args.max_val_samples)
    mean,scale,global_rms=input_norm(tr.gy)
    ranges=ranges_from_meta(meta,args.parameter_bounds_source)
    model=make_model(args,tr.gy.shape[1],ranges)
    model.set_input_normalization(mean,scale,global_rms)
    model=model.to(device)

    a1r,mr,gr=ranges
    spans=(a1r[1]-a1r[0],mr[1]-mr[0],math.log10(gr[1])-math.log10(gr[0]))
    weights=(args.a1_loss_weight,args.m_loss_weight,args.gamma_loss_weight)

    gen=torch.Generator();gen.manual_seed(args.seed+350035)
    tl=DataLoader(tr,batch_size=args.batch_size,shuffle=True,generator=gen,num_workers=args.num_workers)
    vl=DataLoader(va,batch_size=args.eval_batch_size,shuffle=False,num_workers=args.num_workers)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(args.epochs,1),eta_min=max(args.lr*1e-3,1e-7))

    best=math.inf;best_epoch=0;bad=0;hist=[];ckpt=out/("best_exp35_%s.pt"%args.model)
    for epoch in range(1,args.epochs+1):
        model.train();tot=0;n=0
        for b in tl:
            gy=b["gy"].to(device);a=b["a1"].to(device);m=b["m"].to(device);g=b["gamma"].to(device)
            opt.zero_grad(set_to_none=True);pred=model(gy)
            loss,la,lm,lg=loss_parts(pred,a,m,g,spans,weights)
            loss.backward();opt.step();tot+=float(loss.item())*len(a);n+=len(a)
        vloss,vmet=eval_loader(model,vl,device,spans,weights,args)
        improved=vloss<best-1e-7
        if improved:
            best=vloss;best_epoch=epoch;bad=0
            torch.save({
                "model_state":model.state_dict(),"epoch":epoch,"val_loss":vloss,
                "model":args.model,"ranges":ranges,
                "architecture":{
                    "hidden_sizes":list(parse_int_tuple(args.hidden_sizes)),
                    "channels":args.channels,"dilations":list(parse_int_tuple(args.dilations)),
                    "head_hidden":args.head_hidden,"amplitude_bins":args.amplitude_bins,
                    "amplitude_hidden":args.amplitude_hidden,
                },
            },ckpt)
        else: bad+=1
        hist.append({"epoch":epoch,"train_loss":tot/max(n,1),"val_loss":vloss,**vmet,
                     "lr":opt.param_groups[0]["lr"],"is_best":int(improved)})
        if improved or epoch%5==0:
            print("%s %s epoch %3d | train %.6g val %.6g | a1 %.2f%% m %.2f%% gamma %.2f%% joint %.2f%%"%(
                "*" if improved else " ",args.model,epoch,tot/max(n,1),vloss,
                100*vmet["a1_recovery_rate"],100*vmet["m_recovery_rate"],
                100*vmet["gamma_recovery_rate"],100*vmet["joint_recovery_rate"]))
        sched.step()
        if bad>=args.patience:break
    write_csv(out/"training_history.csv",hist)

    try: ck=torch.load(ckpt,map_location=device,weights_only=True)
    except TypeError: ck=torch.load(ckpt,map_location=device)
    model.load_state_dict(ck["model_state"]);model.eval()

    summary=[];samples=[]
    for noise,p in noise_dirs(data_dir):
        ds=TripleNpzDataset(p/"test.npz",args.max_test_samples)
        pa,pm,pg,met,frel,grel=predict_dataset(model,ds,device,args.eval_batch_size,args)
        summary.append({"model":args.model,"noise_dir":p.name,"noise_level":noise,
                        "split":"test_independent",**met})
        ea,em,gf,gl,ar,mr,grr,jr=metric_arrays(
            pa,ds.a1.astype(float),pm,ds.m.astype(float),pg,ds.gamma.astype(float),
            args.a1_success_tol,args.m_success_tol,args.gamma_success_factor
        )
        for i in range(len(pa)):
            samples.append({
                "model":args.model,"noise_dir":p.name,"noise_level":noise,
                "sample_index":i,
                "true_a1":float(ds.a1[i]),"pred_a1":float(pa[i]),"a1_abs_error":float(ea[i]),
                "true_m":float(ds.m[i]),"pred_m":float(pm[i]),"m_abs_error":float(em[i]),
                "true_gamma":float(ds.gamma[i]),"pred_gamma":float(pg[i]),
                "gamma_factor_error":float(gf[i]),"gamma_abs_log10_error":float(gl[i]),
                "a1_recovered":int(ar[i]),"m_recovered":int(mr[i]),
                "gamma_recovered":int(grr[i]),"joint_recovered":int(jr[i]),
                "f_relative_l2":float(frel[i]),"g_clean_relative_l2":float(grel[i]),
            })
        print("test %-14s | a1 %.2f%% m %.2f%% gamma %.2f%% joint %.2f%%"%(
            p.name,100*met["a1_recovery_rate"],100*met["m_recovery_rate"],
            100*met["gamma_recovery_rate"],100*met["joint_recovery_rate"]))

    prefix=args.result_prefix.strip() or "exp35"
    write_csv(out/(prefix+"_summary.csv"),summary)
    write_csv(out/(prefix+"_samples.csv"),samples)
    (out/(prefix+"_metadata.json")).write_text(json.dumps({
        "experiment":"Exp35 3P model baseline",
        "model":args.model,"best_epoch":best_epoch,
        "parameter_bounds_source":args.parameter_bounds_source,
        "model_ranges":{"a1":a1r,"m":mr,"gamma":gr},
        "recovery_tolerances":{
            "a1_abs":args.a1_success_tol,"m_abs":args.m_success_tol,
            "gamma_factor":args.gamma_success_factor,
        },
        "loss":"equal range-normalized MSE for a1, m and log10(gamma), with configurable weights",
        "args":vars(args),
    },ensure_ascii=False,indent=2,default=str),encoding="utf-8")


if __name__=="__main__":
    main()
