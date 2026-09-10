#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the Exp36 3P TCN on the frozen physical-state split."""
from __future__ import annotations
import argparse, csv, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from TripleInverseTCN3P import TripleInverseTCN3P
from mc_physics import scaled_forward_observation_numpy


class NPZ3P(Dataset):
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z:
            self.gy=z["gy"].astype(np.float32)
            self.a1=z["a1"].astype(np.float32)
            self.m=z["m"].astype(np.float32)
            self.gamma=z["gamma"].astype(np.float32)
            self.state_id=z["state_id"].astype(np.int64)
    def __len__(self): return len(self.gy)
    def __getitem__(self,i):
        return {
            "gy":torch.from_numpy(self.gy[i]),
            "a1":torch.tensor(self.a1[i]),
            "m":torch.tensor(self.m[i]),
            "gamma":torch.tensor(self.gamma[i]),
            "state_id":torch.tensor(self.state_id[i]),
        }


def write_csv(path,rows):
    if not rows:
        Path(path).write_text("",encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with open(path,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def device_of(text):
    if text=="cuda" and torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


def noise_dirs(data_dir):
    result=[]
    for p in Path(data_dir).glob("noise_*"):
        if (p/"test.npz").exists():
            with np.load(p/"test.npz",allow_pickle=False) as z:
                lev=float(np.median(z["noise_level"]))
            result.append((lev,p))
    return sorted(result,key=lambda x:x[0])


def quant(x,q): return float(np.quantile(np.asarray(x,dtype=float),q))


def metric_block(pa,ta,pm,tm,pg,tg,tol):
    pa=np.asarray(pa,float); ta=np.asarray(ta,float)
    pm=np.asarray(pm,float); tm=np.asarray(tm,float)
    pg=np.clip(np.asarray(pg,float),1e-30,None); tg=np.clip(np.asarray(tg,float),1e-30,None)
    ea=np.abs(pa-ta); em=np.abs(pm-tm)
    gf=np.exp(np.abs(np.log(pg)-np.log(tg)))
    relg=np.abs(pg-tg)/tg
    aok=ea<=tol["a1_abs"]; mok=em<=tol["m_abs"]; gok=gf<=tol["gamma_factor"]; jok=aok&mok&gok
    return {
        "count":len(pa),
        "a1_mae":float(ea.mean()),"a1_p90":quant(ea,.90),"a1_p95":quant(ea,.95),"a1_recovery":float(aok.mean()),
        "m_mae":float(em.mean()),"m_p90":quant(em,.90),"m_p95":quant(em,.95),"m_recovery":float(mok.mean()),
        "gamma_relative_mae":float(relg.mean()),"gamma_p90_factor":quant(gf,.90),"gamma_p95_factor":quant(gf,.95),"gamma_recovery":float(gok.mean()),
        "joint_recovery":float(jok.mean()),
    }


@torch.no_grad()
def predict(model,ds,device,batch_size):
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False)
    out={k:[] for k in ("pa","pm","pg","ta","tm","tg","sid")}
    model.eval()
    for b in loader:
        gy=b["gy"].to(device)
        a,m,lg=model(gy); g=torch.pow(10.0,lg)
        for key,val in [("pa",a),("pm",m),("pg",g),("ta",b["a1"]),("tm",b["m"]),("tg",b["gamma"]),("sid",b["state_id"])]:
            out[key].append(val.detach().cpu().numpy())
    return {k:np.concatenate(v) for k,v in out.items()}


def state_aggregate(pred):
    rows={k:[] for k in ("pa","pm","pg","ta","tm","tg")}
    for sid in np.unique(pred["sid"]):
        mask=pred["sid"]==sid
        for k in ("pa","pm","pg"):
            rows[k].append(float(np.mean(pred[k][mask])))
        for k in ("ta","tm","tg"):
            rows[k].append(float(pred[k][mask][0]))
    return {k:np.asarray(v) for k,v in rows.items()}


def physics_metrics(pred,fixed,integration_points,max_samples=0):
    n=len(pred["pa"])
    ids=np.arange(n)
    if max_samples and n>max_samples:
        ids=np.linspace(0,n-1,max_samples,dtype=int)
    pp=np.column_stack([
        pred["pa"][ids],np.full(len(ids),fixed["a2"]),np.full(len(ids),fixed["a3"]),
        pred["pm"][ids],pred["pg"][ids]
    ])
    tt=np.column_stack([
        pred["ta"][ids],np.full(len(ids),fixed["a2"]),np.full(len(ids),fixed["a3"]),
        pred["tm"][ids],pred["tg"][ids]
    ])
    gp=scaled_forward_observation_numpy(pp,integration_points=integration_points).astype(np.float64)
    gt=scaled_forward_observation_numpy(tt,integration_points=integration_points).astype(np.float64)
    rel=np.sqrt(np.mean((gp-gt)**2,axis=1))/np.maximum(np.sqrt(np.mean(gt**2,axis=1)),1e-30)
    return {"g_reconstruction_median_relL2":float(np.median(rel)),"g_reconstruction_p90_relL2":quant(rel,.90)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="exp36_3p_config.json")
    ap.add_argument("--data-dir",default="data_exp36_3p")
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--epochs",type=int,default=0)
    ap.add_argument("--quick",action="store_true")
    args=ap.parse_args()

    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    meta=json.loads((Path(args.data_dir)/"metadata.json").read_text(encoding="utf-8"))
    if not meta["validation"]["pass"]: raise RuntimeError("dataset metadata validation did not PASS")
    trcfg=cfg["training"]; epochs=args.epochs or int(trcfg["epochs"])
    if args.quick: epochs=min(epochs,3)
    set_seed(args.seed); device=device_of(args.device)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)

    train_dir=min(noise_dirs(Path(args.data_dir)),key=lambda x:abs(x[0]-float(cfg["training_noise"])))[1]
    train=NPZ3P(train_dir/"train.npz"); val=NPZ3P(train_dir/"val.npz")
    mean=train.gy.mean(axis=0,dtype=np.float64).astype(np.float32)
    scale=train.gy.std(axis=0,dtype=np.float64).astype(np.float32)
    scale=np.maximum(scale,1e-8*np.maximum(np.abs(mean),1.0)).astype(np.float32)

    pr=cfg["parameter_ranges"]
    model=TripleInverseTCN3P(
        input_points=train.gy.shape[1],channels=int(trcfg["tcn_channels"]),
        dilations=tuple(trcfg["tcn_dilations"]),kernel_size=int(trcfg["kernel_size"]),
        head_hidden=int(trcfg["head_hidden"]),dropout=float(trcfg["dropout"]),
        a1_range=tuple(pr["a1"]),m_range=tuple(pr["m"]),gamma_range=tuple(pr["gamma"])
    )
    model.set_input_normalization(mean,scale); model=model.to(device)

    gen=torch.Generator().manual_seed(args.seed+360036)
    tl=DataLoader(train,batch_size=int(trcfg["batch_size"]),shuffle=True,generator=gen)
    vl=DataLoader(val,batch_size=int(trcfg["eval_batch_size"]),shuffle=False)
    opt=torch.optim.AdamW(model.parameters(),lr=float(trcfg["lr"]),weight_decay=float(trcfg["weight_decay"]))
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(epochs,1),eta_min=float(trcfg["lr"])*1e-3)

    a_span=pr["a1"][1]-pr["a1"][0]; m_span=pr["m"][1]-pr["m"][0]
    lg_span=math.log10(pr["gamma"][1])-math.log10(pr["gamma"][0])

    def loss_batch(a,m,lg,b):
        la=((a-b["a1"].to(device))/a_span)**2
        lm=((m-b["m"].to(device))/m_span)**2
        true_lg=torch.log10(b["gamma"].to(device))
        lgerr=((lg-true_lg)/lg_span)**2
        return torch.mean(la+lm+lgerr)

    @torch.no_grad()
    def eval_loss(loader):
        model.eval(); total=0.;n=0
        for b in loader:
            a,m,lg=model(b["gy"].to(device))
            L=loss_batch(a,m,lg,b)
            total+=float(L)*len(b["a1"]);n+=len(b["a1"])
        return total/max(n,1)

    best=math.inf; bad=0; hist=[]; ck=out/"best_exp36_tcn3p.pt"
    for epoch in range(1,epochs+1):
        model.train();total=0.;n=0
        for b in tl:
            opt.zero_grad(set_to_none=True)
            a,m,lg=model(b["gy"].to(device))
            L=loss_batch(a,m,lg,b); L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step(); total+=float(L.detach().item())*len(b["a1"]);n+=len(b["a1"])
        vlv=eval_loss(vl); trv=total/max(n,1)
        improved=vlv<best-1e-8
        if improved:
            best=vlv;bad=0
            torch.save({"model_state":model.state_dict(),"seed":int(args.seed),"epoch":int(epoch),"val_loss":float(vlv)},ck)
        else: bad+=1
        hist.append({"epoch":epoch,"train_loss":trv,"val_loss":vlv,"lr":opt.param_groups[0]["lr"],"is_best":int(improved)})
        if improved or epoch%5==0: print(f"{'*' if improved else ' '} epoch {epoch:3d} train={trv:.6g} val={vlv:.6g}")
        sched.step()
        if bad>=int(trcfg["patience"]): break
    write_csv(out/"training_history.csv",hist)

    try: state=torch.load(ck,map_location=device,weights_only=True)
    except TypeError: state=torch.load(ck,map_location=device)
    model.load_state_dict(state["model_state"]); model.eval()

    summaries=[]; pred_rows=[]
    tol=cfg["recovery_tolerances"]
    for level,ndir in noise_dirs(Path(args.data_dir)):
        ds=NPZ3P(ndir/"test.npz")
        p=predict(model,ds,device,int(trcfg["eval_batch_size"]))
        sm=metric_block(p["pa"],p["ta"],p["pm"],p["tm"],p["pg"],p["tg"],tol)
        agg=state_aggregate(p)
        stm=metric_block(agg["pa"],agg["ta"],agg["pm"],agg["tm"],agg["pg"],agg["tg"],tol)
        phys=physics_metrics(p,cfg["fixed_parameters"],int(cfg["physics_integration_points"]),max_samples=2000 if args.quick else 0)
        summaries.append({
            "seed":args.seed,"noise_level":level,"noise_dir":ndir.name,
            **{"sample_"+k:v for k,v in sm.items()},
            **{"state_"+k:v for k,v in stm.items()},
            **phys
        })
        a1_abs=np.abs(p["pa"]-p["ta"])
        m_abs=np.abs(p["pm"]-p["tm"])
        gamma_factor=np.exp(np.abs(np.log(np.clip(p["pg"],1e-30,None))-np.log(np.clip(p["tg"],1e-30,None))))
        a1_ok=a1_abs<=tol["a1_abs"]
        m_ok=m_abs<=tol["m_abs"]
        gamma_ok=gamma_factor<=tol["gamma_factor"]
        joint_ok=a1_ok&m_ok&gamma_ok
        for i in range(len(p["pa"])):
            pred_rows.append({
                "seed":args.seed,"noise_level":level,"state_id":int(p["sid"][i]),
                "true_a1":p["ta"][i],"pred_a1":p["pa"][i],
                "a1_abs_error":a1_abs[i],"a1_recovered":int(a1_ok[i]),
                "true_m":p["tm"][i],"pred_m":p["pm"][i],
                "m_abs_error":m_abs[i],"m_recovered":int(m_ok[i]),
                "true_gamma":p["tg"][i],"pred_gamma":p["pg"][i],
                "gamma_factor_error":gamma_factor[i],"gamma_recovered":int(gamma_ok[i]),
                "joint_recovered":int(joint_ok[i]),
            })
        print(f"noise={level:g} sample joint={100*sm['joint_recovery']:.2f}% state joint={100*stm['joint_recovery']:.2f}%")
    write_csv(out/"exp36_summary.csv",summaries)
    write_csv(out/"exp36_predictions.csv",pred_rows)
    (out/"run_metadata.json").write_text(json.dumps({"seed":args.seed,"best_epoch":state["epoch"],"best_val_loss":state["val_loss"],"dataset":str(args.data_dir),"training_noise_dir":train_dir.name},indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
