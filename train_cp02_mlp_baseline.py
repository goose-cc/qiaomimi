from pathlib import Path
import argparse, csv, json, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_norm(path):
    z = np.load(path, allow_pickle=False)
    im = np.asarray(z["input_mean"], np.float32)
    istd = np.maximum(np.asarray(z["input_std"], np.float32), 1e-12)
    tm = np.asarray(z["target_mean"], np.float32)
    ts = np.maximum(np.asarray(z["target_std"], np.float32), 1e-12)
    if tm.shape != (3,) or ts.shape != (3,):
        raise ValueError(f"target normalization must be shape (3,), got {tm.shape}/{ts.shape}")
    return im, istd, tm, ts

class CP02Dataset(Dataset):
    def __init__(self, path, norm):
        self.path = Path(path)
        im, istd, tm, ts = norm
        z = np.load(self.path, allow_pickle=False)
        self.x_raw = np.asarray(z["gy"], np.float32)
        if all(k in z.files for k in ("a1","m","gamma")):
            a1 = np.asarray(z["a1"], np.float32)
            m = np.asarray(z["m"], np.float32)
            gamma = np.asarray(z["gamma"], np.float32)
        else:
            p = np.asarray(z["params3"], np.float32)
            a1, m, gamma = p[:,0], p[:,1], p[:,2]
        if np.any(gamma <= 0): raise ValueError("gamma must be > 0")
        self.y_raw = np.stack([a1, m, np.log(gamma)], axis=1).astype(np.float32)
        self.x = ((self.x_raw - im) / istd).astype(np.float32)
        self.y = ((self.y_raw - tm) / ts).astype(np.float32)
        n = len(self.x)
        self.ids = np.asarray(z["physical_state_id"]).astype(str) if "physical_state_id" in z.files else np.asarray([str(i) for i in range(n)])
        self.margin = np.asarray(z["continuous_margin"], float) if "continuous_margin" in z.files else np.full(n, np.nan)
        self.noise_level = np.asarray(z["noise_level"], float) if "noise_level" in z.files else np.full(n, np.nan)
        self.q2 = np.asarray(z["q2"], float) if "q2" in z.files else None
        z.close()
        if self.x.ndim != 2: raise ValueError(f"gy must be [N,L], got {self.x.shape}")
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all(): raise ValueError("NaN/Inf found")
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i]), i

class MLP(nn.Module):
    def __init__(self, n_in, hidden=(512,256,128), dropout=0.05):
        super().__init__()
        layers=[]; d=n_in
        for h in hidden:
            layers += [nn.Linear(d,h), nn.GELU(), nn.Dropout(dropout)]
            d=h
        self.backbone=nn.Sequential(*layers)
        def head():
            mid=max(32,d//2)
            return nn.Sequential(nn.Linear(d,mid), nn.GELU(), nn.Linear(mid,1))
        self.ha1=head(); self.hm=head(); self.hg=head()
    def forward(self,x):
        h=self.backbone(x)
        return torch.cat([self.ha1(h), self.hm(h), self.hg(h)],1)

def epoch(model, loader, lossfn, device, opt=None):
    model.train(opt is not None); total=0.0; n=0
    for x,y,_ in loader:
        x=x.to(device); y=y.to(device)
        if opt: opt.zero_grad(set_to_none=True)
        p=model(x); loss=lossfn(p,y)
        if opt:
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        total += loss.item()*len(x); n += len(x)
    return total/max(n,1)

@torch.no_grad()
def predict(model, loader, device, n):
    model.eval(); out=np.empty((n,3),np.float32)
    for x,_,idx in loader:
        out[np.asarray(idx)] = model(x.to(device)).cpu().numpy()
    return out

def metrics_and_rows(ds, predn, norm):
    _,_,tm,ts = norm
    raw = predn*ts[None,:] + tm[None,:]
    pa1 = raw[:,0].astype(float); pm = raw[:,1].astype(float); pg = np.exp(raw[:,2].astype(float))
    ta1 = ds.y_raw[:,0].astype(float); tmass = ds.y_raw[:,1].astype(float); tg = np.exp(ds.y_raw[:,2].astype(float))
    ea1=np.abs(pa1-ta1); em=np.abs(pm-tmass); eg=np.abs(pg-tg)
    gf=np.maximum(pg/tg,tg/pg)
    aok=ea1<=0.005; mok=em<=0.03; gok=gf<=1.2; jok=aok&mok&gok
    def stats(e,p):
        return {f"{p}_mae":float(np.mean(e)), f"{p}_rmse":float(np.sqrt(np.mean(e**2))),
                f"{p}_median":float(np.median(e)), f"{p}_p90":float(np.quantile(e,.90)),
                f"{p}_p95":float(np.quantile(e,.95))}
    met={"n":len(ds),"R_a1":float(aok.mean()),"R_m":float(mok.mean()),"R_gamma":float(gok.mean()),"R_joint":float(jok.mean()),
         "gamma_factor_median":float(np.median(gf)),"gamma_factor_p90":float(np.quantile(gf,.90)),"gamma_factor_p95":float(np.quantile(gf,.95))}
    met.update(stats(ea1,"a1")); met.update(stats(em,"m")); met.update(stats(eg,"gamma"))
    rows=dict(true_a1=ta1,pred_a1=pa1,true_m=tmass,pred_m=pm,true_gamma=tg,pred_gamma=pg,
              a1_abs_error=ea1,m_abs_error=em,gamma_abs_error=eg,gamma_factor_error=gf,
              pass_a1=aok,pass_m=mok,pass_gamma=gok,joint_pass=jok)
    return met, rows

def save_pred(path, ds, rows, seed):
    fields=["physical_state_id","true_a1","pred_a1","true_m","pred_m","true_gamma","pred_gamma",
            "a1_abs_error","m_abs_error","gamma_abs_error","gamma_factor_error",
            "pass_a1","pass_m","pass_gamma","joint_pass","continuous_margin","noise_level","seed"]
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i in range(len(ds)):
            w.writerow({"physical_state_id":ds.ids[i],
                        **{k:(int(rows[k][i]) if k.startswith("pass_") or k=="joint_pass" else rows[k][i]) for k in rows},
                        "continuous_margin":ds.margin[i],"noise_level":ds.noise_level[i],"seed":seed})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",default="data_cp02_ml_hybrid_domain_v2")
    ap.add_argument("--output-dir",default="runs_cp02_mlp")
    ap.add_argument("--seeds",type=int,nargs="+",default=[20260920,20260921,20260922])
    ap.add_argument("--epochs",type=int,default=300); ap.add_argument("--batch-size",type=int,default=256)
    ap.add_argument("--lr",type=float,default=1e-3); ap.add_argument("--weight-decay",type=float,default=1e-4)
    ap.add_argument("--dropout",type=float,default=.05); ap.add_argument("--patience",type=int,default=30)
    ap.add_argument("--device",default="auto"); ap.add_argument("--num-workers",type=int,default=0)
    args=ap.parse_args()
    root=Path(args.data_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    norm=load_norm(root/"normalization_train_only.npz")
    tr=CP02Dataset(root/"noise_0p2pct"/"train.npz",norm)
    va=CP02Dataset(root/"noise_0p2pct"/"val.npz",norm)
    tests={"0pct":CP02Dataset(root/"noise_0pct"/"test.npz",norm),
           "0p2pct":CP02Dataset(root/"noise_0p2pct"/"test.npz",norm),
           "1pct":CP02Dataset(root/"noise_1pct"/"test.npz",norm)}
    if set(tr.ids)&set(va.ids) or set(tr.ids)&set(tests["0pct"].ids) or set(va.ids)&set(tests["0pct"].ids):
        raise RuntimeError("physical_state_id leakage across train/val/test")
    base=tests["0pct"].ids
    for k,d in tests.items():
        if not np.array_equal(base,d.ids): raise RuntimeError(f"test ID mismatch at {k}")
    print(f"Loaded train={len(tr)}, val={len(va)}, test={len(base)}, input_dim={tr.x.shape[1]}")
    allres=[]
    for seed in args.seeds:
        seed_all(seed)
        device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device))
        g=torch.Generator().manual_seed(seed)
        tl=DataLoader(tr,batch_size=args.batch_size,shuffle=True,generator=g,num_workers=args.num_workers)
        vl=DataLoader(va,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
        model=MLP(tr.x.shape[1],dropout=args.dropout).to(device)
        lossfn=nn.SmoothL1Loss(); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=max(3,args.patience//4),min_lr=1e-6)
        sd=out/f"seed_{seed}"; sd.mkdir(exist_ok=True)
        best=math.inf; bestep=0; stale=0; hist=[]
        for ep in range(1,args.epochs+1):
            lt=epoch(model,tl,lossfn,device,opt); lv=epoch(model,vl,lossfn,device,None); sched.step(lv)
            hist.append((ep,lt,lv,opt.param_groups[0]["lr"]))
            if lv < best-1e-6:
                best=lv; bestep=ep; stale=0; torch.save(model.state_dict(),sd/"best.pt")
            else: stale+=1
            if ep==1 or ep%10==0: print(f"seed={seed} ep={ep} train={lt:.6f} val={lv:.6f}")
            if stale>=args.patience: break
        with open(sd/"history.csv","w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["epoch","train_loss","val_loss","lr"]); w.writerows(hist)
        model.load_state_dict(torch.load(sd/"best.pt",map_location=device,weights_only=True))
        result={"seed":seed,"best_epoch":bestep,"best_val_loss":best,"tests":{}}
        for name,ds in tests.items():
            dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
            pred=predict(model,dl,device,len(ds)); met,rows=metrics_and_rows(ds,pred,norm)
            result["tests"][name]=met; save_pred(sd/f"test_{name}_predictions.csv",ds,rows,seed)
            print(f"{name}: R_a1={met['R_a1']*100:.2f}% R_m={met['R_m']*100:.2f}% R_gamma={met['R_gamma']*100:.2f}% Joint={met['R_joint']*100:.2f}%")
        with open(sd/"metrics.json","w",encoding="utf-8") as f: json.dump(result,f,indent=2)
        allres.append(result)
    agg={}
    for noise in ("0pct","0p2pct","1pct"):
        agg[noise]={}
        for key in ("R_a1","R_m","R_gamma","R_joint"):
            v=np.array([r["tests"][noise][key] for r in allres],float)
            agg[noise][key]={"mean":float(v.mean()),"std":float(v.std(ddof=1) if len(v)>1 else 0.0)}
    with open(out/"aggregate_recovery.json","w",encoding="utf-8") as f: json.dump(agg,f,indent=2)
    print(json.dumps(agg,indent=2))

if __name__=="__main__":
    main()
