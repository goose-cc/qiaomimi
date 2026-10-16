from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from train_fivep_compare import (
    load_norm, FivePDataset, FivePMLP, FivePTCN,
    make_coordinate_channels, weighted_smooth_l1,
    pred_norm_to_physical, RECOVERY, seed_all, choose_device
)

def metrics(true, pred):
    ae = np.abs(pred - true)
    gf = np.maximum(
        pred[:,4] / np.maximum(true[:,4], 1e-300),
        true[:,4] / np.maximum(pred[:,4], 1e-300),
    )
    pa = ae[:,0] <= RECOVERY["a1_abs"]
    pm = ae[:,3] <= RECOVERY["m_abs"]
    pg = gf <= RECOVERY["gamma_factor"]
    pj = pa & pm & pg
    return {
        "R_a1": pa.mean(), "R_m": pm.mean(), "R_gamma": pg.mean(), "R_joint": pj.mean(),
        "a1_mae": ae[:,0].mean(), "m_mae": ae[:,3].mean(),
        "gamma_p90": np.quantile(gf, 0.9)
    }

@torch.no_grad()
def predict(model, x, norm):
    model.eval()
    p = model(x).float().cpu().numpy()
    return pred_norm_to_physical(p, norm)

def select_states(ds, n_states):
    ids = np.asarray(ds.ids)
    unique = []
    seen = set()
    for s in ids:
        if s not in seen:
            seen.add(s)
            unique.append(s)
        if len(unique) >= n_states:
            break
    mask = np.isin(ids, unique)
    return np.where(mask)[0], unique

def train_model(name, ds, idx, norm, coord, device, epochs, report_every, seed):
    seed_all(seed)
    x = torch.from_numpy(ds.x[idx]).to(device)
    y = torch.from_numpy(ds.y[idx]).to(device)
    true = ds.parameters[idx].astype(np.float64)

    if name == "mlp":
        model = FivePMLP(ds.x.shape[1], hidden=(512,256,128), dropout=0.0).to(device)
        lr = 1e-3
    else:
        model = FivePTCN(
            ds.x.shape[1], coord,
            channels=(64,96,128,160), kernel=5, dropout=0.0,
            pool_bins=12, shortcut_dim=128
        ).to(device)
        lr = 5e-4

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    weights = torch.ones(5, device=device)

    print(f"\n=== {name.upper()} overfit sanity: samples={len(idx)} ===")
    best_joint = -1.0

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = weighted_smooth_l1(pred, y, weights)
        loss.backward()
        opt.step()

        if ep == 1 or ep % report_every == 0 or ep == epochs:
            phys = predict(model, x, norm)
            m = metrics(true, phys)
            print(
                f"[{name}] ep={ep:4d} loss={loss.item():.6f} "
                f"a1={100*m['R_a1']:.1f}% m={100*m['R_m']:.1f}% "
                f"gamma={100*m['R_gamma']:.1f}% joint={100*m['R_joint']:.1f}% "
                f"a1_mae={m['a1_mae']:.5f} m_mae={m['m_mae']:.5f} "
                f"gammaP90={m['gamma_p90']:.3f}"
            )
            best_joint = max(best_joint, float(m["R_joint"]))
            if m["R_joint"] >= 0.98:
                print(f"[{name}] PASS: small-set memorization reached >=98% joint Recovery.")
                break

    return best_joint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--states", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--report-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--models", nargs="+", default=["mlp","tcn"], choices=["mlp","tcn"])
    args = ap.parse_args()

    device = choose_device(args.device)
    norm = load_norm(args.data_dir / "normalization_train_only.npz")
    ds = FivePDataset(args.data_dir / "noise_0p2pct" / "train.npz", norm)
    idx, unique = select_states(ds, args.states)

    q2 = np.load(args.data_dir / "q2.npy")
    coord, _ = make_coordinate_channels(q2, ds.x.shape[1])
    coord = coord.to(device)

    print(f"Device: {device}")
    print(f"Selected unique physical states: {len(unique)}")
    print(f"Selected noisy samples: {len(idx)}")
    print("This diagnostic uses TRAIN only; it does not read Val/Test.")

    results = {}
    for name in args.models:
        results[name] = train_model(
            name, ds, idx, norm, coord, device,
            args.epochs, args.report_every, args.seed
        )

    print("\n=== Interpretation ===")
    for name, joint in results.items():
        if joint >= 0.98:
            print(f"{name}: PASS small-set memorization (best joint >=98%).")
        elif joint >= 0.80:
            print(f"{name}: PARTIAL; model learns strongly but has not fully memorized.")
        else:
            print(f"{name}: FAIL small-set memorization; inspect optimization/model/data interface before blaming dataset size.")

if __name__ == "__main__":
    main()
