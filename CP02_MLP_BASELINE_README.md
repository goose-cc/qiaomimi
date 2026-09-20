# CP02 fully-connected baseline

Use the already validated `data_cp02_ml_hybrid_domain_v2` dataset without rebuilding it.

- Input: `gy`
- Targets: normalized `[a1, m, log(gamma)]`
- Train: `noise_0p2pct/train.npz`
- Validation: `noise_0p2pct/val.npz`
- Tests: the same physical states at 0%, 0.2%, and 1% noise
- Normalization: `normalization_train_only.npz` only

Model: plain MLP with hidden widths `512 -> 256 -> 128` and three parameter-specific fully-connected heads.

Recovery:
- `|Δa1| <= 0.005`
- `|Δm| <= 0.03`
- `max(pred_gamma/gamma, gamma/pred_gamma) <= 1.2`

Quick smoke test:

```powershell
.\run_cp02_mlp_baseline.ps1 -Quick
```

Formal three-seed run:

```powershell
.\run_cp02_mlp_baseline.ps1
```

Results are written under `runs_cp02_mlp/`, including per-seed checkpoints, learning curves, per-sample predictions, and aggregate recovery.
