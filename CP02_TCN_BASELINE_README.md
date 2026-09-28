# CP02 3P TCN baseline

This replaces the incomplete `train_cp02_tcn.py` and old runner.

Frozen supervised task:

- input: `g(q^2)`
- targets: normalized `[a1, m, log(gamma)]`
- train: `data_cp02_ml_hybrid_domain_v2/noise_0p2pct/train.npz`
- validation: `data_cp02_ml_hybrid_domain_v2/noise_0p2pct/val.npz`
- tests: the same test physical states at 0%, 0.2%, and 1% noise
- normalization: `normalization_train_only.npz`
- recovery:
  - `|pred_a1 - true_a1| <= 0.005`
  - `|pred_m - true_m| <= 0.03`
  - `max(pred_gamma/true_gamma, true_gamma/pred_gamma) <= 1.2`

## Important implementation changes

1. The TCN input length is detected from `gy.shape[1]`; 239/240 is not hard-coded.
2. If the NPZ contains `q2` (or `data_dir/q2.npy` exists), the model adds real q2 coordinate channels:
   - linearly normalized q2
   - normalized `log(abs(q2))`
   This is preferable for the non-uniform `hybrid_ultranear_240` design.
3. If q2 is not present, the script falls back to an index coordinate and prints a warning.
4. The model has separate heads for `a1`, `m`, and `log_gamma`.
5. Physical-state leakage is checked before training.
6. The same best validation checkpoint is evaluated on 0%, 0.2%, and 1% test sets.
7. Three formal random seeds are run by default.
8. Per-sample predictions and continuous margins are saved when the dataset provides them.

## Quick smoke test

```powershell
.\run_cp02_tcn_baseline.ps1 -Quick -Device auto
```

This is only a pipeline test, not a scientific result.

## Formal run

```powershell
.\run_cp02_tcn_baseline.ps1 -Device cuda
```

If CUDA is not available:

```powershell
.\run_cp02_tcn_baseline.ps1 -Device cpu
```

Outputs:

```text
runs_cp02_tcn/
  run_manifest.json
  aggregate_metrics.json
  seed_20260920/
    best.pt
    history.csv
    metrics.json
    test_0pct_predictions.csv
    test_0p2pct_predictions.csv
    test_1pct_predictions.csv
  ...
```

The primary scientific result is the 0.2% test recovery. The first stage target is >=90% per parameter; the intended target is >=95% per parameter.

## Scope note

The supplied `data_cp02_ml_hybrid_domain_v2` has the declared candidate ranges:

- `a1 in [0.05, 0.20]`
- `m in [0.40, 1.20]`
- `gamma in [0.01, 1.00]`

and applies `D_M^continuous >= 1.5`.

Treat results from this dataset as a **hybrid-domain-v2 3P baseline**. Do not describe them as performance over the widest CP02-v2 identifiable region unless the actual accepted `physical_states.csv` confirms that wider support.
