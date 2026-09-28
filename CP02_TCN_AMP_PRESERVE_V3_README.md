# CP02 TCN amplitude-preserving v3

This patch is a targeted response to the v2 behavior:

- `m` and `gamma` were already around the low/mid 90% recovery range.
- `a1` recovery stayed near the high-60% range even at 0% noise.
- the v2 prediction CSV showed a strong regression-to-the-mean pattern in `a1`.

The data definition is unchanged.

## Why change the TCN

The v2 residual blocks used `GroupNorm(1, C)`. That normalization is applied per
sample and can suppress absolute trace amplitude. This matters because `a1` is
strongly amplitude-sensitive.

The v2 model also compressed the final sequence using only global mean and max.
That throws away a lot of q2-location information.

## v3 changes

1. Remove GroupNorm completely.
2. Preserve q2 location with adaptive positional avg/max pooling over 12 bins.
3. Add a small direct observation shortcut from the full normalized trace.
4. Add simple whole-trace descriptors: mean/std/RMS/min/max/first/last.
5. Reduce dropout from 0.08 to 0.03 and weight decay from 1e-4 to 3e-5.
6. Select the best checkpoint using validation recovery first:
   - maximize the worst of `R_a1, R_m, R_gamma`
   - then maximize joint recovery
   - then minimize validation loss
7. Report prediction-vs-truth slopes to detect dynamic-range shrinkage.

The direct shortcut is intentionally small. The TCN remains the main feature
extractor, but global amplitude/position information can no longer be erased.

## Run order

Quick smoke test:

```powershell
.\run_cp02_tcn_baseline.ps1 -Quick -Device cuda
```

Recommended first scientific run:

```powershell
.\run_cp02_tcn_baseline.ps1 -SingleSeed -Device cuda
```

Do not run three seeds yet. First compare the v3 single-seed 0.2% result against
the v2 single-seed result.

The main success criteria for this iteration are:

```text
R_a1 > 90%
R_m  >= 90%
R_gamma >= 90%
a1_pred_vs_true_slope close to 1
```

If the shortcut version improves strongly, an ablation can check how much of the
gain came simply from removing normalization:

```powershell
.\run_cp02_tcn_baseline.ps1 -SingleSeed -Device cuda -NoShortcut
```

That run is a pure no-normalization TCN with positional pooling.

Only after one-seed v3 passes the stage gate should the formal 3-seed run be used:

```powershell
.\run_cp02_tcn_baseline.ps1 -Device cuda
```

## Outputs

```text
runs_cp02_tcn_v3/
  run_manifest.json
  aggregate_metrics.json
  seed_20260920/
    best.pt
    history.csv
    metrics.json
    test_0pct_predictions.csv
    test_0p2pct_predictions.csv
    test_1pct_predictions.csv
```
