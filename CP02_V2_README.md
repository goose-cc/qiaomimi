# CP02-v2 Threshold / Coverage Sweep

This patch does **not** rerun the expensive 2058-state continuous refinement.
It reuses:

- `cp02_identifiable_bank/cp02_refined_prebank.csv`
- `cp02_identifiable_bank/q2.npy`

and asks a different question from CP02-v1:

> What is the **lowest** continuous Mahalanobis threshold that keeps the largest identifiable region while the physical VarPro recovery at 0.2% noise still meets the target?

## What changed

CP02-v1 selected the first high threshold that could still provide 260 states, which was D_M >= 3.0. That produced excellent physical recovery but a narrow selected parameter region.

CP02-v2 sweeps D_M = 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3. For every feasible threshold it:

1. forms the eligible continuously identifiable region;
2. coverage-selects 260 states with the existing g-separation gate;
3. reports coverage relative to both the **eligible region** and the **whole refined prebank**;
4. audits physical recovery on a coverage sample of the **whole eligible region**, not only the selected 260 states;
5. confirms low candidate thresholds with a larger 0.2%-noise VarPro audit;
6. recommends the lowest threshold that confirms all three per-parameter recoveries >=95%.

If no threshold reaches 95%, it reports whether a 90% stage gate passes. It never silently relabels a narrowed subset as full-domain coverage.

## Speed / resume

The slow continuous alias refinement is not repeated. VarPro inversion results are cached in:

`cp02_threshold_sweep_v2/cp02_v2_prediction_cache.csv`

so rerunning resumes previously completed state/repetition inversions. `-Workers 4` uses shared-memory threads for independent VarPro inversions; if the machine becomes memory/CPU constrained, use 2 or 1.

## Run

First optional smoke test:

```powershell
.\run_cp02_threshold_sweep.ps1 -Quick -Workers 2
```

Official sweep:

```powershell
.\run_cp02_threshold_sweep.ps1 -Workers 4
```

## Main outputs

- `cp02_v2_threshold_sweep_summary.csv`: threshold vs eligible count, coverage, quick physical recovery
- `cp02_v2_threshold_confirm_summary.csv`: larger confirmation audits
- `cp02_v2_region_coverage.csv`: per-axis/bin survival and selected coverage
- `cp02_v2_recommendation.json`: chosen threshold and gate status
- `cp02_v2_selected_states.csv`: final 260-state bank table
- `cp02_v2_selected_bank.npz`: clean selected bank for the next train/val/test build

Interpretation of statuses:

- `TARGET95_PASS`: all a1/m/gamma physical recoveries >=95% on the confirmed eligible-region audit
- `STAGE90_PASS_ONLY`: >=90% but at least one parameter <95%
- `FALLBACK_HIGHEST_MARGIN`: no physical stage gate passed; do not start network training yet
