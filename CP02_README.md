# CP02 — corrected-formula 3P full-domain main line

This experiment does **not** train a neural network.

It starts directly from the corrected 3P task

`g -> (a1, m, gamma)`

with the corrected resonance center `m^2` and the original nominal parameter domain.

## Domain policy

Nominal scientific domain is unchanged:

- a1: 0 .. 0.2
- m: 0 .. 2
- gamma: 0 .. 1

Exact `m=0` and `gamma=0` are singular/zero-width endpoints, so numerical continuous searches start at the original first positive grid value 0.01. `a1=0` remains part of the nominal domain and is **not replaced by an artificial a1>=0.05 rule**. States near a1=0 are allowed into the candidate pool and survive only if the recovery-aligned identifiability test says they are distinguishable.

Recovery tolerances remain:

- |delta a1| <= 0.005
- |delta m| <= 0.03
- gamma factor <= 1.2

Reference noise is 0.2%.

## Pipeline

1. Re-scan q2/g observation designs under the corrected physics.
2. Use the same full-domain anchors for all designs.
3. Rank by Jacobian conditioning and recovery-aligned alias distance.
4. Continuous-audit the finalists.
5. Build a 48k full-domain candidate bank for the winning design.
6. Grid screen -> stratified prebank -> continuous alias refinement.
7. Select 260 states from the **empirical identifiable region**, enforcing g separation.
8. Report how much of the original domain survives; do not disguise a lost low-a1 region as "full coverage".
9. Run physics/VarPro recovery at 0%, 0.2%, and 1% noise.

The first stage gate is per-parameter physical recovery >=90% at 0.2% noise. The target is >=95%. Joint recovery is report-only.

## Run

Smoke test:

```powershell
.\run_cp02_corrected_3p.ps1 -Quick
```

Recommended formal sequence:

```powershell
.\run_cp02_corrected_3p.ps1 -ScanOnly
```

Inspect `cp02_design_scan/cp02_g_design_continuous.csv`. If desired, force one design:

```powershell
.\run_cp02_corrected_3p.ps1 -DesignId hybrid_ultranear_240
```

Otherwise the runner uses the best continuous-audit design automatically.

## Important interpretation

A high recovery number here is a **physical/VarPro recovery ceiling**, not neural-network accuracy. Only after CP02 produces a sufficiently identifiable 3P bank do we generate final noisy train/val/test data and train TCN.
