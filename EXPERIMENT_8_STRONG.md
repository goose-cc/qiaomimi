# Exp8 STRONG — resonance-focused parameter learning

Based directly on Exp7. This is intentionally an aggressive combined experiment because time is limited.

## Actual physical parameterization in this code
The spectral formula is
`rho(s)=a1/pi*(m*gamma)/((s-m)^2+(m*gamma)^2)+a2*s+a3`.
Therefore the resonance **center is m** (not m^2) and the width parameter is `m*gamma`.

## Changes
- Keep 100-point g input and 1000-point analytic physical decoder.
- Shared Transformer encoder.
- Separate heads:
  - resonance: `a1, m, log(gamma)`
  - background: `a2, a3`
- Gamma uses log-scale output, useful for narrow widths.
- Strong transformed parameter weights: `a1=3, a2=1, a3=1, m=5, log_gamma=6`.
- Joint differentiable loss:
  - parameter 1.0
  - total spectrum 1.0
  - resonance component 3.0
  - spectrum gradient 0.25
  - forward g 0.25
  - log width 2.0
  - log peak height 1.0
- Training noise remains 0% for the first decisive test.

## Train
```powershell
.\run_exp8_parametric_strong_noise0.ps1 `
  -PoolDir ".\truth_pool_v2_160m" `
  -CheckpointDir ".\model\v2_exp8_parametric_strong_noise0_160m" `
  -Mode fresh `
  -MaxSteps 30000 `
  -MaxHours 0
```

## Validate
```powershell
python .\validate_mc_parametric_strong.py `
  --validation-pool-dir ".\truth_pool_val_10k" `
  --checkpoint-dir ".\model\v2_exp8_parametric_strong_noise0_160m" `
  --weights latest `
  --num-samples 10000 `
  --batch-size 64 `
  --noise-level 0.0 `
  --seed 20260802 `
  --output-dir ".\validation_results\v2_exp8_strong_latest_val0" `
  --plot-count 18 `
  --device cuda
```
Run again with `--weights best` and a different output directory.
