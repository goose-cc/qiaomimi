# Experiment 9 — Conditional resonance-gated loss

## Why this experiment exists

Exp8 applied strong `m`, `gamma`, resonance, width and peak-height losses to every
training sample.  Validation showed a failure mode: background-like/near-linear
truth curves received a generic predicted bump.  The model learned "there should
usually be a resonance" instead of learning when a resonance is actually present.

Exp9 keeps the Exp8 physical parameter model and analytic physical decoder, but
changes the loss design.

## Core rule

The true training curve is used to compute a resonance visibility

`visibility = RMS(true_resonance) / RMS(true_total)`.

A smooth gate is then used:

- visibility <= 0.05 -> gate = 0
- visibility >= 0.20 -> gate = 1
- between them -> smooth interpolation

The gate is used ONLY during supervised training because the true curve is known.
No gate is needed at inference time.

## Loss design

All samples always receive:
- base parameter loss
- complete-spectrum loss
- gradient loss
- forward-g physics loss
- resonance-visibility loss

Only visible-resonance samples receive the extra:
- boosted a1/m/log(gamma) parameter weights
- resonance-component loss
- log-width loss
- peak-height loss

Weak/background-like samples instead receive:
- weak-resonance excess penalty

This penalty only punishes predicted resonance that exceeds the true small
resonance by a tolerance. It does not force the resonance exactly to zero.

## Warmup

The visible-resonance strong terms ramp from 0 to full strength over 3000 steps.
This is intended to prevent the generic-peak collapse seen in Exp8.

## Files

Replace/add:
- `mc_parametric_gated.py`
- `train_mc_parameter_pool_parametric_gated.py`
- `validate_mc_parametric_gated.py`
- `run_exp9_parametric_gated_noise0.ps1`
- `test_exp9_parametric_gated.py`

`TransformerInverse.py` does NOT need another change if the Exp8
`PeakParametricInverseTransformer1D` model is already present.
