# CP01 — Corrected Physics Reset

Correct spectral formula:

`rho(s) = a1/pi * (m*gamma) / ((s-m^2)^2 + (m*gamma)^2) + a2*s + a3`

## Official nominal parameter domain (restored)

- `a1 ∈ [0, 0.2]`, original grid increment `0.001`
- `a2 ∈ [0, 0.05]`, original grid increment `0.001`
- `a3 ∈ [-0.05, 0.05]`, original grid increment `0.001`
- `m ∈ [0, 2]`, original grid increment `0.01`
- `gamma ∈ [0, 1]`, original grid increment `0.01`

The official domain is **not** replaced by the previous development domain.
For numerical finite-width Lorentzian integration, `m=0` and `gamma=0` are singular/degenerate endpoints, so continuous audits begin at the original first positive grid value `0.01`. This is recorded explicitly in metadata rather than silently shrinking the nominal domain.

Also note the structural boundary `a1=0`: the resonance term vanishes exactly, so `m` and `gamma` cannot be recovered from `g` there when the background is fixed. The code reports this boundary explicitly. The later identifiable-data selector may screen such non-identifiable states; it must not pretend they are recoverable.

## What is reset vs what is kept

Keep:
- staged 2P -> 3P -> 4P -> 5P strategy;
- recovery-aligned alias analysis;
- Jacobian/VarPro diagnostics;
- TCN / parameter-specific network ideas.

Reset:
- all datasets produced with `(s-m)^2`;
- all numerical recovery/alias/Jacobian/VarPro conclusions from those datasets;
- Exp38/39 q2 rankings until they are rerun with corrected physics.

## First experiment

Do **not** train a neural network yet.
CP01 compares:

1. 2P development domain (`a1 + gamma`, fixed `m=0.8`)
2. 2P restored full domain
3. 3P development domain (`a1 + m + gamma`)
4. 3P restored full domain

using the corrected formula and the baseline q2 design.

Outputs include:
- old-vs-corrected `g` change;
- Jacobian conditioning;
- recovery-aligned grid + continuous alias margin;
- physical VarPro ceiling at 0% and 0.2% noise;
- full-domain recovery split by `a1` bins, so the low-amplitude structural degeneracy is visible rather than hidden.

## Run

Quick smoke test:

```powershell
.\run_corrected_physics_reset.ps1 -Quick
```

Formal reset audit:

```powershell
.\run_corrected_physics_reset.ps1
```

Send back first:

```text
corrected_physics_reset_results/
  corrected_physics_reset_summary.csv
  full_domain_recovery_by_a1.csv
  continuous_alias_audit.csv
  boundary_identifiability.csv
  stale_formula_references.csv
```

## Important

`stale_formula_references.csv` may list old Exp31-39/legacy validation scripts that still contain `(s-m)^2`. That does not invalidate CP01 as long as the four active source-of-truth files pass. Do not use a flagged legacy script for new corrected-physics results until it is migrated.
