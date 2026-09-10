# Exp39 — fixed-m slice diagnosis + worst-case-targeted g(q²) design

## Fixed project constraints

Exp39 **does not change parameter definition domains**:

- `a1 ∈ [0.05, 0.20]`
- `m ∈ [0.40, 1.20]`
- `gamma ∈ [0.01, 1.0]`
- fixed `a2=0.025, a3=0`

Recovery tolerances also stay unchanged:

- `|Δa1| <= 0.005`
- `|Δm| <= 0.03`
- gamma factor `<= 1.2`

Long-term target: each 3P parameter recovery >= 95% at 0.2% noise. Exp39 itself is an observation-design experiment, not a neural-network experiment.

## Why Exp39

Exp38 showed that moving observation support toward q²≈0 dramatically improves 3P information and largely fixes the new m direction, but continuous aliases are still mainly associated with a1/gamma. Exp39 therefore:

1. freezes m at multiple slices across the **entire original m domain**;
2. diagnoses where the fixed-m `(a1,gamma)` inverse is worst;
3. automatically selects q² points using tolerance-scaled Fisher/E-optimal criteria targeted at those worst slices;
4. direct-verifies candidate designs with the same continuous 3P alias machinery used in Exp38.

The greedy Fisher objective is only a search heuristic. Final ranking is by direct-verified continuous 3P alias separation.

## Required existing project files

Keep these existing files unchanged:

- `mc_physics.py`
- `mc_pool_config.py`
- `exp38_core.py`
- `exp38_observation.py`

Add the Exp39 files from this patch.

## Run

Smoke test:

```powershell
.\run_exp39_targeted_g.ps1 -Quick -AuditAll
```

Formal diagnosis + automatic design construction:

```powershell
.\run_exp39_targeted_g.ps1
```

This produces:

```text
exp39_slice_diagnosis/m_slice_diagnosis.csv
exp39_slice_diagnosis/worst_m_slices.json
exp39_targeted_designs/design_build_summary.csv
```

Then deep-audit all candidate designs:

```powershell
.\run_exp39_targeted_g.ps1 -AuditAll
```

Important output:

```text
exp39_targeted_audit/exp39_deep_audit_ranking.csv
exp39_targeted_audit/<design>_audit_summary.csv
exp39_targeted_audit/<design>_continuous_margin_scan.csv
exp39_targeted_audit/<design>_worst_slice_audit.csv
exp39_targeted_audit/<design>_profiled_alias.csv
```

Or audit one design only:

```powershell
.\run_exp39_targeted_g.ps1 -DesignId mixed_targeted_200
```

## Designs generated

- `robust3p_eopt_200`: robust 3P E-optimal design across all m slices.
- `worstslice_agamma_200`: specifically maximizes a1/gamma information on diagnosed worst m slices.
- `mixed_targeted_200`: first targets worst-slice a1/gamma, then balances full 3P information.
- `mixed_targeted_256`: higher-point version of the mixed strategy.
- `hybrid_plus_targeted_240`: preserves the Exp38 champion and adds targeted points.
- `hybrid_near_200`: unchanged Exp38 champion comparator.

## Decision rule

Do **not** choose the final g by the greedy/Fisher rank alone. Use the deep continuous audit. The current best design should improve both:

- full-domain continuous 3P alias Mahalanobis P10;
- the worst fixed-m a1/gamma slice;

while keeping the full parameter domains unchanged.

After a winner is identified, the next stage is full 48k candidate construction + continuous near-degeneracy filtering + coverage-controlled 260-state selection + 0.2% VarPro ceiling. Only after the physics ceiling approaches the >=95% per-parameter target should TCN training resume.
