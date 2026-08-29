# Universal 3P/4P/5P data pipeline

This directory is data-only infrastructure. It does **not** train MLP/TCN/Transformer models.
The physical forward model is still provided by the existing `mc_physics.py`; this patch does not
change the physics formula or the locked q2 domain `[-100, -6]`.

## Structure

- `pipeline_core.py` — shared config, sampling, forward, distance, generalized alias, box-QP profiling, coverage/noise utilities.
- `data_generate_candidate_pool.py` — clean physical-state candidate generation only.
- `analyze_identifiability.py` — physical candidate NN alias + continuous `(m,gamma)` bank / linear coefficient profiling.
- `select_identifiable_states.py` — runtime alias cutoff + g separation + normalized parameter coverage.
- `analyze_parameter_coverage.py` — 1D statistics and all 2D free-parameter projection coverage.
- `split_physical_states.py` — physical-state-first train/val/test split and cross-split g-space leakage metrics.
- `build_noisy_dataset.py` — noise expansion only after physical split.
- `validate_dataset.py` — hard validation; failures raise errors.
- `configs/stage_3p.json`, `stage_4p.json`, `stage_5p.json` — all ranges, fixed/free parameters, seeds and thresholds.
- `run_stage.py` — common one-command orchestrator used by the three PowerShell wrappers in the project root.

## Current run policy

### 3P — formal candidate + identifiability now

```powershell
.\run_stage_3p.ps1 -Overwrite
```

No final alias cutoff is stored in the 3P config. Therefore the command stops after:

`candidate -> identifiability`

and writes `alias_summary.csv` / `hard_states.csv` for cutoff review.

Once a cutoff is approved, e.g. 0.40:

```powershell
.\run_stage_3p.ps1 -AliasThreshold 0.40 -Overwrite
```

Then the full chain runs:

`candidate -> identifiability -> selection -> coverage -> split -> noise -> validate`.

### 4P / 5P — smoke by default

```powershell
.\run_stage_4p.ps1 -Overwrite
.\run_stage_5p.ps1 -Overwrite
```

The configs intentionally have `alias_threshold: null`. Their default action is smoke and the smoke
runner uses a temporary alias threshold of 0 only to exercise code paths. Smoke outputs are explicitly
marked and must not be released for training.

To run a formal 4P/5P candidate + identifiability analysis without freezing a training dataset:

```powershell
.\run_stage_4p.ps1 -Formal -Overwrite
.\run_stage_5p.ps1 -Formal -Overwrite
```

Later, after a stage-specific cutoff is approved:

```powershell
.\run_stage_4p.ps1 -Formal -AliasThreshold 0.40 -Overwrite
```

## Generalized alias definition

For every truth state, `analyze_identifiability.py` records two complementary searches:

1. **Physical candidate-pool alias** — nearest parameter-unacceptable physical candidate in g-space.
   Small/medium pools use exact `scipy.spatial.cKDTree`; large pools switch to deterministic random-projection
   KD-tree coarse search followed by exact g-distance refinement. No `N x N` matrix is materialized.
2. **Continuous profiled alias** — the existing Exp35-37 idea generalized from configuration. The nonlinear
   `(m,gamma)` bank is scanned in chunks and whichever of `a1/a2/a3` are free are box-profiled jointly.

The reported generalized `alias_score` is the more conservative of the two. For 5P a profiled winner can
violate the full rho non-negativity constraint; that case is explicitly marked
`profiled_alias_is_conservative_lower_bound=true`. This can reject a state conservatively but cannot falsely
claim that an unsafe state is identifiable.

## Parameter distance

All free parameters are normalized to configured ranges before Euclidean coverage/distance calculations.
`gamma` is normalized in `log10(gamma)` because its configured range spans orders of magnitude. Raw and
normalized distances are both retained in identifiability results.

## Required outputs

A completed full run contains at least:

- `candidate_states.csv`
- `candidate_g_clean.npz`
- `identifiability_results.csv`
- `alias_summary.csv`
- `hard_states.csv`
- `selected_states.csv`
- `coverage_summary.csv`
- `parameter_coverage_statistics.csv`
- `parameter_projection_coverage.csv`
- `split_manifest.csv`
- `split_separation_summary.csv`
- `metadata.json`
- `validation_report.json`
- `noise_xxx/train.npz`, `val.npz`, `test.npz`

Every noisy NPZ stores `gy`, `gy_noisy`, `gy_clean`, `a1`, `a2`, `a3`, `m`, `gamma`, `parameters`, and `state_id`.
