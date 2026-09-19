# CP02 ML Dataset Builder — immutable-domain update

This repository now supports the paper/immutable build:

```text
a1    in [0.05, 0.20]
m     in [0.40, 1.20]
gamma in [0.01, 1.00]
a2    = 0.025
a3    = 0
q^2   in [-300, -6] GeV^2, 240 fixed linear points
recovery: |Δa1|<=0.005, |Δm|<=0.03, F_gamma<=1.2
acceptance: D_M^continuous >= 1.5
```

The older `hybrid_ultranear_240` / broad-domain dataset is not compatible with this immutable build. Rebuild the dataset into a new output directory, for example `data_cp02_ml_immutable_v2`, before training. Do not reuse the old `data_cp02_ml_v1` NPZ files for this definition.

# CP02 ML Physical-State Dataset v1

This builder starts **after** the physics/domain decision is frozen. It does not optimize q² and it does not retune the identifiability threshold.

Frozen definition:

- physics formula version: `rho-m2-center-v1` (resonance center `m^2`)
- observation design: `immutable_linear_240_q2_m300_m6`
- scientific acceptance rule: `D_M^continuous >= 1.5`
- reference noise used by the margin: `0.2%`
- recovery tolerances: `|da1|<=0.005`, `|dm|<=0.03`, gamma factor `<=1.2`

## Why this is a new builder

`build_cp02_identifiable_bank.py` was designed to produce a small coverage/ceiling bank. Its final 260 states are useful for physics verification, but they are not a formal neural-network dataset. The ML builder therefore creates many more **distinct physical states**, makes the physical-state split before adding noise, and records permanent state identity and continuous margin for every accepted state.

Candidate states are sampled only inside the immutable `regular_numerical_domain` stated above. No q²/design/domain search is performed inside this builder.


### Important q² note

The immutable build uses `immutable_linear_240_q2_m300_m6`: exactly 240 linear Euclidean q² points on `[-300,-6]`. It intentionally does **not** reuse `cp02_identifiable_bank/q2.npy` from the earlier `hybrid_ultranear_240` run, because that artifact was calibrated under a different observation design.


## Screening/refinement logic

1. Generate 65,536 Sobol candidate physical states over the existing regular numerical domain.
2. Compute a cheap coarse grid alias margin in batches.
3. Safely discard candidates whose coarse grid margin is already below 1.5. A discretized grid minimum is an upper bound on the continuous minimum, so these cannot pass the continuous threshold.
4. Interleave surviving candidates across 3D `(a1,m,gamma)` coverage cells so refinement does not collapse into the middle of parameter space.
5. Continuously refine candidates in that coverage-balanced order until 12,000 accepted physical states are obtained.
6. A new state is accepted only when its **continuous** margin is at least 1.5. The coarse grid value is never written as the scientific label.
7. The prior broad-domain CP02 refined prebank is disabled for this immutable build, because it used a different q² design/domain. New accepted states are continuously refined under the immutable definition.
8. A deterministic continuous audit re-runs the **same frozen CP02 continuous definition** on both near-threshold states and a random accepted sample. Any disagreement/failure aborts publication instead of silently lowering the threshold.

Continuous-refinement results are cached in `continuous_refinement_cache.csv`, keyed by permanent physical-state ID and a refinement signature, so interrupted runs can resume without recomputing completed states.

## Physical split and noise

The default final bank contains 12,000 unique physical states and is split approximately 80/10/10 **before noise** (normally 9,600/1,200/1,200). The split is stratified by the same coverage bins used in the report.

Noise model:

`g_noisy = g_clean + noise_level * RMS(g_clean) * Normal(0,1)`

Outputs are:

- train: 0.2% noise, 3 independent replicas per train physical state
- validation: 0.2% noise, 1 replica per validation physical state
- test: the same test physical states at 0%, 0.2%, and 1% noise

Train-only input/target normalization statistics are saved separately. Targets are `[a1, m, log(gamma)]`; gamma uses natural-log space before standardization.

## Run

Smoke test:

```powershell
.\run_cp02_ml_dataset_v1.ps1 -Quick -Workers 2 -OutputDir data_cp02_ml_immutable_v2_quick
```

Formal build:

```powershell
.\run_cp02_ml_dataset_v1.ps1 -Workers 4 -OutputDir data_cp02_ml_immutable_v2
```

The prior broad-domain refined prebank is intentionally disabled for the immutable build. Run:

```powershell
.\run_cp02_ml_dataset_v1.ps1 -Workers 4 -NoVerifiedSeed -OutputDir data_cp02_ml_immutable_v2
```

The immutable config sets `verified_seed.use_if_present=false`, so `-NoVerifiedSeed` is optional. Keeping it in the command is harmless and makes the no-reuse intent explicit.

## Main outputs

- `physical_states.npz`: one row per unique physical state; float64 `params`, `g_clean`, `continuous_margin`, ID, split, q²
- `physical_states.csv`: human-readable physical-state manifest with alias diagnostics
- `noise_0p2pct/train.npz`: train samples with replicas
- `noise_0p2pct/val.npz`: validation samples
- `noise_0pct/test.npz`, `noise_0p2pct/test.npz`, `noise_1pct/test.npz`: identical test physical states at the three noise levels
- `normalization_train_only.npz/json`: statistics derived only from train data
- `coverage_by_axis.csv`: candidate / grid-survivor / accepted / train / val / test counts by `a1`, `m`, `gamma` bin
- `coverage_3d_cells.csv`: the same accounting by 3D parameter cell
- `continuous_audit.csv`: post-acceptance re-audit using the same frozen CP02 continuous definition
- `candidate_screen.npz`: reproducible coarse-screen record for the full new candidate pool
- `metadata.json`: formula/design/threshold/seeds/noise/split/refinement metadata
- `validation_summary.csv` and `validation_result.json`: integrity checks produced by the validator

## Leakage guarantees

`physical_state_id` is a stable hash of the frozen physics/design context plus the float64 five-parameter state. A state receives exactly one of `train`, `val`, or `test` before any noisy replicas are generated. Every sample also records `noise_level` and `noise_replica_id`, so leakage can be audited directly.

The builder also stores an integer `state_id` hash for compatibility with simple PyTorch loaders, but `physical_state_id` is the canonical permanent identity.
