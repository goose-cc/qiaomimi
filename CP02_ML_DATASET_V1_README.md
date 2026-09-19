# CP02 ML Physical-State Dataset v1

This builder starts **after** the CP02 physics/identifiability decision is frozen. It does not redesign q² and it does not retune the identifiability threshold.

Frozen definition:

- physics formula version: `rho-m2-center-v1` (resonance center `m^2`)
- observation design: `hybrid_ultranear_240`
- scientific acceptance rule: `D_M^continuous >= 1.5`
- reference noise used by the margin: `0.2%`
- recovery tolerances: `|da1|<=0.005`, `|dm|<=0.03`, gamma factor `<=1.2`

## Why this is a new builder

`build_cp02_identifiable_bank.py` was designed to produce a small coverage/ceiling bank. Its final 260 states are useful for physics verification, but they are not a formal neural-network dataset. The ML builder therefore creates many more **distinct physical states**, makes the physical-state split before adding noise, and records permanent state identity and continuous margin for every accepted state.

The empirical identifiable ranges seen in CP02 are treated as coverage references only. Candidate states are still sampled from the existing CP02 `regular_numerical_domain`; no new hand-written hard range such as `a1>=0.028` is introduced.


### Important q² point-count note

The current repository has a naming/count mismatch that should **not** be silently "fixed" inside the ML builder: `hybrid_ultranear_240` requests 48 + 192 segment points, but both segments contain `q²=-20`, and `cp02_observation.make_q2()` removes duplicate q² values. Therefore the current generated design contains **239 unique q² points**.

Because the `D_M=1.5` threshold was calibrated on the CP02 observation artifact, this builder treats the saved `cp02_identifiable_bank/q2.npy` as authoritative when it exists. It will not append a duplicate measurement merely to make the tensor length 240, because that would change the Mahalanobis geometry after calibration. Metadata records both the nominal design label (240) and the actual frozen q² length/hash.

If your existing `q2.npy` is genuinely 240 points, the builder preserves those exact 240 points. If it is 239, train the first ML baseline on those 239 frozen points unless you intentionally decide to redesign/revalidate CP02.

## Screening/refinement logic

1. Generate 65,536 Sobol candidate physical states over the existing regular numerical domain.
2. Compute a cheap coarse grid alias margin in batches.
3. Safely discard candidates whose coarse grid margin is already below 1.5. A discretized grid minimum is an upper bound on the continuous minimum, so these cannot pass the continuous threshold.
4. Interleave surviving candidates across 3D `(a1,m,gamma)` coverage cells so refinement does not collapse into the middle of parameter space.
5. Continuously refine candidates in that coverage-balanced order until 12,000 accepted physical states are obtained.
6. A new state is accepted only when its **continuous** margin is at least 1.5. The coarse grid value is never written as the scientific label.
7. If the prior CP02 refined prebank is present, states already verified at `D_M^continuous>=1.5` are reused as seeds; their q² grid must exactly match `hybrid_ultranear_240`.
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
.\run_cp02_ml_dataset_v1.ps1 -Quick -Workers 2 -OutputDir data_cp02_ml_v1_quick
```

Formal build:

```powershell
.\run_cp02_ml_dataset_v1.ps1 -Workers 4 -OutputDir data_cp02_ml_v1
```

If the prior refined prebank is intentionally unavailable:

```powershell
.\run_cp02_ml_dataset_v1.ps1 -Workers 4 -NoVerifiedSeed -OutputDir data_cp02_ml_v1
```

Do **not** use `-NoVerifiedSeed` merely to avoid a q² mismatch error; a mismatch indicates that results from a different observation design are being mixed.

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
