# Data-pipeline audit against the team handoff

## Scope conclusion

This deliverable is **data-only**. It does not define or invoke MLP, TCN, Transformer, optimizer, loss, epoch,
checkpoint, or model-training code. Existing network-training files in the repository are not modified by this
overlay.

The uploaded working tree before this corrected overlay was **not yet deliverable** under the new handoff:

1. Exp35/36/37 data logic existed, but remained legacy experiment-specific entry points.
2. Root `run_stage_3p.ps1`, `run_stage_4p.ps1`, and `run_stage_5p.ps1` referenced `data_pipeline/`, but the uploaded ZIP did not actually contain that directory.
3. The previous `universal_3p_4p_5p_data_pipeline.patch` was malformed at the 3P/4P/5P config boundaries.
4. The earlier common core imported three observation-grid helpers from an old Exp22 data script. The corrected core is self-contained and no longer depends on that legacy experiment file.

Historical Exp35/36/37 builders can be kept locally as regression references, but the new task should be submitted
through the common `data_pipeline/` framework rather than as three copied experiment implementations.

## Requirement status after this corrected overlay

| Team requirement | Status | Implementation |
|---|---|---|
| One shared implementation | PASS | `data_pipeline/pipeline_core.py` + seven stage scripts |
| Configurable arbitrary `free_params` | PASS | every parameter must be free or fixed in JSON; no experiment-number branching in the core |
| 2P-compatible config pattern | PASS | smoke-tested `free_params=[a1,gamma]`, with `a2/a3/m` fixed |
| 3P config | PASS | `configs/stage_3p.json`: `a1,m,gamma`; `a2=0.025,a3=0` |
| 4P config | PASS | `configs/stage_4p.json`: `a1,a2,m,gamma`; `a3=0` |
| Alternative 4P ordering | PASS | smoke-tested `a1,a3,m,gamma` with only JSON changes |
| 5P config | PASS | `configs/stage_5p.json`: all five free |
| Clean candidate generation only | PASS | candidate script writes no noise |
| Unique physical `state_id` | PASS | deterministic stage-prefixed IDs |
| Candidate table + large g array separated | PASS | `candidate_states.csv` + `candidate_g_clean.npz` |
| Ranges/seeds/sampling from config | PASS | LHS/random/uniform-grid supported and recorded in metadata |
| Existing physics formula/ranges preserved | PASS | calls `mc_physics.py`; stage ranges match current Exp35/36/37 definitions |
| Fixed q2 domain preserved | PASS | config validation rejects changes from `[-100,-6]` |
| Generalized unacceptable alternative | PASS | per-parameter abs thresholds + gamma factor from config |
| Raw and normalized parameter distance | PASS | range normalization; gamma uses log10 |
| `nearest_alias_rms_snr` retained | PASS | written per candidate |
| Hard-state report | PASS | `hard_states.csv` |
| No full candidate NxN matrix | PASS | cKDTree / projected cKDTree; continuous profiler is chunked |
| Runtime alias cutoff | PASS | `--alias-threshold`; no final 4P/5P cutoff in config |
| Threshold scan | PASS | `alias_summary.csv` and `selection_threshold_scan.csv` |
| Separation + parameter coverage selection | PASS | min g separation + max normalized parameter cover radius |
| Coverage statistics | PASS | min/max/mean/std/p05/p25/p50/p75/p95 |
| All 2D free-parameter projections | PASS | occupancy/retention table generated automatically |
| Retention summary | PASS | `coverage_summary.csv` |
| Physical-state-first split | PASS | split is frozen before noise expansion |
| Cross-split g leakage check | PASS | train-val/train-test/val-test RMS-SNR summary |
| Configurable noise levels | PASS | levels and samples-per-state / target totals are config driven |
| Unified NPZ fields | PASS | `gy,a1,a2,a3,m,gamma,state_id` plus clean/noisy/parameter metadata |
| Hard validation | PASS | finite/bounds/physics/duplicates/split ownership/counts/test span/direct forward |
| Complete metadata | PASS | stage/free/fixed/ranges/seeds/counts/cutoff/coverage/split/noise/code version |
| One-command PowerShell runners | PASS | root `run_stage_3p.ps1`, `run_stage_4p.ps1`, `run_stage_5p.ps1` |
| Current 3P policy | PASS | without cutoff: formal candidate + identifiability only; no training dataset frozen |
| Current 4P/5P policy | PASS | default smoke only; temporary smoke cutoff is not a formal scientific cutoff |
| Network training excluded | PASS | no trainer/model/loss/optimizer code in the new data layer |

## Alias-score convention

The directional candidate alias score uses the same RMS-noise idea as the earlier experiments:

`RMS(g_truth - g_alias) / (reference_noise * RMS(g_truth))`.

This is directional because the noise realization is generated from the truth state. Cross-split separation, where
neither state is privileged as truth, uses the symmetric average of the two state RMS values. Both definitions and
the reference noise level are recorded in metadata.

The optional continuous linear profiler is an additional conservative data-side diagnostic. Candidate physical aliases
remain separately available in `identifiability_results.csv`, so later cutoff decisions can distinguish sampled physical
neighbors from continuous profiled lower bounds.

## Validation performed on the corrected overlay

- Python compilation: PASS.
- 3P end-to-end smoke (`candidate -> identifiability -> selection -> coverage -> split -> noise -> validation`): PASS.
- 4P end-to-end smoke: PASS.
- 5P end-to-end smoke: PASS.
- Alternative `free_params=[a1,a3,m,gamma]`, `a2=0.025`: PASS with JSON change only.
- `free_params=[a1,gamma]`, `a2=0.025,a3=0,m=0.65`: PASS with JSON change only.
- 3P analysis-only mode: PASS; confirms no `selected_states.csv` or noisy dataset is frozen when no cutoff is supplied.
- 5P noisy NPZ contract includes all five physical parameters, `state_id`, clean/noisy `gy`, noise fields, q2 fields, and `free_params`.

## Submission policy

For the new team handoff, stage and submit the common `data_pipeline/` directory, the three `run_stage_*.ps1`
runners, and this audit. Legacy `build_exp35/36/37_*`, `data_generate_exp35/36/37_*`, and previous audit/equivalence
scripts are data-side and may be kept as local regression references, but they are not required as the primary new
infrastructure. Generated `data_exp*`, `data_pipeline_outputs/`, `validation_results/`, patch manifests, and temporary
full-project copies are run artifacts and should not be committed unless the team explicitly asks for them.
