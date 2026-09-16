param(
    [int]$Workers = 4,
    [switch]$Quick
)
$ErrorActionPreference = "Stop"
$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Need = @(
  ".\mc_physics.py", ".\mc_pool_config.py", ".\cp02_core.py", ".\cp02_observation.py",
  ".\analyze_cp02_varpro_ceiling.py", ".\cp02_corrected_3p_config.json",
  ".\cp02_threshold_sweep_config.json", ".\sweep_cp02_thresholds.py", ".\validate_cp02_threshold_sweep.py",
  ".\cp02_identifiable_bank\cp02_refined_prebank.csv", ".\cp02_identifiable_bank\q2.npy"
)
foreach ($f in $Need) { if (-not (Test-Path $f)) { throw "Missing required file: $f" } }

$args2 = @(
  "sweep_cp02_thresholds.py",
  "--config","cp02_corrected_3p_config.json",
  "--sweep-config","cp02_threshold_sweep_config.json",
  "--bank-dir","cp02_identifiable_bank",
  "--output-dir","cp02_threshold_sweep_v2",
  "--workers",$Workers
)
if ($Quick) { $args2 += "--quick" }

& $Python @args2
if ($LASTEXITCODE -ne 0) { throw "CP02-v2 threshold sweep failed" }

& $Python validate_cp02_threshold_sweep.py --config cp02_corrected_3p_config.json --output-dir cp02_threshold_sweep_v2
if ($LASTEXITCODE -ne 0) { throw "CP02-v2 validation failed" }

Write-Host "CP02-v2 complete. Key outputs:"
Write-Host "  cp02_threshold_sweep_v2\cp02_v2_threshold_sweep_summary.csv"
Write-Host "  cp02_threshold_sweep_v2\cp02_v2_threshold_confirm_summary.csv"
Write-Host "  cp02_threshold_sweep_v2\cp02_v2_region_coverage.csv"
Write-Host "  cp02_threshold_sweep_v2\cp02_v2_recommendation.json"
Write-Host "  cp02_threshold_sweep_v2\cp02_v2_selected_bank.npz"
