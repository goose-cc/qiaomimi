param(
    [int]$Workers = 4,
    [string]$OutputDir = "data_cp02_ml_immutable_v2",
    [switch]$Quick,
    [switch]$NoVerifiedSeed
)

$buildArgs = @(
    "build_cp02_ml_dataset_v1.py",
    "--physics-config", "cp02_corrected_3p_config.json",
    "--dataset-config", "cp02_ml_dataset_v1_config.json",
    "--output-dir", $OutputDir,
    "--workers", $Workers
)
if ($Quick) { $buildArgs += "--quick" }
if ($NoVerifiedSeed) { $buildArgs += "--no-verified-seed" }

python @buildArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python validate_cp02_ml_dataset_v1.py `
    --physics-config cp02_corrected_3p_config.json `
    --dataset-dir $OutputDir
exit $LASTEXITCODE
