param(
    [string]$ValidationPoolDir = ".\truth_pool_val_10k",

    [string]$CheckpointDir = ".\model\v2_exp3_output1000_noise5_160m",

    [ValidateSet("best", "latest")]
    [string]$Weights = "best",

    [string]$OutputRoot = ".\validation_results\v2_exp3_output1000_noise5",

    [int]$NumSamples = 10000,

    [int]$BatchSize = 64,

    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$Cases = @(
    @{ Label = "noise0"; Value = "0.0" },
    @{ Label = "noise5"; Value = "0.05" },
    @{ Label = "noise9"; Value = "0.09" }
)

foreach ($Case in $Cases) {
    $OutputDir = "${OutputRoot}_${Weights}_$($Case.Label)"
    Write-Host "============================================================"
    Write-Host "Validating $Weights at noise=$($Case.Value)"
    Write-Host "Output: $OutputDir"

    & $Python "validate_mc_transformer.py" `
        "--validation-pool-dir" $ValidationPoolDir `
        "--checkpoint-dir" $CheckpointDir `
        "--weights" $Weights `
        "--num-samples" "$NumSamples" `
        "--batch-size" "$BatchSize" `
        "--seed" "20260802" `
        "--noise-level" $Case.Value `
        "--output-dir" $OutputDir `
        "--plot-count" "12" `
        "--device" $Device

    if ($LASTEXITCODE -ne 0) {
        throw "Validation failed at noise=$($Case.Value), exit code $LASTEXITCODE"
    }
}
