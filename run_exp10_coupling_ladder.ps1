param(
    [ValidateSet(
        "pairwise",
        "a1m",
        "a1gamma",
        "mg",
        "a1mg_strong",
        "a1mg_full",
        "all5_strong",
        "all5_full"
    )]
    [string]$Mode = "pairwise",

    [string]$OutputRoot = ".\model\exp10_coupling",

    [int]$MaxSteps = 30000,

    [int]$ValSamples = 10000,

    [ValidateSet("auto", "cpu", "cuda", "xpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

if ($Mode -eq "pairwise") {
    # 现在最需要的两个实验。
    $Modes = @(
        "a1m",
        "a1gamma"
    )
}
else {
    $Modes = @($Mode)
}

foreach ($M in $Modes) {
    $CheckpointDir = Join-Path $OutputRoot $M

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Exp10 pairwise coupling diagnostic"
    Write-Host "mode       : $M"
    Write-Host "output     : $CheckpointDir"
    Write-Host "max steps  : $MaxSteps"
    Write-Host "============================================================"

    $Args = @(
        "train_exp10_coupling_ladder.py",
        "--mode", $M,
        "--checkpoint-dir", $CheckpointDir,
        "--input-points", "100",
        "--output-points", "1000",
        "--integration-points", "128",
        "--noise-level", "0.0",
        "--batch-size", "64",
        "--learning-rate", "1e-3",
        "--weight-decay", "1e-5",
        "--transformer-d-model", "64",
        "--transformer-nhead", "4",
        "--transformer-num-layers", "3",
        "--transformer-dim-feedforward", "128",
        "--transformer-dropout", "0.1",
        "--max-steps", [string]$MaxSteps,
        "--log-every-steps", "100",
        "--checkpoint-every-steps", "2000",
        "--best-window", "100",
        "--val-samples", [string]$ValSamples,
        "--val-batch-size", "256",
        "--seed", "20260808",
        "--device", $Device,
        "--amp"
    )

    & $Python @Args

    if ($LASTEXITCODE -ne 0) {
        throw "Exp10 mode $M failed with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "All requested Exp10 pairwise modes finished."
Write-Host "Check:"
Write-Host "  model\exp10_coupling\a1m\exp10_summary.json"
Write-Host "  model\exp10_coupling\a1gamma\exp10_summary.json"
