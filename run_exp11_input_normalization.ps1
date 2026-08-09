param(
    [ValidateSet(
        "a1_compare",
        "a1",
        "a1m",
        "a1gamma",
        "pairwise_global"
    )]
    [string]$Mode = "a1_compare",

    [ValidateSet("rms", "global")]
    [string]$TokenNormalization = "global",

    [double]$GlobalInputScale = 0.0175,

    [string]$OutputRoot = ".\model\exp11_input_norm",

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

function Run-One {
    param(
        [string]$ExperimentMode,
        [string]$NormMode
    )

    $CheckpointDir = Join-Path `
        (Join-Path $OutputRoot $NormMode) `
        $ExperimentMode

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Exp11 input-normalization ablation"
    Write-Host "mode                : $ExperimentMode"
    Write-Host "token normalization : $NormMode"
    Write-Host "global input scale  : $GlobalInputScale"
    Write-Host "output              : $CheckpointDir"
    Write-Host "============================================================"

    $Args = @(
        "train_exp11_input_normalization.py",
        "--mode", $ExperimentMode,
        "--checkpoint-dir", $CheckpointDir,
        "--token-normalization", $NormMode,
        "--global-input-scale", [string]$GlobalInputScale,
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
        throw "Exp11 failed: mode=$ExperimentMode norm=$NormMode"
    }
}

if ($Mode -eq "a1_compare") {
    # Cheapest and most decisive first test:
    # same a1-only problem, original RMS tokens vs global-scale tokens.
    Run-One -ExperimentMode "a1" -NormMode "rms"
    Run-One -ExperimentMode "a1" -NormMode "global"
}
elseif ($Mode -eq "pairwise_global") {
    Run-One -ExperimentMode "a1m" -NormMode "global"
    Run-One -ExperimentMode "a1gamma" -NormMode "global"
}
else {
    Run-One -ExperimentMode $Mode -NormMode $TokenNormalization
}

Write-Host ""
Write-Host "Exp11 requested runs finished."
