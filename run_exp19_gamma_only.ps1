param(
    [string]$Device = "cuda",
    [string]$DataDir = ".\data_gamma_easy",
    [string]$OutputDir = ".\validation_results\exp19_gamma_only",
    [string]$TrainNoiseDir = "noise_0p2pct",
    [switch]$Quick,
    [switch]$SkipPhysicsMetrics
)

$ErrorActionPreference = "Stop"

$Python = "python"
if (Test-Path ".\venv\Scripts\python.exe") {
    $Python = ".\venv\Scripts\python.exe"
}

$Epochs = 80
$BatchSize = 256
$EvalBatchSize = 512
$Patience = 15
$MaxTrain = 0
$MaxVal = 0
$MaxTest = 0

if ($Quick) {
    $Epochs = 3
    $BatchSize = 128
    $EvalBatchSize = 256
    $Patience = 3
    $MaxTrain = 2000
    $MaxVal = 500
    $MaxTest = 500
    $SkipPhysicsMetrics = $true
}

Write-Host ""
Write-Host ("=" * 88)
Write-Host "Exp19A: gamma-only Easy MLP"
Write-Host "Physical formula: UNCHANGED"
Write-Host "data          : $DataDir"
Write-Host "training noise: $TrainNoiseDir"
Write-Host "output        : $OutputDir"
Write-Host "device        : $Device"
Write-Host "epochs        : $Epochs"
Write-Host "quick         : $Quick"
Write-Host ("=" * 88)

$ArgsList = @(
    "train_exp19_gamma_only.py",
    "--data-dir", $DataDir,
    "--train-noise-dir", $TrainNoiseDir,
    "--output-dir", $OutputDir,
    "--device", $Device,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--eval-batch-size", "$EvalBatchSize",
    "--patience", "$Patience",
    "--max-train-samples", "$MaxTrain",
    "--max-val-samples", "$MaxVal",
    "--max-test-samples", "$MaxTest"
)

if ($SkipPhysicsMetrics) {
    $ArgsList += "--skip-physics-metrics"
}

& $Python @ArgsList

if ($LASTEXITCODE -ne 0) {
    throw "Exp19A failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Exp19A finished. Read first:"
Write-Host "  $OutputDir\exp19_summary.csv"
Write-Host "  $OutputDir\exp19_samples.csv"
