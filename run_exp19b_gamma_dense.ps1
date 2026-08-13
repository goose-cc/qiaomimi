param(
    [string]$Device = "cuda",
    [int]$GammaCount = 11,
    [double]$GammaMin = 0.01,
    [double]$GammaMax = 1.0,
    [string]$DataDir = "",
    [string]$OutputDir = "",
    [string]$TrainNoiseDir = "noise_0p2pct",
    [switch]$SkipGenerate,
    [switch]$Quick,
    [switch]$SkipPhysicsMetrics
)

$ErrorActionPreference = "Stop"

if ($GammaCount -lt 3) {
    throw "GammaCount must be >= 3"
}
if ($GammaMin -le 0 -or $GammaMax -le $GammaMin) {
    throw "Require 0 < GammaMin < GammaMax"
}

$Python = "python"
if (Test-Path ".\venv\Scripts\python.exe") {
    $Python = ".\venv\Scripts\python.exe"
}

$Suffix = "$GammaCount"
if ($Quick) {
    $Suffix = "${GammaCount}_quick"
}
if ([string]::IsNullOrWhiteSpace($DataDir)) {
    $DataDir = ".\data_gamma_dense_$Suffix"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = ".\validation_results\exp19b_gamma_dense_$Suffix"
}

# Keep the total number of examples close to Exp19A so that gamma coverage,
# not simply a much larger dataset, is the main changed variable.
$TrainTotal = 20000
$ValTotal = 2500
$TestSeenTotal = 5000
$TestInterpTotal = 4000
$Epochs = 80
$BatchSize = 256
$EvalBatchSize = 512
$Patience = 15
$MaxTrain = 0
$MaxVal = 0
$MaxTest = 0

if ($Quick) {
    $TrainTotal = 2200
    $ValTotal = 550
    $TestSeenTotal = 550
    $TestInterpTotal = 500
    $Epochs = 3
    $BatchSize = 128
    $EvalBatchSize = 256
    $Patience = 3
    $MaxTrain = 2200
    $MaxVal = 550
    $MaxTest = 550
    $SkipPhysicsMetrics = $true
}

$TrainPerGamma = [Math]::Ceiling($TrainTotal / [double]$GammaCount)
$ValPerGamma = [Math]::Ceiling($ValTotal / [double]$GammaCount)
$TestSeenPerGamma = [Math]::Ceiling($TestSeenTotal / [double]$GammaCount)
$InterpCount = $GammaCount - 1
$TestInterpPerGamma = [Math]::Ceiling($TestInterpTotal / [double]$InterpCount)

Write-Host ""
Write-Host ("=" * 96)
Write-Host "Exp19B: dense gamma-only coverage ladder"
Write-Host "Physical formula : UNCHANGED"
Write-Host "gamma anchors    : $GammaCount log-spaced points in [$GammaMin, $GammaMax]"
Write-Host "interp test      : geometric midpoint of every adjacent training pair"
Write-Host "training noise   : $TrainNoiseDir"
Write-Host "data             : $DataDir"
Write-Host "output           : $OutputDir"
Write-Host "device           : $Device"
Write-Host "epochs           : $Epochs"
Write-Host "quick            : $Quick"
Write-Host ("=" * 96)

if (-not $SkipGenerate) {
    $GenerateArgs = @(
        "data_generate_gamma_curriculum.py",
        "--output-dir", $DataDir,
        "--tier", "dense_log",
        "--tier-gamma-count", "$GammaCount",
        "--dense-train-gamma-min", "$GammaMin",
        "--dense-train-gamma-max", "$GammaMax",
        "--noise-levels", "0,0.002,0.01",
        "--train-per-gamma", "$TrainPerGamma",
        "--val-per-gamma", "$ValPerGamma",
        "--test-seen-per-gamma", "$TestSeenPerGamma",
        "--test-interp-per-gamma", "$TestInterpPerGamma",
        "--compressed",
        "--overwrite"
    )

    Write-Host ""
    Write-Host "[1/3] Generating dense-log gamma data ..."
    & $Python @GenerateArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Exp19B data generation failed with exit code $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "[2/3] Checking generated data ..."
    & $Python "check_gamma_curriculum.py" $DataDir
    if ($LASTEXITCODE -ne 0) {
        throw "Exp19B data check failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host ""
    Write-Host "[1/3] Data generation skipped."
    Write-Host "[2/3] Checking existing data ..."
    & $Python "check_gamma_curriculum.py" $DataDir
    if ($LASTEXITCODE -ne 0) {
        throw "Exp19B data check failed with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "[3/3] Training the SAME MLP architecture used in Exp19A ..."

$TrainArgs = @(
    "train_exp19b_gamma_dense.py",
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
    $TrainArgs += "--skip-physics-metrics"
}

& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Exp19B training failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Exp19B finished. Read first:"
Write-Host "  $OutputDir\exp19b_summary.csv"
Write-Host "  $OutputDir\exp19b_samples.csv"
