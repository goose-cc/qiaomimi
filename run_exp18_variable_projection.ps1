param(
    [string]$OutputDir = ".\validation_results\exp18_variable_projection",

    [ValidateSet("auto", "cpu", "cuda", "xpu")]
    [string]$Device = "cuda",

    [string]$NoiseLevels = "0,0.002,0.01,0.05",

    [int]$SamplesPerGamma = 40,

    [switch]$Quick
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$MPoints = 161
$GammaPoints = 161
$SampleBatch = 16
$BasisChunk = 2048
$CDIters = 24

if ($Quick) {
    $SamplesPerGamma = 4
    $MPoints = 61
    $GammaPoints = 81
    $SampleBatch = 4
    $BasisChunk = 1024
    $CDIters = 12
}

Write-Host ""
Write-Host "================================================================================"
Write-Host "Exp18A: pure-physics variable projection"
Write-Host "NO neural-network training"
Write-Host "Physical formula: UNCHANGED"
Write-Host "output       : $OutputDir"
Write-Host "device       : $Device"
Write-Host "noise levels : $NoiseLevels  (fractions; 0.002 = 0.2%)"
Write-Host "samples/gamma: $SamplesPerGamma"
Write-Host "grid         : m=$MPoints, gamma=$GammaPoints"
Write-Host "================================================================================"

$Args = @(
    "analyze_exp18_variable_projection.py",
    "--output-dir", $OutputDir,
    "--device", $Device,
    "--noise-levels", $NoiseLevels,
    "--samples-per-gamma", [string]$SamplesPerGamma,

    "--input-points", "100",
    "--output-points", "1000",
    "--integration-points", "128",
    "--data-scale", "160000",
    "--shift", "400",
    "--s-min", "0.1764",
    "--s-max", "6.0",
    "--q2-min", "-100",
    "--q2-max", "-6",
    "--physics-dtype", "float32",

    "--a1-min", "0.05",
    "--a1-max", "0.20",
    "--m-true-min", "0.0001",
    "--m-true-max", "2.0",
    "--true-gammas", "0.01,0.05,0.2,0.5",

    "--scan-m-min", "0.0001",
    "--scan-m-max", "2.0",
    "--scan-m-points", [string]$MPoints,
    "--scan-gamma-min", "0.001",
    "--scan-gamma-max", "1.0",
    "--scan-gamma-points", [string]$GammaPoints,

    "--basis-chunk", [string]$BasisChunk,
    "--sample-batch", [string]$SampleBatch,
    "--coordinate-descent-iters", [string]$CDIters,
    "--linearity-check-samples", "64",
    "--linearity-tolerance", "0.00002",
    "--seed", "20260809"
)

& $Python @Args

if ($LASTEXITCODE -ne 0) {
    throw "Exp18 failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Exp18 finished. Read these first:"
Write-Host "  $OutputDir\exp18_summary.csv"
Write-Host "  $OutputDir\exp18_samples.csv"
