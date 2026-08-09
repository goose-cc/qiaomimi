param(
    [string]$OutputDir = ".\validation_results\exp15_a1_constraint",

    [ValidateSet("auto", "cpu", "cuda", "xpu")]
    [string]$Device = "cuda",

    [string]$TrueGammas = "0.01,0.05,0.2,0.5"
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host ""
Write-Host "======================================================================"
Write-Host "Exp15: independent a1 constraint -> gamma identifiability"
Write-Host "NO neural-network training"
Write-Host "output      : $OutputDir"
Write-Host "device      : $Device"
Write-Host "true gammas : $TrueGammas"
Write-Host "======================================================================"

$Args = @(
    "analyze_exp15_a1_constraint.py",
    "--output-dir", $OutputDir,
    "--device", $Device,
    "--true-gammas", $TrueGammas,

    "--a1-true", "0.10",
    "--a2", "0.025",
    "--a3", "0.0",
    "--m", "0.8",

    "--a1-min", "0.05",
    "--a1-max", "0.20",

    "--a1-constraints", "free,0.5,0.2,0.1,0.05,0.02,0.01,0",

    "--gamma-min", "0.001",
    "--gamma-max", "1.0",
    "--gamma-points", "1601",
    "--far-gamma-factor", "2.0",

    # Original observation design. Exp13 q optimization is no longer the main line.
    "--q2-min", "-100",
    "--q2-max", "-6",
    "--input-points", "100",

    "--output-points", "1000",
    "--integration-points", "128",
    "--data-scale", "160000",
    "--shift", "400",
    "--s-min", "0.1764",
    "--s-max", "6.0",
    "--physics-dtype", "float64",
    "--forward-batch-size", "512"
)

& $Python @Args

if ($LASTEXITCODE -ne 0) {
    throw "Exp15 failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Exp15 finished."
Write-Host "First read:"
Write-Host "  $OutputDir\exp15_ranking.csv"
Write-Host "  $OutputDir\exp15_a1_constraint_detail.csv"
