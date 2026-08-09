param(
    [string]$OutputDir = ".\validation_results\exp16_gamma_noise_threshold",

    [ValidateSet("auto", "cpu", "cuda", "xpu")]
    [string]$Device = "cuda",

    [string]$TrueGammas = "0.01,0.05,0.2,0.5",

    [int]$Trials = 500
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host ""
Write-Host "======================================================================"
Write-Host "Exp16: gamma observability threshold under CURRENT physics"
Write-Host "NO neural-network training"
Write-Host "output      : $OutputDir"
Write-Host "device      : $Device"
Write-Host "true gammas : $TrueGammas"
Write-Host "trials      : $Trials"
Write-Host "======================================================================"

$Args = @(
    "analyze_exp16_gamma_noise_threshold.py",
    "--output-dir", $OutputDir,
    "--device", $Device,
    "--true-gammas", $TrueGammas,
    "--trials", [string]$Trials,

    "--a1-true", "0.10",
    "--a2", "0.025",
    "--a3", "0.0",
    "--m", "0.8",

    "--a1-min", "0.05",
    "--a1-max", "0.20",

    "--gamma-min", "0.001",
    "--gamma-max", "1.0",
    "--gamma-points", "1201",
    "--far-gamma-factor", "2.0",

    "--noise-levels",
        "0.09,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0",
    "--base-noise-levels", "0.05,0.09",
    "--seed", "20260809",

    # Keep the current/original observable unchanged.
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
    throw "Exp16 failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Exp16 finished."
Write-Host "First read:"
Write-Host "  $OutputDir\exp16_factor2_matched_filter.csv"
Write-Host "  $OutputDir\exp16_critical_noise_summary.csv"
Write-Host "  $OutputDir\exp16_noise_sweep.csv"
