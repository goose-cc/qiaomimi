param(
    [string]$OutputDir = ".\validation_results\exp14_gamma_prior_interval",

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
Write-Host "Exp14 gamma prior / interval / posterior diagnostic"
Write-Host "NO neural-network training"
Write-Host "output      : $OutputDir"
Write-Host "device      : $Device"
Write-Host "true gammas : $TrueGammas"
Write-Host "======================================================================"

$Args = @(
    "analyze_exp14_gamma_prior_interval.py",
    "--output-dir", $OutputDir,
    "--device", $Device,
    "--true-gammas", $TrueGammas,

    "--a1-true", "0.10",
    "--a2", "0.025",
    "--a3", "0.0",
    "--m", "0.8",

    "--a1-min", "0.05",
    "--a1-max", "0.20",
    "--a1-points", "401",

    "--gamma-min", "0.001",
    "--gamma-max", "1.0",
    "--gamma-points", "1601",

    "--prior-factors", "1000,100,30,10,5,3,2,1.5,1.2",
    "--compatibility-thresholds", "0.001,0.005,0.01,0.05,0.09",
    "--posterior-noise-levels", "0.05,0.09",

    # Keep the original observable. Exp13 showed q redesign was only a weak gain.
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
    "--forward-batch-size", "512",
    "--seed", "20260809"
)

& $Python @Args

if ($LASTEXITCODE -ne 0) {
    throw "Exp14 failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Exp14 finished."
Write-Host "First read:"
Write-Host "  $OutputDir\exp14_prior_shrinkage.csv"
Write-Host "  $OutputDir\exp14_posterior_summary.csv"
