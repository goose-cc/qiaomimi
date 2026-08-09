param(
    [string]$OutputDir = ".\validation_results\exp12_a1_gamma_landscape",

    [ValidateSet("auto", "cpu", "cuda", "xpu")]
    [string]$Device = "cuda",

    [string]$TrueGammas = "0.01,0.05,0.2,0.5",

    [double]$A1True = 0.10,

    [double]$M = 0.8,

    [int]$A1Points = 241,

    [int]$GammaPoints = 361
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "Exp12 pure-physics a1-gamma landscape"
Write-Host "NO neural-network training"
Write-Host "output      : $OutputDir"
Write-Host "device      : $Device"
Write-Host "true gammas : $TrueGammas"
Write-Host "a1 true     : $A1True"
Write-Host "m fixed     : $M"
Write-Host "============================================================"

$Args = @(
    "analyze_exp12_a1_gamma_landscape.py",
    "--output-dir", $OutputDir,
    "--device", $Device,
    "--true-gammas", $TrueGammas,
    "--a1-true", [string]$A1True,
    "--a2", "0.025",
    "--a3", "0.0",
    "--m", [string]$M,
    "--a1-min", "0.05",
    "--a1-max", "0.20",
    "--a1-points", [string]$A1Points,
    "--gamma-min", "0.001",
    "--gamma-max", "1.0",
    "--gamma-points", [string]$GammaPoints,
    "--far-gamma-factor", "2.0",
    "--input-points", "100",
    "--output-points", "1000",
    "--integration-points", "128",
    "--data-scale", "160000",
    "--shift", "400",
    "--s-min", "0.1764",
    "--s-max", "6.0",
    "--q2-min", "-100",
    "--q2-max", "-6",
    "--physics-dtype", "float64",
    "--forward-batch-size", "512",
    "--linearity-checks", "5",
    "--seed", "20260808"
)

& $Python @Args

if ($LASTEXITCODE -ne 0) {
    throw "Exp12 failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Exp12 finished."
Write-Host "First read:"
Write-Host "  $OutputDir\exp12_summary.csv"
Write-Host ""
Write-Host "Then inspect each gamma_* folder:"
Write-Host "  a1_gamma_heatmap.png"
Write-Host "  gamma_profile.png"
Write-Host "  degenerate_g_comparison.png"
Write-Host "  degenerate_f_comparison.png"
