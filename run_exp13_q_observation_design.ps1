param(
    [string]$OutputDir = ".\validation_results\exp13_q_observation_design",

    [ValidateSet("auto", "cpu", "cuda", "xpu")]
    [string]$Device = "cuda",

    [string]$Ranges = "-200,-6;-100,-6;-50,-6;-20,-6;-10,-6;-20,-1;-10,-1;-6,-0.5",

    [string]$TrueGammas = "0.01,0.05,0.2,0.5",

    [int]$InputPoints = 100
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host ""
Write-Host "======================================================================"
Write-Host "Exp13 q^2 observation-design study"
Write-Host "NO neural-network training"
Write-Host "output      : $OutputDir"
Write-Host "device      : $Device"
Write-Host "q points    : $InputPoints"
Write-Host "true gammas : $TrueGammas"
Write-Host "ranges      : $Ranges"
Write-Host "======================================================================"

# IMPORTANT:
# Values such as "-200,-6;..." begin with '-'.
# argparse can mistake them for a new command-line option if they are passed
# as a separate token.  Bind them to the option with --name=value.
$Args = @(
    "analyze_exp13_q_observation_design.py",
    "--output-dir", $OutputDir,
    "--device", $Device,
    "--ranges=$Ranges",
    "--baseline-range=-100,-6",
    "--true-gammas", $TrueGammas,
    "--input-points", [string]$InputPoints,
    "--sampling-strategies",
        "uniform,qmax_power2,qmax_power4,chebyshev,sensitivity",

    "--a1-true", "0.10",
    "--a2", "0.025",
    "--a3", "0.0",
    "--m", "0.8",

    "--a1-min", "0.05",
    "--a1-max", "0.20",
    "--gamma-min", "0.001",
    "--gamma-max", "1.0",
    "--gamma-points", "361",
    "--far-gamma-factor", "2.0",

    "--output-points", "1000",
    "--integration-points", "128",
    "--data-scale", "160000",
    "--shift", "400",
    "--s-min", "0.1764",
    "--s-max", "6.0",
    "--physics-dtype", "float64",

    "--sensitivity-candidate-points", "2001",
    "--sensitivity-fd-rel", "0.02",
    "--forward-batch-size", "512",
    "--seed", "20260809"
)

& $Python @Args

if ($LASTEXITCODE -ne 0) {
    throw "Exp13 failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Exp13 finished."
Write-Host "First read:"
Write-Host "  $OutputDir\exp13_ranking.csv"
Write-Host ""
Write-Host "Do NOT retrain the network yet."
Write-Host "First check whether a physically allowed q^2 design improves gamma separation."
