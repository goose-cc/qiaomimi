param(
    [string]$CheckpointDir = ".\checkpoints\exp17_tcn_physics_profile_noise0p2",

    [ValidateSet("auto", "cpu", "cuda", "xpu")]
    [string]$Device = "cuda",

    # Fraction, not percent:
    # 0.002 = 0.2%, which is the Exp16 threshold scale for gamma=0.01
    # when nuisance parameters are known accurately.
    [double]$NoiseLevel = 0.002,

    [int]$MaxSteps = 30000
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host ""
Write-Host "================================================================================"
Write-Host "Exp17: TCN nuisance prediction + exact-physics gamma profiling"
Write-Host "Physical formula: UNCHANGED"
Write-Host "checkpoint : $CheckpointDir"
Write-Host "device     : $Device"
Write-Host "noise      : $($NoiseLevel * 100)%"
Write-Host "steps      : $MaxSteps"
Write-Host "================================================================================"

$Args = @(
    "train_exp17_tcn_physics_profile.py",
    "--checkpoint-dir", $CheckpointDir,
    "--device", $Device,
    "--noise-level", [string]$NoiseLevel,
    "--max-steps", [string]$MaxSteps,

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

    # Target the physically meaningful resonance region first.
    "--a1-train-min", "0.05",
    "--a1-train-max", "0.20",
    "--m-train-min", "0.0001",
    "--m-train-max", "2.0",
    "--gamma-train-min", "0.001",
    "--gamma-train-max", "1.0",
    "--gamma-sampling", "log",

    # Global scale is estimated from calibration samples.
    "--global-input-scale", "0",
    "--scale-calibration-samples", "4096",

    "--tcn-channels", "64",
    "--tcn-dilations", "1,2,4,8,16",
    "--tcn-kernel-size", "3",
    "--tcn-dropout", "0.05",
    "--tcn-head-hidden", "128",

    # a1 is most important for subsequent gamma profiling; m is second.
    "--nuisance-loss-weights", "3,1,1,2",

    "--batch-size", "64",
    "--learning-rate", "0.001",
    "--weight-decay", "0.00001",
    "--grad-clip", "1.0",
    "--log-every-steps", "100",
    "--validate-every-steps", "2000",
    "--checkpoint-every-steps", "2000",

    "--val-samples", "2000",
    "--val-batch-size", "256",

    "--profile-true-gammas", "0.01,0.05,0.2,0.5",
    "--profile-samples-per-gamma", "250",
    "--profile-batch-size", "32",
    "--profile-gamma-min", "0.001",
    "--profile-gamma-max", "1.0",
    "--profile-gamma-points", "241",
    "--profile-gamma-chunk", "24",

    "--seed", "20260809"
)

& $Python @Args

if ($LASTEXITCODE -ne 0) {
    throw "Exp17 failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Exp17 finished."
Write-Host "Read these first:"
Write-Host "  $CheckpointDir\validation_best_nuisance.json"
Write-Host "  $CheckpointDir\exp17_gamma_profile_summary.csv"
