param(
    [Parameter(Mandatory = $true)]
    [string]$PoolDir,

    [string]$CheckpointDir = ".\model\v2_exp1_random",

    [ValidateSet("fresh", "resume")]
    [string]$Mode = "fresh",

    [int]$MaxSteps = 0,

    [double]$MaxHours = 23.0
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$ModeFlag = if ($Mode -eq "resume") { "--resume" } else { "--fresh" }

$TrainArgs = @(
    "train_mc_parameter_pool_transformer_loss.py",
    "--pool-dir", $PoolDir,
    "--checkpoint-dir", $CheckpointDir,
    "--model-type", "transformer",
    "--sampling-mode", "random",
    "--active-block-size", "200000",
    "--precompute-chunk-size", "4096",
    "--integration-points", "128",
    "--input-points", "100",
    "--output-points", "100",
    "--batch-size", "64",
    "--noise-level", "0.09",
    "--loss-profile", "pinn",
    "--loss-normalization", "relative",
    "--physics-target", "clean",
    "--lambda-grad", "0.1",
    "--lambda-physics", "0.1",
    "--learning-rate", "1e-3",
    "--weight-decay", "1e-5",
    "--max-hours", "$MaxHours",
    "--max-steps", "$MaxSteps",
    "--checkpoint-every-steps", "2000",
    "--log-every-steps", "100",
    "--amp",
    "--require-complete-pool",
    $ModeFlag
)

Write-Host "Experiment 1: V2 full-pool random sampling"
Write-Host "Pool:       $PoolDir"
Write-Host "Checkpoint: $CheckpointDir"
Write-Host "Mode:       $Mode"

& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}
