param(
    [Parameter(Mandatory = $true)]
    [string]$PoolDir,

    [string]$CheckpointDir = ".\model\v2_exp2_output1000",

    [ValidateSet("fresh", "resume")]
    [string]$Mode = "fresh",

    [int]$MaxSteps = 0,

    [double]$MaxHours = 23.0,

    [ValidateRange(1, 64)]
    [int]$MicroBatchSize = 2
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
    "--sampling-mode", "shuffle",
    "--shuffle-block-size", "200000",
    "--active-block-size", "200000",
    "--precompute-chunk-size", "1024",
    "--integration-points", "128",
    "--input-points", "100",
    "--output-points", "1000",
    "--batch-size", "64",
    "--micro-batch-size", "$MicroBatchSize",
    "--noise-level", "0.09",
    "--loss-profile", "pinn",
    "--loss-normalization", "relative",
    "--physics-target", "clean",
    "--lambda-grad", "0.1",
    "--gradient-reference-points", "100",
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

Write-Host "Experiment 2: V2 shuffled sampling + 1000-point output"
Write-Host "Pool:             $PoolDir"
Write-Host "Checkpoint:       $CheckpointDir"
Write-Host "Mode:             $Mode"
Write-Host "Input points:     100"
Write-Host "Output points:    1000"
Write-Host "Logical batch:    64"
Write-Host "Micro batch:      $MicroBatchSize"
Write-Host "Gradient ref grid:100 points"

& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}
