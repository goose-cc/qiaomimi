param(
    [Parameter(Mandatory = $true)]
    [string]$PoolDir,

    [string]$CheckpointDir = ".\model\v2_exp7_parametric_noise0_160m",

    [ValidateSet("fresh", "resume")]
    [string]$Mode = "fresh",

    [int]$MaxSteps = 30000,
    [double]$MaxHours = 0
)

$ErrorActionPreference = "Stop"
$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$ModeFlag = if ($Mode -eq "resume") { "--resume" } else { "--fresh" }

$TrainArgs = @()
$TrainArgs += "train_mc_parameter_pool_parametric.py"
$TrainArgs += @("--pool-dir", $PoolDir)
$TrainArgs += @("--checkpoint-dir", $CheckpointDir)
$TrainArgs += @("--input-points", "100")
$TrainArgs += @("--output-points", "1000")
$TrainArgs += @("--integration-points", "128")
$TrainArgs += @("--noise-level", "0.0")
$TrainArgs += @("--batch-size", "64")
$TrainArgs += @("--shuffle-block-size", "200000")
$TrainArgs += @("--learning-rate", "1e-3")
$TrainArgs += @("--weight-decay", "1e-5")
$TrainArgs += @("--parameter-weights", "1,1,1,1,1")
$TrainArgs += @("--transformer-d-model", "64")
$TrainArgs += @("--transformer-nhead", "4")
$TrainArgs += @("--transformer-num-layers", "3")
$TrainArgs += @("--transformer-dim-feedforward", "128")
$TrainArgs += @("--transformer-dropout", "0.1")
$TrainArgs += @("--seed", "20260721")
$TrainArgs += @("--max-steps", [string]$MaxSteps)
$TrainArgs += @("--max-hours", [string]$MaxHours)
$TrainArgs += @("--checkpoint-every-steps", "2000")
$TrainArgs += @("--log-every-steps", "100")
$TrainArgs += "--amp"
$TrainArgs += "--require-complete-pool"
$TrainArgs += $ModeFlag

Write-Host "Experiment 7: parameter identifiability, clean g -> five physical parameters"
Write-Host "Pool:           $PoolDir"
Write-Host "Checkpoint:     $CheckpointDir"
Write-Host "Training noise: 0%"
Write-Host "Input:          100-point g(q^2)"
Write-Host "Target:         a1,a2,a3,m,gamma"
Write-Host "Loss:           normalized parameter MSE only"
Write-Host "Sampling:       shuffled without replacement"

& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
