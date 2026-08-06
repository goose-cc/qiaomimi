param(
    [Parameter(Mandatory = $true)]
    [string]$PoolDir,

    [string]$CheckpointDir = ".\model\v2_exp4_output1000_noise0_160m",

    [ValidateSet("fresh", "resume")]
    [string]$Mode = "fresh",

    [int]$MaxSteps = 0,

    [double]$MaxHours = 23.0,

    [ValidateRange(1, 64)]
    [int]$MicroBatchSize = 64
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

if ($Mode -eq "resume") {
    $ModeFlag = "--resume"
}
else {
    $ModeFlag = "--fresh"
}

# 使用逐项追加，避免部分 Windows PowerShell 环境解析参数数组时报 UnexpectedToken。
$TrainArgs = @()
$TrainArgs += "train_mc_parameter_pool_transformer_loss.py"
$TrainArgs += @("--pool-dir", $PoolDir)
$TrainArgs += @("--checkpoint-dir", $CheckpointDir)
$TrainArgs += @("--model-type", "transformer")
$TrainArgs += @("--sampling-mode", "shuffle")
$TrainArgs += @("--shuffle-block-size", "200000")
$TrainArgs += @("--active-block-size", "200000")
$TrainArgs += @("--precompute-chunk-size", "1024")
$TrainArgs += @("--integration-points", "128")
$TrainArgs += @("--input-points", "100")
$TrainArgs += @("--output-points", "1000")
$TrainArgs += @("--batch-size", "64")
$TrainArgs += @("--micro-batch-size", [string]$MicroBatchSize)
$TrainArgs += @("--noise-level", "0.0")
$TrainArgs += @("--loss-profile", "pinn")
$TrainArgs += @("--loss-normalization", "relative")
$TrainArgs += @("--physics-target", "clean")
$TrainArgs += @("--lambda-grad", "0.1")
$TrainArgs += @("--gradient-reference-points", "100")
$TrainArgs += @("--lambda-physics", "0.1")
$TrainArgs += @("--learning-rate", "1e-3")
$TrainArgs += @("--weight-decay", "1e-5")
$TrainArgs += @("--seed", "20260721")
$TrainArgs += @("--max-hours", [string]$MaxHours)
$TrainArgs += @("--max-steps", [string]$MaxSteps)
$TrainArgs += @("--checkpoint-every-steps", "2000")
$TrainArgs += @("--log-every-steps", "100")
$TrainArgs += "--amp"
$TrainArgs += "--require-complete-pool"
$TrainArgs += $ModeFlag

Write-Host "Experiment 4: V2 shuffled sampling + 1000-point output + 0% training noise"
Write-Host "Pool:             $PoolDir"
Write-Host "Checkpoint:       $CheckpointDir"
Write-Host "Mode:             $Mode"
Write-Host "Input points:     100"
Write-Host "Output points:    1000"
Write-Host "Training noise:   0% - clean input"
Write-Host "Logical batch:    64"
Write-Host "Micro batch:      $MicroBatchSize"
Write-Host "Gradient ref grid:100 points"
Write-Host "Seed:             20260721"

& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}