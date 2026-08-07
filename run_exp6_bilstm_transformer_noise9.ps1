param(
    [Parameter(Mandatory = $true)]
    [string]$PoolDir,

    [string]$CheckpointDir = ".\model\v2_exp6_bilstm_transformer_noise9_160m",

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

$TrainArgs = @()
$TrainArgs += "train_mc_parameter_pool_transformer_loss.py"
$TrainArgs += @("--pool-dir", $PoolDir)
$TrainArgs += @("--checkpoint-dir", $CheckpointDir)

# Exp6 = Exp2 training conditions + Exp5 BiLSTM front-end.
$TrainArgs += @("--model-type", "bilstm_transformer")
$TrainArgs += @("--sampling-mode", "shuffle")
$TrainArgs += @("--shuffle-block-size", "200000")
$TrainArgs += @("--active-block-size", "200000")
$TrainArgs += @("--precompute-chunk-size", "1024")
$TrainArgs += @("--integration-points", "128")
$TrainArgs += @("--input-points", "100")
$TrainArgs += @("--output-points", "1000")
$TrainArgs += @("--batch-size", "64")
$TrainArgs += @("--micro-batch-size", [string]$MicroBatchSize)

# Restore Exp2 target training noise.
$TrainArgs += @("--noise-level", "0.09")

# Keep the same loss setup used by Exp2/Exp5.
$TrainArgs += @("--loss-profile", "pinn")
$TrainArgs += @("--loss-normalization", "relative")
$TrainArgs += @("--physics-target", "clean")
$TrainArgs += @("--lambda-grad", "0.1")
$TrainArgs += @("--gradient-reference-points", "100")
$TrainArgs += @("--lambda-physics", "0.1")
$TrainArgs += @("--learning-rate", "1e-3")
$TrainArgs += @("--weight-decay", "1e-5")

# Same BiLSTM structure as Exp5.
$TrainArgs += @("--lstm-hidden-size", "32")
$TrainArgs += @("--lstm-num-layers", "2")
$TrainArgs += @("--lstm-dropout", "0.1")
$TrainArgs += "--lstm-residual"

$TrainArgs += @("--seed", "20260721")
$TrainArgs += @("--max-hours", [string]$MaxHours)
$TrainArgs += @("--max-steps", [string]$MaxSteps)
$TrainArgs += @("--checkpoint-every-steps", "2000")
$TrainArgs += @("--log-every-steps", "100")
$TrainArgs += "--amp"
$TrainArgs += "--require-complete-pool"
$TrainArgs += $ModeFlag

Write-Host "Experiment 6: Exp2 conditions + BiLSTM + Transformer"
Write-Host "Pool:             $PoolDir"
Write-Host "Checkpoint:       $CheckpointDir"
Write-Host "Mode:             $Mode"
Write-Host "Model:            BiLSTM(2 layers, bidirectional) + Transformer"
Write-Host "Sampling:         shuffled without replacement"
Write-Host "Input points:     100"
Write-Host "Output points:    1000"
Write-Host "Training noise:   9% RMS Gaussian"
Write-Host "Logical batch:    64"
Write-Host "Micro batch:      $MicroBatchSize"
Write-Host "Gradient ref grid:100 points"
Write-Host "Seed:             20260721"

& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}
