param(
    [Parameter(Mandatory = $true)]
    [string]$PoolDir,

    [string]$CheckpointDir = ".\model\v2_exp9_parametric_gated_noise0_160m",

    [ValidateSet("fresh", "resume")]
    [string]$Mode = "fresh",

    [int]$MaxSteps = 30000,

    [double]$MaxHours = 0
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$ModeFlag = if ($Mode -eq "resume") { "--resume" } else { "--fresh" }

$A = @()
$A += "train_mc_parameter_pool_parametric_gated.py"
$A += @("--pool-dir", $PoolDir)
$A += @("--checkpoint-dir", $CheckpointDir)

$A += @("--input-points", "100")
$A += @("--output-points", "1000")
$A += @("--integration-points", "128")
$A += @("--noise-level", "0.0")

$A += @("--batch-size", "64")
$A += @("--shuffle-block-size", "200000")
$A += @("--learning-rate", "7e-4")
$A += @("--weight-decay", "1e-5")
$A += @("--grad-clip", "1.0")

# Mild supervision for every sample; resonance boost is conditional.
$A += @("--base-parameter-weights", "1,1,1,0.25,0.25")
$A += @("--resonance-boost-weights", "0.5,0,0,1.75,2.25")

# Based on the same resonance-visibility idea used in the previous peak diagnostics.
$A += @("--visibility-low", "0.05")
$A += @("--visibility-high", "0.20")
$A += @("--weak-excess-tolerance", "0.02")
$A += @("--resonance-warmup-steps", "3000")

$A += @("--lambda-param", "1.0")
$A += @("--lambda-spectrum", "1.0")
$A += @("--lambda-gradient", "0.10")
$A += @("--lambda-physics", "0.25")

$A += @("--lambda-resonance", "1.5")
$A += @("--lambda-width", "1.0")
$A += @("--lambda-peak-height", "0.5")

# Anti-false-peak terms.
$A += @("--lambda-visibility", "0.75")
$A += @("--lambda-weak-excess", "1.0")

$A += @("--gamma-log-floor", "1e-5")
$A += @("--transformer-d-model", "64")
$A += @("--transformer-nhead", "4")
$A += @("--transformer-num-layers", "3")
$A += @("--transformer-dim-feedforward", "128")
$A += @("--transformer-dropout", "0.1")

$A += @("--seed", "20260721")
$A += @("--max-steps", [string]$MaxSteps)
$A += @("--max-hours", [string]$MaxHours)
$A += @("--checkpoint-every-steps", "2000")
$A += @("--log-every-steps", "100")
$A += "--amp"
$A += "--require-complete-pool"
$A += $ModeFlag

Write-Host "Experiment 9: conditional resonance-gated parameter learning"
Write-Host "Based on: Exp8 model, but strong resonance losses are no longer applied to every sample"
Write-Host "Weak samples: suppress EXCESS resonance / false bumps"
Write-Host "Visible resonance samples: strengthen a1, m, log(gamma), width and resonance shape"
Write-Host "Training noise: 0%"
Write-Host "Checkpoint: $CheckpointDir"

& $Python @A
if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}
