param(
    [Parameter(Mandatory = $true)] [string]$PoolDir,
    [string]$CheckpointDir = ".\model\v2_exp8_parametric_strong_noise0_160m",
    [ValidateSet("fresh", "resume")] [string]$Mode = "fresh",
    [int]$MaxSteps = 30000,
    [double]$MaxHours = 0
)
$ErrorActionPreference = "Stop"
$Python = ".\venv\Scripts\python.exe"; if (-not (Test-Path $Python)) { $Python = "python" }
$ModeFlag = if ($Mode -eq "resume") { "--resume" } else { "--fresh" }
$A=@(); $A += "train_mc_parameter_pool_parametric_strong.py"
$A += @("--pool-dir",$PoolDir); $A += @("--checkpoint-dir",$CheckpointDir)
$A += @("--input-points","100"); $A += @("--output-points","1000"); $A += @("--integration-points","128")
$A += @("--noise-level","0.0"); $A += @("--batch-size","64"); $A += @("--shuffle-block-size","200000")
$A += @("--learning-rate","7e-4"); $A += @("--weight-decay","1e-5"); $A += @("--grad-clip","1.0")
$A += @("--parameter-weights","3,1,1,5,6")
$A += @("--lambda-param","1.0"); $A += @("--lambda-spectrum","1.0"); $A += @("--lambda-resonance","3.0")
$A += @("--lambda-gradient","0.25"); $A += @("--lambda-physics","0.25"); $A += @("--lambda-width","2.0"); $A += @("--lambda-peak-height","1.0")
$A += @("--gamma-log-floor","1e-5")
$A += @("--transformer-d-model","64"); $A += @("--transformer-nhead","4"); $A += @("--transformer-num-layers","3"); $A += @("--transformer-dim-feedforward","128"); $A += @("--transformer-dropout","0.1")
$A += @("--seed","20260721"); $A += @("--max-steps",[string]$MaxSteps); $A += @("--max-hours",[string]$MaxHours)
$A += @("--checkpoint-every-steps","2000"); $A += @("--log-every-steps","100"); $A += "--amp"; $A += "--require-complete-pool"; $A += $ModeFlag
Write-Host "Experiment 8 STRONG: resonance-focused parameter learning"
Write-Host "Model: shared Transformer + resonance/background heads"
Write-Host "Resonance target: a1, m, log(gamma)"
Write-Host "Loss: strong param + spectrum + resonance + gradient + physics + log-width + peak-height"
Write-Host "Training noise: 0%"; Write-Host "Checkpoint: $CheckpointDir"
& $Python @A
if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
