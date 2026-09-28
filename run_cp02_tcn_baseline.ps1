param(
    [string]$Device = "cuda",
    [int]$Epochs = 180,
    [int]$BatchSize = 512,
    [switch]$Quick,
    [switch]$SingleSeed,
    [switch]$NoShortcut,
    [switch]$Deterministic,
    [switch]$NoAMP,
    [switch]$NoTF32
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$DataDir = ".\data_cp02_ml_hybrid_domain_v2"
$OutDir = ".\runs_cp02_tcn_v3"

$Need = @(
    "$DataDir\normalization_train_only.npz",
    "$DataDir\noise_0p2pct\train.npz",
    "$DataDir\noise_0p2pct\val.npz",
    "$DataDir\noise_0pct\test.npz",
    "$DataDir\noise_0p2pct\test.npz",
    "$DataDir\noise_1pct\test.npz",
    ".\train_cp02_tcn.py"
)
foreach ($f in $Need) {
    if (-not (Test-Path $f)) { throw "Missing required file: $f" }
}

$Seeds = @(20260920, 20260921, 20260922)
$Patience = 25
$Channels = @(64, 96, 128, 160)

if ($SingleSeed) {
    $Seeds = @(20260920)
    if (-not $PSBoundParameters.ContainsKey("Epochs")) { $Epochs = 140 }
    $Patience = 20
}

if ($Quick) {
    $Seeds = @(20260920)
    $Epochs = [Math]::Min($Epochs, 3)
    $Patience = 3
    $Channels = @(24, 32, 48)
    $BatchSize = [Math]::Min($BatchSize, 256)
}

Write-Host "CP02 3P TCN amplitude-preserving v3"
Write-Host "data=$DataDir output=$OutDir device=$Device epochs=$Epochs batch=$BatchSize"
Write-Host "seeds=$($Seeds -join ',') shortcut=$(-not $NoShortcut)"

$Args2 = @(
    "train_cp02_tcn.py",
    "--data-dir", $DataDir,
    "--output-dir", $OutDir,
    "--device", $Device,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--patience", $Patience,
    "--log-every-batches", 10,
    "--seeds"
)
$Args2 += $Seeds
$Args2 += "--channels"
$Args2 += $Channels

if ($NoShortcut) { $Args2 += "--no-shortcut" }
if ($Deterministic) { $Args2 += "--deterministic" }
if ($NoAMP) { $Args2 += "--no-amp" }
if ($NoTF32) { $Args2 += "--no-tf32" }

& $Python @Args2
if ($LASTEXITCODE -ne 0) { throw "CP02 TCN v3 training failed." }

Write-Host "Done."
Write-Host "Key output: $OutDir\aggregate_metrics.json"
