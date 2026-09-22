param(
    [string]$Device = "cuda",
    [int]$Seed = 20260920,
    [int]$Epochs = 200
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

Write-Host "CP02 TCN training"
Write-Host "device=$Device seed=$Seed epochs=$Epochs"

# Reuse the same CP02 dataset used by the MLP baseline.
# Only replace the backbone with TCN.
& $Python train_cp02_tcn.py `
    --data-dir ".\data_cp02_ml_v1" `
    --device $Device `
    --seed $Seed `
    --epochs $Epochs

if ($LASTEXITCODE -ne 0) {
    throw "TCN training failed."
}
