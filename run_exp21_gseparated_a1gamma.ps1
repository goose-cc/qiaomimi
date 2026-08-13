param(
    [string]$Device = "cuda",
    [double]$TargetSNR = 5.0,
    [switch]$Quick,
    [switch]$Regenerate
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$SNRTag = ("{0:g}" -f $TargetSNR).Replace(".", "p")
$DataDir = ".\data_exp21_gsep_snr$SNRTag"
$OutputDir = ".\validation_results\exp21_gsep_snr$SNRTag"

if ($Quick) {
    $DataDir = "${DataDir}_quick"
    $OutputDir = "${OutputDir}_quick"
    $Epochs = 3
    $TargetTrain = 1200
    $TargetVal = 400
    $TargetTestSeen = 500
    $TargetTestInterp = 400
}
else {
    $Epochs = 80
    $TargetTrain = 20000
    $TargetVal = 2500
    $TargetTestSeen = 4400
    $TargetTestInterp = 2200
}

Write-Host ""
Write-Host ("=" * 96)
Write-Host "Exp21: g-separation-controlled a1 + gamma"
Write-Host "Physical formula : UNCHANGED"
Write-Host "device           : $Device"
Write-Host "reference noise  : 0.2%"
Write-Host "target g-SNR     : >= $TargetSNR"
Write-Host "candidate counts : a1=3..11, gamma=3..11"
Write-Host "data             : $DataDir"
Write-Host "output           : $OutputDir"
Write-Host "quick            : $Quick"
Write-Host ("=" * 96)

$Metadata = Join-Path $DataDir "metadata.json"
if ($Regenerate -or -not (Test-Path $Metadata)) {
    Write-Host ""
    Write-Host "[1/2] Selecting g-separated grid and generating data ..."
    & $Python "data_generate_exp21_gseparated_a1gamma.py" `
        --output-dir $DataDir `
        --target-snr $TargetSNR `
        --reference-noise 0.002 `
        --a1-min 0.05 `
        --a1-max 0.20 `
        --gamma-min 0.01 `
        --gamma-max 1.0 `
        --min-a1-count 3 `
        --max-a1-count 11 `
        --min-gamma-count 3 `
        --max-gamma-count 11 `
        --noise-levels "0,0.002,0.01" `
        --train-noise-level 0.002 `
        --target-train-samples $TargetTrain `
        --target-val-samples $TargetVal `
        --target-test-seen-samples $TargetTestSeen `
        --target-test-interp-samples $TargetTestInterp `
        --compressed `
        --overwrite

    if ($LASTEXITCODE -ne 0) {
        throw "Exp21 data generation failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "[1/2] Existing Exp21 data found; generation skipped."
    Write-Host "      Use -Regenerate to rebuild it."
}

Write-Host ""
Write-Host "[2/2] Training the SAME pairwise MLP as Exp20 ..."
$TrainArgs = @(
    "train_exp20_pairwise.py",
    "--mode", "a1gamma",
    "--data-dir", $DataDir,
    "--output-dir", $OutputDir,
    "--result-prefix", "exp21",
    "--device", $Device,
    "--epochs", $Epochs,
    "--batch-size", 256,
    "--eval-batch-size", 512,
    "--patience", 15
)
if ($Quick) {
    $TrainArgs += "--skip-physics-metrics"
}

& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Exp21 training failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host ("=" * 96)
Write-Host "Exp21 complete."
Write-Host "Read first:"
Write-Host "  $DataDir\grid_search.csv"
Write-Host "  $DataDir\state_nearest_neighbor_separation.csv"
Write-Host "  $OutputDir\exp21_summary.csv"
Write-Host "  $OutputDir\representative_curve_samples.csv"
Write-Host "  $OutputDir\curve_fits"
Write-Host ("=" * 96)
