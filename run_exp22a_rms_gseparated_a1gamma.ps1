param(
    [string]$Device = "cuda",
    [double]$TargetRMSSNR = 2.0,
    [switch]$Quick,
    [switch]$Regenerate
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$SNRTag = ("{0:g}" -f $TargetRMSSNR).Replace(".", "p")
$DataDir = ".\data_exp22a_rms_snr$SNRTag"
$OutputDir = ".\validation_results\exp22a_rms_snr$SNRTag"

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
Write-Host "Exp22A: RMS-SNR-controlled a1 + gamma"
Write-Host "Physical formula : UNCHANGED"
Write-Host "Network / loss   : SAME as Exp20/Exp21"
Write-Host "device           : $Device"
Write-Host "q2 points        : 100"
Write-Host "reference noise  : 0.2%"
Write-Host "target RMS-SNR   : >= $TargetRMSSNR"
Write-Host "candidate counts : a1=3..11, gamma=3..11"
Write-Host "data             : $DataDir"
Write-Host "output           : $OutputDir"
Write-Host "quick            : $Quick"
Write-Host ("=" * 96)

$Metadata = Join-Path $DataDir "metadata.json"
if ($Regenerate -or -not (Test-Path $Metadata)) {
    Write-Host ""
    Write-Host "[1/3] Selecting grid by RMS-SNR and generating data ..."
    & $Python "data_generate_exp22_rms_gseparated_a1gamma.py" `
        --output-dir $DataDir `
        --target-rms-snr $TargetRMSSNR `
        --reference-noise 0.002 `
        --a1-min 0.05 `
        --a1-max 0.20 `
        --gamma-min 0.01 `
        --gamma-max 1.0 `
        --min-a1-count 3 `
        --max-a1-count 11 `
        --min-gamma-count 3 `
        --max-gamma-count 11 `
        --q2-points 100 `
        --noise-levels "0,0.002,0.01" `
        --train-noise-level 0.002 `
        --target-train-samples $TargetTrain `
        --target-val-samples $TargetVal `
        --target-test-seen-samples $TargetTestSeen `
        --target-test-interp-samples $TargetTestInterp `
        --compressed `
        --overwrite

    if ($LASTEXITCODE -ne 0) {
        throw "Exp22A data generation failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "[1/3] Existing Exp22A data found; generation skipped."
    Write-Host "      Use -Regenerate to rebuild it."
}

Write-Host ""
Write-Host "[2/3] Training the SAME pairwise MLP ..."
$TrainArgs = @(
    "train_exp20_pairwise.py",
    "--mode", "a1gamma",
    "--data-dir", $DataDir,
    "--output-dir", $OutputDir,
    "--result-prefix", "exp22a",
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
    throw "Exp22A training failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "[3/3] Converting test_seen predictions to classification-style metrics ..."
& $Python "analyze_exp22_classification.py" `
    --data-dir $DataDir `
    --result-dir $OutputDir `
    --result-prefix "exp22a"

if ($LASTEXITCODE -ne 0) {
    throw "Exp22A classification analysis failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host ("=" * 96)
Write-Host "Exp22A complete."
Write-Host "Read first:"
Write-Host "  $DataDir\grid_search.csv"
Write-Host "  $DataDir\state_nearest_neighbor_separation.csv"
Write-Host "  $DataDir\grid_search_min_rms_snr.png"
Write-Host "  $OutputDir\exp22a_summary.csv"
Write-Host "  $OutputDir\exp22a_classification_summary.csv"
Write-Host "  $OutputDir\classification_diagnostics"
Write-Host "  $OutputDir\representative_curve_samples.csv"
Write-Host "  $OutputDir\curve_fits"
Write-Host ("=" * 96)
