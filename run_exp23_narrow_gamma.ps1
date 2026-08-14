param(
    [string]$Device = "cuda",
    [switch]$Quick,
    [switch]$Regenerate
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$SourceMetadata = ".\data_exp22a_rms_snr2\metadata.json"
$DataDir = ".\data_exp23_narrow_gamma_q500"
$BaselineDir = ".\validation_results\exp23a_dense_gamma_baseline"
$WeightedDir = ".\validation_results\exp23b_narrow_weighted"
$CompareDir = ".\validation_results\exp23_narrow_comparison"

$RequiredFiles = @(
    ".\data_generate_exp23_narrow_gamma.py",
    ".\train_exp23_narrow_peak.py",
    ".\analyze_exp23_narrow_peaks.py",
    ".\train_exp20_pairwise.py",
    ".\analyze_exp22_classification.py",
    ".\PairwiseInverseMLP.py",
    ".\data_generate_exp20_pairwise.py",
    ".\data_generate_exp22c_fixed_capacity.py"
)
foreach ($File in $RequiredFiles) {
    if (-not (Test-Path $File)) {
        throw "Required file not found: $File"
    }
}
if (-not (Test-Path $SourceMetadata)) {
    throw "Exp22A metadata not found: $SourceMetadata"
}

if ($Quick) {
    $Epochs = 3
    $IntegrationPoints = 32
    $TargetTrain = 2700
    $TargetVal = 900
    $TargetTestSeen = 900
    $TargetTestInterp = 700
    $DataDir = "${DataDir}_quick"
    $BaselineDir = "${BaselineDir}_quick"
    $WeightedDir = "${WeightedDir}_quick"
    $CompareDir = "${CompareDir}_quick"
}
else {
    $Epochs = 80
    $IntegrationPoints = 128
    $TargetTrain = 20000
    $TargetVal = 2500
    $TargetTestSeen = 4400
    $TargetTestInterp = 2200
}

Write-Host ""
Write-Host ("=" * 100)
Write-Host "Exp23: narrow-gamma curriculum + narrow-peak weighted-loss ablation"
Write-Host "Scientific controls:"
Write-Host "  unknowns remain a1 + gamma"
Write-Host "  q2 physical observations = 500"
Write-Host "  MLP input width = 1000"
Write-Host "  a1 anchors unchanged"
Write-Host "  gamma train anchors = 0.01, 0.0316228, 0.1, 0.316228, 1"
Write-Host "  total training budget stays near 20k"
Write-Host "  Exp23A: standard loss"
Write-Host "  Exp23B: SAME data/model + narrow-gamma weighting + peak-height proxy loss"
Write-Host ("=" * 100)

$Metadata = Join-Path $DataDir "metadata.json"
if ($Regenerate -or -not (Test-Path $Metadata)) {
    $DataArgs = @(
        "data_generate_exp23_narrow_gamma.py",
        "--source-metadata", $SourceMetadata,
        "--output-dir", $DataDir,
        "--gamma-anchors", "0.01,0.0316227766,0.1,0.316227766,1.0",
        "--observation-points", 500,
        "--model-input-points", 1000,
        "--reference-noise", 0.002,
        "--noise-levels", "0,0.002,0.01",
        "--train-noise-level", 0.002,
        "--integration-points", $IntegrationPoints,
        "--target-train-samples", $TargetTrain,
        "--target-val-samples", $TargetVal,
        "--target-test-seen-samples", $TargetTestSeen,
        "--target-test-interp-samples", $TargetTestInterp,
        "--compressed",
        "--overwrite"
    )
    & $Python @DataArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Exp23 data generation failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "Existing Exp23 data found; generation skipped. Use -Regenerate to rebuild."
}

Write-Host ""
Write-Host "[1/2] Exp23A baseline: denser gamma anchors, STANDARD loss"
$BaselineArgs = @(
    "train_exp20_pairwise.py",
    "--mode", "a1gamma",
    "--data-dir", $DataDir,
    "--output-dir", $BaselineDir,
    "--result-prefix", "exp23a",
    "--device", $Device,
    "--epochs", $Epochs,
    "--batch-size", 256,
    "--eval-batch-size", 512,
    "--patience", 15
)
if ($Quick) {
    $BaselineArgs += "--skip-physics-metrics"
}
& $Python @BaselineArgs
if ($LASTEXITCODE -ne 0) {
    throw "Exp23A baseline training failed with exit code $LASTEXITCODE"
}

& $Python "analyze_exp22_classification.py" `
    --data-dir $DataDir `
    --result-dir $BaselineDir `
    --result-prefix "exp23a" `
    --experiment-label "Exp23A dense-gamma baseline"
if ($LASTEXITCODE -ne 0) {
    throw "Exp23A classification analysis failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "[2/2] Exp23B narrow-weighted objective on SAME data/model"
$WeightedArgs = @(
    "train_exp23_narrow_peak.py",
    "--mode", "a1gamma",
    "--data-dir", $DataDir,
    "--output-dir", $WeightedDir,
    "--result-prefix", "exp23b",
    "--device", $Device,
    "--epochs", $Epochs,
    "--batch-size", 256,
    "--eval-batch-size", 512,
    "--patience", 15,
    "--narrow-focus-gamma", 0.1,
    "--narrow-weight-strength", 1.0,
    "--narrow-weight-power", 0.5,
    "--narrow-weight-max", 4.0,
    "--peak-proxy-loss-weight", 0.25
)
if ($Quick) {
    $WeightedArgs += "--skip-physics-metrics"
}
& $Python @WeightedArgs
if ($LASTEXITCODE -ne 0) {
    throw "Exp23B weighted training failed with exit code $LASTEXITCODE"
}

& $Python "analyze_exp22_classification.py" `
    --data-dir $DataDir `
    --result-dir $WeightedDir `
    --result-prefix "exp23b" `
    --experiment-label "Exp23B narrow-weighted"
if ($LASTEXITCODE -ne 0) {
    throw "Exp23B classification analysis failed with exit code $LASTEXITCODE"
}

if (-not $Quick) {
    & $Python "analyze_exp23_narrow_peaks.py" `
        --data-dir $DataDir `
        --baseline-dir $BaselineDir `
        --baseline-prefix "exp23a" `
        --weighted-dir $WeightedDir `
        --weighted-prefix "exp23b" `
        --output-dir $CompareDir `
        --noise-dir "noise_0p2pct" `
        --narrow-threshold 0.06 `
        --integration-points 512
    if ($LASTEXITCODE -ne 0) {
        throw "Exp23 narrow-peak comparison failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "Quick mode skips physics-based narrow-curve comparison."
}

Write-Host ""
Write-Host ("=" * 100)
Write-Host "Exp23 complete."
Write-Host "Read first:"
Write-Host "  $DataDir\metadata.json"
Write-Host "  $BaselineDir\exp23a_summary.csv"
Write-Host "  $WeightedDir\exp23b_summary.csv"
Write-Host "  $CompareDir\exp23_narrow_band_summary.csv"
Write-Host "  $CompareDir\exp23_per_gamma_summary.csv"
Write-Host "  $CompareDir\narrow_curve_fits"
Write-Host ("=" * 100)
