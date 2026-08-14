param(
    [string]$Device = "cuda",
    [switch]$Quick,
    [switch]$Regenerate
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$SourceMetadata = ".\data_exp22a_rms_snr2\metadata.json"
$DataDir = ".\data_exp24_dense_gamma_q500"
$DirectDir = ".\validation_results\exp24a_direct_dense"
$CoarseDir = ".\validation_results\exp24b_coarse_to_fine"
$CompareDir = ".\validation_results\exp24_comparison"

$Required = @(
    ".\data_generate_exp23_narrow_gamma.py",
    ".\train_exp20_pairwise.py",
    ".\PairwiseInverseMLP.py",
    ".\CoarseToFineA1GammaMLP.py",
    ".\train_exp24_coarse_to_fine.py",
    ".\analyze_exp24_coarse_to_fine.py"
)
foreach ($f in $Required) {
    if (-not (Test-Path $f)) { throw "Required file not found: $f" }
}
if (-not (Test-Path $SourceMetadata)) {
    throw "Exp22A metadata not found: $SourceMetadata"
}

# 17 log-spaced anchors from 0.01 to 1.0.
# The four coarse intervals are:
# [0.01,0.0316228], [0.0316228,0.1], [0.1,0.316228], [0.316228,1].
$GammaAnchors = "0.01,0.0133352143,0.0177827941,0.0237137371,0.0316227766,0.0421696503,0.0562341325,0.0749894209,0.1,0.133352143,0.177827941,0.237137371,0.316227766,0.421696503,0.562341325,0.749894209,1.0"
$GammaEdges = "0.01,0.0316227766,0.1,0.316227766,1.0"

if ($Quick) {
    $Epochs = 3
    $IntegrationPoints = 32
    $TargetTrain = 3060
    $TargetVal = 765
    $TargetTestSeen = 1224
    $TargetTestInterp = 900
    $DataDir = "${DataDir}_quick"
    $DirectDir = "${DirectDir}_quick"
    $CoarseDir = "${CoarseDir}_quick"
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

Write-Host ("=" * 100)
Write-Host "Exp24: direct continuous regression vs coarse classification + within-bin regression"
Write-Host "Controls:"
Write-Host "  unknowns             = a1 + gamma"
Write-Host "  physical q2 obs      = 500"
Write-Host "  network input width  = 1000"
Write-Host "  dense gamma anchors  = 17"
Write-Host "  train budget         ~= 20k"
Write-Host "  same hidden sizes/noise/data for Exp24A and Exp24B"
Write-Host ("=" * 100)

$Meta = Join-Path $DataDir "metadata.json"
if ($Regenerate -or -not (Test-Path $Meta)) {
    & $Python "data_generate_exp23_narrow_gamma.py" `
        --source-metadata $SourceMetadata `
        --output-dir $DataDir `
        --gamma-anchors $GammaAnchors `
        --observation-points 500 `
        --model-input-points 1000 `
        --reference-noise 0.002 `
        --noise-levels "0,0.002,0.01" `
        --train-noise-level 0.002 `
        --integration-points $IntegrationPoints `
        --target-train-samples $TargetTrain `
        --target-val-samples $TargetVal `
        --target-test-seen-samples $TargetTestSeen `
        --target-test-interp-samples $TargetTestInterp `
        --compressed `
        --overwrite
    if ($LASTEXITCODE -ne 0) {
        throw "Exp24 data generation failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "Existing Exp24 data found; use -Regenerate to rebuild."
}

Write-Host ""
Write-Host "[1/3] Exp24A direct continuous regression baseline"
$DirectArgs = @(
    "train_exp20_pairwise.py",
    "--mode", "a1gamma",
    "--data-dir", $DataDir,
    "--output-dir", $DirectDir,
    "--result-prefix", "exp24a",
    "--device", $Device,
    "--epochs", $Epochs,
    "--batch-size", 256,
    "--eval-batch-size", 512,
    "--patience", 15
)
if ($Quick) { $DirectArgs += "--skip-physics-metrics" }
& $Python @DirectArgs
if ($LASTEXITCODE -ne 0) {
    throw "Exp24A training failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "[2/3] Exp24B coarse-to-fine gamma"
$CoarseArgs = @(
    "train_exp24_coarse_to_fine.py",
    "--data-dir", $DataDir,
    "--output-dir", $CoarseDir,
    "--result-prefix", "exp24b",
    "--device", $Device,
    "--epochs", $Epochs,
    "--batch-size", 256,
    "--eval-batch-size", 512,
    "--patience", 15,
    "--gamma-edges", $GammaEdges
)
if ($Quick) { $CoarseArgs += "--skip-physics-metrics" }
& $Python @CoarseArgs
if ($LASTEXITCODE -ne 0) {
    throw "Exp24B training failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "[3/3] Direct-vs-coarse comparison"
if (-not $Quick) {
    & $Python "analyze_exp24_coarse_to_fine.py" `
        --baseline-dir $DirectDir `
        --baseline-prefix "exp24a" `
        --coarse-dir $CoarseDir `
        --coarse-prefix "exp24b" `
        --output-dir $CompareDir `
        --noise-dir "noise_0p2pct" `
        --gamma-edges $GammaEdges
    if ($LASTEXITCODE -ne 0) {
        throw "Exp24 analysis failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "Quick mode skips final comparison because physics metrics are skipped."
}

Write-Host ("=" * 100)
Write-Host "Exp24 complete."
Write-Host "Read first:"
Write-Host "  $DirectDir\exp24a_summary.csv"
Write-Host "  $CoarseDir\exp24b_summary.csv"
Write-Host "  $CoarseDir\exp24b_confusion.csv"
Write-Host "  $CompareDir\exp24_comparison_summary.csv"
Write-Host "  $CompareDir\exp24_catastrophic_branch_jumps.csv"
Write-Host "  $CompareDir\predicted_vs_oracle_bin_test_interp_gamma.png"
Write-Host ("=" * 100)
