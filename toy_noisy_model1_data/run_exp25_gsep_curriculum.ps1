param(
    [string]$Device = "cuda",
    [switch]$Quick
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

# Reuse the EXACT dense continuous-regression data created for Exp24.
$DataDir = ".\data_exp24_dense_gamma_q500"
$BaselineDir = ".\validation_results\exp25a_full_from_start"
$CurriculumDir = ".\validation_results\exp25b_gsep_curriculum"
$CompareDir = ".\validation_results\exp25_comparison"

$Required = @(
    ".\train_exp25_gsep_curriculum.py",
    ".\analyze_exp25_gsep_curriculum.py",
    ".\train_exp20_pairwise.py",
    ".\PairwiseInverseMLP.py",
    ".\data_generate_exp20_pairwise.py",
    ".\data_generate_exp22c_fixed_capacity.py"
)
foreach ($f in $Required) {
    if (-not (Test-Path $f)) { throw "Required file not found: $f" }
}
if (-not (Test-Path (Join-Path $DataDir "metadata.json"))) {
    throw "Exp24 dense data not found: $DataDir. Keep/recreate the Exp24 dense data before Exp25."
}
if (-not (Test-Path (Join-Path $DataDir "state_nearest_neighbor_separation.csv"))) {
    throw "Missing g-separation diagnostics in $DataDir."
}

if ($Quick) {
    $StageEpochs = "2,2,3"
    $MaxTrain = 4000
    $MaxVal = 1200
    $MaxTest = 1000
    $BaselineDir = "${BaselineDir}_quick"
    $CurriculumDir = "${CurriculumDir}_quick"
    $CompareDir = "${CompareDir}_quick"
}
else {
    $StageEpochs = "20,20,40"
    $MaxTrain = 0
    $MaxVal = 0
    $MaxTest = 0
}

Write-Host ("=" * 100)
Write-Host "Exp25: continuous one-to-one g-space curriculum"
Write-Host "IMPORTANT:"
Write-Host "  This does NOT merely test that large g differences are easier."
Write-Host "  Both models are evaluated on the SAME dense/hard Exp24 test distribution."
Write-Host "  Exp25A sees the full hard training set from epoch 1."
Write-Host "  Exp25B sees easy -> medium -> full, then finishes on the same full hard set."
Write-Host "  Same model/loss/seed/epoch length/total epochs."
Write-Host ("=" * 100)

$Common = @(
    "--data-dir", $DataDir,
    "--device", $Device,
    "--seed", 20260814,
    "--stage-epochs", $StageEpochs,
    "--easy-fraction-per-gamma", 0.45,
    "--medium-fraction-per-gamma", 0.75,
    "--batch-size", 256,
    "--eval-batch-size", 512,
    "--max-train-samples", $MaxTrain,
    "--max-val-samples", $MaxVal,
    "--max-test-samples", $MaxTest
)

Write-Host ""
Write-Host "[1/3] Exp25A full dense training from epoch 1"
& $Python "train_exp25_gsep_curriculum.py" `
    --strategy "full" `
    --output-dir $BaselineDir `
    --result-prefix "exp25a" `
    @Common
if ($LASTEXITCODE -ne 0) {
    throw "Exp25A failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "[2/3] Exp25B easy -> medium -> full g-space curriculum"
& $Python "train_exp25_gsep_curriculum.py" `
    --strategy "curriculum" `
    --output-dir $CurriculumDir `
    --result-prefix "exp25b" `
    @Common
if ($LASTEXITCODE -ne 0) {
    throw "Exp25B failed with exit code $LASTEXITCODE"
}

if (-not $Quick) {
    Write-Host ""
    Write-Host "[3/3] Separation-conditioned analysis on the SAME hard test set"
    & $Python "analyze_exp25_gsep_curriculum.py" `
        --data-dir $DataDir `
        --baseline-dir $BaselineDir `
        --baseline-prefix "exp25a" `
        --curriculum-dir $CurriculumDir `
        --curriculum-prefix "exp25b" `
        --output-dir $CompareDir `
        --noise-dir "noise_0p2pct" `
        --reference-noise 0.002 `
        --observation-points 500 `
        --model-input-points 1000 `
        --integration-points 128
    if ($LASTEXITCODE -ne 0) {
        throw "Exp25 analysis failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "Quick mode skips the final physics separation analysis."
}

Write-Host ("=" * 100)
Write-Host "Exp25 complete."
Write-Host "Read first:"
Write-Host "  $BaselineDir\exp25a_summary.csv"
Write-Host "  $CurriculumDir\exp25b_summary.csv"
Write-Host "  $CompareDir\exp25_comparison_summary.csv"
Write-Host "  $CompareDir\exp25_error_by_gseparation_quartile.csv"
Write-Host "  $CompareDir\exp25_catastrophic_gamma_cases.csv"
Write-Host ("=" * 100)
