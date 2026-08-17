param(
    [string]$Device = "cuda",
    [double]$MinRMSSNR = 1.0,
    [switch]$Quick,
    [switch]$Regenerate
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Tag = ("{0:g}" -f $MinRMSSNR).Replace(".", "p")
$DataDir = ".\data_exp26_gsep_snr$Tag"
$ResultDir = ".\validation_results\exp26_gsep_snr$Tag"
$AnalysisDir = ".\validation_results\exp26_gsep_snr${Tag}_analysis"

$Required = @(
    ".\data_generate_exp26_gseparated_independent.py",
    ".\train_exp26_continuous.py",
    ".\analyze_exp26_gseparated.py",
    ".\data_generate_exp20_pairwise.py",
    ".\data_generate_exp22c_fixed_capacity.py",
    ".\train_exp20_pairwise.py",
    ".\PairwiseInverseMLP.py"
)
foreach ($f in $Required) {
    if (-not (Test-Path $f)) { throw "Required file not found: $f" }
}

if ($Quick) {
    $CandidateCount = 1200
    $MaxStates = 90
    $IntegrationPoints = 32
    $TargetTrain = 2400
    $TargetVal = 600
    $TargetTest = 900
    $Epochs = 3
    $DataDir = "${DataDir}_quick"
    $ResultDir = "${ResultDir}_quick"
    $AnalysisDir = "${AnalysisDir}_quick"
}
else {
    $CandidateCount = 6000
    $MaxStates = 260
    $IntegrationPoints = 128
    $TargetTrain = 20000
    $TargetVal = 3000
    $TargetTest = 5000
    $Epochs = 80
}

Write-Host ("=" * 100)
Write-Host "Exp26: directly construct a globally g-separated continuous dataset"
Write-Host "NO classification / NO curriculum / NO narrow-gamma loss"
Write-Host "Unknowns             : a1 + gamma"
Write-Host "Min global RMS-SNR   : $MinRMSSNR"
Write-Host "Reference noise      : 0.2%"
Write-Host "Physical q2 obs      : 500"
Write-Host "Model input width    : 1000"
Write-Host "Train/val/test truths: physically disjoint"
Write-Host ("=" * 100)

$Meta = Join-Path $DataDir "metadata.json"
if ($Regenerate -or -not (Test-Path $Meta)) {
    Write-Host ""
    Write-Host "[1/3] Generate candidate pool -> g-space max-min packing -> independent splits"
    & $Python "data_generate_exp26_gseparated_independent.py" `
        --output-dir $DataDir `
        --candidate-count $CandidateCount `
        --max-selected-states $MaxStates `
        --min-rms-snr $MinRMSSNR `
        --reference-noise 0.002 `
        --a1-min 0.05 `
        --a1-max 0.20 `
        --gamma-min 0.01 `
        --gamma-max 1.0 `
        --observation-points 500 `
        --model-input-points 1000 `
        --integration-points $IntegrationPoints `
        --noise-levels "0,0.002,0.01" `
        --train-noise-level 0.002 `
        --target-train-samples $TargetTrain `
        --target-val-samples $TargetVal `
        --target-test-samples $TargetTest `
        --compressed `
        --overwrite
    if ($LASTEXITCODE -ne 0) {
        throw "Exp26 data generation failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "[1/3] Existing Exp26 data found; use -Regenerate to rebuild."
}

Write-Host ""
Write-Host "[2/3] Train unchanged continuous a1+gamma MLP"
$TrainArgs = @(
    "train_exp26_continuous.py",
    "--data-dir", $DataDir,
    "--output-dir", $ResultDir,
    "--result-prefix", "exp26",
    "--device", $Device,
    "--epochs", $Epochs,
    "--batch-size", 256,
    "--eval-batch-size", 512,
    "--patience", 15
)
if ($Quick) { $TrainArgs += "--skip-physics-metrics" }
& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Exp26 training failed with exit code $LASTEXITCODE"
}

if (-not $Quick) {
    Write-Host ""
    Write-Host "[3/3] Verify split separation and plot independent-test curves"
    & $Python "analyze_exp26_gseparated.py" `
        --data-dir $DataDir `
        --result-dir $ResultDir `
        --result-prefix "exp26" `
        --output-dir $AnalysisDir `
        --noise-dir "noise_0p2pct" `
        --integration-points 512
    if ($LASTEXITCODE -ne 0) {
        throw "Exp26 analysis failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "[3/3] Quick mode skips physics plots."
}

Write-Host ("=" * 100)
Write-Host "Exp26 complete."
Write-Host "Read first:"
Write-Host "  $DataDir\selected_states.csv"
Write-Host "  $DataDir\split_pair_gseparation_summary.csv"
Write-Host "  $DataDir\selected_states_by_split.png"
Write-Host "  $ResultDir\exp26_summary.csv"
Write-Host "  $AnalysisDir\exp26_per_test_state_summary.csv"
Write-Host "  $AnalysisDir\representative_curves"
Write-Host ("=" * 100)
