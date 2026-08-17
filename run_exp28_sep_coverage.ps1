param(
    [string]$Device = "cuda",
    [string]$SourceDataDir = ".\data_exp27_gsep_snr1p5",
    [double]$MaxCoverRMSSNR = 2.0,
    [switch]$Quick,
    [switch]$Regenerate,
    [switch]$Retrain
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Tag = ("{0:g}" -f $MaxCoverRMSSNR).Replace(".", "p")
$DataDir = ".\data_exp28_sep1p5_cover$Tag"
$ResultDir = ".\validation_results\exp28_sep1p5_cover$Tag"
$PhysicsAnalysisDir = ".\validation_results\exp28_sep1p5_cover${Tag}_curves"
$CompareDir = ".\validation_results\exp28_sep1p5_cover${Tag}_comparison"
$Exp27Scan = ".\validation_results\exp27_gsep_threshold_scan\exp27_threshold_scan_summary.csv"

if ($Quick) {
    $DataDir = "${DataDir}_quick"
    $ResultDir = "${ResultDir}_quick"
    $PhysicsAnalysisDir = "${PhysicsAnalysisDir}_quick"
    $CompareDir = "${CompareDir}_quick"
    $Epochs = 3
    $TargetTrain = 3000
    $TargetVal = 800
    $TargetTest = 1200
    $IntegrationPoints = 32
}
else {
    $Epochs = 80
    $TargetTrain = 20000
    $TargetVal = 3000
    $TargetTest = 5000
    $IntegrationPoints = -1
}

$Required = @(
    ".\data_generate_exp28_sep_coverage.py",
    ".\analyze_exp28_sep_coverage.py",
    ".\data_generate_exp26_gseparated_independent.py",
    ".\train_exp26_continuous.py",
    ".\analyze_exp26_gseparated.py",
    ".\train_exp20_pairwise.py",
    ".\PairwiseInverseMLP.py"
)
foreach ($f in $Required) {
    if (-not (Test-Path $f)) { throw "Required file not found: $f" }
}
if (-not (Test-Path (Join-Path $SourceDataDir "metadata.json"))) {
    throw "Exp27 threshold=1.5 source data not found: $SourceDataDir"
}
if (-not (Test-Path (Join-Path $SourceDataDir "selected_states.csv"))) {
    throw "Source selected_states.csv not found: $SourceDataDir"
}

Write-Host ("=" * 100)
Write-Host "Exp28: separation + coverage constrained a1+gamma inversion"
Write-Host ""
Write-Host "Keeps the SAME Exp27 SNR~1.5 globally separated physical state bank."
Write-Host "Only the train/val/test split policy changes:"
Write-Host "  - global g separation remains ~1.5"
Write-Host "  - every val/test state must lie within RMS-SNR <= $MaxCoverRMSSNR of TRAIN"
Write-Host "  - train/val/test truths remain physically disjoint"
Write-Host ""
Write-Host "Network / loss / forward physics are unchanged."
Write-Host ("=" * 100)

$Meta = Join-Path $DataDir "metadata.json"
if ($Regenerate -or -not (Test-Path $Meta)) {
    Write-Host "[1/4] Build separation + coverage constrained split"
    & $Python "data_generate_exp28_sep_coverage.py" `
        --source-data-dir $SourceDataDir `
        --output-dir $DataDir `
        --max-train-cover-rms-snr $MaxCoverRMSSNR `
        --integration-points $IntegrationPoints `
        --noise-levels "0,0.002,0.01" `
        --train-noise-level 0.002 `
        --target-train-samples $TargetTrain `
        --target-val-samples $TargetVal `
        --target-test-samples $TargetTest `
        --compressed `
        --overwrite
    if ($LASTEXITCODE -ne 0) {
        throw "Exp28 data generation failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "[1/4] Existing Exp28 data found; use -Regenerate to rebuild."
}

$Summary = Join-Path $ResultDir "exp28_summary.csv"
if ($Retrain -or -not (Test-Path $Summary)) {
    Write-Host "[2/4] Train unchanged continuous a1+gamma MLP"
    $TrainArgs = @(
        "train_exp26_continuous.py",
        "--data-dir", $DataDir,
        "--output-dir", $ResultDir,
        "--result-prefix", "exp28",
        "--device", $Device,
        "--epochs", $Epochs,
        "--batch-size", 256,
        "--eval-batch-size", 512,
        "--patience", 15
    )
    if ($Quick) { $TrainArgs += "--skip-physics-metrics" }
    & $Python @TrainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Exp28 training failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "[2/4] Existing Exp28 result found; use -Retrain to rerun."
}

if (-not $Quick) {
    Write-Host "[3/4] Representative independent-test f/g curves"
    & $Python "analyze_exp26_gseparated.py" `
        --data-dir $DataDir `
        --result-dir $ResultDir `
        --result-prefix "exp28" `
        --output-dir $PhysicsAnalysisDir `
        --noise-dir "noise_0p2pct" `
        --integration-points 512
    if ($LASTEXITCODE -ne 0) {
        throw "Exp28 curve analysis failed with exit code $LASTEXITCODE"
    }

    Write-Host "[4/4] Coverage + stage-gate comparison"
    $AnalysisArgs = @(
        "analyze_exp28_sep_coverage.py",
        "--data-dir", $DataDir,
        "--result-dir", $ResultDir,
        "--result-prefix", "exp28",
        "--output-dir", $CompareDir,
        "--noise-dir", "noise_0p2pct",
        "--target-gamma-within-x1p2", 0.95,
        "--target-catastrophic-ge5", 0.0,
        "--target-median-f-rel-l2", 0.05
    )
    if (Test-Path $Exp27Scan) {
        $AnalysisArgs += @("--exp27-scan-summary", $Exp27Scan)
    }
    & $Python @AnalysisArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Exp28 aggregate analysis failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "[3/4] Quick mode skips physics curve analysis."
    Write-Host "[4/4] Quick mode skips aggregate stage-gate analysis."
}

Write-Host ("=" * 100)
Write-Host "Exp28 complete."
Write-Host "Read first:"
Write-Host "  $DataDir\coverage_summary.csv"
Write-Host "  $DataDir\holdout_to_train_coverage.csv"
Write-Host "  $ResultDir\exp28_summary.csv"
Write-Host "  $CompareDir\exp28_comparison_summary.csv"
Write-Host "  $CompareDir\exp28_stage_gate.csv"
Write-Host "  $CompareDir\exp28_top20_worst_gamma_cases.csv"
Write-Host "  $PhysicsAnalysisDir\representative_curves"
Write-Host ("=" * 100)
