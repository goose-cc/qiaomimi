param(
    [ValidateSet("a1gamma", "mgamma", "exp20c")]
    [string]$Mode = "a1gamma",

    [string]$Device = "cuda",
    [int]$GammaCount = 11,

    [switch]$Quick,
    [switch]$Regenerate
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$TrainMode = $Mode
$A1Values = ""

if ($Mode -eq "a1gamma") {
    $DataDir = ".\data_exp20_a1_gamma"
    $OutputDir = ".\validation_results\exp20a_a1_gamma"
}
elseif ($Mode -eq "mgamma") {
    $DataDir = ".\data_exp20_m_gamma"
    $OutputDir = ".\validation_results\exp20b_m_gamma"
}
else {
    # Exp20C: keep the successful 11 gamma anchors, but densify a1 from
    # 5 anchors to 11 evenly spaced anchors in [0.05, 0.20].
    $TrainMode = "a1gamma"
    $A1Values = "0.05,0.065,0.08,0.095,0.11,0.125,0.14,0.155,0.17,0.185,0.20"
    $DataDir = ".\data_exp20c_a1_gamma_dense"
    $OutputDir = ".\validation_results\exp20c_a1_gamma_dense"
}

if ($Quick) {
    $DataDir = "${DataDir}_quick"
    $OutputDir = "${OutputDir}_quick"
    $TrainPerState = 4
    $ValPerState = 2
    $TestSeenPerState = 2
    $TestInterpPerState = 2
    $Epochs = 3
}
elseif ($Mode -eq "exp20c") {
    # 11 a1 x 11 gamma = 121 physical states.  Keep the total training size
    # close to Exp20A/B (~20k) so the controlled change is anchor density.
    $TrainPerState = 165       # 121 * 165 = 19,965
    $ValPerState = 20          # 2,420
    $TestSeenPerState = 36     # 4,356
    $TestInterpPerState = 20   # ~2,000-2,200 per interpolation split
    $Epochs = 80
}
else {
    $TrainPerState = 360
    $ValPerState = 45
    $TestSeenPerState = 80
    $TestInterpPerState = 50
    $Epochs = 80
}

Write-Host ""
Write-Host ("=" * 92)
Write-Host "Exp20 pairwise curriculum (A/B/C)"
Write-Host "experiment    : $Mode"
Write-Host "training mode : $TrainMode"
Write-Host "device        : $Device"
Write-Host "gamma anchors : $GammaCount"
Write-Host "data          : $DataDir"
Write-Host "output        : $OutputDir"
Write-Host "quick         : $Quick"
Write-Host ("=" * 92)

$Metadata = Join-Path $DataDir "metadata.json"
if ($Regenerate -or -not (Test-Path $Metadata)) {
    Write-Host ""
    Write-Host "[1/2] Generating data ..."
    $GenerateArgs = @(
        "data_generate_exp20_pairwise.py",
        "--mode", $TrainMode,
        "--output-dir", $DataDir,
        "--gamma-count", $GammaCount,
        "--noise-levels", "0,0.002,0.01",
        "--train-noise-level", "0.002",
        "--reference-noise", "0.002",
        "--train-per-state", $TrainPerState,
        "--val-per-state", $ValPerState,
        "--test-seen-per-state", $TestSeenPerState,
        "--test-interp-per-state", $TestInterpPerState,
        "--compressed",
        "--overwrite"
    )

    if ($Mode -eq "exp20c") {
        $GenerateArgs += @("--a1-values", $A1Values)
    }

    & $Python @GenerateArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Exp20 data generation failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "[1/2] Existing data found; generation skipped."
    Write-Host "      Use -Regenerate to rebuild it."
}

Write-Host ""
Write-Host "[2/2] Training ..."

$TrainArgs = @(
    "train_exp20_pairwise.py",
    "--mode", $TrainMode,
    "--data-dir", $DataDir,
    "--output-dir", $OutputDir,
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
    throw "Exp20 training failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Exp20 complete."
Write-Host "Read first:"
Write-Host "  $OutputDir\exp20_summary.csv"
Write-Host "  $OutputDir\exp20_samples.csv"
