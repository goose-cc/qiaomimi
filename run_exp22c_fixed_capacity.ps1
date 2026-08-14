param(
    [string]$Device = "cuda",
    [int[]]$ObservationPoints = @(100, 500, 1000),
    [int]$ModelInputPoints = 1000,
    [switch]$Quick,
    [switch]$Regenerate
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$SourceMetadata = ".\data_exp22a_rms_snr2\metadata.json"
if (-not (Test-Path $SourceMetadata)) {
    throw "Exp22A metadata not found: $SourceMetadata. Run Exp22A first."
}

$RequiredFiles = @(
    ".\data_generate_exp22c_fixed_capacity.py",
    ".\analyze_exp22c_fixed_capacity.py",
    ".\analyze_exp22_classification.py",
    ".\train_exp20_pairwise.py",
    ".\PairwiseInverseMLP.py"
)
foreach ($File in $RequiredFiles) {
    if (-not (Test-Path $File)) {
        throw "Required file not found: $File"
    }
}

if ($Quick) {
    $Epochs = 3
    $TrainPerState = 20
    $ValPerState = 8
    $TestSeenPerState = 10
    $TestInterpPerState = 8
    $IntegrationPoints = 32
}
else {
    $Epochs = 80
    # 0 => reuse Exp22A counts_per_state.
    $TrainPerState = 0
    $ValPerState = 0
    $TestSeenPerState = 0
    $TestInterpPerState = 0
    $IntegrationPoints = 128
}

Write-Host ""
Write-Host ("=" * 100)
Write-Host "Exp22C: FIXED NETWORK CAPACITY control"
Write-Host "Scientific control:"
Write-Host "  same Exp22A 27 physical classes"
Write-Host "  same physical formula / q2 range / train noise / sample counts"
Write-Host "  same MLP input width for EVERY condition: $ModelInputPoints"
Write-Host "  same hidden widths and therefore same trainable parameter count"
Write-Host "  ONLY independent physical q2 observations change"
Write-Host "physical observations : $($ObservationPoints -join ', ')"
Write-Host "device                : $Device"
Write-Host "quick                 : $Quick"
Write-Host ("=" * 100)

foreach ($Nobs in $ObservationPoints) {
    if ($Nobs -lt 2 -or $Nobs -gt $ModelInputPoints) {
        throw "Invalid observation count $Nobs; require 2 <= Nobs <= ModelInputPoints=$ModelInputPoints"
    }

    $DataDir = ".\data_exp22c_qobs${Nobs}_fixed${ModelInputPoints}"
    $OutputDir = ".\validation_results\exp22c_qobs${Nobs}_fixed${ModelInputPoints}"
    $Prefix = "exp22c_qobs$Nobs"

    if ($Quick) {
        $DataDir = "${DataDir}_quick"
        $OutputDir = "${OutputDir}_quick"
    }

    Write-Host ""
    Write-Host ("-" * 100)
    Write-Host "Exp22C Nobs=$Nobs -> fixed model input=$ModelInputPoints"
    Write-Host "data   : $DataDir"
    Write-Host "output : $OutputDir"
    Write-Host ("-" * 100)

    $Metadata = Join-Path $DataDir "metadata.json"
    if ($Regenerate -or -not (Test-Path $Metadata)) {
        $DataArgs = @(
            "data_generate_exp22c_fixed_capacity.py",
            "--source-metadata", $SourceMetadata,
            "--output-dir", $DataDir,
            "--observation-points", $Nobs,
            "--model-input-points", $ModelInputPoints,
            "--reference-noise", 0.002,
            "--noise-levels", "0,0.002,0.01",
            "--train-noise-level", 0.002,
            "--integration-points", $IntegrationPoints,
            "--train-per-state", $TrainPerState,
            "--val-per-state", $ValPerState,
            "--test-seen-per-state", $TestSeenPerState,
            "--test-interp-per-state", $TestInterpPerState,
            "--compressed",
            "--overwrite"
        )

        & $Python @DataArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Exp22C Nobs=$Nobs data generation failed with exit code $LASTEXITCODE"
        }
    }
    else {
        Write-Host "Existing data found; generation skipped. Use -Regenerate to rebuild."
    }

    $TrainArgs = @(
        "train_exp20_pairwise.py",
        "--mode", "a1gamma",
        "--data-dir", $DataDir,
        "--output-dir", $OutputDir,
        "--result-prefix", $Prefix,
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
        throw "Exp22C Nobs=$Nobs training failed with exit code $LASTEXITCODE"
    }

    & $Python "analyze_exp22_classification.py" `
        --data-dir $DataDir `
        --result-dir $OutputDir `
        --result-prefix $Prefix `
        --experiment-label "Exp22C Nobs=$Nobs fixed-input=$ModelInputPoints"

    if ($LASTEXITCODE -ne 0) {
        throw "Exp22C Nobs=$Nobs classification analysis failed with exit code $LASTEXITCODE"
    }
}

if (-not $Quick) {
    $ObsText = ($ObservationPoints | Sort-Object -Unique) -join ","
    Write-Host ""
    Write-Host "[Aggregate] Fixed-capacity comparison for physical observations=$ObsText"

    & $Python "analyze_exp22c_fixed_capacity.py" `
        --project-root "." `
        --observation-points $ObsText `
        --model-input-points $ModelInputPoints `
        --output-dir "validation_results\exp22c_fixed_capacity_comparison"

    if ($LASTEXITCODE -ne 0) {
        throw "Exp22C aggregate comparison failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host ""
    Write-Host "Quick mode skips aggregate comparison because results are written to *_quick folders."
}

Write-Host ""
Write-Host ("=" * 100)
Write-Host "Exp22C complete."
Write-Host "Read first:"
Write-Host "  .\validation_results\exp22c_fixed_capacity_comparison\exp22c_fixed_capacity_comparison.csv"
Write-Host "  .\validation_results\exp22c_fixed_capacity_comparison\classification_accuracy_vs_observations.png"
Write-Host "  .\validation_results\exp22c_fixed_capacity_comparison\gamma_within_x1p2_vs_observations.png"
Write-Host "  .\validation_results\exp22c_fixed_capacity_comparison\f_relative_l2_vs_observations.png"
Write-Host "  .\validation_results\exp22c_fixed_capacity_comparison\model_parameter_count_vs_observations.png"
Write-Host ("=" * 100)
