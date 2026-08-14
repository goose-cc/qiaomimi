param(
    [string]$Device = "cuda",
    [int[]]$Q2Points = @(500, 1000),
    [switch]$Quick,
    [switch]$Regenerate,
    [switch]$Rerun100
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$SourceData = ".\data_exp22a_rms_snr2"
$SourceMetadata = Join-Path $SourceData "metadata.json"
$SourceResult = ".\validation_results\exp22a_rms_snr2"

if (-not (Test-Path $SourceMetadata)) {
    throw "Exp22A metadata not found: $SourceMetadata. Run Exp22A first."
}

if ($Quick) {
    $Epochs = 3
    $TrainPerState = 45
    $ValPerState = 15
    $TestSeenPerState = 20
    $TestInterpPerState = 15
}
else {
    $Epochs = 80
    # 0 = reuse Exp22A counts_per_state exactly.
    $TrainPerState = 0
    $ValPerState = 0
    $TestSeenPerState = 0
    $TestInterpPerState = 0
}

Write-Host ""
Write-Host ("=" * 100)
Write-Host "Exp22B: SAME 27 classes, increase q2 sampling density"
Write-Host "Physical formula : UNCHANGED"
Write-Host "Parameter grid   : UNCHANGED from Exp22A"
Write-Host "Noise            : UNCHANGED (train 0.2%)"
Write-Host "q2 range         : UNCHANGED"
Write-Host "new q2 points    : $($Q2Points -join ', ')"
Write-Host "device           : $Device"
Write-Host "quick            : $Quick"
Write-Host ("=" * 100)

if ($Rerun100) {
    Write-Host ""
    Write-Host "Rerun100 was requested."
    Write-Host "Use Exp22A's original runner if you want a fresh Nq=100 baseline:"
    Write-Host "  .\run_exp22a_rms_gseparated_a1gamma.ps1 -Device $Device -Regenerate"
    Write-Host "Exp22B itself keeps Nq=100 as the existing Exp22A baseline."
}

foreach ($Q in $Q2Points) {
    if ($Q -eq 100) {
        Write-Host ""
        Write-Host "Skipping Q=100 here; Exp22A is the controlled 100-point baseline."
        continue
    }
    if ($Q -lt 2) {
        throw "Invalid q2 point count: $Q"
    }

    $DataDir = ".\data_exp22b_q$Q"
    $OutputDir = ".\validation_results\exp22b_q$Q"
    $Prefix = "exp22b_q$Q"
    if ($Quick) {
        $DataDir = "${DataDir}_quick"
        $OutputDir = "${OutputDir}_quick"
    }

    Write-Host ""
    Write-Host ("-" * 100)
    Write-Host "Exp22B Nq=$Q"
    Write-Host "data   : $DataDir"
    Write-Host "output : $OutputDir"
    Write-Host ("-" * 100)

    $Metadata = Join-Path $DataDir "metadata.json"
    if ($Regenerate -or -not (Test-Path $Metadata)) {
        $DataArgs = @(
            "data_generate_exp22b_qdensity.py",
            "--source-metadata", $SourceMetadata,
            "--output-dir", $DataDir,
            "--q2-points", $Q,
            "--reference-noise", 0.002,
            "--noise-levels", "0,0.002,0.01",
            "--train-noise-level", 0.002,
            "--integration-points", 128,
            "--train-per-state", $TrainPerState,
            "--val-per-state", $ValPerState,
            "--test-seen-per-state", $TestSeenPerState,
            "--test-interp-per-state", $TestInterpPerState,
            "--compressed",
            "--overwrite"
        )
        & $Python @DataArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Exp22B Nq=$Q data generation failed with exit code $LASTEXITCODE"
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
        throw "Exp22B Nq=$Q training failed with exit code $LASTEXITCODE"
    }

    & $Python "analyze_exp22_classification.py" `
        --data-dir $DataDir `
        --result-dir $OutputDir `
        --result-prefix $Prefix `
        --experiment-label "Exp22B Nq=$Q"

    if ($LASTEXITCODE -ne 0) {
        throw "Exp22B Nq=$Q classification analysis failed with exit code $LASTEXITCODE"
    }
}

if (-not $Quick) {
    $ComparePoints = (@(100) + @($Q2Points | Where-Object { $_ -ne 100 })) | Sort-Object -Unique
    $CompareText = ($ComparePoints -join ",")
    Write-Host ""
    Write-Host "[Aggregate] Comparing Nq=$CompareText ..."
    & $Python "analyze_exp22b_qdensity.py" `
        --project-root "." `
        --q2-points $CompareText `
        --output-dir "validation_results\exp22b_qdensity_comparison"

    if ($LASTEXITCODE -ne 0) {
        throw "Exp22B aggregate comparison failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host ""
    Write-Host "Quick mode skips the final 100/500/1000 aggregate because it writes *_quick results."
}

Write-Host ""
Write-Host ("=" * 100)
Write-Host "Exp22B complete."
Write-Host "Read first:"
Write-Host "  .\validation_results\exp22b_qdensity_comparison\exp22b_qdensity_comparison.csv"
Write-Host "  .\validation_results\exp22b_qdensity_comparison\classification_accuracy_vs_q2.png"
Write-Host "  .\validation_results\exp22b_qdensity_comparison\gamma_within_x1p2_vs_q2.png"
Write-Host "  .\validation_results\exp22b_qdensity_comparison\separation_snr_vs_q2.png"
Write-Host "  .\validation_results\exp22b_qdensity_comparison\f_relative_l2_vs_q2.png"
Write-Host ("=" * 100)