param(
    [ValidateSet("compare", "curriculum", "single")]
    [string]$Action = "compare",

    [ValidateSet("gamma", "a1gamma", "a1mgamma", "all5_strong", "all5_full")]
    [string]$Stage = "a1gamma",

    [ValidateSet("coverage", "coverage_gdiverse")]
    [string]$Strategy = "coverage_gdiverse",

    [ValidateSet("transformer", "bilstm_transformer")]
    [string]$ModelType = "bilstm_transformer",

    [string]$PoolDir = ".\truth_pool",
    [string]$ValPoolDir = ".\truth_pool_val_10k",
    [string]$Python = "python",
    [string]$Device = "cuda",
    [int]$MaxSteps = 30000,
    [int]$BatchSize = 64,
    [int]$MicroBatchSize = 4,
    [int]$CandidateMultiplier = 4,
    [double]$TargetGGapPercent = 0.4,
    [double]$NoiseLevel = 0.002,
    [switch]$Amp,
    [switch]$Quick,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PoolDir)) {
    throw "Training pool not found: $PoolDir"
}
if (-not (Test-Path $ValPoolDir)) {
    throw "Validation pool not found: $ValPoolDir. Create a separate validation pool first."
}

if ($Quick) {
    $MaxSteps = 20
    $BatchSize = 16
    $MicroBatchSize = 4
    $CandidateMultiplier = 2
}

function Invoke-Exp23Run {
    param(
        [string]$RunStage,
        [string]$RunStrategy,
        [string]$CheckpointDir,
        [string]$InitFrom = ""
    )

    $ArgsList = @(
        "train_exp23_formal_spectrum_pool_gdiverse.py",
        "--pool-dir", $PoolDir,
        "--val-pool-dir", $ValPoolDir,
        "--checkpoint-dir", $CheckpointDir,
        "--stage", $RunStage,
        "--selection-strategy", $RunStrategy,
        "--model-type", $ModelType,
        "--target-g-gap-percent", ("{0:g}" -f $TargetGGapPercent),
        "--candidate-multiplier", ("{0}" -f $CandidateMultiplier),
        "--batch-size", ("{0}" -f $BatchSize),
        "--micro-batch-size", ("{0}" -f $MicroBatchSize),
        "--max-steps", ("{0}" -f $MaxSteps),
        "--noise-level", ("{0:g}" -f $NoiseLevel),
        "--device", $Device,
        "--loss-profile", "pinn",
        "--loss-normalization", "relative",
        "--lambda-grad", "0.1",
        "--lambda-physics", "0.1",
        "--require-complete-pool",
        "--require-complete-val-pool"
    )

    if ($Quick) {
        $ArgsList += @(
            "--output-points", "100",
            "--integration-points", "64",
            "--log-every-steps", "5",
            "--checkpoint-every-steps", "10",
            "--validation-every-steps", "10",
            "--val-samples", "256",
            "--val-batch-size", "32"
        )
    }
    else {
        $ArgsList += @(
            "--output-points", "1000",
            "--integration-points", "128",
            "--log-every-steps", "100",
            "--checkpoint-every-steps", "2000",
            "--validation-every-steps", "2000",
            "--val-samples", "5000",
            "--val-batch-size", "64"
        )
    }

    if ($Amp -and $Device -eq "cuda") {
        $ArgsList += "--amp"
    }

    if ($Resume) {
        $ArgsList += "--resume"
    }
    else {
        $ArgsList += "--fresh"
        if ($InitFrom -ne "") {
            $ArgsList += @("--init-from", $InitFrom)
        }
    }

    Write-Host ""
    Write-Host ("=" * 112)
    Write-Host "Exp23 formal run"
    Write-Host "  stage    : $RunStage"
    Write-Host "  strategy : $RunStrategy"
    Write-Host "  network  : $ModelType"
    Write-Host "  output   : $CheckpointDir"
    if ($InitFrom -ne "" -and -not $Resume) {
        Write-Host "  init     : $InitFrom"
    }
    Write-Host ("=" * 112)

    & $Python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Exp23 failed with exit code $LASTEXITCODE"
    }
}

if ($Action -eq "compare") {
    # Same formal g->f network, same large pool source, same coverage bins,
    # same validation pool. Only the within-coverage g-diversity preference changes.
    Invoke-Exp23Run `
        -RunStage "a1gamma" `
        -RunStrategy "coverage" `
        -CheckpointDir ".\model\exp23_compare\a1gamma_coverage"

    Invoke-Exp23Run `
        -RunStage "a1gamma" `
        -RunStrategy "coverage_gdiverse" `
        -CheckpointDir ".\model\exp23_compare\a1gamma_coverage_gdiverse"

    & $Python "analyze_exp23_pool_curriculum.py" `
        --root ".\model\exp23_compare" `
        --output ".\validation_results\exp23_compare_summary.csv"
    if ($LASTEXITCODE -ne 0) {
        throw "Exp23 comparison analysis failed"
    }

    Write-Host ""
    Write-Host "Read first: .\validation_results\exp23_compare_summary.csv"
    Write-Host "Decision: keep g-diverse only if coverage stays similar AND unseen validation f/resonance metrics improve."
}
elseif ($Action -eq "curriculum") {
    $Stages = @("gamma", "a1gamma", "a1mgamma", "all5_strong", "all5_full")
    $PreviousBest = ""
    foreach ($S in $Stages) {
        $Dir = ".\model\exp23_curriculum\$S"
        Invoke-Exp23Run `
            -RunStage $S `
            -RunStrategy "coverage_gdiverse" `
            -CheckpointDir $Dir `
            -InitFrom $PreviousBest
        $PreviousBest = Join-Path $Dir "best_model.pth"
    }

    & $Python "analyze_exp23_pool_curriculum.py" `
        --root ".\model\exp23_curriculum" `
        --output ".\validation_results\exp23_curriculum_summary.csv"
    if ($LASTEXITCODE -ne 0) {
        throw "Exp23 curriculum analysis failed"
    }

    Write-Host ""
    Write-Host "Read first: .\validation_results\exp23_curriculum_summary.csv"
}
else {
    $Dir = ".\model\exp23_single\${Stage}_${Strategy}"
    Invoke-Exp23Run `
        -RunStage $Stage `
        -RunStrategy $Strategy `
        -CheckpointDir $Dir
}
