param(
    [string]$Device = "cuda",
    [string]$Thresholds = "1.0,1.5,2.0",
    [switch]$Quick,
    [switch]$Regenerate,
    [switch]$Retrain
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Required = @(
    ".\data_generate_exp26_gseparated_independent.py",
    ".\train_exp26_continuous.py",
    ".\analyze_exp26_gseparated.py",
    ".\analyze_exp27_gsep_threshold_scan.py",
    ".\data_generate_exp20_pairwise.py",
    ".\data_generate_exp22c_fixed_capacity.py",
    ".\train_exp20_pairwise.py",
    ".\PairwiseInverseMLP.py"
)
foreach ($f in $Required) {
    if (-not (Test-Path $f)) { throw "Required file not found: $f" }
}

$ThresholdList = @()
foreach ($piece in $Thresholds.Split(",")) {
    $text = $piece.Trim()
    if ($text.Length -eq 0) { continue }
    $ThresholdList += [double]$text
}
if ($ThresholdList.Count -lt 1) {
    throw "No valid thresholds were provided."
}
$ThresholdList = @($ThresholdList | Sort-Object -Unique)

if ($Quick) {
    $CandidateCount = 1200
    $MaxStates = 90
    $IntegrationPoints = 32
    $TargetTrain = 2400
    $TargetVal = 600
    $TargetTest = 900
    $Epochs = 3
    $ComparisonDir = ".\validation_results\exp27_gsep_threshold_scan_quick"
}
else {
    $CandidateCount = 6000
    $MaxStates = 260
    $IntegrationPoints = 128
    $TargetTrain = 20000
    $TargetVal = 3000
    $TargetTest = 5000
    $Epochs = 80
    $ComparisonDir = ".\validation_results\exp27_gsep_threshold_scan"
}

New-Item -ItemType Directory -Force -Path $ComparisonDir | Out-Null

Write-Host ("=" * 100)
Write-Host "Exp27: scan the minimum allowed g-space separation"
Write-Host "Physics/network/loss are UNCHANGED from Exp26."
Write-Host "Thresholds          : $($ThresholdList -join ', ')"
Write-Host "Reference noise     : 0.2%"
Write-Host "Physical q2 obs     : 500"
Write-Host "Model input width   : 1000"
Write-Host "Candidate seed/count: fixed across thresholds"
Write-Host ""
Write-Host "Question:"
Write-Host "  As near-degenerate g states are removed more aggressively,"
Write-Host "  do gamma/f errors and catastrophic narrow-peak failures decrease?"
Write-Host ("=" * 100)

$ManifestRows = @()

foreach ($T in $ThresholdList) {
    $Tag = ("{0:g}" -f $T).Replace(".", "p")

    # Reuse the already-completed Exp26 SNR=1 formal result when possible.
    $CanReuseExp26 = (
        (-not $Quick) -and
        (-not $Regenerate) -and
        (-not $Retrain) -and
        ([math]::Abs($T - 1.0) -lt 1e-12) -and
        (Test-Path ".\data_exp26_gsep_snr1\metadata.json") -and
        (Test-Path ".\validation_results\exp26_gsep_snr1\exp26_summary.csv") -and
        (Test-Path ".\validation_results\exp26_gsep_snr1\exp26_samples.csv")
    )

    if ($CanReuseExp26) {
        $DataDir = ".\data_exp26_gsep_snr1"
        $ResultDir = ".\validation_results\exp26_gsep_snr1"
        $AnalysisDir = ".\validation_results\exp26_gsep_snr1_analysis"
        $ResultPrefix = "exp26"
        Write-Host ""
        Write-Host "Threshold $T : reusing completed Exp26 SNR=1 baseline."
    }
    else {
        $DataDir = ".\data_exp27_gsep_snr$Tag"
        $ResultDir = ".\validation_results\exp27_gsep_snr$Tag"
        $AnalysisDir = ".\validation_results\exp27_gsep_snr${Tag}_analysis"
        $ResultPrefix = "exp27"

        if ($Quick) {
            $DataDir = "${DataDir}_quick"
            $ResultDir = "${ResultDir}_quick"
            $AnalysisDir = "${AnalysisDir}_quick"
        }

        Write-Host ""
        Write-Host ("-" * 100)
        Write-Host "Threshold MinRMSSNR = $T"
        Write-Host ("-" * 100)

        $Meta = Join-Path $DataDir "metadata.json"
        if ($Regenerate -or -not (Test-Path $Meta)) {
            Write-Host "[1/3] Generate globally g-separated independent data"
            & $Python "data_generate_exp26_gseparated_independent.py" `
                --output-dir $DataDir `
                --candidate-count $CandidateCount `
                --max-selected-states $MaxStates `
                --min-rms-snr $T `
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
                throw "Exp27 data generation failed at threshold $T with exit code $LASTEXITCODE"
            }
        }
        else {
            Write-Host "[1/3] Existing data found; use -Regenerate to rebuild."
        }

        $SummaryPath = Join-Path $ResultDir "${ResultPrefix}_summary.csv"
        if ($Retrain -or -not (Test-Path $SummaryPath)) {
            Write-Host "[2/3] Train unchanged continuous a1+gamma MLP"
            $TrainArgs = @(
                "train_exp26_continuous.py",
                "--data-dir", $DataDir,
                "--output-dir", $ResultDir,
                "--result-prefix", $ResultPrefix,
                "--device", $Device,
                "--epochs", $Epochs,
                "--batch-size", 256,
                "--eval-batch-size", 512,
                "--patience", 15
            )
            if ($Quick) { $TrainArgs += "--skip-physics-metrics" }
            & $Python @TrainArgs
            if ($LASTEXITCODE -ne 0) {
                throw "Exp27 training failed at threshold $T with exit code $LASTEXITCODE"
            }
        }
        else {
            Write-Host "[2/3] Existing trained result found; use -Retrain to rerun."
        }

        if (-not $Quick) {
            $AnalysisMarker = Join-Path $AnalysisDir "exp26_per_test_state_summary.csv"
            if ($Regenerate -or $Retrain -or -not (Test-Path $AnalysisMarker)) {
                Write-Host "[3/3] Analyze independent test states and representative curves"
                & $Python "analyze_exp26_gseparated.py" `
                    --data-dir $DataDir `
                    --result-dir $ResultDir `
                    --result-prefix $ResultPrefix `
                    --output-dir $AnalysisDir `
                    --noise-dir "noise_0p2pct" `
                    --integration-points 512
                if ($LASTEXITCODE -ne 0) {
                    throw "Exp27 per-threshold analysis failed at threshold $T with exit code $LASTEXITCODE"
                }
            }
            else {
                Write-Host "[3/3] Existing per-threshold analysis found."
            }
        }
        else {
            Write-Host "[3/3] Quick mode skips physics curve analysis."
        }
    }

    $ManifestRows += [PSCustomObject]@{
        threshold = $T
        data_dir = $DataDir
        result_dir = $ResultDir
        analysis_dir = $AnalysisDir
        result_prefix = $ResultPrefix
    }
}

$ManifestPath = Join-Path $ComparisonDir "exp27_run_manifest.csv"
$ManifestRows | Export-Csv -Path $ManifestPath -NoTypeInformation -Encoding UTF8

if (-not $Quick) {
    Write-Host ""
    Write-Host ("=" * 100)
    Write-Host "Aggregate threshold scan"
    & $Python "analyze_exp27_gsep_threshold_scan.py" `
        --manifest $ManifestPath `
        --output-dir $ComparisonDir `
        --noise-dir "noise_0p2pct"
    if ($LASTEXITCODE -ne 0) {
        throw "Exp27 aggregate analysis failed with exit code $LASTEXITCODE"
    }
}
else {
    Write-Host "Quick mode skips final aggregate physics comparison."
}

Write-Host ("=" * 100)
Write-Host "Exp27 complete."
Write-Host "Read first:"
Write-Host "  $ComparisonDir\exp27_threshold_scan_summary.csv"
Write-Host "  $ComparisonDir\exp27_top10_worst_gamma_cases.csv"
Write-Host "  $ComparisonDir\gamma_accuracy_vs_min_gseparation.png"
Write-Host "  $ComparisonDir\f_error_vs_min_gseparation.png"
Write-Host "  $ComparisonDir\catastrophic_error_vs_min_gseparation.png"
Write-Host "  $ComparisonDir\physical_state_count_vs_min_gseparation.png"
Write-Host "  $ComparisonDir\support_vs_accuracy_tradeoff.png"
Write-Host ("=" * 100)
