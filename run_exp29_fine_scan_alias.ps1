param(
    [string]$Device = "cuda",
    [string]$Thresholds = "1.0,1.1,1.2,1.3,1.4,1.5",
    [string]$TrainSeeds = "20260815",
    [int]$ProbeCount = 12000,
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
    ".\analyze_exp29_fine_scan_alias.py",
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
    if ($text.Length -gt 0) { $ThresholdList += [double]$text }
}
$ThresholdList = @($ThresholdList | Sort-Object -Unique)
if ($ThresholdList.Count -lt 2) {
    throw "Please provide at least two thresholds."
}

$SeedList = @()
foreach ($piece in $TrainSeeds.Split(",")) {
    $text = $piece.Trim()
    if ($text.Length -gt 0) { $SeedList += [int]$text }
}
$SeedList = @($SeedList | Sort-Object -Unique)
if ($SeedList.Count -lt 1) { throw "No valid training seeds." }

if ($Quick) {
    $CandidateCount = 1200
    $MaxStates = 100
    $IntegrationPoints = 32
    $TargetTrain = 2600
    $TargetVal = 800
    $TargetTest = 1200
    $Epochs = 3
    $ProbeCount = [math]::Min($ProbeCount, 500)
    $CompareDir = ".\validation_results\exp29_fine_scan_alias_quick"
}
else {
    $CandidateCount = 6000
    $MaxStates = 260
    $IntegrationPoints = 128
    $TargetTrain = 20000
    $TargetVal = 3000
    $TargetTest = 5000
    $Epochs = 80
    $CompareDir = ".\validation_results\exp29_fine_scan_alias"
}
New-Item -ItemType Directory -Force -Path $CompareDir | Out-Null

Write-Host ("=" * 100)
Write-Host "Exp29: fine separation scan + continuous off-bank alias audit"
Write-Host "Thresholds  : $($ThresholdList -join ', ')"
Write-Host "Train seeds : $($SeedList -join ', ')"
Write-Host ""
Write-Host "This experiment answers:"
Write-Host "  1) Is the useful separation threshold actually somewhere between 1.0 and 1.5?"
Write-Host "  2) Do physically remote continuous parameter points still alias a selected truth in g-space?"
Write-Host ""
Write-Host "Physics/network/loss remain unchanged."
Write-Host ("=" * 100)

$ManifestRows = @()

foreach ($T in $ThresholdList) {
    $Tag = ("{0:g}" -f $T).Replace(".", "p")

    # Reuse completed formal data when possible. Training result reuse is seed-specific.
    if ((-not $Quick) -and (-not $Regenerate) -and ([math]::Abs($T - 1.0) -lt 1e-12) -and
        (Test-Path ".\data_exp26_gsep_snr1\metadata.json")) {
        $DataDir = ".\data_exp26_gsep_snr1"
        $DataPrefix = "exp26"
    }
    elseif ((-not $Quick) -and (-not $Regenerate) -and ([math]::Abs($T - 1.5) -lt 1e-12) -and
            (Test-Path ".\data_exp27_gsep_snr1p5\metadata.json")) {
        $DataDir = ".\data_exp27_gsep_snr1p5"
        $DataPrefix = "exp27"
    }
    else {
        $DataDir = ".\data_exp29_gsep_snr$Tag"
        if ($Quick) { $DataDir = "${DataDir}_quick" }
        $DataPrefix = "exp29"

        $Meta = Join-Path $DataDir "metadata.json"
        if ($Regenerate -or -not (Test-Path $Meta)) {
            Write-Host ""
            Write-Host "[data] threshold=$T"
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
                throw "Exp29 data generation failed at threshold $T"
            }
        }
    }

    foreach ($Seed in $SeedList) {
        $CanReuseResult = $false
        if ((-not $Quick) -and (-not $Retrain) -and $Seed -eq 20260815 -and
            ([math]::Abs($T - 1.0) -lt 1e-12) -and
            (Test-Path ".\validation_results\exp26_gsep_snr1\exp26_summary.csv") -and
            (Test-Path ".\validation_results\exp26_gsep_snr1\exp26_samples.csv")) {
            $ResultDir = ".\validation_results\exp26_gsep_snr1"
            $ResultPrefix = "exp26"
            $CanReuseResult = $true
        }
        elseif ((-not $Quick) -and (-not $Retrain) -and $Seed -eq 20260815 -and
                ([math]::Abs($T - 1.5) -lt 1e-12) -and
                (Test-Path ".\validation_results\exp27_gsep_snr1p5\exp27_summary.csv") -and
                (Test-Path ".\validation_results\exp27_gsep_snr1p5\exp27_samples.csv")) {
            $ResultDir = ".\validation_results\exp27_gsep_snr1p5"
            $ResultPrefix = "exp27"
            $CanReuseResult = $true
        }
        else {
            $ResultDir = ".\validation_results\exp29_gsep_snr${Tag}_seed$Seed"
            if ($Quick) { $ResultDir = "${ResultDir}_quick" }
            $ResultPrefix = "exp29"
        }

        if (-not $CanReuseResult) {
            $Summary = Join-Path $ResultDir "${ResultPrefix}_summary.csv"
            if ($Retrain -or -not (Test-Path $Summary)) {
                Write-Host "[train] threshold=$T seed=$Seed"
                $TrainArgs = @(
                    "train_exp26_continuous.py",
                    "--data-dir", $DataDir,
                    "--output-dir", $ResultDir,
                    "--result-prefix", $ResultPrefix,
                    "--device", $Device,
                    "--seed", $Seed,
                    "--epochs", $Epochs,
                    "--batch-size", 256,
                    "--eval-batch-size", 512,
                    "--patience", 15
                )
                if ($Quick) { $TrainArgs += "--skip-physics-metrics" }
                & $Python @TrainArgs
                if ($LASTEXITCODE -ne 0) {
                    throw "Exp29 training failed at threshold $T seed $Seed"
                }
            }
        }
        else {
            Write-Host "[reuse] threshold=$T seed=$Seed -> existing result"
        }

        $ManifestRows += [PSCustomObject]@{
            threshold = $T
            seed = $Seed
            data_dir = $DataDir
            result_dir = $ResultDir
            result_prefix = $ResultPrefix
        }
    }
}

$ManifestPath = Join-Path $CompareDir "exp29_manifest.csv"
$ManifestRows | Export-Csv -Path $ManifestPath -NoTypeInformation -Encoding UTF8

if (-not $Quick) {
    Write-Host ""
    Write-Host "[analysis] fine scan + dense continuous alias audit"
    & $Python "analyze_exp29_fine_scan_alias.py" `
        --manifest $ManifestPath `
        --output-dir $CompareDir `
        --noise-dir "noise_0p2pct" `
        --probe-count $ProbeCount `
        --probe-seed 20260829 `
        --remote-a1-abs 0.03 `
        --remote-gamma-factor 2.0
    if ($LASTEXITCODE -ne 0) {
        throw "Exp29 aggregate analysis failed"
    }
}
else {
    Write-Host "Quick mode skips dense continuous alias audit."
}

Write-Host ("=" * 100)
Write-Host "Exp29 complete."
Write-Host "Read first:"
Write-Host "  $CompareDir\exp29_fine_scan_alias_summary.csv"
Write-Host "  $CompareDir\exp29_continuous_alias_summary.csv"
Write-Host "  $CompareDir\exp29_catastrophic_prediction_cases.csv"
Write-Host "  $CompareDir\gamma_within_x1p2_fine_scan.png"
Write-Host "  $CompareDir\continuous_remote_alias_rate.png"
Write-Host ("=" * 100)
