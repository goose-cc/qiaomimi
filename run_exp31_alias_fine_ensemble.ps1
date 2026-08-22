param(
    [string]$Device = "cuda",
    [string]$AliasThresholds = "0.15,0.20,0.25,0.30,0.35,0.40",
    [string]$TrainSeeds = "20260830,20260831,20260832",
    [double]$MinSeparationRMSSNR = 1.3,
    [double]$MaxCoverRMSSNR = 2.0,
    [switch]$Quick,
    [switch]$RegenerateData,
    [switch]$Retrain
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Required = @(
    ".\build_exp30_alias_pool.py",
    ".\data_generate_exp30_identifiable_region.py",
    ".\train_exp26_continuous.py",
    ".\analyze_exp31_alias_fine_ensemble.py",
    ".\PairwiseInverseMLP.py"
)
foreach ($f in $Required) {
    if (-not (Test-Path $f)) { throw "Required file not found: $f" }
}

$PoolDir = ".\data_exp30_alias_pool"
if ($Quick) { $PoolDir = ".\data_exp30_alias_pool_quick" }
if (-not (Test-Path (Join-Path $PoolDir "metadata.json"))) {
    throw "Exp30 alias pool not found: $PoolDir. Run Exp30 pool generation first."
}

$AliasList = @()
foreach ($piece in $AliasThresholds.Split(",")) {
    $text = $piece.Trim()
    if ($text.Length -gt 0) { $AliasList += [double]$text }
}
$AliasList = @($AliasList | Sort-Object -Unique)

$SeedList = @()
foreach ($piece in $TrainSeeds.Split(",")) {
    $text = $piece.Trim()
    if ($text.Length -gt 0) { $SeedList += [int]$text }
}
$SeedList = @($SeedList | Sort-Object -Unique)

if ($AliasList.Count -lt 2) { throw "Need at least two alias thresholds." }
if ($SeedList.Count -lt 1) { throw "Need at least one training seed." }

if ($Quick) {
    $MaxStates = 90
    $MinRequiredStates = 18
    $TargetTrain = 2600
    $TargetVal = 800
    $TargetTest = 1200
    $Epochs = 3
    $CompareDir = ".\validation_results\exp31_alias_fine_ensemble_quick"
}
else {
    $MaxStates = 260
    $MinRequiredStates = 30
    $TargetTrain = 20000
    $TargetVal = 3000
    $TargetTest = 5000
    $Epochs = 80
    $CompareDir = ".\validation_results\exp31_alias_fine_ensemble"
}
New-Item -ItemType Directory -Force -Path $CompareDir | Out-Null

Write-Host ("=" * 100)
Write-Host "Exp31: fine alias scan + 3-seed stability + median ensemble"
Write-Host "Alias thresholds        : $($AliasList -join ', ')"
Write-Host "Training seeds          : $($SeedList -join ', ')"
Write-Host "Selected separation     : $MinSeparationRMSSNR"
Write-Host "Holdout coverage max    : $MaxCoverRMSSNR"
Write-Host ""
Write-Host "Physics / q2 observations / MLP / loss remain unchanged."
Write-Host "The new variables are only alias cutoff and training random seed."
Write-Host ("=" * 100)

$ManifestRows = @()

foreach ($A in $AliasList) {
    $ATag = ("{0:g}" -f $A).Replace(".", "p")
    $STag = ("{0:g}" -f $MinSeparationRMSSNR).Replace(".", "p")
    $CTag = ("{0:g}" -f $MaxCoverRMSSNR).Replace(".", "p")

    $DataDir = ".\data_exp31_alias${ATag}_sep${STag}_cover${CTag}"
    if ($Quick) { $DataDir = "${DataDir}_quick" }

    $Meta = Join-Path $DataDir "metadata.json"
    if ($RegenerateData -or -not (Test-Path $Meta)) {
        Write-Host ""
        Write-Host "[data] alias_min=$A"
        & $Python "data_generate_exp30_identifiable_region.py" `
            --pool-dir $PoolDir `
            --output-dir $DataDir `
            --alias-min-rms-snr $A `
            --min-rms-snr $MinSeparationRMSSNR `
            --max-train-cover-rms-snr $MaxCoverRMSSNR `
            --max-selected-states $MaxStates `
            --min-required-states $MinRequiredStates `
            --noise-levels "0,0.002,0.01" `
            --train-noise-level 0.002 `
            --target-train-samples $TargetTrain `
            --target-val-samples $TargetVal `
            --target-test-samples $TargetTest `
            --compressed `
            --overwrite
        if ($LASTEXITCODE -ne 0) {
            throw "Exp31 data generation failed for alias threshold $A"
        }
    }

    if (-not (Test-Path $Meta)) {
        Write-Host "[skip] alias_min=$A has insufficient identifiable support."
        continue
    }

    foreach ($Seed in $SeedList) {
        $ResultDir = ".\validation_results\exp31_alias${ATag}_seed${Seed}"
        if ($Quick) { $ResultDir = "${ResultDir}_quick" }
        $Summary = Join-Path $ResultDir "exp31_summary.csv"

        if ($Retrain -or -not (Test-Path $Summary)) {
            Write-Host "[train] alias_min=$A seed=$Seed"
            $TrainArgs = @(
                "train_exp26_continuous.py",
                "--data-dir", $DataDir,
                "--output-dir", $ResultDir,
                "--result-prefix", "exp31",
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
                throw "Exp31 training failed for alias=$A seed=$Seed"
            }
        }
        else {
            Write-Host "[reuse] alias_min=$A seed=$Seed"
        }

        $ManifestRows += [PSCustomObject]@{
            alias_min_rms_snr = $A
            seed = $Seed
            data_dir = $DataDir
            result_dir = $ResultDir
            result_prefix = "exp31"
        }
    }
}

if ($ManifestRows.Count -lt 2) {
    throw "Too few successful Exp31 runs."
}

$ManifestPath = Join-Path $CompareDir "exp31_manifest.csv"
$ManifestRows | Export-Csv -Path $ManifestPath -NoTypeInformation -Encoding UTF8

if (-not $Quick) {
    Write-Host ""
    Write-Host "[analysis] multi-seed aggregate + evaluation-only seed ensemble"
    & $Python "analyze_exp31_alias_fine_ensemble.py" `
        --manifest $ManifestPath `
        --output-dir $CompareDir `
        --noise-dir "noise_0p2pct" `
        --target-within-x1p2 0.95
    if ($LASTEXITCODE -ne 0) {
        throw "Exp31 aggregate analysis failed"
    }
}
else {
    Write-Host "Quick mode skips formal aggregate analysis because physics metrics are omitted."
}

Write-Host ("=" * 100)
Write-Host "Exp31 complete."
Write-Host "Read first:"
Write-Host "  $CompareDir\exp31_alias_fine_scan_summary.csv"
Write-Host "  $CompareDir\exp31_threshold_ranking.csv"
Write-Host "  $CompareDir\exp31_stage_gate.csv"
Write-Host "  $CompareDir\exp31_persistent_hard_states.csv"
Write-Host "  $CompareDir\state_accuracy_fine_scan_ensemble.png"
Write-Host ("=" * 100)
