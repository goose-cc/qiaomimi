param(
    [string]$Device = "cuda",
    [string]$AliasThresholds = "0.4",
    [double]$MinSeparationRMSSNR = 1.3,
    [double]$MaxCoverRMSSNR = 2.0,
    [switch]$Quick,
    [switch]$RegeneratePool,
    [switch]$RegenerateData,
    [switch]$Retrain
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Required = @(
    ".\build_exp30_alias_pool.py",
    ".\data_generate_exp30_identifiable_region.py",
    ".\analyze_exp30_identifiable_region.py",
    ".\train_exp30_tcn.py",
    ".\PairwiseInverseTCN.py",
    ".\PairwiseInverseMLP.py",
    ".\train_exp20_pairwise.py"
)
foreach ($f in $Required) {
    if (-not (Test-Path $f)) { throw "Required file not found: $f" }
}

$AliasList = @()
foreach ($piece in $AliasThresholds.Split(",")) {
    $text = $piece.Trim()
    if ($text.Length -gt 0) { $AliasList += [double]$text }
}
$AliasList = @($AliasList | Sort-Object -Unique)
if ($AliasList.Count -lt 1) {
    throw "Please provide at least one alias threshold."
}

if ($Quick) {
    $PoolDir = ".\data_exp30_alias_pool_quick"
    $CandidateCount = 1500
    $GammaPoints = 401
    $IntegrationPoints = 32
    $MaxStates = 90
    $MinRequiredStates = 18
    $TargetTrain = 2600
    $TargetVal = 800
    $TargetTest = 1200
    $Epochs = 3
    $CompareDir = ".\validation_results\exp30_identifiable_region_tcn_quick"
}
else {
    # IMPORTANT: same pool/data directory names as the MLP experiment.
    # Therefore, without -RegeneratePool/-RegenerateData, TCN uses the exact
    # same Exp30 data and only the inverse-network architecture changes.
    $PoolDir = ".\data_exp30_alias_pool"
    $CandidateCount = 20000
    $GammaPoints = 1601
    $IntegrationPoints = 128
    $MaxStates = 260
    $MinRequiredStates = 30
    $TargetTrain = 20000
    $TargetVal = 3000
    $TargetTest = 5000
    $Epochs = 80
    $CompareDir = ".\validation_results\exp30_identifiable_region_tcn"
}
New-Item -ItemType Directory -Force -Path $CompareDir | Out-Null

Write-Host ("=" * 100)
Write-Host "Exp30-TCN: same identifiable a1+gamma data, TCN inverse network"
Write-Host "Alias thresholds        : $($AliasList -join ', ')"
Write-Host "Selected-bank separation: $MinSeparationRMSSNR"
Write-Host "Holdout coverage max    : $MaxCoverRMSSNR"
Write-Host ""
Write-Host "DATA / forward physics / q2 observations / loss are kept unchanged."
Write-Host "Inverse network changes: MLP -> dilated residual TCN."
Write-Host ("=" * 100)

$PoolMeta = Join-Path $PoolDir "metadata.json"
$PoolNpz = Join-Path $PoolDir "alias_pool.npz"
if ($RegeneratePool -or -not (Test-Path $PoolMeta) -or -not (Test-Path $PoolNpz)) {
    Write-Host "[pool] Build one shared continuous alias-scored candidate pool"
    & $Python "build_exp30_alias_pool.py" `
        --output-dir $PoolDir `
        --candidate-count $CandidateCount `
        --alias-gamma-points $GammaPoints `
        --reference-noise 0.002 `
        --a1-min 0.05 `
        --a1-max 0.20 `
        --gamma-min 0.01 `
        --gamma-max 1.0 `
        --remote-a1-abs 0.03 `
        --remote-gamma-factor 2.0 `
        --observation-points 500 `
        --model-input-points 1000 `
        --integration-points $IntegrationPoints `
        --overwrite
    if ($LASTEXITCODE -ne 0) {
        throw "Exp30 alias-pool generation failed"
    }
}
else {
    Write-Host "[pool] Reusing existing alias-scored pool."
}

$ManifestRows = @()

foreach ($A in $AliasList) {
    $ATag = ("{0:g}" -f $A).Replace(".", "p")
    $STag = ("{0:g}" -f $MinSeparationRMSSNR).Replace(".", "p")
    $CTag = ("{0:g}" -f $MaxCoverRMSSNR).Replace(".", "p")

    # Reuse the MLP data directories intentionally.
    $DataDir = ".\data_exp30_alias${ATag}_sep${STag}_cover${CTag}"
    $ResultDir = ".\validation_results\exp30_tcn_alias${ATag}_sep${STag}_cover${CTag}"
    if ($Quick) {
        $DataDir = "${DataDir}_quick"
        $ResultDir = "${ResultDir}_quick"
    }

    $Meta = Join-Path $DataDir "metadata.json"
    $TrainNpz = Join-Path $DataDir "noise_0p2pct\train.npz"
    if ($RegenerateData -or -not (Test-Path $Meta) -or -not (Test-Path $TrainNpz)) {
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
            throw "Exp30 data generation failed for alias threshold $A"
        }
    }
    else {
        Write-Host "[data] Reusing existing MLP Exp30 data for alias_min=$A"
    }

    if (-not (Test-Path $Meta)) {
        Write-Host "[skip] alias_min=$A left insufficient identifiable support."
        continue
    }

    $Summary = Join-Path $ResultDir "exp30_tcn_summary.csv"
    if ($Retrain -or -not (Test-Path $Summary)) {
        Write-Host "[train-TCN] alias_min=$A"
        $TrainArgs = @(
            "train_exp30_tcn.py",
            "--data-dir", $DataDir,
            "--output-dir", $ResultDir,
            "--result-prefix", "exp30_tcn",
            "--device", $Device,
            "--seed", 20260830,
            "--epochs", $Epochs,
            "--batch-size", 256,
            "--eval-batch-size", 512,
            "--patience", 15,
            "--channels", 64,
            "--dilations", "1,2,4,8,16,32,64,128",
            "--head-hidden", 128
        )
        if ($Quick) { $TrainArgs += "--skip-physics-metrics" }
        & $Python @TrainArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Exp30 TCN training failed for alias threshold $A"
        }
    }
    else {
        Write-Host "[train-TCN] Reusing existing TCN result for alias_min=$A"
    }

    $ManifestRows += [PSCustomObject]@{
        alias_min_rms_snr = $A
        data_dir = $DataDir
        result_dir = $ResultDir
        result_prefix = "exp30_tcn"
    }
}

$ManifestPath = Join-Path $CompareDir "exp30_tcn_manifest.csv"
$ManifestRows | Export-Csv -Path $ManifestPath -NoTypeInformation -Encoding UTF8

if (-not $Quick -and $ManifestRows.Count -ge 2) {
    Write-Host ""
    Write-Host "[analysis] compare TCN alias-exclusion strength"
    & $Python "analyze_exp30_identifiable_region.py" `
        --manifest $ManifestPath `
        --output-dir $CompareDir `
        --noise-dir "noise_0p2pct" `
        --target-within-x1p2 0.95
    if ($LASTEXITCODE -ne 0) {
        throw "Exp30 TCN aggregate analysis failed"
    }
}
elseif (-not $Quick) {
    Write-Host ""
    Write-Host "[analysis] One alias threshold only; aggregate cross-threshold analysis skipped."
    if ($ManifestRows.Count -eq 1) {
        Write-Host "Read the direct TCN summary in:"
        Write-Host "  $($ManifestRows[0].result_dir)\exp30_tcn_summary.csv"
    }
}
else {
    Write-Host "Quick mode skips aggregate physics comparison."
}

Write-Host ("=" * 100)
Write-Host "Exp30-TCN complete."
Write-Host "Read first:"
if ($ManifestRows.Count -eq 1) {
    Write-Host "  $($ManifestRows[0].result_dir)\exp30_tcn_summary.csv"
    Write-Host "  $($ManifestRows[0].result_dir)\exp30_tcn_samples.csv"
}
else {
    Write-Host "  $CompareDir\exp30_identifiable_region_summary.csv"
    Write-Host "  $CompareDir\exp30_stage_gate.csv"
    Write-Host "  $CompareDir\gamma_accuracy_vs_alias_exclusion.png"
}
Write-Host ("=" * 100)
