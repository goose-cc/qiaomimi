param(
    [string]$Device = "cuda",
    [double]$AliasThreshold = 0.40,
    [string]$DataSeeds = "20260840,20260841,20260842",
    [string]$TrainSeeds = "20260850,20260851,20260852",
    [double]$MinSeparationRMSSNR = 1.3,
    [double]$MaxCoverRMSSNR = 2.0,
    [ValidateSet("selected", "original")]
    [string]$ParameterBoundsSource = "selected",
    [switch]$Quick,
    [switch]$RegenerateData,
    [switch]$Retrain,
    [switch]$SkipVarPro
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Required = @(
    ".\data_generate_exp30_identifiable_region.py",
    ".\train_exp32_joint_continuous.py",
    ".\analyze_exp32_joint_continuous.py",
    ".\analyze_exp32_varpro_ceiling.py",
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

function Parse-IntList([string]$Text) {
    $Vals = @()
    foreach ($piece in $Text.Split(",")) {
        $s = $piece.Trim()
        if ($s.Length -gt 0) { $Vals += [int]$s }
    }
    return @($Vals | Sort-Object -Unique)
}

$DataSeedList = Parse-IntList $DataSeeds
$TrainSeedList = Parse-IntList $TrainSeeds
if ($DataSeedList.Count -lt 1) { throw "Need at least one data seed." }
if ($TrainSeedList.Count -lt 1) { throw "Need at least one train seed." }

$ATag = ("{0:g}" -f $AliasThreshold).Replace(".", "p")
$STag = ("{0:g}" -f $MinSeparationRMSSNR).Replace(".", "p")
$CTag = ("{0:g}" -f $MaxCoverRMSSNR).Replace(".", "p")

if ($Quick) {
    $MaxStates = 90
    $MinRequiredStates = 18
    $TargetTrain = 2600
    $TargetVal = 800
    $TargetTest = 1200
    $Epochs = 3
    $CompareDir = ".\validation_results\exp32_joint_continuous_quick"
}
else {
    $MaxStates = 260
    $MinRequiredStates = 30
    $TargetTrain = 20000
    $TargetVal = 3000
    $TargetTest = 5000
    $Epochs = 80
    $CompareDir = ".\validation_results\exp32_joint_continuous"
}
New-Item -ItemType Directory -Force -Path $CompareDir | Out-Null

Write-Host ("=" * 100)
Write-Host "Exp32: fixed alias=0.40 + data-seed x train-seed continuous-error validation"
Write-Host "Alias threshold        : $AliasThreshold"
Write-Host "Data seeds             : $($DataSeedList -join ', ')"
Write-Host "Training seeds         : $($TrainSeedList -join ', ')"
Write-Host "Parameter bounds       : $ParameterBoundsSource"
Write-Host ""
Write-Host "Unchanged: physical formula, g construction, MLP backbone, and training loss."
Write-Host "Changed: evaluation metrics and independent data/noise/split seeds."
Write-Host "Primary gamma metric is continuous log/factor error; no x1.2 gate."
Write-Host ("=" * 100)

$ManifestRows = @()
$DataDirs = @{}

foreach ($DataSeed in $DataSeedList) {
    $DataDir = ".\data_exp32_alias${ATag}_sep${STag}_cover${CTag}_dseed${DataSeed}"
    if ($Quick) { $DataDir = "${DataDir}_quick" }
    $Meta = Join-Path $DataDir "metadata.json"

    if ($RegenerateData -or -not (Test-Path $Meta)) {
        Write-Host ""
        Write-Host "[data] alias=$AliasThreshold data_seed=$DataSeed"
        & $Python "data_generate_exp30_identifiable_region.py" `
            --pool-dir $PoolDir `
            --output-dir $DataDir `
            --alias-min-rms-snr $AliasThreshold `
            --min-rms-snr $MinSeparationRMSSNR `
            --max-train-cover-rms-snr $MaxCoverRMSSNR `
            --max-selected-states $MaxStates `
            --min-required-states $MinRequiredStates `
            --noise-levels "0,0.002,0.01" `
            --train-noise-level 0.002 `
            --target-train-samples $TargetTrain `
            --target-val-samples $TargetVal `
            --target-test-samples $TargetTest `
            --seed $DataSeed `
            --compressed `
            --overwrite
        if ($LASTEXITCODE -ne 0) {
            throw "Exp32 data generation failed for data seed $DataSeed"
        }
    }

    if (-not (Test-Path $Meta)) {
        throw "Exp32 data metadata missing after generation: $Meta"
    }
    $DataDirs[$DataSeed] = $DataDir

    foreach ($TrainSeed in $TrainSeedList) {
        $ResultDir = ".\validation_results\exp32_alias${ATag}_d${DataSeed}_t${TrainSeed}"
        if ($Quick) { $ResultDir = "${ResultDir}_quick" }
        $Summary = Join-Path $ResultDir "exp32_summary.csv"

        if ($Retrain -or -not (Test-Path $Summary)) {
            Write-Host "[train] data_seed=$DataSeed train_seed=$TrainSeed"
            $TrainArgs = @(
                "train_exp32_joint_continuous.py",
                "--data-dir", $DataDir,
                "--output-dir", $ResultDir,
                "--result-prefix", "exp32",
                "--device", $Device,
                "--seed", $TrainSeed,
                "--epochs", $Epochs,
                "--batch-size", 256,
                "--eval-batch-size", 512,
                "--patience", 15,
                "--parameter-bounds-source", $ParameterBoundsSource
            )
            if ($Quick) { $TrainArgs += "--skip-physics-metrics" }
            & $Python @TrainArgs
            if ($LASTEXITCODE -ne 0) {
                throw "Exp32 training failed for data_seed=$DataSeed train_seed=$TrainSeed"
            }
        }
        else {
            Write-Host "[reuse] data_seed=$DataSeed train_seed=$TrainSeed"
        }

        $ManifestRows += [PSCustomObject]@{
            alias_min_rms_snr = $AliasThreshold
            data_seed = $DataSeed
            train_seed = $TrainSeed
            data_dir = $DataDir
            result_dir = $ResultDir
            result_prefix = "exp32"
            parameter_bounds_source = $ParameterBoundsSource
        }
    }
}

$ManifestPath = Join-Path $CompareDir "exp32_manifest.csv"
$ManifestRows | Export-Csv -Path $ManifestPath -NoTypeInformation -Encoding UTF8

if (-not $Quick) {
    Write-Host ""
    Write-Host "[analysis] continuous a1/gamma error stability"
    & $Python "analyze_exp32_joint_continuous.py" `
        --manifest $ManifestPath `
        --output-dir $CompareDir `
        --noise-dir "noise_0p2pct"
    if ($LASTEXITCODE -ne 0) { throw "Exp32 aggregate analysis failed" }

    if (-not $SkipVarPro) {
        # One physics ceiling per data seed.  It is deterministic for a fixed test set,
        # so repeating it for every training seed would be redundant.
        foreach ($DataSeed in $DataSeedList) {
            $DataDir = $DataDirs[$DataSeed]
            $VarDir = Join-Path $CompareDir ("varpro_dseed" + $DataSeed)
            Write-Host "[varpro] data_seed=$DataSeed"
            & $Python "analyze_exp32_varpro_ceiling.py" `
                --data-dir $DataDir `
                --output-dir $VarDir `
                --noise-dir "noise_0p2pct" `
                --device $Device `
                --gamma-grid-points 1201
            if ($LASTEXITCODE -ne 0) {
                throw "Exp32 variable-projection baseline failed for data seed $DataSeed"
            }
        }
    }
}
else {
    Write-Host "Quick mode skips formal aggregate plots/physics ceiling."
}

Write-Host ("=" * 100)
Write-Host "Exp32 complete."
Write-Host "Read first:"
Write-Host "  $CompareDir\exp32_overall_continuous_summary.csv"
Write-Host "  $CompareDir\exp32_per_run_metrics.csv"
Write-Host "  $CompareDir\exp32_by_data_seed.csv"
Write-Host "  $CompareDir\exp32_error_coupling_summary.csv"
Write-Host "  $CompareDir\exp32_state_difficulty.csv"
Write-Host ""
Write-Host "Compare each varpro_dseed*/exp32_varpro_summary.csv against the MLP metrics."
Write-Host ("=" * 100)
