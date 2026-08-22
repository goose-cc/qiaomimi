param(
    [string]$Device = "cuda",
    [double]$AliasThreshold = 0.40,
    [string]$DataSeeds = "20260840,20260841,20260842",
    [string]$TrainSeeds = "20260850,20260851,20260852",
    [double]$MinSeparationRMSSNR = 1.3,
    [double]$MaxCoverRMSSNR = 2.0,
    [ValidateSet("selected", "original")]
    [string]$ParameterBoundsSource = "selected",

    # Secondary stage gate. Continuous error remains the primary evaluation.
    [double]$A1SuccessTol = 0.005,
    [double]$GammaSuccessFactor = 1.20,
    [double]$TargetRecoveryRate = 0.98,
    [double]$FallbackRecoveryRate = 0.95,

    [switch]$Quick,
    [switch]$RegenerateData,
    [switch]$RetrainMLP,
    [switch]$RetrainTCN
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Required = @(
    ".\data_generate_exp30_identifiable_region.py",
    ".\PairwiseInverseMLP.py",
    ".\PairwiseInverseTCN.py",
    ".\train_exp20_pairwise.py",
    ".\train_exp33_model_compare.py",
    ".\analyze_exp33_mlp_vs_tcn.py"
)
foreach ($f in $Required) {
    if (-not (Test-Path $f)) { throw "Required file not found: $f" }
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

$PoolDir = ".\data_exp30_alias_pool"
if ($Quick) { $PoolDir = ".\data_exp30_alias_pool_quick" }
if (-not (Test-Path (Join-Path $PoolDir "metadata.json"))) {
    throw "Exp30 alias pool not found: $PoolDir. Run Exp30 pool generation first."
}

if ($Quick) {
    $MaxStates = 90
    $MinRequiredStates = 18
    $TargetTrain = 2600
    $TargetVal = 800
    $TargetTest = 1200
    $Epochs = 3
    $CompareDir = ".\validation_results\exp33_mlp_vs_tcn_quick"
}
else {
    $MaxStates = 260
    $MinRequiredStates = 30
    $TargetTrain = 20000
    $TargetVal = 3000
    $TargetTest = 5000
    $Epochs = 80
    $CompareDir = ".\validation_results\exp33_mlp_vs_tcn"
}
New-Item -ItemType Directory -Force -Path $CompareDir | Out-Null

Write-Host ("=" * 100)
Write-Host "Exp33: MLP vs TCN on FIXED identifiable data"
Write-Host "Alias threshold        : $AliasThreshold"
Write-Host "Data seeds             : $($DataSeedList -join ', ')"
Write-Host "Training seeds         : $($TrainSeedList -join ', ')"
Write-Host "Parameter bounds       : $ParameterBoundsSource"
Write-Host "Recovery gate          : |da1| <= $A1SuccessTol AND gamma-factor <= $GammaSuccessFactor"
Write-Host "Stage targets          : $($TargetRecoveryRate*100)% / fallback $($FallbackRecoveryRate*100)%"
Write-Host ""
Write-Host "Controlled comparison:"
Write-Host "  SAME data / split / noise / normalization / loss / optimizer / seeds"
Write-Host "  ONLY backbone changes: MLP vs TCN"
Write-Host ""
Write-Host "Existing Exp32 MLP runs are reused automatically when available."
Write-Host "Use -RetrainMLP only if you explicitly want to rerun the MLP baseline."
Write-Host ("=" * 100)

$ManifestRows = @()

foreach ($DataSeed in $DataSeedList) {
    # Intentionally reuse the exact Exp32 dataset path so this experiment does
    # not quietly change the data while comparing network architectures.
    $DataDir = ".\data_exp32_alias${ATag}_sep${STag}_cover${CTag}_dseed${DataSeed}"
    if ($Quick) { $DataDir = "${DataDir}_quick" }
    $Meta = Join-Path $DataDir "metadata.json"

    if ($RegenerateData -or -not (Test-Path $Meta)) {
        Write-Host ""
        Write-Host "[data] generating Exp32-compatible fixed data, data_seed=$DataSeed"
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
            throw "Exp33 data generation failed for data seed $DataSeed"
        }
    }

    if (-not (Test-Path $Meta)) {
        throw "Dataset metadata missing: $Meta"
    }

    foreach ($TrainSeed in $TrainSeedList) {
        # ------------------------------------------------------------------
        # MLP baseline.
        # Prefer the already-computed Exp32 run: it is the exact frozen MLP
        # baseline we want. The analyzer recomputes the new joint-rate metric
        # directly from exp32_samples.csv, so retraining is unnecessary.
        # ------------------------------------------------------------------
        $Exp32ResultDir = ".\validation_results\exp32_alias${ATag}_d${DataSeed}_t${TrainSeed}"
        if ($Quick) { $Exp32ResultDir = "${Exp32ResultDir}_quick" }
        $Exp32Samples = Join-Path $Exp32ResultDir "exp32_samples.csv"

        if ((-not $RetrainMLP) -and ($ParameterBoundsSource -eq "selected") -and (Test-Path $Exp32Samples)) {
            Write-Host "[reuse MLP] data_seed=$DataSeed train_seed=$TrainSeed"
            $MLPResultDir = $Exp32ResultDir
            $MLPPrefix = "exp32"
            $MLPSource = "exp32_reuse"
        }
        else {
            $MLPResultDir = ".\validation_results\exp33_mlp_alias${ATag}_d${DataSeed}_t${TrainSeed}"
            if ($Quick) { $MLPResultDir = "${MLPResultDir}_quick" }
            $MLPSummary = Join-Path $MLPResultDir "exp33_summary.csv"

            if ($RetrainMLP -or -not (Test-Path $MLPSummary)) {
                Write-Host "[train MLP] data_seed=$DataSeed train_seed=$TrainSeed"
                $TrainArgs = @(
                    "train_exp33_model_compare.py",
                    "--model", "mlp",
                    "--data-dir", $DataDir,
                    "--output-dir", $MLPResultDir,
                    "--result-prefix", "exp33",
                    "--device", $Device,
                    "--seed", $TrainSeed,
                    "--epochs", $Epochs,
                    "--batch-size", 256,
                    "--eval-batch-size", 512,
                    "--patience", 15,
                    "--parameter-bounds-source", $ParameterBoundsSource,
                    "--a1-success-tol", $A1SuccessTol,
                    "--gamma-success-factor", $GammaSuccessFactor
                )
                if ($Quick) { $TrainArgs += "--skip-physics-metrics" }
                & $Python @TrainArgs
                if ($LASTEXITCODE -ne 0) {
                    throw "MLP training failed for data_seed=$DataSeed train_seed=$TrainSeed"
                }
            }
            else {
                Write-Host "[reuse Exp33 MLP] data_seed=$DataSeed train_seed=$TrainSeed"
            }
            $MLPPrefix = "exp33"
            $MLPSource = "exp33_train"
        }

        $ManifestRows += [PSCustomObject]@{
            model = "mlp"
            source = $MLPSource
            alias_min_rms_snr = $AliasThreshold
            data_seed = $DataSeed
            train_seed = $TrainSeed
            data_dir = $DataDir
            result_dir = $MLPResultDir
            result_prefix = $MLPPrefix
            parameter_bounds_source = $ParameterBoundsSource
        }

        # ------------------------------------------------------------------
        # TCN: same everything else; only backbone changes.
        # ------------------------------------------------------------------
        $TCNResultDir = ".\validation_results\exp33_tcn_alias${ATag}_d${DataSeed}_t${TrainSeed}"
        if ($Quick) { $TCNResultDir = "${TCNResultDir}_quick" }
        $TCNSummary = Join-Path $TCNResultDir "exp33_summary.csv"

        if ($RetrainTCN -or -not (Test-Path $TCNSummary)) {
            Write-Host "[train TCN] data_seed=$DataSeed train_seed=$TrainSeed"
            $TrainArgs = @(
                "train_exp33_model_compare.py",
                "--model", "tcn",
                "--data-dir", $DataDir,
                "--output-dir", $TCNResultDir,
                "--result-prefix", "exp33",
                "--device", $Device,
                "--seed", $TrainSeed,
                "--epochs", $Epochs,
                "--batch-size", 256,
                "--eval-batch-size", 512,
                "--patience", 15,
                "--parameter-bounds-source", $ParameterBoundsSource,
                "--a1-success-tol", $A1SuccessTol,
                "--gamma-success-factor", $GammaSuccessFactor,
                "--tcn-channels", 64,
                "--tcn-dilations", "1,2,4,8,16",
                "--tcn-kernel-size", 3,
                "--tcn-head-hidden", 128
            )
            if ($Quick) { $TrainArgs += "--skip-physics-metrics" }
            & $Python @TrainArgs
            if ($LASTEXITCODE -ne 0) {
                throw "TCN training failed for data_seed=$DataSeed train_seed=$TrainSeed"
            }
        }
        else {
            Write-Host "[reuse TCN] data_seed=$DataSeed train_seed=$TrainSeed"
        }

        $ManifestRows += [PSCustomObject]@{
            model = "tcn"
            source = "exp33_train"
            alias_min_rms_snr = $AliasThreshold
            data_seed = $DataSeed
            train_seed = $TrainSeed
            data_dir = $DataDir
            result_dir = $TCNResultDir
            result_prefix = "exp33"
            parameter_bounds_source = $ParameterBoundsSource
        }
    }
}

$ManifestPath = Join-Path $CompareDir "exp33_manifest.csv"
$ManifestRows | Export-Csv -Path $ManifestPath -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "[analysis] common continuous metrics + joint recovery rate"
& $Python "analyze_exp33_mlp_vs_tcn.py" `
    --manifest $ManifestPath `
    --output-dir $CompareDir `
    --noise-dir "noise_0p2pct" `
    --a1-success-tol $A1SuccessTol `
    --gamma-success-factor $GammaSuccessFactor `
    --target-rate $TargetRecoveryRate `
    --fallback-rate $FallbackRecoveryRate

if ($LASTEXITCODE -ne 0) { throw "Exp33 aggregate analysis failed" }

Write-Host ("=" * 100)
Write-Host "Exp33 complete."
Write-Host "Read first:"
Write-Host "  $CompareDir\exp33_model_summary.csv"
Write-Host "  $CompareDir\exp33_per_run_metrics.csv"
Write-Host "  $CompareDir\exp33_matched_mlp_tcn_deltas.csv"
Write-Host "  $CompareDir\exp33_stage_gate.csv"
Write-Host ("=" * 100)
