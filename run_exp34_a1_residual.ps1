param(
    [string]$Device = "cuda",
    [double]$AliasThreshold = 0.40,
    [double]$MinSeparationRMSSNR = 1.3,
    [double]$MaxCoverRMSSNR = 2.0,
    [ValidateSet("selected", "original")]
    [string]$ParameterBoundsSource = "selected",
    [string]$DataSeeds = "20260840,20260841,20260842",
    [string]$TrainSeeds = "20260850,20260851,20260852",

    [double]$A1SuccessTol = 0.005,
    [double]$GammaSuccessFactor = 1.20,
    [double]$TargetRecoveryRate = 0.98,
    [double]$FallbackRecoveryRate = 0.95,

    [switch]$Quick,
    [switch]$RetrainResidual
)

$ErrorActionPreference = "Stop"

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Required = @(
    ".\PairwiseInverseTCN.py",
    ".\PairwiseInverseTCNA1Residual.py",
    ".\train_exp20_pairwise.py",
    ".\train_exp33_model_compare.py",
    ".\train_exp34_a1_residual.py",
    ".\analyze_exp34_a1_residual.py"
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

if ($Quick) {
    $Epochs = 3
    $CompareDir = ".\validation_results\exp34_a1_residual_compare_quick"
}
else {
    $Epochs = 40
    $CompareDir = ".\validation_results\exp34_a1_residual_compare"
}
New-Item -ItemType Directory -Force -Path $CompareDir | Out-Null

Write-Host ("=" * 100)
Write-Host "Exp34: improve a1 while freezing the successful Exp33 TCN gamma path"
Write-Host "Alias threshold : $AliasThreshold"
Write-Host "Min separation  : $MinSeparationRMSSNR"
Write-Host "Max train cover : $MaxCoverRMSSNR"
Write-Host "Bounds source   : $ParameterBoundsSource"
Write-Host "Data seeds      : $($DataSeedList -join ', ')"
Write-Host "Train seeds     : $($TrainSeedList -join ', ')"
Write-Host "Recovery gate   : |da1| <= $A1SuccessTol AND gamma-factor <= $GammaSuccessFactor"
Write-Host ""
Write-Host "Controlled change:"
Write-Host "  Exp33 TCN is loaded and frozen."
Write-Host "  Only a new amplitude-aware residual branch for a1 is trained."
Write-Host "  Gamma prediction should remain sample-by-sample identical to Exp33."
Write-Host ("=" * 100)

$ManifestRows = @()

foreach ($DataSeed in $DataSeedList) {
    # Exactly the same fixed identifiable dataset used by Exp32/33.
    $DataDir = ".\data_exp32_alias${ATag}_sep${STag}_cover${CTag}_dseed${DataSeed}"
    if ($Quick) { $DataDir = "${DataDir}_quick" }

    if (-not (Test-Path (Join-Path $DataDir "metadata.json"))) {
        throw "Fixed Exp32/33 dataset missing: $DataDir"
    }

    foreach ($TrainSeed in $TrainSeedList) {
        $BaseDir = ".\validation_results\exp33_tcn_alias${ATag}_d${DataSeed}_t${TrainSeed}"
        if ($Quick) { $BaseDir = "${BaseDir}_quick" }

        $BaseCheckpoint = Join-Path $BaseDir "best_exp33_tcn.pt"
        $BaseSamples = Join-Path $BaseDir "exp33_samples.csv"
        if (-not (Test-Path $BaseCheckpoint)) {
            throw "Exp33 TCN checkpoint missing: $BaseCheckpoint"
        }
        if (-not (Test-Path $BaseSamples)) {
            throw "Exp33 TCN samples missing: $BaseSamples"
        }

        $ManifestRows += [PSCustomObject]@{
            model = "tcn"
            source = "exp33_frozen_baseline"
            data_seed = $DataSeed
            train_seed = $TrainSeed
            result_dir = $BaseDir
            result_prefix = "exp33"
        }

        $ResultDir = ".\validation_results\exp34_a1res_alias${ATag}_d${DataSeed}_t${TrainSeed}"
        if ($Quick) { $ResultDir = "${ResultDir}_quick" }
        $Summary = Join-Path $ResultDir "exp34_summary.csv"

        if ($RetrainResidual -or -not (Test-Path $Summary)) {
            Write-Host ""
            Write-Host "[train a1 residual] data_seed=$DataSeed train_seed=$TrainSeed"
            $Args = @(
                "train_exp34_a1_residual.py",
                "--data-dir", $DataDir,
                "--base-checkpoint", $BaseCheckpoint,
                "--output-dir", $ResultDir,
                "--result-prefix", "exp34",
                "--device", $Device,
                "--seed", $TrainSeed,
                "--epochs", $Epochs,
                "--batch-size", 256,
                "--eval-batch-size", 512,
                "--patience", 10,
                "--a1-success-tol", $A1SuccessTol,
                "--gamma-success-factor", $GammaSuccessFactor,
                "--parameter-bounds-source", $ParameterBoundsSource,
                "--amplitude-bins", 16,
                "--amplitude-hidden", "96,64",
                "--correction-limit", 2.0
            )
            if ($Quick) { $Args += "--skip-physics-metrics" }

            & $Python @Args
            if ($LASTEXITCODE -ne 0) {
                throw "Exp34 residual training failed for D=$DataSeed T=$TrainSeed"
            }
        }
        else {
            Write-Host "[reuse a1 residual] data_seed=$DataSeed train_seed=$TrainSeed"
        }

        $ManifestRows += [PSCustomObject]@{
            model = "a1res"
            source = "exp34_train"
            data_seed = $DataSeed
            train_seed = $TrainSeed
            result_dir = $ResultDir
            result_prefix = "exp34"
        }
    }
}

$ManifestPath = Join-Path $CompareDir "exp34_manifest.csv"
$ManifestRows | Export-Csv -Path $ManifestPath -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "[analysis] Exp33 TCN vs Exp34 a1 residual"
& $Python "analyze_exp34_a1_residual.py" `
    --manifest $ManifestPath `
    --output-dir $CompareDir `
    --noise-dir "noise_0p2pct" `
    --a1-success-tol $A1SuccessTol `
    --gamma-success-factor $GammaSuccessFactor `
    --target-rate $TargetRecoveryRate `
    --fallback-rate $FallbackRecoveryRate

if ($LASTEXITCODE -ne 0) { throw "Exp34 analysis failed" }

Write-Host ("=" * 100)
Write-Host "Exp34 complete."
Write-Host "Read first:"
Write-Host "  $CompareDir\exp34_model_summary.csv"
Write-Host "  $CompareDir\exp34_matched_tcn_a1res_deltas.csv"
Write-Host "  $CompareDir\exp34_stage_gate.csv"
Write-Host "  $CompareDir\exp34_gamma_invariance_check.csv"
Write-Host ("=" * 100)
