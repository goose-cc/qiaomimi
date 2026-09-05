param(
    [string]$Device = "cuda",
    [double]$AliasThreshold = 0.40,
    [double]$MinSeparationRMSSNR = 1.30,
    [double]$MaxCoverRMSSNR = 2.0,

    [double]$A1SuccessTol = 0.005,
    [double]$MSuccessTol = 0.03,
    [double]$GammaSuccessFactor = 1.20,

    [string]$Models = "mlp,tcn,pspec",
    [string]$DataSeeds = "20260870,20260871,20260872",
    [string]$TrainSeeds = "20260880,20260881,20260882",
    [int]$PoolSeed = 20260860,

    [ValidateSet("selected","original")]
    [string]$ParameterBoundsSource = "selected",

    [switch]$Quick,
    [switch]$RegeneratePool,
    [switch]$RegenerateData,
    [switch]$Retrain
)

$ErrorActionPreference = "Stop"
$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Required = @(
    ".\build_exp35_3p_alias_pool.py",
    ".\data_generate_exp35_3p_identifiable.py",
    ".\TripleInverseModels.py",
    ".\train_exp35_3p_model_compare.py",
    ".\analyze_exp35_3p_model_compare.py",
    ".\data_generate_exp20_pairwise.py",
    ".\data_generate_exp26_gseparated_independent.py",
    ".\data_generate_exp22c_fixed_capacity.py",
    ".\mc_physics.py",
    ".\mc_pool_config.py"
)
foreach ($f in $Required) {
    if (-not (Test-Path $f)) { throw "Required file not found: $f" }
}

function Parse-IntList([string]$Text) {
    $Vals=@()
    foreach($x in $Text.Split(",")){ $s=$x.Trim(); if($s){$Vals += [int]$s} }
    return @($Vals | Sort-Object -Unique)
}
function Parse-StringList([string]$Text) {
    $Vals=@()
    foreach($x in $Text.Split(",")){ $s=$x.Trim().ToLower(); if($s){$Vals += $s} }
    return @($Vals | Sort-Object -Unique)
}
$DS=Parse-IntList $DataSeeds
$TS=Parse-IntList $TrainSeeds
$MS=Parse-StringList $Models
foreach($m in $MS){
    if($m -notin @("mlp","tcn","pspec")){throw "Unknown model: $m"}
}

$ATag=("{0:g}" -f $AliasThreshold).Replace(".","p")
$STag=("{0:g}" -f $MinSeparationRMSSNR).Replace(".","p")
$CTag=("{0:g}" -f $MaxCoverRMSSNR).Replace(".","p")

if($Quick){
    $PoolDir=".\data_exp35_3p_alias_pool_quick"
    $CandidateCount=2500
    $NeighborK=96
    $ProjectionRepeats=2
    $MaxStates=100
    $MinStates=24
    $TrainN=3500;$ValN=900;$TestN=1400
    $Epochs=3
    $CompareDir=".\validation_results\exp35_3p_compare_quick"
}else{
    $PoolDir=".\data_exp35_3p_alias_pool"
    $CandidateCount=20000
    $NeighborK=384
    $ProjectionRepeats=3
    $MaxStates=420
    $MinStates=60
    $TrainN=30000;$ValN=5000;$TestN=8000
    $Epochs=80
    $CompareDir=".\validation_results\exp35_3p_compare"
}
New-Item -ItemType Directory -Force -Path $CompareDir | Out-Null

Write-Host ("="*100)
Write-Host "Exp35: 3P a1 + m + gamma"
Write-Host "Step A: rebuild 3P identifiability"
Write-Host "Step B: MLP / TCN / parameter-specific direct regression baselines"
Write-Host "Alias threshold  : $AliasThreshold"
Write-Host "Recovery gate    : |da1|<=$A1SuccessTol, |dm|<=$MSuccessTol, gamma-factor<=$GammaSuccessFactor"
Write-Host "Models           : $($MS -join ', ')"
Write-Host ("="*100)

$PoolMeta=Join-Path $PoolDir "metadata.json"
if($RegeneratePool -or -not (Test-Path $PoolMeta)){
    & $Python "build_exp35_3p_alias_pool.py" `
        --output-dir $PoolDir `
        --candidate-count $CandidateCount `
        --reference-noise 0.002 `
        --a1-min 0.05 --a1-max 0.20 `
        --m-min 0.40 --m-max 1.20 `
        --gamma-min 0.01 --gamma-max 1.0 `
        --remote-a1-abs $A1SuccessTol `
        --remote-m-abs $MSuccessTol `
        --remote-gamma-factor $GammaSuccessFactor `
        --observation-points 500 `
        --model-input-points 1000 `
        --projection-dim 24 `
        --projection-repeats $ProjectionRepeats `
        --neighbor-k $NeighborK `
        --seed $PoolSeed `
        --overwrite
    if($LASTEXITCODE -ne 0){throw "Exp35 alias pool generation failed"}
}

$Manifest=@()
foreach($d in $DS){
    $DataDir=".\data_exp35_3p_alias${ATag}_sep${STag}_cover${CTag}_dseed${d}"
    if($Quick){$DataDir="${DataDir}_quick"}
    $Meta=Join-Path $DataDir "metadata.json"
    if($RegenerateData -or -not(Test-Path $Meta)){
        & $Python "data_generate_exp35_3p_identifiable.py" `
            --pool-dir $PoolDir `
            --output-dir $DataDir `
            --alias-min-rms-snr $AliasThreshold `
            --min-rms-snr $MinSeparationRMSSNR `
            --max-train-cover-rms-snr $MaxCoverRMSSNR `
            --max-selected-states $MaxStates `
            --min-required-states $MinStates `
            --noise-levels "0,0.002,0.01" `
            --train-noise-level 0.002 `
            --target-train-samples $TrainN `
            --target-val-samples $ValN `
            --target-test-samples $TestN `
            --seed $d `
            --compressed `
            --overwrite
        if($LASTEXITCODE -ne 0){throw "Exp35 data generation failed for $d"}
    }

    foreach($t in $TS){
        foreach($model in $MS){
            $ResultDir=".\validation_results\exp35_${model}_alias${ATag}_d${d}_t${t}"
            if($Quick){$ResultDir="${ResultDir}_quick"}
            $Summary=Join-Path $ResultDir "exp35_summary.csv"
            if($Retrain -or -not(Test-Path $Summary)){
                $TrainArgs=@(
                    "train_exp35_3p_model_compare.py",
                    "--model",$model,
                    "--data-dir",$DataDir,
                    "--output-dir",$ResultDir,
                    "--result-prefix","exp35",
                    "--device",$Device,
                    "--seed",$t,
                    "--epochs",$Epochs,
                    "--batch-size",256,
                    "--eval-batch-size",512,
                    "--patience",15,
                    "--parameter-bounds-source",$ParameterBoundsSource,
                    "--a1-success-tol",$A1SuccessTol,
                    "--m-success-tol",$MSuccessTol,
                    "--gamma-success-factor",$GammaSuccessFactor
                )
                if($Quick){$TrainArgs += "--skip-physics-metrics"}
                & $Python @TrainArgs
                if($LASTEXITCODE -ne 0){throw "Training failed: $model D=$d T=$t"}
            }
            $Manifest += [PSCustomObject]@{
                model=$model;data_seed=$d;train_seed=$t;
                data_dir=$DataDir;result_dir=$ResultDir;result_prefix="exp35"
            }
        }
    }
}

$ManifestPath=Join-Path $CompareDir "exp35_manifest.csv"
$Manifest | Export-Csv -Path $ManifestPath -NoTypeInformation -Encoding UTF8

if(-not $Quick){
    & $Python "analyze_exp35_3p_model_compare.py" `
        --manifest $ManifestPath `
        --output-dir $CompareDir `
        --noise-dir "noise_0p2pct" `
        --target-rate 0.98 `
        --fallback-rate 0.95
    if($LASTEXITCODE -ne 0){throw "Exp35 analysis failed"}
}else{
    Write-Host "Quick smoke run complete; formal aggregate analysis is skipped."
}

Write-Host ("="*100)
Write-Host "Exp35 complete."
Write-Host "Read first:"
Write-Host "  $CompareDir\exp35_data_summary.csv"
Write-Host "  $CompareDir\exp35_model_summary.csv"
Write-Host "  $CompareDir\exp35_stage_gate.csv"
Write-Host "  $CompareDir\exp35_matched_model_deltas.csv"
Write-Host ("="*100)
