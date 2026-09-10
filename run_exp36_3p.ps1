param(
    [string]$Device = "cuda",
    [switch]$Quick,
    [switch]$RebuildData,
    [switch]$DataOnly,
    [switch]$SkipContinuousRefine
)
$ErrorActionPreference="Stop"
$Python=".\venv\Scripts\python.exe"; if(-not(Test-Path $Python)){$Python="python"}
$Required=@('.\mc_physics.py','.\mc_pool_config.py','.\exp36_forward64.py','.\selfcheck_exp36_alias_numerics.py','.\exp36_3p_config.json','.\build_exp36_3p_dataset.py','.\validate_exp36_3p_dataset.py','.\TripleInverseTCN3P.py','.\train_exp36_3p_tcn.py','.\analyze_exp36_3p_tcn.py')
foreach($f in $Required){if(-not(Test-Path $f)){throw "Missing required file: $f"}}
$DataDir=if($Quick){'.\data_exp36_3p_quick'}else{'.\data_exp36_3p'}
Write-Host ('='*100); Write-Host 'EXP36-v2 ALIAS NUMERICS SELF-CHECK'
& $Python 'selfcheck_exp36_alias_numerics.py'; if($LASTEXITCODE -ne 0){throw 'Exp36-v2 numerical self-check failed'}
$Meta=Join-Path $DataDir 'metadata.json'
if($RebuildData -or -not(Test-Path $Meta)){
    Write-Host ('='*100); Write-Host 'BUILD EXP36-v2 3P DATA (FLOAT64 DIRECT-VERIFIED ALIAS)'
    $A=@('build_exp36_3p_dataset.py','--config','exp36_3p_config.json','--output-dir',$DataDir)
    if($Quick){$A+='--quick'}
    if($SkipContinuousRefine){$A+='--skip-continuous-refine'}
    & $Python @A; if($LASTEXITCODE -ne 0){throw 'Exp36-v2 data build failed'}
}else{Write-Host "Reuse existing Exp36-v2 data: $DataDir"}
Write-Host ('='*100); Write-Host 'STRICT DATA + ALIAS NUMERICS VALIDATION'
$V=@('validate_exp36_3p_dataset.py','--data-dir',$DataDir); if($Quick){$V+=@('--max-forward-states','36')}
& $Python @V; if($LASTEXITCODE -ne 0){throw 'Exp36-v2 validation failed'}
if($DataOnly){Write-Host 'DataOnly requested. Stop after validated data build.'; exit 0}
$Cfg=Get-Content '.\exp36_3p_config.json' -Raw|ConvertFrom-Json; $Seeds=@($Cfg.training.seeds); if($Quick){$Seeds=@($Seeds[0])}
foreach($Seed in $Seeds){
    $RunDir=".\validation_results\exp36_v2_tcn3p_seed${Seed}"; if($Quick){$RunDir="${RunDir}_quick"}
    $T=@('train_exp36_3p_tcn.py','--config','exp36_3p_config.json','--data-dir',$DataDir,'--output-dir',$RunDir,'--seed',[string]$Seed,'--device',$Device); if($Quick){$T+='--quick'}
    & $Python @T; if($LASTEXITCODE -ne 0){throw "TCN training failed seed=$Seed"}
}
if(-not $Quick){& $Python 'analyze_exp36_3p_tcn.py' --config 'exp36_3p_config.json' --runs-root '.\validation_results' --output-dir '.\validation_results\exp36_v2_3p_tcn_summary'; if($LASTEXITCODE -ne 0){throw 'analysis failed'}}
Write-Host ('='*100); Write-Host 'EXP36-v2 COMPLETE'
Write-Host "Inspect first: $DataDir\exp36_data_summary.csv"; Write-Host "               $DataDir\alias_threshold_scan.csv"; Write-Host "               $DataDir\profiled_alias_selected.csv"; Write-Host "               $DataDir\postbuild_validation.csv"
