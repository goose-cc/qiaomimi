param(
    [switch]$Quick,
    [switch]$RebuildData,
    [switch]$SkipVarPro
)

$ErrorActionPreference = "Stop"
$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Required = @(
    ".\mc_physics.py",
    ".\mc_pool_config.py",
    ".\exp37_3p_config.json",
    ".\exp37_core.py",
    ".\build_exp37_3p_continuous_dataset.py",
    ".\validate_exp37_3p_dataset.py",
    ".\analyze_exp37_jacobian.py",
    ".\analyze_exp37_varpro_ceiling.py",
    ".\selfcheck_exp37.py"
)
foreach ($f in $Required) {
    if (-not (Test-Path $f)) { throw "Missing required file: $f" }
}

$DataDir = if ($Quick) { ".\data_exp37_3p_quick" } else { ".\data_exp37_3p" }

Write-Host ("=" * 100)
Write-Host "EXP37 NUMERICAL SELF-CHECK"
& $Python "selfcheck_exp37.py"
if ($LASTEXITCODE -ne 0) { throw "Exp37 numerical self-check failed" }

$Meta = Join-Path $DataDir "metadata.json"
if ($RebuildData -or -not (Test-Path $Meta)) {
    Write-Host ("=" * 100)
    Write-Host "EXP37 BUILD: CONTINUOUS-IDENTIFIABILITY-FIRST 3P DATA"
    $A = @(
        "build_exp37_3p_continuous_dataset.py",
        "--config", "exp37_3p_config.json",
        "--output-dir", $DataDir
    )
    if ($Quick) { $A += "--quick" }
    & $Python @A
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Exp37 selection failed. This is not automatically a code error."
        Write-Host "Inspect:"
        Write-Host "  $DataDir\continuous_threshold_scan.csv"
        Write-Host "  $DataDir\selection_failure.json"
        throw "No bank satisfied the configured continuous-identifiability + coverage constraints."
    }
} else {
    Write-Host "Reuse existing Exp37 data: $DataDir"
}

Write-Host ("=" * 100)
Write-Host "EXP37 STRICT DATA VALIDATION"
$V = @(
    "validate_exp37_3p_dataset.py",
    "--config", "exp37_3p_config.json",
    "--data-dir", $DataDir
)
if ($Quick) { $V += "--quick" }
& $Python @V
if ($LASTEXITCODE -ne 0) { throw "Exp37 data validation failed" }

Write-Host ("=" * 100)
Write-Host "EXP37 JACOBIAN CONDITIONING DIAGNOSTIC"
& $Python "analyze_exp37_jacobian.py" `
    --config "exp37_3p_config.json" `
    --data-dir $DataDir
if ($LASTEXITCODE -ne 0) { throw "Exp37 Jacobian analysis failed" }

if (-not $SkipVarPro) {
    Write-Host ("=" * 100)
    Write-Host "EXP37 VARPRO PHYSICS CEILING DIAGNOSTIC"
    & $Python "analyze_exp37_varpro_ceiling.py" `
        --config "exp37_3p_config.json" `
        --data-dir $DataDir
    if ($LASTEXITCODE -ne 0) { throw "Exp37 VarPro analysis failed" }
}

Write-Host ("=" * 100)
Write-Host "EXP37 COMPLETE"
Write-Host "Send these files first:"
Write-Host "  $DataDir\exp37_data_summary.csv"
Write-Host "  $DataDir\continuous_threshold_scan.csv"
Write-Host "  $DataDir\selected_refined_margin_scan.csv"
Write-Host "  $DataDir\profiled_alias_selected.csv"
Write-Host "  $DataDir\coverage_summary.csv"
Write-Host "  $DataDir\postbuild_validation.csv"
Write-Host "  $DataDir\jacobian_summary.csv"
if (-not $SkipVarPro) {
    Write-Host "  $DataDir\varpro_summary.csv"
}
Write-Host ("=" * 100)
