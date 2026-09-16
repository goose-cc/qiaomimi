param(
    [switch]$Quick,
    [switch]$ScanOnly,
    [string]$DesignId = ""
)
$ErrorActionPreference = "Stop"
$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Required = @(
  ".\mc_physics.py", ".\mc_pool_config.py", ".\cp02_corrected_3p_config.json",
  ".\cp02_observation.py", ".\cp02_core.py", ".\scan_cp02_g_designs.py",
  ".\build_cp02_identifiable_bank.py", ".\analyze_cp02_varpro_ceiling.py",
  ".\validate_cp02.py", ".\selfcheck_cp02.py"
)
foreach ($f in $Required) { if (-not (Test-Path $f)) { throw "Missing required file: $f" } }

& $Python selfcheck_cp02.py
if ($LASTEXITCODE -ne 0) { throw "CP02 selfcheck failed" }

$ScanDir = if ($Quick) { ".\cp02_design_scan_quick" } else { ".\cp02_design_scan" }
$scanArgs = @("scan_cp02_g_designs.py","--config","cp02_corrected_3p_config.json","--output-dir",$ScanDir)
if ($Quick) { $scanArgs += "--quick" }
& $Python @scanArgs
if ($LASTEXITCODE -ne 0) { throw "CP02 g-design scan failed" }
if ($ScanOnly) { Write-Host "ScanOnly complete. Inspect $ScanDir\cp02_g_design_continuous.csv"; exit 0 }

if ([string]::IsNullOrWhiteSpace($DesignId)) {
  $fin = Get-Content (Join-Path $ScanDir "cp02_finalists.json") -Raw | ConvertFrom-Json
  $DesignId = [string]$fin.recommended_by_continuous
}
Write-Host "Using design: $DesignId"

$BankDir = if ($Quick) { ".\cp02_identifiable_bank_quick" } else { ".\cp02_identifiable_bank" }
$buildArgs = @("build_cp02_identifiable_bank.py","--config","cp02_corrected_3p_config.json","--design-id",$DesignId,"--output-dir",$BankDir)
if ($Quick) { $buildArgs += "--quick" }
& $Python @buildArgs
if ($LASTEXITCODE -ne 0) { throw "CP02 identifiable-bank build failed" }
& $Python validate_cp02.py --config cp02_corrected_3p_config.json --bank-dir $BankDir
if ($LASTEXITCODE -ne 0) { throw "CP02 validation failed" }

$VarDir = if ($Quick) { ".\cp02_varpro_quick" } else { ".\cp02_varpro" }
$varArgs = @("analyze_cp02_varpro_ceiling.py","--config","cp02_corrected_3p_config.json","--bank-dir",$BankDir,"--output-dir",$VarDir)
if ($Quick) { $varArgs += "--quick" }
& $Python @varArgs
if ($LASTEXITCODE -ne 0) { throw "CP02 VarPro failed" }

Write-Host "CP02 complete. Key outputs:"
Write-Host "  $ScanDir\cp02_g_design_continuous.csv"
Write-Host "  $BankDir\cp02_bank_summary.csv"
Write-Host "  $BankDir\cp02_selection_threshold_scan.csv"
Write-Host "  $BankDir\cp02_identifiable_region_coverage.csv"
Write-Host "  $VarDir\cp02_varpro_summary.csv"
