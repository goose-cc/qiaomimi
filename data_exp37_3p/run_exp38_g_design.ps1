param(
    [switch]$Quick,
    [switch]$CoarseOnly,
    [string]$DesignId = ""
)
$ErrorActionPreference = "Stop"
$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
$Required = @(
    ".\mc_physics.py", ".\mc_pool_config.py",
    ".\exp38_g_design_config.json", ".\exp38_observation.py", ".\exp38_core.py",
    ".\scan_exp38_g_designs.py", ".\audit_exp38_g_design.py", ".\selfcheck_exp38.py"
)
foreach ($f in $Required) { if (-not (Test-Path $f)) { throw "Missing required file: $f" } }
Write-Host ("="*100)
Write-Host "EXP38 SELF-CHECK"
& $Python "selfcheck_exp38.py"
if ($LASTEXITCODE -ne 0) { throw "Exp38 self-check failed" }

if ($DesignId -ne "") {
    $Out = if ($Quick) { ".\exp38_design_audit_quick" } else { ".\exp38_design_audit" }
    $A = @("audit_exp38_g_design.py","--config","exp38_g_design_config.json","--design-id",$DesignId,"--output-dir",$Out)
    if ($Quick) { $A += "--quick" }
    & $Python @A
    if ($LASTEXITCODE -ne 0) { throw "Exp38 deep design audit failed" }
    Write-Host "Send the *_audit_summary.csv and *_continuous_margin_scan.csv files from $Out"
    exit 0
}

$ScanDir = if ($Quick) { ".\exp38_design_scan_quick" } else { ".\exp38_design_scan" }
$S = @("scan_exp38_g_designs.py","--config","exp38_g_design_config.json","--output-dir",$ScanDir)
if ($Quick) { $S += "--quick" }
& $Python @S
if ($LASTEXITCODE -ne 0) { throw "Exp38 coarse scan failed" }
Write-Host ("="*100)
Write-Host "COARSE SCAN COMPLETE"
Write-Host "Read/send: $ScanDir\g_design_coarse_scan.csv"
Write-Host "Read/send: $ScanDir\coarse_finalists.json"
Write-Host "Then deep-audit a finalist, for example:"
Write-Host ".\run_exp38_g_design.ps1 -DesignId hybrid_near_200"
Write-Host ("="*100)
