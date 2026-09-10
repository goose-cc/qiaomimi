param(
    [switch]$Quick,
    [switch]$DiagnoseOnly,
    [switch]$BuildOnly,
    [switch]$AuditAll,
    [string]$DesignId = ""
)
$ErrorActionPreference = "Stop"
$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
$Required = @(
    ".\mc_physics.py", ".\mc_pool_config.py",
    ".\exp38_core.py", ".\exp38_observation.py",
    ".\exp39_targeted_g_config.json", ".\exp39_core.py",
    ".\diagnose_exp39_m_slices.py", ".\design_exp39_targeted_q2.py",
    ".\audit_exp39_targeted_design.py", ".\compare_exp39_audits.py", ".\selfcheck_exp39.py"
)
foreach ($f in $Required) { if (-not (Test-Path $f)) { throw "Missing required file: $f" } }

Write-Host ("="*100)
Write-Host "EXP39 SELF-CHECK"
& $Python "selfcheck_exp39.py"
if ($LASTEXITCODE -ne 0) { throw "Exp39 self-check failed" }

$DiagDir = if ($Quick) { ".\exp39_slice_diagnosis_quick" } else { ".\exp39_slice_diagnosis" }
$DesignDir = if ($Quick) { ".\exp39_targeted_designs_quick" } else { ".\exp39_targeted_designs" }
$AuditDir = if ($Quick) { ".\exp39_targeted_audit_quick" } else { ".\exp39_targeted_audit" }

# A design-only/audit call still needs the fixed-m diagnosis because it defines the targeted slices.
if (-not (Test-Path (Join-Path $DiagDir "worst_m_slices.json")) -or $DiagnoseOnly -or ($DesignId -eq "" -and -not $AuditAll)) {
    $A=@("diagnose_exp39_m_slices.py","--config","exp39_targeted_g_config.json","--output-dir",$DiagDir)
    if ($Quick) { $A += "--quick" }
    & $Python @A
    if ($LASTEXITCODE -ne 0) { throw "Exp39 fixed-m diagnosis failed" }
}
if ($DiagnoseOnly) {
    Write-Host "Send: $DiagDir\m_slice_diagnosis.csv"
    Write-Host "Send: $DiagDir\worst_m_slices.json"
    exit 0
}

if (-not (Test-Path (Join-Path $DesignDir "design_build_summary.csv")) -or $BuildOnly -or ($DesignId -eq "" -and -not $AuditAll)) {
    $B=@("design_exp39_targeted_q2.py","--config","exp39_targeted_g_config.json","--worst-slices",(Join-Path $DiagDir "worst_m_slices.json"),"--output-dir",$DesignDir)
    if ($Quick) { $B += "--quick" }
    & $Python @B
    if ($LASTEXITCODE -ne 0) { throw "Exp39 targeted q2 design failed" }
}
if ($BuildOnly) {
    Write-Host "Send: $DiagDir\m_slice_diagnosis.csv"
    Write-Host "Send: $DesignDir\design_build_summary.csv"
    exit 0
}

if ($DesignId -ne "") {
    $C=@("audit_exp39_targeted_design.py","--config","exp39_targeted_g_config.json","--design-dir",$DesignDir,"--design-id",$DesignId,"--worst-slices",(Join-Path $DiagDir "worst_m_slices.json"),"--output-dir",$AuditDir)
    if ($Quick) { $C += "--quick" }
    & $Python @C
    if ($LASTEXITCODE -ne 0) { throw "Exp39 deep audit failed: $DesignId" }
    Write-Host "Send: $AuditDir\${DesignId}_audit_summary.csv"
    Write-Host "Send: $AuditDir\${DesignId}_continuous_margin_scan.csv"
    Write-Host "Send: $AuditDir\${DesignId}_worst_slice_audit.csv"
    exit 0
}

if ($AuditAll) {
    $Ids = Get-ChildItem $DesignDir -Filter "*_q2.npy" | ForEach-Object { $_.BaseName -replace '_q2$','' }
    foreach ($Id in $Ids) {
        Write-Host ("="*100); Write-Host "DEEP AUDIT $Id"
        $C=@("audit_exp39_targeted_design.py","--config","exp39_targeted_g_config.json","--design-dir",$DesignDir,"--design-id",$Id,"--worst-slices",(Join-Path $DiagDir "worst_m_slices.json"),"--output-dir",$AuditDir)
        if ($Quick) { $C += "--quick" }
        & $Python @C
        if ($LASTEXITCODE -ne 0) { throw "Exp39 deep audit failed: $Id" }
    }
    & $Python "compare_exp39_audits.py" --audit-dir $AuditDir
    if ($LASTEXITCODE -ne 0) { throw "Exp39 audit comparison failed" }
    Write-Host "Send: $AuditDir\exp39_deep_audit_ranking.csv"
    Write-Host "Also send the winner's *_audit_summary.csv and *_worst_slice_audit.csv"
    exit 0
}

Write-Host ("="*100)
Write-Host "EXP39 DIAGNOSIS + TARGETED DESIGN COMPLETE"
Write-Host "Read/send: $DiagDir\m_slice_diagnosis.csv"
Write-Host "Read/send: $DesignDir\design_build_summary.csv"
Write-Host "Then deep-audit all generated designs:"
Write-Host ".\run_exp39_targeted_g.ps1 -AuditAll"
Write-Host ("="*100)
