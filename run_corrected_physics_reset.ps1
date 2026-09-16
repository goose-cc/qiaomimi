param(
    [switch]$Quick,
    [string]$OutputDir = ".\corrected_physics_reset_results"
)

$ErrorActionPreference = "Stop"
$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Required = @(
    ".\mc_pool_config.py",
    ".\mc_physics.py",
    ".\mc_parametric.py",
    ".\mc_online_physics.py",
    ".\corrected_physics_reset_config.json",
    ".\selfcheck_corrected_physics.py",
    ".\check_corrected_formula_consistency.py",
    ".\audit_corrected_physics_reset.py"
)
foreach ($f in $Required) {
    if (-not (Test-Path $f)) { throw "Missing required file: $f" }
}

Write-Host ("=" * 100)
Write-Host "CORRECTED PHYSICS SOURCE-OF-TRUTH SELF-CHECK"
& $Python "selfcheck_corrected_physics.py" --config "corrected_physics_reset_config.json"
if ($LASTEXITCODE -ne 0) { throw "corrected physics self-check failed" }

Write-Host ("=" * 100)
Write-Host "FORMULA CONSISTENCY AUDIT"
& $Python "check_corrected_formula_consistency.py" --root "." --output (Join-Path $OutputDir "stale_formula_references.csv")
if ($LASTEXITCODE -ne 0) { throw "active physics files still contain the old formula" }

Write-Host ("=" * 100)
Write-Host "CP01: 2P/3P DEVELOPMENT-vs-FULL PHYSICS AUDIT"
$AuditArgs = @(
    "audit_corrected_physics_reset.py",
    "--config", "corrected_physics_reset_config.json",
    "--output-dir", $OutputDir
)
if ($Quick) { $AuditArgs += "--quick" }
& $Python @AuditArgs
if ($LASTEXITCODE -ne 0) { throw "corrected physics reset audit failed" }

Write-Host ("=" * 100)
Write-Host "DONE. Inspect first:"
Write-Host "  $OutputDir\corrected_physics_reset_summary.csv"
Write-Host "  $OutputDir\full_domain_recovery_by_a1.csv"
Write-Host "  $OutputDir\continuous_alias_audit.csv"
Write-Host "  $OutputDir\boundary_identifiability.csv"
Write-Host "  $OutputDir\stale_formula_references.csv"
Write-Host ("=" * 100)
