param(
    [switch]$Smoke
)
$ErrorActionPreference = "Stop"
$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$Required = @(
    "data_pipeline\pipeline_core.py",
    "data_pipeline\data_generate_candidate_pool.py",
    "data_pipeline\analyze_identifiability.py",
    "data_pipeline\select_identifiable_states.py",
    "data_pipeline\analyze_parameter_coverage.py",
    "data_pipeline\split_physical_states.py",
    "data_pipeline\build_noisy_dataset.py",
    "data_pipeline\validate_dataset.py",
    "data_pipeline\run_stage.py",
    "data_pipeline\configs\stage_3p.json",
    "data_pipeline\configs\stage_4p.json",
    "data_pipeline\configs\stage_5p.json",
    "run_stage_3p.ps1",
    "run_stage_4p.ps1",
    "run_stage_5p.ps1"
)

$Missing = @($Required | Where-Object { -not (Test-Path $_) })
if ($Missing.Count -gt 0) {
    Write-Host "Missing required data-pipeline files:" -ForegroundColor Red
    $Missing | ForEach-Object { Write-Host "  $_" }
    throw "data-pipeline structure is incomplete"
}

Write-Host "[PASS] required data-pipeline files exist"

$CodeFiles = @(
    Get-ChildItem .\data_pipeline -Recurse -File -Include *.py
    Get-Item .\run_stage_3p.ps1, .\run_stage_4p.ps1, .\run_stage_5p.ps1
)
$Forbidden = '(^|\s)(import|from)\s+(torch|tensorflow|keras)(\.|\s|$)|nn\.Module|\.backward\s*\(|optimizer\s*=|model\.fit\s*\(|train_exp\d+'
$Hits = @($CodeFiles | Select-String -Pattern $Forbidden -CaseSensitive:$false)
if ($Hits.Count -gt 0) {
    Write-Host "Network/training implementation patterns found in data scope:" -ForegroundColor Red
    $Hits | ForEach-Object { Write-Host $_ }
    throw "data-pipeline scope check failed"
}
Write-Host "[PASS] no network-training implementation found in data-pipeline code"

& $Python -m py_compile @(
    Get-ChildItem .\data_pipeline -File -Filter *.py | ForEach-Object { $_.FullName }
)
if ($LASTEXITCODE -ne 0) { throw "Python compilation failed" }
Write-Host "[PASS] Python compilation"

if ($Smoke) {
    Write-Host "Running 3P/4P/5P data-only smoke tests..."
    & $Python .\data_pipeline\run_stage.py --config .\data_pipeline\configs\stage_3p.json --smoke --overwrite
    if ($LASTEXITCODE -ne 0) { throw "3P smoke failed" }
    & $Python .\data_pipeline\run_stage.py --config .\data_pipeline\configs\stage_4p.json --smoke --overwrite
    if ($LASTEXITCODE -ne 0) { throw "4P smoke failed" }
    & $Python .\data_pipeline\run_stage.py --config .\data_pipeline\configs\stage_5p.json --smoke --overwrite
    if ($LASTEXITCODE -ne 0) { throw "5P smoke failed" }
    Write-Host "[PASS] 3P/4P/5P data-only smoke tests"
}

Write-Host "Data-pipeline scope verification PASSED." -ForegroundColor Green
