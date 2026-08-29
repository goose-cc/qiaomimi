param(
    [Nullable[double]]$AliasThreshold = $null,
    [switch]$Smoke,
    [switch]$Overwrite
)
$ErrorActionPreference = "Stop"
$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
$ArgsList = @("data_pipeline\run_stage.py", "--config", "data_pipeline\configs\stage_3p.json")
if ($AliasThreshold -ne $null) { $ArgsList += @("--alias-threshold", [string]$AliasThreshold) }
if ($Smoke) { $ArgsList += "--smoke" }
if ($Overwrite) { $ArgsList += "--overwrite" }
& $Python @ArgsList
if ($LASTEXITCODE -ne 0) { throw "3P data pipeline failed with exit code $LASTEXITCODE" }
