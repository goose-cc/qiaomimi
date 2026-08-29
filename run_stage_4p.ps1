param(
    [Nullable[double]]$AliasThreshold = $null,
    [switch]$Formal,
    [switch]$Overwrite
)
$ErrorActionPreference = "Stop"
$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
$ArgsList = @("data_pipeline\run_stage.py", "--config", "data_pipeline\configs\stage_4p.json")
if ($AliasThreshold -ne $null) { $ArgsList += @("--alias-threshold", [string]$AliasThreshold) }
if ($MinSelectedStates -ne $null) { $ArgsList += @("--min-selected-states", [string]$MinSelectedStates) }
if ($Formal) { $ArgsList += "--formal" }
if ($Overwrite) { $ArgsList += "--overwrite" }
& $Python @ArgsList
if ($LASTEXITCODE -ne 0) { throw "4P data pipeline failed with exit code $LASTEXITCODE" }
