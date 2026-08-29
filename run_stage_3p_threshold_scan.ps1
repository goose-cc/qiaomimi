param(
    [string]$Thresholds = "0.10,0.15,0.20,0.25,0.30,0.35,0.40",
    [int]$MinTrainStates = 200,
    [int]$MinValStates = 30,
    [int]$MinTestStates = 30,
    [int]$MaxSelectedStates = 500,
    [double]$MinSeparationRmsSnr = 1.3,
    [double]$MaxParameterCoverRadius = 0.35,
    [double]$MaxTrainCoverRmsSnr = 2.0,
    [double]$MinTestParameterSpan = 0.5,
    [int]$HoldoutSearchTrials = 500,
    [int]$AssignmentTrials = 4000
)

$ErrorActionPreference = "Stop"
$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$ArgsList = @(
    "data_pipeline\scan_stage_3p_thresholds.py",
    "--config", "data_pipeline\configs\stage_3p.json",
    "--source-output", "data_pipeline_outputs\stage_3p",
    "--output-dir", "data_pipeline_outputs\stage_3p_threshold_scan",
    "--thresholds=$Thresholds",
    "--min-train-states", [string]$MinTrainStates,
    "--min-val-states", [string]$MinValStates,
    "--min-test-states", [string]$MinTestStates,
    "--max-selected-states", [string]$MaxSelectedStates,
    "--min-separation-rms-snr", [string]$MinSeparationRmsSnr,
    "--max-parameter-cover-radius", [string]$MaxParameterCoverRadius,
    "--max-train-cover-rms-snr", [string]$MaxTrainCoverRmsSnr,
    "--min-test-parameter-span", [string]$MinTestParameterSpan,
    "--holdout-search-trials", [string]$HoldoutSearchTrials,
    "--assignment-trials", [string]$AssignmentTrials
)

Write-Host "===================================================================================================="
Write-Host "3P threshold scan (DATA ONLY; existing 16000-state identifiability bank is reused)"
Write-Host "thresholds        : $Thresholds"
Write-Host "hard split minima : train >= $MinTrainStates, val >= $MinValStates, test >= $MinTestStates"
Write-Host "NO noise dataset is generated."
Write-Host "===================================================================================================="

& $Python @ArgsList
if ($LASTEXITCODE -ne 0) {
    throw "3P threshold scan failed with exit code $LASTEXITCODE"
}
