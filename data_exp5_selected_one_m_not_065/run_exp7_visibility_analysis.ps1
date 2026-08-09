param(
    [string]$ValidationPoolDir = ".\truth_pool_val_10k",
    [string]$CheckpointDir = ".\model\v2_exp7_parametric_noise0_160m",
    [string]$OutputRoot = ".\validation_results\exp7_visibility",
    [ValidateSet("best","latest","both")]
    [string]$Weights = "both",
    [int]$NumSamples = 10000,
    [int]$BatchSize = 64,
    [double]$NoiseLevel = 0.0,
    [string]$Device = "cuda",
    [double]$WeakThreshold = 0.05,
    [double]$VisibleThreshold = 0.20
)

$ErrorActionPreference = "Stop"

function Run-One([string]$W) {
    $outDir = Join-Path $OutputRoot ("{0}_noise{1}" -f $W, [int]($NoiseLevel * 100))

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Exp7 visibility analysis"
    Write-Host "weights    : $W"
    Write-Host "noise      : $NoiseLevel"
    Write-Host "output     : $outDir"
    Write-Host "============================================================"

    python .\analyze_exp7_visibility.py `
        --validation-pool-dir $ValidationPoolDir `
        --checkpoint-dir $CheckpointDir `
        --weights $W `
        --num-samples $NumSamples `
        --batch-size $BatchSize `
        --noise-level $NoiseLevel `
        --weak-threshold $WeakThreshold `
        --visible-threshold $VisibleThreshold `
        --output-dir $outDir `
        --device $Device

    if ($LASTEXITCODE -ne 0) {
        throw "analyze_exp7_visibility.py failed for weights=$W"
    }
}

if ($Weights -eq "both") {
    Run-One "best"
    Run-One "latest"
} else {
    Run-One $Weights
}
