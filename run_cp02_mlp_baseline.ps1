param(
  [string]$DataDir = "data_cp02_ml_hybrid_domain_v2",
  [string]$OutputDir = "runs_cp02_mlp",
  [switch]$Quick
)
$ErrorActionPreference = "Stop"
if ($Quick) {
  python .\train_cp02_mlp_baseline.py --data-dir $DataDir --output-dir "${OutputDir}_quick" --seeds 20260920 --epochs 20 --patience 8
} else {
  python .\train_cp02_mlp_baseline.py --data-dir $DataDir --output-dir $OutputDir --seeds 20260920 20260921 20260922 --epochs 300 --patience 30
}
