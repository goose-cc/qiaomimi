param(
    [string]$DataDir = ".\fivep_train_val_v1\data",
    [string]$OutputDir = ".\runs_fivep_compare_v1",
    [string]$Device = "auto"
)

python .\train_fivep_compare.py `
  --data-dir $DataDir `
  --output-dir $OutputDir `
  --models tcn mlp `
  --seeds 20260920 20260921 20260922 `
  --epochs 220 `
  --batch-size 256 `
  --patience 35 `
  --device $Device
