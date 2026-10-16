param(
    [string]$DataDir = ".\fivep_train_val_v1\data",
    [string]$OutputDir = ".\runs_fivep_compare_v2",
    [string]$Device = "auto"
)

python .\train_fivep_compare_v2.py `
  --data-dir $DataDir `
  --output-dir $OutputDir `
  --models tcn mlp `
  --seeds 20260920 `
  --epochs 1500 `
  --batch-size 96 `
  --eval-batch-size 512 `
  --eval-every 20 `
  --device $Device `
  --no-amp `
  --no-tf32 `
  --scheduler none `
  --tcn-dropout 0 `
  --tcn-weight-decay 0 `
  --mlp-dropout 0 `
  --mlp-weight-decay 0
