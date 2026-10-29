@echo off
python train_fivep_compare_v2.py --data-dir "%~1" --output-dir runs_fivep_tcn --models tcn --seeds 20260920 --epochs 1500 --batch-size 96 --eval-every 20 --device auto --no-amp --no-tf32
pause
