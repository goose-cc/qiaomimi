Five-parameter TCN training (a1,a2,a3,m,gamma)
Run from folder containing train_fivep_compare_v2.py:
python train_fivep_compare_v2.py --data-dir "PATH_TO_DATA" --output-dir runs_fivep_tcn --models tcn --seeds 20260920 --epochs 1500 --batch-size 96 --eval-every 20 --device auto --no-amp --no-tf32
Required under PATH_TO_DATA:
noise_0p2pct/train.npz
noise_0p2pct/val.npz
normalization_train_only.npz
Optional: q2.npy
No test split is accessed.
