# CP02 TCN GPU-fast v2

Replace these two project-root files:

- `train_cp02_tcn.py`
- `run_cp02_tcn_baseline.ps1`

The scientific task is unchanged. This patch only changes the execution path and progress reporting.

## What changed

1. Default batch size: 256 -> 512.
2. CUDA AMP / FP16 is enabled by default.
3. TF32 is enabled by default on CUDA.
4. cuDNN benchmark mode is enabled by default.
5. Strict deterministic cuDNN is no longer the default.
6. The normalized train/val/test tensors are cached on GPU by default.
   For this dataset, `28800 x 239` float32 inputs are only about 26 MB,
   so CPU->GPU batch transfer should not be the hot-path bottleneck.
7. Training prints progress every 10 batches:
   - batch / total batches
   - running train loss
   - samples/sec
   - GPU allocated/reserved memory
8. Every epoch prints elapsed seconds and throughput.

The model, targets, train/val/test split, recovery criteria, and checkpoint selection are unchanged.

## First run: quick smoke test

```powershell
.\run_cp02_tcn_baseline.ps1 -Quick -Device cuda
```

You should immediately see lines similar to:

```text
device=cuda gpu=NVIDIA GeForce RTX 4050 Laptop GPU ...
fast_backend: amp=True tf32=True cudnn_benchmark=True deterministic=False gpu_cache=True
Caching normalized train/val/test tensors on GPU...
--- seed=20260920 start ...
seed=20260920 ep=1 batch=10/... train_loss=... ... samples/s ...
```

## Recommended first scientific run: one seed

```powershell
.\run_cp02_tcn_baseline.ps1 -SingleSeed -Device cuda
```

Defaults for `-SingleSeed`:

- 1 seed
- max 150 epochs
- patience 20
- batch 512
- full TCN channels

If this first seed reaches acceptable recovery, run the formal three-seed experiment:

```powershell
.\run_cp02_tcn_baseline.ps1 -Device cuda
```

## If CUDA runs out of memory

Use:

```powershell
.\run_cp02_tcn_baseline.ps1 -SingleSeed -Device cuda -BatchSize 256
```

## If GPU cache causes a problem

Disable only the cache path:

```powershell
.\run_cp02_tcn_baseline.ps1 -SingleSeed -Device cuda -NoGPUCache
```

## Final strict reproducibility rerun

The fast path is appropriate for model development and the first baseline.
If you later want a stricter deterministic rerun:

```powershell
.\run_cp02_tcn_baseline.ps1 -Device cuda -Deterministic -NoTF32
```

AMP can remain on unless you specifically want an FP32 reproducibility control.

## Important

Do not compare wall-clock speed between v1 and v2 while simultaneously changing
the TCN channels or data. The default formal architecture is still:

```text
64 -> 96 -> 128 -> 160
```

and the data definition is unchanged.
