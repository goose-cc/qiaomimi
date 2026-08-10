# Gamma-only 数据课程脚本

这组脚本对应 `data-curriculum-gamma` 分支的第一阶段数据任务，并直接复用当前项目的 `mc_physics.py` / `mc_pool_config.py`，不修改物理公式。

## 1. 放置方式

把下面两个文件放到仓库根目录（与 `mc_physics.py` 同级）：

- `data_generate_gamma_curriculum.py`
- `check_gamma_curriculum.py`

它们不会覆盖现有 `data_generate_model3_param_sweep.py`，因此不会破坏旧实验。

## 2. 第一阶段：先只做可区分度分析

```powershell
python data_generate_gamma_curriculum.py `
  --output-dir .\data_gamma_easy `
  --tier easy `
  --analysis-only
```

默认固定：

- a1 = 0.10
- a2 = 0.025
- a3 = 0
- m = 0.8
- q² = [-100, -6]
- Nq = 100
- gamma = 0.01, 0.05, 0.20, 0.50, 1.00
- separation 参考噪声 = 0.2%

会先生成：

- `metadata.json`
- `gamma_values.csv`
- `gamma_pair_separation.csv`
- `gamma_dense_adjacent_separation.csv`

`gamma_pair_separation.csv` 中包含：

- max_abs_g_difference
- rms_g_difference
- l2_g_difference
- relative_l2_difference
- sigma_reference
- snr_sep
- snr_sep_rms

其中项目方案要求的定义严格保存为：

`snr_sep = ||g_i-g_j||_2 / [noise_level * mean(RMS(g_i), RMS(g_j))]`

另外增加 `snr_sep_rms`，便于以后比较 100 / 500 / 1000 个 q² 点时排除 L2 自带的 sqrt(Nq) 因子。

## 3. 生成 Easy 完整数据

```powershell
python data_generate_gamma_curriculum.py `
  --output-dir .\data_gamma_easy `
  --tier easy `
  --noise-levels 0,0.002,0.01 `
  --overwrite
```

默认每个 gamma：

- train: 4000 个独立噪声 realization
- val: 500
- test_seen: 1000
- test_interp: 1000

Easy 的训练 gamma：

`0.01, 0.05, 0.20, 0.50, 1.00`

Easy 的 interpolation gamma：

`0.025, 0.10, 0.32, 0.70`

输出目录：

```text
data_gamma_easy/
  metadata.json
  gamma_values.csv
  gamma_pair_separation.csv
  gamma_dense_adjacent_separation.csv
  noise_0pct/
    train.npz
    val.npz
    test_seen.npz
    test_interp.npz
  noise_0p2pct/
    ...
  noise_1pct/
    ...
```

## 4. NPZ 字段

每个数据文件保存：

- `fx`: 当前项目原有的 scaled target u(s)，可直接给 PeakInversionDataset
- `gy_clean`
- `gy_noisy`
- `x`
- `y`
- `q2`
- `parameters`
- `a1`, `a2`, `a3`, `m`, `gamma`
- `gamma_class`
- `gamma_index`
- `noise_level`
- `noise_sigma`
- `seed`
- `noise_realization_id`

Seen splits 的 `gamma_class` 为 0..K-1；`test_interp` 为 -1，因为这些 gamma 从未出现在训练类别中。

现有 `PIDataset.PeakInversionDataset` 会优先读取 `gy_noisy` 并读取 `fx`，因此无需为了第一阶段反演训练修改 Dataset 类。

## 5. 检查数据独立性和格式

```powershell
python check_gamma_curriculum.py .\data_gamma_easy
```

它会检查：

- 四个 split 是否存在
- train/val/test_seen/test_interp seed 范围是否互不重叠
- interpolation gamma 是否未进入训练
- a1/a2/a3/m 是否真的固定
- q² 点数是否正确
- 0% 时 gy_clean == gy_noisy
- 非零噪声的经验 RMS 是否与 sigma 一致
- 是否有 NaN/Inf

## 6. Medium / Hard

在 Easy baseline 跑通之后：

```powershell
python data_generate_gamma_curriculum.py `
  --output-dir .\data_gamma_medium `
  --tier medium

python data_generate_gamma_curriculum.py `
  --output-dir .\data_gamma_hard `
  --tier hard
```

默认使用 0.2% 噪声作为可区分度参考：

- Medium：相邻点自动寻找 `SNR_sep ≈ 3`
- Hard：相邻点自动寻找 `SNR_sep ≈ 1`

自动选点来自 `gamma ∈ [0.001, 1]` 的 log grid，并且实际 gamma 会写入 `gamma_values.csv`，不会只依赖文字说明。

## 7. 自定义 gamma

```powershell
python data_generate_gamma_curriculum.py `
  --output-dir .\data_gamma_custom `
  --gamma-values 0.01,0.03,0.08,0.25,0.8 `
  --interp-gammas 0.02,0.05,0.15,0.5
```

## 8. 后续 q² density 实验

第一阶段保持默认 `--q2-points 100`。

以后单独开 `data-q-density` 时，可以在**完全相同 gamma 真值**下重新运行：

```powershell
--q2-points 100
--q2-points 500
--q2-points 1000
```

脚本会调用物理 forward 在新的 q² 网格上重新积分；不会把 100 点结果插值成 500/1000 点。
