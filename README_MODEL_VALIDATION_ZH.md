# Transformer + PINN 独立验证指南

验证不会更新权重。流程是：

```text
独立参数 -> f_true -> g_clean -> 固定 9% 噪声
         -> 已训练 Transformer -> f_pred
```

修正版验证噪声由独立 NumPy 随机生成器产生。在相同 `seed` 下，噪声不依赖 CPU/GPU 或 batch size，便于公平比较 `best` 和 `latest`。

## 1. 生成独立验证池

```powershell
python generate_mc_parameter_pool.py `
  --output-dir ./truth_pool_val_10k `
  --num-truths 10000 `
  --candidate-batch-size 100000 `
  --seed 20260802
```

```powershell
python check_mc_parameter_pool.py `
  --pool-dir ./truth_pool_val_10k `
  --sample-size 10000 `
  --require-complete
```

不要使用训练参数池代替独立验证池。

## 2. 验证 best 权重

```powershell
python validate_mc_transformer.py `
  --validation-pool-dir ./truth_pool_val_10k `
  --checkpoint-dir ./model/mc_transformer_pinn `
  --weights best `
  --num-samples 10000 `
  --batch-size 256 `
  --seed 20260802 `
  --noise-level 0.09 `
  --output-dir ./validation_results/pinn_best `
  --plot-count 12 `
  --device cuda
```

验证 `latest` 时只把 `--weights best` 改成 `--weights latest`，其他参数保持一致。

## 3. 输出

```text
validation_results/pinn_best/
  validation_summary.json
  validation_metrics.csv
  plots/
```

## 4. 主要指标

- `f_rmse`：预测曲线与真值曲线的绝对 RMSE。
- `f_true_rms`：真值曲线自身的 RMS 尺度。
- `f_nrmse_rms = f_rmse/f_true_rms`：相对重建误差。
- `f_relative_l2`：相对 L2；对等长向量，它与上述 NRMSE 数值相同。
- `g_relative_l2_vs_clean`：`K(f_pred)` 与高精度 `g_clean` 的相对误差。
- `g_relative_l2_vs_discrete_clean`：`K(f_pred)` 与 `K(f_true_100点)` 的相对误差。
- `clean_discrete_representation_gap`：100 点标签与高精度正演之间的先天表示缺口。
- `peak_grid_error`：预测曲线全局最大点与 100 点真值全局最大点的距离。
- `negative_prediction_point_fraction`：预测为负的网格点比例。

### 关于 m 和峰位置

`m` 是 Lorentz 共振项的中心参数，但它不一定等于完整 `u(s)` 的全局最大点，因为：

1. 还有线性背景 `a2*s+a3`；
2. `u(s)` 还除以 `(s+400)^2`；
3. 当共振很弱时，全局最大点可能由背景决定。

因此验证文件使用：

- `true_argmax_to_m_error`
- `pred_argmax_to_m_error`

作为诊断量，而不把它们当成严格的“峰中心误差”。

## 5. 如何判断表示冲突

- `f_nrmse_rms` 大、`clean_discrete_representation_gap` 小：主要是模型尚未学好。
- `clean_discrete_representation_gap` 在窄峰样本中很大：100 点输出表示不足。
- `g vs discrete-clean` 小但 `g vs clean` 大：模型已学会离散问题，差异主要来自输出分辨率。

只有验证确认存在表示冲突后，才建立新的 `discrete-clean` 对照实验，不能覆盖原 checkpoint。
