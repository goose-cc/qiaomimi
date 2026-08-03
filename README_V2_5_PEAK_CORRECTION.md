# V2.5 峰恢复纠正版：先诊断，再决定是否继续训练

本版本从用户当前 `8.zip` 的 V2 自由曲线模型出发。它没有改变：

- 参数范围 `a1,a2,a3,m,gamma`；
- `rho(s)`、`f(s)` 和 `g(q²)` 的物理公式；
- `s=[0.1764,6]`、`q²=[-100,-6]`；
- `data_scale=160000`；
- 最终 9% RMS 高斯噪声定义。

它也没有重新使用 V3 的“唯一五参数输出”。模型仍然直接预测完整 100 点谱 `f_pred(s)`。

## 为什么要改

当前 V2 的普通完整谱 MSE 容易被大量背景点主导，典型情况是：

- 完整谱和背景看起来较好；
- 共振峰被明显压平；
- 错误的谱重新正演后仍能给出很接近的 `g`。

因此不能只继续增加普通训练步数。

## 本版本做了什么

### 1. 新增峰感知 Loss

新增两个 profile：

```text
peak_grad
peak_pinn
```

`peak_grad`：

```text
完整谱相对 MSE
+ 梯度 Loss
+ 峰区加权完整谱 MSE
+ 可见共振分量监督
```

`peak_pinn` 在此基础上加入较弱的正演物理项。

峰区加权仍然比较完整的 `f_pred` 和 `f_true`，只是让真实共振附近的网格点获得更高权重。

共振监督只对满足以下条件的合成样本启用：

- `m` 位于当前 `s` 区间内；
- 共振可见度不低于阈值；
- `m*gamma` 至少覆盖指定数量的输出网格。

这不会删除数据，也不会改变物理问题；不可清晰评价的样本仍参加完整谱和梯度训练，只是不强行计算峰中心监督。

### 2. 新增噪声课程

可以令训练噪声逐步从 0% 增加到最终 9%：

```text
--noise-start-level 0
--noise-curriculum-steps 5000
--noise-level 0.09
```

最终训练目标仍然是 9% 噪声，只是先学习基本映射，再逐步增加难度。

### 3. 新增绝对总步数

```text
--target-global-step 20000
```

表示训练到总步数 20,000，而不是“本次再增加 20,000 步”。

### 4. 修正验证峰指标

原验证脚本的完整谱 `argmax` 不等于 Lorentz 共振中心。本版本增加：

- `resonance_relative_l2`
- `resonance_center_error`
- `resonance_height_relative_error`
- `resonance_visibility`
- `width_grid_cells`
- `visible_peak_eligible`

旧的完整谱 argmax 指标保留为兼容字段，但终端会明确标为 `legacy`。

### 5. 新增诊断和数据平衡工具

```text
diagnose_peak_identifiability.py
make_width_balanced_pool.py
analyze_validation_by_width.py
```

## 现在先做什么

### 第一步：只重新验证现有 V2 权重，不训练

```powershell
python -u validate_mc_transformer.py `
  --validation-pool-dir ./truth_pool_val_10k `
  --checkpoint-dir ./local_models/mc_transformer_base_v2 `
  --weights latest `
  --num-samples 10000 `
  --batch-size 256 `
  --seed 20260802 `
  --noise-level 0.09 `
  --output-dir ./validation_results/base_v2_peak_metrics_corrected `
  --plot-count 12 `
  --device cuda
```

新 CSV 会包含正确的共振诊断指标。

### 第二步：按峰宽分组统计

```powershell
python analyze_validation_by_width.py `
  --metrics-csv ./validation_results/base_v2_peak_metrics_corrected/validation_metrics.csv `
  --output-dir ./validation_results/base_v2_width_groups
```

重点比较：

```text
m*gamma < 0.10
0.10 <= m*gamma < 0.20
0.20 <= m*gamma < 0.50
m*gamma >= 0.50
```

### 第三步：检查物理可辨识性

```powershell
python diagnose_peak_identifiability.py `
  --pool-dir ./truth_pool_val_10k `
  --num-samples 5000 `
  --integration-points 128 `
  --noise-level 0.09 `
  --neighbors 8 `
  --max-pairs 100 `
  --output-dir ./diagnostics/peak_identifiability
```

查看：

```text
diagnostics/peak_identifiability/ambiguous_pairs.csv
```

若出现 `g_relative_difference < 0.09`，但 `f_relative_difference` 很大的样本对，说明在 9% 噪声水平下存在实际不可区分的谱。

## 之后才做的小规模训练

不要直接跑完整 1.6 亿池。先制作峰宽平衡的小池：

```powershell
python make_width_balanced_pool.py `
  --input-pool-dir ./truth_pool_smoke `
  --output-dir ./truth_pool_peak_balanced_100k `
  --per-bin 25000 `
  --width-edges 0,0.1,0.2,0.5,2.000001 `
  --seed 20260802 `
  --overwrite
```

然后做 `peak_grad` 小规模实验：

```powershell
python -u train_mc_parameter_pool_transformer_loss.py `
  --pool-dir ./truth_pool_peak_balanced_100k `
  --checkpoint-dir ./local_models/mc_transformer_peak_grad_v25 `
  --model-type transformer `
  --loss-profile peak_grad `
  --loss-normalization relative `
  --lambda-grad 0.05 `
  --lambda-peak 0.5 `
  --lambda-resonance 0.02 `
  --peak-alpha 5 `
  --min-resonance-visibility 0.10 `
  --min-width-grid-cells 1.0 `
  --peak-in-domain-only `
  --noise-start-level 0 `
  --noise-curriculum-steps 5000 `
  --noise-level 0.09 `
  --transformer-normalize-coordinates `
  --transformer-rms-normalize-io `
  --transformer-d-model 128 `
  --transformer-nhead 4 `
  --transformer-num-layers 4 `
  --transformer-dim-feedforward 256 `
  --batch-size 128 `
  --learning-rate 3e-4 `
  --weight-decay 1e-4 `
  --integration-points 128 `
  --target-global-step 20000 `
  --max-hours 6 `
  --checkpoint-every-steps 1000 `
  --log-every-steps 100 `
  --amp `
  --fresh
```

若 `peak_grad` 确实改善共振指标，再单独测试弱物理项：

```text
--loss-profile peak_pinn
--lambda-physics 0.01
--physics-target discrete-clean
```

不要先使用大的物理权重。

## 结果是否算改善

不能只看完整谱平均 NRMSE。至少要同时满足：

- `resonance_relative_l2` 下降；
- `resonance_center_error` 下降；
- `resonance_height_relative_error` 下降；
- `m*gamma < 0.2` 分组明显改善；
- 完整谱 NRMSE 和 `g vs clean` 没有明显恶化。

如果这些峰指标没有改善，就停止扩大训练规模，继续做物理可辨识性分析。
