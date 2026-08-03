# 160M 参数池：物理参数化 Transformer + PINN（V3）

## 1. 为什么需要 V3

旧的自由曲线模型让 Transformer 直接输出 100 个 `f(s)` 点。普通 MSE 容易被大量平滑背景点主导，因此网络可以输出一条平滑平均曲线，丢失局部 Lorentz 共振峰，同时仍获得很小的正演 `g` 误差。

V3 不改变原物理问题，而是把已知的物理函数族写进模型输出层：

```text
g_noisy(q²)
  -> Transformer encoder
  -> 预测 a1, a2, a3, m, gamma
  -> 精确物理公式生成 f_pred(s)
  -> 稳定积分生成 g_pred(q²)
```

原公式保持不变：

```text
rho(s) = a1/pi * (m*gamma)/((s-m)^2+(m*gamma)^2) + a2*s + a3
f(s)   = rho(s)/(s+400)^2
g(q²)  = integral f(s)/(s-q²) ds
```

参数范围、`s`/`q²` 区间、`data_scale=160000` 和 9% RMS 高斯噪声均未改变。

## 2. 主要逻辑修正

1. **物理参数化输出**：模型预测五个物理参数，不再自由输出任意 100 点曲线。
2. **绝对幅度没有丢失**：网络同时输入 `g/RMS(g)` 和 `log10(RMS(g))`。
3. **高精度 PINN 正演**：参数化模型的 physics loss 直接由预测参数做稳定积分，不再受 100 点离散峰遗漏影响。
4. **峰专用监督**：增加归一化参数 loss 和 Lorentz 共振分量 loss。
5. **正确验证指标**：报告 `m`、`gamma`、`m*gamma`、共振分量误差，不再把完整曲线全局最大值错误地称为共振峰。
6. **独立验证选 best**：提供独立验证池后，`best_model.pth` 按固定验证指标保存，不再按随机训练 batch loss 保存。
7. **不可辨识峰参数降权**：当 Lorentz 分量相对整条曲线极弱时，`m` 和 `gamma` 的监督会按共振可见度自动降权，避免强迫网络从几乎没有峰信息的观测中猜参数。
8. **步数语义明确**：`--max-steps` 表示本次新增步数；`--target-global-step` 表示绝对总步数。
9. **Python 3.8 兼容**：核心训练和验证文件不依赖 `BooleanOptionalAction` 或 `X | None` 语法。

## 3. 旧权重不能续训

V3 的输出结构与旧的自由曲线 Transformer 不同。必须使用新的 checkpoint 目录和 `--fresh`：

```text
./model/mc_transformer_parametric_pinn_v3
```

不能从以下旧目录 `--resume`：

```text
mc_transformer_base_v2
mc_transformer_pinn_v2
mc_transformer_pinn_smoke
```

## 4. 先运行测试

```powershell
python smoke_test_mc_inverse_loss.py
python smoke_test_parametric_model.py
python smoke_test_160m_pipeline.py
```

应看到：

```text
all loss profiles passed
parametric model forward/backward passed
NumPy/Torch physics cross-check passed
```

## 5. 推荐 smoke 训练

训练池和验证池必须是不同目录：

```powershell
python -u train_mc_parameter_pool_transformer_loss.py `
  --pool-dir ./truth_pool_smoke `
  --validation-pool-dir ./truth_pool_val_10k `
  --checkpoint-dir ./model/mc_transformer_parametric_pinn_v3_smoke `
  --transformer-output-mode parametric `
  --loss-profile pinn `
  --physics-target clean `
  --lambda-grad 0.05 `
  --lambda-physics 0.01 `
  --lambda-parameter 1.0 `
  --lambda-resonance 1.0 `
  --lambda-nonnegative 0.1 `
  --parameter-loss-weights 1 1 1 2 2 `
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
  --resonance-grid-points 512 `
  --validation-samples 2048 `
  --validation-every-steps 1000 `
  --best-metric composite `
  --target-global-step 20000 `
  --max-hours 6 `
  --checkpoint-every-steps 1000 `
  --log-every-steps 100 `
  --amp `
  --fresh
```

`composite` 定义为：

```text
f_nrmse + 0.5 * resonance_relative + 0.25 * visibility_weighted_parameter_normalized_rmse
```

它同时关注整条曲线、共振峰和五个参数。

## 6. 续训到绝对 40000 步

使用绝对步数，避免再次出现“原有 10000 步又新增 20000 步，结果到 30000”的误解：

```powershell
python -u train_mc_parameter_pool_transformer_loss.py `
  --pool-dir ./truth_pool_smoke `
  --validation-pool-dir ./truth_pool_val_10k `
  --checkpoint-dir ./model/mc_transformer_parametric_pinn_v3_smoke `
  --transformer-output-mode parametric `
  --loss-profile pinn `
  --physics-target clean `
  --lambda-grad 0.05 `
  --lambda-physics 0.01 `
  --lambda-parameter 1.0 `
  --lambda-resonance 1.0 `
  --lambda-nonnegative 0.1 `
  --parameter-loss-weights 1 1 1 2 2 `
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
  --resonance-grid-points 512 `
  --validation-samples 2048 `
  --validation-every-steps 1000 `
  --best-metric composite `
  --target-global-step 40000 `
  --max-hours 6 `
  --checkpoint-every-steps 1000 `
  --log-every-steps 100 `
  --amp `
  --resume
```

## 7. 验证和图片

```powershell
python -u validate_mc_transformer.py `
  --validation-pool-dir ./truth_pool_val_10k `
  --checkpoint-dir ./model/mc_transformer_parametric_pinn_v3_smoke `
  --weights best `
  --num-samples 10000 `
  --batch-size 256 `
  --seed 20260802 `
  --noise-level 0.09 `
  --output-dir ./validation_results/parametric_pinn_v3_best `
  --plot-count 12 `
  --device cuda
```

图片位于：

```text
validation_results/parametric_pinn_v3_best/plots/
```

每个样本有三张图：

- `*_f_total.png`：完整 `f_true` 与 `f_pred`；
- `*_resonance.png`：仅 Lorentz 共振分量，能直接判断峰是否恢复；
- `*_g.png`：`g_noisy`、`g_clean` 和由预测参数正演得到的 `g_pred`。

## 8. 重点指标

优先查看：

```text
f NRMSE mean
resonance relative mean
parameter normalized RMSE
visibility-weighted parameter RMSE
m absolute error mean
gamma absolute error mean
width relative error mean
g vs clean mean
```

`full_curve_argmax_error` 只描述完整曲线最大点，不再把它解释为共振中心，因为线性背景可能让完整曲线在 `s=6` 最大。

## 9. 正式大参数池训练

确认 smoke 模型的共振图和参数误差明显改善后，再把：

```text
--pool-dir ./truth_pool_smoke
```

替换为：

```text
--pool-dir ./truth_pool
```

验证池仍然必须保持独立，不能参与训练。

## 10. 仍然无法保证的事情

参数化输出保证预测属于当前五参数物理函数族，但不能凭空创造观测中不存在的信息。若 `m*gamma` 极小，或 9% 噪声已经淹没区分不同峰参数的高阶信息，`m` 和 `gamma` 仍可能存在较大不确定性。此时应报告参数误差分布或不确定性，而不能只看 `g` 正演误差。
