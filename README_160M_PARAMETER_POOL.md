# 1.6 亿五维参数池：Transformer + PINN 分块训练

## 1. 这次的数据是什么

每条有效样本只在磁盘保存五个 `float32` 参数：

```text
a1, a2, a3, m, gamma
```

1.6 亿条参数的逻辑大小为：

```text
160000000 × 5 × 4 bytes = 3.2 GB（十进制，约 2.98 GiB）
```

训练时在线计算：

```text
参数 -> rho(s) -> u_scaled(s) -> g_clean(q²)
     -> 加 9% RMS 高斯噪声 -> Transformer -> u_pred(s)
```

不把 1.6 亿条长度为 100 的曲线全部落盘，因此磁盘占用远低于完整曲线数据集。

## 2. 物理定义

参数范围：

```text
a1 ∈ [0, 0.2]
a2 ∈ [0, 0.05]
a3 ∈ [-0.05, 0.05]
m  ∈ (0, 2]
gamma ∈ (0, 1]
```

谱函数：

```text
rho(s) = a1/pi * (m*gamma)/((s-m)^2+(m*gamma)^2) + a2*s + a3
```

网格范围：

```text
s  ∈ [0.1764, 6]
q² ∈ [-100, -6]
```

目标与观测：

```text
u(s) = rho(s)/(s+400)^2
g(q²) = integral u(s)/(s-q²) ds
```

训练采用相同固定缩放：

```text
u_scaled = 160000 * u
g_scaled = 160000 * g
```

在线噪声：

```text
g_noisy = g_clean + 0.09 * RMS(g_clean) * N(0,1)
```

参数生成阶段会剔除在整个研究区间内出现 `rho(s)<0` 的参数。

## 3. 文件职责

```text
mc_pool_config.py                       集中保存物理常数和 JSON 工具
mc_physics.py                           参数筛选、曲线和稳定正演
 generate_mc_parameter_pool.py          生成 memmap 参数池
check_mc_parameter_pool.py              检查参数池
mc_inverse_loss.py                      本问题专用 PINN Loss
train_mc_parameter_pool_transformer_loss.py  推荐训练入口
train_mc_parameter_pool_day.py          纯 MSE 基线入口
validate_mc_transformer.py              独立验证
```

## 4. 必做的 smoke test

### 4.1 Loss 单元测试

```powershell
python smoke_test_mc_inverse_loss.py
```

脚本会检查全部 Loss profile 的有限值、反向梯度和完美预测误差，最后应看到：

```text
all loss profiles passed
```

### 4.2 生成 1 万条小参数池

```powershell
python generate_mc_parameter_pool.py `
  --output-dir ./truth_pool_smoke `
  --num-truths 10000 `
  --candidate-batch-size 50000 `
  --overwrite
```

### 4.3 检查参数池

```powershell
python check_mc_parameter_pool.py `
  --pool-dir ./truth_pool_smoke `
  --sample-size 10000 `
  --require-complete
```

应看到 `CHECK PASSED`。

### 4.4 跑 100 步 Transformer + PINN

```powershell
python train_mc_parameter_pool_transformer_loss.py `
  --pool-dir ./truth_pool_smoke `
  --checkpoint-dir ./model/mc_transformer_pinn_smoke `
  --model-type transformer `
  --active-block-size 10000 `
  --precompute-chunk-size 1000 `
  --integration-points 128 `
  --batch-size 32 `
  --loss-profile pinn `
  --loss-normalization relative `
  --physics-target clean `
  --lambda-grad 0.1 `
  --lambda-physics 0.1 `
  --learning-rate 1e-3 `
  --weight-decay 1e-5 `
  --max-hours 0.1 `
  --max-steps 100 `
  --log-every-steps 10 `
  --checkpoint-every-steps 50 `
  --amp `
  --fresh `
  --require-complete-pool
```

CPU 测试时删掉 `--amp`，并可把 `--batch-size` 改成 8。

日志应包含：

```text
loss=... data=... grad=... phys=...
```

## 5. 正式生成 1.6 亿参数

首次生成：

```powershell
python generate_mc_parameter_pool.py `
  --output-dir ./truth_pool `
  --num-truths 160000000 `
  --candidate-batch-size 1000000
```

按 `Ctrl+C` 后会在当前候选批结束时保存。继续：

```powershell
python generate_mc_parameter_pool.py `
  --output-dir ./truth_pool `
  --num-truths 160000000 `
  --candidate-batch-size 1000000 `
  --resume
```

完成后检查：

```powershell
python check_mc_parameter_pool.py `
  --pool-dir ./truth_pool `
  --sample-size 100000 `
  --require-complete
```

建议预留至少 6 GB 空间，避免临时文件、检查点和文件系统开销导致空间不足。

## 6. 正式 Transformer + PINN 训练

```powershell
python train_mc_parameter_pool_transformer_loss.py `
  --pool-dir ./truth_pool `
  --checkpoint-dir ./model/mc_transformer_pinn `
  --model-type transformer `
  --active-block-size 200000 `
  --precompute-chunk-size 4096 `
  --integration-points 128 `
  --batch-size 64 `
  --loss-profile pinn `
  --loss-normalization relative `
  --physics-target clean `
  --lambda-grad 0.1 `
  --lambda-physics 0.1 `
  --learning-rate 1e-3 `
  --weight-decay 1e-5 `
  --max-hours 23 `
  --checkpoint-every-steps 2000 `
  --log-every-steps 100 `
  --amp `
  --fresh `
  --require-complete-pool
```

`active-block-size=200000` 只是每次读入内存的参数块，不代表只训练 20 万条。

第二天继续时使用同一条命令，仅把 `--fresh` 改为 `--resume`。修正版会核对模型、物理网格、Loss、学习率、batch size 和参数池，避免误把不同实验接在同一个 checkpoint 上。

## 7. 输出文件

```text
model/mc_transformer_pinn/
  latest_checkpoint.pth   完整断点：模型、优化器、随机状态、训练位置
  best_model.pth          纯模型 state_dict
  best_model_info.json    模型和 best score 信息
  training_summary.json   本次运行汇总
```

当前 `best_model.pth` 是按滚动训练 Loss 选择的，不是按独立验证集选择的。正式汇报必须另外做固定验证。

## 8. 建立独立验证池

```powershell
python generate_mc_parameter_pool.py `
  --output-dir ./truth_pool_val_10k `
  --num-truths 10000 `
  --candidate-batch-size 100000 `
  --seed 20260802
```

然后参考 `README_MODEL_VALIDATION_ZH.md` 验证。验证池不能参与训练。

## 9. clean 与 discrete-clean

- `--physics-target clean`：物理项对齐高精度连续正演，默认推荐。
- `--physics-target discrete-clean`：物理项对齐 `100` 点真值经过离散矩阵后的观测。

当 `m*gamma` 极小时，100 点输出可能无法表示窄峰，但高精度 `g_clean` 仍能看到它。这时两种目标会发生表示冲突，应通过独立验证中的 `clean_discrete_representation_gap` 判断，而不是盲目增大物理 Loss 权重。

## 10. Git 权重上传

`.gitignore` 已设置为：

- 忽略参数池和完整 `latest_checkpoint.pth`；
- 允许提交 `model/**/best_model.pth`、`best_model_info.json` 和 `training_summary.json`。

训练后执行：

```powershell
git add model/mc_transformer_pinn/best_model.pth
git add model/mc_transformer_pinn/best_model_info.json
git add model/mc_transformer_pinn/training_summary.json
git commit -m "Add 160M Transformer PINN model"
git push
```

权重过大时应使用 Git LFS。
