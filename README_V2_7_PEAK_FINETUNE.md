# V2.7：在原 V2 权重上保守强化可辨识峰

本版本不改变 Transformer 结构、不改变输入/输出点数、不改变物理公式，也不使用 V2.6 的双分支或 256 点输出。

核心目标：保住原 V2 已经较好的完整谱和正演结果，只对可见、可分辨峰增加小权重局部监督。

## 修改文件

- `mc_inverse_loss.py`
- `train_mc_parameter_pool_transformer_loss.py`
- `validate_mc_transformer.py`
- `smoke_test_mc_inverse_loss.py`

## 新增训练模式

```text
--loss-profile peak_finetune
```

总损失：

```text
full spectrum relative MSE
+ lambda_grad * gradient loss
+ lambda_peak * local peak-window loss
+ lambda_physics * forward consistency loss
```

默认保守权重：

```text
lambda_grad    = 0.02
lambda_peak    = 0.10
lambda_physics = 0.03
```

局部峰监督默认只用于：

```text
0.06 <= m*gamma < 0.50
resonance visibility >= 0.10
m 位于 s 输出区间内
```

峰窗口：

```text
|s-m| <= 3 * max(m*gamma, ds)
```

局部项仍比较完整 `f_pred` 和完整 `f_true`，不减真实背景，也不按很小的共振能量归一化，因此比 V2.6 更稳定。

## 新增从 V2 权重启动新实验

```text
--init-model-weights <path>
```

支持：

- `best_model.pth` 纯 state_dict
- `latest_checkpoint.pth`，自动取其中 `model_state_dict`
- 其他纯 state_dict 文件

只加载模型权重，不加载旧优化器、旧步数和随机状态。

## 第一步：覆盖文件并测试

```powershell
python smoke_test_mc_inverse_loss.py
```

应看到：

```text
all loss profiles passed
```

## 第二步：确认原 V2 权重路径

推荐使用原 V2 验证表现更好的 latest：

```powershell
Get-ChildItem ./local_models/mc_transformer_base_v2
```

通常使用：

```text
./local_models/mc_transformer_base_v2/latest_checkpoint.pth
```

## 第三步：从原 V2 权重微调 5,000 步

继续使用原始参数池，不先更换窄峰训练池。

```powershell
python -u train_mc_parameter_pool_transformer_loss.py `
  --pool-dir ./truth_pool_160m `
  --checkpoint-dir ./local_models/mc_transformer_peak_finetune_v27 `
  --model-type transformer `
  --loss-profile peak_finetune `
  --input-points 100 `
  --output-points 100 `
  --integration-points 512 `
  --active-block-size 200000 `
  --precompute-chunk-size 4096 `
  --batch-size 64 `
  --learning-rate 2e-5 `
  --weight-decay 0 `
  --grad-clip 1.0 `
  --transformer-d-model 64 `
  --transformer-nhead 4 `
  --transformer-num-layers 3 `
  --transformer-dim-feedforward 128 `
  --transformer-dropout 0.1 `
  --lambda-grad 0.02 `
  --lambda-peak 0.10 `
  --lambda-physics 0.03 `
  --peak-window-widths 3.0 `
  --peak-width-min 0.06 `
  --peak-width-max 0.50 `
  --min-resonance-visibility 0.10 `
  --physics-target discrete-clean `
  --noise-level 0.09 `
  --max-hours 23 `
  --max-steps 5000 `
  --checkpoint-every-steps 1000 `
  --log-every-steps 100 `
  --best-window 200 `
  --init-model-weights ./local_models/mc_transformer_base_v2/latest_checkpoint.pth `
  --save-step-models `
  --amp `
  --fresh `
  --device cuda
```

将 `--pool-dir` 改成你原 V2 实际使用的训练池目录。模型结构参数必须与原 V2 完全一致。

## 第四步：验证 1,000、2,000、... 步模型

新版验证脚本支持：

```text
--weights-file
```

例如验证第 1,000 步：

```powershell
python -u validate_mc_transformer.py `
  --validation-pool-dir ./truth_pool_val_10k `
  --checkpoint-dir ./local_models/mc_transformer_peak_finetune_v27 `
  --weights-file ./local_models/mc_transformer_peak_finetune_v27/model_step_00001000.pth `
  --num-samples 10000 `
  --batch-size 256 `
  --seed 20260802 `
  --noise-level 0.09 `
  --output-dir ./validation_results/v27_step_1000 `
  --plot-count 12 `
  --device cuda
```

其他步数只需替换文件名和输出目录。

## 停止标准

相对原 V2：

- `g vs clean mean` 超过 0.02：停止；
- 完整谱 `f relative L2 mean` 超过 0.25：停止；
- 峰误差没有改善：不继续延长训练；
- 优先选择独立验证结果最好的 step，不使用训练 rolling loss 自动判断最终模型。

## 重要限制

本版本重点强化 `m*gamma >= 0.06` 的可辨识峰。对于明显小于一个 100 点输出网格的超窄峰，不强制模型猜测唯一峰形。
