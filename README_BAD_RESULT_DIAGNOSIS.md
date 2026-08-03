# 当前 NRMSE≈0.59 的原因与重训方式

## 结论

当前模型并非只是训练时间短，而是发生了输入塌缩：网络基本忽略 `g(q²)`，输出接近训练集平均曲线。

证据：9% 噪声训练池和 0% 噪声独立池的 `f NRMSE` 都约为 0.594；去掉噪声没有改善。

## 根因

旧 Transformer 把 `q²=-100...-6` 直接输入线性位置嵌入，而 `g_scaled` 的典型 RMS 只有约 0.02。真正由 g 引起的 embedding 变化会被巨大位置 embedding 和 bias 淹没。

本版本新增：

- 坐标统一映射到 `[-1,1]`；
- 每条输入按自身 RMS 归一化，输出再乘回同一 RMS；
- source/target token LayerNorm；
- 默认学习率由 `1e-3` 降为 `3e-4`；
- checkpoint 保存并核对上述结构参数。

旧权重不能通过修改验证脚本变好，必须换新目录从头训练。

## 先运行尺度诊断

```powershell
python diagnose_mc_signal_scale.py `
  --pool-dir ./truth_pool_smoke `
  --num-samples 10000 `
  --device cuda
```

## 建议先做 base 对照实验

```powershell
python train_mc_parameter_pool_transformer_loss.py `
  --pool-dir ./truth_pool_smoke `
  --checkpoint-dir ./local_models/mc_transformer_base_v2 `
  --model-type transformer `
  --loss-profile base `
  --loss-normalization relative `
  --transformer-normalize-coordinates `
  --transformer-rms-normalize-io `
  --transformer-d-model 128 `
  --transformer-num-layers 4 `
  --transformer-dim-feedforward 256 `
  --batch-size 128 `
  --learning-rate 3e-4 `
  --weight-decay 1e-4 `
  --integration-points 128 `
  --max-steps 10000 `
  --max-hours 2 `
  --checkpoint-every-steps 1000 `
  --log-every-steps 100 `
  --amp `
  --fresh
```

先验证 base。只有当 base 的独立验证 NRMSE 明显低于旧模型后，再训练 PINN：

```powershell
python train_mc_parameter_pool_transformer_loss.py `
  --pool-dir ./truth_pool_smoke `
  --checkpoint-dir ./local_models/mc_transformer_pinn_v2 `
  --model-type transformer `
  --loss-profile pinn `
  --loss-normalization relative `
  --physics-target discrete-clean `
  --lambda-grad 0.05 `
  --lambda-physics 0.02 `
  --transformer-normalize-coordinates `
  --transformer-rms-normalize-io `
  --transformer-d-model 128 `
  --transformer-num-layers 4 `
  --transformer-dim-feedforward 256 `
  --batch-size 128 `
  --learning-rate 3e-4 `
  --weight-decay 1e-4 `
  --integration-points 128 `
  --max-steps 20000 `
  --max-hours 4 `
  --checkpoint-every-steps 1000 `
  --log-every-steps 100 `
  --amp `
  --fresh
```

`discrete-clean` 先让 PINN 正向算子与 100 点训练标签严格一致。待 base 和该 PINN 都能正常学习后，再单独比较 `clean` 物理目标。

## 验证

分别验证 `best` 和 `latest`，因为旧版的 best 只是最近训练 batch 的最低组合 loss，不是验证集最优：

```powershell
python validate_mc_transformer.py --validation-pool-dir ./truth_pool_val_10k --checkpoint-dir ./local_models/mc_transformer_pinn_v2 --weights best --num-samples 10000 --batch-size 256 --seed 20260802 --noise-level 0.09 --output-dir ./validation_results/pinn_v2_best --plot-count 12 --device cuda

python validate_mc_transformer.py --validation-pool-dir ./truth_pool_val_10k --checkpoint-dir ./local_models/mc_transformer_pinn_v2 --weights latest --num-samples 10000 --batch-size 256 --seed 20260802 --noise-level 0.09 --output-dir ./validation_results/pinn_v2_latest --plot-count 12 --device cuda
```

## 数据本身的第二个限制

`m*gamma` 可以无限接近 0，但输出只有 100 个 s 网格点，网格间距约为 0.0588。参数池中约一成样本的峰宽小于一个网格间距，这些极窄峰本身难以被 100 点标签稳定表示。先修复 Transformer 尺度问题；之后仍应按 `m*gamma` 分段报告 NRMSE，必要时提高输出点数或设置物理允许的最小峰宽。
