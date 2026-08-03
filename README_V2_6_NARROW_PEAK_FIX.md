# V2.6：直接针对窄峰恢复的训练版

这一版不再增加大量诊断步骤，目标只有一个：让网络在当前物理问题下尽可能优先恢复窄峰。

## 核心修改

1. **输出网格从 100 点提高到 256 点**：原网格间距约 0.0588，256 点后约 0.0228。低于原网格宽度的峰不再全部挤在一两个点上。
2. **完整谱拆成两个模型分支**：
   - 背景分支只使用物理上正确的两个背景基函数；
   - 共振分支输出非负自由曲线，不直接回归五个参数，避免 V3 的平均参数塌缩。
3. **增加峰位置分布头**：用整条输出网格上的概率分布监督峰位置，不仅靠完整谱 MSE。
4. **Loss 直接监督**：背景、局部峰形、峰梯度、峰面积、峰中心和峰位置分布。
5. **不在第一阶段加入 PINN 项**：当前错误平滑谱也能得到很小的 g 误差，强物理项反而可能继续奖励平滑解。
6. **训练池主动偏向窄峰**：65% 样本位于 m*gamma=0.02~0.20，同时保留 15% 原始均匀分布样本，避免背景能力完全丢失。
7. **噪声课程**：前 10,000 步从 0% 逐渐提高到 9%。

## 分支

建议从当前已保存的 V2/V2.5 分支再建一个分支：

```powershell
git switch -c experiment/v2.6-narrow-peak
```

覆盖补丁后：

```powershell
python smoke_test_mc_inverse_loss.py
```

## 生成一个专门训练窄峰的小参数池

这不是诊断实验，而是正式的定向训练数据。它不会更改物理公式或参数允许范围。

```powershell
python generate_narrow_peak_training_pool.py `
  --output-dir ./truth_pool_narrow_v26_400k `
  --num-samples 400000 `
  --seed 20260803 `
  --overwrite
```

组成：

- 40%：0.02 <= m*gamma < 0.10
- 25%：0.10 <= m*gamma < 0.20
- 15%：0.20 <= m*gamma < 0.50
- 5%：0.50 <= m*gamma < 1.80
- 15%：原始五参数独立均匀分布

## 正式进行一次聚焦训练

必须使用全新的 checkpoint 目录，不能加载旧 V2 权重，因为模型结构和输出点数已经改变。

```powershell
python -u train_mc_parameter_pool_transformer_loss.py `
  --pool-dir ./truth_pool_narrow_v26_400k `
  --checkpoint-dir ./local_models/mc_transformer_narrow_v26 `
  --model-type transformer_peak `
  --loss-profile narrow_peak `
  --input-points 100 `
  --output-points 256 `
  --integration-points 512 `
  --active-block-size 200000 `
  --precompute-chunk-size 2048 `
  --batch-size 32 `
  --learning-rate 2e-4 `
  --weight-decay 0 `
  --grad-clip 1.0 `
  --transformer-d-model 96 `
  --transformer-nhead 4 `
  --transformer-num-layers 4 `
  --transformer-dim-feedforward 256 `
  --transformer-dropout 0.05 `
  --lambda-grad 0.05 `
  --lambda-peak 0.25 `
  --lambda-background 0.20 `
  --lambda-peak-shape 1.0 `
  --lambda-peak-grad 0.05 `
  --lambda-peak-area 0.25 `
  --lambda-peak-center 0.05 `
  --lambda-peak-distribution 0.20 `
  --peak-alpha 20 `
  --min-resonance-visibility 0.03 `
  --min-width-grid-cells 0.25 `
  --noise-start-level 0 `
  --noise-curriculum-steps 10000 `
  --noise-level 0.09 `
  --target-global-step 30000 `
  --max-hours 23 `
  --checkpoint-every-steps 2000 `
  --log-every-steps 100 `
  --best-window 200 `
  --amp `
  --fresh `
  --device cuda
```

第一阶段不要使用 `narrow_peak_pinn`，也不要设置平滑、TV 或 Tikhonov 项。这些正则化会压低尖峰。

## 只做一次必要验证

```powershell
python -u validate_mc_transformer.py `
  --validation-pool-dir ./truth_pool_val_10k `
  --checkpoint-dir ./local_models/mc_transformer_narrow_v26 `
  --weights latest `
  --num-samples 10000 `
  --batch-size 128 `
  --seed 20260802 `
  --noise-level 0.09 `
  --output-dir ./validation_results/narrow_v26_30k `
  --plot-count 12 `
  --device cuda
```

V2.6 的 `resonance_*` 指标直接使用模型显式输出的共振分支，不再使用“完整预测减真实背景”的诊断残差。

## 是否有效只看这些结果

与旧 V2 的同一验证池比较：

- m*gamma < 0.10 的共振平均误差是否从约 91% 明显下降；
- 0.10~0.20 组是否从约 85% 明显下降；
- 峰高误差是否下降；
- 完整谱平均误差不要明显高于原来的 23.76%；
- g 误差允许略有增加，不要求继续维持最低，因为当前 g 误差小并不代表峰正确。

## 物理边界

这一版会尽最大可能纠正“模型和训练目标偏向平滑背景”的问题，但不能保证恢复所有超窄峰。宽度显著小于 256 点网格间距约 0.0228、且其 g 差异远低于 9% 噪声的峰，仍可能没有足够观测信息得到唯一解。此时模型最多恢复统计上更合理的峰，而无法保证逐样本精确宽度。
