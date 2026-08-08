# 实验七：物理参数可辨识性诊断

这不是继续堆网络，而是回答一个更基础的问题：

> 100 个干净的 g(q²) 是否包含足够信息，让网络恢复 a1、a2、a3、m、gamma？

训练时只使用 5 个参数的归一化 MSE，不使用完整谱 loss、峰 loss 或 physics loss。
这样可以避免“平滑谱 loss”干扰结论。

## 第一步：替换/新增文件

- 替换 `TransformerInverse.py`（原类完全保留，只新增 `ParametricInverseTransformer1D`）
- 新增 `train_mc_parameter_pool_parametric.py`
- 新增 `validate_mc_parametric.py`
- 新增 `run_exp7_parametric_noise0.ps1`
- 新增 `test_exp7_parametric.py`

## 测试

```powershell
python .\test_exp7_parametric.py
```

应看到 `EXP7 PARAMETRIC TEST PASSED`。

## 训练（第一阶段只用 0% 噪声）

```powershell
.\run_exp7_parametric_noise0.ps1 `
  -PoolDir ".\truth_pool_v2_160m" `
  -CheckpointDir ".\model\v2_exp7_parametric_noise0_160m" `
  -Mode fresh `
  -MaxSteps 30000 `
  -MaxHours 0
```

## 验证 0% 噪声

```powershell
python .\validate_mc_parametric.py `
  --validation-pool-dir ".\truth_pool_val_10k" `
  --checkpoint-dir ".\model\v2_exp7_parametric_noise0_160m" `
  --weights latest `
  --num-samples 10000 `
  --batch-size 64 `
  --noise-level 0.0 `
  --seed 20260802 `
  --output-dir ".\validation_results\v2_exp7_parametric_latest_val0" `
  --plot-count 12 `
  --device cuda
```

也建议再用 `--weights best` 跑一次。

## 如何解释

优先看 `m`、`gamma` 和 `m*gamma` 的误差，以及用预测参数解析重建出来的 f(s)。

- 如果 0% 输入下参数误差很小，而且解析重建能恢复窄峰：说明物理信息存在，之前主要是“直接预测 1000 点谱 + loss”的问题。
- 如果 0% 输入下 m/gamma 仍然很差：说明 100 点 g 对峰参数本身就缺乏稳定可辨识信息，问题更接近物理逆问题的极限。
- 如果 0% 好、5%/9% 急剧变差：说明无噪声可辨识，但噪声把峰参数信息淹没了。
