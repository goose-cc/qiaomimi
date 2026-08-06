# 实验 2：输出谱从 100 点提高到 1000 点

本包只包含需要新增或覆盖的文件，不是完整项目。

## 分支基础

从已经完成的实验 1 分支创建：

```bash
git switch experiment/v2-shuffle-no-replacement
git switch -c experiment/v2-output1000
```

把本包文件复制到项目根目录：

- 覆盖 `train_mc_parameter_pool_transformer_loss.py`
- 覆盖 `mc_inverse_loss.py`
- 新增 `run_exp2_output1000.ps1`
- 新增 `test_exp2_output1000.py`

`TransformerInverse.py` 不需要修改：原类已经根据 `output_length` 动态建立查询网格，能够输出 1000 点。为了保持控制变量，本实验不改变 Transformer 结构。

## 本次核心修改

1. `input_points=100` 保持不变；`output_points=1000`。
2. 保留实验 1 的打乱后无放回采样、1.6 亿干净参数池、9% 动态噪声和原损失权重。
3. 新增逻辑 batch 内部微批次：逻辑 batch 仍为 64，默认每次只让 2 条样本进入 1000 点 Transformer；64 条的梯度累积完成后才更新一次优化器。因此不会因显存限制把有效 batch 改小。
4. 修正差分损失的网格尺度：1000 点时相邻差分约缩小 10 倍。代码将梯度、平滑和 TV 差分换算回 100 点参考网格尺度，避免 `lambda_grad=0.1` 的实际作用缩小约 100 倍。100 点条件下比例为 1，不改变原 V2 定义。

## 测试

```powershell
python .\test_exp2_output1000.py
```

应看到：

```text
EXP2 OUTPUT1000 TEST PASSED
model output shape: (1, 1, 1000)
```

## 显存冒烟测试

先使用独立 checkpoint 目录训练 20 步：

```powershell
.\run_exp2_output1000.ps1 `
  -PoolDir ".\truth_pool_v2_160m" `
  -CheckpointDir ".\model\v2_exp2_output1000_smoke" `
  -Mode fresh `
  -MaxSteps 20 `
  -MaxHours 0 `
  -MicroBatchSize 2
```

若 CUDA 显存不足，删除 smoke 目录后将 `-MicroBatchSize` 改为 `1`。若显存宽裕，可试 `4`，但正式实验确定后不要中途改变。

日志应显示：

```text
batch_size (logical): 64
micro_batch_size: 2
output_points: 1000
```

## 正式训练

```powershell
.\run_exp2_output1000.ps1 `
  -PoolDir ".\truth_pool_v2_160m" `
  -CheckpointDir ".\model\v2_exp2_output1000_160m" `
  -Mode fresh `
  -MaxSteps 30000 `
  -MaxHours 0 `
  -MicroBatchSize 2
```

本实验必须使用新 checkpoint 目录，不能从 100 点 checkpoint 续训。

## 验证

现有 `validate_mc_transformer.py` 会从 checkpoint 的 `args.output_points` 自动读取 1000，因此无需修改。验证时继续使用同一个 `truth_pool_val_10k`、同一个噪声水平和随机种子。
