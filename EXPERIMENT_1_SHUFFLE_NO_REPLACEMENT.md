# 实验 1：打乱后无放回依次取 batch

## 实验目的

在原 V2 上只改变训练样本的读取顺序，检验更均匀的数据覆盖是否改善训练稳定性和验证误差。

## 唯一修改

原 V2 按 `parameters.dat` 的存储顺序连续读取。新实验采用：

1. 每个 `pool_cycle` 先随机打乱数据块顺序；
2. 进入一个块后，对块内所有行索引做无放回随机排列；
3. 按排列后的顺序依次组成 chunk 和 batch；
4. 一轮内每条样本恰好使用一次；
5. 全部样本用完后进入下一轮，并重新打乱。

这不是有放回随机抽样，因此一轮内不会重复，也不会遗漏。

## 没有修改

- 模型：原 V2 Transformer
- 输入点：100
- 输出点：100
- 噪声：9% RMS 高斯噪声
- 损失函数及权重
- 参数池内容和分布

## 为什么采用分块打乱

对 1.6 亿行生成完整 `int64` 随机排列约需 1.28 GB 内存。这里使用分块打乱：

- 块顺序随机；
- 块内顺序随机且无放回；
- 默认块大小 200,000，块内索引约占 1.6 MB；
- 仍保证每轮完整覆盖一次。

## 测试

```powershell
python test_shuffle_no_replacement.py
```

应看到：

```text
SHUFFLE WITHOUT REPLACEMENT TEST PASSED
unique rows in cycle 0: 1000
```

## 小规模训练

```powershell
.\run_exp1_shuffle_no_replacement.ps1 `
  -PoolDir ".\truth_pool_smoke" `
  -CheckpointDir ".\model\v2_exp1_shuffle_smoke" `
  -Mode fresh `
  -MaxSteps 100
```

## 正式训练

```powershell
.\run_exp1_shuffle_no_replacement.ps1 `
  -PoolDir ".\truth_pool" `
  -CheckpointDir ".\model\v2_exp1_shuffle" `
  -Mode fresh
```

续训：

```powershell
.\run_exp1_shuffle_no_replacement.ps1 `
  -PoolDir ".\truth_pool" `
  -CheckpointDir ".\model\v2_exp1_shuffle" `
  -Mode resume
```

日志必须显示：

```text
sampling_mode: shuffle
无放回打乱说明: ...
```

## 对结果的预期

这种采样方式主要改善样本覆盖均匀性、降低有限训练过程中的重复和遗漏，并可能让 loss 曲线更稳定。它不保证显著降低误差，也不会单独解决窄峰信息不足。是否提高拟合度必须与原 V2 在相同步数和同一验证集上比较。
