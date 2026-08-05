# V2 实验 1：全参数池随机采样

## 实验目的

只修改训练参数的读取方式，验证“打乱顺序、从整个参数池随机采样”是否优于原 V2 的顺序读取。

本实验保持以下内容不变：

- 模型：原 V2 `InverseTransformer1D`
- 输入点数：100
- 输出点数：100
- 噪声：训练时动态加入 9% RMS 高斯白噪声
- 数据集：`parameters.dat` 仍只保存无噪声参数
- 损失函数、学习率、网络宽度和层数：沿用 V2

唯一实验变量：

```text
原 V2：sequential，按 parameters.dat 的文件顺序读取
实验 1：random，每个预计算块从整个参数池独立随机抽样
```

随机模式采用有放回抽样。对 1.6 亿规模的参数池和 4096 大小的预计算块，块内重复概率很低；同时避免创建和保存一个 1.6 亿长度的全局排列。

## 代码改动

训练脚本新增：

```text
--sampling-mode sequential
--sampling-mode random
```

本实验分支默认值是 `random`，但正式命令仍建议显式写出：

```text
--sampling-mode random
```

随机索引由 NumPy RNG 生成。RNG 状态会随 checkpoint 保存并恢复，因此 `--resume` 后随机序列可以连续复现。

为了减少对超大 `memmap` 的无序磁盘读取，代码会先按随机索引排序读取，再恢复为原随机顺序；这不会改变送入训练的数据顺序。

## 建议 Git 分支

在原项目仓库中执行：

```bash
git switch v2
git switch -c experiment/v2-random-sampling
```

然后用本文件夹中的代码覆盖该分支工作区并提交：

```bash
git add train_mc_parameter_pool_transformer_loss.py \
        run_exp1_random_sampling.ps1 \
        test_random_sampling.py \
        EXPERIMENT_1_RANDOM_SAMPLING.md
git commit -m "exp1: random sampling from full parameter pool"
```

不要使用原 V2 的 checkpoint 目录。随机实验必须使用新的目录，避免把顺序训练和随机训练接在同一个实验中。

## 先做功能测试

```powershell
python test_random_sampling.py
```

然后用小参数池跑 20～100 步：

```powershell
.\run_exp1_random_sampling.ps1 `
  -PoolDir ".\truth_pool_smoke" `
  -CheckpointDir ".\model\v2_exp1_random_smoke" `
  -Mode fresh `
  -MaxSteps 100
```

日志开头必须出现：

```text
sampling_mode: random
随机采样说明: ...
```

训练日志中的：

```text
sample_progress=.../... 
```

表示累计随机抽样数量相对于参数池规模的等量进度，不代表所有行都已无重复地覆盖一遍。

## 正式训练

```powershell
.\run_exp1_random_sampling.ps1 `
  -PoolDir ".\truth_pool" `
  -CheckpointDir ".\model\v2_exp1_random" `
  -Mode fresh
```

第二天续训：

```powershell
.\run_exp1_random_sampling.ps1 `
  -PoolDir ".\truth_pool" `
  -CheckpointDir ".\model\v2_exp1_random" `
  -Mode resume
```

## 对照要求

与 V2 基准比较时，应使用相同的：

- 训练步数或训练样本数
- 验证集
- 随机种子
- 网络结构
- 噪声水平
- 损失权重

本实验暂时不要改为 1000 点、不要降低噪声、也不要加入 LSTM。确认随机采样的独立效果后，再建立下一个分支。
