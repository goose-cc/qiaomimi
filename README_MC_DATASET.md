# 1.6亿真值＋在线9%噪声数据集代码说明

这套代码按照新的方案工作：五个参数在完整先验范围内做连续均匀蒙特卡洛采样，不设置离散参数点、不构造笛卡尔积；程序显式拒绝 `m=0`、`gamma=0`、连续研究区间内出现 `rho(s)<0` 的参数，以及产生 `NaN`、无穷大或无法安全保存的数值。每条有效真值只保存一次 `params`、`u(s)`、`g_clean` 和 `truth_id`。训练时根据 `master_seed + truth_id + noise_id` 在线生成1000组可复现的9% RMS高斯白噪声，因此不会落盘保存1600亿条带噪数组。

## 生成或修改的文件

- `dataset_config.py`：集中记录完整参数范围、物理区间、网格、噪声、随机种子、真值数和分块大小。
- `physics_core.py`：实现谱函数、连续区间非负性检查、`u(s)`、稳定正向积分和数值安全检查。
- `generate_truth_shards.py`：拒绝采样并生成 `truth_000000.npz` 等真值分块，支持断点续跑和按分块分配到多台机器。
- `online_noise_dataset.py`：PyTorch `IterableDataset`，读取真值分块并在线生成确定性的9%噪声批次。
- `check_truth_shards.py`：检查参数范围、连续非负性、真值编号、覆盖统计、正向积分复算和实际噪声比例。
- `main_online_integration_example.py`：说明如何把在线数据集接入现有 `main.py`。
- `gitignore_snippet.txt`：不向 GitHub 上传分块数据、模型权重和训练结果的忽略规则。

## 物理关系

代码使用此前已经确定的关系：

\[
\rho(s)=\frac{a_1}{\pi}
\frac{m\Gamma}{(s-m)^2+(m\Gamma)^2}+a_2s+a_3
\]

\[
u(s)=\frac{\rho(s)}{(s+400)^2}
\]

\[
g_{\mathrm{clean}}(q^2)=
\int_{0.1764}^{6}
\frac{u(s)}{s-q^2}\,ds,
\qquad q^2\in[-100,-6].
\]

对极窄共振峰，正向积分没有直接使用稀疏固定网格，而采用
\(z=\arctan((s-m)/(m\Gamma))\) 变量替换，把窄洛伦兹峰变成平滑积分，从而不需要人为提高 `m` 或 `gamma` 的下限。

连续非负检查也不是只看100个输出点。程序利用该函数导数的结构，检查区间端点及可能出现的内部局部最小值，防止粗网格漏掉局部负值。

## 先做小规模链路测试

在项目根目录执行：

```powershell
python generate_truth_shards.py --output-dir ./truth_shards_smoke --num-truths 1000 --truths-per-shard 100 --candidate-batch-size 1000 --forward-batch-size 64
```

验证：

```powershell
python check_truth_shards.py --root ./truth_shards_smoke --expected-truths 1000
```

在线读取两块真值、每条真值只使用前3个噪声编号进行训练链路测试时，在现有 `main.py` 中设置：

```text
max_shards=2
max_noise_ids=3
```

这只是 smoke test，不代表完整数据训练。

## 正式生成1.6亿条真值

单机命令：

```powershell
python generate_truth_shards.py --output-dir ./truth_shards --num-truths 160000000 --truths-per-shard 100000 --confirm-large-run
```

这会计划生成1600个分块：

```text
truth_shards/
    dataset_config.json
    truth_000000.npz
    truth_000001.npz
    ...
    truth_001599.npz
```

每个分块包含：

```text
params      [最多100000, 5]
u           [最多100000, signal_length]
g_clean     [最多100000, observation_length]
truth_id    [最多100000]
```

默认用 `float64` 计算真值、用 `float32` 保存曲线。按100点标签和100点观测估计，全部真值分块约需136 GB左右；改成 `--storage-dtype float64` 后约需264 GB左右，尚未计文件系统开销。运行前必须确认磁盘容量。

## 多节点或多进程分块生成

不同机器只生成互不重叠的分块区间。例如机器A：

```powershell
python generate_truth_shards.py --output-dir Z:\truth_shards --num-truths 160000000 --truths-per-shard 100000 --shard-start 0 --shard-stop 100 --confirm-large-run
```

机器B：

```powershell
python generate_truth_shards.py --output-dir Z:\truth_shards --num-truths 160000000 --truths-per-shard 100000 --shard-start 100 --shard-stop 200 --confirm-large-run
```

每个分块使用由 `master_seed` 和分块编号派生的独立随机流，所以同一分块可以重复生成，且不同机器不会依赖执行顺序。不要让两台机器同时写同一个分块编号。

## 训练时接入现有 main.py

原来的 `.npz` 全量数据读取方式需要改成：

```python
from torch.utils.data import DataLoader
from online_noise_dataset import OnlineTruthBatchDataset

train_dataset = OnlineTruthBatchDataset(
    root="./truth_shards",
    batch_size=64,
    noise_level=0.09,
    noises_per_truth=1000,
    master_seed=20260801,
    data_scale=160000.0,
    shuffle=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=None,
    num_workers=0,  # Windows先用0；集群环境再增加
)

for g_noisy, u_truth, truth_id, noise_id in train_loader:
    prediction = model(g_noisy.to(device))
    if prediction.ndim == 3 and prediction.shape[1] == 1:
        prediction = prediction[:, 0, :]
    loss = criterion(prediction, u_truth.to(device))
```

`batch_size=None` 不能省略，因为在线数据集已经返回完整批次。代码默认让一个完整迭代遍历每条真值的全部1000个噪声编号，因此它对应方案中定义的1600亿条逻辑样本。

## 可信度检查

完整生成后执行：

```powershell
python check_truth_shards.py --root ./truth_shards --expected-truths 160000000
```

报告包含各参数统计箱覆盖数、连续区间内最小 `rho`、随机真值的正向积分复算误差，以及在线噪声的实际RMS比例。没有验证集和测试集时，这些检查只能验证数据和物理一致性；训练损失本身不能证明对未知参数的泛化能力。

## 必须明确的数学限制

完整范围允许 `m*gamma` 任意接近零。代码能够利用变量替换稳定计算 `g_clean`，但任何固定长度的点值标签（例如100个 `u(s)` 点）都不可能保证解析任意窄的连续峰。当前代码仍按现有网络接口保存固定网格上的 `u(s)`，并额外保存五个真值参数，因此不会丢失生成该连续谱函数所需的信息。若以后要求精确恢复无限窄峰，应改为预测五个参数、预测分箱积分，或使用自适应输出表示；单纯增加到有限个固定网格点不能从数学上彻底解决这个边界问题。

## GitHub

把 `gitignore_snippet.txt` 中的规则合并进项目 `.gitignore`。GitHub只上传生成器、在线数据集和说明文件，不上传 `truth_shards/`。队友拉取代码后先做小规模测试，再根据分配的 `--shard-start/--shard-stop` 生成各自负责的真值分块。
