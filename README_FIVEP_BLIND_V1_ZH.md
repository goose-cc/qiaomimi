# 五参数数据交付：独立未知状态盲测 V1

## 为什么需要这份补充生成器

已上传 `build_fivep_codebook.py` 第 203–206 行先对全部候选生成 `g`，再调用 `select_codebook` 同时选择 Train、Val、Test；`select_codebook` 第 108–116 行的 `nearest` 对所有已选状态更新，三个分组交替选样。因而之前的 Train/Test `physical_state_id` 确实隔离，但训练码本并非在不知道测试观测的前提下独立选出的。

旧数据和旧脚本保留：它们仍是有效的**已选有限码本**预实验，不代表出现了已训练的网络作弊。

## 新方案：训练优先、独立盲测

1. Train：只生成训练候选，用训练候选的 `g` 做 D>=6 筛选，满足上下界端点和各参数边际分箱。
2. 完全冻结 Train 后，再使用独立候选种子生成 Val、Test。Val/Test 仅根据**提前确定的参数边界和边际分箱**选样；选择时不计算其 `g`，也不查看 recovery 成绩。Train/Test `physical_state_id` 及五参数向量不可重复。
3. 所有状态选完后才对 Val、Test 正演生成 `g`，并生成独立 0.2% RMS 噪声。每套数据分组后加噪。
4. 验证器对 Train 的 g 距离执行 D>=6 硬筛查；Test 与 Train 的 g 距离只**报告、不筛除**，否则又会影响盲测独立性。
5. 只用 Train 计算归一化（输入均值标准差、目标 `log(gamma)` 变换）。

这可检验**未见过的参数状态**，但仍只能代表这个预先声明的离散筛选分布，不代表完整连续物理域的可辨识性，也不能预先保证网络 Recovery。

## 硬条件不变

参数顺序 `[a1, a2, a3, m, gamma]`：

- a1: [0.05, 0.20]
- a2: [0, 0.05]
- a3: [-0.05, 0.05]
- m: [0.40, 1.20]
- gamma: [0.01, 1.00]
- q²: [-100, -6], 100 点（默认均匀）
- 参考噪声：`sigma=0.002*RMS(g_clean)`
- Recovery：`|a1hat-a1|<=0.005`; `|mhat-m|<=0.03`; `max(gammahat/gamma, gamma/gammahat)<=1.2`。

`a2/a3` 仍是**连续候选采样**，没有强制 0.001 网格。如果图片中的 0.001 是强制采样步长，须在正式封版前单独确认并修改，不能用本脚本声称已满足。

## Windows PowerShell 操作

从项目根目录执行，先确认新 `.patch` 只新增文件：

```powershell
git apply --check .\fivep_blind_v1.patch
git apply .\fivep_blind_v1.patch
python -m unittest -v test_fivep_blind.py
```

接着用新目录生成完全独立的盲测实验，不覆盖原来的 `data_fivep_codebook_256`：

```powershell
python .\build_fivep_blind.py --states 256 --pool-size 4096 --boundary-each 24 --coverage-bins 4 --min-distance 6 --seed 20261012 --train-replicas 3 --out data_fivep_blind_v1
python .\validate_fivep_blind.py --data .\data_fivep_blind_v1 --forward-count 0
```

会打印 `BLIND HOLDOUT DATA PASS`，并生成 `blind_validation_report.json`。有 `train-test min` 低于 6 **不代表失败**，它是盲测真实风险的报告，不允许为了让其超过 6 删掉测试状态。

## 给网络训练同学的接口

- 输入：`data_fivep_blind_v1/noise_0p2pct/{train,val,test}.npz` 中 `gy`，形状 `[N,100]`。
- 标签：同一文件中的 `parameters`，形状 `[N,5]`，顺序 `[a1,a2,a3,m,gamma]`。
- 数据还有 `gy_clean`, `noise`, `physical_state_id` 等审计字段；**不得把 `gy_clean`、`noise`、`physical_state_id`、`parameters` 当成网络输入**。
- `normalization_train_only.npz`：`input_mean`, `input_std`, `target_mean`, `target_std`; 目标的第 5 个分量是 `log_gamma`，预测评估必须反变换成物理 gamma。
- 训练和调参只能用 Train/Val；测试集留作最后一次正式评估。任何调整筛选规则、种子或模型超参之后都不能继续将已查看结果的旧 Test 当作未使用的盲测集。
- 评价：a1、m、gamma 各自通过率及三条的联合通过率；如需使用 95% 为整体目标，应事先注明，这不是放宽逐样本 recovery 门槛。

## 验证范围与局限

本补丁经真实 `mc_physics.py` 做过 90 状态和 256 状态生成及独立积分验证；参数分箱、真值隔离、噪声与归一化由老验证器额外审计。测试包含拒绝覆盖已有目录、拒绝放宽分离门槛、拒绝修改 recovery 门槛、拒绝篡改盲测筛选声明。

如项目其他分支对物理核有新改动，请重新验证源码哈希一致性，不可仅复制旧验证结果。
