# 五参数 V2 正式数据：历史 hybrid_ultranear_240 观测设计（真实 239 点）

## 本版本做什么

在你的五参数物理核不变、前三个 recovery 门槛不变的前提下，直接使用原 CP02 成功数据中的 `data_cp02_ml_hybrid_domain_v2/q2.npy`，生成 **Train 9,600、Val 1,200、Test 1,200 个独立物理状态**；Train 每状态 **3** 个相互独立的 0.2% RMS 噪声样本（共 **28,800** 条训练样本）。此工具只负责生成和验证数据，**不训练网络，不保证 recovery 达标**。

源码全部新增，不覆盖 `build_fivep_blind.py`、`build_fivep_dataset.py`、任何 V1 或原 3P 数据集。

## 冻结的物理条件和历史文件

- 参数顺序 `[a1,a2,a3,m,gamma]`；范围依次是 `[.05,.20]`、`[0,.05]`、`[-.05,.05]`、`[.40,1.20]`、`[.01,1.00]`。
- `gamma` 用 log-space Sobol 采样；`a1/a2/a3/m` 线性 Sobol 采样；五参数继续满足整个连续谱 `rho(s)>=0`。
- 噪声 `sigma=0.002*RMS(g_clean)`；Train 每状态三次独立噪声，Val/Test 每状态一次。额外保存无噪声和 1% 噪声 Test，使用相同 Test 物理真值。
- Recovery **只作为原项目不变的评估定义**：`|Da1|<=.005`、`|Dm|<=.03`、`Fgamma<=1.2`。**数据生成不等于通过 recovery**。
- 采用**实际历史 q² 文件原始字节**。该文件名含 “240”，但物理文件有 **239 个点**，范围 **[-150,-.05]**，不同于五参数 V1 的 `[-100,-6]`。这是本次在你确定要沿用 CP02 设计后明确采用的 q² 变更。
- 原始 `.npy` 文件 SHA-256 **2624D6CF24873A0E4C60F7BA19E2239A0A5EC961C04BADF8056E1F1568DC653D**。正式命令对字节 SHA 实施硬验证，不接受重建的近似文件；同时保存数组值的独立哈希。

## 训练筛选与盲测协议

从**已经完成的 `v2_candidate_audit_65536`** 读取 65,536 个物理合法候选及其高精度物理正向和近邻报告。请保留整个审计目录，包括 `candidate_parameters.npy`、`candidate_g_clean.npy`、`nearest_g_neighbors.csv`、`audit_report.json`、`q2.npy`、`raw_candidate_parameters.npy`、`physical_accepted_parameters.npy`。

对 **Train** 的风险规则暂定为：

> 仅当已检索到的近邻同时满足 `D_g<6` **且恢复接受窗口互不重叠**（`|Da1|>.010` 或 `|Dm|>.060` 或 `gamma factor>1.44`），才排除对应高风险候选。

不再实行所有 Train 样本之间的全局两两 `D_g>=6`，因此允许参数与观测都很接近的**正常局部邻居**。生成器还会对最终 Train 的投影 kNN 短名单做二次精确 `g` 距离检查；此近邻检索**并非穷举所有五维连续参数组合**，不能证明不存在隐藏 alias。**这是一个明示的保守筛选策略，不是已经由神经网络泛化实验论证的最优准则**。

先生成并冻结 Train，再用**不同的 Sobol 随机种子**独立生成 Val/Test；两者只基于物理合法性和预先声明的五参数 10-bin 覆盖配额筛选，**不得使用 Val/Test 的 g 距离、Recovery 或模型结果删除难样本**。物理状态唯一 ID 跨三集合严格隔离，先分组后加噪。

覆盖规则：在 Train、Val、Test 的每个参数 10 个 marginal bins 内至少 2% 独立状态，`gamma` 用 log-bin、其余用 linear bin；每个集合都包含五个参数的精确上下界，且不得引入违反 rho(s)≥0 的真值。配额相对自然的物理筛选后分布会有采样偏置，交付时应保留覆盖报告并在论文/报告说明。

## 第一步：安装新增脚本

只使用一种方式：

1. 解压 `fivep_v2_formal_hybrid_bundle.zip`，将 4 个 `.py/.md` 文件复制到 `D:\big_create\calc-k-gamma` 根目录；**或者**
2. 将 `.patch` 放在项目根目录，执行 `git apply --check .\fivep_v2_formal_hybrid.patch` 后再执行 `git apply .\fivep_v2_formal_hybrid.patch`。

**不要重复使用 ZIP 解压和 Git 补丁**。本版依赖已安装的 `fivep_v2_prepare.py`、`build_fivep_dataset.py`、`mc_physics.py`、`mc_pool_config.py`。

## 第二步：本机源码单元测试

```powershell
cd D:\big_create\calc-k-gamma
python -m unittest -v test_fivep_v2_hybrid.py
```

应输出 10 项测试通过。单元测试使用**单独生成的临时测试 q²**，不冒充历史文件；正式构建会重新检查历史 SHA。

## 第三步：确认原始文件和审计目录

```powershell
Get-FileHash .\data_cp02_ml_hybrid_domain_v2\q2.npy -Algorithm SHA256
Get-ChildItem .\v2_candidate_audit_65536
```

正确历史文件 SHA 为上方锁定值。若缺失审计数据，请先重新运行已经交付的 `fivep_v2_prepare.py audit`；不要只把 CSV 复制到审计目录，因为正式构建需要原始的 NPY 候选真值和 clean g。

## 第四步：直接构建正式 12,000-state 五参数 V2

```powershell
python .\build_fivep_v2_hybrid.py `
  --audit-dir .\v2_candidate_audit_65536 `
  --q2-file .\data_cp02_ml_hybrid_domain_v2\q2.npy `
  --train-states 9600 --val-states 1200 --test-states 1200 `
  --out .\data_fivep_v2_hybrid_12000
```

此命令可能消耗较多 CPU、RAM 和磁盘 I/O，建议保留足够剩余空间并让进程运行到结束。输出目录已存在则拒绝覆盖。**没有执行这个命令就不能声称已经生成正式 12,000 状态数据集**。

如果想更稳妥地本地预检，可先改用 `--train-states 900 --val-states 120 --test-states 120 --out .\data_fivep_v2_hybrid_smoke_1140`，用同一大候选池构建非正式小样本，然后再按上面的正式命令重新运行。

## 第五步：严格验证生成的正式数据

```powershell
python .\validate_fivep_v2_hybrid.py `
  --data .\data_fivep_v2_hybrid_12000 `
  --q2-file .\data_cp02_ml_hybrid_domain_v2\q2.npy `
  --forward-count 24
```

预期会看到：

```text
V2 HYBRID DATA INTEGRITY PASS | historical q2=239; exact original SHA-256 matched
Split states: {'train': 9600, 'val': 1200, 'test': 1200} | training samples: 28800
COVERAGE/BLIND SPLIT/PHYSICS/0.2% RMS/RECOVERY CONTRACT PASS
NOT RECOVERY CERTIFIED
```

使用真实模型 512 点独立积分重算部分样本，检查历史 q² 文件字节与数组值，10-bin Train/Val/Test coverage，独立物理状态，噪声参数，训练集专用归一化和原硬条件。

## 主要交付文件

```text
data_fivep_v2_hybrid_12000/
    q2.npy                          # 精确复制历史文件，实际 239 点
    state_manifest.csv              # 所有独立物理状态及 split 真值
    metadata.json                   # 所有参数、噪声、网格哈希、blind 与 alias 规则
    coverage_v2.csv                 # candidate/accepted/train/val/test 参数分箱统计
    normalization_train_only.npz
    noise_0p2pct/
        train.npz                   # gy=(28800,239), parameters=(28800,5)
        val.npz                     # gy=(1200,239), parameters=(1200,5)
        test.npz                    # gy=(1200,239), parameters=(1200,5)
    noise_0pct/test.npz
    noise_1pct/test.npz
    train_state_indices_1000.npy   # 嵌套 Train state 子集索引
    train_state_indices_2000.npy
    train_state_indices_5000.npy
    train_state_indices_9600.npy
```

训练侧 `train_fivep_compare_v2.py` 使用 `gy/parameters/physical_state_id` 和 `normalization_train_only.npz`，按 `q2.npy` 自动获取 239 维输入；本版实测训练侧**数据加载**成功，但没有运行网络训练。独立 Test 真值应只在模型冻结后验收；不要用它反复调模型。

## Git 交付

在已存在的 `fivep-dataset-v1` 分支里可直接添加**新增代码和说明**：

```powershell
git add -- build_fivep_v2_hybrid.py validate_fivep_v2_hybrid.py test_fivep_v2_hybrid.py README_FIVEP_V2_HYBRID_ZH.md
git diff --cached --name-only
git diff --cached --check
git commit -m "Add historical-hybrid 5P V2 formal dataset pipeline"
git push origin fivep-dataset-v1
```

**不要运行 `git add .`，不要把 `venv`、65536 候选临时数据或整个大数据集意外加入 Git**。数据集本身生成后再通过团队存储、Release 附件或受控交付包分发。因 V2 包含 Test 真值，如需严格盲测应独立保管测试答案。

## 风险声明

- 65,536 候选的近邻 CSV 只有短名单搜索，因此 Train alias 筛选非全域保证；低 g 距离也不等于自动失败。
- Train/Val/Test 边际 10-bin 覆盖不能证明五维全部联合参数区域充分覆盖。
- 239 点历史 q² 方案取代了 100 点 q² 方案；这属于观测条件改变，不能声称除了 a2/a3 一切物理观测都没变化。
- 数据完整性 PASS 只是证明 **数据可用、可读、物理正确、拆分符合规则**；不能推断未知 Val/Test Recovery 成功。
