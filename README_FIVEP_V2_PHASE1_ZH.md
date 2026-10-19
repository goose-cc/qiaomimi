# 五参数 V2 第一阶段：真实 q² 核对、候选近简并审计、成对 Pilot

**当前阶段不是正式 V2 数据集，也不涉及神经网络训练。** 用新的文件名生成结果，不覆盖 V1，也不修改 `mc_physics.py`、`mc_pool_config.py`、`build_fivep_dataset.py` 和 `build_fivep_blind.py`。

## 1. 为什么不能直接执行旧生成器 `--states 12000`

旧版 `build_fivep_blind.py` 仍强制训练状态彼此满足 `D_g >= 6`，这只适合小型离散码本；会删掉连续回归中应当保留的正常参数近邻。它及相关 `q2_grid` 逻辑还存在写死 100 点的情况。**本阶段不修改旧 V1 程序，而是新增独立预检/Pilot。**

历史 `cp02_corrected_3p_config.json` 中 `hybrid_ultranear_240` 采用 48 点线性段和 192 点几何段，两段在 `q²=-20` 重复。`cp02_observation.make_q2` 会去重，因此该配置计算出 **239 个不同点**，实际范围 `[-150,-0.05]`，与旧 V1 的 `[-100,-6]` 不同。这是根据该配置的推算，正式 V2 必须以成功数据集中的真实 `q2.npy` 为唯一输入；不能按标签再生成近似网格或补点。

## 2. Windows/PowerShell 核对 q²

从项目根目录执行：

```powershell
cd D:\big_create\calc-k-gamma
python -c "import numpy as np; q=np.load(r'data_cp02_ml_hybrid_domain_v2\q2.npy',allow_pickle=False); print(q.shape,q.dtype,q.min(),q.max())"
Get-FileHash .\data_cp02_ml_hybrid_domain_v2\q2.npy -Algorithm SHA256
```

核对历史数据元信息中的 `q2_sha256`。注意旧元信息可能记录 **数组 float64 原始值的 SHA-256**，而 `Get-FileHash` 是 **`.npy` 文件字节 SHA-256**：这两个算法针对的字节不同。新脚本同时记录两者，避免误判；若历史目录的 `metadata.json` 包含 `q2_sha256`，还会自动核对该字段，发现不一致就停止。

**本项目 5P V2 的 q² 设计变更必须写入记录，不能宣称观测范围与 V1 完全未变。** 其余五参数边界、谱密度非负性、0.2% RMS 噪声及 3P recovery 容差不改。

## 3. 先执行小型端到端预检（不训练网络）

请把以下四个新增文件放在当前仓库根目录：

- `fivep_v2_prepare.py`
- `validate_fivep_v2_pilot.py`
- `test_fivep_v2_prepare.py`
- `README_FIVEP_V2_PHASE1_ZH.md`

```powershell
python -m unittest -v test_fivep_v2_prepare.py

python .\fivep_v2_prepare.py audit `
  --q2-file .\data_cp02_ml_hybrid_domain_v2\q2.npy `
  --states 256 --bins 5 --neighbors 24 `
  --out .\v2_candidate_smoke_256

python .\fivep_v2_prepare.py pilot `
  --audit-dir .\v2_candidate_smoke_256 `
  --q2-file .\data_cp02_ml_hybrid_domain_v2\q2.npy `
  --states 100 --bins 5 `
  --out .\v2_paired_pilot_smoke_100

python .\validate_fivep_v2_pilot.py `
  --data .\v2_paired_pilot_smoke_100 `
  --q2-file .\data_cp02_ml_hybrid_domain_v2\q2.npy
```

输出目录已存在时程序拒绝覆盖，换新的 `--out` 目录即可。测试用的历史网格来自源码配置临时构建的 **测试夹具**，不是声称读取了你电脑上的成功数据文件。请务必在本机核对真实哈希。

## 4. 正式候选池审计（建议 65,536 个合法状态）

小预检通过后，建议：

```powershell
python .\fivep_v2_prepare.py audit `
  --q2-file .\data_cp02_ml_hybrid_domain_v2\q2.npy `
  --states 65536 --bins 10 `
  --neighbors 64 --projection-dim 12 `
  --out .\v2_candidate_audit_65536
```

输出：

| 文件 | 作用 |
| --- | --- |
| `candidate_parameters.npy` | 65,536 个物理合法且唯一的候选物理状态 |
| `raw_candidate_parameters.npy` | 物理有效性筛选前的 Sobol 候选（含边界锚点） |
| `physical_accepted_parameters.npy` | 全部通过连续谱非负性筛选的候选 |
| `candidate_g_clean.npy` | 使用历史 q² 的无噪声真实正演 |
| `nearest_g_neighbors.csv` | 每个状态检索到的最近 g 邻居及五参数差异 |
| `alias_threshold_counts.csv` | 针对 `D_g<1,2,3,4,6` 的近邻与回收冲突统计 |
| `coverage_candidate.csv` | 5 参数分 bin 的 raw candidate / accepted 数量 |
| `audit_report.json` | 所有参数、噪声、真实 q² 的两种 SHA-256、检索方法与注意事项 |

注意：为避免 `N²` 距离矩阵，近邻检索先用 SVD 投影和 kNN 建立候选短名单，再在完整 q² 上计算**精确 D_g**，并取短名单最近邻。这**不是对全部可能参数状态的完整近简并认证**：`D<t` 数量是短名单内找到的状态数，不是全候选集的完整 pair 计数。若投影残差偏大、需要对不存在隐藏 alias 作强保证，请增加近邻数、投影维度并补充多起点连续优化；不要因为这份审计没找到 alias 就宣称可恢复。

* `D_g=||g_i-g_j||₂ / max(σ_i,σ_j)`，`σ=0.002 RMS(g_clean)`。
* `physical_near_5pct` 是诊断性“正常局部邻居”指标，不是硬筛选。
* `recovery_gate_conflict` 表示 `a1/m/gamma` 至少一个真值差超过对应单样本容差。
* `recovery_acceptance_windows_disjoint` 是更严格的重叠窗口冲突见证（a1>0.010 或 m>0.060 或 gamma 比值>1.44）。
* 本阶段**不依据 `D_g<6` 删除候选，也不据此选正式训练状态**。

这些设置不改变旧的 recovery 容差。

## 5. 同批五参数状态的 100 点 / 历史 hybrid Pilot

```powershell
python .\fivep_v2_prepare.py pilot `
  --audit-dir .\v2_candidate_audit_65536 `
  --q2-file .\data_cp02_ml_hybrid_domain_v2\q2.npy `
  --states 2000 --bins 10 --train-replicas 3 `
  --out .\v2_paired_pilot_2000

python .\validate_fivep_v2_pilot.py `
  --data .\v2_paired_pilot_2000 `
  --q2-file .\data_cp02_ml_hybrid_domain_v2\q2.npy `
  --forward-count 12
```

会生成：

```
v2_paired_pilot_2000/
    state_manifest.csv
    coverage_pilot.csv
    pilot_manifest.json
    uniform_100/
        q2.npy
        metadata.json
        normalization_train_only.npz
        noise_0p2pct/train.npz
        noise_0p2pct/val.npz
    hybrid_ultranear_240/
        q2.npy
        metadata.json
        normalization_train_only.npz
        noise_0p2pct/train.npz
        noise_0p2pct/val.npz
```

两套 Pilot 使用**完全相同五参数真值、同样的物理状态 Train/Val 划分和相同噪声比例**，仅 q² 位置与点数不同。Train 每个状态 3 份独立噪声，Val 1 份。为避免调参污染，**Pilot 暂不生成正式 Test**。`hybrid_ultranear_240/q2.npy` 在打包时逐字节复制自历史文件；`metadata.json` 同时记录原始文件 SHA-256 和原始数组值 SHA-256。

Pilot 的数据格式兼容项目中的 `train_fivep_compare_v2.py`：该训练程序根据 `q2.npy` 自动读取输入长度，并使用 Train-only 归一化。你作为构建方只需提供两套 Pilot 给负责网络的同学，比较固定 Train/Val、相同模型与随机种子时的物理 alias 指标和 Val recovery。

## 6. 后续才开始正式五参数 V2

**本补丁不生成最终正式 V2。** 需要先由团队根据 alias 报告确定处理规则，不能直接用旧版 `build_fivep_blind.py --states 12000 --min-distance 6`。

正式目标：Train 9,600–12,000 个 **独立**状态，Val 1,200–1,500，Test 1,200–1,500；Train 3 个 0.2% 噪声副本。可先准备 15,000–20,000 个 accepted Train master pool，用 1k/2k/5k/9.6k/15k 嵌套子集做 learning curve。Train/Val/Test 按物理状态分组，先冻结划分再加噪；Test 必须由独立种子产生，不得用 Test g 距离或 recovery 筛样。最终数据报告需补全 candidate/accepted/train/val/**test** 五参数全覆盖，Gamma log-bin，检查 a2≈0、a3 负/零/正区域和全部边界。

不要把 V2 Pilot 的 `alias_threshold_counts.csv` 解读成全定义域 identifiability PASS，也不要把数据完整性 PASS 解读为神经网络 Recovery PASS。
