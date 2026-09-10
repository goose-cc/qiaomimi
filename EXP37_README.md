# Exp37：continuous-identifiability-first 3P 数据实验

这一步**不训练网络**。目标是先把 3P 正式数据构造和物理可辨识极限定清楚，再决定是否继续 TCN。

自由参数：

- `a1`
- `m`
- `gamma`

固定：

- `a2 = 0.025`
- `a3 = 0`

恢复失败条件保持不变：

- `|Δa1| >= 0.005`
- `|Δm| >= 0.03`
- `gamma factor >= 1.2`

三者为 OR。

## 相比 Exp36-v2 的核心变化

Exp36-v2 是：

```text
48k candidates
→ grid alias 预筛
→ 先选 260
→ 再对 260 continuous refine
```

所以最后才发现有些被选状态的 continuous margin 比 grid 估计小很多。

Exp37 改成：

```text
48k candidates
→ verified grid alias
→ 参数空间分层预选约 2.9k 状态
→ 对整个 prebank 做 continuous direct-verified refinement
→ 用 refined margin 选最终数据
→ 同时强制 coverage / split coverage / g separation
```

也就是说，**最终 260 个状态是根据 continuous recovery-aligned margin 选出来的**，不是先选完再检查。

## 硬约束

正式默认：

- candidate = 48,000
- continuous prebank ≈ 2,916
- selected = 260
- train / val / test = 200 / 30 / 30
- min selected g separation >= 1.3 RMS-SNR
- parameter cover radius <= 0.35
- holdout→train normalized parameter distance <= 0.35
- holdout→train g distance <= 2.5 RMS-SNR（hard）
- 1.5 RMS-SNR 仍作为 preferred 值
- train 在 `a1 / m / log(gamma)` 每个轴的 8 个 bins 中，每个 bin 至少 2 个状态
- val/test 每个参数轴 span fraction >= 0.5

如果所有 continuous-margin cutoff 都无法同时满足这些条件，程序会**停止并写出 selection_failure.json**，而不是偷偷放宽约束。

## Jacobian diagnostic

`analyze_exp37_jacobian.py` 对最终 selected states 计算：

- 3P `sigma_min`
- 3P condition number
- 2P `(a1,gamma)` condition number 作为对照
- `cos(J_a1, J_m)`
- `cos(J_a1, J_loggamma)`
- `cos(J_m, J_loggamma)`
- 最小奇异向量，即 3P near-null compensation direction

Jacobian 三列按实际 recovery tolerance 缩放：

```text
a1: 0.005
m: 0.03
log(gamma): ln(1.2)
```

并再按 reference-noise RMS 归一化。

## VarPro physics diagnostic

`analyze_exp37_varpro_ceiling.py` 不用于网络训练。

它在最终 test physical states 上：

```text
g
→ grid search over (m, log gamma)
→ analytic a1 profile
→ continuous refinement
→ direct float64 verification
```

分别评估：

- 0% noise
- 0.2% noise
- 1% noise

目的：

- 如果 VarPro 在 0.2% 下也明显失败，说明当前 g 本身不足以稳定支撑目标 3P 精度；
- 如果 VarPro 很好而 TCN 后面很差，才说明主要是网络问题。

## 运行

先 smoke test：

```powershell
.\run_exp37_3p.ps1 -Quick -RebuildData
```

正式：

```powershell
.\run_exp37_3p.ps1 -RebuildData
```

只想先构造数据 + validation + Jacobian，不跑 VarPro：

```powershell
.\run_exp37_3p.ps1 -RebuildData -SkipVarPro
```

## 正式结果优先发这些 CSV

```text
data_exp37_3p/
    exp37_data_summary.csv
    continuous_threshold_scan.csv
    selected_refined_margin_scan.csv
    profiled_alias_selected.csv
    coverage_summary.csv
    postbuild_validation.csv
    jacobian_summary.csv
    varpro_summary.csv
```

如果构造失败，发：

```text
continuous_threshold_scan.csv
selection_failure.json
```

## 依赖

只依赖项目原有：

- `mc_physics.py`
- `mc_pool_config.py`

不依赖 Exp35/Exp36 的数据文件，也不会重新使用旧 3P NPZ。
