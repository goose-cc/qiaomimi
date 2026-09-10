# Exp38：重新设计 g(q²)，先把 3P 信息量拉开

目标保持不变：

- a1 recovery tolerance: `|Δa1| <= 0.005`
- m recovery tolerance: `|Δm| <= 0.03`
- gamma recovery tolerance: factor `<= 1.2`
- reference noise: `0.2%`

Exp38 **不改谱函数和 forward kernel**。只改变 q² 的观测位置、覆盖范围和观测点数，也就是改变 g 向量的构造。

## 为什么先不训练网络

Exp37 已经证明当前 `q² = [-100,-6], N=100` 的 3P information content 不够。Exp38 先比较多种 g construction，找到能显著提高 continuous recovery-alias separation 的方案。之后再用 Exp37 的 continuous-first selection 去删除剩余近简并状态，最后才训练 TCN。

## 设计族

配置文件同时包含：

- 原始 baseline `[-100,-6], 100 points`
- 只把远端范围拉宽的 conservative designs
- 增加点数但保持原范围的 design
- 把 q² 上端向 0 靠近的 designs
- far + near 的 hybrid / nonuniform designs

注意：`q² > -6` 的设计超出了旧 observation domain。代码会在 CSV 中显式标记 `extends_above_original_q2_max=1`。它们是 Exp38 的“观测设计候选”，需要与你们物理设定允许的 q² 区域一致后才能作为最终正式方案。

## 两阶段运行

先 smoke test：

```powershell
.\run_exp38_g_design.ps1 -Quick
```

正式 coarse scan：

```powershell
.\run_exp38_g_design.ps1 -CoarseOnly
```

输出：

```text
exp38_design_scan/
  g_design_coarse_scan.csv
  coarse_finalists.json
```

coarse scan 用完全相同的 Sobol parameter anchors 比较每个 g design 的：

- recovery-tolerance-scaled Jacobian smallest singular value
- Jacobian condition number
- a1/m/log(gamma) derivative correlation
- verified grid recovery-alias RMS-SNR margin
- `sqrt(Nq) * RMS-SNR` Mahalanobis separation proxy

然后对 shortlist 中某个 design 做 continuous direct-verified audit：

```powershell
.\run_exp38_g_design.ps1 -DesignId hybrid_near_200
```

输出：

```text
exp38_design_audit/
  <design>_audit_summary.csv
  <design>_continuous_margin_scan.csv
  <design>_profiled_alias.csv
  <design>_q2.npy
```

这里真正搜索完整连续 `(a1,m,gamma)` 的 recovery-aligned aliases，而不是只比较离散 selected states。

## 95% 目标怎么用

`Mahalanobis ≈ 3.3` 只作为“两个 Gaussian 均值二选一时约 95% 可区分”的参考尺度，不把它当成最终网络 accuracy 定理。

我们下一步选择 g design 时要看：

1. continuous alias separation 是否相对 baseline 大幅增加；
2. 足够多 parameter cells 中是否还有 safe states；
3. 然后对最佳设计构造正式 48k → continuous screening → 260 states；
4. 用 VarPro 在 0.2% noise 下确认 a1/m/gamma 都能达到约 95% recovery ceiling；
5. 只有物理 ceiling 过关后再训练 TCN。

因此不会通过“放宽 recovery threshold”制造 95%，而是优先真正增加 g 的信息量并删除剩余近简并状态。
