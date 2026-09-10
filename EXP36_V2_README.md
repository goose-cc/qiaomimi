# Exp36-v2 修正版：3P recovery-aligned data + TCN baseline

这是对 Exp36 alias 数值实现的修正版。物理公式、参数范围、恢复阈值和 TCN 都没有改变。

## 修复内容

1. alias profile 全程使用 float64，不再把 `y`/resonance 转成 float32 后做二次型抵消。
2. 每个 OR region 的 grid winner 都用显式残差 `||y-a1*R||^2` 重新验证。
3. continuous refinement 从 verified grid winner、最近合法边界点和二者中点做 multi-start。
4. 插值只用于提出候选；最终 margin 一律用 direct float64 forward 重算。
5. verified grid winner 永远保留为候选，因此强制 `refined margin <= verified grid margin`。
6. post-build validator 会独立重算 grid/refined alias margin、检查 trigger、参数范围和 unacceptable 条件。任何 `refined > grid` 的实质性异常直接 FAIL。
7. 新增 `exp36_forward64.py`：只是项目 forward 的 float64 镜像，不改变 `mc_physics.py`，并会验证 cast 回 float32 后与项目 forward 一致。

## 需要覆盖/新增

覆盖：
- `build_exp36_3p_dataset.py`
- `validate_exp36_3p_dataset.py`
- `exp36_3p_config.json`
- `run_exp36_3p.ps1`

新增：
- `exp36_forward64.py`
- `selfcheck_exp36_alias_numerics.py`

TCN 模型、训练和分析脚本不需要修改，但压缩包里附带原文件方便整套放置。

## 重要：旧数据必须删掉/重建

旧 `data_exp36_3p` 的 alias CSV 是由有数值问题的代码产生的，不能复用。直接运行：

```powershell
.\run_exp36_3p.ps1 -RebuildData -DataOnly
```

这会覆盖重建正式数据并立即跑严格验证。

先看：
- `exp36_data_summary.csv`
- `alias_threshold_scan.csv`
- `profiled_alias_selected.csv`
- `postbuild_validation.csv`

只有 `postbuild_validation.csv` 中 alias numerics 相关项目全部 PASS 后，再运行：

```powershell
.\run_exp36_3p.ps1
```

## recovery-aligned unacceptable alternative

- `|Δa1| >= 0.005` OR
- `|Δm| >= 0.03` OR
- `gamma factor >= 1.2`

reference noise = 0.002。


额外输出 `refined_selected_margin_scan.csv`。注意 `alias_threshold_scan.csv` 的 cutoff 只用于 grid preselection；最终物理可辨识性判断以 direct-verified continuous refined margin 为准。runner 会在每次运行前先执行数值 self-check。
