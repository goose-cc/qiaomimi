# Exp36 Final：3P recovery-aligned data + TCN baseline

目标参数：

- `a1`
- `m`
- `gamma`

固定参数：

- `a2 = 0.025`
- `a3 = 0`

## 这版解决什么

旧 3P generalized alias 使用的 remote 条件是：

- `|Δa1| >= 0.03`
- `|Δm| >= 0.16`
- `gamma factor >= 2`

这适合检查“相差很远的参数有没有全局近简并”，但与网络真正的恢复标准不一致。

Exp36 的 primary recovery-aligned alias 改成：

- `|Δa1| >= 0.005`
- `|Δm| >= 0.03`
- `gamma factor >= 1.2`

三者为 OR。

旧的 remote 条件仍然计算，但只作为 `global_alias_diagnostic`。

## 数据构造

默认正式配置：

- 48,000 个 Sobol 3P candidates
- `a1 ∈ [0.05, 0.20]`
- `m ∈ [0.40, 1.20]`
- `gamma ∈ [0.01, 1.0]`，在 log-space 中处理覆盖
- 最终 260 个 physical states
- train / val / test = 200 / 30 / 30
- selected-state 最小 g separation = 1.3 reference-noise RMS-SNR
- reference noise = 0.002
- noise datasets = 0%、0.2%、1%

physical states 先 split，再加 noise。绝对不要 merge NPZ 后重新 random split。

## recovery-alias 计算

利用当前物理结构对 `(m, log10(gamma))` 建立 dense profile bank，
并对每个 profile point 解析求最优 `a1`。最终 selected states 再进行
continuous constrained refinement，并用 direct forward 复核 winning alias。

输出会同时保存：

- recovery-aligned alias margin
- global/remote alias margin
- 最近 continuous alias 的 `a1 / m / gamma`
- 对应 `Δa1 / Δm / gamma factor`

因此如果三参数仍然困难，可以直接定位是哪一种补偿方向。

## 训练

第一轮只跑 TCN baseline，不跑 MLP/PSPEC。

训练：

- `noise_0p2pct/train.npz`
- `noise_0p2pct/val.npz`

同一个最佳 checkpoint 最后分别测试：

- 0%
- 0.2%
- 1%

三个 training seeds 在 `exp36_3p_config.json` 中。

## 运行

先 smoke test：

```powershell
.\run_exp36_3p.ps1 -Quick -RebuildData
```

正式先只生成和验证数据：

```powershell
.\run_exp36_3p.ps1 -RebuildData -DataOnly
```

先检查：

```text
data_exp36_3p/
    exp36_data_summary.csv
    alias_threshold_scan.csv
    profiled_alias_selected.csv
    validation_summary.csv
    postbuild_validation.csv
    metadata.json
```

确认数据侧结果后，训练三个 TCN seed：

```powershell
.\run_exp36_3p.ps1
```

## 如何理解 identifiability warning

`fraction_refined_margin_ge_1` 表示有多少 selected states 在“超过恢复容差”
的替代解中，最近一个仍至少相隔 1 个 reference-noise RMS。

这个值低不代表数据文件损坏，因此不会让 dataset-integrity validation FAIL；
它代表物理逆问题本身仍可能是病态的，必须和 TCN 的参数恢复结果一起解释。

## 依赖

新增文件只依赖项目中现有：

- `mc_physics.py`
- `mc_pool_config.py`

不依赖旧 `data_pipeline/`，也不依赖队友生成的旧 3P NPZ。
