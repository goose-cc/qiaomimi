# Exp18A：纯物理 Variable Projection

## 目的

Exp17 已经说明：当 `a1,a2,a3,m` 使用真值时，0.2% 噪声下 physics profile 可以很好地恢复 `gamma`；但 TCN 对前四个参数的误差会污染后面的 `gamma`。

Exp18A 不再换网络，而是利用当前物理公式本身的结构，把 5 参数联合反演降维。

固定 `(m, gamma)` 后：

```text
g = a1 * G_res(m,gamma) + a2 * G_a2 + a3 * G_a3
```

因此对每个 `(m,gamma)` 候选点，直接从 100 点 `g` 做有界最小二乘，求该点最优的 `a1,a2,a3`。真正搜索的只有 `(m,gamma)` 两个非线性参数。

**不修改** `mc_online_physics.py`、`mc_parametric.py` 或 `mc_physics.py`。

## 新增文件

- `analyze_exp18_variable_projection.py`
- `run_exp18_variable_projection.ps1`

## 建议分支

```powershell
git switch -c exp18-variable-projection
```

## 先快速检查

```powershell
.\run_exp18_variable_projection.ps1 -Device cuda -Quick
```

## 正式运行

```powershell
.\run_exp18_variable_projection.ps1 -Device cuda
```

默认测试：

```text
0%
0.2%
1%
5%
```

如果只先跑最关键的 0.2%：

```powershell
.\run_exp18_variable_projection.ps1 `
  -Device cuda `
  -NoiseLevels "0.002" `
  -OutputDir ".\validation_results\exp18_noise0p2"
```

## 输出

```text
validation_results/exp18_variable_projection/
    exp18_summary.csv
    exp18_samples.csv
    exp18_metadata.json
    within_factor2_oracle_nuisance_1d.png
    within_factor2_full_varpro_2d.png
    full_varpro_median_gamma_factor_error.png
```

最先看：

```text
exp18_summary.csv
```

重点比较：

- `oracle_nuisance_1d`：真实 `a1,a2,a3,m` 已知，只扫描 `gamma`，作为物理上限；
- `full_varpro_2d`：所有参数未知，扫描 `(m,gamma)`，同时每个候选点重新拟合 `a1,a2,a3`。

重点指标：

- `median_gamma_factor_error`
- `within_factor_2`
- `m_normalized_abs_mean`
- `a1_normalized_abs_mean`
- `median_f_relative_l2`
- `median_g_clean_relative_l2`

## 结果怎么解释

如果 0.2% 下 `full_varpro_2d` 明显优于 Exp17 的 `pred_all`，说明 Exp17 的主要瓶颈确实是 TCN nuisance 参数预测误差，直接利用物理结构消去 `a1,a2,a3` 是有效方向。

如果 0.2% 下 oracle 很好但 `full_varpro_2d` 仍然很差，说明即使把线性参数 profile 掉，`m-gamma` 联合退化仍很强；这时继续换 TCN/ResNet 直接预测五参数就没有充分理由。

1% 或 5% 下性能恶化属于 Exp16 已经预期到的物理噪声极限，不应简单解释为 variable projection 算法失败。
