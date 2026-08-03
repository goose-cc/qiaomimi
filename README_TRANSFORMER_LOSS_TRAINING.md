# 参数池 Transformer + PINN Loss 说明

## 1. 模型任务

```text
g_noisy(q²) -> Transformer -> f_pred(s)
```

这里：

```text
f_true(s) = 160000 * rho(s)/(s+400)^2
g_clean(q²) = 160000 * integral [rho(s)/(s+400)^2/(s-q²)] ds
```

输入是在 `g_clean` 上在线加入 9% RMS 高斯白噪声后的 `g_noisy`。

## 2. 为什么不能使用旧 losses.py

旧物理项使用 `1/(y-x)`，旧坐标范围也不同。本问题使用：

```text
1/(s-q²), s ∈ [0.1764,6], q² ∈ [-100,-6]
```

直接套旧 Loss 会出现积分核符号和坐标范围错误。本项目因此使用 `mc_inverse_loss.py`。

## 3. 默认 PINN Loss

```text
L = L_data + lambda_grad * L_grad + lambda_physics * L_physics
```

相对归一化形式：

```text
L_data = mean_i MSE(f_pred_i,f_true_i) / RMS(f_true_i)^2
L_grad = mean_i MSE(Df_pred_i,Df_true_i) / RMS(f_true_i)^2
L_physics = mean_i MSE(Kf_pred_i,g_ref_i) / RMS(g_ref_i)^2
```

离散物理矩阵：

```text
K[j,k] = trapezoid_weight[k] / (s[k]-q²[j])
```

推荐起点：

```text
lambda_grad = 0.1
lambda_physics = 0.1
lambda_smooth = 0
lambda_tv = 0
lambda_tikhonov = 0
```

真实谱可能有窄峰，不建议一开始加入强平滑或 TV。

## 4. physics-target

### clean

```text
g_ref = 高精度连续正演 g_clean
```

不要求模型拟合随机噪声，默认推荐。

### discrete-clean

```text
g_ref = K * f_true_100_points
```

与 100 点标签严格一致，适合研究极窄峰导致的表示冲突。

### noisy

```text
g_ref = g_noisy
```

会推动输出解释噪声，通常不推荐。

## 5. Loss profile

- `base`：监督相对 MSE。
- `mse_grad`：MSE + 一阶差分。
- `huber_grad`：Huber + 一阶差分。
- `pinn`：MSE + 一阶差分 + 物理项，默认推荐。
- 其他 profile 用于额外正则对照实验。

## 6. 重要限制

参数池允许非常小的 `m*gamma`。高精度积分可以看到极窄峰，但 100 点 `f_true` 可能看不到它。因此：

- `clean` 物理目标可能与 100 点标签冲突；
- 这不是简单调大 Loss 权重能解决的问题；
- 应先用独立验证比较 `g vs clean`、`g vs discrete-clean` 和表示缺口。

## 7. 训练入口

本次 PINN 必须运行：

```text
train_mc_parameter_pool_transformer_loss.py
```

`train_mc_parameter_pool_day.py` 只作为纯 MSE 基线保留。

完整命令见 `README_160M_PARAMETER_POOL.md`。
