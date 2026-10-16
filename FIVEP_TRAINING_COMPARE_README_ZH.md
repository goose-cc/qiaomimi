# 五参数 V1：TCN / MLP 同步对比训练

## 目的
只使用 `fivep_train_val_v1.zip` 中的 Train/Val，同时训练 TCN 与 MLP，保持私有 Test 完全未访问。

## 关键改动
- 目标从 3 维改成 5 维：`[a1,a2,a3,m,log_gamma]`（`log_gamma = ln(gamma)`）。
- TCN 沿用 CP02 amplitude-preserving v3 主干，输出改成 5 个独立 head。
- MLP 使用共享 backbone + 5 个独立 head。
- 两模型共用同一份 Train-only normalization、同一组 seeds、同一批 Train/Val。
- 训练损失默认对 5 个标准化目标等权 SmoothL1。
- 正式 Recovery 仍只计算 `a1/m/gamma/joint`；`a2/a3` 报 MAE/RMSE/P90/P95/correlation，不擅自定义 Recovery 门槛。
- checkpoint 规则：优先最大化 `min(R_a1,R_m,R_gamma)`，再最大化 `R_joint`，最后最小化 `val_loss`。
- 训练脚本**不会加载 Test**。

## 使用
先解压：

```powershell
Expand-Archive .\fivep_train_val_v1.zip -DestinationPath .\fivep_train_val_v1 -Force
```

然后把 `train_fivep_compare.py` 和 `run_fivep_compare.ps1` 放在项目根目录：

```powershell
.\run_fivep_compare.ps1 -DataDir .\fivep_train_val_v1\data -Device auto
```

或者直接：

```powershell
python .\train_fivep_compare.py --data-dir .\fivep_train_val_v1\data --models tcn mlp --seeds 20260920 20260921 20260922
```

## 输出
`runs_fivep_compare_v1/` 下会生成：
- `run_manifest.json`：冻结超参数、种子、Recovery 规则。
- `tcn/seed_xxx/best.pt`、`mlp/seed_xxx/best.pt`：各 seed 最佳权重。
- `history.csv`：逐 epoch Train/Val 轨迹。
- `val_predictions.csv`：五参数逐样本预测。
- `val_metrics.json`：单 seed 验证指标。
- `comparison_summary.csv/json`：TCN vs MLP 汇总对比。

## 注意
现在不要解压或读取 `fivep_private_evaluator_v1.zip`。先依据 Val 选择模型/超参数，冻结后再另写独立 evaluator 做最终 Test Recovery。
