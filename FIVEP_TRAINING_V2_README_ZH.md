# 五参数训练 V2 诊断版

## 为什么改
32-state 强行过拟合实验显示：TCN 在约 800 epoch 可达到 99% joint Recovery，说明数据读取、五参数目标、TCN 前向/反向及 Recovery 计算没有根本错误。
V1 正式训练却只有 220 epoch，并启用 patience=35 的早停，实际最佳 checkpoint 多在 9–28 epoch；此外 V1 默认开启 AMP/TF32，并带 dropout/weight decay。对于本项目这种 0.2% 噪声、细微物理差异的逆问题，这套设置过早结束优化，不能据此判断五参数不可学。

## V2 做了什么
- 默认 1500 epoch；诊断阶段不早停。
- batch size 96，增加每 epoch 的参数更新次数。
- 默认诊断命令关闭 AMP 和 TF32，使用 FP32。
- 诊断阶段 dropout=0、weight_decay=0，先测“能不能学会”。
- 同时输出 Train 与 Val Recovery。
- 同时保存：best_val_loss、best_val_score、best_val_recovery、best_train_recovery、last 五类 checkpoint。
- 仍然只读取 Train/Val，绝不读取私有 Test。
- 先只跑 1 个 seed，等确定 180-state 的拟合/泛化性质后，再扩到 3 seeds。

## 运行
把 `train_fivep_compare_v2.py` 和 `run_fivep_compare_v2_diagnostic.ps1` 放项目根目录：

```powershell
.\run_fivep_compare_v2_diagnostic.ps1 -DataDir .\fivep_train_val_v1\data -Device auto
```

如果 PowerShell 拦截脚本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 跑完后优先发给我
`runs_fivep_compare_v2\diagnostic_summary.csv`

如果需要进一步定位，再发：
- `tcn\seed_20260920\history.csv`
- `tcn\seed_20260920\run_summary.json`
- `mlp\seed_20260920\history.csv`
- `mlp\seed_20260920\run_summary.json`

## 结果判读
1. Train TCN joint Recovery 高（例如 >90%），Val 仍很低：主要转向“180 个独立状态覆盖不足”，应扩独立 physical states。
2. Train 180-state 也长期上不去，但 32-state 能 99%：模型容量/优化与任务复杂度仍不足，先做容量/物理解耦，再决定数据规模。
3. Train 和 Val 都明显提升：说明 V1 的早停/精度/正则化配置是主要问题之一，再做 3-seed 正式比较。
