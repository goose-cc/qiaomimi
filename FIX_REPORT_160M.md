# 1.6 亿参数池项目审查与修正记录

## 已发现并修正

1. **缺少 `mc_pool_config.py`**  
   原压缩包中的生成器、检查器和 `mc_physics.py` 都导入该文件，导致 `ModuleNotFoundError`。已补回并纳入测试。

2. **主 README 仍把纯 MSE 脚本当作正式入口**  
   `train_mc_parameter_pool_day.py` 内部是 `nn.MSELoss()`。已明确其仅为基线，并把正式命令改为 `train_mc_parameter_pool_transformer_loss.py`。

3. **物理常数存在重复和漂移风险**  
   `shift=400` 原来在多个文件硬编码。已加入 `PhysicsConfig.shift`，`mc_physics.py` 和训练入口统一读取配置。

4. **续训可能静默混合不同实验**  
   原代码只检查模型类。改变 Loss、物理网格、噪声、学习率、batch size 或参数池后仍可能续训。已增加关键参数和参数池一致性检查；不一致时要求新目录 `--fresh`。

5. **PyTorch 2.6+ 无法直接恢复完整 checkpoint**  
   新版 `torch.load` 默认 `weights_only=True`，完整断点含优化器和 NumPy 随机状态时会报 `UnpicklingError`。已对本项目自己生成的可信断点显式使用 `weights_only=False`，并保留旧版本兼容。

6. **PyTorch AMP API 产生弃用警告**  
   已优先使用 `torch.amp.GradScaler` 和 `torch.amp.autocast`，并保留兼容回退。

7. **验证噪声受设备和 batch size 影响**  
   已改成独立 NumPy 随机生成器；相同 seed 下 best/latest 可公平比较。

8. **验证缺少 RMSE/NRMSE**  
   已增加 `f_rmse`、`f_true_rms`、`f_nrmse_rms`。对等长曲线，`f_nrmse_rms` 与相对 L2 数值相同。

9. **把 m 直接当作完整曲线峰位置不严谨**  
   完整 `u(s)` 还包含线性背景和 `(s+400)^-2`。已把指标改成诊断性的 `true_argmax_to_m_error` 和 `pred_argmax_to_m_error`，并补充说明。

10. **`.gitignore` 规则相互冲突**  
   原文件先允许 `model/`，后面又重新忽略 `model/*.pth`。已改为允许提交 `model/**/best_model.pth` 及 JSON，继续忽略大体积 `latest_checkpoint.pth`。

11. **未使用的 Loss 项仍在每个 batch 计算**  
    原代码即使选择 `base` 也会计算物理矩阵、平滑、TV 和 Tikhonov 项，浪费 1.6 亿规模训练时间。已按 profile 只计算实际参与总 Loss 的项，日志中未使用项为 0。

12. **环境说明冲突**  
    固定的 NumPy/SciPy 版本不适合任意 Python 3.10+。文档已明确推荐 Python 3.10 或 3.11。

## 仍需注意

- `best_model.pth` 当前按滚动训练 Loss 选择，不是独立验证最优。
- 极窄峰可能无法由 100 点输出表示；应比较 `clean` 与 `discrete-clean`，不能仅靠加大物理权重。
- 1.6 亿参数池没有天然 train/val/test 划分，必须另外生成不同 seed 的验证池和测试池。
