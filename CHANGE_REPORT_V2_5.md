# V2.5 修改记录

1. 保留 V2 `InverseTransformer1D` 自由曲线输出。
2. 删除项目包内 V3 参数化模型残留文件，避免误用。
3. `mc_inverse_loss.py` 新增 `peak_grad` 和 `peak_pinn`。
4. 训练 Loss 可接收真实五参数，仅用于合成训练的峰区监督。
5. 新增 `lambda_peak`、`lambda_resonance`、`peak_alpha` 和可见峰筛选参数。
6. 新增 0% 到 9% 的训练噪声课程。
7. 新增 `target_global_step`，避免续训步数误解。
8. `validate_mc_transformer.py` 新增共振分量指标，旧完整谱 argmax 标为 legacy。
9. 新增按峰宽平衡参数池工具。
10. 新增按峰宽汇总验证指标工具。
11. 新增观测近似相同但谱差异很大的歧义样本搜索工具。
12. 未修改原物理公式、参数范围、积分区间或最终噪声定义。
