Exp22 完整版：按“物理生成的 g 曲线间隔”做阶梯实验
======================================================

需要把本压缩包中的文件覆盖/复制到原项目代码目录。

本版本不直接人为修改 g 数值。所有 g 仍由原来的物理 forward 产生。
新增的是一个更直接的控制方式：先指定希望不同 g 曲线至少相差多少，再由程序自动寻找满足条件的最密参数网格。

默认实验：
  g 的 RMS 间隔 >= 0.4%、0.6%、1.0%（相对于两条曲线的平均 RMS 幅度）
  为了让 0.6% / 1.0% 仍可运行，候选网格允许最少保留 2 个 a1 或 gamma 锚点；
  这正好用于量化“g 拉得越开，参数分辨率会牺牲多少”。

在 0.2% 参考噪声下，这三档分别等价于：
  RMS-SNR >= 2、3、5

运行：
  powershell -ExecutionPolicy Bypass -File .\run_exp22_complete_g_gap_sweep.ps1

快速检查代码流程：
  powershell -ExecutionPolicy Bypass -File .\run_exp22_complete_g_gap_sweep.ps1 -Quick -Device cpu -Regenerate -Retrain

正式重新生成并训练：
  powershell -ExecutionPolicy Bypass -File .\run_exp22_complete_g_gap_sweep.ps1 -Regenerate -Retrain

主要结果：
  validation_results\exp22_g_gap_sweep\exp22_g_gap_sweep_teacher_summary.csv

其中最关键的列是：
  target_g_gap_percent
      设定的最小 g 曲线 RMS 间隔百分比。

  actual_min_g_gap_percent
      最终选中训练网格实际达到的最小 g 间隔。

  training_state_count / a1_anchor_count / gamma_anchor_count
      为了满足该 g 间隔，还能保留多少参数状态，反映“g 可分性”和“参数分辨率”的权衡。

  seen_0pct_gamma_within_x1p2
  seen_0p2pct_gamma_within_x1p2
  interp_both_0pct_gamma_within_x1p2
  interp_both_0p2pct_gamma_within_x1p2
      预测 gamma 与真实 gamma 相差不超过 1.2 倍的样本比例。

注意：
  本版本不调用 analyze_exp22_classification.py，也不使用“吸附到最近类别”的分类结果作为主要结论。
