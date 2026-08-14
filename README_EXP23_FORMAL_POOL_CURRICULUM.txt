Exp23：正式 1.6 亿参数池阶梯 + 保参数覆盖的 g 拉开
=================================================

这版是针对 Exp22 的两个问题改的：

1. Exp22 虽然把 g 拉开后“见过的状态”很好学，但它靠减少参数 anchor 做到，
   0.6%/1% 时 gamma 最后只剩极少取值，连续参数覆盖被破坏。

2. Exp22 用的是小 MLP + 小网格，不是你原来的正式训练框架。

Exp23 现在改成：

    原 1.6 亿 parameters.dat
        -> 无放回抽连续参数候选
        -> 保证参数区间覆盖
        -> 在不破坏覆盖的前提下尽量选择 g 更远的样本
        -> 原项目 Transformer / BiLSTM+Transformer
        -> 1000 点 f(s)
        -> 原 PINN loss

也就是说，这次恢复到正式 g -> f 网络，不再用 Exp22 MLP。
默认网络是：

    InverseBiLSTMTransformer1D

你也可以用 -ModelType transformer 固定成纯 Transformer，但一个完整阶梯里不要中途换网络。

-------------------------------------------------
一、最重要的改动：g gap 不再是“硬筛选”
-------------------------------------------------

Exp22 的做法接近：

    不满足 g gap -> 删除状态

所以 g gap 越大，参数状态越少。

Exp23 改成：

    参数覆盖优先 > g gap 尽量拉开

每个当前自由参数会被分成若干连续区间（bin）。bin 只是采样 bookkeeping，
不是离散标签，也不是 gamma anchor。真正拿去训练的值仍然是 1.6 亿参数池里
连续的 float 参数值。除此以外程序还统计这些 bin 的“联合组合格子”，避免只覆盖
每个参数的边际范围、却长期只重复少数参数组合。

每个 batch：

1. 从大参数池无放回抽候选；
2. 先确保 a1/gamma/m/... 的各个区间都尽量有样本；
3. 在同样覆盖优先级的候选里，优先选 clean g 距离更大的；
4. 如果某个参数区间做不到目标 g gap，仍然保留这个区间；
5. 只在日志中记“target 没达到”，绝不为了 g gap 把这个区间删掉。

因此不会再发生：

    想要 1% g gap -> gamma 被砍到只剩 0.01 和 1

-------------------------------------------------
二、g 没有被人工改数值
-------------------------------------------------

所有 g 仍然由原物理 forward 生成。

代码只是从原参数池候选里“挑哪些样本进入当前 batch”，不做：

    g *= 常数
    g += 人工偏移
    手工改某个 q 点

所以物理问题没有被换掉。

-------------------------------------------------
三、训练和验证也改回正式方式
-------------------------------------------------

训练池：

    .\truth_pool

就是你原来的参数池，正式时可以是 160000000 行。

验证池：

    .\truth_pool_val_10k

必须是另外生成、没有参与训练的池。
验证集不会做 g-gap 筛选，因为我们就是要看模型能不能处理新的连续参数值，
不能只验证训练里挑出来的“容易状态”。

如果你还没有独立验证池：

python generate_mc_parameter_pool.py `
  --output-dir .\truth_pool_val_10k `
  --num-truths 10000 `
  --candidate-batch-size 100000 `
  --seed 20260815

-------------------------------------------------
四、第一步不要直接跑五级：先做 a1+gamma A/B
-------------------------------------------------

先 Quick 检查程序：

.\run_exp23_formal_pool_curriculum.ps1 `
  -Action compare `
  -PoolDir .\truth_pool `
  -ValPoolDir .\truth_pool_val_10k `
  -ModelType bilstm_transformer `
  -Device cuda `
  -Amp `
  -Quick

Quick 只检查代码，不拿结果写结论。

正式比较：

.\run_exp23_formal_pool_curriculum.ps1 `
  -Action compare `
  -PoolDir .\truth_pool `
  -ValPoolDir .\truth_pool_val_10k `
  -ModelType bilstm_transformer `
  -Device cuda `
  -Amp

会跑两组完全相同的正式 a1+gamma 模型：

A. coverage
   只保证参数覆盖，不主动优化 g 距离。

B. coverage_gdiverse
   参数覆盖仍然优先，但在不破坏覆盖时尽量把 g 拉开。

A/B 使用：

    同一大参数池
    同一独立验证池
    同一网络
    同一 loss
    同一噪声
    同一 batch
    同一训练步数

唯一关键差异就是：是否在覆盖约束内主动提高 g 可分性。

-------------------------------------------------
五、训练后先看哪个文件
-------------------------------------------------

    validation_results\exp23_compare_summary.csv

最重要的列：

1. parameter_bin_coverage_fraction

   每个自由参数的边际区间覆盖率。A/B 应该都很高、彼此接近。

2. mean_unique_joint_fraction_of_batch / final_global_joint_bins_coverage_fraction

   看参数“组合格子”是不是也在持续覆盖，而不是只重复少数组合。

3. mean_batch_min_g_gap_percent

   实际训练 batch 里，最近两条 clean g 的平均最小 RMS 相对间隔。
   B 应该明显高于 A，否则“拉 g”没有真正发生。

4. val_train_noise_f_relative_l2

   独立验证池上，预测 f 和真实 f 的相对 L2，越低越好。

5. val_train_noise_resonance_window_relative_l2

   只看真实共振峰附近窗口的恢复误差。
   这是本实验比全局 f error 更重要的指标之一，越低越好。

6. val_train_noise_peak_height_relative_error

   共振窗口内峰高相对误差，越低越好。

7. val_train_noise_physics_g_relative_l2

   把网络预测的 f 再正演回 g 后，与真实 clean g 的相对误差。

判断逻辑：

    A/B 参数覆盖差不多
    + B 的实际 g 间隔更大
    + B 在独立验证池的峰区/f 指标更好

才可以说：

    “不牺牲连续参数覆盖时，把 g 拉开确实帮助正式网络泛化。”

如果 B 的 g gap 大很多，但独立验证并没变好，说明：

    只挑 g 更远的训练样本，并不能解决原本连续逆问题。

-------------------------------------------------
六、A/B 证明有效以后再跑正式阶梯
-------------------------------------------------

.\run_exp23_formal_pool_curriculum.ps1 `
  -Action curriculum `
  -PoolDir .\truth_pool `
  -ValPoolDir .\truth_pool_val_10k `
  -ModelType bilstm_transformer `
  -Device cuda `
  -Amp

顺序：

    gamma
      -> a1 + gamma
      -> a1 + m + gamma
      -> all5_strong（a1 >= 0.05）
      -> all5_full

网络结构从头到尾不变。
后一级默认从前一级 best_model.pth 初始化，才是真正的 curriculum / 阶梯训练。

结果汇总：

    validation_results\exp23_curriculum_summary.csv

每一级目录：

    model\exp23_curriculum\<stage>\
        best_model.pth
        latest_checkpoint.pth
        validation_best.json
        training_summary.json
        selection_diagnostics.csv

-------------------------------------------------
七、这次训练样本不是“只有几对”
-------------------------------------------------

默认：

    30000 steps x batch 64 = 1,920,000 个真正进入训练的样本

这些样本来自大参数池里的连续参数行，不是 8/14/27 个状态重复加噪声。

candidate-multiplier=4 表示每步先从参数池看更多候选，再选 64 个兼顾覆盖和
g 距离的样本。候选来自 ShuffledNoReplacementSampler，因此一个 pool cycle 内
候选来源仍然是无放回的。

-------------------------------------------------
八、当前先保持 0.2% 噪声
-------------------------------------------------

这一步只解决：

    参数覆盖 vs g 可分性

不要同时把噪声从 0.2% 改回 9%，否则又多了一个变量。
当正式阶梯在 0.2% 下逻辑跑通以后，再单独做：

    0.2% -> 1% -> 5% -> 9%

噪声阶梯。

-------------------------------------------------
九、压缩包文件
-------------------------------------------------

exp23_pool_sampler.py
    大参数池无放回 + 参数覆盖优先 + soft g-diverse 采样器。

train_exp23_formal_spectrum_pool_gdiverse.py
    正式 g -> f Transformer/BiLSTM+Transformer 训练入口。

run_exp23_formal_pool_curriculum.ps1
    compare / curriculum / single 一键运行。

analyze_exp23_pool_curriculum.py
    汇总 A/B 和阶梯结果。

没有覆盖你原来的训练文件；把这 4 个文件放在项目根目录即可。
