五参数小样本过拟合诊断

目的：
- 只用 Train，不读取 Val/Test
- 选取 32 个独立物理状态（约 96 条噪声样本）
- MLP 与 TCN 分别强行拟合
- 用于判断当前问题是“训练本身学不会”还是“独立状态数量太少导致泛化不足”

运行：
python .\overfit_fivep_sanity.py --data-dir .\fivep_train_val_v1\data --device auto --states 32 --epochs 1200

结果判断：
- joint Recovery 接近 100%：训练接口基本正常，随后做 32→64→128→180 状态规模诊断
- 32 状态仍明显学不会：优先检查训练结构/目标参数化/优化，而不是先扩到几千状态
