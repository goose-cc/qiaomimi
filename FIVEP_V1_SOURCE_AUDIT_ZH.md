# 五参数源码交付审计记录（上传的审计 ZIP）

审计范围：`build_fivep_codebook.py`、`validate_fivep_codebook.py`、`build_fivep_closed_set.py`、`build_fivep_dataset.py`、`validate_fivep_dataset.py`、`mc_physics.py`、`mc_pool_config.py`、实际数据元信息（不含完整 NPZ 样本）。

## 已证实的事项

- 参数域：`build_fivep_dataset.py` 第 22–26 行：`a1 [.05,.20]`, `a2 [0,.05]`, `a3 [-.05,.05]`, `m [.40,1.20]`, `gamma [.01,1]`，三条恢复门槛未放宽。
- 正向物理：`mc_physics.py` 第 21–35 行共振中心为 `m**2`，背景 `a2*s+a3`。`metadata.json` 存储的 `mc_physics.py` 和 `mc_pool_config.py` 哈希均与上传源码实际 SHA256 相同。
- q² 固定在 [-100,-6] 共100点；`q2.npy` 实际 SHA-256 与 `metadata.json` 相符。
- `build_fivep_dataset.py` 第 215–230 行，归一化只读取 0.2% `train.npz`，使用参数 `log_gamma` 作为训练目标的第5个变换分量；`validate_fivep_dataset.py` 第 175–191 行独立复核。
- `build_fivep_codebook.py` 第 215–225 行，Train/Val/Test 是按物理状态分组后才加入独立噪声，已有提供的运行输出也通过身份重复检查。
- `build_fivep_closed_set.py` 第 119–175、178–215 行故意让同一状态出现在三个集合、噪声独立，且将任务清楚标为封闭集识别，不能用于未知状态泛化结论。

## 未达到正式严格盲测要求的问题

- **共同 g 选择：** `build_fivep_codebook.py` 第 203–206 行先计算所有候选观测，再调用 `select_codebook`；第 108–116 行 `nearest` 统一引用已选各 split 的 g；第 134–164 行依次为 Test、Train、Val 强制端点、分箱、补充选样。Train 的码本设计因而会受到 Test 候选 g 选样决策影响。它不证明存在标签重复或神经网络作弊，但不能称为“测试观测从未参与训练码本选择”的严格盲测。
- **a2/a3 网格：** `build_fivep_dataset.py` 第 101–106 行按连续 Sobol 变量采样，且 `build_fivep_codebook.py` 第 41–44 行的边界候选也是连续值，图片中的 0.001 间隔没有强制保证。需要用户确认间隔是否也是硬条件。
- **数据不充分性：** 上传审计 ZIP 不含现有 Train/Val/Test NPZ 和 manifest，不能重新计算你本地旧数据的逐条验证结果，只能核对源码、元数据和用户提供的运行记录。新生成器已用真实物理核完整执行了样例测试。

## 处理

保留旧文件和旧数据；新增加 `build_fivep_blind.py`, `validate_fivep_blind.py` 和 `test_fivep_blind.py`。这些文件在没有查看未知状态 g 的前提下先锁定 Train，随后通过独立种子生成 Val/Test 物理状态，并仅凭参数域覆盖规则选出测试状态，最后正演和加噪。测试 g 和 Train g 的距离只作诊断，不筛除。

正式交付网络团队前需冻结：这次盲测的生成脚本版本、物理核哈希、配置、种子、NPZ、元数据、独立测试集访问规则及 recovery 评价脚本。网络预测 recovery 是否通过须由训练后计算，不能提前认证。
