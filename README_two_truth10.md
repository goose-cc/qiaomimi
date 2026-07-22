# two_truth10 数据集与训练说明

## 实验定义

只使用两个固定物理真值：

- 真值 1：`a1=1, a2=1, m=0.8, gamma=0.2`
- 真值 2：`a1=1, a2=1, m=0.4, gamma=0.2`

每个真值产生 10000 条独立的 10% 零均值高斯白噪声观测，共 20000 条训练数据。

噪声定义：

```text
sigma_i = 0.10 * RMS(g_clean_i)
noise_i ~ N(0, sigma_i^2)
g_noisy_i = g_clean_i + noise_i
```

这里“白噪声”不是均匀分布，而是各采样点独立、均值为 0 的 Gaussian white noise。

## 替换文件

把以下 3 个文件复制到项目根目录并覆盖同名文件：

```text
config.py
main.py
data_generate_model3_two_truths_10percent.py
```

`PIDataset.py` 不需要修改，因为它会自动优先读取 `gy_noisy`。

## 生成数据

PowerShell：

```powershell
python data_generate_model3_two_truths_10percent.py
```

需要覆盖旧数据：

```powershell
python data_generate_model3_two_truths_10percent.py --overwrite
```

生成目录：

```text
data_two_truth10/
    train.npz   # 全部 20000 条
    val.npz     # 从 train 固定随机抽取 2000 条
    test.npz    # 从 train 固定随机抽取另 2000 条
```

## Smoke test

```powershell
python main.py --exp two_truth10 --model_type transformer --loss_profile base --num_epochs 1 --max_train_samples 200 --max_val_samples 100 --max_test_samples 100 --model_path ./local_models/two_truth10_smoke.pth --index 0
```

## 正式训练

```powershell
python main.py --exp two_truth10 --model_type transformer --loss_profile base --full_data --num_epochs 50 --model_path ./local_models/two_truth10_transformer_base.pth --index 0
```

## 评价含义

按老师当前要求，全部 20000 条都进入训练，val/test 是训练数据的固定随机子集。因此这里得到的是“已见真值和已见噪声样本上的重建表现”，不是严格的独立泛化能力。后续若要判断模型对新噪声实现或新 `(m, gamma)` 的泛化，应另外生成独立 test 数据。
