# Transformer + 积分物理 Loss：percent10 / percent30 训练说明

## 1. 数据与模型链路

数据生成过程为：

\[
\text{params}=(a_1,a_2,m,\gamma)
\rightarrow f(x)
\rightarrow g_{\mathrm{clean}}(y)
\rightarrow g_{\mathrm{noisy}}(y)
\]

离散积分算子为：

\[
g_i \approx \sum_j \frac{w_j f_j}{y_i-x_j}
\]

相对高斯白噪声为：

\[
g_{\mathrm{noisy}}
=
g_{\mathrm{clean}}
+
p\cdot \operatorname{RMS}(g_{\mathrm{clean}})\cdot \epsilon,
\qquad \epsilon\sim\mathcal N(0,1)
\]

其中 `percent10` 使用 \(p=0.10\)，`percent30` 使用 \(p=0.30\)。
两个数据集使用相同参数划分和相同标准正态噪声样本，只改变噪声缩放倍数。

`PIDataset.py` 会优先选择 `gy_noisy` 作为输入，`fx` 作为监督标签：

\[
\hat f_\theta(x)=\operatorname{Transformer}_\theta(g_{\mathrm{noisy}}(y))
\]

## 2. 环境

建议在项目根目录执行：

### Windows

```bat
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

若固定版本依赖安装冲突，优先使用 Python 3.10 或 3.11，并按你的 CUDA 版本安装相应 PyTorch。

## 3. 生成数据

同时生成 10% 和 30% 数据：

```bash
python data_generate_model3_param_sweep.py
```

分别生成：

```bash
python data_generate_model3_param_sweep.py --datasets percent10
python data_generate_model3_param_sweep.py --datasets percent30
```

覆盖已有数据：

```bash
python data_generate_model3_param_sweep.py --overwrite
```

生成后应出现：

```text
percent10/train.npz
percent10/val.npz
percent10/test.npz
percent30/train.npz
percent30/val.npz
percent30/test.npz
```

## 4. 先做 smoke test

README 原始基线使用 `base`，其损失只有：

\[
\mathcal L_{\mathrm{base}}
=
\operatorname{MSE}(\hat f,f)
\]

```bash
python main.py --exp percent10 --model_type transformer --loss_profile base --num_epochs 1 --max_train_samples 200 --max_val_samples 100 --max_test_samples 100 --model_path ./model/percent10_smoke_base.pth --index 0 --seed 2026

python main.py --exp percent30 --model_type transformer --loss_profile base --num_epochs 1 --max_train_samples 200 --max_val_samples 100 --max_test_samples 100 --model_path ./model/percent30_smoke_base.pth --index 0 --seed 2026
```

如果你所说的“Transformer + loss 公式”是指加入积分物理一致性，请使用 `pinn`：

\[
\mathcal L_{\mathrm{pinn}}
=
\operatorname{MSE}(\hat f,f)
+
\lambda_{\mathrm{grad}}
\operatorname{MSE}(\Delta\hat f,\Delta f)
+
\lambda_{\mathrm{physics}}
\operatorname{MSE}(A\hat f,g_{\mathrm{noisy}})
\]

默认权重位于 `config.py`：

```python
self.lambda_grad = 0.1
self.lambda_physics = 0.01
```

对应 smoke test：

```bash
python main.py --exp percent10 --model_type transformer --loss_profile pinn --num_epochs 1 --max_train_samples 200 --max_val_samples 100 --max_test_samples 100 --model_path ./model/percent10_smoke_pinn.pth --index 0 --seed 2026

python main.py --exp percent30 --model_type transformer --loss_profile pinn --num_epochs 1 --max_train_samples 200 --max_val_samples 100 --max_test_samples 100 --model_path ./model/percent30_smoke_pinn.pth --index 0 --seed 2026
```

## 5. 正式训练

### 严格复现 README 的 base 基线

```bash
python main.py --exp percent10 --model_type transformer --loss_profile base --full_data --num_epochs 50 --index 0 --seed 2026

python main.py --exp percent30 --model_type transformer --loss_profile base --full_data --num_epochs 100 --index 0 --seed 2026
```

### 使用积分物理 loss

为了公平比较噪声水平，建议两组保持相同的 epoch、batch size、学习率和随机种子，例如都训练 100 epoch：

```bash
python main.py --exp percent10 --model_type transformer --loss_profile pinn --full_data --num_epochs 100 --batch_size 64 --learning_rate 1e-3 --index 0 --seed 2026

python main.py --exp percent30 --model_type transformer --loss_profile pinn --full_data --num_epochs 100 --batch_size 64 --learning_rate 1e-3 --index 0 --seed 2026
```

30% 噪声下也可增加一个鲁棒监督损失对照：

```bash
python main.py --exp percent30 --model_type transformer --loss_profile pinn_huber --full_data --num_epochs 100 --batch_size 64 --learning_rate 1e-3 --index 0 --seed 2026
```

## 6. 输出位置

checkpoint 自动分开保存，例如：

```text
model/percent10_transformer_base.pth
model/percent30_transformer_base.pth
model/percent10_transformer_pinn.pth
model/percent30_transformer_pinn.pth
```

每次运行的配置、预测图和预测数据保存在：

```text
picture/training_results/train_xxx_时间_exp_model_loss/
```

其中包含：

```text
run_info.txt
prediction_*.png
prediction_*.npz
```

终端最后会输出测试集 `Test Loss MSE`。

## 7. 建议的实验表

至少记录：

| exp | loss_profile | seed | epochs | best val MSE | test MSE |
|---|---|---:|---:|---:|---:|
| percent10 | base | 2026 | 100 |  |  |
| percent10 | pinn | 2026 | 100 |  |  |
| percent30 | base | 2026 | 100 |  |  |
| percent30 | pinn | 2026 | 100 |  |  |
| percent30 | pinn_huber | 2026 | 100 |  |  |

为了更可靠，最终可用 3 个随机种子（如 2026、2027、2028）重复训练并报告均值与标准差。

## 8. 本修正版处理的问题

1. 原压缩包缺少 `utils/visualize.py`，但 `main.py` 导入并调用了它，会直接触发 `ModuleNotFoundError`。本版移除了该依赖，保留已有的图片保存函数。
2. `argparse` 帮助文本中的 `%` 会导致 `python main.py --help` 报错，本版已转义。
3. 新增 `--seed`，便于 percent10 和 percent30 使用相同模型初始化与数据打乱随机性。
