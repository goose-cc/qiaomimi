# 反问题重建项目

本项目同时保留旧实验代码和新的 **1.6 亿五维参数池 Transformer + PINN** 流程。

## 本次实验请从这里开始

1. 参数池说明与完整操作：`README_160M_PARAMETER_POOL.md`
2. Transformer + PINN Loss 说明：`README_TRANSFORMER_LOSS_TRAINING.md`
3. 独立验证说明：`README_MODEL_VALIDATION_ZH.md`
4. 本次代码审查和修正记录：`FIX_REPORT_160M.md`

## 推荐环境

固定依赖版本适合 **Python 3.10 或 3.11**：

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

检查 GPU：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 重要入口

- `generate_mc_parameter_pool.py`：生成五维参数池。
- `check_mc_parameter_pool.py`：检查参数池。
- `train_mc_parameter_pool_transformer_loss.py`：本次推荐的 Transformer + PINN 训练入口。
- `train_mc_parameter_pool_day.py`：只保留作纯 MSE 基线，不是 PINN 入口。
- `validate_mc_transformer.py`：在独立参数池上验证权重。

旧版 `main.py` 仍用于早期 `.npz` 数据实验，不用于本次 1.6 亿参数池训练。
