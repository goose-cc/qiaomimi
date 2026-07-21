# 10% / 30% 噪声实验修改版

把这三个文件覆盖到项目根目录：

- `data_generate_model3_param_sweep.py`
- `main.py`
- `config.py`

## 生成数据

默认同时生成两个目录：

```bash
python data_generate_model3_param_sweep.py
```

只生成一个：

```bash
python data_generate_model3_param_sweep.py --datasets percent10
python data_generate_model3_param_sweep.py --datasets percent30
```

重新生成并覆盖现有文件：

```bash
python data_generate_model3_param_sweep.py --overwrite
```

## 训练

```bash
python main.py --exp percent10 --model_type transformer --loss_profile base --full_data --num_epochs 50 --index 0
python main.py --exp percent30 --model_type transformer --loss_profile base --full_data --num_epochs 100 --index 0
```

建议先 smoke test：

```bash
python main.py --exp percent10 --model_type transformer --loss_profile base --num_epochs 1 --max_train_samples 200 --max_val_samples 100 --max_test_samples 100 --model_path ./model/percent10_smoke.pth --index 0
python main.py --exp percent30 --model_type transformer --loss_profile base --num_epochs 1 --max_train_samples 200 --max_val_samples 100 --max_test_samples 100 --model_path ./model/percent30_smoke.pth --index 0
```

两个数据集使用相同参数真值划分和相同标准正态随机数，只改变噪声缩放比例，因此比较更公平。
