# 1.6 亿五维参数池：低硬盘占用、分块训练版

这套脚本放在项目根目录即可运行。它不会覆盖现有的 `main.py`、`config.py`、`PIDataset.py`、`ModelTools.py` 或 `losses.py`。

## 一、为什么只保存五个参数

每条有效真值只保存：

```text
a1, a2, a3, m, gamma
```

使用 `float32` 时：

```text
160000000 × 5 × 4 bytes = 3.2 GB
```

不保存全部 `rho(s)`、`u(s)`、`g_clean` 或 `g_noisy`。这些长度为 100 的曲线若全部落盘，会把空间放大到几十或上百 GB；大量重复噪声会进一步达到 TB 级。

训练时才临时执行：

```text
读取连续参数块
→ 计算 rho(s)
→ 计算 u(s)=rho(s)/(s+400)^2
→ 正向积分得到 g_clean
→ 在线加入 9% RMS 高斯白噪声
→ 模型训练
→ 释放活跃缓存
```

默认训练量使用固定缩放：

```text
u_scaled = 160000 × u
g_scaled = 160000 × g
```

这不会改变积分关系，只是避免原始数值过小导致 MSE 训练接近零解。

## 二、参数和物理筛选

连续均匀蒙特卡洛范围：

```text
a1 ∈ [0, 0.2]
a2 ∈ [0, 0.05]
a3 ∈ [-0.05, 0.05]
m  ∈ (0, 2]
Γ  ∈ (0, 1]
```

谱函数：

```text
rho(s) =
a1/pi × (mΓ)/((s-m)^2+(mΓ)^2)
+ a2*s + a3
```

研究区间：

```text
s ∈ [0.1764, 6]
q² ∈ [-100, -6]
```

候选参数只要在连续研究区间中出现 `rho(s)<0` 就会被剔除。检查不是只看 100 个网格点，而是根据该谱函数导数定位可能的内部局部最小值，再与两端点一起判断。

## 三、新增文件

```text
mc_pool_config.py
mc_physics.py
generate_mc_parameter_pool.py
check_mc_parameter_pool.py
train_mc_parameter_pool_day.py
README_160M_PARAMETER_POOL.md
VALIDATION_REPORT.txt
git_info_exclude_snippet.txt
```

- `generate_mc_parameter_pool.py`：建立 `float32` memmap 参数池，支持 `Ctrl+C` 安全停止和 `--resume`。
- `check_mc_parameter_pool.py`：检查文件大小、范围、有限值和连续区间非负性。
- `train_mc_parameter_pool_day.py`：分块计算曲线、在线加噪、按时间停止并保存恢复位置。
- `mc_physics.py`：谱函数、连续非负性检查、固定缩放和稳定正向积分。
- `mc_pool_config.py`：集中保存范围、网格、噪声和缩放常数。

训练脚本内置一个可直接运行的 Transformer、CNN 和 1D U-Net。也可以通过 `--model-module` 和 `--model-class` 导入项目现有模型，不需要覆盖项目模型代码。

## 四、先做小规模测试

PowerShell 多行命令使用反引号 `` ` ``，反引号后不能有空格。

### 1. 生成 10 万条测试参数

```powershell
python generate_mc_parameter_pool.py `
  --output-dir ./truth_pool_smoke `
  --num-truths 100000 `
  --candidate-batch-size 100000
```

### 2. 检查参数池

```powershell
python check_mc_parameter_pool.py `
  --pool-dir ./truth_pool_smoke `
  --sample-size 10000 `
  --require-complete
```

### 3. 做 100 步训练测试

```powershell
python train_mc_parameter_pool_day.py `
  --pool-dir ./truth_pool_smoke `
  --checkpoint-dir ./local_models/mc_pool_smoke `
  --model-type transformer `
  --active-block-size 20000 `
  --precompute-chunk-size 1000 `
  --integration-points 64 `
  --batch-size 32 `
  --max-hours 0.1 `
  --max-steps 100 `
  --fresh
```

## 五、正式生成 1.6 亿参数

```powershell
python generate_mc_parameter_pool.py `
  --output-dir ./truth_pool `
  --num-truths 160000000 `
  --candidate-batch-size 1000000
```

按 `Ctrl+C` 后程序会在当前候选批结束时保存。继续：

```powershell
python generate_mc_parameter_pool.py `
  --output-dir ./truth_pool `
  --num-truths 160000000 `
  --candidate-batch-size 1000000 `
  --resume
```

参数文件会按最终目标预分配，逻辑大小约为 3.2 GB。正式生成前请确保磁盘剩余空间明显高于 4 GB。

## 六、每天最多训练 23 小时

```powershell
python train_mc_parameter_pool_day.py `
  --pool-dir ./truth_pool `
  --checkpoint-dir ./local_models/mc_pool `
  --model-type transformer `
  --active-block-size 200000 `
  --precompute-chunk-size 4096 `
  --integration-points 128 `
  --batch-size 64 `
  --max-hours 23 `
  --checkpoint-every-steps 2000 `
  --fresh
```

第二天继续时参数保持相同，并改为：

```powershell
python train_mc_parameter_pool_day.py `
  --pool-dir ./truth_pool `
  --checkpoint-dir ./local_models/mc_pool `
  --model-type transformer `
  --active-block-size 200000 `
  --precompute-chunk-size 4096 `
  --integration-points 128 `
  --batch-size 64 `
  --max-hours 23 `
  --checkpoint-every-steps 2000 `
  --resume
```

检查点保存：

```text
next_parameter_id
pool_cycle
global_step
samples_seen
模型权重
优化器状态
Python / NumPy / PyTorch 随机状态
```

所以不会从参数池开头重新训练。参数池全部覆盖后，`pool_cycle` 增加，同一参数会获得新的在线噪声。

## 七、使用项目已有模型

假设项目根目录存在：

```python
# my_models.py
class MyTransformer(torch.nn.Module):
    ...
```

可以运行：

```powershell
python train_mc_parameter_pool_day.py `
  --pool-dir ./truth_pool_smoke `
  --checkpoint-dir ./local_models/project_model_smoke `
  --model-module my_models `
  --model-class MyTransformer `
  --model-kwargs-json '{"input_size":100,"output_size":100}' `
  --input-layout auto `
  --max-steps 10 `
  --max-hours 0.1 `
  --fresh
```

若模型只接受 `[B,1,100]`，使用：

```text
--input-layout B1L
```

若接受 `[B,100]`，使用：

```text
--input-layout BL
```

## 八、完整覆盖时间

脚本结束时会根据本机端到端实测吞吐打印：

```text
estimated time to cover all 160000000 available parameters: ... h
```

估算包含参数读取、物理计算、积分、加噪、前向、反向和更新，因此比只计算模型前向更可信。

## 九、重要数值说明

理论范围只要求 `m>0`、`Gamma>0`。当 `m*Gamma` 极小时，共振峰可能比 100 点输出网格窄得多。脚本的正向积分使用：

```text
z = atan((s-m)/(m*Gamma))
```

处理窄峰，因此观测积分不会简单漏掉峰；但任何固定 100 点的点值标签都无法严格解析宽度趋近于零的连续峰。若后续必须精确恢复极窄峰，应考虑让网络输出五个参数、分箱积分或自适应表示。

## 十、依赖

```text
Python 3.10+
numpy
torch
```

只生成和检查参数池时不需要 PyTorch；训练脚本需要 PyTorch。
