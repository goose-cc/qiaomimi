# V3 修改报告

## 未改变的物理定义

- 参数顺序和范围：`a1, a2, a3, m, gamma`
- `rho(s)` 解析公式
- `u(s)=rho(s)/(s+400)^2`
- `g(q²)=∫u(s)/(s-q²)ds`
- `s=[0.1764,6]`
- `q²=[-100,-6]`
- `data_scale=160000`
- 9% RMS 高斯白噪声
- 参数池文件格式

## 修改的建模逻辑

- 新增 `ParametricInverseTransformer1D`。
- 模型预测五参数并用原公式解码曲线。
- 输入包含归一化形状和绝对 RMS 尺度。
- PINN 使用预测参数的稳定连续正演。
- 加入参数 loss、共振分量 loss 和非负约束。
- 对弱共振样本的 `m/gamma` 参数 loss 按共振可见度降权，避免监督不可辨识参数。
- 共振 loss 使用 512 点致密网格，不再受 100 点输出网格漏掉窄峰的影响。
- `best_model.pth` 可按独立固定验证集选择。
- 验证脚本新增参数、共振、宽度指标和三类图片。
- 修复“完整曲线 argmax 被误当共振峰”的评价逻辑。
- 新增绝对 `--target-global-step`。

## 已完成测试

- Python 3.8 语法解析检查；
- 所有 Loss profile 前向/反向传播；
- 参数化 Transformer 前向/反向传播；
- NumPy/Torch 物理正演交叉检查；
- 小参数池生成；
- 参数化 PINN 小规模训练；
- 独立验证 best 选择；
- checkpoint 保存与 `--resume`；
- `--target-global-step` 绝对步数停止；
- CSV、JSON 和三类验证图片生成。

完整 1.6 亿参数池和正式长时间训练未在此环境执行，最终精度需要在用户硬件上验证。
