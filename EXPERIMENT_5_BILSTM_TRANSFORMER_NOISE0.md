# Experiment 5: BiLSTM + Transformer（1000 点，0% 噪声）

## 目的

在实验四的严格条件下，只改变输入特征提取结构：在原 Transformer 前加入两层双向 LSTM，检验序列局部依赖建模是否有助于窄峰恢复。

## 唯一主要变量

- Exp4：Transformer
- Exp5：BiLSTM + Transformer

保持不变：输入 100 点、输出 1000 点、0% 训练噪声、无放回打乱、1.6 亿参数池、损失配置、逻辑 batch 64、训练 30000 步。

## 模型结构

```text
g(q²) -> 数值/坐标嵌入 -> 2 层双向 LSTM -> 残差归一化
      -> 原 Transformer encoder/decoder -> f(s), 1000 点
```

BiLSTM 每个方向隐藏维度为 32，双向拼接后为 64，与原 `d_model=64` 对齐。

## 注意

- 使用 `--fresh` 从头训练，不加载实验四权重。
- 先运行 `python test_exp5_bilstm_transformer.py`。
- 新模型可能比实验四多占少量显存；若 `MicroBatchSize=64` OOM，依次改为 32、16。
- 验证脚本 `validate_mc_transformer.py` 已支持从 checkpoint 自动识别 `bilstm_transformer`。
