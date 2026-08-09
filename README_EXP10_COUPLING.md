# Exp10 coupling ladder

把这 3 个文件放到实验七项目根目录：

- `train_exp10_coupling_ladder.py`
- `run_exp10_coupling_ladder.ps1`
- `README_EXP10_COUPLING.md`

不需要修改实验七原文件，也不需要重新生成 1.6 亿参数池。

## 为什么做这个实验

刚完成的 visibility 分析说明：

- WEAK 样本的 `m` 确实更差；
- 但 V>=0.2 的 VISIBLE 样本中，`m≈0.216`、`gamma≈0.232`，仍然很差；
- 所以“弱峰样本把全局误差拖高”只能解释一小部分，不能解释核心失败。

下一步必须定位：从单参数可学，到五参数联合学习崩溃，究竟在哪一组参数开始发生。

## 模式

### 1. mg

```text
a1 = 0.10 固定
a2 = 0.025 固定
a3 = 0 固定
m, gamma 全范围变化
```

只对 `m,gamma` 算 loss。

如果这里已经明显失败：
`m-gamma` 联合耦合就是第一嫌疑。

### 2. a1mg_strong

```text
a1 ∈ [0.05, 0.20]
m, gamma 全范围变化
a2 = 0.025 固定
a3 = 0 固定
```

只对 `a1,m,gamma` 算 loss。

如果 `mg` 好、这里坏：
`a1` 加入后导致 resonance 参数耦合。

### 3. a1mg_full

和上面一样，但：

```text
a1 ∈ [0, 0.20]
```

如果 `a1mg_strong` 好、这里明显坏：
`a1→0` 附近的弱共振/无意义 m,gamma 标签是重要问题。

### 4. all5_strong

```text
a1 ∈ [0.05,0.20]
a2,a3,m,gamma 按原 Exp7 全范围变化
```

如果 `a1mg_strong` 好、这里坏：
背景 `a2,a3` 与 resonance 参数之间的联合耦合是核心问题。

### 5. all5_full

原 Exp7 五参数范围的在线版本，用作可选复现实验。

## 推荐运行

```powershell
.\run_exp10_coupling_ladder.ps1 `
  -Mode recommended `
  -MaxSteps 30000 `
  -ValSamples 10000 `
  -Device cuda
```

默认依次跑：

```text
mg
→ a1mg_strong
→ a1mg_full
→ all5_strong
```

如果你想先节省时间，可以一个一个跑：

```powershell
.\run_exp10_coupling_ladder.ps1 -Mode mg -MaxSteps 30000 -Device cuda
```

每个实验输出：

```text
model/exp10_coupling/<mode>/
    best_model.pth
    latest_checkpoint.pth
    validation_best.json
    validation_latest.json
    exp10_summary.json
```

## 判断标准

重点看 `validation_best.json` 中变化参数的 normalized absolute mean。

粗略判断：

```text
< 0.05  很好
0.05~0.10  可接受/明显能学
0.10~0.15  已出现困难
> 0.15  明显崩溃
~0.23~0.25  接近 Exp7 的“几乎平均猜”状态
```

不要先加 peak loss、visibility gate 或噪声。
这个实验只负责定位参数耦合发生在哪里。
