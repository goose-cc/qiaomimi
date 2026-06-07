import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_prediction(fx_true, fx_pred, x, title="difference between fx_true and fx_pred", save_path=None):
    """
    可视化真实 f(x) 和预测 f(x)，并支持保存图片。

    参数：
    fx_true: 真实的 f(x)
    fx_pred: 模型预测的 f(x)
    x: x 轴采样点
    title: 图标题
    save_path: 图片保存路径，如果为 None，则只显示不保存
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    fx_true = np.squeeze(fx_true)
    fx_pred = np.squeeze(fx_pred)
    x = np.squeeze(x)

    plt.figure(figsize=(12, 6))
    plt.scatter(x, fx_true, label="True f(x)", s=10)
    plt.scatter(x, fx_pred, label="Predicted f(x)", s=10)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved prediction figure: {save_path}")

    plt.show()
