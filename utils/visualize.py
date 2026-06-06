import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_prediction(fx_true, fx_pred, x):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制散点图
    plt.scatter(x, fx_true, s=5, alpha=0.7, label='True f(x)', color='blue')
    plt.scatter(x, fx_pred, s=5, alpha=0.7, label='Predicted f(x)', color='red')

    # 添加标题和标签
    plt.title('difference between fx_true and fx_pred', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)

    plt.legend()

    # 添加网格
    plt.grid(True, alpha=0.3)

    # 显示图形
    plt.tight_layout()
    plt.show()
