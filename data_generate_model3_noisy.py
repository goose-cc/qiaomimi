import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def trapezoid_weights(x):
    """
    生成梯形积分权重，用于：
    ∫ f(x) dx ≈ sum_i w_i f(x_i)
    """
    x = np.asarray(x, dtype=np.float32)

    if len(x) < 2:
        raise ValueError("x 至少需要两个采样点")

    dx = x[1] - x[0]
    w = np.ones_like(x, dtype=np.float32) * dx
    w[0] = dx / 2.0
    w[-1] = dx / 2.0
    return w


def model3_fx(x, a1, a2, m=0.8, gamma=0.5):
    """
    Model 3:

    f(x)=1/pi * [a1*m*Gamma / ((x-m^2)^2 + (a1*m*Gamma)^2)] + a2*x/5
    """
    width = a1[:, None] * m * gamma

    x_grid = x[None, :]

    peak = (1.0 / np.pi) * width / (
        (x_grid - m ** 2) ** 2 + width ** 2
    )

    background = a2[:, None] * x_grid / 5.0

    fx = peak + background

    return fx.astype(np.float32)


def compute_gy_batch(x, y, fx_all):
    """
    批量计算：

    g(y) = ∫_0^2 f(x)/(y-x) dx

    fx_all shape: [num_samples, num_x_points]
    返回 gy_all shape: [num_samples, num_y_points]
    """
    weights = trapezoid_weights(x)

    # kernel shape: [num_y_points, num_x_points]
    kernel = weights[None, :] / (y[:, None] - x[None, :])

    # fx_all @ kernel.T -> [num_samples, num_y_points]
    gy_all = fx_all @ kernel.T

    return gy_all.astype(np.float32)


def plot_mean_curve(x_or_y, values, title, xlabel, ylabel, save_path):
    mean_curve = values.mean(axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(x_or_y, mean_curve, linewidth=2.5, label="Mean curve")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_overlay_curve(x_or_y, values, title, xlabel, ylabel, save_path, max_plot=500):
    mean_curve = values.mean(axis=0)

    plt.figure(figsize=(10, 6))

    n = min(max_plot, len(values))

    for i in range(n):
        plt.plot(x_or_y, values[i], alpha=0.08, linewidth=0.8)

    plt.plot(x_or_y, mean_curve, linewidth=3.0, label="Mean curve")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    # =========================
    # 基本参数
    # =========================
    random_seed = 2026

    num_samples = 100000

    num_x_points = 100
    num_y_points = 100

    x_min = 0.0
    x_max = 2.0

    y_min = 3.0
    y_max = 8.0

    m = 0.8
    gamma = 0.5

    # a1、a2 参数扰动强度
    param_noise_level = 0.10

    # g(y) 输入端相对高斯噪声强度
    # eta(y) ~ N(0, sigma^2)
    # sigma = 0.01 表示约 1% 相对噪声
    gy_noise_level = 0.01

    overlay_plot_samples = 500

    # =========================
    # 新目录，不覆盖旧数据
    # =========================
    data_dir = "./toy_noisy_model3_data"
    figure_dir = "./toy_noisy_model3_figures/model3_noisy"

    ensure_dir(data_dir)
    ensure_dir(figure_dir)

    save_path = os.path.join(data_dir, "model3_noisy_100000_pairs.npz")

    # 如果已经存在，直接停止，避免误覆盖
    if os.path.exists(save_path):
        raise FileExistsError(
            f"目标文件已经存在，为避免覆盖已停止运行：{save_path}\n"
            f"如果你确实想重新生成，请先手动删除这个文件。"
        )

    np.random.seed(random_seed)

    x = np.linspace(x_min, x_max, num_x_points).astype(np.float32)
    y = np.linspace(y_min, y_max, num_y_points).astype(np.float32)

    # =========================
    # 1. 生成 a1、a2 参数扰动
    # =========================
    a1 = 1.0 + np.random.normal(0.0, param_noise_level, size=num_samples)
    a2 = 1.0 + np.random.normal(0.0, param_noise_level, size=num_samples)

    # 防止极端情况下参数变成负数
    a1 = np.maximum(a1, 0.05).astype(np.float32)
    a2 = np.maximum(a2, 0.05).astype(np.float32)

    # =========================
    # 2. 生成扰动后的 f(x)
    # =========================
    fx_all = model3_fx(x, a1, a2, m=m, gamma=gamma)

    # =========================
    # 3. 积分生成 clean g(y)
    # =========================
    gy_clean_all = compute_gy_batch(x, y, fx_all)

    # =========================
    # 4. 对输入端 g(y) 加相对高斯噪声
    # =========================
    eta = np.random.normal(
        0.0,
        gy_noise_level,
        size=gy_clean_all.shape
    ).astype(np.float32)

    gy_noisy_all = gy_clean_all * (1.0 + eta)

    # =========================
    # 5. 保存 npz 数据
    # =========================
    np.savez_compressed(
        save_path,
        x=x,
        y=y,
        fx=fx_all,
        gy_clean=gy_clean_all,
        gy_noisy=gy_noisy_all,
        a1=a1,
        a2=a2,
        m=np.float32(m),
        gamma=np.float32(gamma),
        param_noise_level=np.float32(param_noise_level),
        gy_noise_level=np.float32(gy_noise_level),
        model_name="model3_noisy",
    )

    # =========================
    # 6. 生成四张图
    # =========================
    plot_mean_curve(
        x,
        fx_all,
        "Model 3 Noisy f(x): Mean Distribution",
        "x",
        "Mean noisy f(x)",
        os.path.join(figure_dir, "fx_noisy_mean.png"),
    )

    plot_overlay_curve(
        x,
        fx_all,
        "Model 3 Noisy f(x): Overlay and Mean Curve",
        "x",
        "Noisy f(x)",
        os.path.join(figure_dir, "fx_noisy_overlay.png"),
        max_plot=overlay_plot_samples,
    )

    plot_mean_curve(
        y,
        gy_noisy_all,
        "Model 3 Noisy g(y): Mean Distribution",
        "y",
        "Mean noisy g(y)",
        os.path.join(figure_dir, "gy_noisy_mean.png"),
    )

    plot_overlay_curve(
        y,
        gy_noisy_all,
        "Model 3 Noisy g(y): Overlay and Mean Curve",
        "y",
        "Noisy g(y)",
        os.path.join(figure_dir, "gy_noisy_overlay.png"),
        max_plot=overlay_plot_samples,
    )

    print("Model 3 noisy 数据生成完成")
    print(f"数据保存位置: {save_path}")
    print(f"图片保存位置: {figure_dir}")
    print("不会覆盖 toy_clean_data、toy_noisy_model1_data、data、data_exp2。")


if __name__ == "__main__":
    main()
