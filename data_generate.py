import os
import numpy as np
import matplotlib.pyplot as plt
from config import Config

cfg = Config()


# =========================
# 1. 三个 toy model
# =========================
def toy_model_1(x, a1, a2):
    # Model 1: f(x) = a1^2 log(x + 0.5) + a2 / (x + 3)
    return a1**2 * np.log(x + 0.5) + a2 / (x + 3.0)


def toy_model_2(x, a1, a2):
    # Model 2: f(x) = a1^4 x exp(-a2 x)
    return a1**4 * x * np.exp(-a2 * x)


def toy_model_3(x, a1, a2, m, gamma):
    # Model 3: f(x) = 1/pi * (a1*m*Gamma)/((x-m^2)^2 + (a1*m*Gamma)^2) + a2*x/5
    width = a1 * m * gamma
    return (1.0 / np.pi) * width / ((x - m**2)**2 + width**2) + a2 * x / 5.0


# =========================
# 2. 采样参数
# =========================
def sample_parameters(n, rng):
    sigma_choices = np.array(cfg.error_levels, dtype=np.float32)
    sigma = rng.choice(sigma_choices, size=n)

    a1 = rng.normal(loc=1.0, scale=sigma)
    a2 = rng.normal(loc=1.0, scale=sigma)

    # 防止接近 0 或出现负值
    a1 = np.clip(a1, 0.05, None)
    a2 = np.clip(a2, 0.05, None)

    return a1.astype(np.float32), a2.astype(np.float32), sigma.astype(np.float32)


# =========================
# 3. 构建积分网格与核
#    g(y) = ∫ f(x)/(y-x) dx
# =========================
def build_grids():
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.num_x_points, dtype=np.float32)
    y = np.linspace(cfg.y_min, cfg.y_max, cfg.num_y_points, dtype=np.float32)
    return x, y


def build_kernel(x, y):
    dx = (x[-1] - x[0]) / (len(x) - 1)
    w = np.ones(len(x), dtype=np.float32) * dx
    w[0] *= 0.5
    w[-1] *= 0.5

    # K_ij = w_j / (y_i - x_j)
    K = w[None, :] / (y[:, None] - x[None, :])
    return K.astype(np.float32)


def compute_gy_batch(fx_batch, K):
    # fx_batch shape: [N, Nx]
    # gy_batch shape: [N, Ny]
    return (fx_batch @ K.T).astype(np.float32)


# =========================
# 4. 绘图函数
# =========================
def save_overlay_plot(axis_values, curves_subset, mean_curve,
                      xlabel, ylabel, title, save_path):
    plt.figure(figsize=(10, 6))

    # 浅色线：多组叠加
    for curve in curves_subset:
        plt.plot(axis_values, curve, alpha=0.15, linewidth=1.2)

    # 深色线：所有数据平均后得到
    plt.plot(axis_values, mean_curve, linewidth=3.0, label="Mean curve")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_mean_plot(axis_values, mean_curve,
                   xlabel, ylabel, title, save_path):
    plt.figure(figsize=(10, 6))

    plt.plot(axis_values, mean_curve, linewidth=3.0)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# 5. 为单个模型生成 100000 对纯净数据
# =========================
def generate_one_model(model_id, rng):
    x, y = build_grids()
    K = build_kernel(x, y)

    n = cfg.num_clean_samples_per_model
    a1, a2, sigma = sample_parameters(n, rng)

    xx = x[None, :]  # shape [1, Nx]

    if model_id == 1:
        fx = toy_model_1(xx, a1[:, None], a2[:, None])
        params = np.stack([a1, a2, np.full_like(a1, np.nan), np.full_like(a1, np.nan), sigma], axis=1)
        model_name = "model1"

    elif model_id == 2:
        fx = toy_model_2(xx, a1[:, None], a2[:, None])
        params = np.stack([a1, a2, np.full_like(a1, np.nan), np.full_like(a1, np.nan), sigma], axis=1)
        model_name = "model2"

    elif model_id == 3:
        m = np.full(n, cfg.model3_m, dtype=np.float32)
        gamma = np.full(n, cfg.model3_gamma, dtype=np.float32)
        fx = toy_model_3(xx, a1[:, None], a2[:, None], m[:, None], gamma[:, None])
        params = np.stack([a1, a2, m, gamma, sigma], axis=1)
        model_name = "model3"

    else:
        raise ValueError("model_id 必须是 1、2、3")

    fx = fx.astype(np.float32)
    gy = compute_gy_batch(fx, K)

    # 保存数据
    os.makedirs(cfg.output_root, exist_ok=True)
    save_data_path = os.path.join(cfg.output_root, f"{model_name}_clean_100000_pairs.npz")
    np.savez_compressed(
        save_data_path,
        x=x,
        y=y,
        fx=fx,
        gy=gy,
        params=params.astype(np.float32),
        param_names=np.array(["a1", "a2", "m", "gamma", "sigma"])
    )

    # 取全部样本平均
    mean_fx = fx.mean(axis=0)
    mean_gy = gy.mean(axis=0)

    # 叠加图只显示一部分样本，否则 100000 条全部画出来太密
    show_n = min(cfg.overlay_plot_samples, n)
    show_indices = rng.choice(n, size=show_n, replace=False)
    fx_subset = fx[show_indices]
    gy_subset = gy[show_indices]

    # 图像目录
    model_fig_dir = os.path.join(cfg.figure_root, model_name)
    os.makedirs(model_fig_dir, exist_ok=True)

    # 1) f(x) 叠加图（浅色线 + 深色平均线）
    save_overlay_plot(
        axis_values=x,
        curves_subset=fx_subset,
        mean_curve=mean_fx,
        xlabel="x",
        ylabel="f(x)",
        title=f"{model_name.upper()} - Overlay of f(x) (light curves) and mean curve (dark line)",
        save_path=os.path.join(model_fig_dir, "fx_overlay.png")
    )

    # 2) f(x) 平均图（单独成图）
    save_mean_plot(
        axis_values=x,
        mean_curve=mean_fx,
        xlabel="x",
        ylabel="Mean f(x)",
        title=f"{model_name.upper()} - Mean distribution of f(x)",
        save_path=os.path.join(model_fig_dir, "fx_mean.png")
    )

    # 3) g(y) 叠加图（浅色线 + 深色平均线）
    save_overlay_plot(
        axis_values=y,
        curves_subset=gy_subset,
        mean_curve=mean_gy,
        xlabel="y",
        ylabel="g(y)",
        title=f"{model_name.upper()} - Overlay of g(y) (light curves) and mean curve (dark line)",
        save_path=os.path.join(model_fig_dir, "gy_overlay.png")
    )

    # 4) g(y) 平均图（单独成图）
    save_mean_plot(
        axis_values=y,
        mean_curve=mean_gy,
        xlabel="y",
        ylabel="Mean g(y)",
        title=f"{model_name.upper()} - Mean distribution of g(y)",
        save_path=os.path.join(model_fig_dir, "gy_mean.png")
    )

    print(f"[OK] {model_name} 数据已保存到: {save_data_path}")
    print(f"[OK] {model_name} 图像已保存到: {model_fig_dir}")


# =========================
# 6. 主函数
# =========================
def main():
    rng = np.random.default_rng(cfg.random_seed)

    # 分别为三个模型构造
    generate_one_model(1, rng)
    generate_one_model(2, rng)
    generate_one_model(3, rng)

    print("全部 toy model 纯净数据与图像生成完成。")


if __name__ == "__main__":
    main()
