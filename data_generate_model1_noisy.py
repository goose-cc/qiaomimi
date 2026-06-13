import os
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 基本设置
# =========================
SEED = 3026

N_SAMPLES = 100000
N_OVERLAY = 500

NX = 100
NY = 100

X_MIN = 0.0
X_MAX = 2.0

Y_MIN = 3.0
Y_MAX = 8.0

# 与原纯净数据集使用不同目录，保证不覆盖、不混合
OUT_DATA_DIR = "./toy_noisy_model1_data"
OUT_FIG_DIR = "./toy_noisy_model1_figures/model1_noisy"

# 参数扰动水平
ERROR_LEVELS = np.array([0.30, 0.10, 0.01], dtype=np.float32)

# 噪声水平
# 这里表示 5% 相对噪声
FX_NOISE_LEVEL = 0.05
GY_NOISE_LEVEL = 0.05


# =========================
# 2. Model 1 公式
# =========================
def toy_model_1(x, a1, a2):
    """
    Model 1:
    f(x) = a1^2 log(x + 0.5) + a2 / (x + 3)
    """
    return a1**2 * np.log(x + 0.5) + a2 / (x + 3.0)


# =========================
# 3. 构造 x, y 与积分核
# =========================
def build_grids():
    x = np.linspace(X_MIN, X_MAX, NX, dtype=np.float32)
    y = np.linspace(Y_MIN, Y_MAX, NY, dtype=np.float32)
    return x, y


def build_kernel(x, y):
    dx = (x[-1] - x[0]) / (len(x) - 1)

    w = np.ones(len(x), dtype=np.float32) * dx
    w[0] *= 0.5
    w[-1] *= 0.5

    # g(y_i) ≈ sum_j w_j * f(x_j) / (y_i - x_j)
    K = w[None, :] / (y[:, None] - x[None, :])
    return K.astype(np.float32)


def compute_gy_batch(fx_batch, K):
    return (fx_batch @ K.T).astype(np.float32)


# =========================
# 4. 参数采样
# =========================
def sample_parameters(n, rng):
    sigma = rng.choice(ERROR_LEVELS, size=n).astype(np.float32)

    a1 = rng.normal(loc=1.0, scale=sigma).astype(np.float32)
    a2 = rng.normal(loc=1.0, scale=sigma).astype(np.float32)

    a1 = np.clip(a1, 0.05, None)
    a2 = np.clip(a2, 0.05, None)

    return a1, a2, sigma


# =========================
# 5. 加噪声
# =========================
def add_relative_noise(data, noise_level, rng):
    """
    相对噪声：
    noisy_data = data + N(0, noise_level * |data|)
    """
    noise = rng.normal(
        loc=0.0,
        scale=noise_level * np.abs(data) + 1e-8,
        size=data.shape
    )
    return (data + noise).astype(np.float32)


# =========================
# 6. 绘图
# =========================
def save_overlay_plot(axis_values, curves_subset, mean_curve,
                      xlabel, ylabel, title, save_path):
    plt.figure(figsize=(10, 6))

    # 浅色线：多组数据叠加
    # 如果你觉得还是太浅，就把 alpha 从 0.12 改成 0.18 或 0.25
    for curve in curves_subset:
        plt.plot(axis_values, curve, alpha=0.12, linewidth=1.0)

    # 深色线：由所有浅色曲线平均得到
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
# 7. 主生成函数
# =========================
def generate_model1_noisy_dataset():
    rng = np.random.default_rng(SEED)

    os.makedirs(OUT_DATA_DIR, exist_ok=True)
    os.makedirs(OUT_FIG_DIR, exist_ok=True)

    x, y = build_grids()
    K = build_kernel(x, y)

    a1, a2, sigma = sample_parameters(N_SAMPLES, rng)

    xx = x[None, :]

    # 原始 clean f(x)
    fx_clean = toy_model_1(xx, a1[:, None], a2[:, None]).astype(np.float32)

    # 对积分前 f(x) 加噪声，只用于画 noisy f(x) 分布图
    fx_noisy = add_relative_noise(fx_clean, FX_NOISE_LEVEL, rng)

# 训练用的干净积分结果：由 clean f(x) 积分得到 clean g(y)
    gy_clean = compute_gy_batch(fx_clean, K)

# 对积分后的 g(y) 加噪声，作为训练输入
    gy_noisy = add_relative_noise(gy_clean, GY_NOISE_LEVEL, rng)

    params = np.stack([a1, a2, sigma], axis=1).astype(np.float32)

    # 保存新的、不与原数据重合的数据集
    save_path = os.path.join(OUT_DATA_DIR, "model1_noisy_100000_pairs.npz")
    np.savez_compressed(
        save_path,
        x=x,
        y=y,

    # 训练代码直接读取这两个字段
    # 输入：gy，也就是带噪声的 g(y)
    # 输出：fx，也就是干净的 f(x)
        fx=fx_clean,
        gy=gy_noisy,

    # 额外保存，方便你之后检查和画图
        fx_clean=fx_clean,
        fx_noisy=fx_noisy,
        gy_clean=gy_clean,
        gy_noisy=gy_noisy,

        model_id=np.ones(N_SAMPLES, dtype=np.int64),
        params=params,
        param_names=np.array(["a1", "a2", "sigma"]),
        dataset_type=np.array(["model1_noisy_only"])
)


    # 计算平均曲线
    mean_fx_noisy = fx_noisy.mean(axis=0)
    mean_gy_noisy = gy_noisy.mean(axis=0)

    # 叠加图只画一部分浅色线，否则 100000 条全画会很卡
    show_n = min(N_OVERLAY, N_SAMPLES)
    show_indices = rng.choice(N_SAMPLES, size=show_n, replace=False)

    fx_subset = fx_noisy[show_indices]
    gy_subset = gy_noisy[show_indices]

    # 1. 积分前 f(x) 加扰动叠加图
    save_overlay_plot(
        axis_values=x,
        curves_subset=fx_subset,
        mean_curve=mean_fx_noisy,
        xlabel="x",
        ylabel="Noisy f(x)",
        title="Model 1 Noisy f(x): Overlay and Mean Curve",
        save_path=os.path.join(OUT_FIG_DIR, "fx_noisy_overlay.png")
    )

    # 2. 积分前 f(x) 加扰动平均图
    save_mean_plot(
        axis_values=x,
        mean_curve=mean_fx_noisy,
        xlabel="x",
        ylabel="Mean noisy f(x)",
        title="Model 1 Noisy f(x): Mean Distribution",
        save_path=os.path.join(OUT_FIG_DIR, "fx_noisy_mean.png")
    )

    # 3. 积分后 g(y) 加扰动叠加图
    save_overlay_plot(
        axis_values=y,
        curves_subset=gy_subset,
        mean_curve=mean_gy_noisy,
        xlabel="y",
        ylabel="Noisy g(y)",
        title="Model 1 Noisy g(y): Overlay and Mean Curve",
        save_path=os.path.join(OUT_FIG_DIR, "gy_noisy_overlay.png")
    )

    # 4. 积分后 g(y) 加扰动平均图
    save_mean_plot(
        axis_values=y,
        mean_curve=mean_gy_noisy,
        xlabel="y",
        ylabel="Mean noisy g(y)",
        title="Model 1 Noisy g(y): Mean Distribution",
        save_path=os.path.join(OUT_FIG_DIR, "gy_noisy_mean.png")
    )

    print("Model 1 加扰动数据集生成完成")
    print(f"数据保存位置: {save_path}")
    print(f"图片保存位置: {OUT_FIG_DIR}")
   


if __name__ == "__main__":
    generate_model1_noisy_dataset()
