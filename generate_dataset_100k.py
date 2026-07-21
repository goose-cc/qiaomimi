import os
import numpy as np
from tqdm import tqdm

# =====================
# 1. 基本设置
# =====================
SEED = 2026

N_TOTAL = 100_000
N_TRAIN = 80_000
N_VAL = 10_000
N_TEST = 10_000

NX = 100   # f(x) 的采样点数
NY = 100   # g(y) 的采样点数

# 图 11 / toy model 常用区间
X_MIN = 0.0
X_MAX = 2.0

# 你的图片中要求 y > x，因此这里取 y in [3, 8]
Y_MIN = 3.0
Y_MAX = 8.0

# 图 11 中使用的输入误差：30%、10%、1%
ERROR_LEVELS = np.array([0.30, 0.10, 0.01], dtype=np.float32)

# Model 3 中的共振峰参数，论文中 m=0.8, Gamma=0.5
RANDOMIZE_M_GAMMA = False
M_FIXED = 0.8
GAMMA_FIXED = 0.5

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)

# =====================
# 2. 构造 x、y 网格和积分矩阵
# =====================
x = np.linspace(X_MIN, X_MAX, NX, dtype=np.float32)
y = np.linspace(Y_MIN, Y_MAX, NY, dtype=np.float32)

# 梯形积分权重
dx = (X_MAX - X_MIN) / (NX - 1)
w = np.ones(NX, dtype=np.float32) * dx
w[0] *= 0.5
w[-1] *= 0.5

# 积分核矩阵 K
# g(y_i) ≈ sum_j w_j * f(x_j) / (y_i - x_j)
K = w[None, :] / (y[:, None] - x[None, :])
K = K.astype(np.float32)

# =====================
# 3. 三个 toy model
# =====================
def generate_batch(batch_size: int):
    """
    生成一个 batch 的数据：
    fx: [batch_size, NX]
    gy: [batch_size, NY]
    """

    # 随机选择模型：1、2、3
    model_id = rng.integers(1, 4, size=batch_size, dtype=np.int32)

    # 每条数据随机选择一个误差水平：30%、10%、1%
    sigma = rng.choice(ERROR_LEVELS, size=batch_size).astype(np.float32)

    # 随机化 a1, a2
    # 均值为 1，标准差为 sigma
    a1 = rng.normal(loc=1.0, scale=sigma).astype(np.float32)
    a2 = rng.normal(loc=1.0, scale=sigma).astype(np.float32)

    # 防止参数变成负数或太接近 0
    a1 = np.clip(a1, 0.05, None)
    a2 = np.clip(a2, 0.05, None)

    # m 和 Gamma 先固定，严格对应论文 toy model
    if RANDOMIZE_M_GAMMA:
        m = rng.uniform(0.6, 1.0, size=batch_size).astype(np.float32)
        gamma = rng.uniform(0.3, 0.7, size=batch_size).astype(np.float32)
    else:
        m = np.full(batch_size, M_FIXED, dtype=np.float32)
        gamma = np.full(batch_size, GAMMA_FIXED, dtype=np.float32)

    xx = x[None, :]
    fx = np.zeros((batch_size, NX), dtype=np.float32)

    mask1 = model_id == 1
    mask2 = model_id == 2
    mask3 = model_id == 3

    # Model 1:
    # f(x) = a1^2 * log(x + 0.5) + a2 / (x + 3)
    if mask1.any():
        aa1 = a1[mask1, None]
        aa2 = a2[mask1, None]
        fx[mask1] = aa1**2 * np.log(xx + 0.5) + aa2 / (xx + 3.0)

    # Model 2:
    # f(x) = a1^4 * x * exp(-a2 * x)
    if mask2.any():
        aa1 = a1[mask2, None]
        aa2 = a2[mask2, None]
        fx[mask2] = aa1**4 * xx * np.exp(-aa2 * xx)

    # Model 3:
    # f(x) = 1/pi * a1*m*Gamma / ((x-m^2)^2 + (a1*m*Gamma)^2) + a2*x/5
    if mask3.any():
        aa1 = a1[mask3, None]
        aa2 = a2[mask3, None]
        mm = m[mask3, None]
        gg = gamma[mask3, None]

        width = aa1 * mm * gg
        fx[mask3] = (
            (1.0 / np.pi)
            * width
            / ((xx - mm**2) ** 2 + width**2)
            + aa2 * xx / 5.0
        )

    # 通过积分关系生成 g(y)
    gy = fx @ K.T

    # 保存参数，方便以后检查
    params = np.stack([a1, a2, m, gamma, sigma], axis=1).astype(np.float32)

    return fx.astype(np.float32), gy.astype(np.float32), model_id, params


# =====================
# 4. 生成 train / val / test
# =====================
def make_split(name: str, n_samples: int, batch_size: int = 5000):
    fx_all = np.empty((n_samples, NX), dtype=np.float32)
    gy_all = np.empty((n_samples, NY), dtype=np.float32)
    model_all = np.empty((n_samples,), dtype=np.int32)
    params_all = np.empty((n_samples, 5), dtype=np.float32)

    start = 0

    with tqdm(total=n_samples, desc=f"Generating {name}") as pbar:
        while start < n_samples:
            b = min(batch_size, n_samples - start)

            fx, gy, model_id, params = generate_batch(b)

            fx_all[start:start + b] = fx
            gy_all[start:start + b] = gy
            model_all[start:start + b] = model_id
            params_all[start:start + b] = params

            start += b
            pbar.update(b)

    out_path = os.path.join(OUT_DIR, f"{name}.npz")

    np.savez(
        out_path,
        x=x,
        y=y,
        fx=fx_all,
        gy=gy_all,
        model_id=model_all,
        params=params_all,
        param_names=np.array(["a1", "a2", "m", "gamma", "sigma"]),
    )

    print(f"Saved {out_path}")
    print(f"fx shape: {fx_all.shape}")
    print(f"gy shape: {gy_all.shape}")


if __name__ == "__main__":
    assert N_TOTAL == N_TRAIN + N_VAL + N_TEST

    make_split("train", N_TRAIN)
    make_split("val", N_VAL)
    make_split("test", N_TEST)

    print("数据集生成完毕，文件在 ./data/ 文件夹中。")
