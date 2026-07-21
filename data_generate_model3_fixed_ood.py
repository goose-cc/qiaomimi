import os
import numpy as np

from config import Config


cfg = Config()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def assert_not_exists(paths):
    existing = [path for path in paths if os.path.exists(path)]

    if existing:
        raise FileExistsError(
            "以下文件已经存在。为避免覆盖旧数据，程序已停止：\n"
            + "\n".join(existing)
            + "\n如需重新生成，请先手动删除这些文件。"
        )


def inclusive_grid(start, stop, step):
    count = int(round((stop - start) / step)) + 1

    return np.linspace(
        start,
        stop,
        count,
        dtype=np.float32,
    )


def build_grids_and_kernel():
    x = np.linspace(
        cfg.x_min,
        cfg.x_max,
        cfg.num_x_points,
        dtype=np.float32,
    )

    y = np.linspace(
        cfg.y_min,
        cfg.y_max,
        cfg.num_y_points,
        dtype=np.float32,
    )

    dx = (x[-1] - x[0]) / (len(x) - 1)

    weights = np.ones_like(
        x,
        dtype=np.float32,
    ) * dx

    weights[0] *= 0.5
    weights[-1] *= 0.5

    kernel = weights[None, :] / (
        y[:, None] - x[None, :]
    )

    return (
        x,
        y,
        kernel.astype(np.float32),
    )


def model3_fx(x, a1, a2, m, gamma):
    """
    根据精确真值参数生成 Model 3 的 f(x)。
    """

    x_grid = x[None, :]

    a1 = np.asarray(
        a1,
        dtype=np.float32,
    ).reshape(-1, 1)

    a2 = np.asarray(
        a2,
        dtype=np.float32,
    ).reshape(-1, 1)

    m = np.asarray(
        m,
        dtype=np.float32,
    ).reshape(-1, 1)

    gamma = np.asarray(
        gamma,
        dtype=np.float32,
    ).reshape(-1, 1)

    width = a1 * m * gamma

    peak = (
        (1.0 / np.pi)
        * width
        / (
            (x_grid - m**2) ** 2
            + width**2
        )
    )

    background = a2 * x_grid / 5.0

    return (
        peak + background
    ).astype(np.float32)


def add_relative_white_noise(
    gy_clean,
    noise_level,
    rng,
):
    """
    对积分后的 g(y) 加相对 RMS 幅值为 1% 的
    加性高斯白噪声。

    sigma = noise_level * RMS(g)

    gy_noisy = gy_clean + N(0, sigma^2)
    """

    rms = np.sqrt(
        np.mean(
            gy_clean**2,
            axis=1,
            keepdims=True,
        )
    )

    sigma = (
        np.maximum(rms, 1e-12)
        * noise_level
    )

    noise = rng.normal(
        0.0,
        1.0,
        size=gy_clean.shape,
    ).astype(np.float32)

    noise *= sigma.astype(np.float32)

    return (
        gy_clean + noise
    ).astype(np.float32)


def build_a1_a2_truth_pairs():
    """
    a1 / a2 从 0.7 到 1.3，
    步长 0.01。

    每个参数取值本身就是真值。

    不再给 a1 / a2 加参数噪声。
    """

    a1_values = inclusive_grid(
        cfg.model3_a1_min,
        cfg.model3_a1_max,
        cfg.model3_param_step,
    )

    a2_values = inclusive_grid(
        cfg.model3_a2_min,
        cfg.model3_a2_max,
        cfg.model3_param_step,
    )

    a1_mesh, a2_mesh = np.meshgrid(
        a1_values,
        a2_values,
        indexing="ij",
    )

    return np.column_stack(
        [
            a1_mesh.ravel(),
            a2_mesh.ravel(),
        ]
    ).astype(np.float32)


def sample_truth_pairs(
    truth_pairs,
    n_samples,
    seed,
):
    """
    从完整 a1/a2 真值网格中抽样。

    当 n_samples <= 网格大小时：
        无放回随机抽样。
        强制保留四个参数边界角点。

    当 n_samples > 网格大小时：
        先完整覆盖整个参数网格若干次。
        剩余样本再无放回随机抽样。

    同一个参数真值可以对应多个独立噪声观测。
    """

    rng = np.random.default_rng(seed)

    n_truths = len(truth_pairs)

    if n_samples <= 0:
        raise ValueError(
            f"n_samples 必须大于 0，"
            f"当前为 {n_samples}"
        )

    if n_samples <= n_truths:
        a1 = truth_pairs[:, 0]
        a2 = truth_pairs[:, 1]

        corner_mask = (
            np.isin(
                a1,
                [
                    a1.min(),
                    a1.max(),
                ],
            )
            & np.isin(
                a2,
                [
                    a2.min(),
                    a2.max(),
                ],
            )
        )

        corner_indices = np.flatnonzero(
            corner_mask
        )

        if n_samples < len(corner_indices):
            selected_indices = rng.choice(
                n_truths,
                size=n_samples,
                replace=False,
            )

        else:
            remaining_indices = np.setdiff1d(
                np.arange(n_truths),
                corner_indices,
                assume_unique=True,
            )

            extra_count = (
                n_samples
                - len(corner_indices)
            )

            extra_indices = rng.choice(
                remaining_indices,
                size=extra_count,
                replace=False,
            )

            selected_indices = np.concatenate(
                [
                    corner_indices,
                    extra_indices,
                ]
            )

            rng.shuffle(
                selected_indices
            )

    else:
        full_repeat_count = (
            n_samples // n_truths
        )

        remainder = (
            n_samples % n_truths
        )

        selected_indices = np.tile(
            np.arange(n_truths),
            full_repeat_count,
        )

        if remainder > 0:
            remainder_indices = rng.choice(
                n_truths,
                size=remainder,
                replace=False,
            )

            selected_indices = np.concatenate(
                [
                    selected_indices,
                    remainder_indices,
                ]
            )

        rng.shuffle(
            selected_indices
        )

    return truth_pairs[
        selected_indices
    ].copy()


def generate_fixed_dataset(
    selected_pairs,
    m_value,
    gamma_value,
    noise_seed,
    source_name,
):
    """
    使用已经选定的 a1/a2 真值对生成数据。

    参数真值抽样和噪声生成分开。

    这样两个 OOD 测试集可以使用完全相同的
    a1/a2 真值。
    """

    x, y, kernel = build_grids_and_kernel()

    rng = np.random.default_rng(
        noise_seed
    )

    selected_pairs = np.asarray(
        selected_pairs,
        dtype=np.float32,
    )

    if (
        selected_pairs.ndim != 2
        or selected_pairs.shape[1] != 2
    ):
        raise ValueError(
            "selected_pairs 必须是 "
            "shape=(N, 2) 的 a1/a2 数组。"
        )

    n_samples = len(
        selected_pairs
    )

    a1 = selected_pairs[:, 0]
    a2 = selected_pairs[:, 1]

    m = np.full(
        n_samples,
        m_value,
        dtype=np.float32,
    )

    gamma = np.full(
        n_samples,
        gamma_value,
        dtype=np.float32,
    )

    # =====================
    # 第一步：
    # 根据精确真值参数生成 f(x)
    # =====================
    fx_true = model3_fx(
        x,
        a1,
        a2,
        m,
        gamma,
    )

    # =====================
    # 第二步：
    # 对 f(x) 做积分变换
    # 得到 clean g(y)
    # =====================
    gy_clean = (
        fx_true @ kernel.T
    ).astype(np.float32)

    # =====================
    # 第三步：
    # 只在积分后的 g(y)
    # 加 1% 白噪声
    # =====================
    gy_noisy = add_relative_white_noise(
        gy_clean,
        noise_level=(
            cfg.model3_white_noise_level
        ),
        rng=rng,
    )

    return {
        "x": x,
        "y": y,
        "fx": fx_true,
        "gy_clean": gy_clean,
        "gy_noisy": gy_noisy,
        "a1": a1,
        "a2": a2,
        "m": m,
        "gamma": gamma,
        "noise_level": np.float32(
            cfg.model3_white_noise_level
        ),
        "source": source_name,
        "a1_a2_grid_step": np.float32(
            cfg.model3_param_step
        ),
    }


def save_dataset(
    path,
    dataset,
):
    np.savez_compressed(
        path,
        **dataset,
    )

    print(
        f"Saved: {path}"
    )

    print(
        f"  fx shape: "
        f"{dataset['fx'].shape}"
    )

    print(
        f"  gy_noisy shape: "
        f"{dataset['gy_noisy'].shape}"
    )

    print(
        "  parameter ranges: "
        f"a1=["
        f"{dataset['a1'].min():.2f}, "
        f"{dataset['a1'].max():.2f}], "
        f"a2=["
        f"{dataset['a2'].min():.2f}, "
        f"{dataset['a2'].max():.2f}], "
        f"m={dataset['m'][0]:.2f}, "
        f"gamma={dataset['gamma'][0]:.2f}"
    )


def main():
    output_dir = "./data_exp4"

    ensure_dir(
        output_dir
    )

    train_path = os.path.join(
        output_dir,
        "train.npz",
    )

    val_path = os.path.join(
        output_dir,
        "val.npz",
    )

    test_gamma05_path = os.path.join(
        output_dir,
        "test_m1p0_gamma0p5.npz",
    )

    test_gamma03_path = os.path.join(
        output_dir,
        "test_m1p0_gamma0p3.npz",
    )

    assert_not_exists(
        [
            train_path,
            val_path,
            test_gamma05_path,
            test_gamma03_path,
        ]
    )

    # =====================
    # 完整 a1/a2 真值网格
    # =====================
    truth_pairs = (
        build_a1_a2_truth_pairs()
    )

    # =====================
    # 训练集真值
    # =====================
    train_pairs = sample_truth_pairs(
        truth_pairs=truth_pairs,
        n_samples=(
            cfg.model3_fixed_train_pairs
        ),
        seed=cfg.random_seed,
    )

    # =====================
    # 验证集真值
    # =====================
    val_pairs = sample_truth_pairs(
        truth_pairs=truth_pairs,
        n_samples=(
            cfg.model3_fixed_val_pairs
        ),
        seed=cfg.random_seed + 1,
    )

    # =====================
    # OOD 测试共用同一批
    # a1/a2 真值
    # =====================
    test_pairs = sample_truth_pairs(
        truth_pairs=truth_pairs,
        n_samples=(
            cfg.model3_fixed_test_pairs
        ),
        seed=cfg.random_seed + 2,
    )

    # =====================
    # 训练集
    # m=0.8
    # Gamma=0.5
    # =====================
    train_data = generate_fixed_dataset(
        selected_pairs=train_pairs,
        m_value=cfg.model3_train_m,
        gamma_value=(
            cfg.model3_train_gamma
        ),
        noise_seed=(
            cfg.random_seed + 100
        ),
        source_name=(
            "model3_fixed_train_"
            "m0p8_gamma0p5"
        ),
    )

    # =====================
    # 验证集
    # m=0.8
    # Gamma=0.5
    # =====================
    val_data = generate_fixed_dataset(
        selected_pairs=val_pairs,
        m_value=cfg.model3_train_m,
        gamma_value=(
            cfg.model3_train_gamma
        ),
        noise_seed=(
            cfg.random_seed + 101
        ),
        source_name=(
            "model3_fixed_val_"
            "m0p8_gamma0p5"
        ),
    )

    # =====================
    # OOD 测试 1
    #
    # m=1.0
    # Gamma=0.5
    # =====================
    test_gamma05_data = generate_fixed_dataset(
        selected_pairs=test_pairs,
        m_value=cfg.model3_test_m,
        gamma_value=(
            cfg.model3_test_gamma_1
        ),
        noise_seed=(
            cfg.random_seed + 102
        ),
        source_name=(
            "model3_ood_test_"
            "m1p0_gamma0p5"
        ),
    )

    # =====================
    # OOD 测试 2
    #
    # 使用完全相同的 a1/a2
    #
    # m=1.0
    # Gamma=0.3
    #
    # noise_seed 与测试1一致。
    #
    # 两个测试使用相同标准高斯噪声模式，
    # 但 sigma 仍按照各自 g(y) 的 RMS
    # 乘 1% 分别计算。
    # =====================
    test_gamma03_data = generate_fixed_dataset(
        selected_pairs=test_pairs,
        m_value=cfg.model3_test_m,
        gamma_value=(
            cfg.model3_test_gamma_2
        ),
        noise_seed=(
            cfg.random_seed + 102
        ),
        source_name=(
            "model3_ood_test_"
            "m1p0_gamma0p3"
        ),
    )

    save_dataset(
        train_path,
        train_data,
    )

    save_dataset(
        val_path,
        val_data,
    )

    save_dataset(
        test_gamma05_path,
        test_gamma05_data,
    )

    save_dataset(
        test_gamma03_path,
        test_gamma03_data,
    )

    print()

    print(
        "Model 3 exp4 数据生成完成。"
    )

    print(
        "训练/验证: "
        "m=0.8, Gamma=0.5"
    )

    print(
        "OOD 测试1: "
        "m=1.0, Gamma=0.5"
    )

    print(
        "OOD 测试2: "
        "m=1.0, Gamma=0.3"
    )

    print(
        "a1/a2 是 0.7~1.3、"
        "步长 0.01 网格中的精确真值。"
    )

    print(
        "两个 OOD 测试集使用同一批 "
        "a1/a2 真值对，"
        "便于公平比较 Gamma 改变带来的影响。"
    )

    print(
        "数据对定义："
        "输入 gy_noisy，"
        "标签 fx_true；"
        "只在积分后的 g(y) "
        "加 1% 白噪声。"
    )

    print(
        "旧数据目录不会被覆盖。"
    )


if __name__ == "__main__":
    main()
