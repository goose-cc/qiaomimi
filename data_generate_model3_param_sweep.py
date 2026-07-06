import os
import numpy as np

from config import Config


cfg = Config()


PARAM_NAMES = np.array(
    [
        "a1",
        "a2",
        "m",
        "gamma",
    ]
)


def ensure_dir(path):
    os.makedirs(
        path,
        exist_ok=True,
    )


def assert_not_exists(paths):
    existing = [
        path
        for path in paths
        if os.path.exists(path)
    ]

    if existing:
        raise FileExistsError(
            "以下文件已经存在。"
            "为避免覆盖旧数据，程序已停止：\n"
            + "\n".join(existing)
            + "\n如需重新生成，"
            "请先手动删除这些文件。"
        )


def inclusive_grid(
    start,
    stop,
    step,
):
    count = int(
        np.floor(
            (stop - start) / step
        )
    ) + 1

    values = (
        start
        + np.arange(
            count,
            dtype=np.float64,
        )
        * step
    )

    if values[-1] < stop - 1e-12:
        values = np.append(
            values,
            stop,
        )
    else:
        values[-1] = stop

    return values.astype(np.float32)


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

    dx = (
        (x[-1] - x[0])
        / (len(x) - 1)
    )

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


def model3_fx(
    x,
    params,
):
    x_grid = x[None, :]

    a1 = params[:, 0:1]
    a2 = params[:, 1:2]
    m = params[:, 2:3]
    gamma = params[:, 3:4]

    width = a1 * m * gamma

    peak = (
        (1.0 / np.pi)
        * width
        / (
            (x_grid - m**2) ** 2
            + width**2
        )
    )

    background = (
        a2 * x_grid / 5.0
    )

    return (
        peak + background
    ).astype(np.float32)


def add_relative_white_noise(
    gy_clean,
    noise_level,
    rng,
):
    rms = np.sqrt(
        np.mean(
            gy_clean**2,
            axis=1,
            keepdims=True,
        )
    )

    sigma = (
        np.maximum(
            rms,
            1e-12,
        )
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


def build_one_parameter_at_a_time_truths(
    step,
):
    """
    固定 3 个参数，只改变 1 个参数。

    基准：

    a1 = 1.0
    a2 = 1.0
    m = 0.8
    Gamma = 0.5
    """

    baseline = np.array(
        [
            cfg.model3_truth_a1,
            cfg.model3_truth_a2,
            cfg.model3_train_m,
            cfg.model3_train_gamma,
        ],
        dtype=np.float32,
    )

    ranges = [
        inclusive_grid(
            cfg.model3_a1_min,
            cfg.model3_a1_max,
            step,
        ),
        inclusive_grid(
            cfg.model3_a2_min,
            cfg.model3_a2_max,
            step,
        ),
        inclusive_grid(
            cfg.model3_m_min,
            cfg.model3_m_max,
            step,
        ),
        inclusive_grid(
            cfg.model3_gamma_min,
            cfg.model3_gamma_max,
            step,
        ),
    ]

    params = []
    varying_param = []

    for param_index, values in enumerate(ranges):

        # 先复制基准参数
        block = np.repeat(
            baseline[None, :],
            len(values),
            axis=0,
        )

        # 只改变当前一个参数
        block[:, param_index] = values

        params.append(block)

        varying_param.append(
            np.full(
                len(values),
                param_index,
                dtype=np.int8,
            )
        )

    params = np.concatenate(
        params,
        axis=0,
    )

    varying_param = np.concatenate(
        varying_param,
        axis=0,
    )

    # 去掉重复的基准参数
    unique_params, unique_indices = np.unique(
        params,
        axis=0,
        return_index=True,
    )

    # 恢复原来的参数扫描顺序
    order = np.argsort(unique_indices)

    return (
        unique_params[order],
        varying_param[
            unique_indices[order]
        ],
    )


def resolve_step_for_unique_pairs(
    target_pairs,
):
    """
    首先尝试步长 0.01。

    如果固定三变一产生的数据数量不足，
    自动缩小步长，
    直到不同真值数量达到要求。
    """

    initial_step = (
        cfg.model3_param_step
    )

    params, labels = (
        build_one_parameter_at_a_time_truths(
            initial_step
        )
    )

    if len(params) >= target_pairs:
        return (
            initial_step,
            params,
            labels,
        )

    total_span = (
        (
            cfg.model3_a1_max
            - cfg.model3_a1_min
        )
        + (
            cfg.model3_a2_max
            - cfg.model3_a2_min
        )
        + (
            cfg.model3_m_max
            - cfg.model3_m_min
        )
        + (
            cfg.model3_gamma_max
            - cfg.model3_gamma_min
        )
    )

    step = (
        total_span
        / max(
            target_pairs - 4,
            1,
        )
    )

    for _ in range(20):
        params, labels = (
            build_one_parameter_at_a_time_truths(
                step
            )
        )

        if len(params) >= target_pairs:
            return (
                step,
                params,
                labels,
            )

        step *= 0.999

    raise RuntimeError(
        f"自动缩小步长后仍不足 "
        f"{target_pairs} 组不同真值，"
        f"最后得到 {len(params)} 组。"
    )


def generate_from_params(
    params,
    varying_param,
    seed,
    source_name,
    used_step,
):
    x, y, kernel = build_grids_and_kernel()

    rng = np.random.default_rng(seed)

    # 根据精确参数产生真实 f(x)
    fx_true = model3_fx(
        x,
        params,
    )

    # 对 f(x) 做积分变换
    gy_clean = (
        fx_true @ kernel.T
    ).astype(np.float32)

    # 只在积分后的 g(y) 上加 1% 白噪声
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
        "a1": params[:, 0],
        "a2": params[:, 1],
        "m": params[:, 2],
        "gamma": params[:, 3],
        "params": params,
        "param_names": PARAM_NAMES,
        "varying_param": varying_param,
        "varying_param_names": PARAM_NAMES,
        "noise_level": np.float32(
            cfg.model3_white_noise_level
        ),
        "requested_step": np.float32(
            cfg.model3_param_step
        ),
        "used_step": np.float32(
            used_step
        ),
        "source": source_name,
    }


def save_dataset(
    path,
    dataset,
):
    np.savez_compressed(
        path,
        **dataset,
    )

    print(f"Saved: {path}")

    print(
        f"  pairs: "
        f"{len(dataset['fx'])}"
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
        "  used parameter step: "
        f"{float(dataset['used_step']):.8f}"
    )


def main():
    output_dir = "./data_exp5"

    ensure_dir(output_dir)

    train_path = os.path.join(
        output_dir,
        "train.npz",
    )

    val_path = os.path.join(
        output_dir,
        "val.npz",
    )

    test_path = os.path.join(
        output_dir,
        "test.npz",
    )

    assert_not_exists(
        [
            train_path,
            val_path,
            test_path,
        ]
    )

    n_train = (
        cfg.model3_sweep_train_pairs
    )

    n_val = (
        cfg.model3_sweep_val_pairs
    )

    n_test = (
        cfg.model3_sweep_test_pairs
    )

    total_pairs = (
        n_train
        + n_val
        + n_test
    )

    used_step, truth_params, truth_labels = (
        resolve_step_for_unique_pairs(
            total_pairs
        )
    )

    rng = np.random.default_rng(
        cfg.random_seed + 10
    )

    # 随机打乱所有不同参数真值
    order = rng.permutation(
        len(truth_params)
    )[:total_pairs]

    truth_params = truth_params[order]
    truth_labels = truth_labels[order]

    # =====================
    # train
    # =====================
    train_params = truth_params[
        :n_train
    ]

    train_labels = truth_labels[
        :n_train
    ]

    # =====================
    # val
    # =====================
    val_start = n_train
    val_end = val_start + n_val

    val_params = truth_params[
        val_start:val_end
    ]

    val_labels = truth_labels[
        val_start:val_end
    ]

    # =====================
    # test
    # =====================
    test_params = truth_params[
        val_end:val_end + n_test
    ]

    test_labels = truth_labels[
        val_end:val_end + n_test
    ]

    print(
        "用户请求初始步长: "
        f"{cfg.model3_param_step}"
    )

    print(
        "固定三变一在 0.01 步长下"
        f"不足 {total_pairs} 组不同真值。"
    )

    print(
        "自动缩小后的实际步长: "
        f"{used_step:.8f}"
    )

    print(
        "可用不同真值数量: "
        f"{len(truth_params)}"
    )

    print(
        "train/val/test "
        "使用互不重复的参数真值。"
    )

    train_data = generate_from_params(
        train_params,
        train_labels,
        seed=cfg.random_seed + 20,
        source_name=(
            "model3_oat_unique_train"
        ),
        used_step=used_step,
    )

    val_data = generate_from_params(
        val_params,
        val_labels,
        seed=cfg.random_seed + 21,
        source_name=(
            "model3_oat_unique_val"
        ),
        used_step=used_step,
    )

    test_data = generate_from_params(
        test_params,
        test_labels,
        seed=cfg.random_seed + 22,
        source_name=(
            "model3_oat_unique_test"
        ),
        used_step=used_step,
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
        test_path,
        test_data,
    )

    print()

    print(
        "Model 3 exp5 参数范围扩展"
        "数据生成完成。"
    )

    print(
        "规则：固定 3 个参数，"
        "只改变 1 个参数。"
    )

    print(
        "每一对数据使用一个精确参数真值"
        "和对应的 1% noisy g(y)。"
    )

    print(
        "train/val/test "
        "的参数真值不重复。"
    )

    print(
        "旧数据目录不会被覆盖。"
    )


if __name__ == "__main__":
    main()
