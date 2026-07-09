import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


PARAM_NAMES = ["a1", "a2", "m", "Gamma"]
PARAM_KEYS = ["a1", "a2", "m", "gamma"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_dataset(path):
    """
    读取 exp4 / exp5 绘图所需字段。

    exp4:
        x, y, fx, gy_noisy
        没有 varying_param

    exp5:
        x, y, fx, gy_noisy
        params, varying_param
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    with np.load(path) as data:
        keys = set(data.files)

        if "fx" not in keys:
            raise KeyError(
                f"'fx' not found in {path}. "
                f"Available keys: {sorted(keys)}"
            )

        if "gy_noisy" in keys:
            gy = data["gy_noisy"].copy()
            gy_name = "gy_noisy"
        elif "gy" in keys:
            gy = data["gy"].copy()
            gy_name = "gy"
        else:
            raise KeyError(
                f"'gy_noisy' or 'gy' not found in {path}. "
                f"Available keys: {sorted(keys)}"
            )

        fx = data["fx"].copy()

        x = (
            data["x"].copy()
            if "x" in keys
            else np.linspace(
                0.0,
                2.0,
                fx.shape[-1],
                dtype=np.float32,
            )
        )

        y = (
            data["y"].copy()
            if "y" in keys
            else np.linspace(
                3.0,
                8.0,
                gy.shape[-1],
                dtype=np.float32,
            )
        )

        params = (
            data["params"].copy()
            if "params" in keys
            else None
        )

        varying_param = (
            data["varying_param"].copy()
            if "varying_param" in keys
            else None
        )

        parameter_arrays = {}

        for key in PARAM_KEYS:
            if key in keys:
                parameter_arrays[key] = data[key].copy()

    return {
        "x": x,
        "y": y,
        "fx": fx,
        "gy": gy,
        "gy_name": gy_name,
        "params": params,
        "varying_param": varying_param,
        "parameter_arrays": parameter_arrays,
    }


def save_overlay_plot(
    axis_values,
    curves_subset,
    mean_curve,
    xlabel,
    ylabel,
    title,
    save_path,
):
    """
    叠加图逻辑：

    浅色细线：
        多条样本曲线。

    粗线 Mean curve：
        当前完整数据组的平均曲线。

    注意：
        mean_curve 不是只对显示出来的 500 条求平均。
        它由该组全部样本计算。
    """

    ensure_dir(
        os.path.dirname(save_path)
    )

    plt.figure(figsize=(10, 6))

    for curve in curves_subset:
        plt.plot(
            axis_values,
            curve,
            alpha=0.08,
            linewidth=0.8,
        )

    plt.plot(
        axis_values,
        mean_curve,
        linewidth=3.0,
        label="Mean curve",
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.close()

    print(f"Saved: {save_path}")


def plot_exp4_overlay(
    dataset,
    dataset_name,
    out_dir,
    max_plot,
):
    """
    exp4 使用原 Model3 noisy / exp3 的叠加图思想：

    整个数据文件作为一组。

    生成：
        f(x) overlay + mean
        gy_noisy overlay + mean

    显示前 max_plot 条曲线。
    Mean curve 使用整个数据文件的全部样本。
    """

    fx = dataset["fx"]
    gy = dataset["gy"]

    show_n = min(
        max_plot,
        len(fx),
    )

    show_indices = np.arange(
        show_n,
        dtype=np.int64,
    )

    mean_fx = fx.mean(axis=0)
    mean_gy = gy.mean(axis=0)

    save_overlay_plot(
        axis_values=dataset["x"],
        curves_subset=fx[show_indices],
        mean_curve=mean_fx,
        xlabel="x",
        ylabel="f(x)",
        title=(
            f"Exp4 {dataset_name} f(x): "
            "Overlay and Mean Curve"
        ),
        save_path=os.path.join(
            out_dir,
            "fx_overlay_mean.png",
        ),
    )

    save_overlay_plot(
        axis_values=dataset["y"],
        curves_subset=gy[show_indices],
        mean_curve=mean_gy,
        xlabel="y",
        ylabel=dataset["gy_name"],
        title=(
            f"Exp4 {dataset_name} "
            f"{dataset['gy_name']}: "
            "Overlay and Mean Curve"
        ),
        save_path=os.path.join(
            out_dir,
            "gy_noisy_overlay_mean.png",
        ),
    )

    print(
        f"Exp4 {dataset_name}: "
        f"total={len(fx)}, "
        f"displayed={len(show_indices)}"
    )


def get_parameter_values(
    dataset,
    param_index,
):
    """
    exp5 优先从 params 四列读取参数。

    params[:, 0] -> a1
    params[:, 1] -> a2
    params[:, 2] -> m
    params[:, 3] -> gamma
    """

    if dataset["params"] is not None:
        return dataset["params"][
            :,
            param_index,
        ]

    key = PARAM_KEYS[param_index]

    if key not in dataset["parameter_arrays"]:
        raise KeyError(
            f"Cannot find parameter array: {key}"
        )

    return dataset["parameter_arrays"][key]


def select_range_covered_indices(
    group_indices,
    group_values,
    max_plot,
):
    """
    exp5 的每一组都是一个参数扫描轴。

    为了让叠加图覆盖当前参数的完整范围：

    1. 按正在变化的参数排序；
    2. 从最小值到最大值等间隔选择最多 max_plot 条。

    例如 varying m：
        不只画 m 范围中某一小段，
        而是让显示曲线覆盖 m=0.1~2.0。
    """

    order = np.argsort(group_values)

    sorted_indices = group_indices[
        order
    ]

    show_n = min(
        max_plot,
        len(sorted_indices),
    )

    if show_n == len(sorted_indices):
        return sorted_indices

    positions = np.linspace(
        0,
        len(sorted_indices) - 1,
        show_n,
    )

    positions = np.rint(
        positions
    ).astype(np.int64)

    positions = np.unique(
        positions
    )

    return sorted_indices[
        positions
    ]


def plot_exp5_grouped_overlay(
    dataset,
    dataset_name,
    out_dir,
    max_plot,
):
    """
    exp5 按 varying_param 分为四组：

        0 -> varying a1
        1 -> varying a2
        2 -> varying m
        3 -> varying Gamma

    每组生成两张图：

        f(x) overlay + mean
        gy_noisy overlay + mean
    """

    varying_param = dataset[
        "varying_param"
    ]

    if varying_param is None:
        raise KeyError(
            "Exp5 grouped overlay requires "
            "'varying_param'."
        )

    fx = dataset["fx"]
    gy = dataset["gy"]

    for param_index, param_name in enumerate(
        PARAM_NAMES
    ):
        group_indices = np.flatnonzero(
            varying_param
            == param_index
        )

        if len(group_indices) == 0:
            print(
                f"Skip varying {param_name}: "
                "no samples"
            )
            continue

        parameter_values = get_parameter_values(
            dataset,
            param_index,
        )

        group_values = parameter_values[
            group_indices
        ]

        show_indices = (
            select_range_covered_indices(
                group_indices=group_indices,
                group_values=group_values,
                max_plot=max_plot,
            )
        )

        mean_fx = fx[
            group_indices
        ].mean(
            axis=0
        )

        mean_gy = gy[
            group_indices
        ].mean(
            axis=0
        )

        safe_name = (
            "gamma"
            if param_name == "Gamma"
            else param_name
        )

        group_dir = os.path.join(
            out_dir,
            f"varying_{safe_name}",
        )

        value_min = float(
            group_values.min()
        )

        value_max = float(
            group_values.max()
        )

        range_text = (
            f"[{value_min:.6g}, "
            f"{value_max:.6g}]"
        )

        save_overlay_plot(
            axis_values=dataset["x"],
            curves_subset=fx[show_indices],
            mean_curve=mean_fx,
            xlabel="x",
            ylabel="f(x)",
            title=(
                f"Exp5 {dataset_name} | "
                f"Varying {param_name} {range_text} | "
                "f(x) Overlay and Mean Curve"
            ),
            save_path=os.path.join(
                group_dir,
                "fx_overlay_mean.png",
            ),
        )

        save_overlay_plot(
            axis_values=dataset["y"],
            curves_subset=gy[show_indices],
            mean_curve=mean_gy,
            xlabel="y",
            ylabel=dataset["gy_name"],
            title=(
                f"Exp5 {dataset_name} | "
                f"Varying {param_name} {range_text} | "
                f"{dataset['gy_name']} "
                "Overlay and Mean Curve"
            ),
            save_path=os.path.join(
                group_dir,
                "gy_noisy_overlay_mean.png",
            ),
        )

        print(
            f"varying {param_name}: "
            f"group samples={len(group_indices)}, "
            f"displayed={len(show_indices)}, "
            f"range={range_text}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate exp4 whole-dataset overlays "
            "or exp5 OAT grouped overlays."
        )
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset npz file.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./picture/dataset_overlay",
        help="Output directory.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=[
            "auto",
            "exp4",
            "exp5",
        ],
        help=(
            "auto: exp5 when varying_param exists; "
            "otherwise exp4."
        ),
    )

    parser.add_argument(
        "--max_plot",
        type=int,
        default=500,
        help=(
            "Maximum number of light curves "
            "shown in each overlay."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_plot <= 0:
        raise ValueError(
            "--max_plot must be greater than 0."
        )

    dataset = load_dataset(
        args.data
    )

    dataset_name = os.path.splitext(
        os.path.basename(
            args.data
        )
    )[0]

    mode = args.mode

    if mode == "auto":
        mode = (
            "exp5"
            if dataset["varying_param"]
            is not None
            else "exp4"
        )

    print(f"Dataset: {args.data}")
    print(f"Mode: {mode}")
    print(
        f"fx shape: "
        f"{dataset['fx'].shape}"
    )
    print(
        f"{dataset['gy_name']} shape: "
        f"{dataset['gy'].shape}"
    )

    if mode == "exp4":
        plot_exp4_overlay(
            dataset=dataset,
            dataset_name=dataset_name,
            out_dir=args.out_dir,
            max_plot=args.max_plot,
        )

    elif mode == "exp5":
        plot_exp5_grouped_overlay(
            dataset=dataset,
            dataset_name=dataset_name,
            out_dir=args.out_dir,
            max_plot=args.max_plot,
        )

    else:
        raise ValueError(
            f"Unsupported mode: {mode}"
        )


if __name__ == "__main__":
    main()
