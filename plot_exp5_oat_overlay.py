import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_PARAM_NAMES = np.array(["a1", "a2", "m", "gamma"])


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def decode_param_names(param_names):
    if param_names is None:
        return DEFAULT_PARAM_NAMES

    decoded = []
    for item in param_names:
        if isinstance(item, bytes):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))

    return np.array(decoded)


def load_one_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = np.load(path, allow_pickle=False)
    keys = set(data.files)

    required = ["x", "y", "fx", "varying_param"]
    missing = [key for key in required if key not in keys]
    if missing:
        raise KeyError(
            f"{path} is missing keys: {missing}. "
            f"Available keys: {sorted(keys)}"
        )

    if "gy_noisy" in keys:
        gy_for_plot = data["gy_noisy"]
        gy_name = "gy_noisy"
    elif "gy" in keys:
        gy_for_plot = data["gy"]
        gy_name = "gy"
    elif "gy_clean" in keys:
        gy_for_plot = data["gy_clean"]
        gy_name = "gy_clean"
    else:
        raise KeyError(
            f"{path} must contain one of gy_noisy, gy, or gy_clean. "
            f"Available keys: {sorted(keys)}"
        )

    param_names = decode_param_names(
        data["param_names"] if "param_names" in keys else None
    )

    return {
        "x": data["x"],
        "y": data["y"],
        "fx": data["fx"],
        "gy": gy_for_plot,
        "gy_name": gy_name,
        "varying_param": data["varying_param"].astype(np.int64),
        "param_names": param_names,
        "a1": data["a1"] if "a1" in keys else None,
        "a2": data["a2"] if "a2" in keys else None,
        "m": data["m"] if "m" in keys else None,
        "gamma": data["gamma"] if "gamma" in keys else None,
    }


def load_exp5(data_dir, splits):
    loaded = []

    for split in splits:
        path = os.path.join(data_dir, f"{split}.npz")
        loaded.append(load_one_npz(path))

    x = loaded[0]["x"]
    y = loaded[0]["y"]
    param_names = loaded[0]["param_names"]
    gy_name = loaded[0]["gy_name"]

    for item in loaded[1:]:
        if not np.allclose(x, item["x"]):
            raise ValueError("x grids are not consistent across splits.")
        if not np.allclose(y, item["y"]):
            raise ValueError("y grids are not consistent across splits.")
        if list(param_names) != list(item["param_names"]):
            raise ValueError("param_names are not consistent across splits.")

    merged = {
        "x": x,
        "y": y,
        "fx": np.concatenate([item["fx"] for item in loaded], axis=0),
        "gy": np.concatenate([item["gy"] for item in loaded], axis=0),
        "gy_name": gy_name,
        "varying_param": np.concatenate(
            [item["varying_param"] for item in loaded],
            axis=0,
        ),
        "param_names": param_names,
    }

    for name in ["a1", "a2", "m", "gamma"]:
        arrays = [item[name] for item in loaded if item[name] is not None]
        if len(arrays) == len(loaded):
            merged[name] = np.concatenate(arrays, axis=0)
        else:
            merged[name] = None

    return merged


def choose_overlay_indices(indices, sort_values, max_plot):
    indices = np.asarray(indices)

    if len(indices) == 0:
        return indices

    if sort_values is not None:
        order = np.argsort(sort_values[indices])
        indices = indices[order]

    if len(indices) <= max_plot:
        return indices

    # 均匀抽取，避免只画到某一个局部参数范围。
    positions = np.linspace(0, len(indices) - 1, max_plot).astype(np.int64)
    return indices[positions]


def save_overlay_plot(
    axis_values,
    values,
    selected_indices,
    mean_indices,
    xlabel,
    ylabel,
    title,
    save_path,
):
    ensure_dir(os.path.dirname(save_path))

    plt.figure(figsize=(10, 6))

    for idx in selected_indices:
        plt.plot(
            axis_values,
            values[idx],
            alpha=0.12,
            linewidth=1.0,
        )

    mean_curve = values[mean_indices].mean(axis=0)

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
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def format_range(values):
    if values is None or len(values) == 0:
        return ""

    return f", range=[{float(np.min(values)):.4g}, {float(np.max(values)):.4g}]"


def plot_exp5_oat_overlay(data, out_dir, max_plot):
    x = data["x"]
    y = data["y"]
    fx = data["fx"]
    gy = data["gy"]
    gy_name = data["gy_name"]
    varying_param = data["varying_param"]
    param_names = data["param_names"]

    for param_index, param_name in enumerate(param_names):
        all_indices = np.flatnonzero(varying_param == param_index)

        if len(all_indices) == 0:
            print(f"Skip {param_name}: no samples found.")
            continue

        param_values = data.get(param_name, None)
        selected_indices = choose_overlay_indices(
            indices=all_indices,
            sort_values=param_values,
            max_plot=max_plot,
        )

        range_text = ""
        if param_values is not None:
            range_text = format_range(param_values[all_indices])

        info = (
            f"only {param_name} varies "
            f"(N={len(all_indices)}{range_text})"
        )

        save_overlay_plot(
            axis_values=x,
            values=fx,
            selected_indices=selected_indices,
            mean_indices=all_indices,
            xlabel="x",
            ylabel="f(x)",
            title=f"Exp5 f(x) overlay: {info}",
            save_path=os.path.join(
                out_dir,
                f"fx_overlay_only_{param_name}.png",
            ),
        )

        save_overlay_plot(
            axis_values=y,
            values=gy,
            selected_indices=selected_indices,
            mean_indices=all_indices,
            xlabel="y",
            ylabel=gy_name,
            title=f"Exp5 {gy_name} overlay: {info}",
            save_path=os.path.join(
                out_dir,
                f"{gy_name}_overlay_only_{param_name}.png",
            ),
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Exp5 one-parameter-at-a-time overlay plots. "
            "This script only reads existing npz files and saves figures; "
            "it does not regenerate or overwrite datasets."
        )
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data_exp5",
        help="Directory containing exp5 train.npz, val.npz, and test.npz.",
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help=(
            "Dataset splits to merge for plotting. "
            "Examples: --splits train val test, or --splits test"
        ),
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="figures_exp5/exp5_oat_overlay",
        help="Output directory for overlay figures.",
    )

    parser.add_argument(
        "--max_plot",
        type=int,
        default=500,
        help="Maximum number of light overlay curves per parameter group.",
    )

    args = parser.parse_args()

    if args.max_plot <= 0:
        raise ValueError("--max_plot must be positive.")

    data = load_exp5(
        data_dir=args.data_dir,
        splits=args.splits,
    )

    plot_exp5_oat_overlay(
        data=data,
        out_dir=args.out_dir,
        max_plot=args.max_plot,
    )

    print()
    print("Exp5 overlay figures generated successfully.")
    print(f"Data directory: {args.data_dir}")
    print(f"Splits: {args.splits}")
    print(f"Figure directory: {args.out_dir}")
    print("Dataset files were only read, not modified.")


if __name__ == "__main__":
    main()
