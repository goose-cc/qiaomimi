import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def load_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = np.load(path)
    keys = data.files

    if "fx" not in keys:
        raise KeyError(f"'fx' not found in {path}, available keys: {keys}")

    fx = data["fx"]

    gy_clean = data["gy_clean"] if "gy_clean" in keys else None
    gy_noisy = data["gy_noisy"] if "gy_noisy" in keys else None
    gy = data["gy"] if "gy" in keys else None

    if gy is None and gy_clean is None and gy_noisy is None:
        raise KeyError(
            f"'gy', 'gy_clean', and 'gy_noisy' are all missing in {path}, available keys: {keys}"
        )

    x = data["x"] if "x" in keys else np.linspace(0, 2, fx.shape[-1])

    if gy is not None:
        y_len = gy.shape[-1]
    elif gy_clean is not None:
        y_len = gy_clean.shape[-1]
    else:
        y_len = gy_noisy.shape[-1]

    y = data["y"] if "y" in keys else np.linspace(2.1, 10, y_len)

    return x, y, fx, gy, gy_clean, gy_noisy



def plot_one(x, values, index, title, xlabel, ylabel, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(x, values[index])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset pairs.")
    parser.add_argument("--data", type=str, default="data/train.npz", help="Path to dataset npz file.")
    parser.add_argument("--index", type=int, default=0, help="Sample index to visualize.")
    parser.add_argument("--out_dir", type=str, default="picture", help="Output directory.")
    args = parser.parse_args()

    x, y, fx, gy, gy_clean, gy_noisy = load_npz(args.data)

    if args.index < 0 or args.index >= len(fx):
        raise IndexError(f"index {args.index} out of range, dataset size = {len(fx)}")

    plot_one(
        x=x,
        values=fx,
        index=args.index,
        title=f"Before Integral: f(x), sample {args.index}",
        xlabel="x",
        ylabel="f(x)",
        save_path=os.path.join(args.out_dir, f"sample_{args.index}_before_integral_fx.png"),
    )

    if gy is not None:
        plot_one(
            x=y,
            values=gy,
            index=args.index,
            title=f"After Integral: g(y), sample {args.index}",
            xlabel="y",
            ylabel="gy",
            save_path=os.path.join(args.out_dir, f"sample_{args.index}_after_integral_gy.png"),
        )

    if gy_clean is not None:
        plot_one(
            x=y,
            values=gy_clean,
            index=args.index,
            title=f"After Integral (Clean): g_clean(y), sample {args.index}",
            xlabel="y",
            ylabel="gy_clean",
            save_path=os.path.join(args.out_dir, f"sample_{args.index}_after_integral_gy_clean.png"),
        )

    if gy_noisy is not None:
        plot_one(
            x=y,
            values=gy_noisy,
            index=args.index,
            title=f"After Integral (Noisy): g_noisy(y), sample {args.index}",
            xlabel="y",
            ylabel="gy_noisy",
            save_path=os.path.join(args.out_dir, f"sample_{args.index}_after_integral_gy_noisy.png"),
        )
