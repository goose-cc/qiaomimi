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

    if "gy" in keys:
        gy_key = "gy"
    elif "gy_noisy" in keys:
        gy_key = "gy_noisy"
    else:
        raise KeyError(f"'gy' or 'gy_noisy' not found in {path}, available keys: {keys}")

    fx = data["fx"]
    gy = data[gy_key]

    x = data["x"] if "x" in keys else np.linspace(0, 2, fx.shape[-1])
    y = data["y"] if "y" in keys else np.linspace(2.1, 10, gy.shape[-1])

    return x, y, fx, gy, gy_key


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
    parser = argparse.ArgumentParser(description="Visualize constructed dataset before and after integral.")
    parser.add_argument("--data", type=str, default="data/train.npz", help="Path to dataset npz file.")
    parser.add_argument("--index", type=int, default=0, help="Sample index to visualize.")
    parser.add_argument("--out_dir", type=str, default="picture", help="Output directory.")
    args = parser.parse_args()

    x, y, fx, gy, gy_key = load_npz(args.data)

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

    plot_one(
        x=y,
        values=gy,
        index=args.index,
        title=f"After Integral: g(y), sample {args.index}",
        xlabel="y",
        ylabel=gy_key,
        save_path=os.path.join(args.out_dir, f"sample_{args.index}_after_integral_gy.png"),
    )


if __name__ == "__main__":
    main()
