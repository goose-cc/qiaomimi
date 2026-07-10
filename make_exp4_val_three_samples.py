import os
import numpy as np


INPUT_PATH = "./data_exp4/val.npz"
OUTPUT_DIR = "./data_exp4_val_three_samples"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_one(data, idx, save_path):
    save_dict = {}
    n = len(data["fx"])

    for key in data.files:
        value = data[key]

        # 样本级字段，取第 idx 条
        if hasattr(value, "shape") and value.ndim > 0 and value.shape[0] == n:
            save_dict[key] = value[idx:idx + 1]
        else:
            # x, y, source 等非样本级字段直接保存
            save_dict[key] = value

    np.savez_compressed(save_path, **save_dict)

    print(f"Saved: {save_path}")
    print(f"  index = {idx}")
    print(f"  a1 = {float(data['a1'][idx]):.4f}")
    print(f"  a2 = {float(data['a2'][idx]):.4f}")
    print(f"  m = {float(data['m'][idx]):.4f}")
    print(f"  gamma = {float(data['gamma'][idx]):.4f}")
    print()


def main():
    data = np.load(INPUT_PATH, allow_pickle=True)

    print("Loaded:", INPUT_PATH)
    print("samples:", len(data["fx"]))

    print("unique m:", np.unique(data["m"]))
    print("unique gamma:", np.unique(data["gamma"]))
    print()

    a1 = data["a1"]
    a2 = data["a2"]

    # 三条代表性样本：
    # 1. a1+a2 最小
    # 2. a1/a2 接近中心值 1.0
    # 3. a1+a2 最大
    idx_low = int(np.argmin(a1 + a2))
    idx_mid = int(np.argmin((a1 - 1.0) ** 2 + (a2 - 1.0) ** 2))
    idx_high = int(np.argmax(a1 + a2))

    selected = [
        ("val_sample_low_a1a2.npz", idx_low),
        ("val_sample_mid_a1a2.npz", idx_mid),
        ("val_sample_high_a1a2.npz", idx_high),
    ]

    for filename, idx in selected:
        save_one(
            data=data,
            idx=idx,
            save_path=os.path.join(OUTPUT_DIR, filename),
        )


if __name__ == "__main__":
    main()