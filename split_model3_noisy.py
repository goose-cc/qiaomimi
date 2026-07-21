import os
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_split(
    output_path,
    fx,
    gy_noisy,
    gy_clean,
    x,
    y,
    a1,
    a2,
    idx,
    random_seed,
):
    save_dict = {
        "fx": fx[idx],
        "gy_noisy": gy_noisy[idx],
        "x": x,
        "y": y,
        "a1": a1[idx],
        "a2": a2[idx],
        "source": "model3_noisy",
        "random_seed": random_seed,
    }

    if gy_clean is not None:
        save_dict["gy_clean"] = gy_clean[idx]

    np.savez_compressed(output_path, **save_dict)


def split_dataset():
    input_path = "./toy_noisy_model3_data/model3_noisy_100000_pairs.npz"
    output_dir = "./data_exp3"

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    random_seed = 2026

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"找不到输入数据文件: {input_path}\n"
            f"请先运行：python data_generate_model3_noisy.py"
        )

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1")

    ensure_dir(output_dir)

    train_path = os.path.join(output_dir, "train.npz")
    val_path = os.path.join(output_dir, "val.npz")
    test_path = os.path.join(output_dir, "test.npz")

    # 如果已经存在，直接停止，避免误覆盖
    existing_files = [
        path for path in [train_path, val_path, test_path]
        if os.path.exists(path)
    ]

    if existing_files:
        raise FileExistsError(
            "以下切分文件已经存在，为避免覆盖已停止运行：\n"
            + "\n".join(existing_files)
            + "\n如果你确实想重新切分，请先手动删除 data_exp3 里的 train/val/test.npz。"
        )

    data = np.load(input_path, allow_pickle=True)

    print("读取数据:", input_path)
    print("数据字段:", data.files)

    if "fx" not in data.files:
        raise KeyError("数据文件中缺少 fx")

    if "gy_noisy" not in data.files:
        raise KeyError("数据文件中缺少 gy_noisy")

    fx = data["fx"].astype(np.float32)
    gy_noisy = data["gy_noisy"].astype(np.float32)

    gy_clean = (
        data["gy_clean"].astype(np.float32)
        if "gy_clean" in data.files
        else None
    )

    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)

    a1 = data["a1"].astype(np.float32)
    a2 = data["a2"].astype(np.float32)

    num_samples = len(fx)

    if len(gy_noisy) != num_samples:
        raise ValueError(
            f"fx 和 gy_noisy 样本数不一致: fx={len(fx)}, gy_noisy={len(gy_noisy)}"
        )

    print("总样本数:", num_samples)
    print("fx shape:", fx.shape)
    print("gy_noisy shape:", gy_noisy.shape)

    rng = np.random.default_rng(random_seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    print("train:", len(train_idx))
    print("val:", len(val_idx))
    print("test:", len(test_idx))

    save_split(
        train_path,
        fx,
        gy_noisy,
        gy_clean,
        x,
        y,
        a1,
        a2,
        train_idx,
        random_seed,
    )

    save_split(
        val_path,
        fx,
        gy_noisy,
        gy_clean,
        x,
        y,
        a1,
        a2,
        val_idx,
        random_seed,
    )

    save_split(
        test_path,
        fx,
        gy_noisy,
        gy_clean,
        x,
        y,
        a1,
        a2,
        test_idx,
        random_seed,
    )

    print()
    print("Model3 noisy 数据集切分完成")
    print(f"train 保存到: {train_path}")
    print(f"val 保存到: {val_path}")
    print(f"test 保存到: {test_path}")
    print("不会覆盖 data 或 data_exp2。")


if __name__ == "__main__":
    split_dataset()

