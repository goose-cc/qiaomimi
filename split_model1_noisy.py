import os
import numpy as np

INPUT_PATH = "./toy_noisy_model1_data/model1_noisy_100000_pairs.npz"
OUTPUT_DIR = "./data_exp2"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEED = 4026


def save_split(data, indices, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savez_compressed(
        save_path,
        x=data["x"],
        y=data["y"],

        # 训练用字段
        fx=data["fx"][indices],
        gy=data["gy"][indices],

        # 额外保存字段
        fx_clean=data["fx_clean"][indices],
        fx_noisy=data["fx_noisy"][indices],
        gy_clean=data["gy_clean"][indices],
        gy_noisy=data["gy_noisy"][indices],

        model_id=data["model_id"][indices],
        params=data["params"][indices],
        param_names=data["param_names"]
    )


def main():
    data = np.load(INPUT_PATH)

    n = data["fx"].shape[0]

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(n)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    save_split(data, train_idx, os.path.join(OUTPUT_DIR, "train.npz"))
    save_split(data, val_idx, os.path.join(OUTPUT_DIR, "val.npz"))
    save_split(data, test_idx, os.path.join(OUTPUT_DIR, "test.npz"))

    print("Model1 noisy 数据集切分完成")
    print(f"train: {len(train_idx)}")
    print(f"val: {len(val_idx)}")
    print(f"test: {len(test_idx)}")
    print(f"保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
