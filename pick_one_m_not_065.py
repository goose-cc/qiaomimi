import os
import numpy as np


INPUT_FILES = [
    ("train", "./data_exp5/train.npz"),
    ("val", "./data_exp5/val.npz"),
    ("test", "./data_exp5/test.npz"),
]

OUTPUT_DIR = "./data_exp5_selected_one_m_not_065"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 目标：找 m≈1.0, gamma≈0.5 的样本
# 这样 m^2≈1.0，不是 0.65
TARGET_M = 1.3
TARGET_GAMMA = 0.5


def find_best_sample():
    best = None
    best_score = float("inf")

    for split_name, path in INPUT_FILES:
        print(f"正在读取 {path} ...", flush=True)

        data = np.load(path, allow_pickle=True)

        m = data["m"].astype(np.float64)
        gamma = data["gamma"].astype(np.float64)

        score = (m - TARGET_M) ** 2 + (gamma - TARGET_GAMMA) ** 2

        local_idx = int(np.argmin(score))
        local_score = float(score[local_idx])

        print(
            f"{split_name}: best local_idx={local_idx}, "
            f"m={m[local_idx]:.6f}, gamma={gamma[local_idx]:.6f}, "
            f"m^2={m[local_idx] ** 2:.6f}, score={local_score:.8f}",
            flush=True
        )

        if local_score < best_score:
            best_score = local_score
            best = {
                "split": split_name,
                "path": path,
                "local_idx": local_idx,
                "m": float(m[local_idx]),
                "gamma": float(gamma[local_idx]),
            }

    return best


def save_one_sample(best):
    data = np.load(best["path"], allow_pickle=True)
    n = len(data["fx"])
    idx = best["local_idx"]

    save_dict = {}

    for key in data.files:
        value = data[key]

        if hasattr(value, "shape") and value.ndim > 0 and value.shape[0] == n:
            save_dict[key] = value[idx:idx + 1]
        else:
            save_dict[key] = value

    save_dict["original_split"] = np.array(best["split"])
    save_dict["original_local_index"] = np.array(idx, dtype=np.int64)

    m_value = float(save_dict["m"][0])
    gamma_value = float(save_dict["gamma"][0])

    filename = (
        f"selected_m_not_065_"
        f"{best['split']}_local{idx}_"
        f"m{m_value:.4f}_gamma{gamma_value:.4f}.npz"
    )

    filename = filename.replace(".", "p")
    filename = filename.replace("pnpz", ".npz")

    save_path = os.path.join(OUTPUT_DIR, filename)

    np.savez_compressed(save_path, **save_dict)

    print("\n已保存样本：")
    print(save_path)
    print(f"m = {m_value:.6f}")
    print(f"gamma = {gamma_value:.6f}")
    print(f"m^2 = {m_value ** 2:.6f}")

    print("\n直接复制下面命令画图：\n")
    print(
        "python main.py "
        "--exp exp5 "
        "--model_type transformer "
        "--loss_profile pinn "
        "--load_model "
        "--model_path ./model/exp5_transformer_pinn.pth "
        f"--test_path {save_path} "
        "--max_train_samples 1 "
        "--max_val_samples 1 "
        "--max_test_samples 1 "
        "--index 0"
    )


def main():
    best = find_best_sample()
    save_one_sample(best)


if __name__ == "__main__":
    main()