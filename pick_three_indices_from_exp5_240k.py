import os
import csv
import numpy as np


INPUT_FILES = [
    ("train", "./data_exp5/train.npz"),
    ("val", "./data_exp5/val.npz"),
    ("test", "./data_exp5/test.npz"),
]

OUTPUT_DIR = "./data_exp5_selected_three_from_240k"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = "./model/exp5_transformer_pinn.pth"


def load_m_gamma_only():
    all_m = []
    all_gamma = []
    split_info = []

    global_offset = 0

    for split_name, path in INPUT_FILES:
        print(f"正在读取 {path} ...", flush=True)

        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件: {path}")

        data = np.load(path, allow_pickle=True)

        if "m" not in data.files:
            raise KeyError(f"{path} 里面没有 m 字段，当前字段: {data.files}")
        if "gamma" not in data.files:
            raise KeyError(f"{path} 里面没有 gamma 字段，当前字段: {data.files}")

        m = data["m"].astype(np.float64)
        gamma = data["gamma"].astype(np.float64)

        n = len(m)

        all_m.append(m)
        all_gamma.append(gamma)

        split_info.append({
            "split": split_name,
            "path": path,
            "start": global_offset,
            "end": global_offset + n,
            "n": n,
        })

        print(
            f"完成读取 {split_name}: {n} 条, "
            f"m范围={m.min():.6f}~{m.max():.6f}, "
            f"gamma范围={gamma.min():.6f}~{gamma.max():.6f}",
            flush=True
        )

        global_offset += n

    all_m = np.concatenate(all_m)
    all_gamma = np.concatenate(all_gamma)

    print(f"\n总样本数: {len(all_m)}", flush=True)

    return all_m, all_gamma, split_info


def global_to_split_index(global_index, split_info):
    for info in split_info:
        if info["start"] <= global_index < info["end"]:
            local_index = global_index - info["start"]
            return info["split"], info["path"], int(local_index)

    raise IndexError(f"global_index 超出范围: {global_index}")


def pick_three_far_apart(m, gamma):
    # 归一化，避免 m 和 gamma 尺度不同
    m_norm = (m - m.min()) / (m.max() - m.min() + 1e-12)
    gamma_norm = (gamma - gamma.min()) / (gamma.max() - gamma.min() + 1e-12)

    points = np.stack([m_norm, gamma_norm], axis=1)

    # 选左下角附近：m 小、gamma 小
    idx1 = int(np.argmin(points[:, 0] + points[:, 1]))

    # 选距离 idx1 最远的点
    dist1 = np.sum((points - points[idx1]) ** 2, axis=1)
    idx2 = int(np.argmax(dist1))

    # 第三个点：距离前两个点的最小距离最大
    dist_to_1 = np.sum((points - points[idx1]) ** 2, axis=1)
    dist_to_2 = np.sum((points - points[idx2]) ** 2, axis=1)
    min_dist = np.minimum(dist_to_1, dist_to_2)

    min_dist[idx1] = -1
    min_dist[idx2] = -1

    idx3 = int(np.argmax(min_dist))

    return [idx1, idx2, idx3]


def save_one_sample(rank, global_index, split_name, path, local_index):
    print(
        f"\n正在保存第 {rank} 条样本: "
        f"global_index={global_index}, split={split_name}, local_index={local_index}",
        flush=True
    )

    data = np.load(path, allow_pickle=True)
    n = len(data["fx"])

    save_dict = {}

    for key in data.files:
        value = data[key]

        # 样本级字段：取一条
        if hasattr(value, "shape") and value.ndim > 0 and value.shape[0] == n:
            save_dict[key] = value[local_index:local_index + 1]
        else:
            # x, y 等非样本字段原样保存
            save_dict[key] = value

    save_dict["original_split"] = np.array(split_name)
    save_dict["original_local_index"] = np.array(local_index, dtype=np.int64)
    save_dict["original_global_index"] = np.array(global_index, dtype=np.int64)

    m_value = float(save_dict["m"][0])
    gamma_value = float(save_dict["gamma"][0])

    filename = (
        f"selected_{rank}_"
        f"global{global_index}_"
        f"{split_name}_local{local_index}_"
        f"m{m_value:.4f}_gamma{gamma_value:.4f}.npz"
    )

    # 文件名里不要有小数点
    filename = filename.replace(".", "p")
    filename = filename.replace("pnpz", ".npz")

    save_path = os.path.join(OUTPUT_DIR, filename)
    np.savez_compressed(save_path, **save_dict)

    print(f"保存完成: {save_path}", flush=True)

    return {
        "rank": rank,
        "global_index": global_index,
        "split": split_name,
        "local_index": local_index,
        "m": m_value,
        "gamma": gamma_value,
        "save_path": save_path,
    }


def write_csv(records):
    csv_path = os.path.join(OUTPUT_DIR, "selected_three_indices.csv")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "global_index",
            "split",
            "local_index",
            "m",
            "gamma",
            "npz_path",
        ])

        for r in records:
            writer.writerow([
                r["rank"],
                r["global_index"],
                r["split"],
                r["local_index"],
                f"{r['m']:.6f}",
                f"{r['gamma']:.6f}",
                r["save_path"],
            ])

    print(f"\n索引表已保存: {csv_path}", flush=True)


def main():
    print("开始从 exp5 的 24 万数据中挑选 m/gamma 差异最大的三条样本...\n", flush=True)

    m, gamma, split_info = load_m_gamma_only()

    selected_global_indices = pick_three_far_apart(m, gamma)

    print("\n选中的三个 global index:", selected_global_indices, flush=True)

    records = []

    for rank, global_index in enumerate(selected_global_indices, start=1):
        split_name, path, local_index = global_to_split_index(global_index, split_info)

        record = save_one_sample(
            rank=rank,
            global_index=global_index,
            split_name=split_name,
            path=path,
            local_index=local_index,
        )

        records.append(record)

    write_csv(records)

    print("\n三条样本抽取完成。接下来你可以用 main.py 手动跑三次。", flush=True)

    print("\n三条测试命令模板如下：\n", flush=True)

    for r in records:
        print(
            "python main.py "
            "--exp exp5 "
            "--model_type transformer "
            "--loss_profile pinn "
            "--load_model "
            f"--model_path {MODEL_PATH} "
            f"--test_path {r['save_path']} "
            "--max_train_samples 1 "
            "--max_val_samples 1 "
            "--max_test_samples 1 "
            "--index 0",
            flush=True
        )
        print()


if __name__ == "__main__":
    main()