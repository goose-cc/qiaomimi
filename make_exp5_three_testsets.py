import os
import numpy as np


INPUT_PATHS = [
    "./data_exp5/train.npz",
    "./data_exp5/val.npz",
    "./data_exp5/test.npz",
]

OUTPUT_DIR = "./data_exp4_extra_tests_from_exp5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_PER_TEST = 1000
TOL = 0.01


def load_all(paths):
    arrays = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True)
        arrays.append(data)

    keys = arrays[0].files
    merged = {}

    for key in keys:
        first = arrays[0][key]

        # x, y, param_names 这类非样本维度字段直接取第一份
        if key in ["x", "y", "param_names", "varying_param_names"]:
            merged[key] = first
            continue

        # 标量字段直接取第一份
        if first.ndim == 0:
            merged[key] = first
            continue

        # 样本字段按第 0 维拼接
        try:
            merged[key] = np.concatenate([d[key] for d in arrays], axis=0)
        except Exception:
            merged[key] = first

    return merged


def save_subset(data, idx, save_path, source_name):
    save_dict = {}
    n = len(data["fx"])

    for key, value in data.items():
        value = np.asarray(value)

        if value.ndim > 0 and value.shape[0] == n:
            save_dict[key] = value[idx]
        else:
            save_dict[key] = value

    save_dict["source"] = np.array(source_name)

    np.savez_compressed(save_path, **save_dict)

    print(f"Saved: {save_path}")
    print(f"  samples: {len(idx)}")
    print(f"  m range: {save_dict['m'].min():.4f} ~ {save_dict['m'].max():.4f}")
    print(f"  gamma range: {save_dict['gamma'].min():.4f} ~ {save_dict['gamma'].max():.4f}")
    print()


def pick_near(data, target_m=None, target_gamma=None, n=1000, tol=0.01):
    m = data["m"]
    gamma = data["gamma"]

    mask = np.ones_like(m, dtype=bool)

    if target_m is not None:
        mask &= np.abs(m - target_m) <= tol

    if target_gamma is not None:
        mask &= np.abs(gamma - target_gamma) <= tol

    idx = np.flatnonzero(mask)

    if len(idx) == 0:
        raise RuntimeError(
            f"找不到样本: target_m={target_m}, target_gamma={target_gamma}, tol={tol}"
        )

    if len(idx) > n:
        rng = np.random.default_rng(2026)
        idx = rng.choice(idx, size=n, replace=False)

    return idx


def main():
    data = load_all(INPUT_PATHS)

    print("merged samples:", len(data["fx"]))
    print("m range:", data["m"].min(), data["m"].max())
    print("gamma range:", data["gamma"].min(), data["gamma"].max())

    # 三组 m 差异很大的测试集，gamma 固定在 0.5 附近
    tests = [
        ("test_m0p3_gamma0p5.npz", 0.3, 0.5),
        ("test_m0p8_gamma0p5.npz", 0.8, 0.5),
        ("test_m1p2_gamma0p5.npz", 1.2, 0.5),
    ]

    for filename, target_m, target_gamma in tests:
        idx = pick_near(
            data,
            target_m=target_m,
            target_gamma=target_gamma,
            n=N_PER_TEST,
            tol=TOL,
        )

        save_subset(
            data,
            idx,
            os.path.join(OUTPUT_DIR, filename),
            source_name=f"exp5_subset_m{target_m}_gamma{target_gamma}",
        )


if __name__ == "__main__":
    main()