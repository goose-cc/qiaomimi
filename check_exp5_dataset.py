import os
import numpy as np

from config import Config


cfg = Config()

DATASETS = {
    "train": ("./data_exp5/train.npz", cfg.model3_sweep_train_pairs),
    "val": ("./data_exp5/val.npz", cfg.model3_sweep_val_pairs),
    "test": ("./data_exp5/test.npz", cfg.model3_sweep_test_pairs),
}

REQUIRED_KEYS = {
    "x",
    "y",
    "fx",
    "gy_clean",
    "gy_noisy",
    "a1",
    "a2",
    "m",
    "gamma",
    "params",
    "param_names",
    "varying_param",
    "varying_param_names",
    "noise_level",
    "requested_step",
    "used_step",
    "source",
}

BASELINE = np.array(
    [
        cfg.model3_truth_a1,
        cfg.model3_truth_a2,
        cfg.model3_train_m,
        cfg.model3_train_gamma,
    ],
    dtype=np.float32,
)

PARAM_NAMES = ["a1", "a2", "m", "gamma"]


def model3_fx(x, params):
    x_grid = x[None, :]
    a1 = params[:, 0:1]
    a2 = params[:, 1:2]
    m = params[:, 2:3]
    gamma = params[:, 3:4]

    width = a1 * m * gamma

    peak = (
        (1.0 / np.pi)
        * width
        / ((x_grid - m**2) ** 2 + width**2)
    )

    background = a2 * x_grid / 5.0
    return (peak + background).astype(np.float32)


def build_kernel(x, y):
    dx = (x[-1] - x[0]) / (len(x) - 1)
    weights = np.ones_like(x, dtype=np.float32) * dx
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return (
        weights[None, :] / (y[:, None] - x[None, :])
    ).astype(np.float32)


def row_keys(params):
    params = np.ascontiguousarray(params)
    return params.view(
        np.dtype((np.void, params.dtype.itemsize * params.shape[1]))
    ).ravel()


def validate_one(name, path, expected_n):
    print()
    print("=" * 72)
    print(f"CHECK {name.upper()}: {path}")
    print("=" * 72)

    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    with np.load(path) as d:
        missing = REQUIRED_KEYS - set(d.files)
        if missing:
            raise AssertionError(f"{name} 缺少 keys: {sorted(missing)}")

        x = d["x"]
        y = d["y"]
        fx = d["fx"]
        gy_clean = d["gy_clean"]
        gy_noisy = d["gy_noisy"]
        params = d["params"]
        varying_param = d["varying_param"]

        print("keys: PASS")
        print(f"N={len(fx)}, fx={fx.shape}, gy_clean={gy_clean.shape}, gy_noisy={gy_noisy.shape}")

        assert len(fx) == expected_n, (name, len(fx), expected_n)
        assert fx.shape == (expected_n, cfg.num_x_points)
        assert gy_clean.shape == (expected_n, cfg.num_y_points)
        assert gy_noisy.shape == (expected_n, cfg.num_y_points)
        assert params.shape == (expected_n, 4)
        assert varying_param.shape == (expected_n,)
        print("shape/count: PASS")

        arrays_match = (
            np.array_equal(d["a1"], params[:, 0])
            and np.array_equal(d["a2"], params[:, 1])
            and np.array_equal(d["m"], params[:, 2])
            and np.array_equal(d["gamma"], params[:, 3])
        )
        assert arrays_match
        print("a1/a2/m/gamma 与 params 四列一致: PASS")

        lower = np.array(
            [
                cfg.model3_a1_min,
                cfg.model3_a2_min,
                cfg.model3_m_min,
                cfg.model3_gamma_min,
            ],
            dtype=np.float32,
        )
        upper = np.array(
            [
                cfg.model3_a1_max,
                cfg.model3_a2_max,
                cfg.model3_m_max,
                cfg.model3_gamma_max,
            ],
            dtype=np.float32,
        )

        assert np.all(params >= lower[None, :] - 1e-6)
        assert np.all(params <= upper[None, :] + 1e-6)
        print("parameter bounds: PASS")

        changed = ~np.isclose(
            params,
            BASELINE[None, :],
            rtol=0.0,
            atol=1e-7,
        )
        changed_count = changed.sum(axis=1)

        oat_bad = int(np.sum(changed_count > 1))
        baseline_count = int(np.sum(changed_count == 0))
        assert oat_bad == 0, f"{name} 有 {oat_bad} 条样本同时改变多个参数"

        one_changed = changed_count == 1
        changed_index = np.argmax(changed, axis=1)
        label_bad = int(
            np.sum(
                one_changed
                & (changed_index != varying_param)
            )
        )
        assert label_bad == 0, f"{name} varying_param 标签错误 {label_bad} 条"

        print(
            "OAT rule: PASS | "
            f"同时改变>1参数={oat_bad}, "
            f"基准样本={baseline_count}"
        )

        label_counts = np.bincount(varying_param, minlength=4)
        print(
            "varying_param counts: "
            + ", ".join(
                f"{PARAM_NAMES[i]}={int(label_counts[i])}"
                for i in range(4)
            )
        )

        noise = gy_noisy - gy_clean
        noise_rms = np.sqrt(np.mean(noise**2, axis=1))
        signal_rms = np.sqrt(np.mean(gy_clean**2, axis=1))
        noise_ratio = noise_rms / np.maximum(signal_rms, 1e-12)

        print(
            "noise RMS ratio: "
            f"mean={noise_ratio.mean():.8f}, "
            f"std={noise_ratio.std():.8f}, "
            f"min={noise_ratio.min():.8f}, "
            f"max={noise_ratio.max():.8f}"
        )
        assert 0.009 <= float(noise_ratio.mean()) <= 0.011
        print("1% additive white-noise level: PASS")

        check_n = min(1000, expected_n)
        check_params = params[:check_n]
        expected_fx = model3_fx(x, check_params)
        fx_max_abs_error = float(
            np.max(np.abs(expected_fx - fx[:check_n]))
        )
        print(f"fx formula max abs error ({check_n} samples): {fx_max_abs_error:.3e}")
        assert fx_max_abs_error < 1e-5
        print("fx formula reconstruction: PASS")

        kernel = build_kernel(x, y)
        expected_gy = (fx[:check_n] @ kernel.T).astype(np.float32)
        gy_max_abs_error = float(
            np.max(np.abs(expected_gy - gy_clean[:check_n]))
        )
        print(f"gy integral max abs error ({check_n} samples): {gy_max_abs_error:.3e}")
        assert gy_max_abs_error < 1e-5
        print("gy_clean integral reconstruction: PASS")

        centers = params[:, 2] ** 2
        center_inside = (
            (centers >= float(x.min()))
            & (centers <= float(x.max()))
        )
        outside_count = int(np.sum(~center_inside))
        print(
            "theoretical peak center m^2 inside x-domain: "
            f"{int(center_inside.sum())}/{expected_n}; "
            f"outside={outside_count}"
        )

        local_peak = np.any(
            (fx[:, 1:-1] > fx[:, :-2])
            & (fx[:, 1:-1] > fx[:, 2:]),
            axis=1,
        )
        no_local_peak = int(np.sum(~local_peak))
        print(
            "visible interior local maximum in sampled f(x): "
            f"{int(local_peak.sum())}/{expected_n}; "
            f"no interior local max={no_local_peak}"
        )

        print(
            f"requested_step={float(d['requested_step']):.10f}, "
            f"used_step={float(d['used_step']):.10f}, "
            f"noise_level={float(d['noise_level']):.6f}, "
            f"source={d['source']}"
        )

        return params.copy()


def main():
    params_by_split = {}

    for name, (path, expected_n) in DATASETS.items():
        params_by_split[name] = validate_one(
            name,
            path,
            expected_n,
        )

    print()
    print("=" * 72)
    print("CHECK TRAIN / VAL / TEST PARAMETER OVERLAP")
    print("=" * 72)

    keys = {
        name: row_keys(params)
        for name, params in params_by_split.items()
    }

    overlap_train_val = len(
        np.intersect1d(keys["train"], keys["val"])
    )
    overlap_train_test = len(
        np.intersect1d(keys["train"], keys["test"])
    )
    overlap_val_test = len(
        np.intersect1d(keys["val"], keys["test"])
    )

    print(f"train ∩ val = {overlap_train_val}")
    print(f"train ∩ test = {overlap_train_test}")
    print(f"val ∩ test = {overlap_val_test}")

    assert overlap_train_val == 0
    assert overlap_train_test == 0
    assert overlap_val_test == 0
    print("parameter truth overlap: PASS")

    all_params = np.concatenate(
        [
            params_by_split["train"],
            params_by_split["val"],
            params_by_split["test"],
        ],
        axis=0,
    )

    print()
    print("=" * 72)
    print("COMBINED EXP5 PARAMETER COVERAGE")
    print("=" * 72)

    for i, param_name in enumerate(PARAM_NAMES):
        print(
            f"{param_name}: "
            f"min={all_params[:, i].min():.8f}, "
            f"max={all_params[:, i].max():.8f}, "
            f"unique={len(np.unique(all_params[:, i]))}"
        )

    print()
    print("ALL CORE EXP5 DATA CHECKS PASSED.")
    print()
    print(
        "注意：如果 theoretical peak center m^2 outside x-domain 数量较多，"
        "这不是代码错误。原因是当前 m 最大为 2.0，而 x 最大为 2.0；"
        "当 m > sqrt(2) 时，峰中心 m^2 > 2，会落在当前 x 观测区间之外。"
    )


if __name__ == "__main__":
    main()
