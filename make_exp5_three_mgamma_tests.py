import os
from config import Config

from data_generate_model3_fixed_ood import (
    ensure_dir,
    build_a1_a2_truth_pairs,
    sample_truth_pairs,
    generate_fixed_dataset,
    save_dataset,
)


def main():
    cfg = Config()

    output_dir = "./data_exp5_three_mgamma_tests"
    ensure_dir(output_dir)

    # 每组测试样本数
    # 正式可以用 10000 或 20000；先跑图用 5000 就够
    n_test = 5000

    test_specs = [
        {
            "filename": "test_A_m0p5_gamma0p2.npz",
            "m": 0.5,
            "gamma": 0.2,
            "source": "exp5_extra_test_A_m0p5_gamma0p2",
        },
        {
            "filename": "test_B_m1p0_gamma0p5.npz",
            "m": 1.0,
            "gamma": 0.5,
            "source": "exp5_extra_test_B_m1p0_gamma0p5",
        },
        {
            "filename": "test_C_m1p3_gamma0p8.npz",
            "m": 1.3,
            "gamma": 0.8,
            "source": "exp5_extra_test_C_m1p3_gamma0p8",
        },
    ]

    output_paths = [
        os.path.join(output_dir, spec["filename"])
        for spec in test_specs
    ]

    existing = [p for p in output_paths if os.path.exists(p)]
    if existing:
        raise FileExistsError(
            "以下测试集已经存在，为避免覆盖已停止运行：\n"
            + "\n".join(existing)
            + "\n如果要重新生成，请先手动删除这些文件。"
        )

    # 三组测试共用同一批 a1/a2，保证只比较 m/gamma 的影响
    truth_pairs = build_a1_a2_truth_pairs()

    test_pairs = sample_truth_pairs(
        truth_pairs=truth_pairs,
        n_samples=n_test,
        seed=cfg.random_seed + 502,
    )

    print("开始生成 exp5 三组 m/gamma 测试集")
    print(f"每组测试样本数: {len(test_pairs)}")
    print(f"输出目录: {output_dir}")
    print()

    for spec in test_specs:
        path = os.path.join(output_dir, spec["filename"])

        dataset = generate_fixed_dataset(
            selected_pairs=test_pairs,
            m_value=spec["m"],
            gamma_value=spec["gamma"],
            noise_seed=cfg.random_seed + 503,
            source_name=spec["source"],
        )

        save_dataset(path, dataset)

        print(
            f"完成: {spec['filename']} | "
            f"m={spec['m']}, gamma={spec['gamma']}, "
            f"峰位置约 m^2={spec['m'] ** 2:.3f}"
        )
        print()

    print("三组 exp5 测试集全部生成完成。")


if __name__ == "__main__":
    main()