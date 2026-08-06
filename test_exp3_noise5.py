from __future__ import annotations

from pathlib import Path

import numpy as np

from mc_online_physics import fixed_rms_noise_numpy


def main() -> None:
    root = Path(__file__).resolve().parent
    run_script = root / "run_exp3_output1000_noise5.ps1"
    if not run_script.exists():
        raise FileNotFoundError(run_script)

    text = run_script.read_text(encoding="utf-8-sig")
    required_fragments = (
        '"--input-points", "100"',
        '"--output-points", "1000"',
        '"--noise-level", "0.05"',
        '"--sampling-mode", "shuffle"',
        '"--batch-size", "64"',
    )
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        raise AssertionError(f"运行脚本缺少预期配置：{missing}")

    # 用大量固定样本检查实现中的噪声 RMS 比例确实约为 5%。
    clean = np.ones((4096, 100), dtype=np.float32)
    noisy = fixed_rms_noise_numpy(clean, noise_level=0.05, seed=20260802)
    noise = noisy.astype(np.float64) - clean.astype(np.float64)
    clean_rms = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2)))
    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    measured_ratio = noise_rms / clean_rms

    if not (0.0495 <= measured_ratio <= 0.0505):
        raise AssertionError(
            f"测得噪声 RMS 比例为 {measured_ratio:.8f}，不在 5% 附近。"
        )

    print("EXP3 NOISE5 TEST PASSED")
    print(f"measured RMS noise ratio: {measured_ratio:.6%}")
    print("input_points=100, output_points=1000, sampling_mode=shuffle")


if __name__ == "__main__":
    main()
