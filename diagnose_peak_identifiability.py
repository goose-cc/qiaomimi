from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from mc_physics import (
    add_rms_noise_numpy,
    scaled_forward_observation_numpy,
    scaled_target_u_numpy,
)
from mc_pool_config import DEFAULT_PHYSICS


def rel_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检查当前物理观测下是否存在 g 很接近但 f/峰参数差异很大的样本对。"
    )
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--integration-points", type=int, default=128)
    parser.add_argument("--noise-level", type=float, default=0.09)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--output-dir", default="./diagnostics/peak_identifiability")
    args = parser.parse_args()

    pool_path = Path(args.pool_dir) / "parameters.dat"
    row_bytes = 5 * np.dtype(np.float32).itemsize
    total = pool_path.stat().st_size // row_bytes
    pool = np.memmap(pool_path, dtype=np.float32, mode="r", shape=(total, 5))
    rng = np.random.default_rng(args.seed)
    n = min(args.num_samples, total)
    indices = np.sort(rng.choice(total, size=n, replace=False))
    params = np.asarray(pool[indices], dtype=np.float32)

    f = scaled_target_u_numpy(params)
    g_clean = scaled_forward_observation_numpy(
        params, integration_points=args.integration_points
    )
    g_noisy, _ = add_rms_noise_numpy(g_clean, args.noise_level, rng)

    # Feature includes normalized shape and log amplitude.  Candidate pairs are
    # then rechecked with exact relative L2 in the original g space.
    g_rms = np.sqrt(np.mean(g_clean.astype(np.float64) ** 2, axis=1, keepdims=True))
    shape = g_clean.astype(np.float64) / np.maximum(g_rms, 1e-12)
    log_amp = np.log(np.maximum(g_rms, 1e-12))
    feature = np.concatenate([shape, log_amp], axis=1)
    tree = cKDTree(feature)
    _, neighbors = tree.query(feature, k=min(args.neighbors + 1, n))

    pairs = []
    used = set()
    for i in range(n):
        for j in np.atleast_1d(neighbors[i])[1:]:
            j = int(j)
            key = (min(i, j), max(i, j))
            if key in used:
                continue
            used.add(key)
            gdiff = float(
                np.linalg.norm(g_clean[i] - g_clean[j])
                / max(np.linalg.norm(g_clean[i]), 1e-12)
            )
            fdiff = float(
                np.linalg.norm(f[i] - f[j])
                / max(np.linalg.norm(f[i]), 1e-12)
            )
            if gdiff <= args.noise_level:
                p1, p2 = params[i], params[j]
                pairs.append({
                    "row_i": int(indices[i]),
                    "row_j": int(indices[j]),
                    "g_relative_difference": gdiff,
                    "f_relative_difference": fdiff,
                    "m_difference": float(abs(p1[3] - p2[3])),
                    "gamma_difference": float(abs(p1[4] - p2[4])),
                    "width_i": float(p1[3] * p1[4]),
                    "width_j": float(p2[3] * p2[4]),
                    "a1_i": float(p1[0]),
                    "a1_j": float(p2[0]),
                })
    pairs.sort(key=lambda x: (-x["f_relative_difference"], x["g_relative_difference"]))
    pairs = pairs[: args.max_pairs]

    width = params[:, 3] * params[:, 4]
    s = np.linspace(DEFAULT_PHYSICS.s_min, DEFAULT_PHYSICS.s_max, DEFAULT_PHYSICS.output_points)
    a1 = params[:, 0:1].astype(np.float64)
    m = params[:, 3:4].astype(np.float64)
    gamma = params[:, 4:5].astype(np.float64)
    w = m * gamma
    resonance = (
        DEFAULT_PHYSICS.data_scale
        * (a1 / np.pi) * w
        / ((s.reshape(1, -1) - m) ** 2 + w**2)
        / (s.reshape(1, -1) + DEFAULT_PHYSICS.shift) ** 2
    )
    visibility = np.linalg.norm(resonance, axis=1) / np.maximum(
        np.linalg.norm(f.astype(np.float64), axis=1), 1e-12
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pair_path = output / "ambiguous_pairs.csv"
    with pair_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = list(pairs[0].keys()) if pairs else [
            "row_i", "row_j", "g_relative_difference", "f_relative_difference"
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)

    edges = np.asarray([0.0, 0.1, 0.2, 0.5, np.inf])
    width_stats = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (width >= left) & (width < right)
        width_stats.append({
            "left": float(left),
            "right": None if np.isinf(right) else float(right),
            "count": int(mask.sum()),
            "visibility_mean": float(np.mean(visibility[mask])) if np.any(mask) else None,
        })

    summary = {
        "pool": str(Path(args.pool_dir).resolve()),
        "sample_count": int(n),
        "noise_level_reference": float(args.noise_level),
        "ambiguous_pair_definition": (
            "clean g relative L2 difference <= noise_level; nearest-neighbor candidates only"
        ),
        "ambiguous_pairs_found": int(len(pairs)),
        "largest_f_difference_pair": pairs[0] if pairs else None,
        "width_stats": width_stats,
        "visibility_mean": float(np.mean(visibility)),
        "visibility_median": float(np.median(visibility)),
        "clean_to_noisy_relative_mean": float(np.mean(rel_l2(g_noisy, g_clean))),
        "notes": (
            "This is a diagnostic search, not a mathematical uniqueness proof. "
            "Pairs with g differences below the noise level but large f differences "
            "are direct evidence of practical ambiguity."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("pairs:", pair_path.resolve())


if __name__ == "__main__":
    main()
