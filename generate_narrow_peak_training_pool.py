from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from mc_pool_config import DEFAULT_PHYSICS, atomic_write_json


PARAMETER_NAMES = ("a1", "a2", "a3", "m", "gamma")


def sample_target_width_group(
    rng: np.random.Generator,
    count: int,
    width_range: Tuple[float, float],
    a1_min: float = 0.05,
) -> np.ndarray:
    lo, hi = width_range
    if not (0.0 < lo < hi <= DEFAULT_PHYSICS.m_max * DEFAULT_PHYSICS.gamma_max):
        raise ValueError("invalid width range")
    width = rng.uniform(lo, hi, size=count).astype(np.float64)
    m_lower = np.maximum(DEFAULT_PHYSICS.s_min, width)
    mass = m_lower + rng.random(count) * (DEFAULT_PHYSICS.m_max - m_lower)
    gamma = width / mass

    rows = np.empty((count, 5), dtype=np.float32)
    rows[:, 0] = rng.uniform(a1_min, DEFAULT_PHYSICS.a1_max, size=count)
    rows[:, 1] = rng.uniform(
        DEFAULT_PHYSICS.a2_min, DEFAULT_PHYSICS.a2_max, size=count
    )
    rows[:, 2] = rng.uniform(
        DEFAULT_PHYSICS.a3_min, DEFAULT_PHYSICS.a3_max, size=count
    )
    rows[:, 3] = mass.astype(np.float32)
    rows[:, 4] = gamma.astype(np.float32)
    return rows


def sample_uniform_original(rng: np.random.Generator, count: int) -> np.ndarray:
    tiny = np.nextafter(np.float32(0.0), np.float32(1.0), dtype=np.float32)
    rows = np.empty((count, 5), dtype=np.float32)
    rows[:, 0] = rng.uniform(
        DEFAULT_PHYSICS.a1_min, DEFAULT_PHYSICS.a1_max, size=count
    )
    rows[:, 1] = rng.uniform(
        DEFAULT_PHYSICS.a2_min, DEFAULT_PHYSICS.a2_max, size=count
    )
    rows[:, 2] = rng.uniform(
        DEFAULT_PHYSICS.a3_min, DEFAULT_PHYSICS.a3_max, size=count
    )
    rows[:, 3] = rng.uniform(tiny, DEFAULT_PHYSICS.m_max, size=count)
    rows[:, 4] = rng.uniform(tiny, DEFAULT_PHYSICS.gamma_max, size=count)
    return rows


def allocate_counts(total: int, fractions: Dict[str, float]) -> Dict[str, int]:
    raw = {name: total * value for name, value in fractions.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="直接生成偏向窄峰、同时保留原始分布样本的 V2.6 训练参数池。"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=400000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError("output directory is not empty; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    fractions = {
        "narrow_002_010": 0.40,
        "narrow_010_020": 0.25,
        "medium_020_050": 0.15,
        "broad_050_180": 0.05,
        "uniform_original": 0.15,
    }
    counts = allocate_counts(args.num_samples, fractions)
    rng = np.random.default_rng(args.seed)

    groups = [
        sample_target_width_group(rng, counts["narrow_002_010"], (0.02, 0.10)),
        sample_target_width_group(rng, counts["narrow_010_020"], (0.10, 0.20)),
        sample_target_width_group(rng, counts["medium_020_050"], (0.20, 0.50)),
        sample_target_width_group(rng, counts["broad_050_180"], (0.50, 1.80)),
        sample_uniform_original(rng, counts["uniform_original"]),
    ]
    rows = np.concatenate(groups, axis=0)
    rows = rows[rng.permutation(len(rows))]

    data_path = output_dir / "parameters.dat"
    mm = np.memmap(data_path, dtype=np.float32, mode="w+", shape=rows.shape)
    mm[:] = rows
    mm.flush()
    del mm

    width = rows[:, 3].astype(np.float64) * rows[:, 4].astype(np.float64)
    metadata = {
        "complete": True,
        "generated_truths": int(len(rows)),
        "parameter_names": list(PARAMETER_NAMES),
        "physics": DEFAULT_PHYSICS.to_dict(),
        "selection": (
            "targeted narrow-peak mixture: 65% m*gamma in [0.02,0.2), "
            "20% wider visible peaks, 15% original uniform distribution"
        ),
        "fractions": fractions,
        "counts": counts,
        "seed": int(args.seed),
        "width_summary": {
            "min": float(np.min(width)),
            "median": float(np.median(width)),
            "mean": float(np.mean(width)),
            "max": float(np.max(width)),
        },
        "does_not_change_physics_formula": True,
        "purpose": "train the narrow-peak correction model before any 160M run",
    }
    atomic_write_json(output_dir / "metadata.json", metadata)
    print("written:", output_dir.resolve())
    print("rows:", len(rows))
    for name, count in counts.items():
        print("%s: %d" % (name, count))


if __name__ == "__main__":
    main()
