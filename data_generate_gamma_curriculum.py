#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gamma-only curriculum dataset generator.

Designed for the current five-parameter physics branch:
    parameters = [a1, a2, a3, m, gamma]
    rho(s) = a1/pi * (m*gamma) / ((s-m)^2 + (m*gamma)^2) + a2*s + a3
    u(s)   = rho(s) / (s+400)^2
    g(q^2) = forward integral implemented by mc_physics.py

Default first-stage settings follow the agreed experiment:
    a1 = 0.10
    a2 = 0.025
    a3 = 0
    m = 0.8
    q^2 in [-100, -6]
    Nq = 100
    Easy gamma = [0.01, 0.05, 0.20, 0.50, 1.00]
    noise levels = [0%, 0.2%, 1%]

Outputs:
    metadata.json
    gamma_values.csv
    gamma_pair_separation.csv
    gamma_dense_adjacent_separation.csv
    noise_*/train.npz
    noise_*/val.npz
    noise_*/test_seen.npz
    noise_*/test_interp.npz

The npz format remains compatible with PIDataset.PeakInversionDataset because
it contains fx, gy_clean, gy_noisy, x and y.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np

from mc_pool_config import DEFAULT_PHYSICS, PARAMETER_NAMES
from mc_physics import output_grids_numpy, scaled_curves_numpy


DEFAULT_EASY_GAMMAS = (0.01, 0.05, 0.20, 0.50, 1.00)
DEFAULT_EASY_INTERP_GAMMAS = (0.025, 0.10, 0.32, 0.70)
DEFAULT_NOISE_LEVELS = (0.0, 0.002, 0.01)

FIXED_A1 = 0.10
FIXED_A2 = 0.025
FIXED_A3 = 0.0
FIXED_M = 0.8

SPLIT_CODES = {
    "train": 11,
    "val": 22,
    "test_seen": 33,
    "test_interp": 44,
}


def parse_float_list(text: str | None) -> list[float] | None:
    if text is None:
        return None
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("list cannot be empty")
    return values


def noise_dir_name(level: float) -> str:
    pct = level * 100.0
    label = f"{pct:.8g}".replace(".", "p")
    return f"noise_{label}pct"


def parameter_rows(gammas: Iterable[float]) -> np.ndarray:
    gammas = np.asarray(list(gammas), dtype=np.float64)
    p = np.empty((len(gammas), 5), dtype=np.float64)
    p[:, 0] = FIXED_A1
    p[:, 1] = FIXED_A2
    p[:, 2] = FIXED_A3
    p[:, 3] = FIXED_M
    p[:, 4] = gammas
    return p


def curve_bank(
    gammas: Iterable[float],
    physics,
    integration_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    p = parameter_rows(gammas)
    fx, gy = scaled_curves_numpy(
        p,
        integration_points=integration_points,
        config=physics,
    )
    return fx.astype(np.float32), gy.astype(np.float32)


def pair_metrics(g1: np.ndarray, g2: np.ndarray, noise_level: float) -> dict[str, float]:
    a = np.asarray(g1, dtype=np.float64)
    b = np.asarray(g2, dtype=np.float64)
    diff = a - b

    l2 = float(np.linalg.norm(diff))
    rms_diff = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))

    l2_a = float(np.linalg.norm(a))
    l2_b = float(np.linalg.norm(b))
    denominator = 0.5 * (l2_a + l2_b)
    relative_l2 = l2 / denominator if denominator > 0.0 else math.inf

    rms_a = float(np.sqrt(np.mean(a * a)))
    rms_b = float(np.sqrt(np.mean(b * b)))
    mean_rms_g = 0.5 * (rms_a + rms_b)
    sigma = float(noise_level) * mean_rms_g

    # This is the definition requested by the project plan:
    # SNR_sep = ||g_i - g_j||_2 / sigma,
    # sigma = noise_level * average RMS(g).
    snr_sep = l2 / sigma if sigma > 0.0 else math.inf

    # Extra diagnostic: average per-observation separation measured in sigma.
    # This removes the sqrt(Nq) factor from the L2 norm and is useful when
    # comparing different q^2 point counts later.
    snr_sep_rms = rms_diff / sigma if sigma > 0.0 else math.inf

    return {
        "max_abs_g_difference": max_abs,
        "rms_g_difference": rms_diff,
        "l2_g_difference": l2,
        "relative_l2_difference": relative_l2,
        "mean_rms_g": mean_rms_g,
        "sigma_reference": sigma,
        "snr_sep": snr_sep,
        "snr_sep_rms": snr_sep_rms,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dense_scan(
    gamma_min: float,
    gamma_max: float,
    points: int,
    physics,
    integration_points: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    dense_gammas = np.geomspace(gamma_min, gamma_max, points, dtype=np.float64)
    _, dense_g = curve_bank(dense_gammas, physics, integration_points)
    return dense_gammas, dense_g, []


def snr_to_candidates(
    reference_g: np.ndarray,
    candidate_g: np.ndarray,
    noise_level: float,
) -> np.ndarray:
    ref = reference_g.astype(np.float64)
    cand = candidate_g.astype(np.float64)
    diff_l2 = np.linalg.norm(cand - ref[None, :], axis=1)
    ref_rms = np.sqrt(np.mean(ref * ref))
    cand_rms = np.sqrt(np.mean(cand * cand, axis=1))
    sigma = float(noise_level) * 0.5 * (ref_rms + cand_rms)
    result = np.full_like(diff_l2, np.inf, dtype=np.float64)
    mask = sigma > 0.0
    result[mask] = diff_l2[mask] / sigma[mask]
    return result


def select_gamma_sequence_by_snr(
    dense_gammas: np.ndarray,
    dense_g: np.ndarray,
    target_snr: float,
    count: int,
    start_gamma: float,
    reference_noise: float,
) -> list[float]:
    if reference_noise <= 0.0:
        raise ValueError("reference noise must be > 0 for automatic tier selection")
    if count < 2:
        raise ValueError("gamma count must be at least 2")

    start_idx = int(np.argmin(np.abs(dense_gammas - start_gamma)))
    selected = [start_idx]

    for _ in range(count - 1):
        current = selected[-1]
        candidate_ids = np.arange(current + 1, len(dense_gammas))
        if len(candidate_ids) == 0:
            raise RuntimeError("dense gamma grid is too short for requested tier")

        scores = snr_to_candidates(
            dense_g[current],
            dense_g[candidate_ids],
            reference_noise,
        )
        chosen = int(candidate_ids[np.argmin(np.abs(scores - target_snr))])
        if chosen <= current:
            raise RuntimeError("automatic gamma selection did not move forward")
        selected.append(chosen)

    return [float(dense_gammas[i]) for i in selected]


def default_interp_gammas(train_gammas: list[float], tier: str) -> list[float]:
    if tier == "easy" and np.allclose(
        np.asarray(train_gammas),
        np.asarray(DEFAULT_EASY_GAMMAS),
        rtol=0.0,
        atol=1e-12,
    ):
        return list(DEFAULT_EASY_INTERP_GAMMAS)

    # Positive gamma values: geometric midpoint is natural on a log-scale scan.
    values = []
    for a, b in zip(train_gammas[:-1], train_gammas[1:]):
        values.append(float(math.sqrt(a * b)))
    return values


def group_seed(base_seed: int, split: str, gamma_slot: int) -> int:
    # Disjoint deterministic seed ranges across splits/classes.
    return int(
        base_seed
        + SPLIT_CODES[split] * 1_000_003
        + (gamma_slot + 1) * 10_007
    )


def make_split_arrays(
    split: str,
    gammas: list[float],
    count_per_gamma: int,
    noise_level: float,
    physics,
    integration_points: int,
    base_seed: int,
    gamma_class_is_seen: bool,
) -> dict[str, np.ndarray]:
    fx_bank, gy_bank = curve_bank(gammas, physics, integration_points)
    s_grid, q2_grid = output_grids_numpy(physics)

    blocks = {
        "fx": [],
        "gy_clean": [],
        "gy_noisy": [],
        "parameters": [],
        "gamma_class": [],
        "gamma_index": [],
        "noise_level": [],
        "noise_sigma": [],
        "seed": [],
        "noise_realization_id": [],
    }

    for gamma_index, gamma in enumerate(gammas):
        clean = gy_bank[gamma_index].astype(np.float32)
        target = fx_bank[gamma_index].astype(np.float32)
        rms = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2)))
        sigma = float(noise_level) * rms

        seed = group_seed(base_seed, split, gamma_index)
        rng = np.random.default_rng(seed)
        if noise_level > 0.0:
            z = rng.standard_normal((count_per_gamma, physics.q2_points)).astype(np.float32)
            noisy = clean[None, :] + np.float32(sigma) * z
        else:
            noisy = np.repeat(clean[None, :], count_per_gamma, axis=0)

        clean_rows = np.repeat(clean[None, :], count_per_gamma, axis=0)
        target_rows = np.repeat(target[None, :], count_per_gamma, axis=0)
        params = np.repeat(
            parameter_rows([gamma]).astype(np.float32),
            count_per_gamma,
            axis=0,
        )

        blocks["fx"].append(target_rows)
        blocks["gy_clean"].append(clean_rows)
        blocks["gy_noisy"].append(noisy.astype(np.float32))
        blocks["parameters"].append(params)
        class_value = gamma_index if gamma_class_is_seen else -1
        blocks["gamma_class"].append(
            np.full(count_per_gamma, class_value, dtype=np.int16)
        )
        blocks["gamma_index"].append(
            np.full(count_per_gamma, gamma_index, dtype=np.int16)
        )
        blocks["noise_level"].append(
            np.full(count_per_gamma, noise_level, dtype=np.float32)
        )
        blocks["noise_sigma"].append(
            np.full(count_per_gamma, sigma, dtype=np.float32)
        )
        blocks["seed"].append(
            np.full(count_per_gamma, seed, dtype=np.int64)
        )
        blocks["noise_realization_id"].append(
            np.arange(count_per_gamma, dtype=np.int32)
        )

    result = {key: np.concatenate(parts, axis=0) for key, parts in blocks.items()}

    # Shuffle class blocks while keeping exact reproducibility.
    perm_seed = base_seed + SPLIT_CODES[split] * 9_999_991
    perm = np.random.default_rng(perm_seed).permutation(len(result["fx"]))
    for key in list(result.keys()):
        result[key] = result[key][perm]

    p = result["parameters"]
    result["a1"] = p[:, 0].copy()
    result["a2"] = p[:, 1].copy()
    result["a3"] = p[:, 2].copy()
    result["m"] = p[:, 3].copy()
    result["gamma"] = p[:, 4].copy()

    # Compatibility aliases for the current PIDataset.
    result["x"] = s_grid.astype(np.float32)
    result["y"] = q2_grid.astype(np.float32)
    result["q2"] = q2_grid.astype(np.float32)
    return result


def save_npz(path: Path, arrays: dict[str, np.ndarray], compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate controlled gamma-only curriculum datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="./data_gamma_curriculum_easy")
    parser.add_argument(
        "--tier",
        choices=("easy", "medium", "hard", "custom"),
        default="easy",
        help="easy uses the agreed five coarse gamma values; medium/hard are auto-selected by SNR_sep.",
    )
    parser.add_argument(
        "--gamma-values",
        default=None,
        help="comma-separated custom training gammas; implies custom selection.",
    )
    parser.add_argument(
        "--interp-gammas",
        default=None,
        help="comma-separated interpolation gammas; otherwise chosen automatically.",
    )

    parser.add_argument("--a1", type=float, default=FIXED_A1)
    parser.add_argument("--a2", type=float, default=FIXED_A2)
    parser.add_argument("--a3", type=float, default=FIXED_A3)
    parser.add_argument("--m", type=float, default=FIXED_M)

    parser.add_argument("--q2-points", type=int, default=100)
    parser.add_argument("--integration-points", type=int, default=512)
    parser.add_argument(
        "--noise-levels",
        default="0,0.002,0.01",
        help="fractions, so 0.002 means 0.2%%.",
    )
    parser.add_argument(
        "--reference-noise",
        type=float,
        default=0.002,
        help="noise fraction used for separation tables/tier selection.",
    )

    parser.add_argument("--train-per-gamma", type=int, default=4000)
    parser.add_argument("--val-per-gamma", type=int, default=500)
    parser.add_argument("--test-seen-per-gamma", type=int, default=1000)
    parser.add_argument("--test-interp-per-gamma", type=int, default=1000)

    parser.add_argument("--dense-gamma-min", type=float, default=0.001)
    parser.add_argument("--dense-gamma-max", type=float, default=1.0)
    parser.add_argument("--dense-gamma-points", type=int, default=1000)
    parser.add_argument("--tier-start-gamma", type=float, default=0.01)
    parser.add_argument("--medium-target-snr", type=float, default=3.0)
    parser.add_argument("--hard-target-snr", type=float, default=1.0)
    parser.add_argument("--tier-gamma-count", type=int, default=5)

    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    global FIXED_A1, FIXED_A2, FIXED_A3, FIXED_M
    args = parse_args()

    FIXED_A1 = float(args.a1)
    FIXED_A2 = float(args.a2)
    FIXED_A3 = float(args.a3)
    FIXED_M = float(args.m)

    if args.q2_points < 2:
        raise ValueError("--q2-points must be >= 2")
    if args.integration_points < 8:
        raise ValueError("--integration-points must be >= 8")
    if args.reference_noise <= 0.0:
        raise ValueError("--reference-noise must be > 0")
    if args.dense_gamma_min <= 0.0 or args.dense_gamma_max <= args.dense_gamma_min:
        raise ValueError("invalid dense gamma range")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} is not empty. Use --overwrite only if you intend to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    physics = replace(
        DEFAULT_PHYSICS,
        q2_points=int(args.q2_points),
    )

    dense_gammas, dense_g, _ = build_dense_scan(
        args.dense_gamma_min,
        args.dense_gamma_max,
        args.dense_gamma_points,
        physics,
        args.integration_points,
    )

    custom_gammas = parse_float_list(args.gamma_values)
    if custom_gammas is not None:
        train_gammas = custom_gammas
        effective_tier = "custom"
    elif args.tier == "easy":
        train_gammas = list(DEFAULT_EASY_GAMMAS)
        effective_tier = "easy"
    elif args.tier == "medium":
        train_gammas = select_gamma_sequence_by_snr(
            dense_gammas,
            dense_g,
            target_snr=args.medium_target_snr,
            count=args.tier_gamma_count,
            start_gamma=args.tier_start_gamma,
            reference_noise=args.reference_noise,
        )
        effective_tier = "medium"
    elif args.tier == "hard":
        train_gammas = select_gamma_sequence_by_snr(
            dense_gammas,
            dense_g,
            target_snr=args.hard_target_snr,
            count=args.tier_gamma_count,
            start_gamma=args.tier_start_gamma,
            reference_noise=args.reference_noise,
        )
        effective_tier = "hard"
    else:
        raise ValueError("--tier custom requires --gamma-values")

    train_gammas = sorted(float(x) for x in train_gammas)
    if len(set(train_gammas)) != len(train_gammas):
        raise ValueError("training gamma values must be unique")
    if min(train_gammas) <= 0.0:
        raise ValueError("gamma must be positive")

    custom_interp = parse_float_list(args.interp_gammas)
    if custom_interp is not None:
        interp_gammas = sorted(float(x) for x in custom_interp)
    else:
        interp_gammas = default_interp_gammas(train_gammas, effective_tier)

    for gamma in interp_gammas:
        if np.any(np.isclose(gamma, train_gammas, rtol=0.0, atol=1e-12)):
            raise ValueError(f"interpolation gamma {gamma} appears in training gammas")

    # Separation table for all training gamma pairs.
    _, train_g = curve_bank(train_gammas, physics, args.integration_points)
    pair_rows = []
    for i, j in combinations(range(len(train_gammas)), 2):
        metrics = pair_metrics(train_g[i], train_g[j], args.reference_noise)
        pair_rows.append({
            "gamma_1": train_gammas[i],
            "gamma_2": train_gammas[j],
            "delta_gamma": train_gammas[j] - train_gammas[i],
            "reference_noise_level": args.reference_noise,
            **metrics,
        })

    pair_fields = [
        "gamma_1",
        "gamma_2",
        "delta_gamma",
        "reference_noise_level",
        "max_abs_g_difference",
        "rms_g_difference",
        "l2_g_difference",
        "relative_l2_difference",
        "mean_rms_g",
        "sigma_reference",
        "snr_sep",
        "snr_sep_rms",
    ]
    write_csv(output_dir / "gamma_pair_separation.csv", pair_fields, pair_rows)

    # Dense adjacent scan for selecting future curriculum points.
    dense_rows = []
    for i in range(len(dense_gammas) - 1):
        metrics = pair_metrics(dense_g[i], dense_g[i + 1], args.reference_noise)
        dense_rows.append({
            "gamma_1": float(dense_gammas[i]),
            "gamma_2": float(dense_gammas[i + 1]),
            "delta_gamma": float(dense_gammas[i + 1] - dense_gammas[i]),
            "reference_noise_level": args.reference_noise,
            **metrics,
        })
    write_csv(
        output_dir / "gamma_dense_adjacent_separation.csv",
        pair_fields,
        dense_rows,
    )

    gamma_rows = []
    for i, gamma in enumerate(train_gammas):
        gamma_rows.append({
            "set": "train_seen",
            "gamma": gamma,
            "gamma_class": i,
        })
    for i, gamma in enumerate(interp_gammas):
        gamma_rows.append({
            "set": "test_interp",
            "gamma": gamma,
            "gamma_class": -1,
        })
    write_csv(
        output_dir / "gamma_values.csv",
        ["set", "gamma", "gamma_class"],
        gamma_rows,
    )

    noise_levels = parse_float_list(args.noise_levels)
    assert noise_levels is not None
    if any(level < 0.0 for level in noise_levels):
        raise ValueError("noise levels cannot be negative")

    metadata = {
        "experiment": "gamma-only curriculum",
        "tier": effective_tier,
        "fixed_parameters": {
            "a1": FIXED_A1,
            "a2": FIXED_A2,
            "a3": FIXED_A3,
            "m": FIXED_M,
        },
        "train_gammas": train_gammas,
        "interp_gammas": interp_gammas,
        "q2_range": [physics.q2_min, physics.q2_max],
        "q2_points": physics.q2_points,
        "s_range": [physics.s_min, physics.s_max],
        "output_points": physics.output_points,
        "integration_points": args.integration_points,
        "data_scale": physics.data_scale,
        "noise_definition": "g_noisy = g_clean + noise_level * RMS(g_clean) * N(0,1)",
        "noise_levels": noise_levels,
        "reference_noise_for_separation": args.reference_noise,
        "snr_sep_definition": "||g_i-g_j||_2 / [noise_level * mean(RMS(g_i), RMS(g_j))]",
        "snr_sep_rms_definition": "RMS(g_i-g_j) / [noise_level * mean(RMS(g_i), RMS(g_j))]",
        "split_counts_per_gamma": {
            "train": args.train_per_gamma,
            "val": args.val_per_gamma,
            "test_seen": args.test_seen_per_gamma,
            "test_interp": args.test_interp_per_gamma,
        },
        "independence_rule": "train/val/test_seen/test_interp use disjoint deterministic RNG seed ranges",
        "seed_base": args.seed,
        "npz_fields": [
            "fx",
            "gy_clean",
            "gy_noisy",
            "x",
            "y",
            "q2",
            "parameters",
            "a1",
            "a2",
            "a3",
            "m",
            "gamma",
            "gamma_class",
            "gamma_index",
            "noise_level",
            "noise_sigma",
            "seed",
            "noise_realization_id",
        ],
        "parameter_names": list(PARAMETER_NAMES),
        "notes": [
            "fx is the existing scaled target u(s), so PeakInversionDataset can load these files directly.",
            "gamma_class is 0..K-1 for seen splits and -1 for interpolation samples.",
            "q2_points defaults to 100. Increasing to 500/1000 recomputes the physical forward model; no interpolation is used.",
        ],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"tier={effective_tier}")
    print("train gammas:", ", ".join(f"{x:.9g}" for x in train_gammas))
    print("interp gammas:", ", ".join(f"{x:.9g}" for x in interp_gammas))
    print(f"separation reference noise={args.reference_noise:.6g}")
    print(f"q2=[{physics.q2_min}, {physics.q2_max}], Nq={physics.q2_points}")
    print(f"wrote analysis tables to: {output_dir}")

    if args.analysis_only:
        print("analysis-only: dataset generation skipped")
        return

    split_specs = [
        ("train", train_gammas, args.train_per_gamma, True),
        ("val", train_gammas, args.val_per_gamma, True),
        ("test_seen", train_gammas, args.test_seen_per_gamma, True),
        ("test_interp", interp_gammas, args.test_interp_per_gamma, False),
    ]

    for noise_level in noise_levels:
        noise_dir = output_dir / noise_dir_name(noise_level)
        noise_dir.mkdir(parents=True, exist_ok=True)
        for split, gammas, count, is_seen in split_specs:
            arrays = make_split_arrays(
                split=split,
                gammas=gammas,
                count_per_gamma=count,
                noise_level=noise_level,
                physics=physics,
                integration_points=args.integration_points,
                base_seed=args.seed,
                gamma_class_is_seen=is_seen,
            )
            save_npz(
                noise_dir / f"{split}.npz",
                arrays,
                compressed=args.compressed,
            )
            print(
                f"saved {noise_dir / (split + '.npz')} "
                f"samples={len(arrays['fx']):,}"
            )

    print("done")


if __name__ == "__main__":
    main()
