#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp20 pairwise curriculum dataset generator.

This is the next controlled step after Exp19B.

Modes
-----
a1gamma:
    vary a1 and gamma
    fixed a2=0.025, a3=0, m=0.8

mgamma:
    vary m and gamma
    fixed a1=0.10, a2=0.025, a3=0

Defaults
--------
gamma anchors: 11 log-spaced values in [0.01, 1.0]
a1 anchors   : [0.05, 0.0875, 0.125, 0.1625, 0.20]
m anchors    : [0.4, 0.6, 0.8, 1.0, 1.2]
noise        : 0%, 0.2%, 1%
training     : only the 0.2% directory contains train/val

Interpolation tests are split into three diagnostic sets:
    test_interp_gamma  : second parameter seen, gamma unseen
    test_interp_second : second parameter unseen, gamma seen
    test_interp_both   : both coordinates unseen

No core physical formula is modified.
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


DEFAULT_A1_VALUES = (0.05, 0.0875, 0.125, 0.1625, 0.20)
DEFAULT_M_VALUES = (0.4, 0.6, 0.8, 1.0, 1.2)
FIXED_A1 = 0.10
FIXED_A2 = 0.025
FIXED_A3 = 0.0
FIXED_M = 0.8

SPLIT_CODES = {
    "train": 11,
    "val": 22,
    "test_seen": 33,
    "test_interp_gamma": 44,
    "test_interp_second": 55,
    "test_interp_both": 66,
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


def geometric_midpoints(values: list[float]) -> list[float]:
    return [float(math.sqrt(a * b)) for a, b in zip(values[:-1], values[1:])]


def arithmetic_midpoints(values: list[float]) -> list[float]:
    return [float(0.5 * (a + b)) for a, b in zip(values[:-1], values[1:])]


def second_name(mode: str) -> str:
    return "a1" if mode == "a1gamma" else "m"


def build_parameter_rows(
    mode: str,
    second_values: Iterable[float],
    gamma_values: Iterable[float],
) -> np.ndarray:
    rows = []
    for second in second_values:
        for gamma in gamma_values:
            if mode == "a1gamma":
                rows.append([second, FIXED_A2, FIXED_A3, FIXED_M, gamma])
            elif mode == "mgamma":
                rows.append([FIXED_A1, FIXED_A2, FIXED_A3, second, gamma])
            else:
                raise ValueError(mode)
    return np.asarray(rows, dtype=np.float64)


def forward_bank(
    params: np.ndarray,
    *,
    physics,
    integration_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    fx, gy = scaled_curves_numpy(
        params,
        integration_points=integration_points,
        config=physics,
    )
    return fx.astype(np.float32), gy.astype(np.float32)


def pair_metrics(g1: np.ndarray, g2: np.ndarray, noise_level: float) -> dict[str, float]:
    a = np.asarray(g1, dtype=np.float64)
    b = np.asarray(g2, dtype=np.float64)
    d = a - b
    l2 = float(np.linalg.norm(d))
    rms_d = float(np.sqrt(np.mean(d * d)))
    max_abs = float(np.max(np.abs(d)))
    denom = 0.5 * (np.linalg.norm(a) + np.linalg.norm(b))
    rel_l2 = l2 / denom if denom > 0 else math.inf
    mean_rms = 0.5 * (
        float(np.sqrt(np.mean(a * a))) + float(np.sqrt(np.mean(b * b)))
    )
    sigma = float(noise_level) * mean_rms
    return {
        "max_abs_g_difference": max_abs,
        "rms_g_difference": rms_d,
        "l2_g_difference": l2,
        "relative_l2_difference": float(rel_l2),
        "mean_rms_g": mean_rms,
        "sigma_reference": sigma,
        "snr_sep": l2 / sigma if sigma > 0 else math.inf,
        "snr_sep_rms": rms_d / sigma if sigma > 0 else math.inf,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty csv: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def state_seed(base_seed: int, split: str, second_slot: int, gamma_slot: int) -> int:
    return int(
        base_seed
        + SPLIT_CODES[split] * 1_000_003
        + (second_slot + 1) * 100_003
        + (gamma_slot + 1) * 10_007
    )


def make_split_arrays(
    *,
    mode: str,
    split: str,
    second_values: list[float],
    gamma_values: list[float],
    count_per_state: int,
    noise_level: float,
    physics,
    integration_points: int,
    base_seed: int,
    second_seen: bool,
    gamma_seen: bool,
) -> dict[str, np.ndarray]:
    params_bank = build_parameter_rows(mode, second_values, gamma_values)
    fx_bank, gy_bank = forward_bank(
        params_bank,
        physics=physics,
        integration_points=integration_points,
    )
    s_grid, q2_grid = output_grids_numpy(physics)

    keys = [
        "fx", "gy_clean", "gy_noisy", "parameters", "a1", "a2", "a3", "m",
        "gamma", "second_value", "second_index", "gamma_index", "state_index",
        "second_seen", "gamma_seen", "noise_level", "noise_sigma", "seed",
        "noise_realization_id",
    ]
    blocks: dict[str, list[np.ndarray]] = {key: [] for key in keys}

    state = 0
    for i, second in enumerate(second_values):
        for j, gamma in enumerate(gamma_values):
            p = params_bank[state].astype(np.float32)
            clean = gy_bank[state].astype(np.float32)
            target = fx_bank[state].astype(np.float32)
            rms = float(np.sqrt(np.mean(clean.astype(np.float64) ** 2)))
            sigma = float(noise_level) * rms
            seed = state_seed(base_seed, split, i, j)
            rng = np.random.default_rng(seed)

            if noise_level > 0:
                z = rng.standard_normal(
                    (count_per_state, physics.q2_points)
                ).astype(np.float32)
                noisy = clean[None, :] + np.float32(sigma) * z
            else:
                noisy = np.repeat(clean[None, :], count_per_state, axis=0)

            def repeat_scalar(value, dtype=np.float32):
                return np.full(count_per_state, value, dtype=dtype)

            blocks["fx"].append(np.repeat(target[None, :], count_per_state, axis=0))
            blocks["gy_clean"].append(np.repeat(clean[None, :], count_per_state, axis=0))
            blocks["gy_noisy"].append(noisy.astype(np.float32))
            blocks["parameters"].append(np.repeat(p[None, :], count_per_state, axis=0))
            blocks["a1"].append(repeat_scalar(p[0]))
            blocks["a2"].append(repeat_scalar(p[1]))
            blocks["a3"].append(repeat_scalar(p[2]))
            blocks["m"].append(repeat_scalar(p[3]))
            blocks["gamma"].append(repeat_scalar(p[4]))
            blocks["second_value"].append(repeat_scalar(second))
            blocks["second_index"].append(repeat_scalar(i if second_seen else -1, np.int16))
            blocks["gamma_index"].append(repeat_scalar(j if gamma_seen else -1, np.int16))
            blocks["state_index"].append(repeat_scalar(state, np.int32))
            blocks["second_seen"].append(repeat_scalar(1 if second_seen else 0, np.int8))
            blocks["gamma_seen"].append(repeat_scalar(1 if gamma_seen else 0, np.int8))
            blocks["noise_level"].append(repeat_scalar(noise_level))
            blocks["noise_sigma"].append(repeat_scalar(sigma))
            blocks["seed"].append(repeat_scalar(seed, np.int64))
            blocks["noise_realization_id"].append(
                np.arange(count_per_state, dtype=np.int32)
            )
            state += 1

    out = {key: np.concatenate(parts, axis=0) for key, parts in blocks.items()}
    perm_seed = base_seed + SPLIT_CODES[split] * 9_999_991
    perm = np.random.default_rng(perm_seed).permutation(len(out["gamma"]))
    for key in list(out):
        out[key] = out[key][perm]

    out["x"] = s_grid.astype(np.float32)
    out["y"] = q2_grid.astype(np.float32)
    out["q2"] = q2_grid.astype(np.float32)
    return out


def save_npz(path: Path, arrays: dict[str, np.ndarray], compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Exp20 pairwise curriculum data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=("a1gamma", "mgamma"), required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gamma-count", type=int, default=11)
    p.add_argument("--gamma-min", type=float, default=0.01)
    p.add_argument("--gamma-max", type=float, default=1.0)
    p.add_argument("--a1-values", default=",".join(map(str, DEFAULT_A1_VALUES)))
    p.add_argument("--m-values", default=",".join(map(str, DEFAULT_M_VALUES)))
    p.add_argument("--noise-levels", default="0,0.002,0.01")
    p.add_argument("--train-noise-level", type=float, default=0.002)
    p.add_argument("--reference-noise", type=float, default=0.002)
    p.add_argument("--q2-points", type=int, default=100)
    p.add_argument("--integration-points", type=int, default=512)

    p.add_argument("--train-per-state", type=int, default=360)
    p.add_argument("--val-per-state", type=int, default=45)
    p.add_argument("--test-seen-per-state", type=int, default=80)
    p.add_argument("--test-interp-per-state", type=int, default=50)

    p.add_argument("--seed", type=int, default=20260813)
    p.add_argument("--compressed", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.gamma_count < 3:
        raise ValueError("--gamma-count must be >= 3")
    if not (0 < args.gamma_min < args.gamma_max):
        raise ValueError("require 0 < gamma_min < gamma_max")
    if args.reference_noise <= 0:
        raise ValueError("--reference-noise must be > 0")
    if args.q2_points < 2:
        raise ValueError("--q2-points must be >= 2")

    second_values = (
        parse_float_list(args.a1_values)
        if args.mode == "a1gamma"
        else parse_float_list(args.m_values)
    )
    assert second_values is not None
    second_values = sorted(float(x) for x in second_values)
    if len(second_values) < 3 or len(set(second_values)) != len(second_values):
        raise ValueError("second-parameter anchors must contain >=3 unique values")

    gamma_values = np.geomspace(
        args.gamma_min, args.gamma_max, args.gamma_count, dtype=np.float64
    ).tolist()
    gamma_mid = geometric_midpoints(gamma_values)
    second_mid = arithmetic_midpoints(second_values)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} is not empty; use --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    physics = replace(DEFAULT_PHYSICS, q2_points=int(args.q2_points))

    # Clean training-anchor bank and separation diagnostics.
    train_params = build_parameter_rows(args.mode, second_values, gamma_values)
    _, train_g = forward_bank(
        train_params,
        physics=physics,
        integration_points=args.integration_points,
    )

    state_rows = []
    n_gamma = len(gamma_values)
    for i, second in enumerate(second_values):
        for j, gamma in enumerate(gamma_values):
            state_rows.append({
                "state_index": i * n_gamma + j,
                second_name(args.mode): second,
                "gamma": gamma,
            })
    write_csv(output_dir / "train_states.csv", state_rows)

    pair_rows: list[dict] = []
    nearest_rows: list[dict] = []
    for a, b in combinations(range(len(train_params)), 2):
        metrics = pair_metrics(train_g[a], train_g[b], args.reference_noise)
        pair_rows.append({
            "state_1": a,
            "state_2": b,
            f"{second_name(args.mode)}_1": state_rows[a][second_name(args.mode)],
            f"{second_name(args.mode)}_2": state_rows[b][second_name(args.mode)],
            "gamma_1": state_rows[a]["gamma"],
            "gamma_2": state_rows[b]["gamma"],
            "reference_noise_level": args.reference_noise,
            **metrics,
        })
    write_csv(output_dir / "state_pair_separation.csv", pair_rows)

    # For every training state, record its closest alternative in g space.
    for i in range(len(train_params)):
        candidates = [row for row in pair_rows if row["state_1"] == i or row["state_2"] == i]
        best = min(candidates, key=lambda r: r["relative_l2_difference"])
        j = best["state_2"] if best["state_1"] == i else best["state_1"]
        nearest_rows.append({
            "state": i,
            "nearest_state": j,
            second_name(args.mode): state_rows[i][second_name(args.mode)],
            f"nearest_{second_name(args.mode)}": state_rows[j][second_name(args.mode)],
            "gamma": state_rows[i]["gamma"],
            "nearest_gamma": state_rows[j]["gamma"],
            "relative_l2_difference": best["relative_l2_difference"],
            "snr_sep": best["snr_sep"],
            "snr_sep_rms": best["snr_sep_rms"],
        })
    write_csv(output_dir / "state_nearest_neighbor_separation.csv", nearest_rows)

    gamma_adjacent = []
    for i, second in enumerate(second_values):
        for j in range(len(gamma_values) - 1):
            a = i * n_gamma + j
            b = i * n_gamma + j + 1
            gamma_adjacent.append({
                second_name(args.mode): second,
                "gamma_1": gamma_values[j],
                "gamma_2": gamma_values[j + 1],
                "reference_noise_level": args.reference_noise,
                **pair_metrics(train_g[a], train_g[b], args.reference_noise),
            })
    write_csv(output_dir / "gamma_adjacent_separation.csv", gamma_adjacent)

    second_adjacent = []
    for i in range(len(second_values) - 1):
        for j, gamma in enumerate(gamma_values):
            a = i * n_gamma + j
            b = (i + 1) * n_gamma + j
            second_adjacent.append({
                f"{second_name(args.mode)}_1": second_values[i],
                f"{second_name(args.mode)}_2": second_values[i + 1],
                "gamma": gamma,
                "reference_noise_level": args.reference_noise,
                **pair_metrics(train_g[a], train_g[b], args.reference_noise),
            })
    write_csv(output_dir / f"{second_name(args.mode)}_adjacent_separation.csv", second_adjacent)

    noise_levels = parse_float_list(args.noise_levels)
    assert noise_levels is not None
    if not any(np.isclose(args.train_noise_level, x, atol=1e-12, rtol=0) for x in noise_levels):
        raise ValueError("--train-noise-level must be included in --noise-levels")

    split_specs = {
        "test_seen": (second_values, gamma_values, args.test_seen_per_state, True, True),
        "test_interp_gamma": (
            second_values, gamma_mid, args.test_interp_per_state, True, False
        ),
        "test_interp_second": (
            second_mid, gamma_values, args.test_interp_per_state, False, True
        ),
        "test_interp_both": (
            second_mid, gamma_mid, args.test_interp_per_state, False, False
        ),
    }

    for noise_level in noise_levels:
        d = output_dir / noise_dir_name(noise_level)
        d.mkdir(parents=True, exist_ok=True)

        if np.isclose(noise_level, args.train_noise_level, atol=1e-12, rtol=0):
            for split, count in (
                ("train", args.train_per_state),
                ("val", args.val_per_state),
            ):
                arrays = make_split_arrays(
                    mode=args.mode,
                    split=split,
                    second_values=second_values,
                    gamma_values=gamma_values,
                    count_per_state=count,
                    noise_level=noise_level,
                    physics=physics,
                    integration_points=args.integration_points,
                    base_seed=args.seed,
                    second_seen=True,
                    gamma_seen=True,
                )
                save_npz(d / f"{split}.npz", arrays, args.compressed)
                print(f"saved {d / (split + '.npz')} samples={len(arrays['gamma']):,}")

        for split, (sv, gv, count, s_seen, g_seen) in split_specs.items():
            arrays = make_split_arrays(
                mode=args.mode,
                split=split,
                second_values=sv,
                gamma_values=gv,
                count_per_state=count,
                noise_level=noise_level,
                physics=physics,
                integration_points=args.integration_points,
                base_seed=args.seed,
                second_seen=s_seen,
                gamma_seen=g_seen,
            )
            save_npz(d / f"{split}.npz", arrays, args.compressed)
            print(f"saved {d / (split + '.npz')} samples={len(arrays['gamma']):,}")

    nearest_snr = np.asarray([r["snr_sep"] for r in nearest_rows], dtype=np.float64)
    metadata = {
        "experiment": "Exp20 pairwise curriculum",
        "mode": args.mode,
        "second_parameter": second_name(args.mode),
        "fixed_parameters": (
            {"a2": FIXED_A2, "a3": FIXED_A3, "m": FIXED_M}
            if args.mode == "a1gamma"
            else {"a1": FIXED_A1, "a2": FIXED_A2, "a3": FIXED_A3}
        ),
        "second_train_values": second_values,
        "second_interp_values": second_mid,
        "gamma_train_values": gamma_values,
        "gamma_interp_values": gamma_mid,
        "noise_levels": noise_levels,
        "train_noise_level": args.train_noise_level,
        "reference_noise": args.reference_noise,
        "q2_range": [physics.q2_min, physics.q2_max],
        "q2_points": physics.q2_points,
        "output_points": physics.output_points,
        "integration_points": args.integration_points,
        "data_scale": physics.data_scale,
        "counts_per_state": {
            "train": args.train_per_state,
            "val": args.val_per_state,
            "test_seen": args.test_seen_per_state,
            "test_interp_*": args.test_interp_per_state,
        },
        "training_state_count": len(second_values) * len(gamma_values),
        "nearest_neighbor_snr_at_reference_noise": {
            "min": float(np.min(nearest_snr)),
            "median": float(np.median(nearest_snr)),
            "p90": float(np.quantile(nearest_snr, 0.90)),
        },
        "noise_definition": "g_noisy = g_clean + noise_level * RMS(g_clean) * N(0,1)",
        "parameter_names": list(PARAMETER_NAMES),
        "notes": [
            "Physical formula is unchanged.",
            "train/val exist only at the designated train-noise level.",
            "test_interp_gamma isolates unseen gamma with seen second parameter.",
            "test_interp_second isolates unseen second parameter with seen gamma.",
            "test_interp_both tests genuine 2D interpolation.",
            "state_nearest_neighbor_separation.csv checks whether different 2D physical states are already close in g-space before training.",
        ],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("=" * 88)
    print(f"Exp20 data ready: mode={args.mode}, states={metadata['training_state_count']}")
    print(f"second values: {second_values}")
    print(f"gamma count: {len(gamma_values)}")
    print(
        "nearest-state SNR @ reference noise: "
        f"min={np.min(nearest_snr):.4g}, median={np.median(nearest_snr):.4g}"
    )
    print(f"output: {output_dir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
