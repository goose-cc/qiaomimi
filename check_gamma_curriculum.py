#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Consistency checker for data_generate_gamma_curriculum.py outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED = {
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
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    return p.parse_args()


def unique_sorted(x):
    return np.unique(np.asarray(x, dtype=np.float64))


def check_file(path: Path, metadata: dict) -> dict:
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED - set(data.files)
        if missing:
            raise AssertionError(f"{path}: missing keys {sorted(missing)}")

        n = len(data["fx"])
        for key in (
            "gy_clean", "gy_noisy", "parameters", "a1", "a2", "a3", "m",
            "gamma", "gamma_class", "gamma_index", "noise_level", "noise_sigma",
            "seed", "noise_realization_id",
        ):
            if len(data[key]) != n:
                raise AssertionError(f"{path}: {key} length mismatch")

        if data["fx"].shape[1] != metadata["output_points"]:
            raise AssertionError(f"{path}: fx point count mismatch")
        if data["gy_clean"].shape[1] != metadata["q2_points"]:
            raise AssertionError(f"{path}: gy point count mismatch")
        if not np.isfinite(data["fx"]).all():
            raise AssertionError(f"{path}: non-finite fx")
        if not np.isfinite(data["gy_clean"]).all():
            raise AssertionError(f"{path}: non-finite gy_clean")
        if not np.isfinite(data["gy_noisy"]).all():
            raise AssertionError(f"{path}: non-finite gy_noisy")

        fixed = metadata["fixed_parameters"]
        for key in ("a1", "a2", "a3", "m"):
            if not np.allclose(data[key], fixed[key], rtol=0.0, atol=1e-7):
                raise AssertionError(f"{path}: {key} is not fixed at {fixed[key]}")

        if not np.allclose(data["y"], data["q2"], rtol=0.0, atol=0.0):
            raise AssertionError(f"{path}: y and q2 are inconsistent")

        level_values = unique_sorted(data["noise_level"])
        if len(level_values) != 1:
            raise AssertionError(f"{path}: multiple noise levels in one file")
        noise_level = float(level_values[0])

        if noise_level == 0.0:
            if not np.array_equal(data["gy_clean"], data["gy_noisy"]):
                raise AssertionError(f"{path}: zero-noise file differs from clean")
            empirical = 0.0
        else:
            noise = data["gy_noisy"].astype(np.float64) - data["gy_clean"].astype(np.float64)
            sigma = data["noise_sigma"].astype(np.float64)
            row_rms = np.sqrt(np.mean(noise * noise, axis=1))
            empirical = float(np.mean(row_rms / np.maximum(sigma, 1e-30)))

        return {
            "n": n,
            "noise_level": noise_level,
            "gammas": unique_sorted(data["gamma"]),
            "seeds": set(np.asarray(data["seed"], dtype=np.int64).tolist()),
            "empirical_noise_rms_over_sigma": empirical,
        }


def main():
    args = parse_args()
    root = Path(args.data_dir)
    with (root / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    train_gammas = np.asarray(metadata["train_gammas"], dtype=np.float64)
    interp_gammas = np.asarray(metadata["interp_gammas"], dtype=np.float64)
    if np.any(np.isclose(train_gammas[:, None], interp_gammas[None, :], rtol=0.0, atol=1e-12)):
        raise AssertionError("test_interp contains a training gamma")

    noise_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("noise_"))
    if not noise_dirs:
        raise AssertionError("no noise_* directories found")

    for noise_dir in noise_dirs:
        summaries = {}
        for split in ("train", "val", "test_seen", "test_interp"):
            path = noise_dir / f"{split}.npz"
            if not path.exists():
                raise AssertionError(f"missing {path}")
            summaries[split] = check_file(path, metadata)

        # Seen split gamma support must match exactly.
        for split in ("train", "val", "test_seen"):
            if not np.allclose(
                summaries[split]["gammas"],
                train_gammas,
                rtol=0.0,
                atol=1e-7,
            ):
                raise AssertionError(f"{noise_dir}/{split}: seen gamma set mismatch")

        if not np.allclose(
            summaries["test_interp"]["gammas"],
            interp_gammas,
            rtol=0.0,
            atol=1e-7,
        ):
            raise AssertionError(f"{noise_dir}/test_interp: interpolation gamma set mismatch")

        # Seed ranges must be disjoint across all four splits.
        split_names = list(summaries)
        for i, a in enumerate(split_names):
            for b in split_names[i + 1:]:
                overlap = summaries[a]["seeds"] & summaries[b]["seeds"]
                if overlap:
                    raise AssertionError(f"{noise_dir}: seed overlap between {a} and {b}")

        print(f"[OK] {noise_dir.name}")
        for split in split_names:
            s = summaries[split]
            print(
                f"  {split:11s} n={s['n']:6d} "
                f"gammas={len(s['gammas'])} "
                f"noise={s['noise_level']:.6g} "
                f"RMS(noise)/sigma={s['empirical_noise_rms_over_sigma']:.3f}"
            )

    print("all checks passed")


if __name__ == "__main__":
    main()
