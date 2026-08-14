#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp22B: regenerate the SAME Exp22A parameter classes at denser q^2 sampling.

Scientific control
------------------
Exp22A selected a fixed 9 x 3 a1-gamma grid using RMS-SNR >= 2 at N_q=100.
Exp22B must NOT re-select the parameter grid.  It reads the selected anchors
from the Exp22A metadata, keeps the same truths/noise levels/sample counts, and
changes only the number of q^2 observation points.

This lets us test whether denser *true forward evaluations* of g(q^2) add
useful information.  No interpolation of the original 100-point curves is used.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from data_generate_exp20_pairwise import (
    DEFAULT_PHYSICS,
    FIXED_A2,
    FIXED_A3,
    FIXED_M,
    arithmetic_midpoints,
    geometric_midpoints,
    make_split_arrays,
    noise_dir_name,
    save_npz,
    write_csv,
)
from data_generate_exp22_rms_gseparated_a1gamma import pair_diagnostics


def parse_float_list(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exp22B fixed-class q^2 sampling-density dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source-metadata",
        required=True,
        help="Exp22A metadata.json containing the selected a1/gamma anchors",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--q2-points", type=int, required=True)
    p.add_argument("--reference-noise", type=float, default=0.002)
    p.add_argument("--noise-levels", type=parse_float_list, default=[0.0, 0.002, 0.01])
    p.add_argument("--train-noise-level", type=float, default=0.002)
    p.add_argument("--integration-points", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260813)

    # 0 means: reuse the Exp22A count-per-state stored in source metadata.
    p.add_argument("--train-per-state", type=int, default=0)
    p.add_argument("--val-per-state", type=int, default=0)
    p.add_argument("--test-seen-per-state", type=int, default=0)
    p.add_argument("--test-interp-per-state", type=int, default=0)

    lean_group = p.add_mutually_exclusive_group()
    lean_group.add_argument(
        "--lean-train-files",
        dest="lean_train_files",
        action="store_true",
        help="remove repeated clean/physics arrays from train/val NPZ to reduce disk use",
    )
    lean_group.add_argument(
        "--no-lean-train-files",
        dest="lean_train_files",
        action="store_false",
        help="keep repeated clean/physics arrays in train/val NPZ",
    )
    p.set_defaults(lean_train_files=True)
    p.add_argument("--compressed", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def source_count(meta: dict, key: str) -> int:
    counts = meta.get("counts_per_state", {})
    if key not in counts:
        raise KeyError(
            f"source metadata does not contain counts_per_state[{key!r}]; "
            "use the explicit --*-per-state override"
        )
    return int(counts[key])


def resolve_counts(args: argparse.Namespace, meta: dict) -> dict[str, int]:
    test_interp_default = source_count(meta, "test_interp_gamma")
    out = {
        "train": int(args.train_per_state) if args.train_per_state > 0 else source_count(meta, "train"),
        "val": int(args.val_per_state) if args.val_per_state > 0 else source_count(meta, "val"),
        "test_seen": (
            int(args.test_seen_per_state)
            if args.test_seen_per_state > 0
            else source_count(meta, "test_seen")
        ),
        "test_interp_gamma": (
            int(args.test_interp_per_state)
            if args.test_interp_per_state > 0
            else test_interp_default
        ),
        "test_interp_second": (
            int(args.test_interp_per_state)
            if args.test_interp_per_state > 0
            else source_count(meta, "test_interp_second")
        ),
        "test_interp_both": (
            int(args.test_interp_per_state)
            if args.test_interp_per_state > 0
            else source_count(meta, "test_interp_both")
        ),
    }
    if any(v <= 0 for v in out.values()):
        raise ValueError(f"all per-state counts must be positive, got {out}")
    return out


def slim_for_training(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Keep only arrays needed by train_exp20_pairwise.py for train/val."""
    keep = {
        "gy_noisy",
        "gamma",
        "a1",
        "noise_level",
        "q2",
        "y",
        "x",
    }
    return {k: v for k, v in arrays.items() if k in keep}


def plot_nearest_hist(
    output_dir: Path,
    nearest_rows: list[dict],
    *,
    q2_points: int,
    reference_noise: float,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: plotting skipped: {exc}")
        return

    rms = np.asarray([row["snr_sep_rms"] for row in nearest_rows], dtype=float)
    full = np.asarray([row["snr_sep"] for row in nearest_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rms, bins=min(30, max(5, len(rms) // 2)))
    ax.set_xlabel("nearest-state RMS-SNR")
    ax.set_ylabel("state count")
    ax.set_title(f"Exp22B nearest-state RMS-SNR | Nq={q2_points}")
    fig.tight_layout()
    fig.savefig(output_dir / "nearest_state_rms_snr_hist.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(full, bins=min(30, max(5, len(full) // 2)))
    ax.set_xlabel("nearest-state full-curve SNR")
    ax.set_ylabel("state count")
    ax.set_title(f"Exp22B nearest-state full SNR | Nq={q2_points}")
    fig.tight_layout()
    fig.savefig(output_dir / "nearest_state_full_snr_hist.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_metadata)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if args.q2_points < 2:
        raise ValueError("--q2-points must be >= 2")
    if args.reference_noise <= 0:
        raise ValueError("--reference-noise must be > 0")
    if not any(
        np.isclose(args.train_noise_level, x, atol=1e-12, rtol=0)
        for x in args.noise_levels
    ):
        raise ValueError("--train-noise-level must be included in --noise-levels")

    source_meta = json.loads(source_path.read_text(encoding="utf-8"))
    if source_meta.get("mode") != "a1gamma":
        raise ValueError("Exp22B expects an Exp22A a1gamma metadata file")

    a1_values = [float(x) for x in source_meta["second_train_values"]]
    gamma_values = [float(x) for x in source_meta["gamma_train_values"]]
    if len(a1_values) < 2 or len(gamma_values) < 2:
        raise ValueError("source metadata contains too few anchors")

    # Recompute the midpoint test truths from the fixed Exp22A training anchors.
    a1_mid = arithmetic_midpoints(a1_values)
    gamma_mid = geometric_midpoints(gamma_values)
    counts = resolve_counts(args, source_meta)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} is not empty; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: same q^2 range, new physical forward evaluations at Nq points.
    physics = replace(DEFAULT_PHYSICS, q2_points=int(args.q2_points))

    print("=" * 100)
    print("Exp22B: fixed Exp22A classes, denser q^2 observations")
    print("Physical formula : UNCHANGED")
    print("Parameter classes: UNCHANGED from Exp22A")
    print(f"source metadata  : {source_path}")
    print(f"q2 range         : [{physics.q2_min}, {physics.q2_max}]")
    print(f"q2 points        : {physics.q2_points}")
    print(f"reference noise  : {100 * args.reference_noise:.4g}%")
    print(f"a1 anchors       : {len(a1_values)}")
    print(f"gamma anchors    : {len(gamma_values)} -> {np.asarray(gamma_values)}")
    print(f"training states  : {len(a1_values) * len(gamma_values)}")
    print("=" * 100)

    pair_rows, nearest_rows, sep = pair_diagnostics(
        a1_values,
        gamma_values,
        physics=physics,
        integration_points=args.integration_points,
        reference_noise=args.reference_noise,
    )
    write_csv(output_dir / "state_pair_separation.csv", pair_rows)
    write_csv(output_dir / "state_nearest_neighbor_separation.csv", nearest_rows)

    sep_summary = {
        **sep,
        "q2_points": int(physics.q2_points),
        "q2_min": float(physics.q2_min),
        "q2_max": float(physics.q2_max),
        "reference_noise": float(args.reference_noise),
        "sqrt_q2_points": float(np.sqrt(physics.q2_points)),
        "expected_full_over_rms": float(np.sqrt(physics.q2_points)),
        "observed_min_full_over_rms": float(sep["min_snr"] / sep["min_snr_rms"]),
    }
    (output_dir / "sampling_separation_summary.json").write_text(
        json.dumps(sep_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_nearest_hist(
        output_dir,
        nearest_rows,
        q2_points=physics.q2_points,
        reference_noise=args.reference_noise,
    )

    split_defs = {
        "test_seen": (a1_values, gamma_values, True, True),
        "test_interp_gamma": (a1_values, gamma_mid, True, False),
        "test_interp_second": (a1_mid, gamma_values, False, True),
        "test_interp_both": (a1_mid, gamma_mid, False, False),
    }

    for noise_level in args.noise_levels:
        ndir = output_dir / noise_dir_name(noise_level)
        ndir.mkdir(parents=True, exist_ok=True)

        if np.isclose(noise_level, args.train_noise_level, atol=1e-12, rtol=0):
            for split in ("train", "val"):
                arrays = make_split_arrays(
                    mode="a1gamma",
                    split=split,
                    second_values=a1_values,
                    gamma_values=gamma_values,
                    count_per_state=counts[split],
                    noise_level=noise_level,
                    physics=physics,
                    integration_points=args.integration_points,
                    base_seed=args.seed,
                    second_seen=True,
                    gamma_seen=True,
                )
                if args.lean_train_files:
                    arrays = slim_for_training(arrays)
                save_npz(ndir / f"{split}.npz", arrays, args.compressed)
                print(
                    f"saved {ndir / (split + '.npz')} "
                    f"samples={len(arrays['gamma']):,}"
                )

        for split, (avals, gvals, a_seen, g_seen) in split_defs.items():
            arrays = make_split_arrays(
                mode="a1gamma",
                split=split,
                second_values=avals,
                gamma_values=gvals,
                count_per_state=counts[split],
                noise_level=noise_level,
                physics=physics,
                integration_points=args.integration_points,
                base_seed=args.seed,
                second_seen=a_seen,
                gamma_seen=g_seen,
            )
            save_npz(ndir / f"{split}.npz", arrays, args.compressed)
            print(
                f"saved {ndir / (split + '.npz')} "
                f"samples={len(arrays['gamma']):,}"
            )

    metadata = {
        "experiment": "Exp22B fixed-class q2-density control",
        "mode": "a1gamma",
        "second_parameter": "a1",
        "fixed_parameters": {"a2": FIXED_A2, "a3": FIXED_A3, "m": FIXED_M},
        "source_exp22a_metadata": str(source_path),
        "controlled_change": "q2_points only",
        "second_train_values": a1_values,
        "second_interp_values": a1_mid,
        "gamma_train_values": gamma_values,
        "gamma_interp_values": gamma_mid,
        "training_state_count": len(a1_values) * len(gamma_values),
        "selected_separation": sep,
        "sampling_separation_summary": sep_summary,
        "noise_levels": [float(x) for x in args.noise_levels],
        "train_noise_level": float(args.train_noise_level),
        "reference_noise": float(args.reference_noise),
        "q2_range": [float(physics.q2_min), float(physics.q2_max)],
        "q2_points": int(physics.q2_points),
        "output_points": int(physics.output_points),
        "integration_points": int(args.integration_points),
        "counts_per_state": counts,
        "noise_definition": "g_noisy = g_clean + noise_level * RMS(g_clean) * N(0,1)",
        "notes": [
            "The 9x3 Exp22A a1-gamma training classes are fixed and are not re-selected.",
            "g(q^2) is recomputed by the physical forward model at the requested q2_points; no interpolation from the 100-point input is used.",
            "RMS-SNR should be nearly sampling-density invariant when the q2 range is unchanged.",
            "Full-curve SNR may grow approximately as sqrt(Nq) if the added noisy observations contribute independent evidence.",
            "The MLP hidden widths are unchanged, but its first linear layer has more parameters when Nq increases; the aggregate report records this caveat.",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("Exp22B data ready")
    print(
        "nearest-state separation: "
        f"min RMS-SNR={sep['min_snr_rms']:.4g}, "
        f"min full-SNR={sep['min_snr']:.4g}, "
        f"median RMS-SNR={sep['median_nearest_snr_rms']:.4g}"
    )
    print(f"output: {output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
