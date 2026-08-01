from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np

from dataset_config import DatasetConfig
from physics_core import ForwardOperator, valid_parameter_mask
from online_noise_dataset import stateless_standard_normal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated truth shards.")
    parser.add_argument("--root", default="./truth_shards")
    parser.add_argument("--expected-truths", type=int, default=None)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--reintegrate-samples", type=int, default=20)
    parser.add_argument("--noise-test-truths", type=int, default=100)
    parser.add_argument("--noise-test-count", type=int, default=1000)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    config_path = root / "dataset_config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing {config_path}")

    cfg_data = json.loads(config_path.read_text(encoding="utf-8"))
    valid_fields = DatasetConfig.__dataclass_fields__.keys()
    cfg = DatasetConfig(**{k: cfg_data[k] for k in valid_fields if k in cfg_data})
    operator = ForwardOperator(cfg)

    shards = sorted(root.glob("truth_*.npz"))
    if not shards:
        raise SystemExit(f"No shards found under {root}")

    ranges = [
        (cfg.a1_min, cfg.a1_max),
        (cfg.a2_min, cfg.a2_max),
        (cfg.a3_min, cfg.a3_max),
        (cfg.m_min, cfg.m_max),
        (cfg.gamma_min, cfg.gamma_max),
    ]
    hist = np.zeros((5, args.bins), dtype=np.int64)
    total = 0
    expected_next_id = 0
    reintegrate_pool = []
    noise_pool = []
    worst_min_rho = np.inf
    max_forward_relative_error = 0.0

    for path in shards:
        with np.load(path, allow_pickle=False) as data:
            params = np.asarray(data["params"], dtype=np.float64)
            u = np.asarray(data["u"], dtype=np.float64)
            g = np.asarray(data["g_clean"], dtype=np.float64)
            ids = np.asarray(data["truth_id"], dtype=np.int64)

        n = len(ids)
        if params.shape != (n, 5):
            raise RuntimeError(f"Bad params shape in {path}: {params.shape}")
        if u.shape[0] != n or g.shape[0] != n:
            raise RuntimeError(f"Bad truth array shape in {path}")
        if ids[0] != expected_next_id:
            raise RuntimeError(
                f"truth_id discontinuity at {path}: expected {expected_next_id}, got {ids[0]}"
            )
        if not np.array_equal(ids, np.arange(ids[0], ids[0] + n, dtype=np.int64)):
            raise RuntimeError(f"Non-consecutive truth_id inside {path}")
        expected_next_id += n
        total += n

        mask, minimum = valid_parameter_mask(params, cfg)
        if not np.all(mask):
            bad = np.flatnonzero(~mask)[:10]
            raise RuntimeError(f"Invalid parameters in {path}, rows {bad.tolist()}")
        worst_min_rho = min(worst_min_rho, float(np.min(minimum)))

        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(g))):
            raise RuntimeError(f"NaN/Inf found in {path}")

        for j, (lo, hi) in enumerate(ranges):
            counts, _ = np.histogram(params[:, j], bins=args.bins, range=(lo, hi))
            hist[j] += counts

        # Deterministic reservoir-like sample from each shard.
        take_re = min(args.reintegrate_samples, n)
        if take_re > 0:
            idx = np.linspace(0, n - 1, take_re, dtype=int)
            reintegrate_pool.extend((params[i], u[i], g[i]) for i in idx)

        take_noise = min(args.noise_test_truths, n)
        if take_noise > 0 and len(noise_pool) < args.noise_test_truths:
            idx = np.linspace(0, n - 1, take_noise, dtype=int)
            for i in idx:
                noise_pool.append((int(ids[i]), g[i]))
                if len(noise_pool) >= args.noise_test_truths:
                    break

        print(f"[ok] {path.name}: {n:,} truths")

    if args.expected_truths is not None and total != args.expected_truths:
        raise RuntimeError(f"Expected {args.expected_truths:,} truths, found {total:,}")

    # Recompute selected truths and compare.
    if reintegrate_pool:
        params_sample = np.stack([x[0] for x in reintegrate_pool], axis=0)
        _, u_new, g_new = operator.compute_batch(params_sample)
        u_old = np.stack([x[1] for x in reintegrate_pool], axis=0)
        g_old = np.stack([x[2] for x in reintegrate_pool], axis=0)

        u_rel = np.linalg.norm(u_new - u_old) / max(np.linalg.norm(u_new), 1e-300)
        g_rel = np.linalg.norm(g_new - g_old) / max(np.linalg.norm(g_new), 1e-300)
        max_forward_relative_error = float(max(u_rel, g_rel))

    # Check the online 9% RMS noise rule over a pool of truths/noise IDs.
    ratios = []
    for truth_id, g_clean in noise_pool:
        g_clean = np.asarray(g_clean, dtype=np.float64)
        rms = np.sqrt(np.mean(g_clean * g_clean))
        if rms == 0.0:
            continue
        zs = []
        for noise_id in range(args.noise_test_count):
            z = stateless_standard_normal(
                np.array([truth_id]),
                noise_id,
                len(g_clean),
                cfg.master_seed,
            )[0]
            zs.append(z)
        z_all = np.stack(zs, axis=0)
        noisy = g_clean[None, :] + cfg.noise_level * rms * z_all
        ratio = (
            np.sqrt(np.mean((noisy - g_clean[None, :]) ** 2))
            / rms
        )
        ratios.append(float(ratio))

    result = {
        "root": str(root),
        "shards": len(shards),
        "truths": total,
        "worst_continuous_min_rho": worst_min_rho,
        "recompute_relative_error": max_forward_relative_error,
        "noise_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "noise_ratio_std": float(np.std(ratios)) if ratios else None,
        "histogram_bins": args.bins,
        "histogram_counts": {
            name: hist[i].tolist()
            for i, name in enumerate(["a1", "a2", "a3", "m", "gamma"])
        },
    }

    output = Path(args.output_json) if args.output_json else root / "validation_report.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Validation report saved to: {output}")


if __name__ == "__main__":
    main()
