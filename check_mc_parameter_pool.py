from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mc_pool_config import DEFAULT_PHYSICS, read_json
from mc_physics import minimum_rho_on_interval_numpy, valid_parameter_mask_numpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a parameter memmap pool.")
    parser.add_argument("--pool-dir", default="./truth_pool")
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--report-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    root = Path(args.pool_dir)
    metadata = read_json(root / "metadata.json")
    state = read_json(root / "generation_state.json")
    path = root / metadata["parameter_file"]

    target = int(metadata["target_truths"])
    generated = int(state["generated_truths"])
    state_complete = bool(state.get("complete", False))
    metadata_complete = bool(metadata.get("complete", False))
    if state_complete != metadata_complete:
        raise RuntimeError(
            "metadata.json and generation_state.json disagree about completion"
        )
    complete = state_complete

    if metadata.get("dtype") != "float32":
        raise RuntimeError(f"Unexpected pool dtype: {metadata.get('dtype')!r}")
    if list(metadata.get("shape", [])) != [target, 5]:
        raise RuntimeError(f"Unexpected pool shape metadata: {metadata.get('shape')!r}")
    if list(metadata.get("parameter_names", [])) != ["a1", "a2", "a3", "m", "gamma"]:
        raise RuntimeError("Unexpected parameter_names in metadata.json")
    if generated > target:
        raise RuntimeError("generated_truths exceeds target_truths")

    expected_bytes = target * 5 * 4
    actual_bytes = path.stat().st_size

    if args.require_complete and not complete:
        raise RuntimeError("The pool is not marked complete")
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"File size mismatch: expected {expected_bytes}, found {actual_bytes}"
        )
    if generated <= 0:
        raise RuntimeError("No generated parameters are available")

    pool = np.memmap(path, dtype=np.float32, mode="r", shape=(target, 5))
    sample_size = min(int(args.sample_size), generated)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(generated, size=sample_size, replace=False)
    sample = np.asarray(pool[indices], dtype=np.float64)

    finite = np.isfinite(sample).all(axis=1)
    valid = valid_parameter_mask_numpy(sample)
    minima = minimum_rho_on_interval_numpy(sample)
    quantiles = {
        str(q): np.quantile(sample, q, axis=0).tolist()
        for q in (0.0, 0.01, 0.5, 0.99, 1.0)
    }

    report = {
        "pool_dir": str(root),
        "target_truths": target,
        "generated_truths": generated,
        "complete": complete,
        "file_bytes": actual_bytes,
        "file_gb_decimal": actual_bytes / 1e9,
        "sample_size": sample_size,
        "finite_rows": int(finite.sum()),
        "valid_rows": int(valid.sum()),
        "minimum_rho_in_sample": float(np.min(minima)),
        "maximum_rho_minimum_in_sample": float(np.max(minima)),
        "parameter_quantiles": quantiles,
        "acceptance_rate": float(state.get("acceptance_rate", 0.0)),
    }

    print("=" * 72)
    print("MC PARAMETER POOL CHECK")
    print("=" * 72)
    print(f"pool: {root}")
    print(f"generated: {generated:,}/{target:,}")
    print(f"complete: {complete}")
    print(f"file size: {actual_bytes:,} bytes ({actual_bytes/1e9:.3f} GB)")
    print(f"sampled rows: {sample_size:,}")
    print(f"finite rows: {finite.sum():,}/{sample_size:,}")
    print(f"valid rows: {valid.sum():,}/{sample_size:,}")
    print(f"minimum continuous rho(s): {np.min(minima):.8e}")
    print(f"acceptance rate during generation: {report['acceptance_rate']:.2%}")
    names = ("a1", "a2", "a3", "m", "gamma")
    for i, name in enumerate(names):
        print(
            f"{name:>6}: min={sample[:, i].min():.8g} "
            f"median={np.median(sample[:, i]):.8g} "
            f"max={sample[:, i].max():.8g}"
        )

    if not finite.all():
        raise RuntimeError("Non-finite values found")
    if not valid.all():
        bad = np.flatnonzero(~valid)[:10]
        raise RuntimeError(f"Invalid sampled rows found at local sample ids: {bad}")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"JSON report saved: {report_path}")

    print("CHECK PASSED")


if __name__ == "__main__":
    main()
