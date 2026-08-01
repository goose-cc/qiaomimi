from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np

from dataset_config import DatasetConfig
from physics_core import (
    PARAM_NAMES,
    ForwardOperator,
    storage_safety_mask,
    valid_parameter_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Monte Carlo truth shards. Only one copy of params, u(s), "
            "and g_clean is stored per truth; noisy observations are generated online."
        )
    )
    parser.add_argument("--output-dir", default="./truth_shards")
    parser.add_argument("--num-truths", type=int, default=160_000_000)
    parser.add_argument("--truths-per-shard", type=int, default=100_000)
    parser.add_argument("--candidate-batch-size", type=int, default=20_000)
    parser.add_argument("--forward-batch-size", type=int, default=256)
    parser.add_argument("--signal-length", type=int, default=100)
    parser.add_argument("--observation-length", type=int, default=100)
    parser.add_argument("--background-order", type=int, default=96)
    parser.add_argument("--resonance-order", type=int, default=96)
    parser.add_argument("--master-seed", type=int, default=20260801)
    parser.add_argument("--storage-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--data-scale", type=float, default=160_000.0)
    parser.add_argument("--rho-negative-tolerance", type=float, default=0.0)
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument(
        "--shard-stop",
        type=int,
        default=None,
        help="Exclusive shard index. Default: generate through the final shard.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use np.savez_compressed. Saves disk but is substantially slower.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shard files.",
    )
    parser.add_argument(
        "--confirm-large-run",
        action="store_true",
        help="Required when num_truths exceeds one million.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DatasetConfig:
    return DatasetConfig(
        num_truths=args.num_truths,
        truths_per_shard=args.truths_per_shard,
        signal_length=args.signal_length,
        observation_length=args.observation_length,
        background_quadrature_order=args.background_order,
        resonance_quadrature_order=args.resonance_order,
        master_seed=args.master_seed,
        storage_dtype=args.storage_dtype,
        data_scale=args.data_scale,
        rho_negative_tolerance=args.rho_negative_tolerance,
    )


def config_fingerprint(cfg: DatasetConfig) -> str:
    payload = json.dumps(cfg.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def sample_candidates(
    rng: np.random.Generator,
    n: int,
    cfg: DatasetConfig,
) -> np.ndarray:
    unit = rng.random((n, 5), dtype=np.float64)
    params = np.empty_like(unit)
    params[:, 0] = cfg.a1_min + (cfg.a1_max - cfg.a1_min) * unit[:, 0]
    params[:, 1] = cfg.a2_min + (cfg.a2_max - cfg.a2_min) * unit[:, 1]
    params[:, 2] = cfg.a3_min + (cfg.a3_max - cfg.a3_min) * unit[:, 2]
    params[:, 3] = cfg.m_min + (cfg.m_max - cfg.m_min) * unit[:, 3]
    params[:, 4] = cfg.gamma_min + (cfg.gamma_max - cfg.gamma_min) * unit[:, 4]
    return params


def iter_slices(n: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, n, batch_size):
        yield slice(start, min(n, start + batch_size))


def generate_one_shard(
    shard_index: int,
    cfg: DatasetConfig,
    operator: ForwardOperator,
    output_dir: Path,
    candidate_batch_size: int,
    forward_batch_size: int,
    compress: bool,
    overwrite: bool,
    fingerprint: str,
) -> dict:
    target = cfg.shard_size(shard_index)
    path = output_dir / f"truth_{shard_index:06d}.npz"
    if path.exists() and not overwrite:
        print(f"[skip] {path}")
        return {"shard_index": shard_index, "status": "skipped", "path": str(path)}

    # Each shard has an independent, reproducible random stream.
    seed_sequence = np.random.SeedSequence([cfg.master_seed, shard_index])
    rng = np.random.default_rng(seed_sequence)

    params_chunks: list[np.ndarray] = []
    u_chunks: list[np.ndarray] = []
    g_chunks: list[np.ndarray] = []

    accepted = 0
    proposed = 0
    rejected_nonnegative = 0
    rejected_numeric = 0
    started = time.time()

    while accepted < target:
        needed = target - accepted
        propose_n = max(candidate_batch_size, int(math.ceil(needed * 1.4)))
        candidates = sample_candidates(rng, propose_n, cfg)
        proposed += propose_n

        valid_mask, _ = valid_parameter_mask(candidates, cfg)
        rejected_nonnegative += int(propose_n - np.count_nonzero(valid_mask))
        candidates = candidates[valid_mask]
        if candidates.size == 0:
            continue

        for sl in iter_slices(len(candidates), forward_batch_size):
            batch_params = candidates[sl]
            rho, u, g_clean = operator.compute_batch(batch_params)
            numeric_mask = storage_safety_mask(
                rho,
                u,
                g_clean,
                cfg.storage_dtype,
                cfg.finite_safety_fraction,
            )
            rejected_numeric += int(len(batch_params) - np.count_nonzero(numeric_mask))
            if not np.any(numeric_mask):
                continue

            batch_params = batch_params[numeric_mask]
            u = u[numeric_mask]
            g_clean = g_clean[numeric_mask]

            take = min(len(batch_params), target - accepted)
            params_chunks.append(batch_params[:take].copy())
            u_chunks.append(u[:take].astype(cfg.storage_dtype, copy=False))
            g_chunks.append(g_clean[:take].astype(cfg.storage_dtype, copy=False))
            accepted += take
            if accepted >= target:
                break

        elapsed = time.time() - started
        rate = accepted / elapsed if elapsed > 0 else 0.0
        print(
            f"\r[shard {shard_index:06d}] accepted {accepted:,}/{target:,} "
            f"| proposed {proposed:,} | {rate:,.1f} truths/s",
            end="",
            flush=True,
        )
    print()

    params = np.concatenate(params_chunks, axis=0)
    u = np.concatenate(u_chunks, axis=0)
    g_clean = np.concatenate(g_chunks, axis=0)
    truth_start = shard_index * cfg.truths_per_shard
    truth_id = np.arange(truth_start, truth_start + target, dtype=np.int64)

    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": fingerprint,
        "shard_index": shard_index,
        "truth_start": int(truth_start),
        "truth_count": int(target),
        "proposed_count": int(proposed),
        "rejected_nonnegative": int(rejected_nonnegative),
        "rejected_numeric": int(rejected_numeric),
        "compute_dtype": "float64",
        "storage_dtype": cfg.storage_dtype,
        "formula": (
            "rho(s)=a1/pi*(m*gamma)/((s-m)^2+(m*gamma)^2)+a2*s+a3; "
            "u(s)=rho(s)/(s+400)^2; "
            "g(q2)=integral_[s_min,s_max] u(s)/(s-q2) ds"
        ),
        "noise_rule": (
            "g_noisy=g_clean+0.09*RMS(g_clean)*z; "
            "z is generated online from master_seed, truth_id, noise_id"
        ),
    }

    payload = dict(
        params=params,
        u=u,
        g_clean=g_clean,
        truth_id=truth_id,
        s_grid=operator.s_output,
        q2_grid=operator.q2_grid,
        param_names=PARAM_NAMES,
        noise_level=np.float64(cfg.noise_level),
        noises_per_truth=np.int64(cfg.noises_per_truth),
        master_seed=np.int64(cfg.master_seed),
        data_scale=np.float64(cfg.data_scale),
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
    )

    tmp_path = path.with_suffix(".tmp.npz")
    save_func = np.savez_compressed if compress else np.savez
    save_func(tmp_path, **payload)
    os.replace(tmp_path, path)

    elapsed = time.time() - started
    size_mb = path.stat().st_size / (1024**2)
    print(
        f"[saved] {path} | truths={target:,} | "
        f"size={size_mb:,.1f} MiB | elapsed={elapsed/60:.1f} min"
    )
    return {
        "shard_index": shard_index,
        "status": "generated",
        "path": str(path),
        "truth_count": target,
        "proposed_count": proposed,
        "rejected_nonnegative": rejected_nonnegative,
        "rejected_numeric": rejected_numeric,
        "elapsed_seconds": elapsed,
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    if args.num_truths <= 0 or args.truths_per_shard <= 0:
        raise SystemExit("num_truths and truths_per_shard must be positive")
    if args.num_truths > 1_000_000 and not args.confirm_large_run:
        raise SystemExit(
            "Refusing to start a very large generation accidentally. "
            "Re-run with --confirm-large-run after checking disk and compute resources."
        )

    cfg = build_config(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(output_dir / "dataset_config.json")
    fingerprint = config_fingerprint(cfg)

    stop = cfg.num_shards if args.shard_stop is None else args.shard_stop
    if not (0 <= args.shard_start <= stop <= cfg.num_shards):
        raise SystemExit(
            f"Shard range must satisfy 0 <= start <= stop <= {cfg.num_shards}"
        )

    operator = ForwardOperator(cfg)
    run_summary = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": fingerprint,
        "shard_start": args.shard_start,
        "shard_stop": stop,
        "results": [],
    }

    print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
    print(
        "\nImportant: this script stores only truths. "
        "It does not materialize 160 billion noisy samples.\n"
    )

    for shard_index in range(args.shard_start, stop):
        result = generate_one_shard(
            shard_index=shard_index,
            cfg=cfg,
            operator=operator,
            output_dir=output_dir,
            candidate_batch_size=args.candidate_batch_size,
            forward_batch_size=args.forward_batch_size,
            compress=args.compress,
            overwrite=args.overwrite,
            fingerprint=fingerprint,
        )
        run_summary["results"].append(result)
        (output_dir / f"run_{args.shard_start:06d}_{stop:06d}.json").write_text(
            json.dumps(run_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("Generation range completed.")


if __name__ == "__main__":
    main()
