from __future__ import annotations

import argparse
import shutil
import signal
import time
from pathlib import Path

import numpy as np

from mc_pool_config import (
    DEFAULT_PHYSICS,
    PARAMETER_NAMES,
    atomic_write_json,
    read_json,
)
from mc_physics import minimum_rho_on_interval_numpy, valid_parameter_mask_numpy


STOP_REQUESTED = False


def _request_stop(signum, frame):
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n收到停止信号；当前候选批次完成后会安全保存进度。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a float32 memmap containing valid 5-D parameters."
    )
    parser.add_argument("--output-dir", default="./truth_pool")
    parser.add_argument("--num-truths", type=int, default=160_000_000)
    parser.add_argument("--candidate-batch-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-candidate-batches",
        type=int,
        default=0,
        help="0 means unlimited; useful for testing safe resume.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_truths <= 0:
        raise ValueError("--num-truths must be positive")
    if args.candidate_batch_size <= 0:
        raise ValueError("--candidate-batch-size must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")

    output_dir = Path(args.output_dir)
    data_path = output_dir / "parameters.dat"
    metadata_path = output_dir / "metadata.json"
    state_path = output_dir / "generation_state.json"

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    if args.resume:
        if not (data_path.exists() and metadata_path.exists() and state_path.exists()):
            raise FileNotFoundError(
                "Resume requires parameters.dat, metadata.json, and generation_state.json"
            )
        metadata = read_json(metadata_path)
        state = read_json(state_path)
        if int(metadata["target_truths"]) != int(args.num_truths):
            raise ValueError(
                "The requested --num-truths differs from the existing pool target"
            )
        if metadata.get("dtype") != "float32":
            raise RuntimeError("Existing pool dtype is not float32")
        if list(metadata.get("shape", [])) != [int(args.num_truths), 5]:
            raise RuntimeError("Existing pool shape metadata is inconsistent")
        if list(metadata.get("parameter_names", [])) != list(PARAMETER_NAMES):
            raise RuntimeError("Existing pool parameter order is inconsistent")
        generated = int(state["generated_truths"])
        if not 0 <= generated <= int(args.num_truths):
            raise RuntimeError("Saved generated_truths is outside the valid range")
        candidates_seen = int(state["candidates_seen"])
        accepted_total = int(state["accepted_total"])
        candidate_batches = int(state["candidate_batches"])
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]
        mode = "r+"
        print(f"继续生成：已完成 {generated:,}/{args.num_truths:,}")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"{output_dir} is not empty. Use --resume or --overwrite explicitly."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        generated = 0
        candidates_seen = 0
        accepted_total = 0
        candidate_batches = 0
        rng = np.random.default_rng(args.seed)
        mode = "w+"
        metadata = {
            "format_version": 1,
            "target_truths": int(args.num_truths),
            "generated_truths": 0,
            "dtype": "float32",
            "shape": [int(args.num_truths), 5],
            "parameter_names": list(PARAMETER_NAMES),
            "parameter_file": "parameters.dat",
            "expected_bytes": int(args.num_truths) * 5 * 4,
            "seed": int(args.seed),
            "physics": DEFAULT_PHYSICS.to_dict(),
            "created_unix_time": time.time(),
            "complete": False,
        }
        atomic_write_json(metadata_path, metadata)

    pool = np.memmap(
        data_path,
        dtype=np.float32,
        mode=mode,
        shape=(int(args.num_truths), 5),
    )

    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_stop)

    def save_state(complete: bool = False) -> None:
        pool.flush()
        state_payload = {
            "generated_truths": int(generated),
            "candidates_seen": int(candidates_seen),
            "accepted_total": int(accepted_total),
            "candidate_batches": int(candidate_batches),
            "acceptance_rate": (
                float(accepted_total / candidates_seen)
                if candidates_seen else 0.0
            ),
            "rng_state": rng.bit_generator.state,
            "updated_unix_time": time.time(),
            "complete": bool(complete),
        }
        atomic_write_json(state_path, state_payload)
        metadata["generated_truths"] = int(generated)
        metadata["complete"] = bool(complete)
        metadata["updated_unix_time"] = time.time()
        metadata["acceptance_rate"] = state_payload["acceptance_rate"]
        atomic_write_json(metadata_path, metadata)

    started = time.perf_counter()
    try:
        while generated < args.num_truths and not STOP_REQUESTED:
            if (
                args.max_candidate_batches > 0
                and candidate_batches >= args.max_candidate_batches
            ):
                print("已达到 --max-candidate-batches，保存后停止。")
                break

            n = int(args.candidate_batch_size)
            candidates = np.empty((n, 5), dtype=np.float64)
            candidates[:, 0] = rng.uniform(
                DEFAULT_PHYSICS.a1_min, DEFAULT_PHYSICS.a1_max, n
            )
            candidates[:, 1] = rng.uniform(
                DEFAULT_PHYSICS.a2_min, DEFAULT_PHYSICS.a2_max, n
            )
            candidates[:, 2] = rng.uniform(
                DEFAULT_PHYSICS.a3_min, DEFAULT_PHYSICS.a3_max, n
            )
            # np.random.uniform is [low, high), so exact zero is already
            # probability-zero. nextafter makes the open boundary explicit.
            positive_zero = np.nextafter(0.0, 1.0)
            candidates[:, 3] = rng.uniform(
                positive_zero, DEFAULT_PHYSICS.m_max, n
            )
            candidates[:, 4] = rng.uniform(
                positive_zero, DEFAULT_PHYSICS.gamma_max, n
            )

            valid = valid_parameter_mask_numpy(candidates)
            accepted = candidates[valid]
            remaining = int(args.num_truths - generated)
            take = min(remaining, len(accepted))
            if take:
                pool[generated:generated + take] = accepted[:take].astype(
                    np.float32, copy=False
                )
                generated += take

            candidates_seen += n
            accepted_total += len(accepted)
            candidate_batches += 1
            save_state(complete=False)

            elapsed = max(time.perf_counter() - started, 1e-9)
            rate = generated / elapsed
            minimum = (
                float(np.min(minimum_rho_on_interval_numpy(accepted[:take])))
                if take else float("nan")
            )
            print(
                f"batch={candidate_batches:,} "
                f"generated={generated:,}/{args.num_truths:,} "
                f"accepted={len(accepted):,}/{n:,} "
                f"overall_acceptance={accepted_total/candidates_seen:.2%} "
                f"write_rate={rate:,.0f} truths/s "
                f"accepted_min_rho={minimum:.3e}"
            )

        complete = generated >= args.num_truths
        save_state(complete=complete)
        if complete:
            print("\n参数池生成完成。")
        else:
            print("\n参数池未完成，进度已保存。使用相同参数并增加 --resume 继续。")
        print(f"文件：{data_path}")
        print(f"已生成：{generated:,}/{args.num_truths:,}")
        print(f"逻辑文件大小：{args.num_truths * 5 * 4 / 1e9:.3f} GB")
    finally:
        pool.flush()
        del pool


if __name__ == "__main__":
    main()
