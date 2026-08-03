from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from mc_pool_config import DEFAULT_PHYSICS, atomic_write_json


PARAMETER_COLUMNS = 5


def parse_edges(text: str) -> np.ndarray:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) < 2:
        raise ValueError("--width-edges 至少需要两个边界")
    edges = np.asarray(values, dtype=np.float64)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("--width-edges 必须严格递增")
    return edges


def reservoir_update(
    reservoirs: List[np.ndarray],
    seen: np.ndarray,
    rows: np.ndarray,
    bin_ids: np.ndarray,
    per_bin: int,
    rng: np.random.Generator,
) -> None:
    for local_i, bin_id in enumerate(bin_ids):
        b = int(bin_id)
        if b < 0 or b >= len(reservoirs):
            continue
        seen[b] += 1
        current = reservoirs[b]
        if len(current) < per_bin:
            reservoirs[b] = np.concatenate([current, rows[local_i:local_i + 1]], axis=0)
        else:
            j = int(rng.integers(0, seen[b]))
            if j < per_bin:
                reservoirs[b][j] = rows[local_i]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从原参数池无放回抽取按 m*gamma 分桶平衡的小训练池。"
    )
    parser.add_argument("--input-pool-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-bin", type=int, default=25000)
    parser.add_argument(
        "--width-edges",
        default="0,0.1,0.2,0.5,2.000001",
        help="m*gamma 分桶边界，逗号分隔",
    )
    parser.add_argument("--scan-chunk-size", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.per_bin <= 0 or args.scan_chunk_size <= 0:
        raise ValueError("--per-bin 和 --scan-chunk-size 必须大于0")
    edges = parse_edges(args.width_edges)
    bin_count = len(edges) - 1

    input_dir = Path(args.input_pool_dir)
    output_dir = Path(args.output_dir)
    input_path = input_dir / "parameters.dat"
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError("输出目录非空；请换目录或加 --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    row_bytes = PARAMETER_COLUMNS * np.dtype(np.float32).itemsize
    rows_total = input_path.stat().st_size // row_bytes
    pool = np.memmap(input_path, dtype=np.float32, mode="r", shape=(rows_total, 5))

    rng = np.random.default_rng(args.seed)
    reservoirs = [np.empty((0, 5), dtype=np.float32) for _ in range(bin_count)]
    seen = np.zeros(bin_count, dtype=np.int64)

    for start in range(0, rows_total, args.scan_chunk_size):
        stop = min(start + args.scan_chunk_size, rows_total)
        rows = np.asarray(pool[start:stop], dtype=np.float32)
        width = rows[:, 3].astype(np.float64) * rows[:, 4].astype(np.float64)
        bin_ids = np.searchsorted(edges, width, side="right") - 1
        reservoir_update(reservoirs, seen, rows, bin_ids, args.per_bin, rng)
        print("scanned %d/%d" % (stop, rows_total))

    shortages = [i for i, values in enumerate(reservoirs) if len(values) < args.per_bin]
    if shortages:
        detail = ", ".join(
            "bin%d=%d/%d" % (i, len(reservoirs[i]), args.per_bin)
            for i in shortages
        )
        raise RuntimeError("以下分桶样本不足：" + detail)

    balanced = np.concatenate(reservoirs, axis=0)
    order = rng.permutation(len(balanced))
    balanced = balanced[order]
    output_path = output_dir / "parameters.dat"
    mm = np.memmap(output_path, dtype=np.float32, mode="w+", shape=balanced.shape)
    mm[:] = balanced
    mm.flush()
    del mm

    source_metadata = {}
    source_meta_path = input_dir / "metadata.json"
    if source_meta_path.exists():
        try:
            source_metadata = json.loads(source_meta_path.read_text(encoding="utf-8"))
        except Exception:
            source_metadata = {}

    metadata = {
        "complete": True,
        "generated_truths": int(len(balanced)),
        "parameter_names": ["a1", "a2", "a3", "m", "gamma"],
        "physics": DEFAULT_PHYSICS.to_dict(),
        "source_pool": str(input_dir.resolve()),
        "selection": "reservoir sampling without replacement within m*gamma bins",
        "width_edges": edges.tolist(),
        "per_bin": int(args.per_bin),
        "seen_per_bin": seen.tolist(),
        "seed": int(args.seed),
        "does_not_change_physics": True,
        "source_metadata": source_metadata,
    }
    atomic_write_json(output_dir / "metadata.json", metadata)
    print("balanced pool written:", output_dir.resolve())
    print("rows:", len(balanced))
    for i in range(bin_count):
        print("[%g, %g): %d" % (edges[i], edges[i + 1], len(reservoirs[i])))


if __name__ == "__main__":
    main()
