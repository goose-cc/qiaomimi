from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import IterableDataset, get_worker_info
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "online_noise_dataset.py requires PyTorch. Install torch before importing it."
    ) from exc


_U64 = np.uint64
_MASK53_INV = 1.0 / float(1 << 53)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 hash. uint64 overflow is intentional."""
    with np.errstate(over="ignore"):
        z = x + _U64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> _U64(27))) * _U64(0x94D049BB133111EB)
        z = z ^ (z >> _U64(31))
    return z


def stateless_standard_normal(
    truth_ids: np.ndarray,
    noise_id: int,
    observation_length: int,
    master_seed: int,
) -> np.ndarray:
    """
    Generate deterministic N(0,1) values for each
    (master_seed, truth_id, noise_id, observation_index).

    No per-sample RNG object is created, so this remains vectorized.
    """
    truth_ids = np.asarray(truth_ids, dtype=np.uint64)[:, None]
    obs = np.arange(observation_length, dtype=np.uint64)[None, :]
    n = _U64(noise_id)
    seed = _U64(master_seed)

    with np.errstate(over="ignore"):
        counter = (
            seed
            ^ (truth_ids * _U64(0xD2B74407B1CE6E93))
            ^ (n * _U64(0xCA5A826395121157))
            ^ (obs * _U64(0x9E3779B185EBCA87))
        )

    h1 = _splitmix64(counter)
    h2 = _splitmix64(counter ^ _U64(0xA24BAED4963EE407))

    u1 = (((h1 >> _U64(11)).astype(np.float64)) + 0.5) * _MASK53_INV
    u2 = (((h2 >> _U64(11)).astype(np.float64)) + 0.5) * _MASK53_INV

    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return z


class OnlineTruthBatchDataset(IterableDataset):
    """
    Read truth shards and yield already-batched (g_noisy, u, truth_id, noise_id).

    Use DataLoader(..., batch_size=None). The default configuration traverses all
    1000 noise IDs for every truth, so one complete iteration is the logical
    160-billion-sample epoch requested by the specification.

    Shards are partitioned across distributed ranks and DataLoader workers.
    """

    def __init__(
        self,
        root: str | Path,
        batch_size: int = 64,
        noise_level: float = 0.09,
        noises_per_truth: int = 1000,
        master_seed: int = 20260801,
        data_scale: float = 160_000.0,
        shuffle: bool = True,
        epoch: int = 0,
        max_shards: int | None = None,
        max_noise_ids: int | None = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.batch_size = int(batch_size)
        self.noise_level = float(noise_level)
        self.noises_per_truth = int(noises_per_truth)
        self.master_seed = int(master_seed)
        self.data_scale = float(data_scale)
        self.shuffle = bool(shuffle)
        self.epoch = int(epoch)
        self.max_shards = max_shards
        self.max_noise_ids = max_noise_ids
        self.output_dtype = output_dtype

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.noises_per_truth <= 0:
            raise ValueError("noises_per_truth must be positive")
        if not self.root.exists():
            raise FileNotFoundError(self.root)

        self.shards = sorted(self.root.glob("truth_*.npz"))
        if max_shards is not None:
            self.shards = self.shards[: int(max_shards)]
        if not self.shards:
            raise FileNotFoundError(f"No truth_*.npz shards found in {self.root}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def logical_num_samples(self) -> int:
        total_truths = 0
        for path in self.shards:
            with np.load(path, allow_pickle=False) as data:
                total_truths += int(data["truth_id"].shape[0])
        n_noise = self.noises_per_truth
        if self.max_noise_ids is not None:
            n_noise = min(n_noise, int(self.max_noise_ids))
        return total_truths * n_noise

    @property
    def logical_num_batches(self) -> int:
        return math.ceil(self.logical_num_samples / self.batch_size)

    def _rank_info(self) -> Tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _assigned_shards(self) -> list[Path]:
        rank, world_size = self._rank_info()
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers

        global_worker_id = rank * num_workers + worker_id
        total_workers = world_size * num_workers

        shards = list(self.shards)
        if self.shuffle:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.master_seed, self.epoch, 991])
            )
            rng.shuffle(shards)
        return shards[global_worker_id::total_workers]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        shards = self._assigned_shards()
        n_noise = self.noises_per_truth
        if self.max_noise_ids is not None:
            n_noise = min(n_noise, int(self.max_noise_ids))

        noise_ids = np.arange(n_noise, dtype=np.int64)
        if self.shuffle:
            rng_noise = np.random.default_rng(
                np.random.SeedSequence([self.master_seed, self.epoch, 31337])
            )
            rng_noise.shuffle(noise_ids)

        for shard_path in shards:
            with np.load(shard_path, allow_pickle=False) as data:
                u = np.asarray(data["u"])
                g_clean = np.asarray(data["g_clean"])
                truth_id = np.asarray(data["truth_id"], dtype=np.int64)

            if u.shape[0] != g_clean.shape[0] or u.shape[0] != truth_id.shape[0]:
                raise RuntimeError(f"Inconsistent first dimension in {shard_path}")

            order = np.arange(len(truth_id))
            if self.shuffle:
                shard_number = int(shard_path.stem.split("_")[-1])
                rng_truth = np.random.default_rng(
                    np.random.SeedSequence(
                        [self.master_seed, self.epoch, shard_number, 2718]
                    )
                )
                rng_truth.shuffle(order)
                u = u[order]
                g_clean = g_clean[order]
                truth_id = truth_id[order]

            # Noise-major order: each batch normally contains distinct truths.
            for noise_id in noise_ids:
                for start in range(0, len(truth_id), self.batch_size):
                    end = min(len(truth_id), start + self.batch_size)
                    ids_b = truth_id[start:end]
                    clean_b = g_clean[start:end].astype(np.float64, copy=False)
                    u_b = u[start:end].astype(np.float64, copy=False)

                    rms = np.sqrt(np.mean(clean_b * clean_b, axis=1))
                    z = stateless_standard_normal(
                        ids_b,
                        int(noise_id),
                        clean_b.shape[1],
                        self.master_seed,
                    )
                    noisy = clean_b + self.noise_level * rms[:, None] * z

                    noisy *= self.data_scale
                    u_scaled = u_b * self.data_scale

                    yield (
                        torch.as_tensor(noisy, dtype=self.output_dtype),
                        torch.as_tensor(u_scaled, dtype=self.output_dtype),
                        torch.as_tensor(ids_b, dtype=torch.int64),
                        torch.full(
                            (len(ids_b),),
                            int(noise_id),
                            dtype=torch.int64,
                        ),
                    )
