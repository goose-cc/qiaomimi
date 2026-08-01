from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math


@dataclass(frozen=True)
class DatasetConfig:
    # Full prior ranges from the new specification.
    a1_min: float = 0.0
    a1_max: float = 0.2
    a2_min: float = 0.0
    a2_max: float = 0.05
    a3_min: float = -0.05
    a3_max: float = 0.05
    m_min: float = 0.0
    m_max: float = 2.0
    gamma_min: float = 0.0
    gamma_max: float = 1.0

    # Previously established physical interval and observation grid.
    s_min: float = 0.1764
    s_max: float = 6.0
    q2_min: float = -100.0
    q2_max: float = -6.0

    signal_length: int = 100
    observation_length: int = 100

    # Stable quadrature settings.
    background_quadrature_order: int = 96
    resonance_quadrature_order: int = 96

    # Dataset size requested in the new scheme.
    num_truths: int = 160_000_000
    truths_per_shard: int = 100_000
    noises_per_truth: int = 1000
    noise_level: float = 0.09
    master_seed: int = 20260801

    # Truths are computed in float64. Storage can be float32 or float64.
    storage_dtype: str = "float32"

    # Existing project used a fixed scale near 400^2. Raw values remain on disk;
    # the online dataset applies this scale at training time.
    data_scale: float = 160_000.0

    # Strict non-negativity by default. A small positive tolerance can be
    # supplied explicitly only to absorb floating-point round-off.
    rho_negative_tolerance: float = 0.0

    # Reject values that cannot be represented safely by the chosen storage dtype.
    finite_safety_fraction: float = 0.25

    @property
    def num_shards(self) -> int:
        return math.ceil(self.num_truths / self.truths_per_shard)

    def shard_size(self, shard_index: int) -> int:
        if shard_index < 0 or shard_index >= self.num_shards:
            raise IndexError(f"Invalid shard index: {shard_index}")
        start = shard_index * self.truths_per_shard
        return min(self.truths_per_shard, self.num_truths - start)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["num_shards"] = self.num_shards
        return data

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
