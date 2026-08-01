from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PARAMETER_NAMES = ("a1", "a2", "a3", "m", "gamma")


@dataclass(frozen=True)
class PhysicsConfig:
    a1_min: float = 0.0
    a1_max: float = 0.2
    a2_min: float = 0.0
    a2_max: float = 0.05
    a3_min: float = -0.05
    a3_max: float = 0.05
    m_min_open: float = 0.0
    m_max: float = 2.0
    gamma_min_open: float = 0.0
    gamma_max: float = 1.0

    s_min: float = 0.1764
    s_max: float = 6.0
    q2_min: float = -100.0
    q2_max: float = -6.0

    output_points: int = 100
    q2_points: int = 100
    data_scale: float = 160000.0
    noise_level: float = 0.09

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["parameter_names"] = list(PARAMETER_NAMES)
        result["rho_formula"] = (
            "a1/pi * (m*gamma) / ((s-m)^2 + (m*gamma)^2) + a2*s + a3"
        )
        result["target_formula"] = "u(s) = rho(s)/(s+400)^2"
        result["training_target"] = "data_scale * u(s)"
        result["training_input"] = "data_scale * g_noisy(q^2)"
        result["noise_definition"] = (
            "g_noisy = g_clean + noise_level*RMS(g_clean)*N(0,1)"
        )
        return result


DEFAULT_PHYSICS = PhysicsConfig()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
