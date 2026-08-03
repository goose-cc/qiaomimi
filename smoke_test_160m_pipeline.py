from __future__ import annotations

import numpy as np
import torch

from mc_physics import (
    rho_numpy,
    scaled_curves_numpy,
    scaled_curves_torch,
    valid_parameter_mask_numpy,
)
from mc_pool_config import DEFAULT_PHYSICS


def main() -> None:
    rng = np.random.default_rng(20260802)
    accepted = []
    while sum(len(x) for x in accepted) < 64:
        p = np.empty((256, 5), dtype=np.float64)
        p[:, 0] = rng.uniform(DEFAULT_PHYSICS.a1_min, DEFAULT_PHYSICS.a1_max, len(p))
        p[:, 1] = rng.uniform(DEFAULT_PHYSICS.a2_min, DEFAULT_PHYSICS.a2_max, len(p))
        p[:, 2] = rng.uniform(DEFAULT_PHYSICS.a3_min, DEFAULT_PHYSICS.a3_max, len(p))
        p[:, 3] = rng.uniform(1e-6, DEFAULT_PHYSICS.m_max, len(p))
        p[:, 4] = rng.uniform(1e-6, DEFAULT_PHYSICS.gamma_max, len(p))
        accepted.append(p[valid_parameter_mask_numpy(p)])

    params = np.concatenate(accepted, axis=0)[:64].astype(np.float32)
    assert valid_parameter_mask_numpy(params).all()

    s_dense = np.linspace(DEFAULT_PHYSICS.s_min, DEFAULT_PHYSICS.s_max, 20001)
    dense_min = float(rho_numpy(params, s_dense).min())
    if dense_min < -2e-6:
        raise RuntimeError(f"validity check failed: dense minimum rho={dense_min}")

    f_np, g_np = scaled_curves_numpy(params, integration_points=128)
    f_t, g_t = scaled_curves_torch(
        torch.as_tensor(params, dtype=torch.float64), integration_points=128
    )
    f_t = f_t.cpu().numpy()
    g_t = g_t.cpu().numpy()

    if not np.allclose(f_np, f_t, rtol=2e-5, atol=2e-6):
        raise RuntimeError("NumPy and Torch targets disagree")
    if not np.allclose(g_np, g_t, rtol=2e-5, atol=2e-6):
        raise RuntimeError("NumPy and Torch forward observations disagree")

    print(f"valid rows: {len(params)}")
    print(f"dense minimum rho: {dense_min:.8e}")
    print("NumPy/Torch physics cross-check passed")


if __name__ == "__main__":
    main()
