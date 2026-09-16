from __future__ import annotations
import numpy as np

from cp02_observation import load_config, make_q2
from cp02_core import sample_full_domain, forward
from mc_physics import scaled_forward_observation_at_q2_numpy, rho_numpy
from mc_pool_config import DEFAULT_PHYSICS

cfg = load_config('cp02_corrected_3p_config.json')
assert DEFAULT_PHYSICS.to_dict()['physics_formula_version'] == 'rho-m2-center-v1'

p = sample_full_domain(cfg, 8, 1234)
q = make_q2(cfg['observation_designs'][0])
a = forward(p, q, cfg['integration_points'])
b = np.asarray(
    scaled_forward_observation_at_q2_numpy(
        p, q, integration_points=cfg['integration_points'], output_dtype=np.float64
    ),
    dtype=float,
)
rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
assert rel < 1e-13, rel
assert np.all(p[:, 3] >= 0.01) and np.all(p[:, 4] >= 0.01)

# Independent center sanity check: with background off, rho must peak at s=m^2, not s=m.
p0 = np.array([[0.1, 0.0, 0.0, 0.8, 0.1]], dtype=float)
rr = rho_numpy(p0, np.array([0.64, 0.8]))[0]
assert rr[0] > rr[1] * 2.0, rr

print('CP02 selfcheck PASS; forward relL2=', rel)
print('CP02 corrected m^2-center sanity PASS')
