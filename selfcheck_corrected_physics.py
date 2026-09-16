#!/usr/bin/env python
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

from mc_pool_config import DEFAULT_PHYSICS
from mc_physics import (
    rho_numpy,
    output_grids_numpy,
    scaled_forward_observation_numpy,
    scaled_forward_observation_at_q2_numpy,
)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', default='corrected_physics_reset_config.json')
    args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))

    # Original nominal ranges from the handwritten specification.
    expected={
        'a1':(0.0,0.2),'a2':(0.0,0.05),'a3':(-0.05,0.05),
        'm':(0.0,2.0),'gamma':(0.0,1.0),
    }
    actual={
        'a1':(DEFAULT_PHYSICS.a1_min,DEFAULT_PHYSICS.a1_max),
        'a2':(DEFAULT_PHYSICS.a2_min,DEFAULT_PHYSICS.a2_max),
        'a3':(DEFAULT_PHYSICS.a3_min,DEFAULT_PHYSICS.a3_max),
        'm':(DEFAULT_PHYSICS.m_min_open,DEFAULT_PHYSICS.m_max),
        'gamma':(DEFAULT_PHYSICS.gamma_min_open,DEFAULT_PHYSICS.gamma_max),
    }
    for k in expected:
        if not np.allclose(actual[k],expected[k],rtol=0,atol=1e-15):
            raise AssertionError(f'{k} domain mismatch: {actual[k]} vs {expected[k]}')

    # Peak center must be m^2 for the corrected Lorentzian.
    a1=0.12; m=0.8; gamma=0.2
    p=np.array([[a1,0.0,0.0,m,gamma]],dtype=np.float64)
    center=m*m
    eps=2e-4
    vals=rho_numpy(p,np.array([center-eps,center,center+eps],dtype=np.float64))[0]
    if not (vals[1] >= vals[0] and vals[1] >= vals[2]):
        raise AssertionError(f'corrected peak is not centered at m^2={center}: {vals}')
    expected_peak=(a1/math.pi)/(m*gamma)
    if not np.isclose(vals[1],expected_peak,rtol=1e-10,atol=1e-12):
        raise AssertionError(f'peak-height identity failed: {vals[1]} vs {expected_peak}')

    # Standard q2 forward must be exactly the arbitrary-q2 source of truth.
    _,q2=output_grids_numpy()
    test=np.array([
        [0.12,0.025,0.0,0.8,0.2],
        [0.07,0.025,0.0,1.3,0.05],
    ],dtype=np.float64)
    a=scaled_forward_observation_numpy(test,integration_points=128).astype(np.float64)
    b=scaled_forward_observation_at_q2_numpy(
        test,q2,integration_points=128,output_dtype=np.float64
    )
    rel=np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30)
    # a is intentionally stored float32, so ~1e-8 agreement is expected.
    if rel > 1e-6:
        raise AssertionError(f'forward wrapper mismatch relL2={rel:g}')

    # Formula version must be explicit in metadata.
    meta=DEFAULT_PHYSICS.to_dict()
    if meta.get('physics_formula_version')!='rho-m2-center-v1' or 'm^2' not in meta.get('rho_formula',''):
        raise AssertionError(f'formula metadata not corrected: {meta}')

    print('='*88)
    print('CORRECTED PHYSICS SELF-CHECK PASS')
    print('formula:', cfg['rho_formula'])
    print('original nominal domain:', cfg['original_nominal_domain'])
    print('peak center check: m^2 =',center)
    print('standard-vs-arbitrary q2 forward relL2:',rel)
    print('='*88)

if __name__=='__main__':
    main()
