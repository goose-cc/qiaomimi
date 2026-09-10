#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from exp38_observation import make_q2, design_by_id
from exp38_core import forward64
from mc_physics import scaled_forward_observation_numpy


def main():
    cfg=json.loads(Path('exp38_g_design_config.json').read_text(encoding='utf-8'))
    d=design_by_id(cfg,'baseline_linear_100');q2=make_q2(d)
    p=np.array([[0.12,0.025,0.0,0.72,0.12],[0.18,0.025,0.0,1.05,0.035]],dtype=np.float64)
    a=forward64(p,q2=q2,integration_points=int(cfg['physics_integration_points']))
    b=scaled_forward_observation_numpy(p,integration_points=int(cfg['physics_integration_points'])).astype(np.float64)
    rel=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
    if rel>2e-7: raise RuntimeError(f'baseline custom-q2 forward mismatch: {rel}')
    for dd in cfg['observation_designs']:
        qq=make_q2(dd);g=forward64(p,q2=qq,integration_points=int(cfg['physics_integration_points']))
        if g.shape!=(2,len(qq)) or not np.isfinite(g).all(): raise RuntimeError(f'invalid forward for {dd["id"]}')
    print('EXP38 self-check PASS')
    print('baseline custom-vs-project relL2:',rel)
    print('design count:',len(cfg['observation_designs']))

if __name__=='__main__':main()
