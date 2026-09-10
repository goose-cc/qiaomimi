#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small deterministic regression test for the Exp36-v2 alias numerics."""
from __future__ import annotations
import copy, json, math
from pathlib import Path
import numpy as np

import build_exp36_3p_dataset as b
from exp36_forward64 import forward64_batched
from mc_physics import scaled_forward_observation_numpy


def main():
    cfg=json.loads(Path('exp36_3p_config.json').read_text(encoding='utf-8'))
    small=copy.deepcopy(cfg)
    small['candidate_count']=96
    small['profile_grid']['m_points']=21
    small['profile_grid']['log_gamma_points']=21
    p=b.generate_candidates(small,96,int(small['candidate_seed'])+991)
    integ=int(small['physics_integration_points'])
    g32=scaled_forward_observation_numpy(p,integration_points=integ).astype(np.float64)
    g64=forward64_batched(p,integ)
    rel=float(np.linalg.norm(g64.astype(np.float32).astype(np.float64)-g32)/max(np.linalg.norm(g32),1e-30))
    lim=float(cfg.get('numerics',{}).get('forward64_roundtrip_relL2_max',1e-7))
    if rel>lim:
        raise RuntimeError(f'forward64 roundtrip mismatch: {rel} > {lim}')

    mg,lg,bg,R,r2=b.make_profile_bank(small,integ)
    rec=b.profile_alias_margins(g64,p,small,bg,mg,lg,R,r2,small['recovery_tolerances'],chunk_size=32)
    if not np.isfinite(rec['margin']).all():
        raise RuntimeError('non-finite verified grid margins')
    if np.any(rec['margin']<0):
        raise RuntimeError('negative verified grid margin')
    delta=np.abs(rec['margin_formula']-rec['margin'])
    if float(np.quantile(delta,0.99))>1e-5:
        raise RuntimeError(f'formula/direct profile mismatch too large: p99={np.quantile(delta,.99)}')

    # Refine a small deterministic subset and enforce the key invariant that
    # failed in the old implementation.
    ids=np.array([8,17,31,46,63,79],dtype=np.int64)
    rows=b.refine_selected_continuous(ids,p,g64,small,bg,mg,lg,R,r2,rec,integ)
    for r in rows:
        if r['recovery_alias_margin_refined'] > r['recovery_alias_margin_grid_verified'] + 1e-9:
            raise RuntimeError(f"refined>grid for state {r['state_id']}")
        if not np.isfinite(r['recovery_alias_margin_refined']):
            raise RuntimeError('non-finite refined margin')

    print('EXP36-v2 alias numerics self-check PASS')
    print('forward64 roundtrip relL2:',rel)
    print('grid formula-vs-direct p99 abs margin delta:',float(np.quantile(delta,.99)))
    print('checked refined states:',len(rows))

if __name__=='__main__':
    main()
