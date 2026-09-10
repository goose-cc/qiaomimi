#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from exp39_core import load_config,assert_frozen_parameter_domain,current_q2,candidate_q2,generate_slice_anchors,jacobian_metrics,fixed_m_alias_dense,make_candidate_information,greedy_design

def main():
    cfg=load_config('exp39_targeted_g_config.json');assert_frozen_parameter_domain(cfg);q=current_q2(cfg);pool=candidate_q2(cfg)
    if np.any(q>=0) or np.any(pool>=0):raise RuntimeError('q2 must stay negative')
    mvals=np.array([0.4,0.8,1.2]);p,sid=generate_slice_anchors(cfg,mvals,3,2026091299);jm=jacobian_metrics(p,q,cfg)
    for k,v in jm.items():
        if not np.all(np.isfinite(v)):raise RuntimeError(f'nonfinite jacobian metric {k}')
    am=fixed_m_alias_dense(p[sid==1],q,cfg,gamma_points=51)
    if not np.all(np.isfinite(am)) or np.any(am<0):raise RuntimeError('invalid fixed-m alias margins')
    small_pool=pool[::max(1,len(pool)//80)][:80];B3,B2,_,_=make_candidate_information(cfg,small_pool,mvals,3,2026091288)
    idx=greedy_design(small_pool,B3,B2,16,'mixed',0.1,np.array([0,2]),np.array([-150,-20,-2,-.2]),0.4,progress_every=16)
    if len(idx)!=16 or len(np.unique(idx))!=16:raise RuntimeError('greedy design failed uniqueness/size check')
    print('EXP39 self-check PASS');print('parameter ranges frozen:',cfg['parameter_ranges']);print('current q2 N/range:',len(q),float(q.min()),float(q.max()));print('candidate q2 N/range:',len(pool),float(pool.min()),float(pool.max()));print('fixed-m alias smoke margins:',am.tolist())
if __name__=='__main__':main()
