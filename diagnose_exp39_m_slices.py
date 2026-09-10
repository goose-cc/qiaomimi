#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp39 step A: diagnose fixed-m (a1,gamma) slices under hybrid_near_200."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from exp39_core import (
    load_config, assert_frozen_parameter_domain, current_q2, generate_slice_anchors,
    jacobian_metrics, fixed_m_alias_dense,
)


def write_csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text('',encoding='utf-8');return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def q(x,p):return float(np.quantile(np.asarray(x,float),p))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='exp39_targeted_g_config.json');ap.add_argument('--output-dir',default='exp39_slice_diagnosis');ap.add_argument('--quick',action='store_true');args=ap.parse_args()
    cfg=load_config(args.config);assert_frozen_parameter_domain(cfg);sc=dict(cfg['slice_diagnosis']);
    slices=int(sc['m_slice_count']);anchors=int(sc['anchors_per_slice']);G=int(sc['gamma_grid_points']);worst_n=int(sc['worst_slice_count'])
    if args.quick:slices=5;anchors=8;G=81;worst_n=2
    pr=cfg['parameter_ranges'];mvals=np.linspace(pr['m'][0],pr['m'][1],slices,dtype=np.float64);q2=current_q2(cfg)
    params,sid=generate_slice_anchors(cfg,mvals,anchors,int(sc['anchor_seed']))
    rows=[]
    print('='*100);print('EXP39 FIXED-m SLICE DIAGNOSIS');print('parameter domains are frozen:',cfg['parameter_ranges']);print('current design:',cfg['current_best_design']['id'],'Nq=',len(q2));print('='*100)
    for s,mv in enumerate(mvals):
        ps=params[sid==s];jm=jacobian_metrics(ps,q2,cfg);am=fixed_m_alias_dense(ps,q2,cfg,gamma_points=G);mah=am*math.sqrt(len(q2))
        rows.append({
            'm_slice':float(mv),'anchor_count':len(ps),'q2_points':len(q2),
            'fixed_m_alias_rms_p10':q(am,.1),'fixed_m_alias_rms_median':q(am,.5),
            'fixed_m_alias_mahalanobis_p10':q(mah,.1),'fixed_m_alias_mahalanobis_median':q(mah,.5),
            'agamma_sigma_min_p10':q(jm['sigma_min_agamma'],.1),'agamma_sigma_min_median':q(jm['sigma_min_agamma'],.5),
            'agamma_condition_median':q(jm['condition_agamma'],.5),'agamma_condition_p90':q(jm['condition_agamma'],.9),
            'abs_cos_a1_loggamma_median':q(np.abs(jm['cos_a1_loggamma']),.5),
            'sigma_min_3p_p10':q(jm['sigma_min_3p'],.1),'condition_3p_median':q(jm['condition_3p'],.5),
        })
        print(f"  m={mv:.4f}: fixed-m alias Mah P10={rows[-1]['fixed_m_alias_mahalanobis_p10']:.4g}, |cos(a1,lg)| med={rows[-1]['abs_cos_a1_loggamma_median']:.6f}")
    # Worst slices: alias P10 first (global nonlinear signal), Jacobian P10 as tie-break.
    ranked=sorted(range(len(rows)),key=lambda i:(rows[i]['fixed_m_alias_mahalanobis_p10'],rows[i]['agamma_sigma_min_p10']))
    for rank,i in enumerate(ranked,1): rows[i]['difficulty_rank']=rank;rows[i]['is_target_worst_slice']=int(rank<=worst_n)
    worst=[float(rows[i]['m_slice']) for i in ranked[:worst_n]]
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);write_csv(out/'m_slice_diagnosis.csv',rows)
    payload={'current_design':cfg['current_best_design']['id'],'parameter_ranges_frozen':cfg['parameter_ranges'],'worst_m_slices':worst,'ranking_rule':'ascending fixed-m dense-grid alias Mahalanobis P10; ascending a1-gamma Jacobian sigma_min P10 tie-break','note':'This fixed-m alias scan is a dense-grid diagnosis used to target q2 design. Final decisions use the direct-verified continuous 3P audit.'}
    (out/'worst_m_slices.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print('='*100);print('Worst m slices:',worst);print('Read:',out/'m_slice_diagnosis.csv');print('Read:',out/'worst_m_slices.json');print('='*100)

if __name__=='__main__':main()
