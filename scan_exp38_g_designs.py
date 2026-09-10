#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp38 coarse scan of alternative g(q^2) constructions.

Same physics and same parameter anchors for every design. Only the q^2
observation locations / number of observation points change.

Metrics:
- recovery-tolerance-scaled Jacobian singular values and condition number;
- verified grid recovery-alias margins;
- equivalent Mahalanobis separation sqrt(Nq)*RMS-SNR.

This stage deliberately does not train a neural network.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from mc_physics import valid_parameter_mask_numpy
from exp38_observation import load_config, make_q2, describe_design
from exp38_core import forward64_batched, rms_rows, make_profile_bank, profile_alias_margins


def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text('',encoding='utf-8'); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def generate_anchors(cfg,n,seed):
    pr=cfg['parameter_ranges']; fx=cfg['fixed_parameters']
    sampler=qmc.Sobol(d=3,scramble=True,seed=int(seed)); m2=int(math.ceil(math.log2(max(n,2))))
    u=sampler.random_base2(m2)[:n]
    a1=pr['a1'][0]+u[:,0]*(pr['a1'][1]-pr['a1'][0])
    m=pr['m'][0]+u[:,1]*(pr['m'][1]-pr['m'][0])
    lo,hi=math.log10(pr['gamma'][0]),math.log10(pr['gamma'][1]); g=10**(lo+u[:,2]*(hi-lo))
    p=np.column_stack([a1,np.full(n,fx['a2']),np.full(n,fx['a3']),m,g]).astype(np.float64)
    if not np.all(valid_parameter_mask_numpy(p)): raise RuntimeError('invalid scan anchor generated')
    return p


def jacobian_metrics(params,q2,cfg):
    tol=cfg['recovery_tolerances']; integ=int(cfg['physics_integration_points']); frac=.04
    base=forward64_batched(params,q2,integ); cols=[]
    pp=params.copy(); pm=params.copy(); h=frac*tol['a1_abs']; pp[:,0]+=h; pm[:,0]-=h
    cols.append((forward64_batched(pp,q2,integ)-forward64_batched(pm,q2,integ))/(2*h/tol['a1_abs']))
    pp=params.copy(); pm=params.copy(); h=frac*tol['m_abs']; pp[:,3]+=h; pm[:,3]-=h
    cols.append((forward64_batched(pp,q2,integ)-forward64_batched(pm,q2,integ))/(2*h/tol['m_abs']))
    pp=params.copy(); pm=params.copy(); unit=math.log(tol['gamma_factor']); h=frac*unit; pp[:,4]*=np.exp(h); pm[:,4]*=np.exp(-h)
    cols.append((forward64_batched(pp,q2,integ)-forward64_batched(pm,q2,integ))/(2*h/unit))
    J=np.stack(cols,axis=2); sigma=cfg['reference_noise']*rms_rows(base); J/=np.maximum(sigma[:,None,None],1e-30)
    sv=np.linalg.svd(J,compute_uv=False); cond=sv[:,0]/np.maximum(sv[:,-1],1e-30)
    def cos(a,b): return np.sum(a*b,axis=1)/np.maximum(np.sqrt(np.sum(a*a,axis=1)*np.sum(b*b,axis=1)),1e-30)
    return sv[:,-1],cond,cos(J[:,:,0],J[:,:,1]),cos(J[:,:,0],J[:,:,2]),cos(J[:,:,1],J[:,:,2])


def q(x,p): return float(np.quantile(np.asarray(x,float),p))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='exp38_g_design_config.json'); ap.add_argument('--output-dir',default='exp38_design_scan'); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    cfg=load_config(args.config); sc=dict(cfg['design_scan']); designs=list(cfg['observation_designs'])
    if args.quick:
        sc['anchor_count']=32; sc['profile_grid']={'m_points':21,'log_gamma_points':21}
        wanted={'baseline_linear_100','near_extend_linear_160','hybrid_near_200'}; designs=[d for d in designs if d['id'] in wanted]
    params=generate_anchors(cfg,int(sc['anchor_count']),int(sc['anchor_seed'])); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); rows=[]
    print('='*100); print('EXP38 COARSE G-DESIGN SCAN'); print('same anchors:',len(params),'designs:',len(designs)); print('='*100)
    for i,d in enumerate(designs,1):
        q2=make_q2(d); print(f"[{i}/{len(designs)}] {d['id']} N={len(q2)} q2=[{q2.min():g},{q2.max():g}]")
        clean=forward64_batched(params,q2,int(cfg['physics_integration_points']))
        smin,cond,c12,c13,c23=jacobian_metrics(params,q2,cfg)
        pg=sc['profile_grid']; mgrid,lggrid,bg,R,r2=make_profile_bank(cfg,q2,int(cfg['physics_integration_points']),m_points=int(pg['m_points']),log_gamma_points=int(pg['log_gamma_points']))
        gd=profile_alias_margins(clean,params,cfg,bg,mgrid,lggrid,R,r2,cfg['recovery_tolerances'],chunk_size=96)
        gm=np.asarray(gd['margin'],float); mah=gm*math.sqrt(len(q2))
        rows.append({**describe_design(cfg,d),
            'jac_sigma_min_l2_p10':q(smin,.1),'jac_sigma_min_l2_median':q(smin,.5),
            'jac_condition_median':q(cond,.5),'jac_condition_p90':q(cond,.9),
            'abs_cos_a1_m_median':q(abs(c12),.5),'abs_cos_a1_loggamma_median':q(abs(c13),.5),'abs_cos_m_loggamma_median':q(abs(c23),.5),
            'grid_alias_rms_snr_p10':q(gm,.1),'grid_alias_rms_snr_median':q(gm,.5),
            'grid_alias_mahalanobis_p10':q(mah,.1),'grid_alias_mahalanobis_median':q(mah,.5),
        })
    # rank using local worst-direction information, then nonlinear alias proxy
    rows.sort(key=lambda r:(r['jac_sigma_min_l2_p10'],r['grid_alias_mahalanobis_p10']),reverse=True)
    base=next((r for r in rows if r['design_id']=='baseline_linear_100'),None)
    for rank,r in enumerate(rows,1):
        r['coarse_rank']=rank
        if base:
            r['sigma_min_p10_gain_vs_baseline']=r['jac_sigma_min_l2_p10']/max(base['jac_sigma_min_l2_p10'],1e-30)
            r['alias_mah_p10_gain_vs_baseline']=r['grid_alias_mahalanobis_p10']/max(base['grid_alias_mahalanobis_p10'],1e-30)
    write_csv(out/'g_design_coarse_scan.csv',rows)
    finalists=[r['design_id'] for r in rows[:int(sc['finalist_count'])]]
    if 'baseline_linear_100' not in finalists: finalists.append('baseline_linear_100')
    (out/'coarse_finalists.json').write_text(json.dumps({'finalists':finalists,'note':'Run audit_exp38_g_design.py for each finalist before choosing the official g construction.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('='*100); print('Coarse finalists:',finalists); print('Read:',out/'g_design_coarse_scan.csv'); print('='*100)

if __name__=='__main__': main()
