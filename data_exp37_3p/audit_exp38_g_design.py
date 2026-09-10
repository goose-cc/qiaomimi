#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Direct continuous recovery-alias audit for one Exp38 g design.

This is the deep check after the coarse scan. It uses a fixed Sobol anchor bank,
continuous direct-verified alias refinement, and reports how many states remain
physically distinguishable at several Mahalanobis separation levels.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from mc_physics import valid_parameter_mask_numpy
from exp38_observation import load_config,design_by_id,make_q2,describe_design
from exp38_core import forward64_batched,make_profile_bank,profile_alias_margins,refine_alias_ids,normalize_free_params,coverage_radius


def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text('',encoding='utf-8'); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def anchors(cfg,n,seed):
    pr=cfg['parameter_ranges'];fx=cfg['fixed_parameters'];sam=qmc.Sobol(3,scramble=True,seed=int(seed));u=sam.random_base2(int(math.ceil(math.log2(max(n,2)))))[:n]
    p=np.column_stack([pr['a1'][0]+u[:,0]*(pr['a1'][1]-pr['a1'][0]),np.full(n,fx['a2']),np.full(n,fx['a3']),pr['m'][0]+u[:,1]*(pr['m'][1]-pr['m'][0]),10**(math.log10(pr['gamma'][0])+u[:,2]*(math.log10(pr['gamma'][1])-math.log10(pr['gamma'][0])))]).astype(float)
    if not np.all(valid_parameter_mask_numpy(p)):raise RuntimeError('invalid anchor')
    return p


def occupied_cells(x,bins):
    b=np.floor(np.clip(x,0,1-1e-14)*bins).astype(int); c=(b[:,0]*bins+b[:,1])*bins+b[:,2]; return len(np.unique(c))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='exp38_g_design_config.json');ap.add_argument('--design-id',required=True);ap.add_argument('--output-dir',default='exp38_design_audit');ap.add_argument('--quick',action='store_true');args=ap.parse_args()
    cfg=load_config(args.config);d=design_by_id(cfg,args.design_id);q2=make_q2(d);sc=dict(cfg['design_scan'])
    n=int(sc['continuous_anchor_count']);M=int(sc['profile_grid']['m_points']);G=int(sc['profile_grid']['log_gamma_points']);maxiter=int(sc['continuous_maxiter']);mult=int(sc['continuous_multistart'])
    if args.quick:n=6;M=21;G=21;maxiter=8;mult=1
    p=anchors(cfg,n,int(sc['anchor_seed'])+17);clean=forward64_batched(p,q2,int(cfg['physics_integration_points']));mgrid,lggrid,bg,R,r2=make_profile_bank(cfg,q2,int(cfg['physics_integration_points']),m_points=M,log_gamma_points=G)
    grid=profile_alias_margins(clean,p,cfg,bg,mgrid,lggrid,R,r2,cfg['recovery_tolerances'],chunk_size=64)
    ids=np.arange(n,dtype=int);ref=refine_alias_ids(ids,p,clean,cfg,bg,mgrid,lggrid,R,r2,grid,q2,int(cfg['physics_integration_points']),maxiter=maxiter,multistart=mult,progress_every=max(1,n//4))
    rm=np.array([r['recovery_alias_margin_refined'] for r in ref],float);mah=rm*math.sqrt(len(q2));pn=normalize_free_params(p,cfg);bins=4
    rows=[]
    for thr in [0.25,0.5,1.0,1.5,2.0,2.5,3.0,3.3,4.0,5.0]:
        mask=mah>=thr; ids2=np.flatnonzero(mask)
        rows.append({'design_id':args.design_id,'mahalanobis_threshold':thr,'eligible_states':len(ids2),'eligible_fraction':float(mask.mean()),'occupied_4x4x4_cells':occupied_cells(pn[ids2],bins) if len(ids2) else 0,'parameter_cover_radius_if_eligible':coverage_radius(pn,pn[ids2]) if len(ids2) else float('inf')})
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);write_csv(out/f'{args.design_id}_continuous_margin_scan.csv',rows)
    detail=[]
    for i,r in enumerate(ref):detail.append({'state_id':i,'a1':p[i,0],'m':p[i,3],'gamma':p[i,4],'q2_points':len(q2),'recovery_alias_margin_rms_snr':rm[i],'recovery_alias_mahalanobis':mah[i],**{k:v for k,v in r.items() if k!='state_id'}})
    write_csv(out/f'{args.design_id}_profiled_alias.csv',detail)
    summary={**describe_design(cfg,d),'continuous_anchor_count':n,'alias_rms_min':float(rm.min()),'alias_rms_p10':float(np.quantile(rm,.1)),'alias_rms_median':float(np.median(rm)),'alias_mahalanobis_min':float(mah.min()),'alias_mahalanobis_p10':float(np.quantile(mah,.1)),'alias_mahalanobis_median':float(np.median(mah)),'fraction_mahalanobis_ge_3p3':float(np.mean(mah>=3.3)),'note':'Mahalanobis 3.3 is only a pairwise Gaussian 95%-separation reference, not a neural-network accuracy guarantee.'}
    write_csv(out/f'{args.design_id}_audit_summary.csv',[summary]);np.save(out/f'{args.design_id}_q2.npy',q2)
    print('='*100);print('EXP38 DEEP DESIGN AUDIT:',args.design_id);print('q2:',q2.min(),q2.max(),'N=',len(q2));print('continuous alias Mahalanobis p10/median:',summary['alias_mahalanobis_p10'],summary['alias_mahalanobis_median']);print('fraction >=3.3:',summary['fraction_mahalanobis_ge_3p3']);print('Read:',out/f'{args.design_id}_audit_summary.csv');print('='*100)

if __name__=='__main__':main()
