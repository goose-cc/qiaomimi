#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp39 step C: direct-verified continuous 3P audit of one generated q2 design."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from mc_physics import valid_parameter_mask_numpy
from exp38_core import forward64_batched,make_profile_bank,profile_alias_margins,refine_alias_ids
from exp39_core import load_config,assert_frozen_parameter_domain,jacobian_metrics,generate_slice_anchors,fixed_m_alias_dense


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text('',encoding='utf-8');return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def anchors3p(cfg,n,seed):
    pr=cfg['parameter_ranges'];fx=cfg['fixed_parameters'];sam=qmc.Sobol(3,scramble=True,seed=int(seed));u=sam.random_base2(int(math.ceil(math.log2(max(n,2)))))[:n]
    lo,hi=math.log10(pr['gamma'][0]),math.log10(pr['gamma'][1])
    p=np.column_stack([pr['a1'][0]+u[:,0]*(pr['a1'][1]-pr['a1'][0]),np.full(n,fx['a2']),np.full(n,fx['a3']),pr['m'][0]+u[:,1]*(pr['m'][1]-pr['m'][0]),10**(lo+u[:,2]*(hi-lo))]).astype(np.float64)
    if not np.all(valid_parameter_mask_numpy(p)):raise RuntimeError('invalid 3P audit anchor')
    return p


def q(x,p):return float(np.quantile(np.asarray(x,float),p))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='exp39_targeted_g_config.json');ap.add_argument('--design-dir',default='exp39_targeted_designs');ap.add_argument('--design-id',required=True);ap.add_argument('--worst-slices',default='exp39_slice_diagnosis/worst_m_slices.json');ap.add_argument('--output-dir',default='exp39_targeted_audit');ap.add_argument('--quick',action='store_true');args=ap.parse_args()
    cfg=load_config(args.config);assert_frozen_parameter_domain(cfg);da=dict(cfg['deep_audit']);did=args.design_id;qpath=Path(args.design_dir)/f'{did}_q2.npy'
    if not qpath.exists():raise FileNotFoundError(qpath)
    q2=np.load(qpath).astype(np.float64);q2.sort()
    if len(np.unique(q2))!=len(q2) or np.any(q2>=0) or np.any(~np.isfinite(q2)):raise RuntimeError('invalid generated q2 design')
    n=int(da['anchor_count']);M=int(da['profile_grid']['m_points']);G=int(da['profile_grid']['log_gamma_points']);maxiter=int(da['continuous_maxiter']);mult=int(da['continuous_multistart']);sliceA=int(da['slice_alias_anchors_per_worst_slice']);sliceG=int(da['slice_alias_gamma_grid_points'])
    if args.quick:n=8;M=21;G=21;maxiter=8;mult=1;sliceA=4;sliceG=81
    p=anchors3p(cfg,n,int(da['anchor_seed']));clean=forward64_batched(p,q2,int(cfg['physics_integration_points']));mgrid,lggrid,bg,R,r2=make_profile_bank(cfg,q2,int(cfg['physics_integration_points']),m_points=M,log_gamma_points=G)
    print('='*100);print('EXP39 DEEP CONTINUOUS 3P AUDIT:',did);print('q2:',float(q2.min()),float(q2.max()),'N=',len(q2));print('='*100)
    grid=profile_alias_margins(clean,p,cfg,bg,mgrid,lggrid,R,r2,cfg['recovery_tolerances'],chunk_size=64)
    ref=refine_alias_ids(np.arange(n,dtype=int),p,clean,cfg,bg,mgrid,lggrid,R,r2,grid,q2,int(cfg['physics_integration_points']),maxiter=maxiter,multistart=mult,progress_every=max(1,n//4))
    rm=np.asarray([r['recovery_alias_margin_refined'] for r in ref],float);mah=rm*math.sqrt(len(q2));jm=jacobian_metrics(p,q2,cfg)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    thresholds=[]
    for thr in da['mahalanobis_thresholds']:
        mask=mah>=float(thr);thresholds.append({'design_id':did,'mahalanobis_threshold':float(thr),'eligible_states':int(mask.sum()),'eligible_fraction':float(mask.mean())})
    write_csv(out/f'{did}_continuous_margin_scan.csv',thresholds)
    detail=[]
    trigger_counts={}
    for i,r in enumerate(ref):
        trig=str(r['alias_trigger_refined']);trigger_counts[trig]=trigger_counts.get(trig,0)+1
        detail.append({'state_id':i,'a1':p[i,0],'m':p[i,3],'gamma':p[i,4],'q2_points':len(q2),'recovery_alias_margin_rms_snr':rm[i],'recovery_alias_mahalanobis':mah[i],**r})
    write_csv(out/f'{did}_profiled_alias.csv',detail)

    # Re-audit the previously diagnosed worst fixed-m slices for a1-gamma specifically.
    worst=np.asarray(json.loads(Path(args.worst_slices).read_text(encoding='utf-8'))['worst_m_slices'],dtype=np.float64)
    ps,sid=generate_slice_anchors(cfg,worst,sliceA,int(da['anchor_seed'])+101)
    slice_rows=[]
    for s,mv in enumerate(worst):
        pp=ps[sid==s];am=fixed_m_alias_dense(pp,q2,cfg,gamma_points=sliceG);mh=am*math.sqrt(len(q2));jj=jacobian_metrics(pp,q2,cfg)
        slice_rows.append({'design_id':did,'m_slice':float(mv),'anchors':len(pp),'fixed_m_alias_mahalanobis_p10':q(mh,.1),'fixed_m_alias_mahalanobis_median':q(mh,.5),'agamma_sigma_min_p10':q(jj['sigma_min_agamma'],.1),'agamma_condition_median':q(jj['condition_agamma'],.5),'abs_cos_a1_loggamma_median':q(np.abs(jj['cos_a1_loggamma']),.5)})
    write_csv(out/f'{did}_worst_slice_audit.csv',slice_rows)
    summary={
        'design_id':did,'q2_points':len(q2),'q2_min':float(q2.min()),'q2_max':float(q2.max()),'continuous_anchor_count':n,
        'alias_mahalanobis_min':float(mah.min()),'alias_mahalanobis_p10':q(mah,.1),'alias_mahalanobis_median':q(mah,.5),'alias_mahalanobis_p90':q(mah,.9),'fraction_mahalanobis_ge_3p3':float(np.mean(mah>=3.3)),
        'jac_sigma_min_3p_p10':q(jm['sigma_min_3p'],.1),'jac_sigma_min_3p_median':q(jm['sigma_min_3p'],.5),'jac_condition_3p_median':q(jm['condition_3p'],.5),'jac_condition_3p_p90':q(jm['condition_3p'],.9),
        'abs_cos_a1_loggamma_median':q(np.abs(jm['cos_a1_loggamma']),.5),'abs_cos_a1_m_median':q(np.abs(jm['cos_a1_m']),.5),'abs_cos_m_loggamma_median':q(np.abs(jm['cos_m_loggamma']),.5),
        'worst_target_slice_alias_mahalanobis_p10':float(min(r['fixed_m_alias_mahalanobis_p10'] for r in slice_rows)),
        'median_target_slice_alias_mahalanobis_p10':float(np.median([r['fixed_m_alias_mahalanobis_p10'] for r in slice_rows])),
        'worst_target_slice_abs_cos_a1_loggamma_median':float(max(r['abs_cos_a1_loggamma_median'] for r in slice_rows)),
        'trigger_a1_count':int(sum(v for k,v in trigger_counts.items() if k.startswith('a1_'))),
        'trigger_m_count':int(sum(v for k,v in trigger_counts.items() if k.startswith('m_'))),
        'trigger_gamma_count':int(sum(v for k,v in trigger_counts.items() if k.startswith('gamma_'))),
        'target_parameter_recovery_goal':float(cfg['target_parameter_recovery']),
        'note':'This audit measures physical information/aliasing. 95% network recovery is tested only after a winning g design passes dataset construction and a physics ceiling test.'
    }
    write_csv(out/f'{did}_audit_summary.csv',[summary])
    print('continuous alias Mah P10/median:',summary['alias_mahalanobis_p10'],summary['alias_mahalanobis_median']);print('worst target-slice alias Mah P10:',summary['worst_target_slice_alias_mahalanobis_p10']);print('triggers a1/m/gamma:',summary['trigger_a1_count'],summary['trigger_m_count'],summary['trigger_gamma_count']);print('Read:',out/f'{did}_audit_summary.csv');print('='*100)

if __name__=='__main__':main()
