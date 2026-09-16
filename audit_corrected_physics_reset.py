#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CP01: corrected-formula 2P/3P narrow-vs-full physics audit.

No neural network is trained here.  The experiment answers four questions:
1) How much did correcting (s-m)^2 -> (s-m^2)^2 change g?
2) Does the old 2P a1+gamma problem remain recoverable?
3) What changes when the original parameter domain is restored?
4) Is corrected 3P already identifiable enough before redesigning g again?

All current observations use the baseline configured q^2 grid.  If corrected 3P still
needs more information, observation design is revisited only after this reset audit.
"""
from __future__ import annotations

import argparse, csv, json, math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from mc_pool_config import DEFAULT_PHYSICS
from mc_physics import (
    legendre_rule,
    output_grids_numpy,
    scaled_forward_observation_at_q2_numpy,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def rms(x: np.ndarray) -> float:
    a=np.asarray(x,dtype=np.float64)
    return float(np.sqrt(np.mean(a*a)))


def gamma_factor(a: float, b: float) -> float:
    a=max(float(a),1e-300); b=max(float(b),1e-300)
    return max(a/b,b/a)


def correct_forward(p: np.ndarray, q2: np.ndarray, integration_points: int) -> np.ndarray:
    return scaled_forward_observation_at_q2_numpy(
        np.asarray(p,dtype=np.float64), q2=q2,
        integration_points=integration_points,
        output_dtype=np.float64,
    )


def legacy_wrong_forward(p: np.ndarray, q2: np.ndarray, integration_points: int) -> np.ndarray:
    """Old wrong formula, kept only to quantify the reset; never used to build data."""
    p=np.asarray(p,dtype=np.float64)
    q2=np.asarray(q2,dtype=np.float64).reshape(-1)
    x,w=legendre_rule(int(integration_points)); x=np.asarray(x); w=np.asarray(w)
    c=DEFAULT_PHYSICS
    s_mid=.5*(c.s_max+c.s_min); s_half=.5*(c.s_max-c.s_min)
    sf=s_mid+s_half*x; sw=s_half*w
    bg=(p[:,1:2]*sf[None,:]+p[:,2:3])/(sf[None,:]+c.shift)**2
    result=bg @ (sw[:,None]/(sf[:,None]-q2[None,:]))
    a1=p[:,0:1]; m=p[:,3:4]; width=m*p[:,4:5]
    if np.any(width<=0): raise ValueError('legacy comparison requires positive m,gamma')
    z0=np.arctan((c.s_min-m)/width); z1=np.arctan((c.s_max-m)/width)
    zm=.5*(z0+z1); zh=.5*(z1-z0); z=zm+zh*x[None,:]
    sr=m+width*np.tan(z)
    coef=(a1/np.pi)*zh*w[None,:]/(sr+c.shift)**2
    for st in range(0,len(q2),25):
        sp=min(st+25,len(q2))
        result[:,st:sp]+=np.sum(coef[:,:,None]/(sr[:,:,None]-q2[None,None,st:sp]),axis=1)
    return c.data_scale*result


def case_ranges(cfg: dict, stage: str, domain: str) -> dict[str, tuple[float,float]]:
    floor=cfg['numerical_positive_floor']
    if domain=='development':
        d=cfg['development_domain_legacy']
        return {'a1':tuple(d['a1']), 'm':tuple(d['m']), 'gamma':tuple(d['gamma'])}
    if domain!='full': raise ValueError(domain)
    d=cfg['original_nominal_domain']
    return {
        'a1':tuple(d['a1']),
        'm':(max(float(d['m'][0]),float(floor['m'])),float(d['m'][1])),
        'gamma':(max(float(d['gamma'][0]),float(floor['gamma'])),float(d['gamma'][1])),
    }


def sample_params(cfg: dict, stage: str, domain: str, n: int, seed: int) -> np.ndarray:
    rr=case_ranges(cfg,stage,domain); fixed=cfg['staged_fixed_parameters']
    dim=2 if stage=='2p' else 3
    sob=qmc.Sobol(d=dim, scramble=True, seed=int(seed))
    m2=int(math.ceil(math.log2(max(int(n),1))))
    u=sob.random_base2(m2)[:int(n)]
    p=np.empty((n,5),dtype=np.float64)
    p[:,1]=fixed['a2']; p[:,2]=fixed['a3']
    p[:,0]=rr['a1'][0]+u[:,0]*(rr['a1'][1]-rr['a1'][0])
    if stage=='2p':
        p[:,3]=fixed['m_for_2p']
        p[:,4]=rr['gamma'][0]+u[:,1]*(rr['gamma'][1]-rr['gamma'][0])
    else:
        p[:,3]=rr['m'][0]+u[:,1]*(rr['m'][1]-rr['m'][0])
        p[:,4]=rr['gamma'][0]+u[:,2]*(rr['gamma'][1]-rr['gamma'][0])
    return p


def make_grid(cfg: dict, stage: str, domain: str, quick: bool):
    rr=case_ranges(cfg,stage,domain); inc=cfg['original_grid_increment']; fixed=cfg['staged_fixed_parameters']
    ms=float(inc['m']); gs=float(inc['gamma'])
    if quick:
        ms*=int(cfg['quick']['m_grid_stride']); gs*=int(cfg['quick']['gamma_grid_stride'])
    if stage=='2p':
        mgrid=np.array([fixed['m_for_2p']],dtype=np.float64)
    else:
        mgrid=np.arange(rr['m'][0],rr['m'][1]+.5*ms,ms,dtype=np.float64)
        mgrid=mgrid[mgrid<=rr['m'][1]+1e-12]
        if abs(mgrid[-1]-rr['m'][1])>1e-10: mgrid=np.r_[mgrid,rr['m'][1]]
    ggrid=np.arange(rr['gamma'][0],rr['gamma'][1]+.5*gs,gs,dtype=np.float64)
    ggrid=ggrid[ggrid<=rr['gamma'][1]+1e-12]
    if abs(ggrid[-1]-rr['gamma'][1])>1e-10: ggrid=np.r_[ggrid,rr['gamma'][1]]
    return mgrid,ggrid


def build_profile_bank(cfg, stage, domain, q2, integration_points, quick):
    rr=case_ranges(cfg,stage,domain); fixed=cfg['staged_fixed_parameters']
    mgrid,ggrid=make_grid(cfg,stage,domain,quick)
    mm,gg=np.meshgrid(mgrid,ggrid,indexing='ij')
    bgp=np.array([[0.0,fixed['a2'],fixed['a3'],fixed['m_for_2p'],0.5]],dtype=np.float64)
    background=correct_forward(bgp,q2,integration_points)[0]
    pp=np.column_stack([
        np.ones(mm.size), np.full(mm.size,fixed['a2']), np.full(mm.size,fixed['a3']),
        mm.ravel(),gg.ravel()
    ])
    parts=[]
    for st in range(0,len(pp),512): parts.append(correct_forward(pp[st:st+512],q2,integration_points))
    R=np.concatenate(parts,axis=0)-background[None,:]
    r2=np.einsum('ij,ij->i',R,R)
    return {'mgrid':mgrid,'ggrid':ggrid,'mm':mm.ravel(),'gg':gg.ravel(),'background':background,'R':R,'r2':r2,'a1_bounds':rr['a1']}


def constrained_best(y,R,r2,a1_bounds,mask=None,a_bounds=None):
    amin,amax=a1_bounds if a_bounds is None else a_bounds
    if amax < amin: return None
    idx=np.arange(len(R)) if mask is None else np.flatnonzero(mask)
    if len(idx)==0: return None
    RR=R[idx]; r2s=r2[idx]
    dot=RR@np.asarray(y,dtype=np.float64)
    aa=np.clip(dot/r2s,float(amin),float(amax))
    y2=float(np.dot(y,y)); sse=np.maximum(y2+aa*aa*r2s-2*aa*dot,0.0)
    order=np.argsort(sse)[:min(8,len(sse))]
    best=None
    for j in order:
        resid=y-aa[j]*RR[j]; sd=float(np.dot(resid,resid))
        rec=(sd,int(idx[j]),float(aa[j]))
        if best is None or rec[0]<best[0]: best=rec
    return best


def grid_alias(anchor, gtrue, bank, stage, tol, ref_noise):
    y=gtrue-bank['background']; a=float(anchor[0]); m=float(anchor[3]); ga=float(anchor[4])
    aa=bank['a1_bounds']; mm=bank['mm']; gg=bank['gg']; R=bank['R']; r2=bank['r2']
    candidates=[]
    def add(name,b):
        if b is not None: candidates.append((b[0],name,b[1],b[2]))
    add('a1_low',constrained_best(y,R,r2,aa,a_bounds=(aa[0],min(aa[1],a-tol['a1_abs']))))
    add('a1_high',constrained_best(y,R,r2,aa,a_bounds=(max(aa[0],a+tol['a1_abs']),aa[1])))
    if stage=='3p':
        add('m_low',constrained_best(y,R,r2,aa,mask=mm<=m-tol['m_abs']+1e-12))
        add('m_high',constrained_best(y,R,r2,aa,mask=mm>=m+tol['m_abs']-1e-12))
    add('gamma_low',constrained_best(y,R,r2,aa,mask=gg<=ga/tol['gamma_factor']+1e-12))
    add('gamma_high',constrained_best(y,R,r2,aa,mask=gg>=ga*tol['gamma_factor']-1e-12))
    if not candidates: return {'margin':math.inf,'trigger':'none'}
    sd,name,idx,ahat=min(candidates,key=lambda z:z[0])
    sigma=float(ref_noise)*max(rms(gtrue),1e-30)
    return {'margin':math.sqrt(sd/len(gtrue))/sigma,'trigger':name,'idx':idx,'a1':ahat,
            'm':float(mm[idx]),'gamma':float(gg[idx])}


def unit_R(bank, m, gamma, cfg, q2, ip):
    fixed=cfg['staged_fixed_parameters']
    p=np.array([[1.0,fixed['a2'],fixed['a3'],m,gamma]],dtype=np.float64)
    return correct_forward(p,q2,ip)[0]-bank['background']


def continuous_alias(anchor,gtrue,bank,stage,cfg,domain,q2,ip,gridwinner,tol,ref_noise,maxiter,multistart):
    rr=case_ranges(cfg,stage,domain); y=gtrue-bank['background']; ta=float(anchor[0]); tm=float(anchor[3]); tg=float(anchor[4])
    region=gridwinner['trigger']; full_a=rr['a1']
    if region=='a1_low': ab=(full_a[0],min(full_a[1],ta-tol['a1_abs']))
    elif region=='a1_high': ab=(max(full_a[0],ta+tol['a1_abs']),full_a[1])
    else: ab=full_a
    if ab[1]<ab[0]: return gridwinner
    mb=list(rr['m']); lgb=[math.log(rr['gamma'][0]),math.log(rr['gamma'][1])]
    if stage=='2p': mb=[tm,tm]
    elif region=='m_low': mb[1]=min(mb[1],tm-tol['m_abs'])
    elif region=='m_high': mb[0]=max(mb[0],tm+tol['m_abs'])
    if region=='gamma_low': lgb[1]=min(lgb[1],math.log(tg/tol['gamma_factor']))
    elif region=='gamma_high': lgb[0]=max(lgb[0],math.log(tg*tol['gamma_factor']))
    if mb[1]<mb[0] or lgb[1]<lgb[0]: return gridwinner

    def eval_at(mv,lg):
        gv=math.exp(float(lg)); r=unit_R(bank,float(mv),gv,cfg,q2,ip); r2=float(np.dot(r,r))
        ah=float(np.dot(y,r)/max(r2,1e-300)); ah=float(np.clip(ah,ab[0],ab[1])); res=y-ah*r
        return float(np.dot(res,res)),ah,gv
    if stage=='2p':
        def obj(x): return eval_at(tm,x[0])[0]
        starts=[math.log(gridwinner['gamma']), np.clip(math.log(tg),*lgb), lgb[0],lgb[1]]
        best=None
        for st in starts[:max(1,multistart)]:
            r=minimize(obj,[float(st)],bounds=[tuple(lgb)],method='L-BFGS-B',options={'maxiter':int(maxiter),'ftol':1e-15})
            sd,ah,gv=eval_at(tm,r.x[0]); rec=(sd,ah,tm,gv)
            if best is None or sd<best[0]: best=rec
    else:
        def obj(x): return eval_at(x[0],x[1])[0]
        starts=[
            [gridwinner['m'],math.log(gridwinner['gamma'])],
            [np.clip(tm,*mb),np.clip(math.log(tg),*lgb)],
            [mb[0],lgb[0]],[mb[1],lgb[1]],
        ]
        best=None
        for st in starts[:max(1,multistart)]:
            r=minimize(obj,np.asarray(st,float),bounds=[tuple(mb),tuple(lgb)],method='L-BFGS-B',options={'maxiter':int(maxiter),'ftol':1e-15})
            sd,ah,gv=eval_at(r.x[0],r.x[1]); rec=(sd,ah,float(r.x[0]),gv)
            if best is None or sd<best[0]: best=rec
    sigma=float(ref_noise)*max(rms(gtrue),1e-30); margin=math.sqrt(best[0]/len(gtrue))/sigma
    # Grid winner is always a valid candidate; refinement must never make the margin worse.
    if margin > gridwinner['margin']+1e-8: return gridwinner
    return {'margin':margin,'trigger':region,'a1':best[1],'m':best[2],'gamma':best[3]}


def jacobian_metrics(anchor,stage,cfg,domain,q2,ip,ref_noise,tol):
    p=np.asarray(anchor,dtype=np.float64); g0=correct_forward(p[None,:],q2,ip)[0]; sigma=ref_noise*max(rms(g0),1e-30)
    rr=case_ranges(cfg,stage,domain); cols=[]; names=[]
    # derivatives in recovery-tolerance units
    specs=[('a1',0,tol['a1_abs'],rr['a1'])]
    if stage=='3p': specs.append(('m',3,tol['m_abs'],rr['m']))
    # log-gamma handled separately
    for name,idx,scale,bounds in specs:
        h=max(scale*.05,1e-7); lo,hi=bounds
        pm=p.copy(); pp=p.copy(); pm[idx]=max(lo,p[idx]-h); pp[idx]=min(hi,p[idx]+h)
        den=pp[idx]-pm[idx]
        if den<=0: continue
        col=(correct_forward(pp[None,:],q2,ip)[0]-correct_forward(pm[None,:],q2,ip)[0])/den*scale/sigma
        cols.append(col); names.append(name)
    hlog=.05*math.log(tol['gamma_factor']); lg=math.log(p[4]); lo=math.log(rr['gamma'][0]); hi=math.log(rr['gamma'][1])
    lm=max(lo,lg-hlog); lp=min(hi,lg+hlog); pm=p.copy(); pp=p.copy(); pm[4]=math.exp(lm); pp[4]=math.exp(lp)
    col=(correct_forward(pp[None,:],q2,ip)[0]-correct_forward(pm[None,:],q2,ip)[0])/max(lp-lm,1e-15)*math.log(tol['gamma_factor'])/sigma
    cols.append(col); names.append('loggamma')
    J=np.column_stack(cols); sv=np.linalg.svd(J,compute_uv=False); cond=float(sv[0]/max(sv[-1],1e-15))
    out={'sigma_min':float(sv[-1]),'condition':cond}
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            a=J[:,i]; b=J[:,j]; out[f'abs_cos_{names[i]}_{names[j]}']=abs(float(np.dot(a,b)/(max(np.linalg.norm(a)*np.linalg.norm(b),1e-300))))
    return out


def varpro_predict(obs,bank,stage,cfg,domain,q2,ip,maxiter=50,seed_count=4):
    rr=case_ranges(cfg,stage,domain); y=np.asarray(obs)-bank['background']; R=bank['R']; r2=bank['r2']
    dot=R@y; aa=np.clip(dot/r2,rr['a1'][0],rr['a1'][1]); y2=float(np.dot(y,y)); approx=np.maximum(y2+aa*aa*r2-2*aa*dot,0)
    ids=np.argsort(approx)[:min(max(seed_count,1),len(approx))]
    best=None
    def eval_at(mv,gv):
        r=unit_R(bank,mv,gv,cfg,q2,ip); rr2=float(np.dot(r,r)); ah=float(np.clip(np.dot(y,r)/max(rr2,1e-300),rr['a1'][0],rr['a1'][1])); res=y-ah*r
        return float(np.dot(res,res)),ah
    for idx in ids:
        m0=float(bank['mm'][idx]); g0=float(bank['gg'][idx])
        if stage=='2p':
            tm=cfg['staged_fixed_parameters']['m_for_2p']
            def obj(x): return eval_at(tm,math.exp(x[0]))[0]
            r=minimize(obj,[math.log(g0)],bounds=[(math.log(rr['gamma'][0]),math.log(rr['gamma'][1]))],method='L-BFGS-B',options={'maxiter':int(maxiter),'ftol':1e-15})
            gv=math.exp(r.x[0]); sd,ah=eval_at(tm,gv); rec=(sd,ah,tm,gv)
        else:
            def obj(x): return eval_at(x[0],math.exp(x[1]))[0]
            r=minimize(obj,[m0,math.log(g0)],bounds=[rr['m'],(math.log(rr['gamma'][0]),math.log(rr['gamma'][1]))],method='L-BFGS-B',options={'maxiter':int(maxiter),'ftol':1e-15})
            gv=math.exp(r.x[1]); sd,ah=eval_at(r.x[0],gv); rec=(sd,ah,float(r.x[0]),gv)
        if best is None or rec[0]<best[0]: best=rec
    return best[1],best[2],best[3],best[0]


def qtile(a,q):
    a=np.asarray(a,float); return float(np.quantile(a,q)) if len(a) else math.nan


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='corrected_physics_reset_config.json')
    ap.add_argument('--output-dir',default='corrected_physics_reset_results')
    ap.add_argument('--quick',action='store_true')
    args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8')); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    n=int(cfg['quick']['anchor_count'] if args.quick else cfg['anchor_count'])
    ncont=int(cfg['quick']['continuous_alias_anchor_count'] if args.quick else cfg['continuous_alias_anchor_count'])
    reps=int(cfg['quick']['varpro_noise_repetitions'] if args.quick else cfg['varpro_noise_repetitions'])
    maxiter=int(cfg['quick']['continuous_maxiter'] if args.quick else cfg['continuous_maxiter']); multistart=int(cfg['quick']['continuous_multistart'] if args.quick else cfg['continuous_multistart'])
    ip=int(cfg['integration_points']); tol=cfg['recovery_tolerances']; ref=float(cfg['reference_noise']); _,q2=output_grids_numpy()
    rng=np.random.default_rng(int(cfg['noise_seed']))
    summary=[]; pred_rows=[]; alias_rows=[]; jac_rows=[]; bin_rows=[]

    # Structural endpoint audit: these are mathematical facts of the supplied formula.
    boundary_rows=[
        {'boundary':'a1=0','status':'STRUCTURALLY_UNIDENTIFIABLE','reason':'resonance term is exactly zero, so g is independent of m and gamma when a2/a3 are fixed'},
        {'boundary':'m=0','status':'SINGULAR_OR_DEGENERATE_ENDPOINT','reason':'m*gamma=0 gives zero width; exact endpoint is not a regular Lorentzian state'},
        {'boundary':'gamma=0','status':'SINGULAR_ZERO_WIDTH_ENDPOINT','reason':'zero-width Lorentzian is not a regular finite-width function; numerical audit starts at original first positive step 0.01'},
    ]
    write_csv(out/'boundary_identifiability.csv',boundary_rows)

    for ci,(stage,domain) in enumerate([('2p','development'),('2p','full'),('3p','development'),('3p','full')]):
        print('='*96); print('CASE',stage,domain)
        anchors=sample_params(cfg,stage,domain,n,int(cfg['candidate_seed'])+ci)
        clean=correct_forward(anchors,q2,ip); old=legacy_wrong_forward(anchors,q2,ip)
        delta=np.linalg.norm(clean-old,axis=1)/np.maximum(np.linalg.norm(clean,axis=1),1e-30)
        bank=build_profile_bank(cfg,stage,domain,q2,ip,args.quick)
        gridm=[]; contm=[]
        for i,(p,g) in enumerate(zip(anchors,clean)):
            gw=grid_alias(p,g,bank,stage,tol,ref); gridm.append(gw['margin'])
            if i<ncont:
                cw=continuous_alias(p,g,bank,stage,cfg,domain,q2,ip,gw,tol,ref,maxiter,multistart); contm.append(cw['margin'])
                alias_rows.append({'stage':stage,'domain':domain,'anchor_id':i,'true_a1':p[0],'true_m':p[3],'true_gamma':p[4],
                                   'grid_margin':gw['margin'],'continuous_margin':cw['margin'],'trigger':cw['trigger'],
                                   'alias_a1':cw.get('a1',math.nan),'alias_m':cw.get('m',math.nan),'alias_gamma':cw.get('gamma',math.nan)})
            jm=jacobian_metrics(p,stage,cfg,domain,q2,ip,ref,tol); jac_rows.append({'stage':stage,'domain':domain,'anchor_id':i,'a1':p[0],'m':p[3],'gamma':p[4],**jm})

        # VarPro ceiling on the same anchors, clean + requested reference noise.
        case_preds=[]
        for noise in cfg['noise_levels']:
            nr=1 if float(noise)==0 else reps
            for rep in range(nr):
                if float(noise)==0: obs=clean.copy()
                else:
                    sig=float(noise)*np.sqrt(np.mean(clean*clean,axis=1,keepdims=True)); obs=clean+sig*rng.standard_normal(clean.shape)
                for i,(p,o) in enumerate(zip(anchors,obs)):
                    pa,pm,pg,sse=varpro_predict(o,bank,stage,cfg,domain,q2,ip,maxiter=maxiter,seed_count=max(2,multistart))
                    aok=abs(pa-p[0])<=tol['a1_abs']; gok=gamma_factor(pg,p[4])<=tol['gamma_factor']; mok=True if stage=='2p' else abs(pm-p[3])<=tol['m_abs']
                    row={'stage':stage,'domain':domain,'noise_level':noise,'rep':rep,'anchor_id':i,'true_a1':p[0],'pred_a1':pa,
                         'true_m':p[3],'pred_m':pm,'true_gamma':p[4],'pred_gamma':pg,'a1_abs_error':abs(pa-p[0]),
                         'm_abs_error':abs(pm-p[3]),'gamma_factor_error':gamma_factor(pg,p[4]),'a1_recovered':int(aok),
                         'm_recovered':int(mok),'gamma_recovered':int(gok),'joint_recovered':int(aok and mok and gok),'fit_sse':sse}
                    pred_rows.append(row); case_preds.append(row)
        # Summary by noise
        jcase=[r for r in jac_rows if r['stage']==stage and r['domain']==domain]
        for noise in cfg['noise_levels']:
            rrn=[r for r in case_preds if float(r['noise_level'])==float(noise)]
            summary.append({
                'stage':stage,'domain':domain,'formula_version':cfg['physics_formula_version'],'anchor_count':n,'noise_level':noise,
                'old_vs_correct_g_relL2_median':float(np.median(delta)),
                'grid_alias_margin_p10':qtile(gridm,.1),'grid_alias_margin_median':qtile(gridm,.5),
                'continuous_alias_margin_p10':qtile(contm,.1),'continuous_alias_margin_median':qtile(contm,.5),
                'jac_sigma_min_median':float(np.median([x['sigma_min'] for x in jcase])),
                'jac_condition_median':float(np.median([x['condition'] for x in jcase])),
                'a1_recovery':float(np.mean([x['a1_recovered'] for x in rrn])),
                'm_recovery':float(np.mean([x['m_recovered'] for x in rrn])),
                'gamma_recovery':float(np.mean([x['gamma_recovered'] for x in rrn])),
                'joint_recovery':float(np.mean([x['joint_recovered'] for x in rrn])),
                'a1_mae':float(np.mean([x['a1_abs_error'] for x in rrn])),
                'm_mae':float(np.mean([x['m_abs_error'] for x in rrn])),
                'gamma_factor_p90':qtile([x['gamma_factor_error'] for x in rrn],.9),
            })
        # Full-domain recovery by a1 bins, to expose the a1->0 structural issue rather than hide it.
        if domain=='full':
            edges=[0.0,0.01,0.025,0.05,0.10,0.15,0.2000001]
            for noise in cfg['noise_levels']:
                rrn=[r for r in case_preds if float(r['noise_level'])==float(noise)]
                for lo,hi in zip(edges[:-1],edges[1:]):
                    ss=[r for r in rrn if lo<=r['true_a1']<hi]
                    if ss:
                        bin_rows.append({'stage':stage,'noise_level':noise,'a1_bin_lo':lo,'a1_bin_hi':hi,'count':len(ss),
                                         'a1_recovery':float(np.mean([r['a1_recovered'] for r in ss])),
                                         'm_recovery':float(np.mean([r['m_recovered'] for r in ss])),
                                         'gamma_recovery':float(np.mean([r['gamma_recovered'] for r in ss])),
                                         'joint_recovery':float(np.mean([r['joint_recovered'] for r in ss]))})

    write_csv(out/'corrected_physics_reset_summary.csv',summary)
    write_csv(out/'varpro_predictions.csv',pred_rows)
    write_csv(out/'continuous_alias_audit.csv',alias_rows)
    write_csv(out/'jacobian_per_state.csv',jac_rows)
    write_csv(out/'full_domain_recovery_by_a1.csv',bin_rows)
    meta={'formula_version':cfg['physics_formula_version'],'rho_formula':cfg['rho_formula'],
          'original_nominal_domain':cfg['original_nominal_domain'],'numerical_positive_floor':cfg['numerical_positive_floor'],
          'note':'Old Exp31-39 numeric results are legacy/wrong-formula results and must not be mixed with CP01.'}
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('='*96); print('CP01 CORRECTED-PHYSICS RESET AUDIT COMPLETE'); print('read:',out/'corrected_physics_reset_summary.csv')
    print('read:',out/'full_domain_recovery_by_a1.csv'); print('='*96)

if __name__=='__main__': main()
