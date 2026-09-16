from __future__ import annotations
import csv, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from mc_physics import scaled_forward_observation_at_q2_numpy

REGIONS = ('a1_low','a1_high','m_low','m_high','gamma_low','gamma_high')


def write_csv(path: str | Path, rows: list[dict]):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8'); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def rms(x):
    a=np.asarray(x,dtype=np.float64)
    return float(np.sqrt(np.mean(a*a)))


def rms_rows(x):
    a=np.asarray(x,dtype=np.float64)
    return np.sqrt(np.mean(a*a,axis=1))


def forward(params,q2,ip):
    return np.asarray(scaled_forward_observation_at_q2_numpy(
        np.asarray(params,dtype=np.float64), np.asarray(q2,dtype=np.float64),
        integration_points=int(ip), output_dtype=np.float64), dtype=np.float64)


def sample_full_domain(cfg,n,seed,stratified_a1=False):
    d=cfg['regular_numerical_domain']; fixed=cfg['fixed_parameters']
    n=int(n)
    if not stratified_a1:
        u=qmc.Sobol(d=3,scramble=True,seed=int(seed)).random_base2(int(math.ceil(math.log2(max(n,2)))))[:n]
        a=d['a1'][0]+u[:,0]*(d['a1'][1]-d['a1'][0])
        m=d['m'][0]+u[:,1]*(d['m'][1]-d['m'][0])
        g=d['gamma'][0]+u[:,2]*(d['gamma'][1]-d['gamma'][0])
    else:
        edges=np.asarray(cfg['design_scan']['a1_strata'],dtype=float)
        per=int(math.ceil(n/(len(edges)-1))); chunks=[]
        for bi in range(len(edges)-1):
            u=qmc.Sobol(d=3,scramble=True,seed=int(seed)+bi).random_base2(int(math.ceil(math.log2(max(per,2)))))[:per]
            aa=edges[bi]+u[:,0]*(edges[bi+1]-edges[bi])
            mm=d['m'][0]+u[:,1]*(d['m'][1]-d['m'][0])
            gg=d['gamma'][0]+u[:,2]*(d['gamma'][1]-d['gamma'][0])
            chunks.append(np.column_stack([aa,mm,gg]))
        x=np.concatenate(chunks,axis=0)[:n]; a,m,g=x[:,0],x[:,1],x[:,2]
    p=np.column_stack([a,np.full(n,fixed['a2']),np.full(n,fixed['a3']),m,g]).astype(np.float64)
    return p


def normalize_params(p,cfg):
    d=cfg['regular_numerical_domain']; p=np.asarray(p,float)
    return np.column_stack([
        (p[:,0]-d['a1'][0])/(d['a1'][1]-d['a1'][0]),
        (p[:,3]-d['m'][0])/(d['m'][1]-d['m'][0]),
        (p[:,4]-d['gamma'][0])/(d['gamma'][1]-d['gamma'][0]),
    ])


def build_profile_bank(cfg,q2,ip,m_points,gamma_points):
    d=cfg['regular_numerical_domain']; fixed=cfg['fixed_parameters']
    mg=np.linspace(d['m'][0],d['m'][1],int(m_points),dtype=np.float64)
    # log grid is deliberately used for search accuracy at small gamma, even though the nominal domain is linear.
    lg=np.linspace(math.log(d['gamma'][0]),math.log(d['gamma'][1]),int(gamma_points),dtype=np.float64)
    mm,ll=np.meshgrid(mg,lg,indexing='ij'); gg=np.exp(ll.ravel())
    bgp=np.array([[0.0,fixed['a2'],fixed['a3'],1.0,0.5]],dtype=np.float64)
    bg=forward(bgp,q2,ip)[0]
    pp=np.column_stack([np.ones(mm.size),np.full(mm.size,fixed['a2']),np.full(mm.size,fixed['a3']),mm.ravel(),gg])
    parts=[]
    for st in range(0,len(pp),512): parts.append(forward(pp[st:st+512],q2,ip))
    R=np.concatenate(parts,axis=0)-bg[None,:]
    r2=np.einsum('ij,ij->i',R,R)
    return {'m_grid':mg,'lg_grid':lg,'mm':mm.ravel(),'gg':gg,'background':bg,'R':R,'r2':r2}


def _best_region(y,bank,true,tol,cfg,region):
    a,m,g=float(true[0]),float(true[3]),float(true[4]); d=cfg['regular_numerical_domain']
    R,r2=bank['R'],bank['r2']; dot=R@y; y2=float(y@y); free=np.clip(dot/r2,d['a1'][0],d['a1'][1])
    sse=np.maximum(y2+free*free*r2-2*free*dot,0.0); mask=np.ones(len(R),dtype=bool); aa=free.copy()
    if region=='a1_low':
        hi=a-float(tol['a1_abs'])
        if hi<d['a1'][0]: return None
        aa=np.clip(dot/r2,d['a1'][0],min(hi,d['a1'][1])); sse=np.maximum(y2+aa*aa*r2-2*aa*dot,0.0)
    elif region=='a1_high':
        lo=a+float(tol['a1_abs'])
        if lo>d['a1'][1]: return None
        aa=np.clip(dot/r2,max(lo,d['a1'][0]),d['a1'][1]); sse=np.maximum(y2+aa*aa*r2-2*aa*dot,0.0)
    elif region=='m_low': mask=bank['mm']<=m-float(tol['m_abs'])+1e-12
    elif region=='m_high': mask=bank['mm']>=m+float(tol['m_abs'])-1e-12
    elif region=='gamma_low': mask=bank['gg']<=g/float(tol['gamma_factor'])+1e-12
    elif region=='gamma_high': mask=bank['gg']>=g*float(tol['gamma_factor'])-1e-12
    if not mask.any(): return None
    idxs=np.flatnonzero(mask); j=int(idxs[np.argmin(sse[idxs])]);
    resid=y-aa[j]*R[j]; sd=float(resid@resid)
    return {'region':region,'sse':sd,'a1':float(aa[j]),'m':float(bank['mm'][j]),'gamma':float(bank['gg'][j]),'idx':j}


def grid_alias_one(p,gclean,bank,cfg):
    tol=cfg['recovery_tolerances']; y=np.asarray(gclean,float)-bank['background']; wins=[]
    for reg in REGIONS:
        w=_best_region(y,bank,p,tol,cfg,reg)
        if w is not None: wins.append(w)
    wins=sorted(wins,key=lambda z:z['sse'])
    sig=float(cfg['reference_noise'])*max(rms(gclean),1e-30); nq=len(gclean)
    for w in wins:
        w['rms_snr']=math.sqrt(max(w['sse'],0)/nq)/sig
        w['mahalanobis']=math.sqrt(max(w['sse'],0))/sig
    return wins


def grid_alias_batch(params,gclean,bank,cfg):
    rows=[]
    margin=[]
    for i,(p,g) in enumerate(zip(params,gclean)):
        wins=grid_alias_one(p,g,bank,cfg); w=wins[0]
        margin.append(w['mahalanobis'])
        rows.append({'index':i,'grid_trigger':w['region'],'grid_alias_a1':w['a1'],'grid_alias_m':w['m'],'grid_alias_gamma':w['gamma'],'grid_alias_rms_snr':w['rms_snr'],'grid_alias_mahalanobis':w['mahalanobis']})
        if i==0 or (i+1)%5000==0 or i+1==len(params): print(f'  grid aliases: {i+1}/{len(params)}')
    return np.asarray(margin),rows


def _region_bounds(p,region,cfg):
    d=cfg['regular_numerical_domain']; tol=cfg['recovery_tolerances']; a,m,g=float(p[0]),float(p[3]),float(p[4])
    ab=[d['a1'][0],d['a1'][1]]; mb=[d['m'][0],d['m'][1]]; lb=[math.log(d['gamma'][0]),math.log(d['gamma'][1])]
    if region=='a1_low': ab[1]=min(ab[1],a-tol['a1_abs'])
    elif region=='a1_high': ab[0]=max(ab[0],a+tol['a1_abs'])
    elif region=='m_low': mb[1]=min(mb[1],m-tol['m_abs'])
    elif region=='m_high': mb[0]=max(mb[0],m+tol['m_abs'])
    elif region=='gamma_low': lb[1]=min(lb[1],math.log(g/tol['gamma_factor']))
    elif region=='gamma_high': lb[0]=max(lb[0],math.log(g*tol['gamma_factor']))
    if ab[1]<ab[0] or mb[1]<mb[0] or lb[1]<lb[0]: return None
    return ab,mb,lb


def refine_region(p,gclean,gridwin,region,cfg,q2,ip,maxiter,multistart):
    b=_region_bounds(p,region,cfg)
    if b is None: return None
    ab,mb,lb=b; fixed=cfg['fixed_parameters']; target=np.asarray(gclean,float)
    def evalx(x):
        mv,lg=x; gv=math.exp(float(lg))
        unit=np.array([[1.0,fixed['a2'],fixed['a3'],mv,gv]],dtype=float)
        base=np.array([[0.0,fixed['a2'],fixed['a3'],mv,gv]],dtype=float)
        R=forward(unit,q2,ip)[0]-forward(base,q2,ip)[0]; bg=forward(base,q2,ip)[0]
        y=target-bg; r2=float(R@R); aa=float(np.clip((y@R)/max(r2,1e-300),ab[0],ab[1])); res=y-aa*R
        return float(res@res),aa,gv
    starts=[[gridwin['m'],math.log(gridwin['gamma'])],[np.clip(p[3],*mb),np.clip(math.log(p[4]),*lb)],[mb[0],lb[0]],[mb[1],lb[1]],[mb[0],lb[1]],[mb[1],lb[0]]]
    best=None
    for st in starts[:max(1,int(multistart))]:
        r=minimize(lambda x:evalx(x)[0],np.asarray(st,float),bounds=[tuple(mb),tuple(lb)],method='L-BFGS-B',options={'maxiter':int(maxiter),'ftol':1e-15})
        sd,aa,gv=evalx(r.x); rec={'region':region,'sse':sd,'a1':aa,'m':float(r.x[0]),'gamma':gv}
        if best is None or sd<best['sse']: best=rec
    return best


def continuous_alias_one(p,gclean,bank,cfg,q2,ip,top_regions,maxiter,multistart):
    wins=grid_alias_one(p,gclean,bank,cfg); cand=[]
    for w in wins[:int(top_regions)]:
        r=refine_region(p,gclean,w,w['region'],cfg,q2,ip,maxiter,multistart)
        if r is not None: cand.append(r)
        cand.append({k:w[k] for k in ['region','sse','a1','m','gamma']})
    best=min(cand,key=lambda z:z['sse']); sig=float(cfg['reference_noise'])*max(rms(gclean),1e-30)
    best['rms_snr']=math.sqrt(best['sse']/len(gclean))/sig; best['mahalanobis']=math.sqrt(best['sse'])/sig
    return best


def jacobian_metrics(p,q2,cfg,ip):
    p=np.asarray(p,float); tol=cfg['recovery_tolerances']; d=cfg['regular_numerical_domain']; g0=forward(p[None,:],q2,ip)[0]; sig=cfg['reference_noise']*max(rms(g0),1e-30)
    cols=[]
    for idx,scale,bounds in [(0,tol['a1_abs'],d['a1']),(3,tol['m_abs'],d['m'])]:
        h=max(scale*.05,1e-7); pm=p.copy(); pp=p.copy(); pm[idx]=max(bounds[0],p[idx]-h); pp[idx]=min(bounds[1],p[idx]+h); den=pp[idx]-pm[idx]
        cols.append((forward(pp[None,:],q2,ip)[0]-forward(pm[None,:],q2,ip)[0])/den*scale/sig)
    l0=math.log(p[4]); h=.05*math.log(tol['gamma_factor']); lm=max(math.log(d['gamma'][0]),l0-h); lp=min(math.log(d['gamma'][1]),l0+h); pm=p.copy(); pp=p.copy(); pm[4]=math.exp(lm); pp[4]=math.exp(lp)
    cols.append((forward(pp[None,:],q2,ip)[0]-forward(pm[None,:],q2,ip)[0])/(lp-lm)*math.log(tol['gamma_factor'])/sig)
    J=np.column_stack(cols); sv=np.linalg.svd(J,compute_uv=False); n=np.linalg.norm(J,axis=0)
    return {'sigma_min':float(sv[-1]),'condition':float(sv[0]/max(sv[-1],1e-15)),'cos_a1_m':float((J[:,0]@J[:,1])/max(n[0]*n[1],1e-30)),'cos_a1_loggamma':float((J[:,0]@J[:,2])/max(n[0]*n[2],1e-30)),'cos_m_loggamma':float((J[:,1]@J[:,2])/max(n[1]*n[2],1e-30))}


def make_bins(p,cfg,bins):
    x=normalize_params(p,cfg); b=np.clip((x*np.asarray(bins)[None,:]).astype(int),0,np.asarray(bins)-1)
    return b[:,0]*(bins[1]*bins[2])+b[:,1]*bins[2]+b[:,2]


def prebank_indices(params,score,cfg,bins,top_per_cell,max_states):
    cell=make_bins(params,cfg,bins); ids=[]
    for c in np.unique(cell):
        ii=np.flatnonzero(cell==c); ii=ii[np.argsort(score[ii])[::-1][:int(top_per_cell)]]; ids.extend(ii.tolist())
    ids=np.asarray(sorted(set(ids)),dtype=int)
    if len(ids)>int(max_states): ids=ids[np.argsort(score[ids])[::-1][:int(max_states)]]
    return ids


def select_states(params,gclean,margin,cfg,count,min_g_sep):
    # coverage-first farthest-point selection in the identifiable prebank.
    x=normalize_params(params,cfg); order=np.argsort(margin)[::-1]; sig=cfg['reference_noise']*np.maximum(rms_rows(gclean),1e-30)
    selected=[]; minp=np.full(len(params),np.inf); ming=np.full(len(params),np.inf)
    first=int(order[0]); selected.append(first)
    while len(selected)<int(count):
        last=selected[-1]
        dp=np.linalg.norm(x-x[last],axis=1); minp=np.minimum(minp,dp)
        dg=np.sqrt(np.mean((gclean-gclean[last])**2,axis=1))/np.maximum(sig,1e-30); ming=np.minimum(ming,dg)
        valid=np.ones(len(params),bool); valid[selected]=False; valid &= ming>=float(min_g_sep)-1e-12
        if not valid.any(): break
        # prefer coverage, then margin as tie-breaker.
        utility=minp + 0.05*(margin/ max(np.nanmax(margin),1e-12)); utility[~valid]=-np.inf
        selected.append(int(np.argmax(utility)))
    return np.asarray(selected,dtype=int)


def cover_radius(source,selected,cfg):
    A=normalize_params(source,cfg); B=normalize_params(selected,cfg); best=np.full(len(A),np.inf)
    for st in range(0,len(B),256):
        d=np.linalg.norm(A[:,None,:]-B[None,st:st+256,:],axis=2); best=np.minimum(best,d.min(axis=1))
    return float(best.max()),float(np.quantile(best,.95)),float(np.median(best))
