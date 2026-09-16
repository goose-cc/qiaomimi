from __future__ import annotations
import argparse, csv, json, math, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

from cp02_observation import load_config
from cp02_core import (
    forward, normalize_params, rms, write_csv, build_profile_bank,
    select_states, cover_radius
)
from analyze_cp02_varpro_ceiling import best_grid, refine


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_refined_csv(path, cfg):
    rows=[]
    with Path(path).open('r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        raise RuntimeError(f'No rows in {path}')
    fixed=cfg['fixed_parameters']
    p=np.array([[float(r['a1']), float(fixed['a2']), float(fixed['a3']), float(r['m']), float(r['gamma'])] for r in rows], dtype=np.float64)
    margin=np.array([float(r['continuous_mahalanobis']) for r in rows], dtype=np.float64)
    cid=np.array([int(r['candidate_id']) for r in rows], dtype=np.int64)
    return rows,p,margin,cid


def coverage_sample_indices(params, n, cfg):
    n=min(int(n), len(params))
    if n<=0:
        return np.empty(0,dtype=int)
    x=normalize_params(params,cfg)
    center=np.full(x.shape[1],0.5)
    first=int(np.argmax(np.linalg.norm(x-center,axis=1)))
    sel=[first]
    mind=np.linalg.norm(x-x[first],axis=1)
    while len(sel)<n:
        mind[sel]=-1.0
        j=int(np.argmax(mind))
        sel.append(j)
        mind=np.minimum(mind,np.linalg.norm(x-x[j],axis=1))
    return np.asarray(sel,dtype=int)


def axis_range(p, idx):
    if len(p)==0: return (math.nan,math.nan)
    return float(np.min(p[:,idx])), float(np.max(p[:,idx]))


def g_separation_min(g, noise_ref):
    if len(g)<2: return math.inf
    sig=noise_ref*np.maximum(np.sqrt(np.mean(g*g,axis=1)),1e-30)
    best=math.inf
    for i in range(len(g)):
        d=np.sqrt(np.mean((g[i+1:]-g[i])**2,axis=1))/np.maximum(sig[i+1:],1e-30)
        if len(d): best=min(best,float(np.min(d)))
    return best


def load_prediction_cache(path):
    cache={}
    path=Path(path)
    if not path.exists(): return cache
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            key=(int(r['candidate_id']),int(r['rep']))
            cache[key]=r
    return cache


def cache_rows(cache):
    return [cache[k] for k in sorted(cache)]


def one_inversion(candidate_id, pt, clean, rep, cfg, bank, q2, noise, base_seed, multistart):
    # deterministic noise by candidate_id + repetition, independent of threshold
    seed=(int(base_seed) + 1000003*int(candidate_id) + 1009*int(rep)) % (2**63-1)
    rng=np.random.default_rng(seed)
    obs=clean + float(noise)*rms(clean)*rng.normal(size=len(clean))
    seeds=best_grid(obs,bank,cfg)
    best=None
    for s in seeds[:int(multistart)]:
        rec=refine(obs,s,cfg,q2,cfg['integration_points'])
        if best is None or rec[0]<best[0]: best=rec
    sd,a,m,g=best
    tol=cfg['recovery_tolerances']
    ae=abs(a-pt[0]); me=abs(m-pt[3]); gf=max(g/pt[4],pt[4]/g)
    oka=ae<=tol['a1_abs']; okm=me<=tol['m_abs']; okg=gf<=tol['gamma_factor']
    return {
        'candidate_id':int(candidate_id),'rep':int(rep),'noise_level':float(noise),
        'true_a1':float(pt[0]),'pred_a1':float(a),'a1_abs_error':float(ae),
        'true_m':float(pt[3]),'pred_m':float(m),'m_abs_error':float(me),
        'true_gamma':float(pt[4]),'pred_gamma':float(g),'gamma_factor_error':float(gf),
        'a1_recovered':int(oka),'m_recovered':int(okm),'gamma_recovered':int(okg),
        'joint_recovered':int(oka and okm and okg)
    }


def ensure_predictions(ids, params, gclean, candidate_ids, reps, cfg, bank, q2, noise, base_seed, multistart, workers, cache, cache_path):
    tasks=[]
    for loc in ids:
        cid=int(candidate_ids[loc])
        for rep in range(int(reps)):
            if (cid,rep) not in cache:
                tasks.append((cid,params[loc].copy(),gclean[loc].copy(),rep))
    if tasks:
        print(f'  VarPro new inversions: {len(tasks)} (workers={workers})')
        def run(t):
            cid,pt,gc,rep=t
            return one_inversion(cid,pt,gc,rep,cfg,bank,q2,noise,base_seed,multistart)
        if int(workers)<=1:
            for j,t in enumerate(tasks,1):
                r=run(t); cache[(int(r['candidate_id']),int(r['rep']))]=r
                if j%25==0 or j==len(tasks): print(f'    inversions {j}/{len(tasks)}')
        else:
            with ThreadPoolExecutor(max_workers=int(workers)) as ex:
                futs=[ex.submit(run,t) for t in tasks]
                done=0
                for fut in as_completed(futs):
                    r=fut.result(); cache[(int(r['candidate_id']),int(r['rep']))]=r; done+=1
                    if done%25==0 or done==len(tasks): print(f'    inversions {done}/{len(tasks)}')
        write_csv(cache_path, cache_rows(cache))
    return cache


def summarize_ids(ids, reps, candidate_ids, cache):
    rows=[]
    for loc in ids:
        cid=int(candidate_ids[loc])
        for rep in range(int(reps)):
            r=cache.get((cid,rep))
            if r is not None: rows.append(r)
    if not rows:
        return dict(samples=0,a1_recovery=math.nan,m_recovery=math.nan,gamma_recovery=math.nan,joint_recovery=math.nan,min_recovery=math.nan)
    A=np.array([[int(r['a1_recovered']),int(r['m_recovered']),int(r['gamma_recovered']),int(r['joint_recovered'])] for r in rows],dtype=float)
    a,m,g,j=A.mean(axis=0)
    return dict(samples=len(rows),a1_recovery=float(a),m_recovery=float(m),gamma_recovery=float(g),joint_recovery=float(j),min_recovery=float(min(a,m,g)))


def bin_rows(threshold, p_all, eligible_mask, selected_mask_global, sweep_cfg):
    out=[]
    specs=[('a1',0),('m',3),('gamma',4)]
    for name,idx in specs:
        edges=np.asarray(sweep_cfg['coverage_bins'][name],dtype=float)
        vals=p_all[:,idx]
        for i in range(len(edges)-1):
            hi=edges[i+1]+(1e-12 if i==len(edges)-2 else 0.0)
            base=(vals>=edges[i])&(vals<hi)
            elig=base&eligible_mask
            sel=base&selected_mask_global
            out.append({
                'threshold_mahalanobis':float(threshold),'axis':name,'lo':float(edges[i]),'hi':float(edges[i+1]),
                'prebank_count':int(base.sum()),'eligible_count':int(elig.sum()),'selected_count':int(sel.sum()),
                'eligible_fraction_of_prebank_bin':float(elig.sum()/max(base.sum(),1)),
                'selected_fraction_of_eligible_bin':float(sel.sum()/max(elig.sum(),1))
            })
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='cp02_corrected_3p_config.json')
    ap.add_argument('--sweep-config',default='cp02_threshold_sweep_config.json')
    ap.add_argument('--bank-dir',default='cp02_identifiable_bank')
    ap.add_argument('--output-dir',default='cp02_threshold_sweep_v2')
    ap.add_argument('--workers',type=int,default=None)
    ap.add_argument('--quick',action='store_true')
    args=ap.parse_args()

    cfg=load_config(args.config); sc=read_json(args.sweep_config); qk=sc.get('quick',{})
    bank_dir=Path(args.bank_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    refined_path=bank_dir/'cp02_refined_prebank.csv'; q2_path=bank_dir/'q2.npy'
    if not refined_path.exists(): raise FileNotFoundError(f'Missing expensive prior result: {refined_path}')
    if not q2_path.exists(): raise FileNotFoundError(f'Missing observation grid: {q2_path}')

    rows,p,margin,cids=read_refined_csv(refined_path,cfg); q2=np.load(q2_path).astype(np.float64)
    print(f'Loaded existing refined prebank: {len(p)} states. No continuous refinement will be rerun.')
    print('Recomputing clean g once for threshold selection...')
    g=forward(p,q2,cfg['integration_points'])

    selected_count=int(qk['selected_count'] if args.quick else sc['selected_count'])
    sweep_n=int(qk['sweep_eval_state_count'] if args.quick else sc['sweep_eval_state_count'])
    sweep_reps=int(qk['sweep_repetitions'] if args.quick else sc['sweep_repetitions'])
    confirm_n=int(qk['confirm_eval_state_count'] if args.quick else sc['confirm_eval_state_count'])
    confirm_reps=int(qk['confirm_repetitions'] if args.quick else sc['confirm_repetitions'])
    M=int(qk['profile_m_points'] if args.quick else sc['profile_m_points']); G=int(qk['profile_log_gamma_points'] if args.quick else sc['profile_log_gamma_points'])
    maxiter=int(qk['continuous_maxiter'] if args.quick else sc['continuous_maxiter']); multistart=int(qk['continuous_multistart'] if args.quick else sc['continuous_multistart'])
    workers=int(args.workers if args.workers is not None else (qk.get('workers',1) if args.quick else sc.get('workers',1)))
    cfg['_runtime_varpro_maxiter']=maxiter
    noise=float(sc['noise_level']); base_seed=int(cfg['varpro']['noise_seed'])+99173
    minsep=float(sc['min_selected_g_separation_rms_snr']); stage=float(sc['stage_gate']); target=float(sc['target_gate'])

    print(f'Building one shared VarPro profile bank MxG={M}x{G}...')
    vbank=build_profile_bank(cfg,q2,cfg['integration_points'],M,G)
    cache_path=out/'cp02_v2_prediction_cache.csv'; cache=load_prediction_cache(cache_path)
    print(f'Resumable inversion cache: {len(cache)} existing rows')

    summaries=[]; coverage=[]; threshold_data={}
    thresholds=sorted(float(x) for x in sc['thresholds_mahalanobis'])
    for th in thresholds:
        elig=np.flatnonzero(margin>=th-1e-12)
        rec={'threshold_mahalanobis':th,'prebank_count':len(p),'eligible_count':len(elig)}
        if len(elig)<selected_count:
            rec.update({'selected_count':0,'selection_feasible':0})
            summaries.append(rec); continue
        local=select_states(p[elig],g[elig],margin[elig],cfg,selected_count,minsep)
        if len(local)<selected_count:
            rec.update({'selected_count':len(local),'selection_feasible':0})
            summaries.append(rec); continue
        local=local[:selected_count]; sel_global=elig[local]
        selmask=np.zeros(len(p),bool); selmask[sel_global]=True; eligmask=np.zeros(len(p),bool); eligmask[elig]=True
        cr_e=cover_radius(p[elig],p[sel_global],cfg); cr_f=cover_radius(p,p[sel_global],cfg)
        ar=axis_range(p[elig],0); mr=axis_range(p[elig],3); gr=axis_range(p[elig],4)
        sr_a=axis_range(p[sel_global],0); sr_m=axis_range(p[sel_global],3); sr_g=axis_range(p[sel_global],4)
        rec.update({
            'selected_count':len(sel_global),'selection_feasible':1,
            'eligible_cover_by_selected_max':cr_e[0],'eligible_cover_by_selected_p95':cr_e[1],
            'full_prebank_cover_by_selected_max':cr_f[0],'full_prebank_cover_by_selected_p95':cr_f[1],
            'selected_g_separation_min_rms_snr':g_separation_min(g[sel_global],cfg['reference_noise']),
            'eligible_a1_min':ar[0],'eligible_a1_max':ar[1],'eligible_m_min':mr[0],'eligible_m_max':mr[1],'eligible_gamma_min':gr[0],'eligible_gamma_max':gr[1],
            'selected_a1_min':sr_a[0],'selected_a1_max':sr_a[1],'selected_m_min':sr_m[0],'selected_m_max':sr_m[1],'selected_gamma_min':sr_g[0],'selected_gamma_max':sr_g[1]
        })
        coverage.extend(bin_rows(th,p,eligmask,selmask,sc))
        audit_local=coverage_sample_indices(p[elig],sweep_n,cfg); audit_global=elig[audit_local]
        ensure_predictions(audit_global,p,g,cids,sweep_reps,cfg,vbank,q2,noise,base_seed,multistart,workers,cache,cache_path)
        vs=summarize_ids(audit_global,sweep_reps,cids,cache)
        rec.update({f'sweep_{k}':v for k,v in vs.items()})
        rec['sweep_stage90_pass']=int(vs['min_recovery']>=stage)
        rec['sweep_target95_pass']=int(vs['min_recovery']>=target)
        summaries.append(rec)
        threshold_data[th]=(elig,sel_global)
        write_csv(out/'cp02_v2_threshold_sweep_summary.csv',summaries)
        write_csv(out/'cp02_v2_region_coverage.csv',coverage)
        print(f"threshold {th:g}: eligible={len(elig)} selected={len(sel_global)} sweep min recovery={vs['min_recovery']:.3f}")

    write_csv(out/'cp02_v2_threshold_sweep_summary.csv',summaries)
    write_csv(out/'cp02_v2_region_coverage.csv',coverage)

    feasible=[r for r in summaries if r.get('selection_feasible')==1]
    if not feasible: raise RuntimeError('No threshold can produce the requested selected bank.')
    # Confirm low thresholds first if sweep >=90%; otherwise at least confirm the highest-margin feasible threshold.
    cand=[r for r in feasible if r.get('sweep_stage90_pass')==1]
    cand=sorted(cand,key=lambda r:r['threshold_mahalanobis'])
    if not cand: cand=[max(feasible,key=lambda r:r['threshold_mahalanobis'])]

    confirms=[]; chosen=None; chosen_status='NO_PHYSICAL_GATE_PASS'
    for r in cand:
        th=float(r['threshold_mahalanobis']); elig,sel_global=threshold_data[th]
        audit_global=elig[coverage_sample_indices(p[elig],confirm_n,cfg)]
        ensure_predictions(audit_global,p,g,cids,confirm_reps,cfg,vbank,q2,noise,base_seed,multistart,workers,cache,cache_path)
        cs=summarize_ids(audit_global,confirm_reps,cids,cache)
        crow={'threshold_mahalanobis':th,**cs,'stage90_pass':int(cs['min_recovery']>=stage),'target95_pass':int(cs['min_recovery']>=target)}
        confirms.append(crow); write_csv(out/'cp02_v2_threshold_confirm_summary.csv',confirms)
        print(f"CONFIRM threshold {th:g}: a1={cs['a1_recovery']:.3f} m={cs['m_recovery']:.3f} gamma={cs['gamma_recovery']:.3f} min={cs['min_recovery']:.3f}")
        if cs['min_recovery']>=target:
            chosen=th; chosen_status='TARGET95_PASS'; break
    if chosen is None:
        pass90=[x for x in confirms if x['stage90_pass']==1]
        if pass90:
            chosen=min(pass90,key=lambda x:x['threshold_mahalanobis'])['threshold_mahalanobis']; chosen_status='STAGE90_PASS_ONLY'
        else:
            chosen=max(feasible,key=lambda r:r['threshold_mahalanobis'])['threshold_mahalanobis']; chosen_status='FALLBACK_HIGHEST_MARGIN'

    chosen=float(chosen); elig,sel_global=threshold_data[chosen]
    sp=p[sel_global]; sg=g[sel_global]; sm=margin[sel_global]
    selected_rows=[]
    for sid,gi in enumerate(sel_global):
        selected_rows.append({'state_id':sid,'candidate_id':int(cids[gi]),'a1':float(p[gi,0]),'a2':float(p[gi,1]),'a3':float(p[gi,2]),'m':float(p[gi,3]),'gamma':float(p[gi,4]),'continuous_mahalanobis':float(margin[gi])})
    write_csv(out/'cp02_v2_selected_states.csv',selected_rows)
    np.savez_compressed(out/'cp02_v2_selected_bank.npz',params=sp.astype(np.float64),g_clean=sg.astype(np.float64),q2=q2.astype(np.float64),continuous_mahalanobis=sm.astype(np.float64),candidate_id=cids[sel_global])

    chosen_sweep=next(r for r in summaries if abs(float(r['threshold_mahalanobis'])-chosen)<1e-12)
    chosen_confirm=next((r for r in confirms if abs(float(r['threshold_mahalanobis'])-chosen)<1e-12),None)
    recommendation={
        'status':chosen_status,'recommended_threshold_mahalanobis':chosen,
        'interpretation':'Threshold applies to the continuously refined prebank. Physical recovery is audited on a coverage sample of the whole eligible region, not only on the hand-selected 260 states.',
        'selected_count':len(sel_global),'eligible_count':len(elig),
        'sweep_summary':chosen_sweep,'confirm_summary':chosen_confirm,
        'source_refined_prebank':str(refined_path),'source_q2':str(q2_path),
        'no_continuous_refinement_rerun':True
    }
    (out/'cp02_v2_recommendation.json').write_text(json.dumps(recommendation,indent=2),encoding='utf-8')
    write_csv(out/'cp02_v2_bank_summary.csv',[{
        'status':chosen_status,'threshold_mahalanobis':chosen,'eligible_count':len(elig),'selected_count':len(sel_global),
        'selected_margin_min':float(sm.min()),'selected_margin_p10':float(np.quantile(sm,.1)),'selected_margin_median':float(np.median(sm)),
        'eligible_cover_by_selected_max':chosen_sweep['eligible_cover_by_selected_max'],
        'full_prebank_cover_by_selected_max':chosen_sweep['full_prebank_cover_by_selected_max'],
        'selected_g_separation_min_rms_snr':chosen_sweep['selected_g_separation_min_rms_snr'],
        'selected_a1_min':float(sp[:,0].min()),'selected_a1_max':float(sp[:,0].max()),
        'selected_m_min':float(sp[:,3].min()),'selected_m_max':float(sp[:,3].max()),
        'selected_gamma_min':float(sp[:,4].min()),'selected_gamma_max':float(sp[:,4].max())
    }])
    print(json.dumps(recommendation,indent=2))

if __name__=='__main__':
    main()
