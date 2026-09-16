from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from cp02_observation import load_config, design_by_id, make_q2
from cp02_core import *


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='cp02_corrected_3p_config.json'); ap.add_argument('--design-id',required=True); ap.add_argument('--output-dir',default='cp02_identifiable_bank'); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    cfg=load_config(args.config); off=cfg['official_bank']; qk=cfg['quick']; out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    n=int(qk['candidate_count'] if args.quick else off['candidate_count']); M=int(qk['profile_m_points'] if args.quick else off['profile_m_points']); G=int(qk['profile_log_gamma_points'] if args.quick else off['profile_log_gamma_points']); premax=int(qk['prebank_max_states'] if args.quick else off['prebank_max_states']); selcount=int(qk['selected_count'] if args.quick else off['selected_count'])
    q2=make_q2(design_by_id(cfg,args.design_id)); np.save(out/'q2.npy',q2)
    params=sample_full_domain(cfg,n,off['candidate_seed']); g=forward(params,q2,cfg['integration_points']); bank=build_profile_bank(cfg,q2,cfg['integration_points'],M,G); gridm,gridrows=grid_alias_batch(params,g,bank,cfg)
    pre=prebank_indices(params,gridm,cfg,off['prebank_bins'],off['prebank_top_per_cell'],premax); pp=params[pre]; pg=g[pre]
    refined=[]; rows=[]
    for k,(orig,p,gg) in enumerate(zip(pre,pp,pg)):
        w=continuous_alias_one(p,gg,bank,cfg,q2,cfg['integration_points'], qk['continuous_top_regions'] if args.quick else off['continuous_top_regions'], qk['continuous_maxiter'] if args.quick else off['continuous_maxiter'], qk['continuous_multistart'] if args.quick else off['continuous_multistart']); refined.append(w['mahalanobis']); rows.append({'candidate_id':int(orig),'a1':p[0],'m':p[3],'gamma':p[4],'grid_mahalanobis':gridm[orig],'continuous_mahalanobis':w['mahalanobis'],'continuous_rms_snr':w['rms_snr'],'trigger':w['region'],'alias_a1':w['a1'],'alias_m':w['m'],'alias_gamma':w['gamma']})
        if (k+1)%100==0 or k+1==len(pp): print(f'  refined prebank: {k+1}/{len(pp)}')
    refined=np.asarray(refined,float); write_csv(out/'cp02_refined_prebank.csv',rows)
    scans=[]; chosen=None
    for th in off['selection_thresholds_mahalanobis']:
        eligible=np.flatnonzero(refined>=float(th)-1e-12); rec={'threshold_mahalanobis':th,'eligible':len(eligible)}
        if len(eligible)>=selcount:
            local=select_states(pp[eligible],pg[eligible],refined[eligible],cfg,selcount,off['min_selected_g_separation_rms_snr'])
            rec['selected']=len(local)
            if len(local)>=selcount:
                local=local[:selcount]
                crmax,cr95,crmed=cover_radius(pp[eligible],pp[eligible][local],cfg)
                rec['identifiable_cover_radius_max']=crmax; rec['cover_radius_p95']=cr95
                cover_ok = crmax <= float(off['identifiable_cover_radius_max']) + 1e-12
                rec['cover_gate_pass']=int(cover_ok)
                if chosen is None and (cover_ok or args.quick):
                    chosen=(float(th),eligible[local])
            else:
                rec['identifiable_cover_radius_max']=math.nan; rec['cover_gate_pass']=0
        else:
            rec['selected']=0; rec['identifiable_cover_radius_max']=math.nan; rec['cover_gate_pass']=0
        scans.append(rec)
    write_csv(out/'cp02_selection_threshold_scan.csv',scans)
    if chosen is None:
        (out/'selection_failure.json').write_text(json.dumps({'reason':'could not satisfy selected_count + g-separation + identifiable-region coverage gate','requested':selcount,'cover_radius_max':off['identifiable_cover_radius_max']},indent=2),encoding='utf-8'); raise SystemExit(3)
    th,ids=chosen; sp=pp[ids]; sg=pg[ids]; sm=refined[ids]; crmax,cr95,crmed=cover_radius(pp,sp,cfg)
    # cover radius is over the continuously refined prebank = empirical identifiable region.
    selected=[]
    for sid,(p,marg) in enumerate(zip(sp,sm)): selected.append({'state_id':sid,'a1':p[0],'a2':p[1],'a3':p[2],'m':p[3],'gamma':p[4],'continuous_mahalanobis':marg})
    write_csv(out/'selected_states.csv',selected)
    # report survival by original-domain bins rather than pretending the exact-zero boundary is recoverable.
    edges_a=np.asarray([0,.01,.025,.05,.10,.15,.20]); edges_m=np.linspace(.01,2,8); edges_g=np.linspace(.01,1,8); surv=[]
    for name,vals,edges in [('a1',pp[:,0],edges_a),('m',pp[:,3],edges_m),('gamma',pp[:,4],edges_g)]:
        svals = sp[:,0] if name=='a1' else sp[:,3] if name=='m' else sp[:,4]
        for i in range(len(edges)-1):
            hi = edges[i+1] if i < len(edges)-2 else edges[i+1] + 1e-12
            mask=(vals>=edges[i])&(vals<hi); smask=(svals>=edges[i])&(svals<hi)
            surv.append({'axis':name,'lo':edges[i],'hi':edges[i+1],'prebank_count':int(mask.sum()),'selected_count':int(smask.sum()),'selection_fraction':float(smask.sum()/max(mask.sum(),1))})
    write_csv(out/'cp02_identifiable_region_coverage.csv',surv)
    summary=[{'design_id':args.design_id,'candidate_count':n,'prebank_count':len(pp),'selected_count':len(sp),'selection_threshold_mahalanobis':th,'selected_margin_min':float(sm.min()),'selected_margin_p10':float(np.quantile(sm,.1)),'selected_margin_median':float(np.median(sm)),'identifiable_region_cover_radius_max':crmax,'cover_radius_p95':cr95,'cover_radius_median':crmed,'selected_a1_min':float(sp[:,0].min()),'selected_a1_max':float(sp[:,0].max()),'selected_m_min':float(sp[:,3].min()),'selected_m_max':float(sp[:,3].max()),'selected_gamma_min':float(sp[:,4].min()),'selected_gamma_max':float(sp[:,4].max())}]
    write_csv(out/'cp02_bank_summary.csv',summary)
    np.savez_compressed(out/'selected_bank.npz',params=sp.astype(np.float64),g_clean=sg.astype(np.float64),q2=q2.astype(np.float64),continuous_mahalanobis=sm.astype(np.float64))
    print(summary[0])

if __name__=='__main__': main()
