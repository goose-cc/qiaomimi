from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from cp02_observation import load_config, make_q2, describe
from cp02_core import sample_full_domain, forward, build_profile_bank, grid_alias_batch, continuous_alias_one, jacobian_metrics, write_csv


def qv(a,p): return float(np.quantile(np.asarray(a,float),p))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='cp02_corrected_3p_config.json'); ap.add_argument('--output-dir',default='cp02_design_scan'); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    cfg=load_config(args.config); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); ds=cfg['design_scan']; qk=cfg['quick']
    n=int(qk['anchor_count'] if args.quick else ds['anchor_count']); M=int(qk['profile_m_points'] if args.quick else ds['profile_m_points']); G=int(qk['profile_log_gamma_points'] if args.quick else ds['profile_log_gamma_points']); nc=int(qk['continuous_anchor_count'] if args.quick else ds['continuous_anchor_count'])
    anchors=sample_full_domain(cfg,n,ds['anchor_seed'],stratified_a1=True); rows=[]; deep=[]
    for design in cfg['observation_designs']:
        q2=make_q2(design); g=forward(anchors,q2,cfg['integration_points']); bank=build_profile_bank(cfg,q2,cfg['integration_points'],M,G); margins,_=grid_alias_batch(anchors,g,bank,cfg)
        jrows=[jacobian_metrics(p,q2,cfg,cfg['integration_points']) for p in anchors]
        row=describe(cfg,design); row.update({
            'grid_alias_mah_p10':qv(margins,.1),'grid_alias_mah_median':qv(margins,.5),
            'jac_sigma_min_p10':qv([r['sigma_min'] for r in jrows],.1),'jac_sigma_min_median':qv([r['sigma_min'] for r in jrows],.5),
            'jac_condition_median':qv([r['condition'] for r in jrows],.5),'jac_condition_p90':qv([r['condition'] for r in jrows],.9),
            'abs_cos_a1_gamma_median':qv([abs(r['cos_a1_loggamma']) for r in jrows],.5)
        }); rows.append(row)
    # coarse ranking emphasizes tail alias and sigma_min, penalizes condition.
    base=next(r for r in rows if r['design_id']=='baseline_linear_100')
    for r in rows:
        r['alias_p10_gain_vs_baseline']=r['grid_alias_mah_p10']/max(base['grid_alias_mah_p10'],1e-30); r['sigma_p10_gain_vs_baseline']=r['jac_sigma_min_p10']/max(base['jac_sigma_min_p10'],1e-30)
        r['_score']=math.log(max(r['grid_alias_mah_p10'],1e-12))+0.5*math.log(max(r['jac_sigma_min_p10'],1e-12))-0.15*math.log(max(r['jac_condition_median'],1.0))
    rows.sort(key=lambda z:z['_score'],reverse=True)
    finalists=[r['design_id'] for r in rows[:int(ds['finalist_count'])]]
    if 'baseline_linear_100' not in finalists: finalists.append('baseline_linear_100')
    # continuous audit of finalists on a common subset.
    ids=np.linspace(0,len(anchors)-1,nc,dtype=int)
    for did in finalists:
        design=next(d for d in cfg['observation_designs'] if d['id']==did); q2=make_q2(design); aa=anchors[ids]; gg=forward(aa,q2,cfg['integration_points']); bank=build_profile_bank(cfg,q2,cfg['integration_points'],M,G); vals=[]; tr=[]
        for i,(p,g) in enumerate(zip(aa,gg)):
            w=continuous_alias_one(p,g,bank,cfg,q2,cfg['integration_points'],ds['continuous_top_regions'],ds['continuous_maxiter'],ds['continuous_multistart']); vals.append(w['mahalanobis']); tr.append(w['region']);
            if (i+1)%20==0 or i+1==len(aa): print(f'  continuous {did}: {i+1}/{len(aa)}')
        deep.append({'design_id':did,'continuous_mah_min':float(np.min(vals)),'continuous_mah_p10':qv(vals,.1),'continuous_mah_median':qv(vals,.5),'continuous_mah_p90':qv(vals,.9),'fraction_ge_1':float(np.mean(np.asarray(vals)>=1)),'fraction_ge_2':float(np.mean(np.asarray(vals)>=2)),'fraction_ge_3p3':float(np.mean(np.asarray(vals)>=3.3)),'gamma_trigger_fraction':float(np.mean([x.startswith('gamma') for x in tr])),'m_trigger_fraction':float(np.mean([x.startswith('m_') for x in tr])),'a1_trigger_fraction':float(np.mean([x.startswith('a1_') for x in tr]))})
    deep.sort(key=lambda z:(z['continuous_mah_p10'],z['continuous_mah_median']),reverse=True)
    for r in rows: r.pop('_score',None)
    write_csv(out/'cp02_g_design_coarse.csv',rows); write_csv(out/'cp02_g_design_continuous.csv',deep)
    (out/'cp02_finalists.json').write_text(json.dumps({'finalists':finalists,'recommended_by_continuous':deep[0]['design_id']},indent=2),encoding='utf-8')
    print('recommended:',deep[0]['design_id'])

if __name__=='__main__': main()
