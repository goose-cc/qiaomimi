#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp39 step B: automatically build worst-slice-targeted q2 observation designs."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from exp39_core import (
    load_config,assert_frozen_parameter_domain,current_q2,candidate_q2,make_candidate_information,
    greedy_design,design_objective_summary,
)


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text('',encoding='utf-8');return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def load_worst(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8'));return np.asarray(obj['worst_m_slices'],dtype=np.float64)


def nearest_slice_ids(mvals,worst):
    ids=[]
    for x in worst:
        j=int(np.argmin(np.abs(mvals-float(x))))
        if j not in ids:ids.append(j)
    return np.asarray(ids,dtype=int)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='exp39_targeted_g_config.json');ap.add_argument('--worst-slices',default='exp39_slice_diagnosis/worst_m_slices.json');ap.add_argument('--output-dir',default='exp39_targeted_designs');ap.add_argument('--quick',action='store_true');args=ap.parse_args()
    cfg=load_config(args.config);assert_frozen_parameter_domain(cfg);td=dict(cfg['targeted_design']);pr=cfg['parameter_ranges']
    S=int(td['anchor_m_slice_count']);A=int(td['anchors_per_slice']);designs=list(td['designs'])
    if args.quick:S=5;A=5;designs=[{'id':'mixed_targeted_40','kind':'mixed','points':40},{'id':'hybrid_plus_targeted_210','kind':'augment_current','points':210}]
    mvals=np.linspace(pr['m'][0],pr['m'][1],S,dtype=np.float64);worst=load_worst(args.worst_slices);target_ids=nearest_slice_ids(mvals,worst);qpool=candidate_q2(cfg)
    print('='*100);print('EXP39 TARGETED q2 DESIGN');print('parameter domains are frozen:',cfg['parameter_ranges']);print('q2 candidate pool:',len(qpool),qpool.min(),qpool.max());print('target m slices:',mvals[target_ids].tolist());print('='*100)
    B3,B2,_,_=make_candidate_information(cfg,qpool,mvals,A,int(td['anchor_seed']))
    # Current hybrid indices are exact because candidate_q2 explicitly unions them.
    cur=current_q2(cfg);cur_idx=[]
    for x in cur:
        j=int(np.argmin(np.abs(qpool-x)))
        if abs(float(qpool[j]-x))>1e-10:raise RuntimeError('failed to map current design into candidate pool')
        if j not in cur_idx:cur_idx.append(j)
    seed=np.asarray(td['seed_q2'],dtype=np.float64);rows=[];out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    # Always save current champion unchanged as comparator.
    np.save(out/'hybrid_near_200_q2.npy',cur)
    base_summary=design_objective_summary(np.asarray(cur_idx,dtype=int),B3,B2,target_ids)
    rows.append({'design_id':'hybrid_near_200','kind':'exp38_current_best','q2_points':len(cur),'q2_min':float(cur.min()),'q2_max':float(cur.max()),'overlap_with_current_fraction':1.0,**base_summary})
    for d in designs:
        did=str(d['id']);kind=str(d['kind']);n=int(d['points'])
        print(f'  building {did}: kind={kind}, points={n}')
        init=cur_idx if kind=='augment_current' else None
        # For augmentation the current design already supplies broad multi-scale seeds.
        seed_use=np.array([],dtype=float) if kind=='augment_current' else seed
        idx=greedy_design(qpool,B3,B2,n,kind,float(td['robust_slice_quantile']),target_ids,seed_use,float(td['mixed_target_fraction']),initial_indices=init,progress_every=max(10,n//5))
        q=qpool[idx];np.save(out/f'{did}_q2.npy',q)
        # Human-readable q2 list too.
        write_csv(out/f'{did}_q2.csv',[{'point_index':i,'q2':float(x)} for i,x in enumerate(q)])
        sm=design_objective_summary(idx,B3,B2,target_ids)
        overlap=sum(np.min(np.abs(cur[:,None]-q[None,:]),axis=1)<1e-10)/len(cur)
        rows.append({'design_id':did,'kind':kind,'q2_points':len(q),'q2_min':float(q.min()),'q2_max':float(q.max()),'overlap_with_current_fraction':float(overlap),**sm})
    # Rank only as heuristic; final winner comes from continuous 3P audit.
    rows.sort(key=lambda r:(r['fim3_lambda_min_p10_slice'],r['agamma_lambda_min_min_target_slice']),reverse=True)
    for i,r in enumerate(rows,1):r['heuristic_rank']=i
    write_csv(out/'design_build_summary.csv',rows)
    (out/'design_manifest.json').write_text(json.dumps({'parameter_ranges_frozen':cfg['parameter_ranges'],'target_m_slices':[float(x) for x in mvals[target_ids]],'designs':[r['design_id'] for r in rows],'note':'Heuristic rank is not the final choice. Deep direct-verified continuous 3P alias audit is required.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('='*100);print('Read:',out/'design_build_summary.csv');print('Deep-audit the leading designs before selecting g.');print('='*100)

if __name__=='__main__':main()
