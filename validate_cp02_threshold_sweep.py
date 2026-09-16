from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from cp02_observation import load_config
from cp02_core import forward, rms_rows


def read_rows(path):
    with Path(path).open('r',encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='cp02_corrected_3p_config.json'); ap.add_argument('--output-dir',default='cp02_threshold_sweep_v2'); args=ap.parse_args()
    cfg=load_config(args.config); root=Path(args.output_dir)
    need=['cp02_v2_recommendation.json','cp02_v2_threshold_sweep_summary.csv','cp02_v2_region_coverage.csv','cp02_v2_selected_states.csv','cp02_v2_selected_bank.npz','cp02_v2_bank_summary.csv']
    missing=[x for x in need if not (root/x).exists()]
    if missing: raise SystemExit(f'Missing outputs: {missing}')
    rec=json.loads((root/'cp02_v2_recommendation.json').read_text(encoding='utf-8')); th=float(rec['recommended_threshold_mahalanobis'])
    z=np.load(root/'cp02_v2_selected_bank.npz'); p=z['params'].astype(float); g=z['g_clean'].astype(float); q2=z['q2'].astype(float); m=z['continuous_mahalanobis'].astype(float)
    checks=[]
    checks.append(('selected_nonempty',len(p)>0))
    checks.append(('margin_threshold',bool(np.all(m>=th-1e-10))))
    d=cfg['regular_numerical_domain']
    checks.append(('a1_bounds',bool(np.all((p[:,0]>=d['a1'][0])&(p[:,0]<=d['a1'][1])))))
    checks.append(('m_bounds',bool(np.all((p[:,3]>=d['m'][0])&(p[:,3]<=d['m'][1])))))
    checks.append(('gamma_bounds',bool(np.all((p[:,4]>=d['gamma'][0])&(p[:,4]<=d['gamma'][1])))))
    gd=forward(p,q2,cfg['integration_points']); rel=np.linalg.norm(gd-g)/max(np.linalg.norm(g),1e-30)
    checks.append(('direct_forward_roundtrip_relL2_lt_1e-12',rel<1e-12))
    # pairwise selected g separation, normalized by reference noise of the comparison state
    sig=cfg['reference_noise']*np.maximum(rms_rows(g),1e-30); minsep=np.inf
    for i in range(len(g)):
        if i+1<len(g):
            dd=np.sqrt(np.mean((g[i+1:]-g[i])**2,axis=1))/np.maximum(sig[i+1:],1e-30)
            if len(dd): minsep=min(minsep,float(dd.min()))
    checks.append(('selected_g_separation_ge_1p3',minsep>=1.3-1e-8))
    print(f"status={rec['status']} threshold={th} selected={len(p)} direct_relL2={rel:.3e} min_g_sep={minsep:.4f}")
    bad=[name for name,ok in checks if not ok]
    for name,ok in checks: print(('PASS' if ok else 'FAIL'),name)
    if bad: raise SystemExit('Validation failed: '+', '.join(bad))
    print('CP02-v2 validation PASS')

if __name__=='__main__': main()
