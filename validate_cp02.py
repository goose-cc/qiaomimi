from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from cp02_observation import load_config
from cp02_core import forward, write_csv


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='cp02_corrected_3p_config.json'); ap.add_argument('--bank-dir',default='cp02_identifiable_bank'); args=ap.parse_args(); cfg=load_config(args.config); root=Path(args.bank_dir); z=np.load(root/'selected_bank.npz'); p=z['params']; g=z['g_clean']; q=z['q2']; checks=[]; errs=[]
    d=cfg['regular_numerical_domain']; ok=(np.isfinite(p).all() and np.isfinite(g).all() and np.isfinite(q).all()); checks.append({'check':'finite_arrays','status':'PASS' if ok else 'FAIL'}); errs += [] if ok else ['nonfinite']
    bounds=((p[:,0]>=d['a1'][0])&(p[:,0]<=d['a1'][1])&(p[:,3]>=d['m'][0])&(p[:,3]<=d['m'][1])&(p[:,4]>=d['gamma'][0])&(p[:,4]<=d['gamma'][1])).all(); checks.append({'check':'full_domain_bounds','status':'PASS' if bounds else 'FAIL'}); errs += [] if bounds else ['bounds']
    rec=forward(p,q,cfg['integration_points']); rel=float(np.linalg.norm(rec-g)/max(np.linalg.norm(g),1e-30)); checks.append({'check':'direct_forward_relL2','status':'PASS' if rel<1e-11 else 'FAIL','value':rel}); errs += [] if rel<1e-11 else ['forward']
    checks.append({'check':'q2_euclidean_negative','status':'PASS' if np.all(q<0) else 'FAIL','value':float(q.max())}); errs += [] if np.all(q<0) else ['q2']
    write_csv(root/'cp02_validation.csv',checks); (root/'cp02_validation.json').write_text(json.dumps({'pass':not errs,'errors':errs},indent=2),encoding='utf-8'); print('CP02 validation', 'PASS' if not errs else 'FAIL', errs); raise SystemExit(0 if not errs else 2)
if __name__=='__main__': main()
