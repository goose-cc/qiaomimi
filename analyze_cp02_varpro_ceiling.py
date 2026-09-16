from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from cp02_observation import load_config
from cp02_core import forward, build_profile_bank, rms, write_csv


def best_grid(obs,bank,cfg):
    y=obs-bank['background']; dot=bank['R']@y; aa=np.clip(dot/bank['r2'],cfg['regular_numerical_domain']['a1'][0],cfg['regular_numerical_domain']['a1'][1]); y2=float(y@y); sse=np.maximum(y2+aa*aa*bank['r2']-2*aa*dot,0); ids=np.argsort(sse)[:12]; return [(float(sse[i]),float(aa[i]),float(bank['mm'][i]),float(bank['gg'][i])) for i in ids]

def refine(obs,seed,cfg,q2,ip):
    d=cfg['regular_numerical_domain']; fixed=cfg['fixed_parameters']
    def ev(x):
        m,lg=x; ga=math.exp(lg); base=np.array([[0,fixed['a2'],fixed['a3'],m,ga]],float); unit=base.copy(); unit[0,0]=1
        bg=forward(base,q2,ip)[0]; R=forward(unit,q2,ip)[0]-bg; y=obs-bg; a=float(np.clip((y@R)/max(R@R,1e-300),d['a1'][0],d['a1'][1])); r=y-a*R; return float(r@r),a,ga
    maxiter=cfg.get('_runtime_varpro_maxiter', cfg['varpro']['continuous_maxiter']); r=minimize(lambda x:ev(x)[0],[seed[2],math.log(seed[3])],bounds=[tuple(d['m']),(math.log(d['gamma'][0]),math.log(d['gamma'][1]))],method='L-BFGS-B',options={'maxiter':int(maxiter),'ftol':1e-15}); sd,a,g=ev(r.x); return sd,a,float(r.x[0]),g

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='cp02_corrected_3p_config.json'); ap.add_argument('--bank-dir',default='cp02_identifiable_bank'); ap.add_argument('--output-dir',default='cp02_varpro'); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    cfg=load_config(args.config); vd=cfg['varpro']; qk=cfg['quick']; cfg['_runtime_varpro_maxiter']=qk['varpro_continuous_maxiter'] if args.quick else vd['continuous_maxiter']; root=Path(args.bank_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); z=np.load(root/'selected_bank.npz'); p=z['params']; gc=z['g_clean']; q2=z['q2']
    ne=int(qk['varpro_eval_state_count'] if args.quick else vd['eval_state_count']); ids=np.linspace(0,len(p)-1,min(ne,len(p)),dtype=int); p=p[ids]; gc=gc[ids]
    M=int(qk['varpro_m_points'] if args.quick else vd['m_points']); G=int(qk['varpro_log_gamma_points'] if args.quick else vd['log_gamma_points']); bank=build_profile_bank(cfg,q2,cfg['integration_points'],M,G); rng=np.random.default_rng(vd['noise_seed']); tol=cfg['recovery_tolerances']; rows=[]; summary=[]
    for noise in vd['noise_levels']:
        reps=int(qk['varpro_repetitions_0p2pct'] if args.quick and abs(noise-.002)<1e-12 else 1 if args.quick else vd['repetitions'][str(noise)])
        oks=[]
        for si,(pt,clean) in enumerate(zip(p,gc)):
            for rix in range(reps):
                obs=clean.copy() if noise==0 else clean+float(noise)*rms(clean)*rng.normal(size=len(clean)); seeds=best_grid(obs,bank,cfg); best=None
                for sd,a,m,g in seeds[:int(qk['varpro_continuous_multistart'] if args.quick else vd['continuous_multistart'])]:
                    rec=refine(obs,(sd,a,m,g),cfg,q2,cfg['integration_points']);
                    if best is None or rec[0]<best[0]: best=rec
                sd,a,m,g=best; ae=abs(a-pt[0]); me=abs(m-pt[3]); gf=max(g/pt[4],pt[4]/g); ok=(ae<=tol['a1_abs'],me<=tol['m_abs'],gf<=tol['gamma_factor']); oks.append(ok); rows.append({'noise_level':noise,'state_index':int(ids[si]),'rep':rix,'true_a1':pt[0],'pred_a1':a,'a1_abs_error':ae,'true_m':pt[3],'pred_m':m,'m_abs_error':me,'true_gamma':pt[4],'pred_gamma':g,'gamma_factor_error':gf,'a1_recovered':int(ok[0]),'m_recovered':int(ok[1]),'gamma_recovered':int(ok[2]),'joint_recovered':int(all(ok))})
        A=np.asarray(oks,bool); summary.append({'noise_level':noise,'samples':len(A),'a1_recovery':float(A[:,0].mean()),'m_recovery':float(A[:,1].mean()),'gamma_recovery':float(A[:,2].mean()),'joint_recovery':float(A.all(axis=1).mean())})
    gate=cfg['stage_gate']; target=next(x for x in summary if abs(x['noise_level']-gate['noise_level'])<1e-12); target['stage_gate_90_pass']=int(target['a1_recovery']>=gate['minimum_a1_recovery'] and target['m_recovery']>=gate['minimum_m_recovery'] and target['gamma_recovery']>=gate['minimum_gamma_recovery']); target['target_95_pass']=int(target['a1_recovery']>=gate['target_a1_recovery'] and target['m_recovery']>=gate['target_m_recovery'] and target['gamma_recovery']>=gate['target_gamma_recovery'])
    write_csv(out/'cp02_varpro_predictions.csv',rows); write_csv(out/'cp02_varpro_summary.csv',summary); print(summary)

if __name__=='__main__': main()
