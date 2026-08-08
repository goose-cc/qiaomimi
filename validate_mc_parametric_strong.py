#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np
import torch
from TransformerInverse import PeakParametricInverseTransformer1D
from mc_online_physics import OnlinePhysics
from mc_parametric import normalize_parameters
from mc_parametric_strong import transformed_parameter_coordinates,width_metrics,resonance_peak_height
from train_mc_parameter_pool_transformer_loss import choose_device,open_parameter_pool

PARAMETER_NAMES=('a1','a2','a3','m','gamma')

def parse_args():
    p=argparse.ArgumentParser(description='Validate Exp8 strong resonance-focused parameter model')
    p.add_argument('--validation-pool-dir',required=True); p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--weights',choices=('best','latest'),default='latest'); p.add_argument('--num-samples',type=int,default=10000)
    p.add_argument('--batch-size',type=int,default=64); p.add_argument('--noise-level',type=float,default=0.0); p.add_argument('--seed',type=int,default=20260802)
    p.add_argument('--output-dir',required=True); p.add_argument('--plot-count',type=int,default=18); p.add_argument('--device',choices=('auto','cpu','cuda','xpu'),default='auto')
    return p.parse_args()

def stat(x):
    x=np.asarray(x,dtype=np.float64); return {'mean':float(np.mean(x)),'median':float(np.median(x)),'p90':float(np.quantile(x,.9))}

def rel_l2(p,t,eps=1e-12): return torch.linalg.vector_norm(p-t,dim=1)/torch.linalg.vector_norm(t,dim=1).clamp_min(eps)

def load_latest_args(d):
    try: ck=torch.load(d/'latest_checkpoint.pth',map_location='cpu',weights_only=False)
    except TypeError: ck=torch.load(d/'latest_checkpoint.pth',map_location='cpu')
    return ck,dict(ck.get('args',{}))

def build_model(saved,device):
    return PeakParametricInverseTransformer1D(
      input_length=int(saved.get('input_points',100)),output_length=int(saved.get('output_points',1000)),
      d_model=int(saved.get('transformer_d_model',64)),nhead=int(saved.get('transformer_nhead',4)),
      num_encoder_layers=int(saved.get('transformer_num_layers',3)),dim_feedforward=int(saved.get('transformer_dim_feedforward',128)),
      dropout=float(saved.get('transformer_dropout',.1)),y_min=float(saved.get('q2_min',-100)),y_max=float(saved.get('q2_max',-6)),
      x_min=float(saved.get('s_min',.1764)),x_max=float(saved.get('s_max',6)),shift=float(saved.get('shift',400)),
      data_scale=float(saved.get('data_scale',160000)),gamma_log_floor=float(saved.get('gamma_log_floor',1e-5))).to(device)

def main():
    a=parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); device=choose_device(a.device); ckd=Path(a.checkpoint_dir)
    latest,saved=load_latest_args(ckd); model=build_model(saved,device)
    if a.weights=='best':
        state=torch.load(ckd/'best_model.pth',map_location='cpu'); selected=json.loads((ckd/'best_model_info.json').read_text(encoding='utf-8')).get('global_step')
    else: state=latest['model_state_dict']; selected=latest.get('global_step')
    model.load_state_dict(state); model.eval()
    pool,meta,usable=open_parameter_pool(Path(a.validation_pool_dir),require_complete=False); n=min(a.num_samples,usable); true_np=np.array(pool[:n],dtype=np.float32,copy=True)
    pa=SimpleNamespace(physics_dtype=str(saved.get('physics_dtype','float32')),noise_level=float(a.noise_level),data_scale=float(saved.get('data_scale',160000)),shift=float(saved.get('shift',400)),s_min=float(saved.get('s_min',.1764)),s_max=float(saved.get('s_max',6)),q2_min=float(saved.get('q2_min',-100)),q2_max=float(saved.get('q2_max',-6)),output_points=int(saved.get('output_points',1000)),input_points=int(saved.get('input_points',100)),integration_points=int(saved.get('integration_points',128)))
    physics=OnlinePhysics(pa,device); rng=np.random.default_rng(a.seed)
    T=[];P=[]; ferr=[];rerr=[];gerr=[]; plots=[]; cur=0
    with torch.no_grad():
      while cur<n:
        stop=min(cur+a.batch_size,n); tp=torch.from_numpy(true_np[cur:stop]).to(device); tf,tr,_=physics.components(tp); gc=physics.forward_from_parameters(tp)
        if a.noise_level>0:
          x=gc.cpu().numpy().astype(np.float64); rms=np.sqrt(np.mean(x*x,axis=1,keepdims=True)); gi=torch.from_numpy((x+a.noise_level*rms*rng.standard_normal(x.shape)).astype(np.float32)).to(device)
        else: gi=gc
        pp=model.predict_parameters(gi.unsqueeze(1)); pf,pr,_=physics.components(pp); pg=physics.forward_from_parameters(pp)
        fe=rel_l2(pf,tf); re=rel_l2(pr,tr); ge=rel_l2(pg,gc)
        T.append(tp.cpu());P.append(pp.cpu());ferr.append(fe.cpu());rerr.append(re.cpu());gerr.append(ge.cpu())
        if len(plots)<a.plot_count:
          take=min(a.plot_count-len(plots),stop-cur)
          for j in range(take): plots.append((cur+j,tf[j].cpu().numpy(),pf[j].cpu().numpy(),tp[j].cpu().numpy(),pp[j].cpu().numpy(),float(fe[j]),float(re[j]),float(ge[j])))
        cur=stop
    true=torch.cat(T);pred=torch.cat(P); ferr=np.asarray(torch.cat(ferr));rerr=np.asarray(torch.cat(rerr));gerr=np.asarray(torch.cat(gerr))
    true_n=normalize_parameters(true,model.parameter_lower.cpu(),model.parameter_upper.cpu()); pred_n=normalize_parameters(pred,model.parameter_lower.cpu(),model.parameter_upper.cpu()); norm_abs=torch.abs(pred_n-true_n).numpy(); abs_err=torch.abs(pred-true).numpy()
    trans_true=transformed_parameter_coordinates(true,model.parameter_lower.cpu(),model.parameter_upper.cpu(),float(saved.get('gamma_log_floor',1e-5))); trans_pred=transformed_parameter_coordinates(pred,model.parameter_lower.cpu(),model.parameter_upper.cpu(),float(saved.get('gamma_log_floor',1e-5))); trans_abs=torch.abs(trans_pred-trans_true).numpy()
    wm=width_metrics(pred,true); width_rel=wm['width_relative_error'].numpy(); width_log=wm['width_log10_abs_error'].numpy()
    ph_pred=resonance_peak_height(pred,float(saved.get('shift',400)),float(saved.get('data_scale',160000))); ph_true=resonance_peak_height(true,float(saved.get('shift',400)),float(saved.get('data_scale',160000))); ph_log=torch.abs(torch.log10(ph_pred.clamp_min(1e-12))-torch.log10(ph_true.clamp_min(1e-12))).numpy()
    sample_rmse=torch.sqrt(torch.mean((trans_pred-trans_true).square(),dim=1)).numpy()
    summary={'weights':a.weights,'selected_model_step':selected,'validation_noise_level':a.noise_level,'num_samples':n,'transformed_parameter_rmse':stat(sample_rmse),'f_relative_l2':stat(ferr),'resonance_relative_l2':stat(rerr),'g_vs_clean_relative_l2':stat(gerr),'width_relative_error':stat(width_rel),'width_log10_abs_error':stat(width_log),'peak_height_log10_abs_error':stat(ph_log),'parameters':{}}
    for k,name in enumerate(PARAMETER_NAMES): summary['parameters'][name]={'absolute_error':stat(abs_err[:,k]),'normalized_absolute_error':stat(norm_abs[:,k])}
    summary['m_transformed_abs_error']=stat(trans_abs[:,3]); summary['log_gamma_normalized_abs_error']=stat(trans_abs[:,4])
    (out/'validation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'parameter_predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
      w=csv.writer(f); w.writerow(['row']+[q for name in PARAMETER_NAMES for q in (f'{name}_true',f'{name}_pred',f'{name}_abs',f'{name}_norm_abs')]+['log_gamma_norm_abs','width_rel','width_log10_abs','peak_height_log10_abs','f_rel','res_rel','g_rel'])
      for i in range(n):
        row=[i]
        for k in range(5): row += [float(true[i,k]),float(pred[i,k]),float(abs_err[i,k]),float(norm_abs[i,k])]
        row += [float(trans_abs[i,4]),float(width_rel[i]),float(width_log[i]),float(ph_log[i]),float(ferr[i]),float(rerr[i]),float(gerr[i])]; w.writerow(row)
    s=physics.s_output.cpu().numpy(); pd=out/'plots';pd.mkdir(exist_ok=True)
    for row,ft,fp,pt,pp,fe,re,ge in plots:
      fig,ax=plt.subplots(figsize=(10,5.5));ax.plot(s,ft,label='f_true');ax.plot(s,fp,label='f_from_predicted_params');ax.grid(alpha=.25);ax.legend();ax.set_xlabel('s');ax.set_ylabel('scaled f(s)');ax.set_title(f'row={row}, f={fe:.4g}, res={re:.4g}, g={ge:.4g}')
      txt='true: '+', '.join(f'{n}={v:.4g}' for n,v in zip(PARAMETER_NAMES,pt))+'\n'+'pred: '+', '.join(f'{n}={v:.4g}' for n,v in zip(PARAMETER_NAMES,pp));ax.text(.01,.01,txt,transform=ax.transAxes,fontsize=8,va='bottom');fig.tight_layout();fig.savefig(pd/f'row_{row:05d}.png',dpi=150);plt.close(fig)
    print('验证完成');print(f'selected model step       : {selected}');print(f'validation noise          : {100*a.noise_level:.2f}%');print(f'transformed param RMSE    : {summary["transformed_parameter_rmse"]["mean"]:.6g}');
    for name in PARAMETER_NAMES:
      it=summary['parameters'][name]['normalized_absolute_error'];print(f'{name:5s} norm abs mean/med : {it["mean"]:.6g} / {it["median"]:.6g}')
    print(f'log_gamma norm abs mean   : {summary["log_gamma_normalized_abs_error"]["mean"]:.6g}');print(f'width log10 abs mean/med  : {summary["width_log10_abs_error"]["mean"]:.6g} / {summary["width_log10_abs_error"]["median"]:.6g}');print(f'width rel error med       : {summary["width_relative_error"]["median"]:.6g}');print(f'peak height log10 mean    : {summary["peak_height_log10_abs_error"]["mean"]:.6g}');print(f'f from params rel L2 mean : {summary["f_relative_l2"]["mean"]:.6g}');print(f'f from params rel L2 med  : {summary["f_relative_l2"]["median"]:.6g}');print(f'resonance rel L2 mean     : {summary["resonance_relative_l2"]["mean"]:.6g}');print(f'g from params rel L2 mean : {summary["g_vs_clean_relative_l2"]["mean"]:.6g}');print(f'output                    : {out}')
if __name__=='__main__': main()
