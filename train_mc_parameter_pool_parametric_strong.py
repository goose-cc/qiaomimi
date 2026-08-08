#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp8 strong resonance-focused parametric training.

Key changes from Exp7:
- separate resonance/background heads;
- gamma predicted on logarithmic scale;
- strong parameter weights for a1/m/log-gamma;
- analytic physical decoder remains mandatory;
- joint spectrum, resonance, gradient, forward-g, width and peak-height losses.
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from TransformerInverse import PeakParametricInverseTransformer1D
from mc_online_physics import OnlinePhysics
from mc_parametric_strong import (
    STRONG_TARGET_NAMES,
    gradient_relative_mse,
    log_peak_height_mse,
    log_width_mse,
    relative_mse,
    strong_parameter_mse,
    transformed_parameter_coordinates,
)
from train_mc_parameter_pool_transformer_loss import (
    ShuffledNoReplacementSampler,
    amp_autocast,
    atomic_json_save,
    atomic_torch_save,
    choose_device,
    make_grad_scaler,
    open_parameter_pool,
    set_random_seed,
)

SCRIPT_VERSION=1


def parse_args():
    p=argparse.ArgumentParser(description='Exp8 strong resonance-focused parametric trainer')
    p.add_argument('--pool-dir',required=True); p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--input-points',type=int,default=100); p.add_argument('--output-points',type=int,default=1000)
    p.add_argument('--integration-points',type=int,default=128); p.add_argument('--noise-level',type=float,default=0.0)
    p.add_argument('--data-scale',type=float,default=160000.0); p.add_argument('--shift',type=float,default=400.0)
    p.add_argument('--s-min',type=float,default=0.1764); p.add_argument('--s-max',type=float,default=6.0)
    p.add_argument('--q2-min',type=float,default=-100.0); p.add_argument('--q2-max',type=float,default=-6.0)
    p.add_argument('--physics-dtype',choices=('float32','float64'),default='float32')
    p.add_argument('--batch-size',type=int,default=64); p.add_argument('--shuffle-block-size',type=int,default=200000)
    p.add_argument('--learning-rate',type=float,default=7e-4); p.add_argument('--weight-decay',type=float,default=1e-5)
    p.add_argument('--grad-clip',type=float,default=1.0)
    p.add_argument('--parameter-weights',default='3,1,1,5,6',help='a1,a2,a3,m,log_gamma')
    p.add_argument('--lambda-param',type=float,default=1.0)
    p.add_argument('--lambda-spectrum',type=float,default=1.0)
    p.add_argument('--lambda-resonance',type=float,default=3.0)
    p.add_argument('--lambda-gradient',type=float,default=0.25)
    p.add_argument('--lambda-physics',type=float,default=0.25)
    p.add_argument('--lambda-width',type=float,default=2.0)
    p.add_argument('--lambda-peak-height',type=float,default=1.0)
    p.add_argument('--gamma-log-floor',type=float,default=1e-5)
    p.add_argument('--transformer-d-model',type=int,default=64); p.add_argument('--transformer-nhead',type=int,default=4)
    p.add_argument('--transformer-num-layers',type=int,default=3); p.add_argument('--transformer-dim-feedforward',type=int,default=128)
    p.add_argument('--transformer-dropout',type=float,default=0.1)
    p.add_argument('--max-steps',type=int,default=30000); p.add_argument('--max-hours',type=float,default=0.0)
    p.add_argument('--checkpoint-every-steps',type=int,default=2000); p.add_argument('--log-every-steps',type=int,default=100)
    p.add_argument('--best-window',type=int,default=100); p.add_argument('--seed',type=int,default=20260721)
    p.add_argument('--device',choices=('auto','cpu','cuda','xpu'),default='auto'); p.add_argument('--amp',action='store_true')
    p.add_argument('--deterministic',action='store_true'); p.add_argument('--require-complete-pool',action='store_true')
    mode=p.add_mutually_exclusive_group(); mode.add_argument('--fresh',action='store_true'); mode.add_argument('--resume',action='store_true')
    return p.parse_args()


def parse_weights(text,device):
    vals=[float(x.strip()) for x in text.split(',')]
    if len(vals)!=5 or any(v<=0 for v in vals): raise ValueError('--parameter-weights needs 5 positive values')
    return torch.tensor(vals,dtype=torch.float32,device=device)


def make_model(args,device):
    return PeakParametricInverseTransformer1D(
        input_length=args.input_points,output_length=args.output_points,d_model=args.transformer_d_model,
        nhead=args.transformer_nhead,num_encoder_layers=args.transformer_num_layers,
        dim_feedforward=args.transformer_dim_feedforward,dropout=args.transformer_dropout,
        y_min=args.q2_min,y_max=args.q2_max,x_min=args.s_min,x_max=args.s_max,
        shift=args.shift,data_scale=args.data_scale,gamma_log_floor=args.gamma_log_floor,
    ).to(device)


def save_latest(path,*,args,model,optimizer,scaler,step,total_samples,cycle,progress,best_score):
    atomic_torch_save({
        'script_version':SCRIPT_VERSION,'args':vars(args),'model_class':model.__class__.__name__,
        'global_step':int(step),'total_samples':int(total_samples),'pool_cycle':int(cycle),
        'next_parameter_id':int(progress),'best_score':float(best_score),'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),'amp_scaler_state_dict':scaler.state_dict() if scaler.is_enabled() else None,
        'saved_at_local':time.strftime('%Y-%m-%d %H:%M:%S')},path)


def load_latest(path,model,optimizer,scaler,device):
    try: ck=torch.load(path,map_location='cpu',weights_only=False)
    except TypeError: ck=torch.load(path,map_location='cpu')
    if ck.get('model_class')!=model.__class__.__name__: raise RuntimeError('checkpoint model class mismatch')
    model.load_state_dict(ck['model_state_dict']); optimizer.load_state_dict(ck['optimizer_state_dict'])
    for st in optimizer.state.values():
        for k,v in st.items():
            if torch.is_tensor(v): st[k]=v.to(device)
    if scaler.is_enabled() and ck.get('amp_scaler_state_dict'): scaler.load_state_dict(ck['amp_scaler_state_dict'])
    return ck


def main():
    args=parse_args()
    if args.max_steps<=0 and args.max_hours<=0: raise ValueError('max-steps and max-hours cannot both be <=0')
    set_random_seed(args.seed,args.deterministic); device=choose_device(args.device); use_amp=bool(args.amp and device.type=='cuda')
    ckdir=Path(args.checkpoint_dir); ckdir.mkdir(parents=True,exist_ok=True)
    latest=ckdir/'latest_checkpoint.pth'; best=ckdir/'best_model.pth'; best_info=ckdir/'best_model_info.json'
    if args.fresh:
        for p in (latest,best,best_info,ckdir/'training_summary.json'):
            if p.exists(): p.unlink()
    pool,metadata,usable=open_parameter_pool(Path(args.pool_dir),require_complete=args.require_complete_pool)
    sampler=ShuffledNoReplacementSampler(pool=pool,usable_rows=usable,block_size=args.shuffle_block_size,seed=args.seed)
    physics=OnlinePhysics(args,device); model=make_model(args,device); pw=parse_weights(args.parameter_weights,device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay); scaler=make_grad_scaler(use_amp)
    step=samples=cycle=progress=0; best_score=float('inf')
    if args.resume:
        if not latest.exists(): raise FileNotFoundError(latest)
        ck=load_latest(latest,model,opt,scaler,device); step=int(ck.get('global_step',0)); samples=int(ck.get('total_samples',0)); cycle=int(ck.get('pool_cycle',0)); progress=int(ck.get('next_parameter_id',0)); best_score=float(ck.get('best_score',float('inf')))

    print('='*78); print('Experiment 8 STRONG: resonance-focused physical-parameter learning')
    print(f'model              : {model.__class__.__name__}'); print(f'model parameters   : {sum(p.numel() for p in model.parameters()):,}')
    print(f'pool rows          : {usable:,}'); print('sampling           : shuffle without replacement')
    print(f'training noise     : {100*args.noise_level:.2f}% RMS Gaussian')
    print('heads              : resonance=[a1,m,log_gamma], background=[a2,a3]')
    print(f'parameter weights  : {args.parameter_weights} for {STRONG_TARGET_NAMES}')
    print('loss weights       : '
          f'param={args.lambda_param:g}, spectrum={args.lambda_spectrum:g}, resonance={args.lambda_resonance:g}, '
          f'grad={args.lambda_gradient:g}, physics={args.lambda_physics:g}, width={args.lambda_width:g}, peakH={args.lambda_peak_height:g}')
    print('='*78)

    recent=deque(maxlen=args.best_window); start=time.monotonic(); start_step=step; stop_reason=None
    while True:
        if args.max_steps>0 and step-start_step>=args.max_steps: stop_reason=f'reached max_steps={args.max_steps}'; break
        if args.max_hours>0 and (time.monotonic()-start)/3600>=args.max_hours: stop_reason=f'reached max_hours={args.max_hours}'; break
        if progress+args.batch_size>usable: cycle+=1; progress=0
        np_params=sampler.sample_chunk(cycle=cycle,progress=progress,chunk_size=args.batch_size); progress+=args.batch_size
        if progress>=usable: cycle+=1; progress=0
        true_params=torch.from_numpy(np_params).to(device=device,dtype=torch.float32)
        with torch.no_grad():
            true_total,true_res,_=physics.components(true_params); g_clean=physics.forward_from_parameters(true_params)
            g_input=physics.add_noise(g_clean) if args.noise_level>0 else g_clean

        opt.zero_grad(set_to_none=True)
        with amp_autocast(use_amp):
            pred_params=model.predict_parameters(g_input.unsqueeze(1))
            pred_total,pred_res,_=physics.components(pred_params)
            pred_g=physics.forward_from_parameters(pred_params)
            l_param=strong_parameter_mse(pred_params,true_params,model.parameter_lower,model.parameter_upper,pw,args.gamma_log_floor)
            l_spec=relative_mse(pred_total,true_total,floor=true_total)
            l_res=relative_mse(pred_res,true_res,floor=true_total,floor_fraction=0.01)
            l_grad=gradient_relative_mse(pred_total,true_total,true_total)
            l_phys=relative_mse(pred_g,g_clean,floor=g_clean,floor_fraction=0.0)
            l_width=log_width_mse(pred_params,true_params)
            l_peak=log_peak_height_mse(pred_params,true_params,args.shift,args.data_scale)
            loss=(args.lambda_param*l_param + args.lambda_spectrum*l_spec + args.lambda_resonance*l_res +
                  args.lambda_gradient*l_grad + args.lambda_physics*l_phys + args.lambda_width*l_width +
                  args.lambda_peak_height*l_peak)
        if not torch.isfinite(loss): raise FloatingPointError(f'non-finite loss at step {step+1}')
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
        scaler.step(opt); scaler.update(); step+=1; samples+=args.batch_size; recent.append(float(loss.detach().cpu()))

        if step%args.log_every_steps==0:
            with torch.no_grad():
                pn=transformed_parameter_coordinates(pred_params,model.parameter_lower,model.parameter_upper,args.gamma_log_floor)
                tn=transformed_parameter_coordinates(true_params,model.parameter_lower,model.parameter_upper,args.gamma_log_floor)
                mae=torch.mean(torch.abs(pn-tn),dim=0).cpu().numpy()
            rolling=float(np.mean(recent))
            print(f'step={step:6d} total={float(loss):.5e} rolling={rolling:.5e} | '
                  f'param={float(l_param):.3e} spec={float(l_spec):.3e} res={float(l_res):.3e} '
                  f'g={float(l_phys):.3e} width={float(l_width):.3e} | '
                  + ' '.join(f'{n}={v:.4f}' for n,v in zip(STRONG_TARGET_NAMES,mae)))
            if len(recent)==recent.maxlen and rolling<best_score:
                best_score=rolling; atomic_torch_save(model.state_dict(),best)
                atomic_json_save({'model_class':model.__class__.__name__,'global_step':step,'best_score':best_score,
                                  'score_definition':'rolling strong joint loss','noise_level':args.noise_level,
                                  'parameter_weights':args.parameter_weights},best_info)
        if step%args.checkpoint_every_steps==0:
            save_latest(latest,args=args,model=model,optimizer=opt,scaler=scaler,step=step,total_samples=samples,cycle=cycle,progress=progress,best_score=best_score)

    save_latest(latest,args=args,model=model,optimizer=opt,scaler=scaler,step=step,total_samples=samples,cycle=cycle,progress=progress,best_score=best_score)
    atomic_json_save({'global_step':step,'total_samples':samples,'best_score':best_score,'stop_reason':stop_reason,
                      'noise_level':args.noise_level,'target':['a1','a2','a3','m','log_gamma'],'mode':'strong_joint_physics'},ckdir/'training_summary.json')
    print(f'training finished: {stop_reason}'); print(f'latest: {latest}'); print(f'best  : {best}')

if __name__=='__main__': main()
