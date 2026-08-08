#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

from TransformerInverse import ParametricInverseTransformer1D
from mc_online_physics import OnlinePhysics
from mc_parametric import normalize_parameters
from train_mc_parameter_pool_transformer_loss import choose_device, open_parameter_pool

PARAMETER_NAMES = ('a1','a2','a3','m','gamma')


def parse_args():
    p = argparse.ArgumentParser(description='Validate Exp7 parameter-identifiability model')
    p.add_argument('--validation-pool-dir', required=True)
    p.add_argument('--checkpoint-dir', required=True)
    p.add_argument('--weights', choices=('best','latest'), default='latest')
    p.add_argument('--num-samples', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--noise-level', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=20260802)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--plot-count', type=int, default=12)
    p.add_argument('--device', choices=('auto','cpu','cuda','xpu'), default='auto')
    return p.parse_args()


def load_training_args(checkpoint_dir):
    latest = checkpoint_dir/'latest_checkpoint.pth'
    if not latest.exists():
        raise FileNotFoundError(latest)
    try:
        ckpt = torch.load(latest, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(latest, map_location='cpu')
    saved = dict(ckpt.get('args', {}))
    return ckpt, saved


def build_model(saved, device):
    return ParametricInverseTransformer1D(
        input_length=int(saved.get('input_points',100)),
        output_length=int(saved.get('output_points',1000)),
        d_model=int(saved.get('transformer_d_model',64)),
        nhead=int(saved.get('transformer_nhead',4)),
        num_encoder_layers=int(saved.get('transformer_num_layers',3)),
        dim_feedforward=int(saved.get('transformer_dim_feedforward',128)),
        dropout=float(saved.get('transformer_dropout',0.1)),
        y_min=float(saved.get('q2_min',-100.0)),
        y_max=float(saved.get('q2_max',-6.0)),
        x_min=float(saved.get('s_min',0.1764)),
        x_max=float(saved.get('s_max',6.0)),
        shift=float(saved.get('shift',400.0)),
        data_scale=float(saved.get('data_scale',160000.0)),
    ).to(device)


def rel_l2(pred, true, eps=1e-12):
    num = torch.linalg.vector_norm(pred-true, dim=1)
    den = torch.linalg.vector_norm(true, dim=1).clamp_min(eps)
    return num/den


def stat(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        'mean': float(np.mean(x)),
        'median': float(np.median(x)),
        'p90': float(np.quantile(x,0.90)),
    }


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    latest_ckpt, saved = load_training_args(checkpoint_dir)
    model = build_model(saved, device)

    if args.weights == 'best':
        weight_path = checkpoint_dir/'best_model.pth'
        state = torch.load(weight_path, map_location='cpu')
        selected_step = json.loads((checkpoint_dir/'best_model_info.json').read_text(encoding='utf-8')).get('global_step')
    else:
        state = latest_ckpt['model_state_dict']
        selected_step = latest_ckpt.get('global_step')
    model.load_state_dict(state)
    model.eval()

    pool, metadata, usable = open_parameter_pool(Path(args.validation_pool_dir), require_complete=False)
    n = min(args.num_samples, usable)
    true_np = np.array(pool[:n], dtype=np.float32, copy=True)

    physics_args = SimpleNamespace(
        physics_dtype=str(saved.get('physics_dtype','float32')),
        noise_level=float(args.noise_level),
        data_scale=float(saved.get('data_scale',160000.0)),
        shift=float(saved.get('shift',400.0)),
        s_min=float(saved.get('s_min',0.1764)),
        s_max=float(saved.get('s_max',6.0)),
        q2_min=float(saved.get('q2_min',-100.0)),
        q2_max=float(saved.get('q2_max',-6.0)),
        output_points=int(saved.get('output_points',1000)),
        input_points=int(saved.get('input_points',100)),
        integration_points=int(saved.get('integration_points',128)),
    )
    physics = OnlinePhysics(physics_args, device)
    rng = np.random.default_rng(args.seed)

    all_true=[]; all_pred=[]; f_err=[]; g_err=[]; width_rel=[]
    plots=[]
    cursor=0
    with torch.no_grad():
        while cursor<n:
            stop=min(cursor+args.batch_size,n)
            true_params=torch.from_numpy(true_np[cursor:stop]).to(device)
            true_f,g_clean=physics.make_clean_batch(true_params)
            if args.noise_level>0:
                # Generate fixed validation noise on CPU so the seed is explicit.
                g_np=g_clean.cpu().numpy().astype(np.float64)
                rms=np.sqrt(np.mean(g_np*g_np,axis=1,keepdims=True))
                g_in_np=g_np + args.noise_level*rms*rng.standard_normal(g_np.shape)
                g_input=torch.from_numpy(g_in_np.astype(np.float32)).to(device)
            else:
                g_input=g_clean

            pred_params=model.predict_parameters(g_input.unsqueeze(1))
            pred_f,_,_=physics.components(pred_params)
            pred_g=physics.forward_from_parameters(pred_params)
            fe=rel_l2(pred_f,true_f)
            ge=rel_l2(pred_g,g_clean)
            true_width=true_params[:,3]*true_params[:,4]
            pred_width=pred_params[:,3]*pred_params[:,4]
            wr=torch.abs(pred_width-true_width)/true_width.abs().clamp_min(1e-8)

            all_true.append(true_params.cpu())
            all_pred.append(pred_params.cpu())
            f_err.append(fe.cpu())
            g_err.append(ge.cpu())
            width_rel.append(wr.cpu())

            if len(plots)<args.plot_count:
                take=min(args.plot_count-len(plots), stop-cursor)
                for j in range(take):
                    plots.append((
                        cursor+j,
                        true_f[j].cpu().numpy(), pred_f[j].cpu().numpy(),
                        true_params[j].cpu().numpy(), pred_params[j].cpu().numpy(),
                        float(fe[j].cpu()), float(ge[j].cpu()),
                    ))
            cursor=stop

    true=torch.cat(all_true,0)
    pred=torch.cat(all_pred,0)
    f_err=np.asarray(torch.cat(f_err,0))
    g_err=np.asarray(torch.cat(g_err,0))
    width_rel=np.asarray(torch.cat(width_rel,0))
    true_n=normalize_parameters(true, model.parameter_lower.cpu(), model.parameter_upper.cpu())
    pred_n=normalize_parameters(pred, model.parameter_lower.cpu(), model.parameter_upper.cpu())
    norm_abs=torch.abs(pred_n-true_n).numpy()
    abs_err=torch.abs(pred-true).numpy()
    sample_norm_rmse=torch.sqrt(torch.mean((pred_n-true_n).square(),dim=1)).numpy()

    summary={
        'weights': args.weights,
        'selected_model_step': selected_step,
        'validation_noise_level': args.noise_level,
        'num_samples': n,
        'parameter_normalized_rmse': stat(sample_norm_rmse),
        'reconstructed_f_relative_l2': stat(f_err),
        'reconstructed_g_vs_clean_relative_l2': stat(g_err),
        'width_m_gamma_relative_error': stat(width_rel),
        'parameters': {},
    }
    for k,name in enumerate(PARAMETER_NAMES):
        summary['parameters'][name]={
            'absolute_error': stat(abs_err[:,k]),
            'normalized_absolute_error': stat(norm_abs[:,k]),
        }

    (out/'validation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

    with (out/'parameter_predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f)
        header=['row']
        for name in PARAMETER_NAMES:
            header += [f'{name}_true', f'{name}_pred', f'{name}_abs_err', f'{name}_norm_abs_err']
        header += ['width_rel_err','f_rel_l2','g_rel_clean']
        w.writerow(header)
        for i in range(n):
            row=[i]
            for k in range(5):
                row += [float(true[i,k]), float(pred[i,k]), float(abs_err[i,k]), float(norm_abs[i,k])]
            row += [float(width_rel[i]), float(f_err[i]), float(g_err[i])]
            w.writerow(row)

    s=physics.s_output.detach().cpu().numpy()
    plot_dir=out/'plots'
    plot_dir.mkdir(exist_ok=True)
    for row,ft,fp,pt,pp,fe,ge in plots:
        fig,ax=plt.subplots(figsize=(10,5.5))
        ax.plot(s,ft,label='f_true')
        ax.plot(s,fp,label='f_from_predicted_params')
        ax.set_xlabel('s'); ax.set_ylabel('scaled f(s)'); ax.grid(alpha=0.25); ax.legend()
        ax.set_title(f'row={row}, f_rel={fe:.4g}, g_rel={ge:.4g}')
        text='true: '+', '.join(f'{n}={v:.4g}' for n,v in zip(PARAMETER_NAMES,pt))+'\n' + \
             'pred: '+', '.join(f'{n}={v:.4g}' for n,v in zip(PARAMETER_NAMES,pp))
        ax.text(0.01,0.01,text,transform=ax.transAxes,fontsize=8,va='bottom')
        fig.tight_layout(); fig.savefig(plot_dir/f'row_{row:05d}.png',dpi=150); plt.close(fig)

    print('验证完成')
    print(f'selected model step       : {selected_step}')
    print(f'validation noise          : {100*args.noise_level:.2f}%')
    print(f'parameter norm RMSE mean  : {summary["parameter_normalized_rmse"]["mean"]:.6g}')
    print(f'parameter norm RMSE median: {summary["parameter_normalized_rmse"]["median"]:.6g}')
    for name in PARAMETER_NAMES:
        item=summary['parameters'][name]['normalized_absolute_error']
        print(f'{name:5s} norm abs mean/med : {item["mean"]:.6g} / {item["median"]:.6g}')
    print(f'width rel error mean      : {summary["width_m_gamma_relative_error"]["mean"]:.6g}')
    print(f'f from params rel L2 mean : {summary["reconstructed_f_relative_l2"]["mean"]:.6g}')
    print(f'f from params rel L2 med  : {summary["reconstructed_f_relative_l2"]["median"]:.6g}')
    print(f'g from params rel L2 mean : {summary["reconstructed_g_vs_clean_relative_l2"]["mean"]:.6g}')
    print(f'output                    : {out}')


if __name__=='__main__':
    main()
