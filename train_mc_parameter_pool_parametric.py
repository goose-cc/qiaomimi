#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp7: 直接预测 5 个物理参数，用于判断 100 点 g 的可辨识性。

核心原则：
- 第一阶段只训练 parameter loss，不使用完整谱 loss，不用峰 loss；
- 默认 0% 训练噪声，先回答“干净 g 是否足以确定参数”；
- 使用与 Exp1/Exp2 相同的 1.6 亿参数池和 shuffle 无放回采样。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from TransformerInverse import ParametricInverseTransformer1D
from mc_online_physics import OnlinePhysics
from mc_parametric import normalize_parameters, normalized_parameter_mse
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

PARAMETER_NAMES = ('a1', 'a2', 'a3', 'm', 'gamma')
SCRIPT_VERSION = 1


def parse_args():
    p = argparse.ArgumentParser(description='Exp7 parameter-identifiability trainer')
    p.add_argument('--pool-dir', required=True)
    p.add_argument('--checkpoint-dir', required=True)
    p.add_argument('--input-points', type=int, default=100)
    p.add_argument('--output-points', type=int, default=1000, help='只用于解析重建 f 的网格')
    p.add_argument('--integration-points', type=int, default=128)
    p.add_argument('--noise-level', type=float, default=0.0)
    p.add_argument('--data-scale', type=float, default=160000.0)
    p.add_argument('--shift', type=float, default=400.0)
    p.add_argument('--s-min', type=float, default=0.1764)
    p.add_argument('--s-max', type=float, default=6.0)
    p.add_argument('--q2-min', type=float, default=-100.0)
    p.add_argument('--q2-max', type=float, default=-6.0)
    p.add_argument('--physics-dtype', choices=('float32','float64'), default='float32')

    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--shuffle-block-size', type=int, default=200000)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--parameter-weights', default='1,1,1,1,1')

    p.add_argument('--transformer-d-model', type=int, default=64)
    p.add_argument('--transformer-nhead', type=int, default=4)
    p.add_argument('--transformer-num-layers', type=int, default=3)
    p.add_argument('--transformer-dim-feedforward', type=int, default=128)
    p.add_argument('--transformer-dropout', type=float, default=0.1)

    p.add_argument('--max-steps', type=int, default=30000)
    p.add_argument('--max-hours', type=float, default=0.0)
    p.add_argument('--checkpoint-every-steps', type=int, default=2000)
    p.add_argument('--log-every-steps', type=int, default=100)
    p.add_argument('--best-window', type=int, default=100)
    p.add_argument('--seed', type=int, default=20260721)
    p.add_argument('--device', choices=('auto','cpu','cuda','xpu'), default='auto')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--require-complete-pool', action='store_true')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--fresh', action='store_true')
    mode.add_argument('--resume', action='store_true')
    return p.parse_args()


def parse_weights(text, device):
    values = [float(x.strip()) for x in text.split(',')]
    if len(values) != 5 or any(v <= 0 for v in values):
        raise ValueError('--parameter-weights 必须是 5 个正数，例如 1,1,1,1,1')
    return torch.tensor(values, dtype=torch.float32, device=device)


def make_model(args, device):
    model = ParametricInverseTransformer1D(
        input_length=args.input_points,
        output_length=args.output_points,
        d_model=args.transformer_d_model,
        nhead=args.transformer_nhead,
        num_encoder_layers=args.transformer_num_layers,
        dim_feedforward=args.transformer_dim_feedforward,
        dropout=args.transformer_dropout,
        y_min=args.q2_min,
        y_max=args.q2_max,
        x_min=args.s_min,
        x_max=args.s_max,
        shift=args.shift,
        data_scale=args.data_scale,
    ).to(device)
    return model


def save_latest(path, *, args, model, optimizer, scaler, step, total_samples,
                cycle, progress, best_score):
    payload = {
        'script_version': SCRIPT_VERSION,
        'args': vars(args),
        'model_class': model.__class__.__name__,
        'global_step': int(step),
        'total_samples': int(total_samples),
        'pool_cycle': int(cycle),
        'next_parameter_id': int(progress),
        'best_score': float(best_score),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'amp_scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
        'saved_at_local': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    atomic_torch_save(payload, path)


def load_latest(path, model, optimizer, scaler, device):
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location='cpu')
    if ckpt.get('model_class') != model.__class__.__name__:
        raise RuntimeError('checkpoint model class mismatch')
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    if scaler.is_enabled() and ckpt.get('amp_scaler_state_dict'):
        scaler.load_state_dict(ckpt['amp_scaler_state_dict'])
    return ckpt


def main():
    args = parse_args()
    if args.max_steps <= 0 and args.max_hours <= 0:
        raise ValueError('max-steps 和 max-hours 至少一个必须 > 0')
    if args.noise_level < 0:
        raise ValueError('noise-level 不能为负数')
    if args.batch_size <= 0:
        raise ValueError('batch-size 必须 > 0')

    set_random_seed(args.seed, args.deterministic)
    device = choose_device(args.device)
    use_amp = bool(args.amp and device.type == 'cuda')
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir/'latest_checkpoint.pth'
    best_path = checkpoint_dir/'best_model.pth'
    best_info_path = checkpoint_dir/'best_model_info.json'

    if args.fresh:
        for p in (latest_path, best_path, best_info_path, checkpoint_dir/'training_summary.json'):
            if p.exists():
                p.unlink()

    pool, metadata, usable_rows = open_parameter_pool(
        Path(args.pool_dir), require_complete=args.require_complete_pool
    )
    sampler = ShuffledNoReplacementSampler(
        pool=pool, usable_rows=usable_rows,
        block_size=args.shuffle_block_size, seed=args.seed,
    )

    physics = OnlinePhysics(args, device)
    model = make_model(args, device)
    weights = parse_weights(args.parameter_weights, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = make_grad_scaler(use_amp)

    global_step = 0
    total_samples = 0
    cycle = 0
    progress = 0
    best_score = float('inf')
    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(latest_path)
        ckpt = load_latest(latest_path, model, optimizer, scaler, device)
        global_step = int(ckpt.get('global_step', 0))
        total_samples = int(ckpt.get('total_samples', 0))
        cycle = int(ckpt.get('pool_cycle', 0))
        progress = int(ckpt.get('next_parameter_id', 0))
        best_score = float(ckpt.get('best_score', float('inf')))

    print('='*72)
    print('Experiment 7: direct physical-parameter identifiability')
    print(f'model              : {model.__class__.__name__}')
    print(f'model parameters   : {sum(p.numel() for p in model.parameters()):,}')
    print(f'pool rows          : {usable_rows:,}')
    print('sampling           : shuffle without replacement')
    print(f'training noise     : {100*args.noise_level:.2f}% RMS Gaussian')
    print('target             : [a1,a2,a3,m,gamma]')
    print(f'parameter weights  : {args.parameter_weights}')
    print('curve/peak loss    : NONE (diagnostic on purpose)')
    print('='*72)

    recent = deque(maxlen=args.best_window)
    start = time.monotonic()
    start_step = global_step
    stop_reason = None

    while True:
        if args.max_steps > 0 and global_step - start_step >= args.max_steps:
            stop_reason = f'reached max_steps={args.max_steps}'
            break
        if args.max_hours > 0 and (time.monotonic()-start)/3600 >= args.max_hours:
            stop_reason = f'reached max_hours={args.max_hours}'
            break

        if progress + args.batch_size > usable_rows:
            cycle += 1
            progress = 0
        params_np = sampler.sample_chunk(
            cycle=cycle, progress=progress, chunk_size=args.batch_size
        )
        progress += args.batch_size
        if progress >= usable_rows:
            cycle += 1
            progress = 0

        true_params = torch.from_numpy(params_np).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            _, g_clean = physics.make_clean_batch(true_params)
            if args.noise_level > 0:
                g_input = physics.add_noise(g_clean)
            else:
                g_input = g_clean

        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(use_amp):
            pred_params = model.predict_parameters(g_input.unsqueeze(1))
            loss = normalized_parameter_mse(
                pred_params, true_params,
                model.parameter_lower, model.parameter_upper,
                weights,
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at step {global_step+1}')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        global_step += 1
        total_samples += args.batch_size
        recent.append(float(loss.detach().cpu()))

        if global_step % args.log_every_steps == 0:
            with torch.no_grad():
                pred_n = normalize_parameters(
                    pred_params, model.parameter_lower, model.parameter_upper
                )
                true_n = normalize_parameters(
                    true_params, model.parameter_lower, model.parameter_upper
                )
                mae_n = torch.mean(torch.abs(pred_n-true_n), dim=0).cpu().numpy()
                rmse_n = torch.sqrt(torch.mean((pred_n-true_n).square())).item()
            rolling = float(np.mean(recent))
            print(
                f'step={global_step:6d} loss={float(loss):.6e} '
                f'rolling={rolling:.6e} norm_rmse={rmse_n:.5f} | '
                + ' '.join(f'{n}={v:.4f}' for n,v in zip(PARAMETER_NAMES, mae_n))
            )
            if len(recent) == recent.maxlen and rolling < best_score:
                best_score = rolling
                atomic_torch_save(model.state_dict(), best_path)
                atomic_json_save({
                    'model_class': model.__class__.__name__,
                    'global_step': global_step,
                    'best_score': best_score,
                    'score_definition': 'rolling normalized parameter MSE',
                    'noise_level': args.noise_level,
                    'parameter_weights': args.parameter_weights,
                }, best_info_path)

        if global_step % args.checkpoint_every_steps == 0:
            save_latest(
                latest_path, args=args, model=model, optimizer=optimizer, scaler=scaler,
                step=global_step, total_samples=total_samples,
                cycle=cycle, progress=progress, best_score=best_score,
            )

    save_latest(
        latest_path, args=args, model=model, optimizer=optimizer, scaler=scaler,
        step=global_step, total_samples=total_samples,
        cycle=cycle, progress=progress, best_score=best_score,
    )
    atomic_json_save({
        'global_step': global_step,
        'total_samples': total_samples,
        'best_score': best_score,
        'stop_reason': stop_reason,
        'noise_level': args.noise_level,
        'parameter_target': list(PARAMETER_NAMES),
    }, checkpoint_dir/'training_summary.json')
    print(f'training finished: {stop_reason}')
    print(f'latest: {latest_path}')
    print(f'best  : {best_path}')


if __name__ == '__main__':
    main()
