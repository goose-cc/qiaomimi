#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp23 formal spectrum curriculum on the original large parameter pool.

This is the production-style counterpart to the parameter-regression diagnostic:

    1.6e8 parameter pool -> physical g(q^2) -> original Transformer/BiLSTM+Transformer
                          -> f(s)

The only new data policy is coverage-first g-diverse sampling.  It preserves
continuous parameter coverage and treats the requested g gap as a soft target,
never as a reason to delete an entire parameter region.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import exp23_pool_sampler as exp23
from TransformerInverse import InverseBiLSTMTransformer1D, InverseTransformer1D
from mc_inverse_loss import MonteCarloInverseLoss
from mc_online_physics import OnlinePhysics
from train_mc_parameter_pool_transformer_loss import (
    amp_autocast,
    atomic_json_save,
    atomic_torch_save,
    choose_device,
    make_grad_scaler,
    open_parameter_pool,
    set_random_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exp23 formal g->f curriculum with coverage-preserving g-diverse pool sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pool-dir", required=True)
    p.add_argument("--val-pool-dir", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--stage", choices=tuple(exp23.STAGE_SPECS), required=True)
    p.add_argument(
        "--selection-strategy",
        choices=("coverage", "coverage_gdiverse"),
        default="coverage_gdiverse",
    )
    p.add_argument("--target-g-gap-percent", type=float, default=0.4)
    p.add_argument("--coverage-bins", type=int, default=0)
    p.add_argument("--candidate-multiplier", type=int, default=4)
    p.add_argument("--shuffle-block-size", type=int, default=200_000)
    p.add_argument("--require-complete-pool", action="store_true")
    p.add_argument("--require-complete-val-pool", action="store_true")

    p.add_argument(
        "--model-type",
        choices=("transformer", "bilstm_transformer"),
        default="bilstm_transformer",
        help="The actual project inverse network. Keep this fixed across all curriculum stages.",
    )
    p.add_argument("--input-points", type=int, default=100)
    p.add_argument("--output-points", type=int, default=1000)
    p.add_argument("--integration-points", type=int, default=128)
    p.add_argument("--noise-level", type=float, default=0.002)
    p.add_argument("--data-scale", type=float, default=160000.0)
    p.add_argument("--shift", type=float, default=400.0)
    p.add_argument("--s-min", type=float, default=0.1764)
    p.add_argument("--s-max", type=float, default=6.0)
    p.add_argument("--q2-min", type=float, default=-100.0)
    p.add_argument("--q2-max", type=float, default=-6.0)
    p.add_argument("--physics-dtype", choices=("float32", "float64"), default="float32")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--log-every-steps", type=int, default=100)
    p.add_argument("--checkpoint-every-steps", type=int, default=2000)
    p.add_argument("--validation-every-steps", type=int, default=2000)
    p.add_argument("--val-samples", type=int, default=10000)
    p.add_argument("--val-batch-size", type=int, default=64)

    # Same formal loss family as train_mc_parameter_pool_transformer_loss.py.
    p.add_argument(
        "--loss-profile",
        choices=(
            "base", "mse_grad", "huber_grad", "pinn", "pinn_smooth",
            "pinn_tikhonov", "pinn_huber", "pinn_full",
        ),
        default="pinn",
    )
    p.add_argument("--loss-normalization", choices=("relative", "absolute"), default="relative")
    p.add_argument("--lambda-grad", type=float, default=0.1)
    p.add_argument("--gradient-reference-points", type=int, default=100)
    p.add_argument("--lambda-physics", type=float, default=0.1)
    p.add_argument("--lambda-smooth", type=float, default=0.0)
    p.add_argument("--lambda-tv", type=float, default=0.0)
    p.add_argument("--lambda-tikhonov", type=float, default=0.0)
    p.add_argument("--huber-beta", type=float, default=0.1)
    p.add_argument("--loss-eps", type=float, default=1e-8)

    p.add_argument("--transformer-d-model", type=int, default=64)
    p.add_argument("--transformer-nhead", type=int, default=4)
    p.add_argument("--transformer-num-layers", type=int, default=3)
    p.add_argument("--transformer-dim-feedforward", type=int, default=128)
    p.add_argument("--transformer-dropout", type=float, default=0.1)
    p.add_argument("--lstm-hidden-size", type=int, default=32)
    p.add_argument("--lstm-num-layers", type=int, default=2)
    p.add_argument("--lstm-dropout", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=20260814)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", default="")
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.fresh and args.resume:
        raise ValueError("--fresh and --resume cannot be used together")
    if args.batch_size <= 0 or args.micro_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.micro_batch_size > args.batch_size:
        args.micro_batch_size = args.batch_size
    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    if args.candidate_multiplier < 1:
        raise ValueError("candidate-multiplier must be >= 1")
    if args.target_g_gap_percent < 0:
        raise ValueError("target g gap cannot be negative")
    if args.transformer_d_model % args.transformer_nhead != 0:
        raise ValueError("d_model must be divisible by nhead")


def make_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    common = dict(
        input_length=args.input_points,
        output_length=args.output_points,
        d_model=args.transformer_d_model,
        nhead=args.transformer_nhead,
        num_encoder_layers=args.transformer_num_layers,
        num_decoder_layers=args.transformer_num_layers,
        dim_feedforward=args.transformer_dim_feedforward,
        dropout=args.transformer_dropout,
        x_min=args.s_min,
        x_max=args.s_max,
        y_min=args.q2_min,
        y_max=args.q2_max,
        normalize_coordinates=True,
        rms_normalize_io=True,
        rms_eps=1e-8,
    )
    if args.model_type == "bilstm_transformer":
        model = InverseBiLSTMTransformer1D(
            **common,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_num_layers=args.lstm_num_layers,
            lstm_dropout=args.lstm_dropout,
            lstm_residual=True,
        )
    else:
        model = InverseTransformer1D(**common)
    return model.to(device)


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model_state(path: Path) -> dict[str, torch.Tensor]:
    obj = torch_load(path)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if not isinstance(obj, dict):
        raise RuntimeError(f"unsupported model file: {path}")
    return obj


def make_criterion(args: argparse.Namespace, device: torch.device) -> MonteCarloInverseLoss:
    return MonteCarloInverseLoss(
        s_min=args.s_min,
        s_max=args.s_max,
        q2_min=args.q2_min,
        q2_max=args.q2_max,
        output_points=args.output_points,
        input_points=args.input_points,
        profile=args.loss_profile,
        normalization=args.loss_normalization,
        lambda_grad=args.lambda_grad,
        gradient_reference_points=args.gradient_reference_points,
        lambda_physics=args.lambda_physics,
        lambda_smooth=args.lambda_smooth,
        lambda_tv=args.lambda_tv,
        lambda_tikhonov=args.lambda_tikhonov,
        huber_beta=args.huber_beta,
        eps=args.loss_eps,
    ).to(device)


def normalize_prediction(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 2:
        pred = pred.unsqueeze(1)
    if pred.shape != target.shape:
        raise RuntimeError(f"prediction shape {tuple(pred.shape)} != target {tuple(target.shape)}")
    return pred


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    physics: OnlinePhysics,
    criterion: MonteCarloInverseLoss,
    params_np: np.ndarray,
    *,
    batch_size: int,
    noise_level: float,
    seed: int,
) -> dict[str, float]:
    model.eval()
    gen = torch.Generator(device=physics.device)
    gen.manual_seed(int(seed))
    s_grid = physics.s_output.to(dtype=torch.float32)
    spacing = float((physics.s_max - physics.s_min) / max(physics.output_points - 1, 1))

    f_rel_sum = 0.0
    g_rel_sum = 0.0
    resonance_rel_sum = 0.0
    peak_height_rel_sum = 0.0
    count = 0

    for start in range(0, len(params_np), batch_size):
        stop = min(start + batch_size, len(params_np))
        truth_params = torch.from_numpy(params_np[start:stop]).to(physics.device, dtype=torch.float32)
        f_true, g_clean = physics.make_clean_batch(truth_params)
        g_in = exp23.add_fractional_noise(g_clean, noise_level, gen)
        pred = normalize_prediction(model(g_in.unsqueeze(1)), f_true.unsqueeze(1))[:, 0]

        f_rel = torch.linalg.vector_norm(pred - f_true, dim=1) / torch.linalg.vector_norm(f_true, dim=1).clamp_min(1e-12)
        g_pred = criterion.physics_forward_integral(pred.unsqueeze(1))[:, 0]
        g_rel = torch.linalg.vector_norm(g_pred - g_clean, dim=1) / torch.linalg.vector_norm(g_clean, dim=1).clamp_min(1e-12)

        mass = truth_params[:, 3:4]
        width = (truth_params[:, 3:4] * truth_params[:, 4:5]).clamp_min(spacing)
        radius = torch.maximum(3.0 * width, torch.full_like(width, 2.0 * spacing))
        mask = (s_grid.view(1, -1) >= (mass - radius)) & (s_grid.view(1, -1) <= (mass + radius))
        mask_f = mask.to(dtype=pred.dtype)
        diff_sq = ((pred - f_true).square() * mask_f).sum(dim=1)
        true_sq = (f_true.square() * mask_f).sum(dim=1).clamp_min(1e-12)
        resonance_rel = torch.sqrt(diff_sq / true_sq)

        neg_inf = torch.full_like(pred, -float("inf"))
        pred_peak = torch.where(mask, pred, neg_inf).max(dim=1).values
        true_peak = torch.where(mask, f_true, neg_inf).max(dim=1).values
        peak_rel = torch.abs(pred_peak - true_peak) / true_peak.abs().clamp_min(1e-8)

        f_rel_sum += float(f_rel.sum().cpu())
        g_rel_sum += float(g_rel.sum().cpu())
        resonance_rel_sum += float(resonance_rel.sum().cpu())
        peak_height_rel_sum += float(peak_rel.sum().cpu())
        count += len(truth_params)

    return {
        "noise_level": float(noise_level),
        "samples": int(count),
        "f_relative_l2_mean": float(f_rel_sum / max(count, 1)),
        "physics_g_relative_l2_mean": float(g_rel_sum / max(count, 1)),
        "resonance_window_relative_l2_mean": float(resonance_rel_sum / max(count, 1)),
        "resonance_peak_height_relative_error_mean": float(peak_height_rel_sum / max(count, 1)),
    }


def write_selection_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [
        "step", "loss", "batch_min_g_gap_percent", "batch_median_nearest_g_gap_percent",
        "target_g_gap_percent", "target_met", "coverage_mean_fraction",
        "unique_joint_fraction_of_batch", "global_joint_bins_coverage_fraction", "pool_rows_read",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    spec = exp23.STAGE_SPECS[args.stage]
    free_idx = tuple(spec["free"])
    coverage_bins = int(args.coverage_bins or spec["default_bins"])

    set_random_seed(args.seed, args.deterministic)
    device = choose_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    pool, _, usable_rows = open_parameter_pool(Path(args.pool_dir), args.require_complete_pool)
    val_pool, _, val_usable_rows = open_parameter_pool(Path(args.val_pool_dir), args.require_complete_val_pool)
    physics = OnlinePhysics(args, device)
    criterion = make_criterion(args, device)
    model = make_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(use_amp)

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "latest_checkpoint.pth"
    best_path = out_dir / "best_model.pth"
    best_info_path = out_dir / "validation_best.json"
    if args.fresh:
        for p in (latest_path, best_path, best_info_path, out_dir / "training_summary.json", out_dir / "selection_diagnostics.csv"):
            if p.exists():
                p.unlink()

    start_step = 0
    best_score = float("inf")
    pool_cycle = 0
    pool_progress = 0
    global_bin_counts = torch.zeros((len(free_idx), coverage_bins), dtype=torch.long, device=device)
    global_joint_counts = torch.zeros(coverage_bins ** len(free_idx), dtype=torch.long, device=device)

    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(latest_path)
        ckpt = torch_load(latest_path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        try:
            scaler.load_state_dict(ckpt.get("scaler_state_dict", {}))
        except Exception:
            pass
        start_step = int(ckpt.get("step", 0))
        pool_cycle = int(ckpt.get("pool_cycle", 0))
        pool_progress = int(ckpt.get("pool_progress", 0))
        best_score = float(ckpt.get("best_score", float("inf")))
        if ckpt.get("global_bin_counts") is not None:
            global_bin_counts = ckpt["global_bin_counts"].to(device=device, dtype=torch.long)
        if ckpt.get("global_joint_counts") is not None:
            global_joint_counts = ckpt["global_joint_counts"].to(device=device, dtype=torch.long)
    elif args.init_from:
        model.load_state_dict(load_model_state(Path(args.init_from)), strict=True)

    cursor = exp23.PoolCursor(
        pool,
        usable_rows,
        args.shuffle_block_size,
        args.seed + 101,
        cycle=pool_cycle,
        progress=pool_progress,
    )
    val_params_np = exp23.collect_validation_rows(
        val_pool, val_usable_rows, args.stage, args.val_samples, args.seed + 7001
    )

    print("=" * 108)
    print("Exp23 FORMAL spectrum curriculum")
    print(f"stage                 : {args.stage} - {spec['description']}")
    print(f"free parameters       : {[exp23.PARAMETER_NAMES[i] for i in free_idx]}")
    print(f"network               : {args.model_type} (same architecture in every stage)")
    print(f"loss                  : {args.loss_profile}, normalization={args.loss_normalization}")
    print(f"training pool         : {Path(args.pool_dir)} ({usable_rows:,} rows)")
    print(f"validation pool       : {Path(args.val_pool_dir)} ({val_usable_rows:,} rows, separate)")
    print(f"selection             : {args.selection_strategy}")
    print(f"coverage bins         : {coverage_bins} per free parameter, continuous values")
    print(f"soft g-gap target     : {args.target_g_gap_percent:g}%")
    print(f"training noise        : {100*args.noise_level:g}% RMS")
    print("g is NEVER edited manually; every g is generated by the unchanged physical forward model.")
    print("=" * 108)

    rng = np.random.default_rng(args.seed + 555)
    selection_rows: list[dict[str, Any]] = []
    g_min_history: list[float] = []
    g_med_history: list[float] = []
    target_history: list[float] = []
    coverage_history: list[float] = []
    joint_batch_history: list[float] = []
    start_time = time.time()

    model.train()
    for step in range(start_step + 1, args.max_steps + 1):
        candidate_n = max(args.batch_size, args.batch_size * args.candidate_multiplier)
        params_np = exp23.collect_candidates(cursor, args.stage, candidate_n)
        params = torch.from_numpy(params_np).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            target_candidates, g_candidates = physics.make_clean_batch(params)

        indices, diag = exp23.select_batch(
            params,
            g_candidates,
            stage=args.stage,
            free_idx=free_idx,
            batch_size=args.batch_size,
            bins=coverage_bins,
            strategy=args.selection_strategy,
            target_gap_fraction=args.target_g_gap_percent / 100.0,
            global_bin_counts=global_bin_counts,
            global_joint_counts=global_joint_counts,
            rng=rng,
        )
        target = target_candidates[indices].unsqueeze(1)
        g_clean = g_candidates[indices]
        g_noisy = physics.add_noise(g_clean).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        logs_acc = {"total": 0.0, "data_mse": 0.0, "data_huber": 0.0, "grad": 0.0, "physics": 0.0, "smooth": 0.0, "tv": 0.0, "tikhonov": 0.0}
        micro = min(args.micro_batch_size, args.batch_size)
        for ms in range(0, args.batch_size, micro):
            me = min(ms + micro, args.batch_size)
            weight = float(me - ms) / float(args.batch_size)
            with amp_autocast(use_amp):
                pred = normalize_prediction(model(g_noisy[ms:me]), target[ms:me])
                micro_loss, micro_logs = criterion(pred, target[ms:me], g_clean[ms:me].unsqueeze(1))
            if not torch.isfinite(micro_loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            scaler.scale(micro_loss * weight).backward()
            loss_value += float(micro_loss.detach().cpu()) * weight
            for key in logs_acc:
                logs_acc[key] += float(micro_logs.get(key, 0.0)) * weight

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        coverage_mean = float(np.mean([v["coverage_fraction"] for v in diag["coverage"].values()]))
        g_min_history.append(float(diag["batch_min_g_gap_percent"]))
        g_med_history.append(float(diag["batch_median_nearest_g_gap_percent"]))
        target_history.append(float(diag["target_met"]))
        coverage_history.append(coverage_mean)
        joint_batch_history.append(float(diag["unique_joint_fraction_of_batch"]))

        if step == 1 or step % args.log_every_steps == 0:
            print(
                f"step {step:6d} | loss {loss_value:.6g} | data {logs_acc['data_mse']:.3g} | phys {logs_acc['physics']:.3g} | "
                f"g-min {diag['batch_min_g_gap_percent']:.4g}% | g-nnmed {diag['batch_median_nearest_g_gap_percent']:.4g}% | "
                f"coverage {100*coverage_mean:.1f}% | joint-unique {100*diag['unique_joint_fraction_of_batch']:.1f}% | "
                f"target {'Y' if diag['target_met'] else 'N'}"
            )
            selection_rows.append({
                "step": step,
                "loss": loss_value,
                "batch_min_g_gap_percent": diag["batch_min_g_gap_percent"],
                "batch_median_nearest_g_gap_percent": diag["batch_median_nearest_g_gap_percent"],
                "target_g_gap_percent": diag["target_g_gap_percent"],
                "target_met": int(diag["target_met"]),
                "coverage_mean_fraction": coverage_mean,
                "unique_joint_fraction_of_batch": diag["unique_joint_fraction_of_batch"],
                "global_joint_bins_coverage_fraction": diag["global_joint_bins_coverage_fraction"],
                "pool_rows_read": cursor.rows_read,
            })

        if step % args.validation_every_steps == 0 or step == args.max_steps:
            val0 = evaluate(model, physics, criterion, val_params_np, batch_size=args.val_batch_size, noise_level=0.0, seed=args.seed + 8001)
            valn = evaluate(model, physics, criterion, val_params_np, batch_size=args.val_batch_size, noise_level=args.noise_level, seed=args.seed + 8002)
            score = float(valn["resonance_window_relative_l2_mean"])
            print(
                f"  VAL step {step}: f-rel={valn['f_relative_l2_mean']:.4g} | "
                f"res-window={valn['resonance_window_relative_l2_mean']:.4g} | "
                f"peak-height={valn['resonance_peak_height_relative_error_mean']:.4g} | "
                f"g-rel={valn['physics_g_relative_l2_mean']:.4g}"
            )
            if score < best_score:
                best_score = score
                atomic_torch_save(model.state_dict(), best_path)
                atomic_json_save({
                    "step": step,
                    "stage": args.stage,
                    "model_type": args.model_type,
                    "selection_strategy": args.selection_strategy,
                    "best_score": best_score,
                    "validation_0pct": val0,
                    "validation_train_noise": valn,
                }, best_info_path)
            model.train()

        if step % args.checkpoint_every_steps == 0 or step == args.max_steps:
            payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "step": step,
                "pool_cycle": cursor.cycle,
                "pool_progress": cursor.progress,
                "global_bin_counts": global_bin_counts.detach().cpu(),
                "global_joint_counts": global_joint_counts.detach().cpu(),
                "best_score": best_score,
                "args": vars(args),
            }
            atomic_torch_save(payload, latest_path)
            write_selection_csv(out_dir / "selection_diagnostics.csv", selection_rows)

    summary = {
        "stage": args.stage,
        "description": spec["description"],
        "model_type": args.model_type,
        "selection_strategy": args.selection_strategy,
        "pool_dir": str(Path(args.pool_dir)),
        "val_pool_dir": str(Path(args.val_pool_dir)),
        "pool_usable_rows": int(usable_rows),
        "val_pool_usable_rows": int(val_usable_rows),
        "free_parameters": [exp23.PARAMETER_NAMES[i] for i in free_idx],
        "coverage_bins": coverage_bins,
        "candidate_multiplier": args.candidate_multiplier,
        "soft_target_g_gap_percent": args.target_g_gap_percent,
        "training_noise_level": args.noise_level,
        "selected_training_samples": int(args.max_steps * args.batch_size),
        "pool_rows_read_this_run": int(cursor.rows_read),
        "mean_batch_min_g_gap_percent": float(np.mean(g_min_history)),
        "median_batch_min_g_gap_percent": float(np.median(g_min_history)),
        "mean_batch_median_nearest_g_gap_percent": float(np.mean(g_med_history)),
        "target_met_fraction": float(np.mean(target_history)),
        "mean_parameter_bin_coverage_fraction": float(np.mean(coverage_history)),
        "mean_unique_joint_fraction_of_batch": float(np.mean(joint_batch_history)),
        "final_global_joint_bins_coverage_fraction": float(torch.mean((global_joint_counts > 0).float()).item()),
        "best_validation_score": float(best_score),
        "elapsed_seconds": float(time.time() - start_time),
        "loss_profile": args.loss_profile,
        "important": "Formal g->f network; large-pool continuous values; coverage-first soft g-gap selection; separate unfiltered validation pool.",
    }
    atomic_json_save(summary, out_dir / "training_summary.json")
    write_selection_csv(out_dir / "selection_diagnostics.csv", selection_rows)

    print("=" * 108)
    print("Exp23 formal spectrum run finished")
    print(f"best model      : {best_path}")
    print(f"best validation : {best_info_path}")
    print(f"summary         : {out_dir / 'training_summary.json'}")
    print("=" * 108)


if __name__ == "__main__":
    main()
