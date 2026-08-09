#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp17: TCN nuisance prediction + exact-physics gamma profiling.

Why this experiment
-------------------
Exp12-Exp16 established that directly regressing all five parameters is a poor
formulation for narrow gamma.  Exp17 changes the inverse algorithm WITHOUT
changing the physical formula:

    g(q^2)
      -> TCN
      -> [a1, a2, a3, m]
      -> exact physical forward scan over gamma
      -> gamma_profile / best gamma
      -> reconstructed f(s)

The TCN never predicts gamma directly.

Key controls
------------
A. "oracle_nuisance":
   gamma profiling uses the true [a1,a2,a3,m].
   This is the physical upper bound under the chosen noise level.

B. "pred_a1_only":
   only a1 is replaced by the TCN prediction; a2,a3,m remain true.
   This isolates how much a1 prediction error hurts gamma.

C. "true_a1_pred_rest":
   a1 is true; a2,a3,m come from the TCN.
   This tests whether the other nuisance parameters are limiting.

D. "pred_all":
   all [a1,a2,a3,m] come from the TCN.
   This is the deployable hybrid pipeline.

Training distribution
---------------------
Parameters are generated online using the SAME current five-parameter physical
family.  No 160M parameter file is required.

For this targeted peak experiment:
- a1 is restricted to a1 >= 0.05 by default, because gamma is not a meaningful
  observable when the resonance strength tends to zero.
- gamma is sampled log-uniformly by default so narrow widths are not almost
  absent from training.
These change the TRAINING DISTRIBUTION, not the physical forward formula.

Input scaling
-------------
Per-sample RMS token normalization is NOT used.  A single global scale is used
for every sample (Exp11).  If --global-input-scale <= 0, the script estimates
the scale once from a calibration sample and then keeps it fixed.

Best checkpoint
---------------
The best model is selected by a fixed validation nuisance-parameter metric,
not by rolling training loss.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from TCNFourParameterInverse import (
    NUISANCE_NAMES,
    TCNFourParameterInverse1D,
)
from mc_online_physics import OnlinePhysics, fixed_rms_noise_numpy
from mc_parametric import normalize_parameters
from mc_physics import valid_parameter_mask_numpy
from train_mc_parameter_pool_transformer_loss import (
    amp_autocast,
    atomic_json_save,
    atomic_torch_save,
    choose_device,
    make_grad_scaler,
    set_random_seed,
)


PARAMETER_NAMES = ("a1", "a2", "a3", "m", "gamma")
NUISANCE_INDICES = (0, 1, 2, 3)

LOWER = np.array(
    [
        0.0,
        0.0,
        -0.05,
        np.nextafter(0.0, 1.0),
        np.nextafter(0.0, 1.0),
    ],
    dtype=np.float64,
)
UPPER = np.array([0.2, 0.05, 0.05, 2.0, 1.0], dtype=np.float64)


def parse_float_list(text: str) -> List[float]:
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    if not out:
        raise ValueError("empty float list")
    return out


def parse_int_list(text: str) -> List[int]:
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise ValueError("empty int list")
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Exp17 TCN nuisance prediction + physics gamma profile"
    )

    p.add_argument(
        "--checkpoint-dir",
        default=r".\checkpoints\exp17_tcn_physics_profile",
    )

    # Physics
    p.add_argument("--input-points", type=int, default=100)
    p.add_argument("--output-points", type=int, default=1000)
    p.add_argument("--integration-points", type=int, default=128)
    p.add_argument("--noise-level", type=float, default=0.002)  # 0.2%
    p.add_argument("--data-scale", type=float, default=160000.0)
    p.add_argument("--shift", type=float, default=400.0)
    p.add_argument("--s-min", type=float, default=0.1764)
    p.add_argument("--s-max", type=float, default=6.0)
    p.add_argument("--q2-min", type=float, default=-100.0)
    p.add_argument("--q2-max", type=float, default=-6.0)
    p.add_argument(
        "--physics-dtype",
        choices=("float32", "float64"),
        default="float32",
    )

    # Targeted training distribution.
    p.add_argument("--a1-train-min", type=float, default=0.05)
    p.add_argument("--a1-train-max", type=float, default=0.20)
    p.add_argument("--m-train-min", type=float, default=1e-4)
    p.add_argument("--m-train-max", type=float, default=2.0)
    p.add_argument("--gamma-train-min", type=float, default=0.001)
    p.add_argument("--gamma-train-max", type=float, default=1.0)
    p.add_argument(
        "--gamma-sampling",
        choices=("log", "uniform"),
        default="log",
    )

    # TCN
    p.add_argument("--tcn-channels", type=int, default=64)
    p.add_argument("--tcn-dilations", default="1,2,4,8,16")
    p.add_argument("--tcn-kernel-size", type=int, default=3)
    p.add_argument("--tcn-dropout", type=float, default=0.05)
    p.add_argument("--tcn-head-hidden", type=int, default=128)
    p.add_argument(
        "--global-input-scale",
        type=float,
        default=0.0,
        help="<=0 means estimate once from calibration data",
    )
    p.add_argument("--scale-calibration-samples", type=int, default=4096)

    # Nuisance loss.
    p.add_argument(
        "--nuisance-loss-weights",
        default="3,1,1,2",
        help="weights for [a1,a2,a3,m]; a1 and m are emphasized",
    )

    # Training.
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--log-every-steps", type=int, default=100)
    p.add_argument("--validate-every-steps", type=int, default=2000)
    p.add_argument("--checkpoint-every-steps", type=int, default=2000)

    # Fixed nuisance validation for best-model selection.
    p.add_argument("--val-samples", type=int, default=2000)
    p.add_argument("--val-batch-size", type=int, default=256)

    # Expensive final gamma-profile validation.
    p.add_argument(
        "--profile-true-gammas",
        default="0.01,0.05,0.2,0.5",
    )
    p.add_argument("--profile-samples-per-gamma", type=int, default=250)
    p.add_argument("--profile-batch-size", type=int, default=32)
    p.add_argument("--profile-gamma-min", type=float, default=0.001)
    p.add_argument("--profile-gamma-max", type=float, default=1.0)
    p.add_argument("--profile-gamma-points", type=int, default=241)
    p.add_argument("--profile-gamma-chunk", type=int, default=24)

    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "xpu"),
        default="auto",
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument("--deterministic", action="store_true")

    return p.parse_args()


def validate_args(args):
    if not (0 <= args.noise_level):
        raise ValueError("noise-level must be >= 0")
    if not (0 < args.a1_train_min < args.a1_train_max <= 0.2):
        raise ValueError("invalid a1 training range")
    if not (0 < args.m_train_min < args.m_train_max <= 2.0):
        raise ValueError("invalid m training range")
    if not (
        0 < args.gamma_train_min < args.gamma_train_max <= 1.0
    ):
        raise ValueError("invalid gamma training range")
    if args.tcn_kernel_size % 2 != 1:
        raise ValueError("TCN kernel size must be odd")
    if args.global_input_scale < 0:
        raise ValueError("global-input-scale must be >= 0")
    if args.profile_gamma_chunk <= 0:
        raise ValueError("profile-gamma-chunk must be > 0")


def sample_valid_parameters(
    rng: np.random.Generator,
    args,
    n: int,
    *,
    fixed_gamma: float | None = None,
) -> np.ndarray:
    accepted = []
    total = 0

    while total < n:
        need = n - total
        candidate_n = max(2 * need, 1024)

        p = np.empty((candidate_n, 5), dtype=np.float64)
        p[:, 0] = rng.uniform(
            args.a1_train_min,
            args.a1_train_max,
            candidate_n,
        )
        p[:, 1] = rng.uniform(LOWER[1], UPPER[1], candidate_n)
        p[:, 2] = rng.uniform(LOWER[2], UPPER[2], candidate_n)
        p[:, 3] = rng.uniform(
            args.m_train_min,
            args.m_train_max,
            candidate_n,
        )

        if fixed_gamma is not None:
            p[:, 4] = float(fixed_gamma)
        elif args.gamma_sampling == "log":
            log_g = rng.uniform(
                math.log(args.gamma_train_min),
                math.log(args.gamma_train_max),
                candidate_n,
            )
            p[:, 4] = np.exp(log_g)
        else:
            p[:, 4] = rng.uniform(
                args.gamma_train_min,
                args.gamma_train_max,
                candidate_n,
            )

        valid = valid_parameter_mask_numpy(p)
        good = p[valid]
        if len(good):
            take = min(need, len(good))
            accepted.append(good[:take])
            total += take

    return np.concatenate(accepted, axis=0).astype(np.float32, copy=False)


def make_model(args, device, global_input_scale: float):
    return TCNFourParameterInverse1D(
        input_length=args.input_points,
        channels=args.tcn_channels,
        dilations=parse_int_list(args.tcn_dilations),
        kernel_size=args.tcn_kernel_size,
        dropout=args.tcn_dropout,
        head_hidden=args.tcn_head_hidden,
        q2_min=args.q2_min,
        q2_max=args.q2_max,
        global_input_scale=global_input_scale,
    ).to(device)


def nuisance_normalized(
    nuisance: torch.Tensor,
    model: TCNFourParameterInverse1D,
) -> torch.Tensor:
    lower = model.parameter_lower.to(
        device=nuisance.device, dtype=nuisance.dtype
    )
    upper = model.parameter_upper.to(
        device=nuisance.device, dtype=nuisance.dtype
    )
    return (nuisance - lower) / (upper - lower)


def nuisance_loss(
    predicted: torch.Tensor,
    truth5: torch.Tensor,
    model: TCNFourParameterInverse1D,
    weights: torch.Tensor,
) -> torch.Tensor:
    truth = truth5[:, :4]
    pred_n = nuisance_normalized(predicted, model)
    true_n = nuisance_normalized(truth, model)
    w = weights.to(device=predicted.device, dtype=predicted.dtype).view(1, 4)
    return torch.mean(
        torch.sum((pred_n - true_n).square() * w, dim=1)
        / torch.sum(w).clamp_min(1e-12)
    )


@torch.no_grad()
def estimate_global_scale(
    physics: OnlinePhysics,
    args,
    rng: np.random.Generator,
) -> float:
    params_np = sample_valid_parameters(
        rng,
        args,
        args.scale_calibration_samples,
    )
    rms_all = []
    for start in range(0, len(params_np), args.val_batch_size):
        stop = min(start + args.val_batch_size, len(params_np))
        truth = torch.from_numpy(params_np[start:stop]).to(
            device=physics.device, dtype=torch.float32
        )
        _, g_clean = physics.make_clean_batch(truth)
        rms = torch.sqrt(torch.mean(g_clean.square(), dim=1))
        rms_all.append(rms.cpu().numpy())
    values = np.concatenate(rms_all)
    scale = float(np.median(values))
    if not np.isfinite(scale) or scale <= 0:
        raise FloatingPointError("failed to estimate positive global input scale")
    return scale


def fixed_validation_set(
    physics: OnlinePhysics,
    args,
    rng: np.random.Generator,
    *,
    samples: int,
    noise_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    params_np = sample_valid_parameters(rng, args, samples)
    clean_parts = []
    for start in range(0, len(params_np), args.val_batch_size):
        stop = min(start + args.val_batch_size, len(params_np))
        truth = torch.from_numpy(params_np[start:stop]).to(
            device=physics.device, dtype=torch.float32
        )
        _, g_clean = physics.make_clean_batch(truth)
        clean_parts.append(g_clean.cpu().numpy())
    g_clean_np = np.concatenate(clean_parts, axis=0)
    if args.noise_level > 0:
        g_input_np = fixed_rms_noise_numpy(
            g_clean_np,
            args.noise_level,
            noise_seed,
        )
    else:
        g_input_np = g_clean_np.astype(np.float32, copy=True)
    return params_np, g_input_np


@torch.no_grad()
def evaluate_nuisance(
    model,
    params_np,
    g_input_np,
    args,
    weights,
) -> Dict[str, float]:
    model.eval()

    sum_abs = np.zeros(4, dtype=np.float64)
    sum_sq_weighted = 0.0
    count = 0

    lower = model.parameter_lower.detach().cpu().numpy()
    upper = model.parameter_upper.detach().cpu().numpy()
    ranges = upper - lower
    w_np = weights.detach().cpu().numpy()

    for start in range(0, len(params_np), args.val_batch_size):
        stop = min(start + args.val_batch_size, len(params_np))

        truth = params_np[start:stop, :4].astype(np.float64)
        g = torch.from_numpy(g_input_np[start:stop]).to(
            device=next(model.parameters()).device,
            dtype=torch.float32,
        )
        pred = model.predict_nuisance(g.unsqueeze(1))
        pred_np = pred.cpu().numpy().astype(np.float64)

        diff_n = (pred_np - truth) / ranges.reshape(1, 4)
        sum_abs += np.abs(diff_n).sum(axis=0)
        sum_sq_weighted += np.sum(
            np.sum(diff_n**2 * w_np.reshape(1, 4), axis=1)
            / np.sum(w_np)
        )
        count += len(truth)

    mae = sum_abs / max(count, 1)
    return {
        "samples": int(count),
        "weighted_normalized_mse": float(
            sum_sq_weighted / max(count, 1)
        ),
        "normalized_abs_mean": {
            name: float(mae[i])
            for i, name in enumerate(NUISANCE_NAMES)
        },
        "normalized_abs_mean_all": float(np.mean(mae)),
    }


def build_gamma_grid(args, anchors: Sequence[float]) -> np.ndarray:
    grid = np.logspace(
        math.log10(args.profile_gamma_min),
        math.log10(args.profile_gamma_max),
        args.profile_gamma_points,
        dtype=np.float64,
    )
    grid = np.unique(
        np.concatenate([grid, np.asarray(anchors, dtype=np.float64)])
    )
    grid.sort()
    return grid.astype(np.float32)


@torch.no_grad()
def profile_gamma_batch(
    physics: OnlinePhysics,
    g_observed: torch.Tensor,
    base_params: torch.Tensor,
    gamma_grid: torch.Tensor,
    gamma_chunk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Profile gamma while holding the first four parameters fixed.

    Returns:
        best_gamma [B]
        best_relative_l2 [B]
    """
    if g_observed.ndim != 2:
        raise ValueError("g_observed must have shape [B,Nq]")
    if base_params.ndim != 2 or base_params.shape[1] != 5:
        raise ValueError("base_params must have shape [B,5]")

    batch = len(base_params)
    best_error = torch.full(
        (batch,),
        float("inf"),
        dtype=torch.float32,
        device=physics.device,
    )
    best_gamma = torch.full(
        (batch,),
        float("nan"),
        dtype=torch.float32,
        device=physics.device,
    )

    denom = torch.linalg.vector_norm(
        g_observed, dim=1
    ).clamp_min(1e-12)

    for start in range(0, len(gamma_grid), int(gamma_chunk)):
        gvals = gamma_grid[start:start + int(gamma_chunk)]
        k = len(gvals)

        candidates = base_params[:, None, :].expand(
            batch, k, 5
        ).clone()
        candidates[:, :, 4] = gvals.view(1, k)
        flat = candidates.reshape(batch * k, 5)

        g_pred = physics.forward_from_parameters(flat).reshape(
            batch, k, -1
        )
        rel = torch.linalg.vector_norm(
            g_pred - g_observed[:, None, :],
            dim=2,
        ) / denom[:, None]

        chunk_error, chunk_idx = torch.min(rel, dim=1)
        improve = chunk_error < best_error
        if torch.any(improve):
            selected_gamma = gvals[chunk_idx]
            best_error = torch.where(
                improve, chunk_error, best_error
            )
            best_gamma = torch.where(
                improve, selected_gamma, best_gamma
            )

    return best_gamma, best_error


def factor_error(pred_gamma: np.ndarray, true_gamma: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(pred_gamma, dtype=np.float64), 1e-12)
    t = np.maximum(np.asarray(true_gamma, dtype=np.float64), 1e-12)
    return np.maximum(p / t, t / p)


def make_regime_params(
    truth: torch.Tensor,
    pred4: torch.Tensor,
    regime: str,
) -> torch.Tensor:
    base = truth.clone()
    if regime == "oracle_nuisance":
        return base
    if regime == "pred_a1_only":
        base[:, 0] = pred4[:, 0]
        return base
    if regime == "true_a1_pred_rest":
        base[:, 1:4] = pred4[:, 1:4]
        return base
    if regime == "pred_all":
        base[:, :4] = pred4
        return base
    raise ValueError(f"unknown regime: {regime}")


@torch.no_grad()
def evaluate_gamma_profile(
    model,
    physics,
    args,
    *,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    model.eval()

    anchors = parse_float_list(args.profile_true_gammas)
    gamma_grid_np = build_gamma_grid(args, anchors)
    gamma_grid = torch.from_numpy(gamma_grid_np).to(
        device=physics.device, dtype=torch.float32
    )

    regimes = (
        "oracle_nuisance",
        "pred_a1_only",
        "true_a1_pred_rest",
        "pred_all",
    )

    rng = np.random.default_rng(seed)
    summary_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []

    for anchor_index, true_gamma in enumerate(anchors):
        params_np = sample_valid_parameters(
            rng,
            args,
            args.profile_samples_per_gamma,
            fixed_gamma=true_gamma,
        )

        # Compute clean g once.
        clean_parts = []
        for start in range(0, len(params_np), args.profile_batch_size):
            stop = min(
                start + args.profile_batch_size,
                len(params_np),
            )
            truth = torch.from_numpy(params_np[start:stop]).to(
                device=physics.device, dtype=torch.float32
            )
            _, g_clean = physics.make_clean_batch(truth)
            clean_parts.append(g_clean.cpu().numpy())
        g_clean_np = np.concatenate(clean_parts, axis=0)

        if args.noise_level > 0:
            g_obs_np = fixed_rms_noise_numpy(
                g_clean_np,
                args.noise_level,
                seed + 1000 + anchor_index,
            )
        else:
            g_obs_np = g_clean_np.astype(np.float32, copy=True)

        regime_gamma = {name: [] for name in regimes}
        regime_fit = {name: [] for name in regimes}
        regime_frel = {name: [] for name in regimes}

        # Track nuisance prediction error on this gamma slice.
        nuisance_abs_sum = np.zeros(4, dtype=np.float64)
        slice_count = 0

        for start in range(0, len(params_np), args.profile_batch_size):
            stop = min(
                start + args.profile_batch_size,
                len(params_np),
            )

            truth = torch.from_numpy(params_np[start:stop]).to(
                device=physics.device, dtype=torch.float32
            )
            g_obs = torch.from_numpy(g_obs_np[start:stop]).to(
                device=physics.device, dtype=torch.float32
            )

            pred4 = model.predict_nuisance(g_obs.unsqueeze(1))

            lower = model.parameter_lower.to(
                device=pred4.device, dtype=pred4.dtype
            )
            upper = model.parameter_upper.to(
                device=pred4.device, dtype=pred4.dtype
            )
            pred4_n = (pred4 - lower) / (upper - lower)
            true4_n = (truth[:, :4] - lower) / (upper - lower)
            nuisance_abs_sum += torch.abs(
                pred4_n - true4_n
            ).sum(dim=0).cpu().numpy()
            slice_count += len(truth)

            f_true, _, _ = physics.components(truth)

            for regime in regimes:
                base = make_regime_params(truth, pred4, regime)
                best_gamma, best_fit = profile_gamma_batch(
                    physics,
                    g_obs,
                    base,
                    gamma_grid,
                    args.profile_gamma_chunk,
                )

                final_params = base.clone()
                final_params[:, 4] = best_gamma
                f_pred, _, _ = physics.components(final_params)
                f_rel = (
                    torch.linalg.vector_norm(
                        f_pred - f_true, dim=1
                    )
                    / torch.linalg.vector_norm(
                        f_true, dim=1
                    ).clamp_min(1e-12)
                )

                bg = best_gamma.cpu().numpy()
                bf = best_fit.cpu().numpy()
                fr = f_rel.cpu().numpy()

                regime_gamma[regime].append(bg)
                regime_fit[regime].append(bf)
                regime_frel[regime].append(fr)

                true_np = params_np[start:stop, 4]
                a1_true_np = params_np[start:stop, 0]
                a1_pred_np = pred4[:, 0].cpu().numpy()

                for j in range(len(bg)):
                    sample_rows.append(
                        {
                            "true_gamma": float(true_np[j]),
                            "regime": regime,
                            "pred_gamma": float(bg[j]),
                            "gamma_factor_error": float(
                                factor_error(
                                    np.array([bg[j]]),
                                    np.array([true_np[j]]),
                                )[0]
                            ),
                            "g_profile_relative_l2": float(bf[j]),
                            "f_relative_l2": float(fr[j]),
                            "true_a1": float(a1_true_np[j]),
                            "pred_a1": float(a1_pred_np[j]),
                        }
                    )

        nuisance_mae = nuisance_abs_sum / max(slice_count, 1)

        for regime in regimes:
            pred_gamma = np.concatenate(regime_gamma[regime])
            fit = np.concatenate(regime_fit[regime])
            frel = np.concatenate(regime_frel[regime])
            truth_gamma = np.full(
                len(pred_gamma), float(true_gamma), dtype=np.float64
            )
            ferr = factor_error(pred_gamma, truth_gamma)

            summary_rows.append(
                {
                    "true_gamma": float(true_gamma),
                    "regime": regime,
                    "samples": int(len(pred_gamma)),
                    "median_gamma_factor_error": float(np.median(ferr)),
                    "p90_gamma_factor_error": float(np.quantile(ferr, 0.90)),
                    "within_factor_1p2": float(np.mean(ferr <= 1.2)),
                    "within_factor_1p5": float(np.mean(ferr <= 1.5)),
                    "within_factor_2": float(np.mean(ferr <= 2.0)),
                    "median_g_profile_relative_l2": float(np.median(fit)),
                    "mean_g_profile_relative_l2": float(np.mean(fit)),
                    "median_f_relative_l2": float(np.median(frel)),
                    "mean_f_relative_l2": float(np.mean(frel)),
                    "nuisance_a1_norm_abs_mean": float(nuisance_mae[0]),
                    "nuisance_a2_norm_abs_mean": float(nuisance_mae[1]),
                    "nuisance_a3_norm_abs_mean": float(nuisance_mae[2]),
                    "nuisance_m_norm_abs_mean": float(nuisance_mae[3]),
                }
            )

    return summary_rows, sample_rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_model_state(path: Path):
    try:
        obj = torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def print_profile_summary(rows):
    print("-" * 100)
    print("Gamma profiling summary (BEST checkpoint)")
    print(
        "true_gamma | regime              | med_factor | <=x2   | "
        "med_g_fit | med_f"
    )
    print("-" * 100)
    for row in rows:
        print(
            f"{row['true_gamma']:10.4g} | "
            f"{row['regime']:19s} | "
            f"{row['median_gamma_factor_error']:10.4f} | "
            f"{row['within_factor_2']:6.3f} | "
            f"{row['median_g_profile_relative_l2']:9.4g} | "
            f"{row['median_f_relative_l2']:7.4f}"
        )
    print("-" * 100)


def main():
    args = parse_args()
    validate_args(args)

    set_random_seed(args.seed, args.deterministic)
    device = choose_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_path = checkpoint_dir / "latest_checkpoint.pth"
    best_path = checkpoint_dir / "best_model.pth"

    physics = OnlinePhysics(args, device)
    rng = np.random.default_rng(args.seed + 17)

    if args.global_input_scale > 0:
        global_scale = float(args.global_input_scale)
    else:
        global_scale = estimate_global_scale(
            physics, args, rng
        )

    model = make_model(args, device, global_scale)

    weights_np = np.asarray(
        parse_float_list(args.nuisance_loss_weights),
        dtype=np.float32,
    )
    if weights_np.shape != (4,) or np.any(weights_np <= 0):
        raise ValueError(
            "--nuisance-loss-weights must contain 4 positive values"
        )
    weights = torch.from_numpy(weights_np).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = make_grad_scaler(use_amp)

    # Fixed validation set used for checkpoint selection.
    val_rng = np.random.default_rng(args.seed + 50001)
    val_params_np, val_g_np = fixed_validation_set(
        physics,
        args,
        val_rng,
        samples=args.val_samples,
        noise_seed=args.seed + 50002,
    )

    best_score = float("inf")
    best_metrics = None
    start_time = time.time()

    print("=" * 100)
    print("Exp17: TCN nuisance prediction + exact-physics gamma profiling")
    print("PHYSICAL FORWARD FORMULA: unchanged")
    print(f"device                 : {device}")
    print(f"noise                  : {100*args.noise_level:.6g}%")
    print(f"training a1            : [{args.a1_train_min}, {args.a1_train_max}]")
    print(
        f"training gamma         : [{args.gamma_train_min}, "
        f"{args.gamma_train_max}], {args.gamma_sampling}"
    )
    print(f"global input scale     : {global_scale:.8g}")
    print(f"TCN dilations          : {parse_int_list(args.tcn_dilations)}")
    print(
        f"estimated receptive RF : {model.receptive_field_estimate} "
        f"for {args.input_points} input points"
    )
    print(f"nuisance weights       : {weights_np.tolist()} for [a1,a2,a3,m]")
    print("gamma direct loss      : NONE")
    print("best checkpoint metric : fixed validation nuisance weighted MSE")
    print("=" * 100)

    model.train()

    for step in range(1, args.max_steps + 1):
        params_np = sample_valid_parameters(
            rng,
            args,
            args.batch_size,
        )
        truth = torch.from_numpy(params_np).to(
            device=device, dtype=torch.float32
        )

        with torch.no_grad():
            _, g_clean = physics.make_clean_batch(truth)
            if args.noise_level > 0:
                g_input = physics.add_noise(g_clean)
            else:
                g_input = g_clean

        optimizer.zero_grad(set_to_none=True)

        with amp_autocast(use_amp):
            pred4 = model.predict_nuisance(
                g_input.unsqueeze(1)
            )
            loss = nuisance_loss(
                pred4,
                truth,
                model,
                weights,
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at step {step}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_every_steps == 0:
            with torch.no_grad():
                pred_n = nuisance_normalized(pred4, model)
                true_n = nuisance_normalized(truth[:, :4], model)
                mae = torch.mean(
                    torch.abs(pred_n - true_n), dim=0
                ).cpu().numpy()

            print(
                f"step={step:6d} "
                f"loss={float(loss.detach().cpu()):.6e} | "
                f"a1={mae[0]:.4f} "
                f"a2={mae[1]:.4f} "
                f"a3={mae[2]:.4f} "
                f"m={mae[3]:.4f}"
            )

        if step % args.validate_every_steps == 0:
            metrics = evaluate_nuisance(
                model,
                val_params_np,
                val_g_np,
                args,
                weights,
            )
            score = metrics["weighted_normalized_mse"]
            m = metrics["normalized_abs_mean"]

            print(
                f"[VAL] step={step:6d} score={score:.6e} | "
                f"a1={m['a1']:.5f} "
                f"a2={m['a2']:.5f} "
                f"a3={m['a3']:.5f} "
                f"m={m['m']:.5f}"
            )

            if score < best_score:
                best_score = float(score)
                best_metrics = dict(metrics)
                best_metrics["global_step"] = int(step)

                atomic_torch_save(
                    {
                        "global_step": step,
                        "global_input_scale": global_scale,
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "validation": metrics,
                    },
                    best_path,
                )
                atomic_json_save(
                    best_metrics,
                    checkpoint_dir / "validation_best_nuisance.json",
                )

        if step % args.checkpoint_every_steps == 0:
            atomic_torch_save(
                {
                    "global_step": step,
                    "global_input_scale": global_scale,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                latest_path,
            )

    atomic_torch_save(
        {
            "global_step": args.max_steps,
            "global_input_scale": global_scale,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        latest_path,
    )

    latest_metrics = evaluate_nuisance(
        model,
        val_params_np,
        val_g_np,
        args,
        weights,
    )
    latest_metrics["global_step"] = int(args.max_steps)
    atomic_json_save(
        latest_metrics,
        checkpoint_dir / "validation_latest_nuisance.json",
    )

    if not best_path.exists():
        atomic_torch_save(
            {
                "global_step": args.max_steps,
                "global_input_scale": global_scale,
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "validation": latest_metrics,
            },
            best_path,
        )
        best_metrics = latest_metrics

    # Reconstruct best model for the expensive gamma-profile evaluation.
    best_model = make_model(args, device, global_scale)
    best_model.load_state_dict(load_model_state(best_path))

    profile_rows, sample_rows = evaluate_gamma_profile(
        best_model,
        physics,
        args,
        seed=args.seed + 900001,
    )

    write_csv(
        checkpoint_dir / "exp17_gamma_profile_summary.csv",
        profile_rows,
    )
    write_csv(
        checkpoint_dir / "exp17_gamma_profile_samples.csv",
        sample_rows,
    )

    print_profile_summary(profile_rows)

    summary = {
        "physical_formula_changed": False,
        "algorithm": (
            "TCN predicts [a1,a2,a3,m]; exact OnlinePhysics profiles gamma"
        ),
        "noise_level": args.noise_level,
        "global_input_scale": global_scale,
        "training_gamma_sampling": args.gamma_sampling,
        "training_a1_min": args.a1_train_min,
        "best_nuisance": best_metrics,
        "latest_nuisance": latest_metrics,
        "gamma_profile_summary": profile_rows,
        "elapsed_seconds": time.time() - start_time,
    }
    atomic_json_save(
        summary,
        checkpoint_dir / "exp17_summary.json",
    )

    print("=" * 100)
    print("Exp17 finished")
    print(f"output: {checkpoint_dir}")
    print("")
    print("FIRST READ:")
    print("  validation_best_nuisance.json")
    print("  exp17_gamma_profile_summary.csv")
    print("")
    print("Most important comparison:")
    print("  oracle_nuisance  vs  pred_a1_only  vs  true_a1_pred_rest  vs  pred_all")
    print("=" * 100)


if __name__ == "__main__":
    main()
