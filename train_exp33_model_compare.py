#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp33: controlled MLP-vs-TCN comparison for continuous a1+gamma inversion.

Controlled variables:
- identical Exp32 identifiable dataset (alias cutoff stays fixed)
- identical train/val/test split and noise realizations
- identical input normalization
- identical targets: a1 and log10(gamma)
- identical regression loss
- identical optimizer/scheduler/early stopping
- identical train seed

Only the backbone changes: MLP vs TCN.

Continuous errors are the primary scientific metrics.  A secondary
"joint recovery rate" is also reported for the stage decision:
    |a1_hat-a1| <= --a1-success-tol
AND gamma factor error <= --gamma-success-factor
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from PairwiseInverseMLP import PairwiseInverseMLP
from PairwiseInverseTCN import PairwiseInverseTCN
from train_exp20_pairwise import (
    PairwiseNpzDataset,
    evaluate_loader,
    parse_hidden_sizes,
    predict_dataset,
    resolve_device,
    set_seed,
    train_one_epoch,
    training_input_normalization,
    write_csv,
)


def parse_int_tuple(text: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in str(text).split(",") if x.strip())
    if not vals or any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return vals


def parse_args():
    p = argparse.ArgumentParser(
        description="Exp33 fixed-data MLP vs TCN a1+gamma regression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=("mlp", "tcn"), required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--result-prefix", default="exp33")
    p.add_argument("--train-noise-dir", default="noise_0p2pct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260850)

    # Keep the training protocol aligned with Exp32.
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=1e-7)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--second-loss-weight", type=float, default=1.0)
    p.add_argument("--gamma-loss-weight", type=float, default=1.0)

    # MLP baseline: unchanged from Exp32.
    p.add_argument(
        "--mlp-hidden-sizes",
        type=parse_hidden_sizes,
        default=(256, 256, 128),
    )

    # TCN first-pass architecture.  These are intentionally fixed for the
    # architecture comparison rather than extensively tuned.
    p.add_argument("--tcn-channels", type=int, default=64)
    p.add_argument("--tcn-dilations", type=parse_int_tuple, default=(1, 2, 4, 8, 16))
    p.add_argument("--tcn-kernel-size", type=int, default=3)
    p.add_argument("--tcn-head-hidden", type=int, default=128)
    p.add_argument(
        "--tcn-no-coordinate-channel",
        action="store_true",
        help="ablation only; default TCN includes a normalized absolute position channel",
    )

    p.add_argument(
        "--parameter-bounds-source",
        choices=("selected", "original"),
        default="selected",
        help="Keep 'selected' to reproduce the Exp32 baseline exactly.",
    )

    # Secondary recovery-rate gate. Continuous errors remain the primary metrics.
    p.add_argument("--a1-success-tol", type=float, default=0.005)
    p.add_argument("--gamma-success-factor", type=float, default=1.20)

    p.add_argument("--physics-integration-points", type=int, default=512)
    p.add_argument("--skip-physics-metrics", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    return p.parse_args()


def noise_dirs_with_test(data_dir: Path):
    rows = []
    for d in Path(data_dir).glob("noise_*"):
        p = d / "test.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=False) as z:
            level = float(np.median(z["noise_level"]))
        rows.append((level, d))
    rows.sort(key=lambda x: x[0])
    return rows


def safe_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.quantile(x, q)) if len(x) else math.nan



def finite_summary(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    return (
        float(np.median(x)),
        float(np.quantile(x, 0.90)),
        float(np.quantile(x, 0.95)),
    )

def choose_bounds(metadata: dict, source: str):
    selected_a1 = [float(x) for x in metadata["a1_range"]]
    selected_gamma = [float(x) for x in metadata["gamma_range"]]
    original_a1 = [float(x) for x in metadata.get("original_a1_range", selected_a1)]
    original_gamma = [float(x) for x in metadata.get("original_gamma_range", selected_gamma)]
    if source == "original":
        model_a1, model_gamma = original_a1, original_gamma
    else:
        model_a1, model_gamma = selected_a1, selected_gamma
    return model_a1, model_gamma, original_a1, original_gamma


def regression_metrics(
    pred_a1: np.ndarray,
    true_a1: np.ndarray,
    pred_gamma: np.ndarray,
    true_gamma: np.ndarray,
    *,
    a1_reference_span: float,
    a1_success_tol: float,
    gamma_success_factor: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    pa = np.asarray(pred_a1, dtype=np.float64)
    ta = np.asarray(true_a1, dtype=np.float64)
    pg = np.clip(np.asarray(pred_gamma, dtype=np.float64), 1e-30, None)
    tg = np.clip(np.asarray(true_gamma, dtype=np.float64), 1e-30, None)

    a1_signed = pa - ta
    a1_abs = np.abs(a1_signed)
    a1_norm = a1_abs / max(float(a1_reference_span), 1e-30)

    gamma_log10_signed = np.log10(pg) - np.log10(tg)
    gamma_log10_abs = np.abs(gamma_log10_signed)
    gamma_rel_abs = np.abs(pg - tg) / tg
    gamma_factor = np.exp(np.abs(np.log(pg) - np.log(tg)))

    a1_ok = a1_abs <= float(a1_success_tol)
    gamma_ok = gamma_factor <= float(gamma_success_factor)
    joint_ok = a1_ok & gamma_ok

    metrics = {
        "count": int(len(pa)),
        "a1_mae": float(np.mean(a1_abs)),
        "a1_rmse": float(np.sqrt(np.mean(a1_signed**2))),
        "a1_median_abs_error": float(np.median(a1_abs)),
        "a1_p90_abs_error": safe_quantile(a1_abs, 0.90),
        "a1_p95_abs_error": safe_quantile(a1_abs, 0.95),
        "a1_max_abs_error": float(np.max(a1_abs)),
        "a1_nmae_original_span": float(np.mean(a1_norm)),
        "a1_p90_norm_error_original_span": safe_quantile(a1_norm, 0.90),
        "gamma_log10_mae": float(np.mean(gamma_log10_abs)),
        "gamma_log10_rmse": float(np.sqrt(np.mean(gamma_log10_signed**2))),
        "gamma_median_abs_log10_error": float(np.median(gamma_log10_abs)),
        "gamma_p90_abs_log10_error": safe_quantile(gamma_log10_abs, 0.90),
        "gamma_p95_abs_log10_error": safe_quantile(gamma_log10_abs, 0.95),
        "gamma_relative_mae": float(np.mean(gamma_rel_abs)),
        "gamma_median_relative_error": float(np.median(gamma_rel_abs)),
        "gamma_p90_relative_error": safe_quantile(gamma_rel_abs, 0.90),
        "gamma_p95_relative_error": safe_quantile(gamma_rel_abs, 0.95),
        "gamma_median_factor": float(np.median(gamma_factor)),
        "gamma_p90_factor": safe_quantile(gamma_factor, 0.90),
        "gamma_p95_factor": safe_quantile(gamma_factor, 0.95),
        "gamma_max_factor": float(np.max(gamma_factor)),
        "a1_success_tol": float(a1_success_tol),
        "gamma_success_factor": float(gamma_success_factor),
        "a1_recovery_rate": float(np.mean(a1_ok)),
        "gamma_recovery_rate": float(np.mean(gamma_ok)),
        "joint_recovery_rate": float(np.mean(joint_ok)),
    }
    arrays = {
        "a1_signed_error": a1_signed,
        "a1_abs_error": a1_abs,
        "a1_norm_abs_error_original_span": a1_norm,
        "gamma_signed_log10_error": gamma_log10_signed,
        "gamma_abs_log10_error": gamma_log10_abs,
        "gamma_relative_error": gamma_rel_abs,
        "gamma_factor_error": gamma_factor,
        "a1_recovered": a1_ok.astype(np.int8),
        "gamma_recovered": gamma_ok.astype(np.int8),
        "joint_recovered": joint_ok.astype(np.int8),
    }
    return metrics, arrays


def build_model(args, train_ds, model_a1, model_gamma, mean, scale, device):
    a1_min, a1_max = model_a1
    gamma_min, gamma_max = model_gamma

    if args.model == "mlp":
        model = PairwiseInverseMLP(
            input_points=train_ds.gy.shape[1],
            hidden_sizes=args.mlp_hidden_sizes,
            dropout=args.dropout,
            second_min=a1_min,
            second_max=a1_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        architecture = {
            "model": "mlp",
            "hidden_sizes": list(args.mlp_hidden_sizes),
            "dropout": float(args.dropout),
        }
    else:
        model = PairwiseInverseTCN(
            input_points=train_ds.gy.shape[1],
            channels=args.tcn_channels,
            dilations=args.tcn_dilations,
            kernel_size=args.tcn_kernel_size,
            dropout=args.dropout,
            head_hidden=args.tcn_head_hidden,
            second_min=a1_min,
            second_max=a1_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            use_coordinate_channel=not args.tcn_no_coordinate_channel,
        )
        architecture = {
            "model": "tcn",
            "channels": int(args.tcn_channels),
            "dilations": list(args.tcn_dilations),
            "kernel_size": int(args.tcn_kernel_size),
            "head_hidden": int(args.tcn_head_hidden),
            "dropout": float(args.dropout),
            "coordinate_channel": not bool(args.tcn_no_coordinate_channel),
            "receptive_field_estimate": int(model.receptive_field_estimate),
        }

    model.set_input_normalization(mean, scale)
    model = model.to(device)
    architecture["trainable_parameters"] = int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    return model, architecture


def main():
    args = parse_args()
    if args.a1_success_tol <= 0:
        raise ValueError("--a1-success-tol must be > 0")
    if args.gamma_success_factor <= 1:
        raise ValueError("--gamma-success-factor must be > 1")

    set_seed(args.seed)
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("mode") != "a1gamma":
        raise ValueError("Exp33 requires a1gamma data")

    model_a1, model_gamma, original_a1, original_gamma = choose_bounds(
        metadata, args.parameter_bounds_source
    )
    training_a1_span = model_a1[1] - model_a1[0]
    metric_a1_span = original_a1[1] - original_a1[0]

    train_ds = PairwiseNpzDataset(
        data_dir / args.train_noise_dir / "train.npz",
        mode="a1gamma",
        max_samples=args.max_train_samples,
    )
    val_ds = PairwiseNpzDataset(
        data_dir / args.train_noise_dir / "val.npz",
        mode="a1gamma",
        max_samples=args.max_val_samples,
    )
    mean, scale, global_rms = training_input_normalization(train_ds.gy)
    model, architecture = build_model(
        args, train_ds, model_a1, model_gamma, mean, scale, device
    )

    # Dedicated shuffle RNG keeps the sample order reproducible and independent
    # of how many random numbers a particular architecture consumes at init.
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(int(args.seed) + 330033)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=shuffle_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=max(args.lr * 1e-3, 1e-7)
    )

    print("=" * 100)
    print("Exp33 controlled model comparison")
    print("model                   : %s" % args.model.upper())
    print("data                    : %s" % data_dir)
    print("train / val samples     : %d / %d" % (len(train_ds), len(val_ds)))
    print("physical states         : %s" % metadata["selected_state_counts"])
    print("alias minimum RMS-SNR   : %s" % metadata.get("alias_min_rms_snr"))
    print("model-bound source      : %s" % args.parameter_bounds_source)
    print("model a1 bounds         : %s" % model_a1)
    print("model gamma bounds      : %s" % model_gamma)
    print("input global RMS        : %.10g" % global_rms)
    print("trainable parameters    : %d" % architecture["trainable_parameters"])
    print(
        "secondary recovery gate : |da1|<=%.6g AND gamma-factor<=%.4g"
        % (args.a1_success_tol, args.gamma_success_factor)
    )
    print("=" * 100)

    best_val = math.inf
    best_epoch = 0
    bad_epochs = 0
    checkpoint = output_dir / ("best_exp33_%s.pt" % args.model)
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            training_a1_span,
            args.second_loss_weight,
            args.gamma_loss_weight,
        )
        va, vm = evaluate_loader(
            model,
            val_loader,
            device,
            training_a1_span,
            args.second_loss_weight,
            args.gamma_loss_weight,
        )

        improved = va < best_val - args.min_delta
        if improved:
            best_val = va
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": va,
                    "architecture": architecture,
                },
                checkpoint,
            )
        else:
            bad_epochs += 1

        history.append(
            {
                "model": args.model,
                "epoch": epoch,
                "train_loss": tr,
                "val_loss": va,
                "gamma_log10_mae": vm["gamma_log10_mae"],
                "gamma_median_factor": vm["gamma_median_factor"],
                "gamma_p90_factor": vm["gamma_p90_factor"],
                "a1_median_abs_error": vm["second_median_abs_error"],
                "a1_p90_abs_error": vm["second_p90_abs_error"],
                "lr": optimizer.param_groups[0]["lr"],
                "is_best": int(improved),
            }
        )
        if improved or epoch % args.log_every == 0:
            mark = "*" if improved else " "
            print(
                "%s epoch %3d | train %.6g | val %.6g | gamma log10-MAE %.5f | "
                "gamma P90 factor %.4f | a1 P90 abs %.6f"
                % (
                    mark,
                    epoch,
                    tr,
                    va,
                    vm["gamma_log10_mae"],
                    vm["gamma_p90_factor"],
                    vm["second_p90_abs_error"],
                )
            )
        scheduler.step()
        if bad_epochs >= args.patience:
            print("early stop; best epoch=%d" % best_epoch)
            break

    write_csv(output_dir / "training_history.csv", history)

    try:
        ck = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        ck = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    summary_rows = []
    sample_rows = []
    for noise_level, noise_dir in noise_dirs_with_test(data_dir):
        test_ds = PairwiseNpzDataset(
            noise_dir / "test.npz",
            mode="a1gamma",
            max_samples=args.max_test_samples,
        )
        pred = predict_dataset(
            model,
            test_ds,
            mode="a1gamma",
            device=device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            compute_physics=not args.skip_physics_metrics,
            physics_integration_points=args.physics_integration_points,
        )

        rm, arrays = regression_metrics(
            pred["pred_second"],
            pred["true_second"],
            pred["pred_gamma"],
            pred["true_gamma"],
            a1_reference_span=metric_a1_span,
            a1_success_tol=args.a1_success_tol,
            gamma_success_factor=args.gamma_success_factor,
        )
        f_rel = np.asarray(pred["f_relative_l2"], dtype=np.float64)
        g_rel = np.asarray(pred["g_clean_relative_l2"], dtype=np.float64)
        f_med, f_p90, f_p95 = finite_summary(f_rel)
        g_med, g_p90, _ = finite_summary(g_rel)
        summary_rows.append(
            {
                "model": args.model,
                "noise_dir": noise_dir.name,
                "noise_level": float(noise_level),
                "split": "test_independent",
                "trainable_parameters": architecture["trainable_parameters"],
                **rm,
                "median_f_relative_l2": f_med,
                "p90_f_relative_l2": f_p90,
                "p95_f_relative_l2": f_p95,
                "median_g_clean_relative_l2": g_med,
                "p90_g_clean_relative_l2": g_p90,
            }
        )

        pa = np.asarray(pred["pred_second"], dtype=np.float64)
        ta = np.asarray(pred["true_second"], dtype=np.float64)
        pg = np.asarray(pred["pred_gamma"], dtype=np.float64)
        tg = np.asarray(pred["true_gamma"], dtype=np.float64)
        for i in range(len(tg)):
            sample_rows.append(
                {
                    "model": args.model,
                    "noise_dir": noise_dir.name,
                    "noise_level": float(noise_level),
                    "split": "test_independent",
                    "sample_index": int(i),
                    "true_a1": float(ta[i]),
                    "pred_a1": float(pa[i]),
                    "a1_signed_error": float(arrays["a1_signed_error"][i]),
                    "a1_abs_error": float(arrays["a1_abs_error"][i]),
                    "a1_norm_abs_error_original_span": float(
                        arrays["a1_norm_abs_error_original_span"][i]
                    ),
                    "true_gamma": float(tg[i]),
                    "pred_gamma": float(pg[i]),
                    "gamma_signed_log10_error": float(
                        arrays["gamma_signed_log10_error"][i]
                    ),
                    "gamma_abs_log10_error": float(
                        arrays["gamma_abs_log10_error"][i]
                    ),
                    "gamma_relative_error": float(arrays["gamma_relative_error"][i]),
                    "gamma_factor_error": float(arrays["gamma_factor_error"][i]),
                    "a1_recovered": int(arrays["a1_recovered"][i]),
                    "gamma_recovered": int(arrays["gamma_recovered"][i]),
                    "joint_recovered": int(arrays["joint_recovered"][i]),
                    "f_relative_l2": float(f_rel[i]),
                    "g_clean_relative_l2": float(g_rel[i]),
                }
            )

        print(
            "noise=%7.4g | a1 MAE=%.6g P90=%.6g | gamma P90-factor=%.4f | "
            "joint=%.2f%% | f P90=%.5f"
            % (
                noise_level,
                rm["a1_mae"],
                rm["a1_p90_abs_error"],
                rm["gamma_p90_factor"],
                100.0 * rm["joint_recovery_rate"],
                f_p90,
            )
        )

    prefix = str(args.result_prefix).strip() or "exp33"
    write_csv(output_dir / (prefix + "_summary.csv"), summary_rows)
    write_csv(output_dir / (prefix + "_samples.csv"), sample_rows)
    (output_dir / (prefix + "_metadata.json")).write_text(
        json.dumps(
            {
                "experiment": "Exp33 fixed identifiable data: MLP vs TCN",
                "controlled_change": "network backbone only",
                "model": args.model,
                "architecture": architecture,
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "primary_evaluation": {
                    "a1": "continuous MAE/RMSE/median/P90/P95 absolute error",
                    "gamma": "continuous log10/relative/factor error",
                    "physics": "f and clean-g relative L2",
                },
                "secondary_stage_gate": {
                    "a1_abs_error_max": float(args.a1_success_tol),
                    "gamma_factor_error_max": float(args.gamma_success_factor),
                    "metric": "joint_recovery_rate",
                },
                "model_bounds": {"a1": model_a1, "gamma": model_gamma},
                "metric_reference_bounds": {
                    "a1": original_a1,
                    "gamma": original_gamma,
                },
                "args": vars(args),
                "dataset_metadata": metadata,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("Exp33 %s finished" % args.model.upper())
    print("Read first:")
    print("  %s" % (output_dir / (prefix + "_summary.csv")))
    print("  %s" % (output_dir / (prefix + "_samples.csv")))
    print("=" * 100)


if __name__ == "__main__":
    main()
