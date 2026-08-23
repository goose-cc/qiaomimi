#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp34: improve a1 while preserving the successful Exp33 gamma predictor.

Strategy
--------
- Load the matched Exp33 TCN checkpoint.
- Add a zero-initialized amplitude-aware residual branch for a1.
- Freeze the entire Exp33 TCN path.
- Train ONLY the new residual branch on the same Exp32/33 identifiable data.

Therefore:
- gamma is intentionally unchanged from Exp33;
- any improvement in joint recovery comes from better a1 recovery;
- a1 is still a direct neural-network prediction.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from PairwiseInverseTCNA1Residual import PairwiseInverseTCNA1Residual
from train_exp20_pairwise import (
    PairwiseNpzDataset,
    predict_dataset,
    resolve_device,
    set_seed,
    training_input_normalization,
    write_csv,
)
from train_exp33_model_compare import (
    choose_bounds,
    finite_summary,
    noise_dirs_with_test,
    parse_int_tuple,
    regression_metrics,
)


def parse_hidden(text: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in str(text).split(",") if x.strip())
    if not vals or any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return vals


def parse_args():
    p = argparse.ArgumentParser(
        description="Exp34 frozen TCN + a1 amplitude residual correction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--result-prefix", default="exp34")
    p.add_argument("--train-noise-dir", default="noise_0p2pct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260850)

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-7)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=2)

    p.add_argument("--amplitude-bins", type=int, default=16)
    p.add_argument("--amplitude-hidden", type=parse_hidden, default=(96, 64))
    p.add_argument("--correction-limit", type=float, default=2.0)

    p.add_argument(
        "--parameter-bounds-source",
        choices=("selected", "original"),
        default="selected",
    )
    p.add_argument("--a1-success-tol", type=float, default=0.005)
    p.add_argument("--gamma-success-factor", type=float, default=1.20)

    p.add_argument("--physics-integration-points", type=int, default=512)
    p.add_argument("--skip-physics-metrics", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    return p.parse_args()


def load_checkpoint(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_from_exp33_checkpoint(args, train_ds, metadata, mean, scale, global_rms, device):
    ck = load_checkpoint(Path(args.base_checkpoint), "cpu")
    arch = dict(ck.get("architecture", {}))
    if str(arch.get("model", "tcn")).lower() != "tcn":
        raise ValueError("base checkpoint is not an Exp33 TCN checkpoint")

    model_a1, model_gamma, original_a1, original_gamma = choose_bounds(
        metadata, args.parameter_bounds_source
    )

    model = PairwiseInverseTCNA1Residual(
        input_points=train_ds.gy.shape[1],
        channels=int(arch.get("channels", 64)),
        dilations=tuple(int(x) for x in arch.get("dilations", [1, 2, 4, 8, 16])),
        kernel_size=int(arch.get("kernel_size", 3)),
        dropout=float(arch.get("dropout", 0.05)),
        head_hidden=int(arch.get("head_hidden", 128)),
        second_min=float(model_a1[0]),
        second_max=float(model_a1[1]),
        gamma_min=float(model_gamma[0]),
        gamma_max=float(model_gamma[1]),
        use_coordinate_channel=bool(arch.get("coordinate_channel", True)),
        amplitude_bins=args.amplitude_bins,
        amplitude_hidden=args.amplitude_hidden,
        correction_limit=args.correction_limit,
    )

    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    allowed_missing_prefixes = (
        "global_input_scale",
        "amplitude_encoder.",
        "a1_logit_correction.",
    )
    bad_missing = [
        k for k in missing if not any(k.startswith(p) for p in allowed_missing_prefixes)
    ]
    if bad_missing:
        raise RuntimeError("unexpected missing base keys: %s" % bad_missing)
    if unexpected:
        raise RuntimeError("unexpected checkpoint keys: %s" % list(unexpected))

    # Keep normalization tied to the current fixed train set.
    model.set_input_normalization(mean, scale, global_rms)
    model.freeze_base()
    model = model.to(device)

    model_meta = {
        "base_architecture": arch,
        "amplitude_bins": int(args.amplitude_bins),
        "amplitude_hidden": list(args.amplitude_hidden),
        "correction_limit": float(args.correction_limit),
        "base_frozen": True,
        "trainable_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    return model, model_meta, model_a1, model_gamma, original_a1, original_gamma


def train_residual_epoch(model, loader, optimizer, device, a1_span):
    model.train()
    model.enforce_frozen_base_eval()

    total = 0.0
    n = 0
    for batch in loader:
        gy = batch["gy"].to(device, non_blocking=True)
        true_a1 = batch["second"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred_a1, _ = model(gy)
        loss = torch.mean(((pred_a1 - true_a1) / a1_span) ** 2)
        loss.backward()
        optimizer.step()

        total += float(loss.item()) * len(true_a1)
        n += len(true_a1)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_a1(model, loader, device, a1_span, a1_tol):
    model.eval()
    pred_a1_parts, true_a1_parts = [], []
    pred_gamma_parts, true_gamma_parts = [], []
    mse_sum = 0.0
    n = 0

    for batch in loader:
        gy = batch["gy"].to(device, non_blocking=True)
        true_a1 = batch["second"].to(device, non_blocking=True)
        true_gamma = batch["gamma"].to(device, non_blocking=True)
        pred_a1, pred_log_gamma = model(gy)
        pred_gamma = torch.pow(10.0, pred_log_gamma)

        loss = torch.mean(((pred_a1 - true_a1) / a1_span) ** 2)
        mse_sum += float(loss.item()) * len(true_a1)
        n += len(true_a1)

        pred_a1_parts.append(pred_a1.cpu().numpy())
        true_a1_parts.append(true_a1.cpu().numpy())
        pred_gamma_parts.append(pred_gamma.cpu().numpy())
        true_gamma_parts.append(true_gamma.cpu().numpy())

    pa = np.concatenate(pred_a1_parts).astype(np.float64)
    ta = np.concatenate(true_a1_parts).astype(np.float64)
    pg = np.clip(np.concatenate(pred_gamma_parts).astype(np.float64), 1e-30, None)
    tg = np.clip(np.concatenate(true_gamma_parts).astype(np.float64), 1e-30, None)

    ae = np.abs(pa - ta)
    gf = np.exp(np.abs(np.log(pg) - np.log(tg)))
    return {
        "a1_normalized_mse": mse_sum / max(n, 1),
        "a1_mae": float(np.mean(ae)),
        "a1_p90_abs_error": float(np.quantile(ae, 0.90)),
        "a1_p95_abs_error": float(np.quantile(ae, 0.95)),
        "a1_recovery_rate": float(np.mean(ae <= float(a1_tol))),
        "gamma_p90_factor": float(np.quantile(gf, 0.90)),
    }


def better_checkpoint(current, best, min_delta):
    if best is None:
        return True
    # Primary goal: raise a1 recovery at the declared stage tolerance.
    if current["a1_recovery_rate"] > best["a1_recovery_rate"] + 1e-6:
        return True
    if current["a1_recovery_rate"] < best["a1_recovery_rate"] - 1e-6:
        return False
    # Tie-breaker: reduce the high-error tail.
    if current["a1_p95_abs_error"] < best["a1_p95_abs_error"] - min_delta:
        return True
    if current["a1_p95_abs_error"] > best["a1_p95_abs_error"] + min_delta:
        return False
    return current["a1_mae"] < best["a1_mae"] - min_delta


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
        raise ValueError("Exp34 requires a1gamma data")

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

    (
        model,
        architecture,
        model_a1,
        model_gamma,
        original_a1,
        original_gamma,
    ) = build_from_exp33_checkpoint(
        args, train_ds, metadata, mean, scale, global_rms, device
    )

    a1_span = float(model_a1[1] - model_a1[0])
    metric_a1_span = float(original_a1[1] - original_a1[0])

    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(int(args.seed) + 340034)
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
        list(model.residual_parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=max(args.lr * 1e-3, 1e-7)
    )

    # Because the residual head is zero-initialized, this is the exact loaded
    # Exp33 TCN validation performance before any Exp34 training.
    base_val = evaluate_a1(
        model, val_loader, device, a1_span, args.a1_success_tol
    )

    print("=" * 100)
    print("Exp34: frozen Exp33 TCN + amplitude residual correction for a1")
    print("data                   : %s" % data_dir)
    print("base checkpoint        : %s" % args.base_checkpoint)
    print("train / val samples    : %d / %d" % (len(train_ds), len(val_ds)))
    print("trainable / total pars : %d / %d" % (
        architecture["trainable_parameters"], architecture["total_parameters"]
    ))
    print("global input RMS       : %.10g" % global_rms)
    print("a1 residual bins       : %d" % args.amplitude_bins)
    print("a1 residual hidden     : %s" % (list(args.amplitude_hidden),))
    print(
        "base val: a1 recovery %.2f%% | P90 %.6g | P95 %.6g | gamma P90 %.4f"
        % (
            100.0 * base_val["a1_recovery_rate"],
            base_val["a1_p90_abs_error"],
            base_val["a1_p95_abs_error"],
            base_val["gamma_p90_factor"],
        )
    )
    print("=" * 100)

    checkpoint = output_dir / "best_exp34_a1_residual.pt"
    best_metrics = None
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_residual_epoch(
            model, train_loader, optimizer, device, a1_span
        )
        va = evaluate_a1(
            model, val_loader, device, a1_span, args.a1_success_tol
        )

        improved = better_checkpoint(va, best_metrics, args.min_delta)
        if improved:
            best_metrics = dict(va)
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation": va,
                    "architecture": architecture,
                    "base_checkpoint": str(args.base_checkpoint),
                },
                checkpoint,
            )
        else:
            bad_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "train_a1_normalized_mse": tr_loss,
                **va,
                "lr": optimizer.param_groups[0]["lr"],
                "is_best": int(improved),
            }
        )

        if improved or epoch % args.log_every == 0:
            mark = "*" if improved else " "
            print(
                "%s epoch %3d | train %.6g | a1 rec %.2f%% | "
                "P90 %.6g | P95 %.6g | gamma P90 %.4f"
                % (
                    mark,
                    epoch,
                    tr_loss,
                    100.0 * va["a1_recovery_rate"],
                    va["a1_p90_abs_error"],
                    va["a1_p95_abs_error"],
                    va["gamma_p90_factor"],
                )
            )

        scheduler.step()
        if bad_epochs >= args.patience:
            print("early stop; best epoch=%d" % best_epoch)
            break

    write_csv(output_dir / "training_history.csv", history)

    ck_best = load_checkpoint(checkpoint, device)
    model.load_state_dict(ck_best["model_state"])
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
        g_med, g_p90, g_p95 = finite_summary(g_rel)

        summary_rows.append(
            {
                "model": "a1res",
                "noise_dir": noise_dir.name,
                "noise_level": float(noise_level),
                "split": "test_independent",
                "trainable_parameters": architecture["trainable_parameters"],
                "total_parameters": architecture["total_parameters"],
                **rm,
                "median_f_relative_l2": f_med,
                "p90_f_relative_l2": f_p90,
                "p95_f_relative_l2": f_p95,
                "median_g_clean_relative_l2": g_med,
                "p90_g_clean_relative_l2": g_p90,
                "p95_g_clean_relative_l2": g_p95,
            }
        )

        pa = np.asarray(pred["pred_second"], dtype=np.float64)
        ta = np.asarray(pred["true_second"], dtype=np.float64)
        pg = np.asarray(pred["pred_gamma"], dtype=np.float64)
        tg = np.asarray(pred["true_gamma"], dtype=np.float64)

        for i in range(len(tg)):
            sample_rows.append(
                {
                    "model": "a1res",
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
            "noise=%7.4g | a1 rec %.2f%% | MAE %.6g | P90 %.6g | "
            "gamma P90 %.4f | joint %.2f%%"
            % (
                noise_level,
                100.0 * rm["a1_recovery_rate"],
                rm["a1_mae"],
                rm["a1_p90_abs_error"],
                rm["gamma_p90_factor"],
                100.0 * rm["joint_recovery_rate"],
            )
        )

    prefix = str(args.result_prefix).strip() or "exp34"
    write_csv(output_dir / (prefix + "_summary.csv"), summary_rows)
    write_csv(output_dir / (prefix + "_samples.csv"), sample_rows)
    (output_dir / (prefix + "_metadata.json")).write_text(
        json.dumps(
            {
                "experiment": "Exp34 frozen TCN + amplitude residual for a1",
                "purpose": "improve a1 without changing the successful gamma mapping",
                "base_checkpoint": str(args.base_checkpoint),
                "base_validation_before_residual": base_val,
                "best_epoch": best_epoch,
                "best_validation": best_metrics,
                "architecture": architecture,
                "training": {
                    "base_tcn_frozen": True,
                    "loss": "a1 normalized MSE only",
                    "checkpoint_selection": "a1 recovery rate, then P95, then MAE",
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
    print("Exp34 finished. Read first:")
    print("  %s" % (output_dir / (prefix + "_summary.csv")))
    print("  %s" % (output_dir / (prefix + "_samples.csv")))
    print("=" * 100)


if __name__ == "__main__":
    main()
