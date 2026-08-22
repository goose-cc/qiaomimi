#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp32: train the unchanged a1+gamma MLP and report continuous errors.

This file intentionally does NOT change the forward physics, MLP backbone, or
training loss used by Exp31.  The change is evaluation: gamma is treated as a
continuous regression target instead of using gamma-within-x1.2 as the main
metric.  a1 is also reported explicitly so the two-parameter stage is judged
jointly rather than by gamma alone.
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Exp32 a1+gamma continuous-error validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--result-prefix", default="exp32")
    p.add_argument("--train-noise-dir", default="noise_0p2pct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260850)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--hidden-sizes", type=parse_hidden_sizes, default=(256, 256, 128))
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=1e-7)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--second-loss-weight", type=float, default=1.0)
    p.add_argument("--gamma-loss-weight", type=float, default=1.0)
    p.add_argument("--physics-integration-points", type=int, default=512)
    p.add_argument("--skip-physics-metrics", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument(
        "--parameter-bounds-source",
        choices=("selected", "original"),
        default="selected",
        help=(
            "selected reproduces Exp31 model-output bounds; original uses the full "
            "pre-filter parameter domain as a stricter control experiment"
        ),
    )
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


def _safe_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.quantile(x, q)) if len(x) else math.nan


def continuous_metrics(
    pred_a1: np.ndarray,
    true_a1: np.ndarray,
    pred_gamma: np.ndarray,
    true_gamma: np.ndarray,
    *,
    a1_reference_span: float,
) -> dict[str, float]:
    """Continuous regression metrics; no hard x1.2 success threshold is used."""
    pa = np.asarray(pred_a1, dtype=np.float64)
    ta = np.asarray(true_a1, dtype=np.float64)
    pg = np.clip(np.asarray(pred_gamma, dtype=np.float64), 1e-30, None)
    tg = np.clip(np.asarray(true_gamma, dtype=np.float64), 1e-30, None)

    a1_abs = np.abs(pa - ta)
    a1_norm = a1_abs / max(float(a1_reference_span), 1e-30)

    log10_signed = np.log10(pg) - np.log10(tg)
    log10_abs = np.abs(log10_signed)
    gamma_rel_abs = np.abs(pg - tg) / tg
    gamma_factor = np.exp(np.abs(np.log(pg) - np.log(tg)))

    return {
        "count": int(len(pa)),
        "a1_mae": float(np.mean(a1_abs)),
        "a1_rmse": float(np.sqrt(np.mean((pa - ta) ** 2))),
        "a1_median_abs_error": float(np.median(a1_abs)),
        "a1_p90_abs_error": _safe_quantile(a1_abs, 0.90),
        "a1_p95_abs_error": _safe_quantile(a1_abs, 0.95),
        "a1_max_abs_error": float(np.max(a1_abs)),
        "a1_nmae_original_span": float(np.mean(a1_norm)),
        "a1_p90_norm_error_original_span": _safe_quantile(a1_norm, 0.90),
        "gamma_log10_mae": float(np.mean(log10_abs)),
        "gamma_log10_rmse": float(np.sqrt(np.mean(log10_signed ** 2))),
        "gamma_median_abs_log10_error": float(np.median(log10_abs)),
        "gamma_p90_abs_log10_error": _safe_quantile(log10_abs, 0.90),
        "gamma_p95_abs_log10_error": _safe_quantile(log10_abs, 0.95),
        "gamma_relative_mae": float(np.mean(gamma_rel_abs)),
        "gamma_median_relative_error": float(np.median(gamma_rel_abs)),
        "gamma_p90_relative_error": _safe_quantile(gamma_rel_abs, 0.90),
        "gamma_median_factor": float(np.median(gamma_factor)),
        "gamma_p90_factor": _safe_quantile(gamma_factor, 0.90),
        "gamma_p95_factor": _safe_quantile(gamma_factor, 0.95),
        "gamma_max_factor": float(np.max(gamma_factor)),
    }


def choose_bounds(metadata: dict, source: str):
    selected_a1 = [float(x) for x in metadata["a1_range"]]
    selected_gamma = [float(x) for x in metadata["gamma_range"]]
    original_a1 = [float(x) for x in metadata.get("original_a1_range", selected_a1)]
    original_gamma = [float(x) for x in metadata.get("original_gamma_range", selected_gamma)]

    if source == "original":
        model_a1 = original_a1
        model_gamma = original_gamma
    else:
        model_a1 = selected_a1
        model_gamma = selected_gamma

    return model_a1, model_gamma, original_a1, original_gamma


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("mode") != "a1gamma":
        raise ValueError("Exp32 requires a1gamma data")

    model_a1, model_gamma, original_a1, original_gamma = choose_bounds(
        metadata, args.parameter_bounds_source
    )
    model_a1_min, model_a1_max = model_a1
    model_gamma_min, model_gamma_max = model_gamma
    training_a1_span = model_a1_max - model_a1_min
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
    model = PairwiseInverseMLP(
        input_points=train_ds.gy.shape[1],
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        second_min=model_a1_min,
        second_max=model_a1_max,
        gamma_min=model_gamma_min,
        gamma_max=model_gamma_max,
    ).to(device)
    model.set_input_normalization(mean, scale)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
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
    print("Exp32: unchanged MLP/loss + continuous a1/gamma evaluation")
    print("data                    : %s" % data_dir)
    print("train / val samples     : %d / %d" % (len(train_ds), len(val_ds)))
    print("physical states         : %s" % metadata["selected_state_counts"])
    print("alias minimum RMS-SNR   : %s" % metadata.get("alias_min_rms_snr"))
    print("model-bound source      : %s" % args.parameter_bounds_source)
    print("model a1 bounds         : %s" % model_a1)
    print("model gamma bounds      : %s" % model_gamma)
    print("metric a1 reference     : %s" % original_a1)
    print("global input RMS        : %.10g" % global_rms)
    print("=" * 100)

    best_val = math.inf
    best_epoch = 0
    bad_epochs = 0
    checkpoint = output_dir / "best_exp32_mlp.pt"
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
                {"model_state": model.state_dict(), "epoch": epoch, "val_loss": va},
                checkpoint,
            )
        else:
            bad_epochs += 1

        history.append(
            {
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

        cm = continuous_metrics(
            pred["pred_second"],
            pred["true_second"],
            pred["pred_gamma"],
            pred["true_gamma"],
            a1_reference_span=metric_a1_span,
        )
        f_rel = np.asarray(pred["f_relative_l2"], dtype=np.float64)
        g_rel = np.asarray(pred["g_clean_relative_l2"], dtype=np.float64)
        summary_rows.append(
            {
                "noise_dir": noise_dir.name,
                "noise_level": float(noise_level),
                "split": "test_independent",
                **cm,
                "median_f_relative_l2": float(np.nanmedian(f_rel)),
                "p90_f_relative_l2": float(np.nanquantile(f_rel, 0.90)),
                "p95_f_relative_l2": float(np.nanquantile(f_rel, 0.95)),
                "median_g_clean_relative_l2": float(np.nanmedian(g_rel)),
                "p90_g_clean_relative_l2": float(np.nanquantile(g_rel, 0.90)),
            }
        )

        pa = np.asarray(pred["pred_second"], dtype=np.float64)
        ta = np.asarray(pred["true_second"], dtype=np.float64)
        pg = np.clip(np.asarray(pred["pred_gamma"], dtype=np.float64), 1e-30, None)
        tg = np.clip(np.asarray(pred["true_gamma"], dtype=np.float64), 1e-30, None)
        a1_abs = np.abs(pa - ta)
        gamma_log10_signed = np.log10(pg) - np.log10(tg)
        gamma_log10_abs = np.abs(gamma_log10_signed)
        gamma_rel_abs = np.abs(pg - tg) / tg
        gamma_factor = np.exp(np.abs(np.log(pg) - np.log(tg)))

        for i in range(len(tg)):
            sample_rows.append(
                {
                    "noise_dir": noise_dir.name,
                    "noise_level": float(noise_level),
                    "split": "test_independent",
                    "sample_index": int(i),
                    "true_a1": float(ta[i]),
                    "pred_a1": float(pa[i]),
                    "a1_signed_error": float(pa[i] - ta[i]),
                    "a1_abs_error": float(a1_abs[i]),
                    "a1_norm_abs_error_original_span": float(a1_abs[i] / metric_a1_span),
                    "true_gamma": float(tg[i]),
                    "pred_gamma": float(pg[i]),
                    "gamma_signed_log10_error": float(gamma_log10_signed[i]),
                    "gamma_abs_log10_error": float(gamma_log10_abs[i]),
                    "gamma_relative_error": float(gamma_rel_abs[i]),
                    "gamma_factor_error": float(gamma_factor[i]),
                    "f_relative_l2": float(f_rel[i]),
                    "g_clean_relative_l2": float(g_rel[i]),
                }
            )

        print(
            "noise=%7.4g | a1 MAE=%.6g P90=%.6g | gamma log10-MAE=%.5f "
            "P90-factor=%.4f | f P90=%.5f"
            % (
                noise_level,
                cm["a1_mae"],
                cm["a1_p90_abs_error"],
                cm["gamma_log10_mae"],
                cm["gamma_p90_factor"],
                float(np.nanquantile(f_rel, 0.90)),
            )
        )

    prefix = str(args.result_prefix).strip() or "exp32"
    write_csv(output_dir / (prefix + "_summary.csv"), summary_rows)
    write_csv(output_dir / (prefix + "_samples.csv"), sample_rows)
    (output_dir / (prefix + "_metadata.json")).write_text(
        json.dumps(
            {
                "experiment": "Exp32 continuous-error validation for a1+gamma",
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "primary_evaluation": {
                    "a1": "MAE/RMSE/median/P90/P95 absolute error",
                    "gamma": "log10 error + continuous factor error; no x1.2 hard gate",
                    "physics": "f and clean-g relative L2",
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
    print("Exp32 training finished")
    print("Read first:")
    print("  %s" % (output_dir / (prefix + "_summary.csv")))
    print("  %s" % (output_dir / (prefix + "_samples.csv")))
    print("=" * 100)


if __name__ == "__main__":
    main()
