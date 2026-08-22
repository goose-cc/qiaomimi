#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train an Exp30 a1+gamma TCN on the same identifiable-region data as the MLP."""
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
    metrics,
    predict_dataset,
    resolve_device,
    set_seed,
    train_one_epoch,
    training_input_normalization,
    write_csv,
)


def parse_dilations(text):
    if isinstance(text, (tuple, list)):
        return tuple(int(x) for x in text)
    values = tuple(int(x.strip()) for x in str(text).split(",") if x.strip())
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("dilations must be positive comma-separated integers")
    return values


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Exp30 a1+gamma TCN regression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--result-prefix", default="exp30_tcn")
    p.add_argument("--train-noise-dir", default="noise_0p2pct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260830)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--dilations", type=parse_dilations, default=(1,2,4,8,16,32,64,128))
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--head-hidden", type=int, default=128)
    p.add_argument("--q2-min", type=float, default=-100.0)
    p.add_argument("--q2-max", type=float, default=-6.0)
    p.add_argument("--no-q2-channel", action="store_true")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=1e-7)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--second-loss-weight", type=float, default=1.0)
    p.add_argument("--gamma-loss-weight", type=float, default=1.0)
    p.add_argument("--gamma-min", type=float, default=0.01)
    p.add_argument("--gamma-max", type=float, default=1.0)
    p.add_argument("--physics-integration-points", type=int, default=512)
    p.add_argument("--skip-physics-metrics", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    return p.parse_args()


def noise_dirs_with_test(data_dir):
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


def detect_mlp_return_style(
    input_points, mean, scale, second_min, second_max, gamma_min, gamma_max
):
    """
    Probe the repository's existing PairwiseInverseMLP output convention.
    This keeps compatibility with train_exp20_pairwise.py without changing it.
    """
    probe = PairwiseInverseMLP(
        input_points=input_points,
        hidden_sizes=(8,),
        dropout=0.0,
        second_min=second_min,
        second_max=second_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )
    probe.set_input_normalization(mean, scale)
    probe.eval()
    x = torch.zeros(2, input_points, dtype=torch.float32)
    with torch.no_grad():
        y = probe(x)

    if torch.is_tensor(y):
        return "tensor", None
    if isinstance(y, (tuple, list)) and len(y) == 2:
        return "tuple", None
    if isinstance(y, dict) and len(y) >= 2:
        # Prefer the obvious keys if present.
        if "second" in y and "gamma" in y:
            return "dict", ("second", "gamma")
        if "pred_second" in y and "pred_gamma" in y:
            return "dict", ("pred_second", "pred_gamma")
        keys = list(y.keys())[:2]
        return "dict", tuple(keys)

    raise TypeError(
        "Could not infer PairwiseInverseMLP output style; "
        f"got type={type(y).__name__}"
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("mode") != "a1gamma":
        raise ValueError("Exp30 TCN requires a1gamma data")

    a1_min, a1_max = [float(x) for x in metadata["a1_range"]]
    second_span = a1_max - a1_min
    gamma_min_meta, gamma_max_meta = [float(x) for x in metadata["gamma_range"]]
    gamma_min = max(float(args.gamma_min), gamma_min_meta)
    gamma_max = min(float(args.gamma_max), gamma_max_meta)

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

    model = PairwiseInverseTCN(
        input_points=train_ds.gy.shape[1],
        channels=args.channels,
        dilations=args.dilations,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        head_hidden=args.head_hidden,
        second_min=a1_min,
        second_max=a1_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        q2_min=args.q2_min,
        q2_max=args.q2_max,
        use_q2_channel=not args.no_q2_channel,
    ).to(device)
    model.set_input_normalization(mean, scale)

    style, keys = detect_mlp_return_style(
        train_ds.gy.shape[1], mean, scale,
        a1_min, a1_max, gamma_min, gamma_max
    )
    model.set_return_style(style, keys)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=max(args.lr * 1e-3, 1e-7)
    )

    print("=" * 100)
    print("Exp30 TCN regression on alias-filtered identifiable a1+gamma data")
    print("data                 : %s" % data_dir)
    print("train / val samples  : %d / %d" % (len(train_ds), len(val_ds)))
    print("physical states      : %s" % metadata["selected_state_counts"])
    print("global min RMS-SNR   : %.6g" % metadata["global_min_rms_snr"])
    print("input points         : %d" % train_ds.gy.shape[1])
    print("global input RMS     : %.10g" % global_rms)
    print("TCN channels         : %d" % args.channels)
    print("TCN dilations        : %s" % (args.dilations,))
    print("TCN receptive field  : %d" % model.receptive_field_estimate)
    print("q2 channel           : %s" % (not args.no_q2_channel))
    print("MLP-compatible output: %s" % style)
    print("=" * 100)

    best_val = math.inf
    best_epoch = 0
    bad_epochs = 0
    checkpoint = output_dir / "best_exp30_tcn.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model, train_loader, optimizer, device, second_span,
            args.second_loss_weight, args.gamma_loss_weight,
        )
        va, vm = evaluate_loader(
            model, val_loader, device, second_span,
            args.second_loss_weight, args.gamma_loss_weight,
        )
        improved = va < best_val - args.min_delta
        if improved:
            best_val = va
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": va,
                "return_style": style,
            }, checkpoint)
        else:
            bad_epochs += 1

        history.append({
            "epoch": epoch,
            "train_loss": tr,
            "val_loss": va,
            "gamma_median_factor": vm["gamma_median_factor"],
            "gamma_within_x1p2": vm["gamma_within_x1p2"],
            "a1_median_norm_error": vm["second_median_norm_error"],
            "lr": optimizer.param_groups[0]["lr"],
            "is_best": int(improved),
        })
        if improved or epoch % args.log_every == 0:
            mark = "*" if improved else " "
            print(
                "%s epoch %3d | train %.6g | val %.6g | gamma med %.4f | <=x1.2 %.3f | a1 norm %.4f"
                % (
                    mark, epoch, tr, va, vm["gamma_median_factor"],
                    vm["gamma_within_x1p2"], vm["second_median_norm_error"],
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
        m = metrics(
            pred["pred_second"], pred["true_second"],
            pred["pred_gamma"], pred["true_gamma"], second_span,
        )
        summary_rows.append({
            "noise_dir": noise_dir.name,
            "noise_level": float(noise_level),
            "split": "test_independent",
            **m,
            "median_f_relative_l2": float(np.nanmedian(pred["f_relative_l2"])),
            "p90_f_relative_l2": float(np.nanquantile(pred["f_relative_l2"], 0.90)),
            "median_g_clean_relative_l2": float(np.nanmedian(pred["g_clean_relative_l2"])),
            "catastrophic_gamma_factor_ge5": float(np.mean(pred["gamma_factor_error"] >= 5.0)),
            "catastrophic_gamma_factor_ge10": float(np.mean(pred["gamma_factor_error"] >= 10.0)),
        })
        for i in range(len(pred["true_gamma"])):
            sample_rows.append({
                "noise_dir": noise_dir.name,
                "noise_level": float(noise_level),
                "split": "test_independent",
                "true_a1": float(pred["true_second"][i]),
                "pred_a1": float(pred["pred_second"][i]),
                "true_gamma": float(pred["true_gamma"][i]),
                "pred_gamma": float(pred["pred_gamma"][i]),
                "gamma_factor_error": float(pred["gamma_factor_error"][i]),
                "f_relative_l2": float(pred["f_relative_l2"][i]),
                "g_clean_relative_l2": float(pred["g_clean_relative_l2"][i]),
            })

        print(
            "noise=%7.4g | gamma med=%.4f | <=x1.2=%.3f | f med=%.5f | cat>=5=%.4f"
            % (
                noise_level, m["gamma_median_factor"], m["gamma_within_x1p2"],
                float(np.nanmedian(pred["f_relative_l2"])),
                float(np.mean(pred["gamma_factor_error"] >= 5.0)),
            )
        )

    prefix = str(args.result_prefix).strip() or "exp30_tcn"
    write_csv(output_dir / (prefix + "_summary.csv"), summary_rows)
    write_csv(output_dir / (prefix + "_samples.csv"), sample_rows)
    (output_dir / (prefix + "_metadata.json")).write_text(
        json.dumps({
            "experiment": "Exp30 TCN regression on alias-filtered identifiable a1+gamma data",
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "model": {
                "type": "PairwiseInverseTCN",
                "channels": args.channels,
                "dilations": list(args.dilations),
                "kernel_size": args.kernel_size,
                "head_hidden": args.head_hidden,
                "q2_min": args.q2_min,
                "q2_max": args.q2_max,
                "use_q2_channel": not args.no_q2_channel,
                "receptive_field_estimate": model.receptive_field_estimate,
            },
            "args": vars(args),
            "dataset_metadata": metadata,
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("=" * 100)
    print("Exp30 TCN training finished")
    print("Read first:")
    print("  %s" % (output_dir / (prefix + "_summary.csv")))
    print("  %s" % (output_dir / (prefix + "_samples.csv")))
    print("=" * 100)


if __name__ == "__main__":
    main()
