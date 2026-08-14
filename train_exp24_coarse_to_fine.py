#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train Exp24 coarse-classification + within-bin gamma regression.

This is the structural test motivated by Exp23 failures where direct continuous
regression sometimes jumped from a moderate true gamma to a very small gamma.

Controlled comparison:
- Exp24A: direct continuous regression (existing train_exp20_pairwise.py)
- Exp24B: this coarse-to-fine model
Both use the SAME dense-gamma data, q500 physical observations, 1000-wide input,
hidden sizes, noise level and sample budget.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from CoarseToFineA1GammaMLP import CoarseToFineA1GammaMLP
from train_exp20_pairwise import (
    PairwiseNpzDataset,
    discover_noise_dirs,
    gamma_factor_error,
    metrics,
    parse_hidden_sizes,
    resolve_device,
    set_seed,
    training_input_normalization,
)


def parse_float_list(text):
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) < 3:
        raise argparse.ArgumentTypeError("need at least 3 gamma edges")
    return tuple(vals)


def parse_args():
    p = argparse.ArgumentParser(
        description="Exp24 coarse-to-fine a1+gamma training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--result-prefix", default="exp24b")
    p.add_argument("--train-noise-dir", default="noise_0p2pct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260814)
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
    p.add_argument(
        "--gamma-edges",
        type=parse_float_list,
        default=(0.01, 0.0316227766, 0.1, 0.316227766, 1.0),
    )
    p.add_argument("--a1-loss-weight", type=float, default=1.0)
    p.add_argument("--class-loss-weight", type=float, default=1.0)
    p.add_argument("--residual-loss-weight", type=float, default=1.0)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--skip-physics-metrics", action="store_true")
    p.add_argument("--physics-integration-points", type=int, default=512)
    return p.parse_args()


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def coarse_metrics(
    pred_a1, true_a1, pred_gamma, true_gamma,
    pred_class, true_class, oracle_gamma, a1_span,
):
    base = metrics(pred_a1, true_a1, pred_gamma, true_gamma, a1_span)
    oracle_ferr = gamma_factor_error(oracle_gamma, true_gamma)
    jump = np.abs(np.asarray(pred_class, dtype=int) - np.asarray(true_class, dtype=int))
    base.update({
        "gamma_class_accuracy": float(np.mean(np.asarray(pred_class) == np.asarray(true_class))),
        "gamma_class_mean_abs_jump": float(np.mean(jump)),
        "gamma_class_jump_ge2": float(np.mean(jump >= 2)),
        "oracle_bin_gamma_median_factor": float(np.median(oracle_ferr)),
        "oracle_bin_gamma_within_x1p2": float(np.mean(oracle_ferr <= 1.2)),
    })
    return base


def loss_batch(
    model, gy, a1, gamma, a1_span,
    a1_w, class_w, residual_w,
):
    pred_a1, logits, residual_all = model(gy)
    true_class = model.gamma_class(gamma)
    target_u = model.residual_target(gamma, true_class)
    pred_u_true_class = residual_all.gather(1, true_class[:, None]).squeeze(1)

    a1_loss = torch.mean(((pred_a1 - a1) / a1_span) ** 2)
    # Normalize CE by random-guess entropy so its natural initial scale is ~1.
    class_loss = F.cross_entropy(logits, true_class) / math.log(float(model.n_classes))
    residual_loss = torch.mean((pred_u_true_class - target_u) ** 2)
    total = a1_w * a1_loss + class_w * class_loss + residual_w * residual_loss
    return total, pred_a1, logits, residual_all, true_class


def train_epoch(model, loader, optimizer, device, a1_span, args):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        gy = batch["gy"].to(device)
        a1 = batch["second"].to(device)
        gamma = batch["gamma"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, _, _ = loss_batch(
            model, gy, a1, gamma, a1_span,
            args.a1_loss_weight, args.class_loss_weight, args.residual_loss_weight,
        )
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(gamma)
        n += len(gamma)
    return total / max(n, 1)


@torch.no_grad()
def eval_loader(model, loader, device, a1_span, args):
    model.eval()
    total = 0.0
    n = 0
    pa, ta, pg, tg, pc, tc, og = [], [], [], [], [], [], []
    for batch in loader:
        gy = batch["gy"].to(device)
        a1 = batch["second"].to(device)
        gamma = batch["gamma"].to(device)
        loss, pred_a1, logits, residual_all, true_class = loss_batch(
            model, gy, a1, gamma, a1_span,
            args.a1_loss_weight, args.class_loss_weight, args.residual_loss_weight,
        )
        pred_class = torch.argmax(logits, dim=1)
        pred_u = residual_all.gather(1, pred_class[:, None]).squeeze(1)
        pred_gamma = model.gamma_from_class_residual(pred_class, pred_u)
        oracle_u = residual_all.gather(1, true_class[:, None]).squeeze(1)
        oracle_gamma = model.gamma_from_class_residual(true_class, oracle_u)

        total += float(loss.item()) * len(gamma)
        n += len(gamma)
        pa.append(pred_a1.cpu().numpy()); ta.append(a1.cpu().numpy())
        pg.append(pred_gamma.cpu().numpy()); tg.append(gamma.cpu().numpy())
        pc.append(pred_class.cpu().numpy()); tc.append(true_class.cpu().numpy())
        og.append(oracle_gamma.cpu().numpy())

    pa = np.concatenate(pa); ta = np.concatenate(ta)
    pg = np.concatenate(pg); tg = np.concatenate(tg)
    pc = np.concatenate(pc); tc = np.concatenate(tc)
    og = np.concatenate(og)
    return (
        total / max(n, 1),
        coarse_metrics(pa, ta, pg, tg, pc, tc, og, a1_span),
    )


@torch.no_grad()
def predict_dataset(model, ds, device, args, compute_physics):
    loader = DataLoader(
        ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    pa, ta, pg, tg, pc, tc, og, noise = [], [], [], [], [], [], [], []
    f_parts, g_parts = [], []
    physics_config = None
    scaled_curves_torch = None

    if compute_physics:
        if ds.fx is None or ds.gy_clean is None or ds.parameters is None:
            raise KeyError("%s lacks physics arrays" % ds.path)
        from mc_pool_config import DEFAULT_PHYSICS
        from mc_physics import scaled_curves_torch as _scaled_curves_torch
        scaled_curves_torch = _scaled_curves_torch
        physics_config = replace(
            DEFAULT_PHYSICS,
            q2_points=int(ds.gy.shape[1]),
            output_points=int(ds.fx.shape[1]),
        )

    model.eval()
    for batch in loader:
        gy = batch["gy"].to(device)
        true_a1 = batch["second"].to(device)
        true_gamma = batch["gamma"].to(device)

        pred_a1, logits, residual_all = model(gy)
        true_class = model.gamma_class(true_gamma)
        pred_class = torch.argmax(logits, dim=1)

        pred_u = residual_all.gather(1, pred_class[:, None]).squeeze(1)
        pred_gamma = model.gamma_from_class_residual(pred_class, pred_u)
        oracle_u = residual_all.gather(1, true_class[:, None]).squeeze(1)
        oracle_gamma = model.gamma_from_class_residual(true_class, oracle_u)

        pa.append(pred_a1.cpu().numpy()); ta.append(true_a1.cpu().numpy())
        pg.append(pred_gamma.cpu().numpy()); tg.append(true_gamma.cpu().numpy())
        pc.append(pred_class.cpu().numpy()); tc.append(true_class.cpu().numpy())
        og.append(oracle_gamma.cpu().numpy())
        noise.append(batch["noise_level"].numpy())

        if compute_physics:
            params = batch["parameters"].to(device).clone()
            params[:, 0] = pred_a1
            params[:, 4] = pred_gamma
            pred_f, pred_g = scaled_curves_torch(
                params,
                integration_points=int(args.physics_integration_points),
                config=physics_config,
            )
            true_f = batch["fx"].to(device)
            true_g = batch["gy_clean"].to(device)
            f_rel = torch.linalg.vector_norm(pred_f - true_f, dim=1) / torch.linalg.vector_norm(
                true_f, dim=1
            ).clamp_min(1e-12)
            g_rel = torch.linalg.vector_norm(pred_g - true_g, dim=1) / torch.linalg.vector_norm(
                true_g, dim=1
            ).clamp_min(1e-12)
            f_parts.append(f_rel.cpu().numpy())
            g_parts.append(g_rel.cpu().numpy())

    result = {
        "pred_a1": np.concatenate(pa),
        "true_a1": np.concatenate(ta),
        "pred_gamma": np.concatenate(pg),
        "true_gamma": np.concatenate(tg),
        "pred_class": np.concatenate(pc).astype(int),
        "true_class": np.concatenate(tc).astype(int),
        "oracle_gamma": np.concatenate(og),
        "noise_level": np.concatenate(noise),
    }
    result["gamma_factor_error"] = gamma_factor_error(
        result["pred_gamma"], result["true_gamma"]
    )
    result["oracle_gamma_factor_error"] = gamma_factor_error(
        result["oracle_gamma"], result["true_gamma"]
    )
    if compute_physics:
        result["f_relative_l2"] = np.concatenate(f_parts)
        result["g_clean_relative_l2"] = np.concatenate(g_parts)
    else:
        n = len(result["true_gamma"])
        result["f_relative_l2"] = np.full(n, np.nan)
        result["g_clean_relative_l2"] = np.full(n, np.nan)
    return result


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("mode") != "a1gamma":
        raise ValueError("Exp24 requires a1gamma data")
    a1_values = np.asarray(metadata["second_train_values"], dtype=float)
    a1_min = float(np.min(a1_values))
    a1_max = float(np.max(a1_values))
    a1_span = a1_max - a1_min

    train_ds = PairwiseNpzDataset(
        data_dir / args.train_noise_dir / "train.npz",
        mode="a1gamma", max_samples=args.max_train_samples,
    )
    val_ds = PairwiseNpzDataset(
        data_dir / args.train_noise_dir / "val.npz",
        mode="a1gamma", max_samples=args.max_val_samples,
    )

    mean, scale, global_rms = training_input_normalization(train_ds.gy)
    model = CoarseToFineA1GammaMLP(
        input_points=train_ds.gy.shape[1],
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        a1_min=a1_min,
        a1_max=a1_max,
        gamma_edges=args.gamma_edges,
    ).to(device)
    model.set_input_normalization(mean, scale)

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
    print("Exp24B coarse-to-fine a1 + gamma")
    print("data            : %s" % data_dir)
    print("input points    : %d" % train_ds.gy.shape[1])
    print("train / val     : %d / %d" % (len(train_ds), len(val_ds)))
    print("gamma edges     : %s" % (list(args.gamma_edges),))
    print("gamma classes   : %d" % model.n_classes)
    print("global input RMS: %.10g" % global_rms)
    print("=" * 100)

    best_val = float("inf")
    best_epoch = 0
    bad = 0
    ckpt = output_dir / "best_exp24_coarse_to_fine.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device, a1_span, args)
        va, vm = eval_loader(model, val_loader, device, a1_span, args)
        improved = va < best_val - args.min_delta
        if improved:
            best_val = va
            best_epoch = epoch
            bad = 0
            torch.save({
                "model_state": model.state_dict(),
                "gamma_edges": tuple(args.gamma_edges),
                "a1_min": a1_min,
                "a1_max": a1_max,
                "hidden_sizes": tuple(args.hidden_sizes),
                "dropout": args.dropout,
                "epoch": epoch,
                "val_loss": va,
            }, ckpt)
        else:
            bad += 1

        history.append({
            "epoch": epoch,
            "train_loss": tr,
            "val_loss": va,
            "gamma_median_factor": vm["gamma_median_factor"],
            "gamma_within_x1p2": vm["gamma_within_x1p2"],
            "gamma_class_accuracy": vm["gamma_class_accuracy"],
            "oracle_bin_gamma_median_factor": vm["oracle_bin_gamma_median_factor"],
            "lr": optimizer.param_groups[0]["lr"],
            "is_best": int(improved),
        })
        if improved or epoch % args.log_every == 0:
            mark = "*" if improved else " "
            print(
                "%s epoch %3d | train %.6g | val %.6g | class %.3f | "
                "gamma med %.4f | <=x1.2 %.3f | oracle med %.4f"
                % (
                    mark, epoch, tr, va, vm["gamma_class_accuracy"],
                    vm["gamma_median_factor"], vm["gamma_within_x1p2"],
                    vm["oracle_bin_gamma_median_factor"],
                )
            )
        scheduler.step()
        if bad >= args.patience:
            print("early stop; best epoch=%d" % best_epoch)
            break

    write_csv(output_dir / "training_history.csv", history)

    try:
        checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    summary_rows = []
    sample_rows = []
    confusion_rows = []

    for noise_dir in discover_noise_dirs(data_dir):
        for split in (
            "test_seen", "test_interp_gamma",
            "test_interp_second", "test_interp_both",
        ):
            path = noise_dir / (split + ".npz")
            if not path.exists():
                continue
            ds = PairwiseNpzDataset(
                path, mode="a1gamma", max_samples=args.max_test_samples
            )
            r = predict_dataset(
                model, ds, device, args,
                compute_physics=not args.skip_physics_metrics,
            )
            cm = coarse_metrics(
                r["pred_a1"], r["true_a1"], r["pred_gamma"], r["true_gamma"],
                r["pred_class"], r["true_class"], r["oracle_gamma"], a1_span,
            )
            level = float(np.nanmedian(r["noise_level"]))
            summary_rows.append({
                "noise_dir": noise_dir.name,
                "noise_level": level,
                "split": split,
                **cm,
                "median_f_relative_l2": float(np.nanmedian(r["f_relative_l2"])),
                "median_g_clean_relative_l2": float(np.nanmedian(r["g_clean_relative_l2"])),
            })

            for i in range(len(r["true_gamma"])):
                sample_rows.append({
                    "noise_dir": noise_dir.name,
                    "noise_level": float(r["noise_level"][i]),
                    "split": split,
                    "true_a1": float(r["true_a1"][i]),
                    "pred_a1": float(r["pred_a1"][i]),
                    "true_gamma": float(r["true_gamma"][i]),
                    "pred_gamma": float(r["pred_gamma"][i]),
                    "oracle_gamma": float(r["oracle_gamma"][i]),
                    "true_gamma_class": int(r["true_class"][i]),
                    "pred_gamma_class": int(r["pred_class"][i]),
                    "class_correct": int(r["true_class"][i] == r["pred_class"][i]),
                    "class_jump": int(abs(r["true_class"][i] - r["pred_class"][i])),
                    "gamma_factor_error": float(r["gamma_factor_error"][i]),
                    "oracle_gamma_factor_error": float(r["oracle_gamma_factor_error"][i]),
                    "f_relative_l2": float(r["f_relative_l2"][i]),
                    "g_clean_relative_l2": float(r["g_clean_relative_l2"][i]),
                })

            if noise_dir.name == args.train_noise_dir:
                ncls = model.n_classes
                for tc in range(ncls):
                    mask_t = r["true_class"] == tc
                    for pc in range(ncls):
                        confusion_rows.append({
                            "split": split,
                            "true_class": tc,
                            "pred_class": pc,
                            "count": int(np.sum(mask_t & (r["pred_class"] == pc))),
                        })

            print(
                "%-16s | %-20s | class=%.3f | jump>=2=%.3f | "
                "gamma med=%.4f | oracle med=%.4f"
                % (
                    noise_dir.name, split, cm["gamma_class_accuracy"],
                    cm["gamma_class_jump_ge2"], cm["gamma_median_factor"],
                    cm["oracle_bin_gamma_median_factor"],
                )
            )

    prefix = str(args.result_prefix).strip() or "exp24b"
    write_csv(output_dir / (prefix + "_summary.csv"), summary_rows)
    write_csv(output_dir / (prefix + "_samples.csv"), sample_rows)
    write_csv(output_dir / (prefix + "_confusion.csv"), confusion_rows)

    meta = {
        "experiment": "Exp24B coarse-to-fine gamma",
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "gamma_edges": list(args.gamma_edges),
        "training_args": vars(args),
        "dataset_metadata": metadata,
    }
    (output_dir / (prefix + "_metadata.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("=" * 100)
    print("Exp24B finished.")
    print("Read first:")
    print("  %s" % (output_dir / (prefix + "_summary.csv")))
    print("  %s" % (output_dir / (prefix + "_samples.csv")))
    print("  %s" % (output_dir / (prefix + "_confusion.csv")))
    print("=" * 100)


if __name__ == "__main__":
    main()
