#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp25: continuous one-to-one curriculum in g-space.

This experiment deliberately returns to the ordinary continuous regression

    g(q^2) -> (a1, gamma)

and does NOT use hard classification or narrow-gamma loss.

Both control runs use the SAME dense Exp24 data, model, normalization, loss,
seed, total number of epochs, and (approximately) the same number of optimizer
updates per epoch.

Control:
    strategy=full
        sample the full dense training set from epoch 1.

Curriculum:
    strategy=curriculum
        easy  -> medium -> full
    where "easy/medium" states are selected by their clean nearest-neighbour
    RMS-SNR in g-space. Selection is stratified within each gamma anchor so the
    full gamma range is present from the first stage.

Early curriculum stages are sampled with replacement to the SAME epoch length
as the full training set. Therefore the comparison is not simply "fewer
updates versus more updates".
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from PairwiseInverseMLP import PairwiseInverseMLP
from train_exp20_pairwise import (
    PairwiseNpzDataset,
    discover_noise_dirs,
    evaluate_loader,
    metrics,
    parse_hidden_sizes,
    predict_dataset,
    resolve_device,
    set_seed,
    train_one_epoch,
    training_input_normalization,
    write_csv,
)


def parse_int_list(text):
    vals = tuple(int(x.strip()) for x in str(text).split(",") if x.strip())
    if len(vals) != 3 or any(x < 1 for x in vals):
        raise argparse.ArgumentTypeError("stage epochs must be three positive integers, e.g. 20,20,40")
    return vals


def parse_args():
    p = argparse.ArgumentParser(
        description="Exp25 g-separation continuous curriculum",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--result-prefix", default="exp25")
    p.add_argument("--strategy", choices=("full", "curriculum"), required=True)
    p.add_argument("--train-noise-dir", default="noise_0p2pct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260814)
    p.add_argument("--stage-epochs", type=parse_int_list, default=(20, 20, 40))
    p.add_argument("--easy-fraction-per-gamma", type=float, default=0.45)
    p.add_argument("--medium-fraction-per-gamma", type=float, default=0.75)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--hidden-sizes", type=parse_hidden_sizes, default=(256, 256, 128))
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--second-loss-weight", type=float, default=1.0)
    p.add_argument("--gamma-loss-weight", type=float, default=1.0)
    p.add_argument("--gamma-min", type=float, default=1e-3)
    p.add_argument("--gamma-max", type=float, default=1.0)
    p.add_argument("--physics-integration-points", type=int, default=512)
    p.add_argument("--skip-physics-metrics", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    return p.parse_args()


def load_state_separation(path):
    if not path.exists():
        raise FileNotFoundError(
            "%s not found. Exp25 expects the separation file generated with the dense Exp24 data." % path
        )
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "a1": float(row["a1"]),
                "gamma": float(row["gamma"]),
                "snr_sep_rms": float(row["snr_sep_rms"]),
                "nearest_a1": float(row["nearest_a1"]),
                "nearest_gamma": float(row["nearest_gamma"]),
            })
    if not rows:
        raise ValueError("empty state separation CSV")
    return rows


def match_samples_to_states(a1, gamma, state_rows):
    sa = np.asarray([r["a1"] for r in state_rows], dtype=np.float64)
    sg = np.asarray([r["gamma"] for r in state_rows], dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    a_span = max(float(sa.max() - sa.min()), 1e-12)
    lg = np.log10(sg)
    lg_span = max(float(lg.max() - lg.min()), 1e-12)

    result = np.empty(len(a1), dtype=np.int32)
    # Chunking avoids a potentially large N x states temporary for future datasets.
    for start in range(0, len(a1), 4096):
        stop = min(start + 4096, len(a1))
        da = (a1[start:stop, None] - sa[None, :]) / a_span
        dg = (np.log10(gamma[start:stop, None]) - lg[None, :]) / lg_span
        dist2 = da * da + dg * dg
        idx = np.argmin(dist2, axis=1)
        # Training samples should be exact grid states. Fail loudly if not.
        matched_a = sa[idx]
        matched_g = sg[idx]
        if np.max(np.abs(matched_a - a1[start:stop])) > 2e-5:
            raise ValueError("could not match training a1 values to separation states")
        if np.max(np.abs(np.log10(matched_g) - np.log10(gamma[start:stop]))) > 2e-5:
            raise ValueError("could not match training gamma values to separation states")
        result[start:stop] = idx
    return result


def select_states_stratified(state_rows, fraction):
    """Keep the top-separation fraction of a1 states independently at each gamma."""
    fraction = float(fraction)
    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in (0,1]")
    by_gamma = {}
    for i, row in enumerate(state_rows):
        by_gamma.setdefault(round(row["gamma"], 12), []).append(i)

    selected = []
    for _, idxs in sorted(by_gamma.items()):
        idxs = sorted(idxs, key=lambda i: state_rows[i]["snr_sep_rms"], reverse=True)
        keep = max(1, int(math.ceil(fraction * len(idxs))))
        selected.extend(idxs[:keep])
    return np.asarray(sorted(set(selected)), dtype=np.int32)


def stage_for_epoch(epoch, stage_epochs):
    e1, e2, e3 = stage_epochs
    if epoch <= e1:
        return "easy"
    if epoch <= e1 + e2:
        return "medium"
    return "full"


def build_epoch_loader(dataset, indices, batch_size, num_workers, device, seed, epoch):
    subset = Subset(dataset, [int(i) for i in indices])
    if len(subset) == 0:
        raise ValueError("empty curriculum subset")
    gen = torch.Generator()
    gen.manual_seed(int(seed) + int(epoch) * 1009)
    # Same number of samples / updates every epoch as full training.
    sampler = RandomSampler(
        subset,
        replacement=True,
        num_samples=len(dataset),
        generator=gen,
    )
    return DataLoader(
        subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def sample_summary(indices, sample_state, state_rows):
    state_ids = sorted(set(int(sample_state[i]) for i in indices))
    scores = np.asarray([state_rows[i]["snr_sep_rms"] for i in state_ids], dtype=np.float64)
    gammas = sorted(set(round(state_rows[i]["gamma"], 12) for i in state_ids))
    return {
        "sample_count_available": int(len(indices)),
        "state_count": int(len(state_ids)),
        "gamma_anchor_count": int(len(gammas)),
        "min_state_rms_snr": float(np.min(scores)),
        "median_state_rms_snr": float(np.median(scores)),
        "max_state_rms_snr": float(np.max(scores)),
    }


def main():
    args = parse_args()
    if not (0 < args.easy_fraction_per_gamma <= args.medium_fraction_per_gamma <= 1):
        raise ValueError("require 0 < easy_fraction <= medium_fraction <= 1")

    set_seed(args.seed)
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("mode") != "a1gamma":
        raise ValueError("Exp25 requires a1gamma data")

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

    state_rows = load_state_separation(data_dir / "state_nearest_neighbor_separation.csv")
    sample_state = match_samples_to_states(train_ds.second, train_ds.gamma, state_rows)

    easy_states = select_states_stratified(state_rows, args.easy_fraction_per_gamma)
    medium_states = select_states_stratified(state_rows, args.medium_fraction_per_gamma)
    full_states = np.arange(len(state_rows), dtype=np.int32)

    easy_idx = np.where(np.isin(sample_state, easy_states))[0]
    medium_idx = np.where(np.isin(sample_state, medium_states))[0]
    full_idx = np.arange(len(train_ds), dtype=np.int64)

    stage_indices = {
        "easy": easy_idx,
        "medium": medium_idx,
        "full": full_idx,
    }

    if args.strategy == "full":
        stage_indices = {"easy": full_idx, "medium": full_idx, "full": full_idx}

    stage_info = {
        name: sample_summary(idx, sample_state, state_rows)
        for name, idx in stage_indices.items()
    }
    write_csv(
        output_dir / "curriculum_stage_summary.csv",
        [{"stage": k, **v} for k, v in stage_info.items()],
    )

    second_values = np.asarray(metadata["second_train_values"], dtype=np.float64)
    second_min = float(second_values.min())
    second_max = float(second_values.max())
    second_span = second_max - second_min

    mean, scale, global_rms = training_input_normalization(train_ds.gy)
    model = PairwiseInverseMLP(
        input_points=train_ds.gy.shape[1],
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        second_min=second_min,
        second_max=second_max,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
    ).to(device)
    model.set_input_normalization(mean, scale)

    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_epochs = int(sum(args.stage_epochs))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs, 1),
        eta_min=max(args.lr * 1e-3, 1e-7),
    )

    print("=" * 100)
    print("Exp25 continuous g-space curriculum")
    print("strategy          : %s" % args.strategy)
    print("data              : %s" % data_dir)
    print("train / val       : %d / %d" % (len(train_ds), len(val_ds)))
    print("input points      : %d" % train_ds.gy.shape[1])
    print("stage epochs      : %s" % (args.stage_epochs,))
    print("easy fraction     : %.3f per gamma" % args.easy_fraction_per_gamma)
    print("medium fraction   : %.3f per gamma" % args.medium_fraction_per_gamma)
    print("global input RMS  : %.10g" % global_rms)
    for stage in ("easy", "medium", "full"):
        print("  %-6s -> %s" % (stage, stage_info[stage]))
    print("=" * 100)

    best_val = math.inf
    best_epoch = 0
    best_stage = ""
    checkpoint = output_dir / "best_exp25_mlp.pt"
    history = []

    for epoch in range(1, total_epochs + 1):
        stage = stage_for_epoch(epoch, args.stage_epochs)
        loader = build_epoch_loader(
            train_ds,
            stage_indices[stage],
            args.batch_size,
            args.num_workers,
            device,
            args.seed,
            epoch,
        )
        train_loss = train_one_epoch(
            model, loader, optimizer, device, second_span,
            args.second_loss_weight, args.gamma_loss_weight,
        )
        val_loss, vm = evaluate_loader(
            model, val_loader, device, second_span,
            args.second_loss_weight, args.gamma_loss_weight,
        )
        lr_now = optimizer.param_groups[0]["lr"]
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_stage = stage
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "stage": stage,
                "val_loss": val_loss,
                "strategy": args.strategy,
            }, checkpoint)

        history.append({
            "epoch": epoch,
            "stage": stage,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "gamma_median_factor": vm["gamma_median_factor"],
            "gamma_within_x1p2": vm["gamma_within_x1p2"],
            "a1_median_norm_error": vm["second_median_norm_error"],
            "lr": lr_now,
            "is_best": int(improved),
        })
        if improved or epoch % args.log_every == 0 or epoch in (
            args.stage_epochs[0],
            args.stage_epochs[0] + 1,
            args.stage_epochs[0] + args.stage_epochs[1],
            args.stage_epochs[0] + args.stage_epochs[1] + 1,
        ):
            mark = "*" if improved else " "
            print(
                "%s epoch %3d | %-6s | train %.6g | val %.6g | gamma med %.4f | <=x1.2 %.3f"
                % (
                    mark, epoch, stage, train_loss, val_loss,
                    vm["gamma_median_factor"], vm["gamma_within_x1p2"],
                )
            )
        scheduler.step()

    write_csv(output_dir / "training_history.csv", history)

    try:
        ck = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        ck = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ck["model_state"])

    summary_rows = []
    sample_rows = []
    for noise_dir in discover_noise_dirs(data_dir):
        for split in ("test_seen", "test_interp_gamma", "test_interp_second", "test_interp_both"):
            path = noise_dir / (split + ".npz")
            if not path.exists():
                continue
            ds = PairwiseNpzDataset(
                path,
                mode="a1gamma",
                max_samples=args.max_test_samples,
            )
            pred = predict_dataset(
                model,
                ds,
                mode="a1gamma",
                device=device,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                compute_physics=not args.skip_physics_metrics,
                physics_integration_points=args.physics_integration_points,
            )
            m = metrics(
                pred["pred_second"], pred["true_second"],
                pred["pred_gamma"], pred["true_gamma"],
                second_span,
            )
            summary_rows.append({
                "strategy": args.strategy,
                "noise_dir": noise_dir.name,
                "noise_level": float(np.nanmedian(pred["noise_level"])),
                "split": split,
                **m,
                "median_f_relative_l2": float(np.nanmedian(pred["f_relative_l2"])),
                "median_g_clean_relative_l2": float(np.nanmedian(pred["g_clean_relative_l2"])),
                "catastrophic_gamma_factor_ge5": float(np.mean(pred["gamma_factor_error"] >= 5.0)),
                "catastrophic_gamma_factor_ge10": float(np.mean(pred["gamma_factor_error"] >= 10.0)),
            })
            for i in range(len(pred["true_gamma"])):
                sample_rows.append({
                    "strategy": args.strategy,
                    "noise_dir": noise_dir.name,
                    "noise_level": float(pred["noise_level"][i]),
                    "split": split,
                    "true_a1": float(pred["true_second"][i]),
                    "pred_a1": float(pred["pred_second"][i]),
                    "true_gamma": float(pred["true_gamma"][i]),
                    "pred_gamma": float(pred["pred_gamma"][i]),
                    "gamma_factor_error": float(pred["gamma_factor_error"][i]),
                    "f_relative_l2": float(pred["f_relative_l2"][i]),
                    "g_clean_relative_l2": float(pred["g_clean_relative_l2"][i]),
                })

    prefix = str(args.result_prefix).strip() or "exp25"
    write_csv(output_dir / (prefix + "_summary.csv"), summary_rows)
    write_csv(output_dir / (prefix + "_samples.csv"), sample_rows)
    (output_dir / (prefix + "_metadata.json")).write_text(
        json.dumps({
            "experiment": "Exp25 continuous g-space curriculum",
            "strategy": args.strategy,
            "best_epoch": best_epoch,
            "best_stage": best_stage,
            "best_val_loss": best_val,
            "stage_info": stage_info,
            "args": vars(args),
            "dataset_metadata": metadata,
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("=" * 100)
    print("Exp25 %s finished | best epoch=%d stage=%s" % (args.strategy, best_epoch, best_stage))
    print("Read first:")
    print("  %s" % (output_dir / (prefix + "_summary.csv")))
    print("  %s" % (output_dir / (prefix + "_samples.csv")))
    print("=" * 100)


if __name__ == "__main__":
    main()
