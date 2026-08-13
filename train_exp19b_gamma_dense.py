#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp19B: dense gamma-only curriculum regression.

Scientific question
-------------------
With a1, a2, a3 and m fixed, can denser log-spaced gamma coverage turn the\nprevious five-anchor recognition problem into a stable continuous g(q^2) -> gamma map?

This script:
1. trains on a dense-log gamma-only dataset at noise_0p2pct by default;
2. predicts log10(gamma) with a small MLP;
3. uses fixed per-q mean/std from the training set (no per-sample normalization);
4. evaluates the same best checkpoint on every available noise_*/test_seen.npz
   and noise_*/test_interp.npz;
5. reports gamma factor error and optional physics-space f/g reconstruction
   errors.

No core physical formula is modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from GammaOnlyMLP import GammaOnlyMLP


class GammaNpzDataset(Dataset):
    """Small in-memory dataset for the gamma curriculum NPZ files."""

    def __init__(self, path: str | Path, max_samples: int = 0):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        with np.load(self.path, allow_pickle=False) as data:
            required = ("gy_noisy", "gamma")
            missing = [key for key in required if key not in data]
            if missing:
                raise KeyError(f"{self.path}: missing keys {missing}")

            self.gy = np.asarray(data["gy_noisy"], dtype=np.float32)
            self.gamma = np.asarray(data["gamma"], dtype=np.float32).reshape(-1)
            self.gy_clean = (
                np.asarray(data["gy_clean"], dtype=np.float32)
                if "gy_clean" in data
                else None
            )
            self.fx = (
                np.asarray(data["fx"], dtype=np.float32)
                if "fx" in data
                else None
            )
            self.parameters = (
                np.asarray(data["parameters"], dtype=np.float32)
                if "parameters" in data
                else None
            )
            self.gamma_class = (
                np.asarray(data["gamma_class"], dtype=np.int64).reshape(-1)
                if "gamma_class" in data
                else np.full(len(self.gamma), -1, dtype=np.int64)
            )
            self.noise_level = (
                np.asarray(data["noise_level"], dtype=np.float32).reshape(-1)
                if "noise_level" in data
                else np.full(len(self.gamma), np.nan, dtype=np.float32)
            )

        n = len(self.gamma)
        if self.gy.ndim != 2 or self.gy.shape[0] != n:
            raise ValueError(f"{self.path}: gy_noisy must have shape [N,Q]")
        if np.any(~np.isfinite(self.gy)) or np.any(~np.isfinite(self.gamma)):
            raise FloatingPointError(f"{self.path}: non-finite gy/gamma values")
        if np.any(self.gamma <= 0.0):
            raise ValueError(f"{self.path}: gamma must be positive")

        if max_samples and max_samples > 0 and n > int(max_samples):
            keep = np.arange(int(max_samples), dtype=np.int64)
            self._take(keep)

    def _take(self, indices: np.ndarray) -> None:
        self.gy = self.gy[indices]
        self.gamma = self.gamma[indices]
        self.gamma_class = self.gamma_class[indices]
        self.noise_level = self.noise_level[indices]
        if self.gy_clean is not None:
            self.gy_clean = self.gy_clean[indices]
        if self.fx is not None:
            self.fx = self.fx[indices]
        if self.parameters is not None:
            self.parameters = self.parameters[indices]

    def __len__(self) -> int:
        return len(self.gamma)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "gy": torch.from_numpy(self.gy[index]),
            "gamma": torch.tensor(self.gamma[index], dtype=torch.float32),
            "gamma_class": torch.tensor(self.gamma_class[index], dtype=torch.long),
            "noise_level": torch.tensor(self.noise_level[index], dtype=torch.float32),
        }
        if self.gy_clean is not None:
            item["gy_clean"] = torch.from_numpy(self.gy_clean[index])
        if self.fx is not None:
            item["fx"] = torch.from_numpy(self.fx[index])
        if self.parameters is not None:
            item["parameters"] = torch.from_numpy(self.parameters[index])
        return item


def parse_hidden_sizes(text: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("hidden sizes must be positive comma-separated integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp19B dense gamma-only regression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="./data_gamma_dense_11")
    parser.add_argument("--train-noise-dir", default="noise_0p2pct")
    parser.add_argument("--output-dir", default="./validation_results/exp19b_gamma_dense_11")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260813)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--hidden-sizes", type=parse_hidden_sizes, default=(256, 256, 128))
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)

    parser.add_argument("--gamma-min", type=float, default=1e-3)
    parser.add_argument("--gamma-max", type=float, default=1.0)

    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)

    parser.add_argument(
        "--skip-physics-metrics",
        action="store_true",
        help="Skip predicted f/g reconstruction metrics; useful for a fast smoke test.",
    )
    parser.add_argument(
        "--physics-integration-points",
        type=int,
        default=512,
        help="Quadrature points for predicted g during final evaluation.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(text: str) -> torch.device:
    text = str(text)
    if text.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(text)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load our own checkpoint without the torch.load future-warning when supported."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch releases that do not expose weights_only.
        return torch.load(path, map_location=device)


def training_input_normalization(
    array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(array, dtype=np.float64)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    global_rms = float(np.sqrt(np.mean(x**2)))
    floor = max(global_rms * 1e-6, 1e-12)
    scale = np.maximum(scale, floor)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("invalid training input normalization")
    return mean.astype(np.float32), scale.astype(np.float32), global_rms


def gamma_factor_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred64 = np.asarray(pred, dtype=np.float64)
    true64 = np.asarray(true, dtype=np.float64)
    return np.maximum(
        pred64 / np.clip(true64, 1e-30, None),
        true64 / np.clip(pred64, 1e-30, None),
    )


def factor_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    ferr = gamma_factor_error(pred, true)
    log_diff = np.log10(np.asarray(pred, dtype=np.float64)) - np.log10(
        np.asarray(true, dtype=np.float64)
    )
    return {
        "count": int(len(ferr)),
        "median_factor_error": float(np.median(ferr)),
        "p90_factor_error": float(np.quantile(ferr, 0.90)),
        "within_x1p1": float(np.mean(ferr <= 1.1)),
        "within_x1p2": float(np.mean(ferr <= 1.2)),
        "within_x1p5": float(np.mean(ferr <= 1.5)),
        "within_x2": float(np.mean(ferr <= 2.0)),
        "log10_mae": float(np.mean(np.abs(log_diff))),
        "log10_rmse": float(np.sqrt(np.mean(log_diff**2))),
    }


@torch.no_grad()
def predict_dataset(
    model: GammaOnlyMLP,
    dataset: GammaNpzDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    compute_physics: bool,
    physics_integration_points: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []
    all_noise: list[np.ndarray] = []
    all_f_rel: list[np.ndarray] = []
    all_g_rel: list[np.ndarray] = []

    physics_config = None
    scaled_curves_torch = None
    if compute_physics:
        if dataset.fx is None or dataset.gy_clean is None or dataset.parameters is None:
            raise KeyError(
                f"{dataset.path}: physics metrics require fx, gy_clean and parameters"
            )
        from mc_pool_config import DEFAULT_PHYSICS
        from mc_physics import scaled_curves_torch as _scaled_curves_torch

        scaled_curves_torch = _scaled_curves_torch
        physics_config = replace(
            DEFAULT_PHYSICS,
            q2_points=int(dataset.gy.shape[1]),
            output_points=int(dataset.fx.shape[1]),
        )

    model.eval()
    for batch in loader:
        gy = batch["gy"].to(device, non_blocking=True)
        log_pred = model(gy)
        pred_gamma = torch.pow(10.0, log_pred)
        true_gamma = batch["gamma"].to(device, non_blocking=True)

        all_true.append(true_gamma.detach().cpu().numpy())
        all_pred.append(pred_gamma.detach().cpu().numpy())
        all_classes.append(batch["gamma_class"].numpy())
        all_noise.append(batch["noise_level"].numpy())

        if compute_physics:
            params = batch["parameters"].to(device, non_blocking=True).clone()
            params[:, 4] = pred_gamma
            pred_f, pred_g = scaled_curves_torch(
                params,
                integration_points=int(physics_integration_points),
                config=physics_config,
            )
            true_f = batch["fx"].to(device, non_blocking=True)
            true_g = batch["gy_clean"].to(device, non_blocking=True)

            f_rel = torch.linalg.vector_norm(pred_f - true_f, dim=1) / (
                torch.linalg.vector_norm(true_f, dim=1).clamp_min(1e-12)
            )
            g_rel = torch.linalg.vector_norm(pred_g - true_g, dim=1) / (
                torch.linalg.vector_norm(true_g, dim=1).clamp_min(1e-12)
            )
            all_f_rel.append(f_rel.detach().cpu().numpy())
            all_g_rel.append(g_rel.detach().cpu().numpy())

    result = {
        "true_gamma": np.concatenate(all_true),
        "pred_gamma": np.concatenate(all_pred),
        "gamma_class": np.concatenate(all_classes),
        "noise_level": np.concatenate(all_noise),
    }
    result["factor_error"] = gamma_factor_error(
        result["pred_gamma"], result["true_gamma"]
    ).astype(np.float64)
    if compute_physics:
        result["f_relative_l2"] = np.concatenate(all_f_rel).astype(np.float64)
        result["g_clean_relative_l2"] = np.concatenate(all_g_rel).astype(np.float64)
    else:
        result["f_relative_l2"] = np.full(len(result["true_gamma"]), np.nan)
        result["g_clean_relative_l2"] = np.full(len(result["true_gamma"]), np.nan)
    return result


@torch.no_grad()
def validation_loss_and_metrics(
    model: GammaOnlyMLP,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    sum_loss = 0.0
    count = 0
    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []

    for batch in loader:
        gy = batch["gy"].to(device, non_blocking=True)
        gamma = batch["gamma"].to(device, non_blocking=True)
        target_log = torch.log10(gamma)
        pred_log = model(gy)
        per_item = (pred_log - target_log).square()
        sum_loss += float(per_item.sum().item())
        count += int(len(gamma))
        pred_parts.append(torch.pow(10.0, pred_log).cpu().numpy())
        true_parts.append(gamma.cpu().numpy())

    pred = np.concatenate(pred_parts)
    true = np.concatenate(true_parts)
    return sum_loss / max(count, 1), factor_metrics(pred, true)


def train_one_epoch(
    model: GammaOnlyMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    sum_loss = 0.0
    count = 0

    for batch in loader:
        gy = batch["gy"].to(device, non_blocking=True)
        gamma = batch["gamma"].to(device, non_blocking=True)
        target_log = torch.log10(gamma)

        optimizer.zero_grad(set_to_none=True)
        pred_log = model(gy)
        loss = torch.mean((pred_log - target_log).square())
        loss.backward()
        optimizer.step()

        sum_loss += float(loss.item()) * len(gamma)
        count += int(len(gamma))

    return sum_loss / max(count, 1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_noise_dirs(data_dir: Path) -> list[Path]:
    result: list[tuple[float, Path]] = []
    for path in data_dir.glob("noise_*"):
        if not path.is_dir():
            continue
        probe = path / "test_seen.npz"
        if not probe.exists():
            continue
        with np.load(probe, allow_pickle=False) as data:
            level = (
                float(np.median(np.asarray(data["noise_level"], dtype=np.float64)))
                if "noise_level" in data
                else math.inf
            )
        result.append((level, path))
    result.sort(key=lambda x: (x[0], x[1].name))
    return [path for _, path in result]


def add_summary_rows(
    summary_rows: list[dict[str, Any]],
    *,
    result: dict[str, np.ndarray],
    split: str,
    noise_dir: str,
) -> None:
    true_gamma = result["true_gamma"]
    unique_gamma = np.unique(np.round(true_gamma.astype(np.float64), 10))
    groups: list[tuple[str, np.ndarray]] = [
        ("ALL", np.ones(len(true_gamma), dtype=bool))
    ]
    for gamma in unique_gamma:
        groups.append(
            (
                f"{float(gamma):.10g}",
                np.isclose(true_gamma, gamma, rtol=1e-6, atol=1e-8),
            )
        )

    noise_pct = float(np.nanmedian(result["noise_level"])) * 100.0
    for label, mask in groups:
        metrics = factor_metrics(
            result["pred_gamma"][mask], result["true_gamma"][mask]
        )
        f_values = result["f_relative_l2"][mask]
        g_values = result["g_clean_relative_l2"][mask]
        row: dict[str, Any] = {
            "noise_dir": noise_dir,
            "noise_percent": noise_pct,
            "split": split,
            "true_gamma_group": label,
            **metrics,
            "median_f_relative_l2": (
                float(np.nanmedian(f_values))
                if np.any(np.isfinite(f_values))
                else math.nan
            ),
            "median_g_clean_relative_l2": (
                float(np.nanmedian(g_values))
                if np.any(np.isfinite(g_values))
                else math.nan
            ),
        }
        summary_rows.append(row)


def save_prediction_rows(
    sample_rows: list[dict[str, Any]],
    *,
    result: dict[str, np.ndarray],
    split: str,
    noise_dir: str,
) -> None:
    for i in range(len(result["true_gamma"])):
        sample_rows.append(
            {
                "noise_dir": noise_dir,
                "noise_percent": float(result["noise_level"][i]) * 100.0,
                "split": split,
                "sample_index": i,
                "gamma_class": int(result["gamma_class"][i]),
                "true_gamma": float(result["true_gamma"][i]),
                "pred_gamma": float(result["pred_gamma"][i]),
                "factor_error": float(result["factor_error"][i]),
                "f_relative_l2": float(result["f_relative_l2"][i]),
                "g_clean_relative_l2": float(result["g_clean_relative_l2"][i]),
            }
        )


def plot_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warning] history plot skipped: {exc}")
        return

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_log_mse"]) for row in history]
    val_loss = [float(row["val_log_mse"]) for row in history]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(epochs, train_loss, label="train")
    ax.plot(epochs, val_loss, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE on log10(gamma)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=160)
    plt.close(fig)


def plot_predictions(
    sample_rows: list[dict[str, Any]],
    output_dir: Path,
    split: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warning] prediction plot skipped: {exc}")
        return

    rows = [row for row in sample_rows if row["split"] == split]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    for noise_dir in sorted({row["noise_dir"] for row in rows}):
        part = [row for row in rows if row["noise_dir"] == noise_dir]
        ax.scatter(
            [row["true_gamma"] for row in part],
            [row["pred_gamma"] for row in part],
            s=10,
            alpha=0.30,
            label=noise_dir,
        )
    lo = min(min(row["true_gamma"], row["pred_gamma"]) for row in rows)
    hi = max(max(row["true_gamma"], row["pred_gamma"]) for row in rows)
    lo = max(lo, 1e-4)
    ax.plot([lo, hi], [lo, hi], linestyle="--", label="ideal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true gamma")
    ax.set_ylabel("predicted gamma")
    ax.set_title(split)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"pred_vs_true_{split}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    data_dir = Path(args.data_dir)
    train_dir = data_dir / args.train_noise_dir
    train_path = train_dir / "train.npz"
    val_path = train_dir / "val.npz"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"expected {train_path} and {val_path}; run/check the gamma curriculum data first"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_gamma_only_dense_mlp.pt"

    train_ds = GammaNpzDataset(train_path, args.max_train_samples)
    val_ds = GammaNpzDataset(val_path, args.max_val_samples)
    if train_ds.gy.shape[1] != val_ds.gy.shape[1]:
        raise ValueError("train/val input point counts differ")

    input_points = int(train_ds.gy.shape[1])
    input_mean, input_scale, global_rms = training_input_normalization(train_ds.gy)

    model = GammaOnlyMLP(
        input_points=input_points,
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
    ).to(device)
    model.set_input_normalization(input_mean, input_scale)

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
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(args.epochs), 1),
    )

    print("=" * 96)
    print("Exp19B: dense gamma-only regression")
    print("NO change to physical formula")
    print(f"device           : {device}")
    print(f"data             : {data_dir}")
    print(f"training noise   : {args.train_noise_dir}")
    print(f"train / val      : {len(train_ds)} / {len(val_ds)}")
    print(f"input points     : {input_points}")
    print(f"global input RMS : {global_rms:.9g}")
    print(
        "feature std     : "
        f"min={float(np.min(input_scale)):.3g}, "
        f"median={float(np.median(input_scale)):.3g}, "
        f"max={float(np.max(input_scale)):.3g}"
    )
    print(f"hidden sizes     : {args.hidden_sizes}")
    print(f"gamma bounds     : [{args.gamma_min:g}, {args.gamma_max:g}]")
    print("=" * 96)

    history: list[dict[str, Any]] = []
    best_val = math.inf
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = validation_loss_and_metrics(model, val_loader, device)
        lr_now = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": epoch,
                "train_log_mse": train_loss,
                "val_log_mse": val_loss,
                "val_median_factor_error": val_metrics["median_factor_error"],
                "val_p90_factor_error": val_metrics["p90_factor_error"],
                "val_within_x1p2": val_metrics["within_x1p2"],
                "lr": lr_now,
            }
        )

        improved = val_loss < (best_val - args.min_delta)
        if improved:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_points": input_points,
                    "hidden_sizes": list(args.hidden_sizes),
                    "dropout": float(args.dropout),
                    "gamma_min": float(args.gamma_min),
                    "gamma_max": float(args.gamma_max),
                    "input_mean": input_mean.tolist(),
                    "input_scale": input_scale.tolist(),
                    "global_input_rms": float(global_rms),
                    "best_epoch": int(best_epoch),
                    "best_val_log_mse": float(best_val),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1

        if (
            epoch == 1
            or epoch % max(args.log_every, 1) == 0
            or improved
            or epoch == args.epochs
        ):
            mark = "*" if improved else " "
            print(
                f"{mark} epoch {epoch:4d} | "
                f"train {train_loss:.6g} | val {val_loss:.6g} | "
                f"med factor {val_metrics['median_factor_error']:.4f} | "
                f"<=x1.2 {val_metrics['within_x1p2']:.3f} | lr {lr_now:.3g}"
            )

        scheduler.step()
        if stale_epochs >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}, val log-MSE={best_val:.6g}"
            )
            break

    write_csv(output_dir / "training_history.csv", history)
    plot_history(history, output_dir)

    if not checkpoint_path.exists():
        raise RuntimeError("best checkpoint was not created")
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    summary_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    noise_dirs = discover_noise_dirs(data_dir)
    if not noise_dirs:
        raise RuntimeError(f"no usable noise_* directories found under {data_dir}")

    compute_physics = not args.skip_physics_metrics
    print("-" * 96)
    print(
        "Final evaluation with best checkpoint: "
        f"epoch={checkpoint['best_epoch']}, physics_metrics={compute_physics}"
    )

    for noise_dir in noise_dirs:
        for split in ("test_seen", "test_interp"):
            path = noise_dir / f"{split}.npz"
            if not path.exists():
                print(f"[warning] missing {path}; skipped")
                continue
            dataset = GammaNpzDataset(path, args.max_test_samples)
            result = predict_dataset(
                model,
                dataset,
                device=device,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                compute_physics=compute_physics,
                physics_integration_points=args.physics_integration_points,
            )
            add_summary_rows(
                summary_rows,
                result=result,
                split=split,
                noise_dir=noise_dir.name,
            )
            save_prediction_rows(
                sample_rows,
                result=result,
                split=split,
                noise_dir=noise_dir.name,
            )
            overall = factor_metrics(result["pred_gamma"], result["true_gamma"])
            print(
                f"{noise_dir.name:14s} | {split:11s} | "
                f"med factor={overall['median_factor_error']:.4f} | "
                f"p90={overall['p90_factor_error']:.4f} | "
                f"<=x1.2={overall['within_x1p2']:.3f} | "
                f"<=x2={overall['within_x2']:.3f}"
            )

    write_csv(output_dir / "exp19b_summary.csv", summary_rows)
    write_csv(output_dir / "exp19b_samples.csv", sample_rows)
    plot_predictions(sample_rows, output_dir, "test_seen")
    plot_predictions(sample_rows, output_dir, "test_interp")

    metadata = {
        "experiment": "Exp19B dense gamma-only regression",
        "scientific_question": (
            "Can denser log-spaced gamma coverage produce stable continuous "
            "g(q^2)->gamma interpolation when a1,a2,a3,m are fixed?"
        ),
        "physical_formula_changed": False,
        "args": vars(args),
        "input_points": input_points,
        "input_normalization": "fixed per-q train-set mean/std (not per-sample)",
        "global_input_rms": global_rms,
        "feature_std_min": float(np.min(input_scale)),
        "feature_std_median": float(np.median(input_scale)),
        "feature_std_max": float(np.max(input_scale)),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_log_mse": float(checkpoint["best_val_log_mse"]),
        "train_unique_gamma": sorted(
            float(x) for x in np.unique(train_ds.gamma.astype(np.float64))
        ),
        "outputs": [
            "best_gamma_only_dense_mlp.pt",
            "training_history.csv",
            "exp19b_summary.csv",
            "exp19b_samples.csv",
            "training_history.png",
            "pred_vs_true_test_seen.png",
            "pred_vs_true_test_interp.png",
        ],
    }
    with (output_dir / "exp19b_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print("=" * 96)
    print("Exp19B finished.")
    print(f"output: {output_dir}")
    print("Read first:")
    print(f"  {output_dir / 'exp19b_summary.csv'}")
    print(f"  {output_dir / 'exp19b_samples.csv'}")
    print("=" * 96)


if __name__ == "__main__":
    main()
