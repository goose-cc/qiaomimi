#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train Exp20A/Exp20B/Exp20C pairwise curriculum models.

Exp20A (mode=a1gamma):
    g(q^2) -> a1, gamma

Exp20B (mode=mgamma):
    g(q^2) -> m, gamma

Exp20C reuses mode=a1gamma with denser a1 anchors.  The runner keeps the
training sample count near 20k so the controlled change is a1 coverage.

The MLP backbone, fixed train-set input normalization and 0.2% default training
noise intentionally stay close to Exp19B.  The controlled change is adding one
extra unknown parameter.
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

from PairwiseInverseMLP import PairwiseInverseMLP


class PairwiseNpzDataset(Dataset):
    def __init__(self, path: str | Path, *, mode: str, max_samples: int = 0):
        self.path = Path(path)
        self.mode = mode
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        second_key = "a1" if mode == "a1gamma" else "m"

        with np.load(self.path, allow_pickle=False) as data:
            required = ("gy_noisy", "gamma", second_key)
            missing = [k for k in required if k not in data]
            if missing:
                raise KeyError(f"{self.path}: missing keys {missing}")
            self.gy = np.asarray(data["gy_noisy"], dtype=np.float32)
            self.gamma = np.asarray(data["gamma"], dtype=np.float32).reshape(-1)
            self.second = np.asarray(data[second_key], dtype=np.float32).reshape(-1)
            self.gy_clean = np.asarray(data["gy_clean"], dtype=np.float32) if "gy_clean" in data else None
            self.fx = np.asarray(data["fx"], dtype=np.float32) if "fx" in data else None
            self.parameters = np.asarray(data["parameters"], dtype=np.float32) if "parameters" in data else None
            self.noise_level = np.asarray(data["noise_level"], dtype=np.float32).reshape(-1) if "noise_level" in data else np.full(len(self.gamma), np.nan, np.float32)
            self.x_grid = np.asarray(data["x"], dtype=np.float32).reshape(-1) if "x" in data else None
            self.q2_grid = np.asarray(data["q2"], dtype=np.float32).reshape(-1) if "q2" in data else (
                np.asarray(data["y"], dtype=np.float32).reshape(-1) if "y" in data else None
            )

        n = len(self.gamma)
        if self.gy.ndim != 2 or self.gy.shape[0] != n:
            raise ValueError(f"{self.path}: gy_noisy must be [N,Q]")
        if len(self.second) != n:
            raise ValueError(f"{self.path}: second parameter length mismatch")
        if np.any(~np.isfinite(self.gy)) or np.any(~np.isfinite(self.gamma)) or np.any(~np.isfinite(self.second)):
            raise FloatingPointError(f"{self.path}: non-finite values")
        if np.any(self.gamma <= 0):
            raise ValueError(f"{self.path}: gamma must be positive")

        if max_samples and n > int(max_samples):
            idx = np.arange(int(max_samples))
            self.gy = self.gy[idx]
            self.gamma = self.gamma[idx]
            self.second = self.second[idx]
            self.noise_level = self.noise_level[idx]
            if self.gy_clean is not None:
                self.gy_clean = self.gy_clean[idx]
            if self.fx is not None:
                self.fx = self.fx[idx]
            if self.parameters is not None:
                self.parameters = self.parameters[idx]

    def __len__(self) -> int:
        return len(self.gamma)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        out = {
            "gy": torch.from_numpy(self.gy[i]),
            "gamma": torch.tensor(self.gamma[i], dtype=torch.float32),
            "second": torch.tensor(self.second[i], dtype=torch.float32),
            "noise_level": torch.tensor(self.noise_level[i], dtype=torch.float32),
        }
        if self.gy_clean is not None:
            out["gy_clean"] = torch.from_numpy(self.gy_clean[i])
        if self.fx is not None:
            out["fx"] = torch.from_numpy(self.fx[i])
        if self.parameters is not None:
            out["parameters"] = torch.from_numpy(self.parameters[i])
        return out


def parse_hidden_sizes(text: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("hidden sizes must be positive integers")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Exp20 pairwise curriculum MLP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=("a1gamma", "mgamma"), required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--train-noise-dir", default="noise_0p2pct")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--result-prefix", default="exp20", help="prefix for summary/samples/metadata output files")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260813)
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
    p.add_argument("--gamma-min", type=float, default=1e-3)
    p.add_argument("--gamma-max", type=float, default=1.0)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--skip-physics-metrics", action="store_true")
    p.add_argument("--physics-integration-points", type=int, default=512)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(text: str) -> torch.device:
    if text.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(text)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def training_input_normalization(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(array, dtype=np.float64)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    global_rms = float(np.sqrt(np.mean(x**2)))
    floor = max(global_rms * 1e-6, 1e-12)
    scale = np.maximum(scale, floor)
    return mean.astype(np.float32), scale.astype(np.float32), global_rms


def gamma_factor_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    p = np.asarray(pred, np.float64)
    t = np.asarray(true, np.float64)
    return np.maximum(p / np.clip(t, 1e-30, None), t / np.clip(p, 1e-30, None))


def metrics(pred_second, true_second, pred_gamma, true_gamma, second_span) -> dict[str, float]:
    ferr = gamma_factor_error(pred_gamma, true_gamma)
    second_abs = np.abs(np.asarray(pred_second, np.float64) - np.asarray(true_second, np.float64))
    second_norm = second_abs / max(float(second_span), 1e-30)
    log_diff = np.log10(np.asarray(pred_gamma, np.float64)) - np.log10(np.asarray(true_gamma, np.float64))
    return {
        "count": int(len(ferr)),
        "gamma_median_factor": float(np.median(ferr)),
        "gamma_p90_factor": float(np.quantile(ferr, 0.90)),
        "gamma_within_x1p1": float(np.mean(ferr <= 1.1)),
        "gamma_within_x1p2": float(np.mean(ferr <= 1.2)),
        "gamma_within_x1p5": float(np.mean(ferr <= 1.5)),
        "gamma_within_x2": float(np.mean(ferr <= 2.0)),
        "gamma_log10_mae": float(np.mean(np.abs(log_diff))),
        "second_median_abs_error": float(np.median(second_abs)),
        "second_p90_abs_error": float(np.quantile(second_abs, 0.90)),
        "second_median_norm_error": float(np.median(second_norm)),
        "second_p90_norm_error": float(np.quantile(second_norm, 0.90)),
    }


def train_one_epoch(model, loader, optimizer, device, second_span, second_w, gamma_w):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        gy = batch["gy"].to(device, non_blocking=True)
        second = batch["second"].to(device, non_blocking=True)
        gamma = batch["gamma"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred_second, pred_log_gamma = model(gy)
        second_loss = torch.mean(((pred_second - second) / second_span) ** 2)
        gamma_loss = torch.mean((pred_log_gamma - torch.log10(gamma)) ** 2)
        loss = second_w * second_loss + gamma_w * gamma_loss
        loss.backward()
        optimizer.step()

        total += float(loss.item()) * len(gamma)
        n += len(gamma)
    return total / max(n, 1)


@torch.no_grad()
def evaluate_loader(model, loader, device, second_span, second_w, gamma_w):
    model.eval()
    total = 0.0
    n = 0
    ps, ts, pg, tg = [], [], [], []
    for batch in loader:
        gy = batch["gy"].to(device, non_blocking=True)
        second = batch["second"].to(device, non_blocking=True)
        gamma = batch["gamma"].to(device, non_blocking=True)
        pred_second, pred_log_gamma = model(gy)
        second_loss = torch.mean(((pred_second - second) / second_span) ** 2)
        gamma_loss = torch.mean((pred_log_gamma - torch.log10(gamma)) ** 2)
        loss = second_w * second_loss + gamma_w * gamma_loss
        total += float(loss.item()) * len(gamma)
        n += len(gamma)
        ps.append(pred_second.cpu().numpy())
        ts.append(second.cpu().numpy())
        pg.append(torch.pow(10.0, pred_log_gamma).cpu().numpy())
        tg.append(gamma.cpu().numpy())
    ps = np.concatenate(ps); ts = np.concatenate(ts)
    pg = np.concatenate(pg); tg = np.concatenate(tg)
    return total / max(n, 1), metrics(ps, ts, pg, tg, second_span)


@torch.no_grad()
def predict_dataset(
    model,
    dataset,
    *,
    mode,
    device,
    batch_size,
    num_workers,
    compute_physics,
    physics_integration_points,
):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    ps, ts, pg, tg, noises = [], [], [], [], []
    f_rel_parts, g_rel_parts = [], []
    physics_config = None
    scaled_curves_torch = None

    if compute_physics:
        if dataset.fx is None or dataset.gy_clean is None or dataset.parameters is None:
            raise KeyError(f"{dataset.path}: physics metrics need fx, gy_clean, parameters")
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
        pred_second, pred_log_gamma = model(gy)
        pred_gamma = torch.pow(10.0, pred_log_gamma)
        true_second = batch["second"].to(device, non_blocking=True)
        true_gamma = batch["gamma"].to(device, non_blocking=True)

        ps.append(pred_second.cpu().numpy())
        ts.append(true_second.cpu().numpy())
        pg.append(pred_gamma.cpu().numpy())
        tg.append(true_gamma.cpu().numpy())
        noises.append(batch["noise_level"].numpy())

        if compute_physics:
            params = batch["parameters"].to(device, non_blocking=True).clone()
            if mode == "a1gamma":
                params[:, 0] = pred_second
            else:
                params[:, 3] = pred_second
            params[:, 4] = pred_gamma
            pred_f, pred_g = scaled_curves_torch(
                params,
                integration_points=int(physics_integration_points),
                config=physics_config,
            )
            true_f = batch["fx"].to(device, non_blocking=True)
            true_g = batch["gy_clean"].to(device, non_blocking=True)
            f_rel = torch.linalg.vector_norm(pred_f - true_f, dim=1) / torch.linalg.vector_norm(true_f, dim=1).clamp_min(1e-12)
            g_rel = torch.linalg.vector_norm(pred_g - true_g, dim=1) / torch.linalg.vector_norm(true_g, dim=1).clamp_min(1e-12)
            f_rel_parts.append(f_rel.cpu().numpy())
            g_rel_parts.append(g_rel.cpu().numpy())

    result = {
        "pred_second": np.concatenate(ps),
        "true_second": np.concatenate(ts),
        "pred_gamma": np.concatenate(pg),
        "true_gamma": np.concatenate(tg),
        "noise_level": np.concatenate(noises),
    }
    result["gamma_factor_error"] = gamma_factor_error(result["pred_gamma"], result["true_gamma"])
    result["second_abs_error"] = np.abs(result["pred_second"] - result["true_second"]).astype(np.float64)
    if compute_physics:
        result["f_relative_l2"] = np.concatenate(f_rel_parts).astype(np.float64)
        result["g_clean_relative_l2"] = np.concatenate(g_rel_parts).astype(np.float64)
    else:
        result["f_relative_l2"] = np.full(len(result["true_gamma"]), np.nan)
        result["g_clean_relative_l2"] = np.full(len(result["true_gamma"]), np.nan)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty csv: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k); fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def discover_noise_dirs(data_dir: Path) -> list[Path]:
    result = []
    for d in data_dir.glob("noise_*"):
        probe = d / "test_seen.npz"
        if not probe.exists():
            continue
        with np.load(probe, allow_pickle=False) as z:
            level = float(np.median(z["noise_level"])) if "noise_level" in z else math.inf
        result.append((level, d))
    result.sort(key=lambda x: (x[0], x[1].name))
    return [d for _, d in result]


def plot_history(path: Path, history: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    epochs = [r["epoch"] for r in history]
    train = [r["train_loss"] for r in history]
    val = [r["val_loss"] for r in history]
    fig = plt.figure(figsize=(7, 4))
    plt.plot(epochs, train, label="train")
    plt.plot(epochs, val, label="val")
    plt.yscale("log")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_scatter(path: Path, x, y, xlabel, ylabel) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=8, alpha=0.35)
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    plt.plot([lo, hi], [lo, hi], "--", linewidth=1)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)



def representative_indices(factor_error: np.ndarray) -> list[tuple[str, int]]:
    """Choose best/median/worst samples by gamma factor error."""
    ferr = np.asarray(factor_error, dtype=np.float64).reshape(-1)
    if ferr.size == 0:
        return []
    order = np.argsort(ferr, kind="mergesort")
    picks = [
        ("best", int(order[0])),
        ("median", int(order[len(order) // 2])),
        ("worst", int(order[-1])),
    ]
    # Keep labels even if a tiny smoke-test set maps two labels to one sample.
    return picks


def save_representative_curve_plots(
    *,
    output_dir: Path,
    dataset: PairwiseNpzDataset,
    result: dict[str, np.ndarray],
    mode: str,
    second_name: str,
    split: str,
    noise_dir_name: str,
    physics_integration_points: int,
) -> list[dict[str, Any]]:
    """Save true/pred g(q^2) and f(s) curves for best/median/worst samples."""
    if (
        dataset.parameters is None
        or dataset.fx is None
        or dataset.gy_clean is None
        or dataset.x_grid is None
        or dataset.q2_grid is None
    ):
        return []

    try:
        import matplotlib.pyplot as plt
        from mc_pool_config import DEFAULT_PHYSICS
        from mc_physics import scaled_curves_numpy
    except Exception as exc:
        print(f"[curve plots skipped] {exc}")
        return []

    physics = replace(
        DEFAULT_PHYSICS,
        q2_points=int(dataset.gy.shape[1]),
        output_points=int(dataset.fx.shape[1]),
    )

    records: list[dict[str, Any]] = []
    curve_dir = output_dir / "curve_fits" / noise_dir_name / split
    curve_dir.mkdir(parents=True, exist_ok=True)

    for rank_label, i in representative_indices(result["gamma_factor_error"]):
        params = dataset.parameters[i : i + 1].astype(np.float64, copy=True)
        if mode == "a1gamma":
            params[0, 0] = float(result["pred_second"][i])
        else:
            params[0, 3] = float(result["pred_second"][i])
        params[0, 4] = float(result["pred_gamma"][i])

        pred_f, pred_g = scaled_curves_numpy(
            params,
            integration_points=int(physics_integration_points),
            config=physics,
        )
        pred_f = np.asarray(pred_f[0], dtype=np.float64)
        pred_g = np.asarray(pred_g[0], dtype=np.float64)

        true_second = float(result["true_second"][i])
        pred_second = float(result["pred_second"][i])
        true_gamma = float(result["true_gamma"][i])
        pred_gamma = float(result["pred_gamma"][i])
        gamma_factor = float(result["gamma_factor_error"][i])
        f_rel = float(result["f_relative_l2"][i])
        g_rel = float(result["g_clean_relative_l2"][i])

        stem = f"{rank_label}_idx{i:05d}"

        fig = plt.figure(figsize=(7.2, 4.4))
        plt.plot(dataset.q2_grid, dataset.gy[i], ".", markersize=2.5, label="observed g noisy")
        plt.plot(dataset.q2_grid, dataset.gy_clean[i], linewidth=1.6, label="true g clean")
        plt.plot(dataset.q2_grid, pred_g, "--", linewidth=1.4, label="predicted-parameter g")
        plt.xlabel(r"$q^2$")
        plt.ylabel("g")
        plt.title(
            f"{split} | {rank_label}\n"
            f"true {second_name}={true_second:.6g}, pred={pred_second:.6g}; "
            f"true gamma={true_gamma:.6g}, pred={pred_gamma:.6g}\n"
            f"gamma factor={gamma_factor:.4g}, g relL2={g_rel:.4g}"
        )
        plt.legend()
        plt.tight_layout()
        g_path = curve_dir / f"g_fit_{stem}.png"
        fig.savefig(g_path, dpi=170)
        plt.close(fig)

        fig = plt.figure(figsize=(7.2, 4.4))
        plt.plot(dataset.x_grid, dataset.fx[i], linewidth=1.6, label="true f")
        plt.plot(dataset.x_grid, pred_f, "--", linewidth=1.4, label="predicted f")
        plt.xlabel("s")
        plt.ylabel("f(s)")
        plt.title(
            f"{split} | {rank_label}\n"
            f"true {second_name}={true_second:.6g}, pred={pred_second:.6g}; "
            f"true gamma={true_gamma:.6g}, pred={pred_gamma:.6g}\n"
            f"gamma factor={gamma_factor:.4g}, f relL2={f_rel:.4g}"
        )
        plt.legend()
        plt.tight_layout()
        f_path = curve_dir / f"f_fit_{stem}.png"
        fig.savefig(f_path, dpi=170)
        plt.close(fig)

        records.append(
            {
                "noise_dir": noise_dir_name,
                "split": split,
                "representative": rank_label,
                "sample_index": i,
                f"true_{second_name}": true_second,
                f"pred_{second_name}": pred_second,
                "true_gamma": true_gamma,
                "pred_gamma": pred_gamma,
                "gamma_factor_error": gamma_factor,
                "f_relative_l2": f_rel,
                "g_clean_relative_l2": g_rel,
                "g_plot": str(g_path),
                "f_plot": str(f_path),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (data_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("mode") != args.mode:
        raise ValueError(f"dataset mode {metadata.get('mode')} != requested {args.mode}")

    second_name = metadata["second_parameter"]
    second_values = np.asarray(metadata["second_train_values"], dtype=np.float64)
    second_min = float(np.min(second_values))
    second_max = float(np.max(second_values))
    second_span = second_max - second_min

    train_path = data_dir / args.train_noise_dir / "train.npz"
    val_path = data_dir / args.train_noise_dir / "val.npz"
    train_ds = PairwiseNpzDataset(train_path, mode=args.mode, max_samples=args.max_train_samples)
    val_ds = PairwiseNpzDataset(val_path, mode=args.mode, max_samples=args.max_val_samples)

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

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=max(args.lr * 1e-3, 1e-7)
    )

    print("=" * 96)
    print(f"Exp20 pairwise training: {args.mode}")
    print(f"g -> [{second_name}, gamma]")
    print(f"device           : {device}")
    print(f"data             : {data_dir}")
    print(f"train / val      : {len(train_ds):,} / {len(val_ds):,}")
    print(f"input points     : {train_ds.gy.shape[1]}")
    print(f"second range     : [{second_min:.6g}, {second_max:.6g}]")
    print(f"gamma range      : [{args.gamma_min:.6g}, {args.gamma_max:.6g}]")
    print(f"global input RMS : {global_rms:.10g}")
    print(f"hidden sizes     : {args.hidden_sizes}")
    print("=" * 96)

    best_val = math.inf
    best_epoch = 0
    bad_epochs = 0
    checkpoint_path = output_dir / f"best_{args.mode}_mlp.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, second_span,
            args.second_loss_weight, args.gamma_loss_weight
        )
        val_loss, val_metrics = evaluate_loader(
            model, val_loader, device, second_span,
            args.second_loss_weight, args.gamma_loss_weight
        )
        lr_now = optimizer.param_groups[0]["lr"]
        improved = val_loss < best_val - args.min_delta

        if improved:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mode": args.mode,
                    "second_name": second_name,
                    "second_min": second_min,
                    "second_max": second_max,
                    "hidden_sizes": tuple(args.hidden_sizes),
                    "dropout": args.dropout,
                    "gamma_min": args.gamma_min,
                    "gamma_max": args.gamma_max,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "gamma_median_factor": val_metrics["gamma_median_factor"],
            "gamma_within_x1p2": val_metrics["gamma_within_x1p2"],
            "second_median_norm_error": val_metrics["second_median_norm_error"],
            "lr": lr_now,
            "is_best": int(improved),
        })

        if improved or epoch % args.log_every == 0:
            mark = "*" if improved else " "
            print(
                f"{mark} epoch {epoch:4d} | train {train_loss:.6g} | val {val_loss:.6g} "
                f"| gamma med {val_metrics['gamma_median_factor']:.4f} "
                f"| gamma <=x1.2 {val_metrics['gamma_within_x1p2']:.3f} "
                f"| {second_name} norm med {val_metrics['second_median_norm_error']:.4f} "
                f"| lr {lr_now:.3g}"
            )
        scheduler.step()
        if bad_epochs >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; best epoch={best_epoch}, "
                f"val loss={best_val:.6g}"
            )
            break

    write_csv(output_dir / "training_history.csv", history)
    plot_history(output_dir / "training_history.png", history)

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("-" * 96)
    print(
        f"Final evaluation: best epoch={best_epoch}, "
        f"physics_metrics={not args.skip_physics_metrics}"
    )

    summary_rows = []
    sample_rows = []
    representative_rows = []
    scatter_done = False

    for noise_dir in discover_noise_dirs(data_dir):
        for split in (
            "test_seen",
            "test_interp_gamma",
            "test_interp_second",
            "test_interp_both",
        ):
            path = noise_dir / f"{split}.npz"
            if not path.exists():
                continue
            ds = PairwiseNpzDataset(path, mode=args.mode, max_samples=args.max_test_samples)
            result = predict_dataset(
                model, ds, mode=args.mode, device=device,
                batch_size=args.eval_batch_size, num_workers=args.num_workers,
                compute_physics=not args.skip_physics_metrics,
                physics_integration_points=args.physics_integration_points,
            )
            met = metrics(
                result["pred_second"], result["true_second"],
                result["pred_gamma"], result["true_gamma"], second_span
            )
            f_med = float(np.nanmedian(result["f_relative_l2"])) if not args.skip_physics_metrics else math.nan
            g_med = float(np.nanmedian(result["g_clean_relative_l2"])) if not args.skip_physics_metrics else math.nan
            level = float(np.nanmedian(result["noise_level"]))
            summary_rows.append({
                "mode": args.mode,
                "second_parameter": second_name,
                "noise_dir": noise_dir.name,
                "noise_level": level,
                "split": split,
                **met,
                "median_f_relative_l2": f_med,
                "median_g_clean_relative_l2": g_med,
            })

            print(
                f"{noise_dir.name:16s} | {split:20s} "
                f"| gamma med={met['gamma_median_factor']:.4f} "
                f"| <=x1.2={met['gamma_within_x1p2']:.3f} "
                f"| {second_name} norm med={met['second_median_norm_error']:.4f}"
            )

            for i in range(len(result["true_gamma"])):
                sample_rows.append({
                    "mode": args.mode,
                    "second_parameter": second_name,
                    "noise_dir": noise_dir.name,
                    "noise_level": float(result["noise_level"][i]),
                    "split": split,
                    f"true_{second_name}": float(result["true_second"][i]),
                    f"pred_{second_name}": float(result["pred_second"][i]),
                    "second_abs_error": float(result["second_abs_error"][i]),
                    "true_gamma": float(result["true_gamma"][i]),
                    "pred_gamma": float(result["pred_gamma"][i]),
                    "gamma_factor_error": float(result["gamma_factor_error"][i]),
                    "f_relative_l2": float(result["f_relative_l2"][i]),
                    "g_clean_relative_l2": float(result["g_clean_relative_l2"][i]),
                })

            # For the same 0.2% condition used in training, automatically save
            # best/median/worst true-vs-pred g and f curves for every diagnostic split.
            if (
                not args.skip_physics_metrics
                and noise_dir.name == args.train_noise_dir
            ):
                representative_rows.extend(
                    save_representative_curve_plots(
                        output_dir=output_dir,
                        dataset=ds,
                        result=result,
                        mode=args.mode,
                        second_name=second_name,
                        split=split,
                        noise_dir_name=noise_dir.name,
                        physics_integration_points=args.physics_integration_points,
                    )
                )

            if (
                not scatter_done
                and noise_dir.name == args.train_noise_dir
                and split == "test_interp_both"
            ):
                plot_scatter(
                    output_dir / "pred_vs_true_gamma_interp_both.png",
                    result["true_gamma"], result["pred_gamma"],
                    "true gamma", "pred gamma"
                )
                plot_scatter(
                    output_dir / f"pred_vs_true_{second_name}_interp_both.png",
                    result["true_second"], result["pred_second"],
                    f"true {second_name}", f"pred {second_name}"
                )
                scatter_done = True

    result_prefix = str(args.result_prefix).strip() or "exp20"
    write_csv(output_dir / f"{result_prefix}_summary.csv", summary_rows)
    write_csv(output_dir / f"{result_prefix}_samples.csv", sample_rows)
    if representative_rows:
        write_csv(output_dir / "representative_curve_samples.csv", representative_rows)

    run_meta = {
        "experiment": "Exp20 pairwise curriculum training",
        "mode": args.mode,
        "second_parameter": second_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "training_args": vars(args),
        "dataset_metadata": metadata,
        "checkpoint": str(checkpoint_path),
    }
    with (output_dir / f"{result_prefix}_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2, default=str)

    print("=" * 96)
    print(f"{result_prefix} training finished.")
    print(f"output: {output_dir}")
    print("Read first:")
    print(f"  {output_dir / (result_prefix + '_summary.csv')}")
    print(f"  {output_dir / (result_prefix + '_samples.csv')}")
    if representative_rows:
        print(f"  {output_dir / 'representative_curve_samples.csv'}")
        print(f"  {output_dir / 'curve_fits'}")
    print("=" * 96)


if __name__ == "__main__":
    main()
