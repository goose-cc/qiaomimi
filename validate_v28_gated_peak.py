#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from GatedPeakResidual import GatedPeakResidualModel, load_v2_base_checkpoint
from mc_inverse_loss import MonteCarloInverseLoss
from mc_online_physics import OnlinePhysics
from mc_peak_visibility import (
    masked_relative_mse_per_sample,
    peak_mask_from_parameters,
    peak_visibility_from_observation,
    relative_mse_per_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate V2.8 gated peak residual against frozen V2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--validation-pool-dir", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--branch-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--noise-level", type=float, default=-1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--physics-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--integration-points", type=int, default=512)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def open_pool(pool_dir: str) -> np.memmap:
    path = Path(pool_dir) / "parameters.dat"
    if not path.exists():
        raise FileNotFoundError(path)
    row_bytes = 5 * np.dtype(np.float32).itemsize
    rows = path.stat().st_size // row_bytes
    return np.memmap(path, mode="r", dtype=np.float32, shape=(rows, 5))


def make_physics_args(
    saved_args: Dict[str, Any], cli: argparse.Namespace
) -> SimpleNamespace:
    noise = float(saved_args.get("noise_level", 0.09))
    if cli.noise_level >= 0:
        noise = float(cli.noise_level)
    return SimpleNamespace(
        physics_dtype=cli.physics_dtype,
        noise_level=noise,
        data_scale=float(saved_args.get("data_scale", 160000.0)),
        shift=float(saved_args.get("shift", 400.0)),
        s_min=float(saved_args.get("s_min", 0.1764)),
        s_max=float(saved_args.get("s_max", 6.0)),
        q2_min=float(saved_args.get("q2_min", -100.0)),
        q2_max=float(saved_args.get("q2_max", -6.0)),
        output_points=int(saved_args.get("output_points", 100)),
        input_points=int(saved_args.get("input_points", 100)),
        integration_points=int(cli.integration_points),
        resonance_grid_points=512,
    )


def describe(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def group_summary(
    name: str,
    mask: np.ndarray,
    f_base: np.ndarray,
    f_new: np.ndarray,
    g_base: np.ndarray,
    g_new: np.ndarray,
    peak_base: np.ndarray,
    peak_new: np.ndarray,
) -> Dict[str, Any]:
    return {
        "name": name,
        "count": int(np.sum(mask)),
        "f_base": describe(f_base[mask]),
        "f_new": describe(f_new[mask]),
        "g_base": describe(g_base[mask]),
        "g_new": describe(g_new[mask]),
        "peak_base": describe(peak_base[mask]),
        "peak_new": describe(peak_new[mask]),
        "f_mean_delta": float(np.mean(f_new[mask] - f_base[mask])) if np.any(mask) else float("nan"),
        "peak_mean_delta": float(np.mean(peak_new[mask] - peak_base[mask])) if np.any(mask) else float("nan"),
        "improved_fraction": float(np.mean(f_new[mask] < f_base[mask])) if np.any(mask) else float("nan"),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    base_model, saved_base_args, _ = load_v2_base_checkpoint(args.base_checkpoint, device)

    branch_path = Path(args.branch_checkpoint)
    branch_checkpoint = torch_load(branch_path)
    if not isinstance(branch_checkpoint, dict) or "model_state_dict" not in branch_checkpoint:
        raise RuntimeError("branch checkpoint must contain model_state_dict")
    branch_args = dict(branch_checkpoint.get("args", {}))
    model = GatedPeakResidualModel(
        base_model,
        nhead=int(saved_base_args.get("transformer_nhead", 4)),
        branch_layers=int(branch_args.get("branch_layers", 1)),
        branch_dim_feedforward=int(branch_args.get("branch_dim_feedforward", 256)),
        branch_dropout=float(branch_args.get("branch_dropout", 0.05)),
        residual_scale=float(branch_args.get("residual_scale", 1.0)),
        gate_activation_threshold=float(
            branch_args.get("gate_activation_threshold", 0.70)
        ),
    ).to(device)
    model.load_state_dict(branch_checkpoint["model_state_dict"], strict=True)
    model.eval()

    pool = open_pool(args.validation_pool_dir)
    sample_count = min(args.num_samples, len(pool))
    params_np_all = np.asarray(pool[:sample_count], dtype=np.float32).copy()
    physics_args = make_physics_args(saved_base_args, args)
    physics = OnlinePhysics(physics_args, device)
    criterion = MonteCarloInverseLoss(
        s_min=physics_args.s_min,
        s_max=physics_args.s_max,
        q2_min=physics_args.q2_min,
        q2_max=physics_args.q2_max,
        output_points=physics_args.output_points,
        input_points=physics_args.input_points,
        profile="base",
        normalization="relative",
    ).to(device)

    visibility_threshold = float(branch_args.get("visibility_threshold", 2.0))
    gate_activation_threshold = float(
        branch_args.get("gate_activation_threshold", 0.70)
    )
    peak_window_widths = float(branch_args.get("peak_window_widths", 3.0))
    peak_min_radius = float(branch_args.get("peak_min_radius", 0.09))
    peak_max_radius = float(branch_args.get("peak_max_radius", 0.50))
    s_grid = physics.s_output.to(device=device, dtype=torch.float32)
    noise_rng = np.random.default_rng(args.seed)

    rows: List[Dict[str, float]] = []
    print("=" * 76)
    print("V2.8 validation against frozen V2")
    print("device            : %s" % device)
    print("samples           : %s" % format(sample_count, ","))
    print("base checkpoint   : %s" % Path(args.base_checkpoint).resolve())
    print("branch checkpoint : %s" % branch_path.resolve())
    print("visibility cutoff : %.3f" % visibility_threshold)
    print("gate activation   : %.3f" % gate_activation_threshold)
    print("=" * 76)

    with torch.inference_mode():
        for start in range(0, sample_count, args.batch_size):
            stop = min(start + args.batch_size, sample_count)
            params_np = params_np_all[start:stop]
            params = torch.from_numpy(params_np).to(device)
            f_true_2d, g_clean_2d = physics.make_clean_batch(params)

            clean_np = g_clean_2d.detach().cpu().numpy().astype(np.float64, copy=False)
            rms_np = np.sqrt(np.mean(clean_np ** 2, axis=1, keepdims=True))
            noise_np = (
                physics_args.noise_level
                * rms_np
                * noise_rng.standard_normal(clean_np.shape)
            ).astype(np.float32)
            g_noisy_2d = g_clean_2d + torch.from_numpy(noise_np).to(device)

            f_true = f_true_2d.unsqueeze(1)
            output = model(g_noisy_2d.unsqueeze(1))
            f_base = output["base"]
            f_new = output["final"]
            g_base = criterion.physics_forward_integral(f_base)
            g_new = criterion.physics_forward_integral(f_new)
            g_clean = g_clean_2d.unsqueeze(1)

            visibility, _ = peak_visibility_from_observation(
                physics,
                params,
                g_clean_2d,
                noise_level=physics_args.noise_level,
            )
            peak_mask = peak_mask_from_parameters(
                params,
                s_grid,
                window_widths=peak_window_widths,
                min_radius=peak_min_radius,
                max_radius=peak_max_radius,
            )

            f_base_rel = torch.sqrt(relative_mse_per_sample(f_base, f_true).clamp_min(0))
            f_new_rel = torch.sqrt(relative_mse_per_sample(f_new, f_true).clamp_min(0))
            g_base_rel = torch.sqrt(relative_mse_per_sample(g_base, g_clean).clamp_min(0))
            g_new_rel = torch.sqrt(relative_mse_per_sample(g_new, g_clean).clamp_min(0))
            peak_base_rel = torch.sqrt(
                masked_relative_mse_per_sample(f_base, f_true, peak_mask).clamp_min(0)
            )
            peak_new_rel = torch.sqrt(
                masked_relative_mse_per_sample(f_new, f_true, peak_mask).clamp_min(0)
            )
            local_gate = output["local_gate"]
            residual = output["residual"]

            inside_count = peak_mask.sum(dim=(1, 2)).clamp_min(1.0)
            outside_mask = 1.0 - peak_mask
            outside_count = outside_mask.sum(dim=(1, 2)).clamp_min(1.0)

            local_gate_peak_mean = (
                (local_gate * peak_mask).sum(dim=(1, 2)) / inside_count
            )

            local_gate_outside_mean = (
               (local_gate * outside_mask).sum(dim=(1, 2)) / outside_count
            )

local_gate_max = local_gate.amax(dim=(1, 2))

residual_rms = torch.sqrt(
    torch.mean(residual.square(), dim=(1, 2))
)

local_residual_rms = torch.sqrt(
    torch.mean((local_gate * residual).square(), dim=(1, 2))
)
            arrays = {
                "visibility": visibility.detach().cpu().numpy(),
                "gate": output["gate"].detach().cpu().numpy(),
                "active_gate": output.get("active_gate", output["gate"]).detach().cpu().numpy(),
                "f_base": f_base_rel.detach().cpu().numpy(),
                "f_new": f_new_rel.detach().cpu().numpy(),
                "g_base": g_base_rel.detach().cpu().numpy(),
                "g_new": g_new_rel.detach().cpu().numpy(),
                "peak_base": peak_base_rel.detach().cpu().numpy(),
                "peak_new": peak_new_rel.detach().cpu().numpy(),
                "correction_rms": torch.sqrt(
                    torch.mean(output["correction"].square(), dim=(1, 2))
                ).detach().cpu().numpy(),
            }
            for local_index in range(stop - start):
                p = params_np[local_index]
                rows.append({
                    "row": float(start + local_index),
                    "a1": float(p[0]),
                    "a2": float(p[1]),
                    "a3": float(p[2]),
                    "m": float(p[3]),
                    "gamma": float(p[4]),
                    "width": float(p[3] * p[4]),
                    "visibility": float(arrays["visibility"][local_index]),
                    "gate": float(arrays["gate"][local_index]),
                    "active_gate": float(arrays["active_gate"][local_index]),
                    "f_base_rel": float(arrays["f_base"][local_index]),
                    "f_new_rel": float(arrays["f_new"][local_index]),
                    "g_base_rel": float(arrays["g_base"][local_index]),
                    "g_new_rel": float(arrays["g_new"][local_index]),
                    "peak_base_rel": float(arrays["peak_base"][local_index]),
                    "peak_new_rel": float(arrays["peak_new"][local_index]),
                    "correction_rms": float(arrays["correction_rms"][local_index]),
                })
            print("validated: %d/%d" % (stop, sample_count), flush=True)

    columns = list(rows[0].keys())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "validation_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    def arr(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    visibility = arr("visibility")
    gate = arr("gate")
    f_base = arr("f_base_rel")
    f_new = arr("f_new_rel")
    g_base = arr("g_base_rel")
    g_new = arr("g_new_rel")
    peak_base = arr("peak_base_rel")
    peak_new = arr("peak_new_rel")

    hard_target = visibility >= visibility_threshold
    # Report metrics at the same calibrated cutoff that actually enables
    # spectral corrections, rather than at the generic 0.5 classifier cutoff.
    predicted = gate >= gate_activation_threshold
    tp = int(np.sum(predicted & hard_target))
    fp = int(np.sum(predicted & (~hard_target)))
    tn = int(np.sum((~predicted) & (~hard_target)))
    fn = int(np.sum((~predicted) & hard_target))

    groups = [
        group_summary("all", np.ones(sample_count, dtype=bool), f_base, f_new, g_base, g_new, peak_base, peak_new),
        group_summary("visibility_lt_1", visibility < 1.0, f_base, f_new, g_base, g_new, peak_base, peak_new),
        group_summary("visibility_1_to_2", (visibility >= 1.0) & (visibility < 2.0), f_base, f_new, g_base, g_new, peak_base, peak_new),
        group_summary("visibility_ge_2", visibility >= 2.0, f_base, f_new, g_base, g_new, peak_base, peak_new),
    ]
    summary = {
        "experiment": "v2.8_gated_peak_residual",
        "selected_step": int(branch_checkpoint.get("global_step", 0)),
        "stage": branch_checkpoint.get("stage", "unknown"),
        "sample_count": sample_count,
        "noise_level": physics_args.noise_level,
        "visibility_threshold": visibility_threshold,
        "gate": {
            "accuracy": float((tp + tn) / max(tp + fp + tn + fn, 1)),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        },
        "groups": groups,
        "overall": {
            "f_base": describe(f_base),
            "f_new": describe(f_new),
            "g_base": describe(g_base),
            "g_new": describe(g_new),
            "peak_base": describe(peak_base),
            "peak_new": describe(peak_new),
            "improved_fraction": float(np.mean(f_new < f_base)),
            "worsened_fraction": float(np.mean(f_new > f_base)),
        },
        "files": {"metrics": str(metrics_path.resolve())},
    }
    summary_path = output_dir / "validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("=" * 76)
    print("selected step       : %s" % format(summary["selected_step"], ","))
    print("f base/new mean     : %.6f / %.6f" % (np.mean(f_base), np.mean(f_new)))
    print("g base/new mean     : %.6f / %.6f" % (np.mean(g_base), np.mean(g_new)))
    print("peak base/new mean  : %.6f / %.6f" % (np.mean(peak_base), np.mean(peak_new)))
    print("improved fraction   : %.3f" % np.mean(f_new < f_base))
    print("gate precision      : %.3f" % summary["gate"]["precision"])
    print("gate recall         : %.3f" % summary["gate"]["recall"])
    print("gate false positive : %.3f" % summary["gate"]["false_positive_rate"])
    print("summary             : %s" % summary_path.resolve())
    print("metrics             : %s" % metrics_path.resolve())
    print("=" * 76)


if __name__ == "__main__":
    main()
