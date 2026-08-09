#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp18A: Pure-physics variable projection for the CURRENT five-parameter model.

No neural-network training. The physical forward formula is NOT changed.

For fixed (m, gamma), the current forward observable is linear in
[a1, a2, a3]:

    g = a1 * G_res(m, gamma) + a2 * G_a2 + a3 * G_a3.

Therefore:
  1) scan only the two nonlinear parameters (m, gamma),
  2) at every grid point solve bounded least squares for (a1,a2,a3),
  3) choose the minimum-residual solution.

The script first verifies this linear decomposition numerically against the
current OnlinePhysics implementation and aborts if it is not accurate enough.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from mc_online_physics import OnlinePhysics, fixed_rms_noise_numpy
from mc_physics import valid_parameter_mask_numpy


LOWER = np.array(
    [0.0, 0.0, -0.05, np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0)],
    dtype=np.float64,
)
UPPER = np.array([0.2, 0.05, 0.05, 2.0, 1.0], dtype=np.float64)
LINEAR_LOWER = np.array([0.0, 0.0, -0.05], dtype=np.float64)
LINEAR_UPPER = np.array([0.2, 0.05, 0.05], dtype=np.float64)


def parse_float_list(text: str) -> List[float]:
    out = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not out:
        raise ValueError("empty float list")
    return out


def choose_device(name: str) -> torch.device:
    name = str(name).lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "xpu":
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("XPU requested but unavailable")
        return torch.device("xpu")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser(
        description="Exp18A pure-physics (m,gamma) variable projection"
    )
    p.add_argument(
        "--output-dir",
        default=r".\validation_results\exp18_variable_projection",
    )

    # Current physics configuration.
    p.add_argument("--input-points", type=int, default=100)
    p.add_argument("--output-points", type=int, default=1000)
    p.add_argument("--integration-points", type=int, default=128)
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

    # Test distribution.
    p.add_argument("--a1-min", type=float, default=0.05)
    p.add_argument("--a1-max", type=float, default=0.20)
    p.add_argument("--m-true-min", type=float, default=1e-4)
    p.add_argument("--m-true-max", type=float, default=2.0)
    p.add_argument("--true-gammas", default="0.01,0.05,0.2,0.5")
    p.add_argument("--samples-per-gamma", type=int, default=40)
    p.add_argument(
        "--noise-levels",
        default="0,0.002,0.01,0.05",
        help="fractions: 0.002=0.2%, 0.01=1%, 0.05=5%",
    )

    # Global 2D search grid.
    p.add_argument("--scan-m-min", type=float, default=1e-4)
    p.add_argument("--scan-m-max", type=float, default=2.0)
    p.add_argument("--scan-m-points", type=int, default=161)
    p.add_argument("--scan-gamma-min", type=float, default=0.001)
    p.add_argument("--scan-gamma-max", type=float, default=1.0)
    p.add_argument("--scan-gamma-points", type=int, default=161)

    # Numerics.
    p.add_argument("--basis-chunk", type=int, default=2048)
    p.add_argument("--sample-batch", type=int, default=16)
    p.add_argument("--coordinate-descent-iters", type=int, default=24)
    p.add_argument("--linearity-check-samples", type=int, default=64)
    p.add_argument("--linearity-tolerance", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "xpu"),
        default="auto",
    )
    return p.parse_args()


def validate_args(args):
    if not (0.0 < args.a1_min < args.a1_max <= 0.2):
        raise ValueError("invalid a1 range")
    if not (0.0 < args.m_true_min < args.m_true_max <= 2.0):
        raise ValueError("invalid true m range")
    if not (0.0 < args.scan_m_min < args.scan_m_max <= 2.0):
        raise ValueError("invalid scan m range")
    if not (0.0 < args.scan_gamma_min < args.scan_gamma_max <= 1.0):
        raise ValueError("invalid scan gamma range")
    if args.scan_m_points < 3 or args.scan_gamma_points < 3:
        raise ValueError("scan grids must contain at least 3 points")
    if args.samples_per_gamma <= 0:
        raise ValueError("samples-per-gamma must be positive")
    if args.coordinate_descent_iters < 0:
        raise ValueError("coordinate-descent-iters must be >= 0")


def sample_valid_parameters(
    rng: np.random.Generator,
    args,
    n: int,
    *,
    fixed_gamma: float,
) -> np.ndarray:
    accepted = []
    total = 0
    while total < n:
        need = n - total
        candidate_n = max(4 * need, 512)
        p = np.empty((candidate_n, 5), dtype=np.float64)
        p[:, 0] = rng.uniform(args.a1_min, args.a1_max, candidate_n)
        p[:, 1] = rng.uniform(0.0, 0.05, candidate_n)
        p[:, 2] = rng.uniform(-0.05, 0.05, candidate_n)
        p[:, 3] = rng.uniform(args.m_true_min, args.m_true_max, candidate_n)
        p[:, 4] = float(fixed_gamma)

        mask = valid_parameter_mask_numpy(p)
        good = p[mask]
        if len(good):
            take = min(need, len(good))
            accepted.append(good[:take])
            total += take

    return np.concatenate(accepted, axis=0).astype(np.float32, copy=False)


def build_test_parameters(args) -> np.ndarray:
    rng = np.random.default_rng(args.seed + 1801)
    parts = []
    for gamma in parse_float_list(args.true_gammas):
        parts.append(
            sample_valid_parameters(
                rng,
                args,
                args.samples_per_gamma,
                fixed_gamma=float(gamma),
            )
        )
    return np.concatenate(parts, axis=0)


@torch.no_grad()
def forward_clean_in_batches(
    physics: OnlinePhysics,
    params_np: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    f_parts, g_parts = [], []
    for start in range(0, len(params_np), batch_size):
        stop = min(start + batch_size, len(params_np))
        p = torch.from_numpy(params_np[start:stop]).to(
            device=physics.device,
            dtype=torch.float32,
        )
        f, g = physics.make_clean_batch(p)
        f_parts.append(f.detach().cpu().numpy())
        g_parts.append(g.detach().cpu().numpy())
    return np.concatenate(f_parts), np.concatenate(g_parts)


@torch.no_grad()
def make_background_bases(
    physics: OnlinePhysics,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ref = torch.tensor(
        [
            [0.0, 0.05, 0.0, 0.8, 0.2],
            [0.0, 0.0, 0.05, 0.8, 0.2],
        ],
        dtype=torch.float32,
        device=physics.device,
    )
    g = physics.forward_from_parameters(ref)
    return (g[0] / 0.05).contiguous(), (g[1] / 0.05).contiguous()


@torch.no_grad()
def resonance_basis_for_pairs(
    physics: OnlinePhysics,
    m_values: np.ndarray,
    gamma_values: np.ndarray,
    *,
    chunk: int,
    a1_reference: float = 0.1,
) -> torch.Tensor:
    if len(m_values) != len(gamma_values):
        raise ValueError("m_values and gamma_values must have same length")

    parts = []
    for start in range(0, len(m_values), chunk):
        stop = min(start + chunk, len(m_values))
        k = stop - start
        p = torch.zeros((k, 5), dtype=torch.float32, device=physics.device)
        p[:, 0] = float(a1_reference)
        p[:, 3] = torch.as_tensor(
            m_values[start:stop],
            dtype=torch.float32,
            device=physics.device,
        )
        p[:, 4] = torch.as_tensor(
            gamma_values[start:stop],
            dtype=torch.float32,
            device=physics.device,
        )
        parts.append(
            physics.forward_from_parameters(p) / float(a1_reference)
        )
    return torch.cat(parts, dim=0).contiguous()


@torch.no_grad()
def verify_linearity(
    physics: OnlinePhysics,
    args,
    b2: torch.Tensor,
    b3: torch.Tensor,
) -> float:
    rng = np.random.default_rng(args.seed + 1811)
    n = args.linearity_check_samples

    params = np.empty((n, 5), dtype=np.float32)
    params[:, 0] = rng.uniform(args.a1_min, args.a1_max, n)
    params[:, 1] = rng.uniform(0.0, 0.05, n)
    params[:, 2] = rng.uniform(-0.05, 0.05, n)
    params[:, 3] = rng.uniform(args.m_true_min, args.m_true_max, n)
    params[:, 4] = np.exp(
        rng.uniform(math.log(0.001), math.log(1.0), n)
    )

    p = torch.from_numpy(params).to(
        device=physics.device,
        dtype=torch.float32,
    )
    g_true = physics.forward_from_parameters(p)
    r = resonance_basis_for_pairs(
        physics,
        params[:, 3],
        params[:, 4],
        chunk=args.basis_chunk,
    )
    g_decomp = (
        p[:, 0:1] * r
        + p[:, 1:2] * b2.view(1, -1)
        + p[:, 2:3] * b3.view(1, -1)
    )

    rel = torch.linalg.vector_norm(
        g_decomp - g_true, dim=1
    ) / torch.linalg.vector_norm(
        g_true, dim=1
    ).clamp_min(1e-20)

    return float(torch.max(rel).detach().cpu())


def build_scan_grid(
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m_grid = np.linspace(
        args.scan_m_min,
        args.scan_m_max,
        args.scan_m_points,
        dtype=np.float64,
    )
    gamma_grid = np.logspace(
        math.log10(args.scan_gamma_min),
        math.log10(args.scan_gamma_max),
        args.scan_gamma_points,
        dtype=np.float64,
    )

    anchors = np.asarray(parse_float_list(args.true_gammas))
    anchors = anchors[
        (anchors >= args.scan_gamma_min)
        & (anchors <= args.scan_gamma_max)
    ]
    gamma_grid = np.unique(np.concatenate([gamma_grid, anchors]))
    gamma_grid.sort()

    mm, gg = np.meshgrid(m_grid, gamma_grid, indexing="ij")
    return (
        m_grid.astype(np.float32),
        gamma_grid.astype(np.float32),
        mm.reshape(-1).astype(np.float32),
        gg.reshape(-1).astype(np.float32),
    )


@torch.no_grad()
def prepare_varpro_gram(
    resonance_basis: torch.Tensor,
    b2: torch.Tensor,
    b3: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    r = resonance_basis

    g00 = torch.sum(r * r, dim=1)
    g01 = r @ b2
    g02 = r @ b3
    g11 = torch.sum(b2 * b2)
    g12 = torch.sum(b2 * b3)
    g22 = torch.sum(b3 * b3)

    c = len(r)
    G = torch.empty(
        (c, 3, 3),
        dtype=r.dtype,
        device=r.device,
    )
    G[:, 0, 0] = g00
    G[:, 0, 1] = G[:, 1, 0] = g01
    G[:, 0, 2] = G[:, 2, 0] = g02
    G[:, 1, 1] = g11
    G[:, 1, 2] = G[:, 2, 1] = g12
    G[:, 2, 2] = g22

    return {
        "G": G,
        "G_pinv": torch.linalg.pinv(G),
        "g00": g00,
        "g01": g01,
        "g02": g02,
        "g11": g11,
        "g12": g12,
        "g22": g22,
    }


@torch.no_grad()
def solve_full_varpro_batch(
    y: torch.Tensor,
    resonance_basis: torch.Tensor,
    b2: torch.Tensor,
    b3: torch.Tensor,
    gram: Dict[str, torch.Tensor],
    *,
    coordinate_descent_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h0 = y @ resonance_basis.T
    h1 = (y @ b2)[:, None].expand_as(h0)
    h2 = (y @ b3)[:, None].expand_as(h0)
    h = torch.stack([h0, h1, h2], dim=2)

    x = torch.einsum(
        "cij,bcj->bci",
        gram["G_pinv"],
        h,
    )

    lo = torch.tensor(
        LINEAR_LOWER,
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3)
    hi = torch.tensor(
        LINEAR_UPPER,
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3)
    x = torch.maximum(torch.minimum(x, hi), lo)

    eps = torch.finfo(x.dtype).eps
    d0 = gram["g00"].clamp_min(eps).view(1, -1)
    d1 = gram["g11"].clamp_min(eps)
    d2 = gram["g22"].clamp_min(eps)
    g01 = gram["g01"].view(1, -1)
    g02 = gram["g02"].view(1, -1)
    g12 = gram["g12"]

    # Convex 3D box-constrained LS: exact coordinate minimizers + projection.
    for _ in range(int(coordinate_descent_iters)):
        x[:, :, 0] = (
            (h0 - g01 * x[:, :, 1] - g02 * x[:, :, 2]) / d0
        ).clamp(0.0, 0.2)

        x[:, :, 1] = (
            (h1 - g01 * x[:, :, 0] - g12 * x[:, :, 2]) / d1
        ).clamp(0.0, 0.05)

        x[:, :, 2] = (
            (h2 - g02 * x[:, :, 0] - g12 * x[:, :, 1]) / d2
        ).clamp(-0.05, 0.05)

    x0, x1, x2 = x[:, :, 0], x[:, :, 1], x[:, :, 2]

    quad = (
        gram["g00"].view(1, -1) * x0.square()
        + gram["g11"] * x1.square()
        + gram["g22"] * x2.square()
        + 2.0 * gram["g01"].view(1, -1) * x0 * x1
        + 2.0 * gram["g02"].view(1, -1) * x0 * x2
        + 2.0 * gram["g12"] * x1 * x2
    )
    linear = x0 * h0 + x1 * h1 + x2 * h2
    y2 = torch.sum(y * y, dim=1, keepdim=True)

    rss = (y2 - 2.0 * linear + quad).clamp_min(0.0)
    best_rss, best_idx = torch.min(rss, dim=1)

    gather_idx = best_idx.view(-1, 1, 1).expand(-1, 1, 3)
    best_x = torch.gather(
        x,
        dim=1,
        index=gather_idx,
    ).squeeze(1)

    rel = torch.sqrt(best_rss) / torch.sqrt(
        y2[:, 0]
    ).clamp_min(1e-20)

    return best_idx, best_x, rel


@torch.no_grad()
def oracle_gamma_profile_batch(
    physics: OnlinePhysics,
    y: torch.Tensor,
    truth: torch.Tensor,
    gamma_grid: np.ndarray,
    *,
    chunk: int,
) -> torch.Tensor:
    b = len(truth)
    best_error = torch.full(
        (b,),
        float("inf"),
        dtype=torch.float32,
        device=physics.device,
    )
    best_gamma = torch.full(
        (b,),
        float("nan"),
        dtype=torch.float32,
        device=physics.device,
    )

    denom = torch.linalg.vector_norm(
        y, dim=1
    ).clamp_min(1e-20)

    ggrid = torch.from_numpy(
        gamma_grid.astype(np.float32)
    ).to(physics.device)

    for start in range(0, len(ggrid), chunk):
        gv = ggrid[start:start + chunk]
        k = len(gv)

        candidates = truth[:, None, :].expand(
            b, k, 5
        ).clone()
        candidates[:, :, 4] = gv.view(1, k)

        pred = physics.forward_from_parameters(
            candidates.reshape(b * k, 5)
        ).reshape(b, k, -1)

        rel = torch.linalg.vector_norm(
            pred - y[:, None, :],
            dim=2,
        ) / denom[:, None]

        e, idx = torch.min(rel, dim=1)
        improve = e < best_error
        selected = gv[idx]

        best_error = torch.where(
            improve, e, best_error
        )
        best_gamma = torch.where(
            improve, selected, best_gamma
        )

    return best_gamma


def factor_error(
    pred: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    p = np.maximum(
        np.asarray(pred, dtype=np.float64),
        1e-12,
    )
    t = np.maximum(
        np.asarray(truth, dtype=np.float64),
        1e-12,
    )
    return np.maximum(p / t, t / p)


@torch.no_grad()
def evaluate_best_parameters(
    physics: OnlinePhysics,
    pred_params: torch.Tensor,
    truth_params: torch.Tensor,
    g_observed: torch.Tensor,
    g_clean: torch.Tensor,
    f_true: torch.Tensor,
) -> Dict[str, np.ndarray]:
    f_pred, g_pred = physics.make_clean_batch(
        pred_params
    )

    f_rel = torch.linalg.vector_norm(
        f_pred - f_true, dim=1
    ) / torch.linalg.vector_norm(
        f_true, dim=1
    ).clamp_min(1e-20)

    g_clean_rel = torch.linalg.vector_norm(
        g_pred - g_clean, dim=1
    ) / torch.linalg.vector_norm(
        g_clean, dim=1
    ).clamp_min(1e-20)

    g_obs_rel = torch.linalg.vector_norm(
        g_pred - g_observed, dim=1
    ) / torch.linalg.vector_norm(
        g_observed, dim=1
    ).clamp_min(1e-20)

    pred_np = pred_params.detach().cpu().numpy().astype(np.float64)
    true_np = truth_params.detach().cpu().numpy().astype(np.float64)

    abs_norm = np.full_like(
        pred_np,
        np.nan,
    )
    ranges = UPPER - LOWER
    abs_norm[:, :4] = (
        np.abs(pred_np[:, :4] - true_np[:, :4])
        / ranges[:4]
    )

    return {
        "pred": pred_np,
        "true": true_np,
        "f_rel": f_rel.detach().cpu().numpy(),
        "g_clean_rel": g_clean_rel.detach().cpu().numpy(),
        "g_obs_rel": g_obs_rel.detach().cpu().numpy(),
        "abs_norm": abs_norm,
    }


def rows_to_csv(
    path: Path,
    rows: List[Dict[str, object]],
) -> None:
    if not rows:
        return
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(
    sample_rows: List[Dict[str, object]],
    *,
    noise_level: float,
    true_gamma: float,
    regime: str,
) -> Dict[str, object]:
    # true_gamma is stored through float32 tensors, so values such as 0.01
    # may come back as 0.009999999776... .  Never group floating-point values
    # with exact equality; otherwise valid groups can become empty.
    rows = [
        r
        for r in sample_rows
        if np.isclose(
            float(r["noise_level"]),
            float(noise_level),
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(
            float(r["true_gamma"]),
            float(true_gamma),
            rtol=1e-6,
            atol=1e-8,
        )
        and r["regime"] == regime
    ]

    if not rows:
        raise RuntimeError(
            "No samples found for summary group: "
            f"noise_level={noise_level}, true_gamma={true_gamma}, regime={regime}. "
            "This indicates a grouping/label mismatch rather than a physics failure."
        )

    ferr = np.asarray(
        [r["gamma_factor_error"] for r in rows],
        dtype=np.float64,
    )
    f_rel = np.asarray(
        [r["f_relative_l2"] for r in rows],
        dtype=np.float64,
    )
    gc = np.asarray(
        [r["g_clean_relative_l2"] for r in rows],
        dtype=np.float64,
    )
    go = np.asarray(
        [r["g_observed_relative_l2"] for r in rows],
        dtype=np.float64,
    )

    out = {
        "noise_level": float(noise_level),
        "noise_percent": 100.0 * float(noise_level),
        "true_gamma": float(true_gamma),
        "regime": regime,
        "samples": len(rows),
        "median_gamma_factor_error": float(np.median(ferr)),
        "p90_gamma_factor_error": float(
            np.quantile(ferr, 0.90)
        ),
        "within_factor_1p2": float(
            np.mean(ferr <= 1.2)
        ),
        "within_factor_1p5": float(
            np.mean(ferr <= 1.5)
        ),
        "within_factor_2": float(
            np.mean(ferr <= 2.0)
        ),
        "median_f_relative_l2": float(
            np.median(f_rel)
        ),
        "mean_f_relative_l2": float(
            np.mean(f_rel)
        ),
        "median_g_clean_relative_l2": float(
            np.median(gc)
        ),
        "median_g_observed_relative_l2": float(
            np.median(go)
        ),
    }

    for name in ("a1", "a2", "a3", "m"):
        vals = np.asarray(
            [
                r[f"{name}_normalized_abs_error"]
                for r in rows
            ],
            dtype=np.float64,
        )
        out[f"{name}_normalized_abs_mean"] = float(
            np.mean(vals)
        )
        out[
            f"{name}_normalized_abs_median"
        ] = float(np.median(vals))

    return out


def make_plots(
    output_dir: Path,
    summary_rows: List[Dict[str, object]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            f"[WARN] matplotlib unavailable, "
            f"skipping plots: {exc}"
        )
        return

    true_gammas = sorted(
        {
            float(r["true_gamma"])
            for r in summary_rows
        }
    )

    for regime in (
        "oracle_nuisance_1d",
        "full_varpro_2d",
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for gamma in true_gammas:
            rows = sorted(
                [
                    r
                    for r in summary_rows
                    if r["regime"] == regime
                    and float(r["true_gamma"]) == gamma
                ],
                key=lambda r: float(r["noise_level"]),
            )
            x = [
                100.0 * float(r["noise_level"])
                for r in rows
            ]
            y = [
                float(r["within_factor_2"])
                for r in rows
            ]
            ax.plot(
                x,
                y,
                marker="o",
                label=f"true gamma={gamma:g}",
            )

        ax.set_xscale(
            "symlog",
            linthresh=0.02,
        )
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Noise level (%)")
        ax.set_ylabel(
            "Fraction within factor 2"
        )
        ax.set_title(
            f"Exp18 gamma recovery: {regime}"
        )
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir
            / f"within_factor2_{regime}.png",
            dpi=180,
        )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for gamma in true_gammas:
        rows = sorted(
            [
                r
                for r in summary_rows
                if r["regime"] == "full_varpro_2d"
                and float(r["true_gamma"]) == gamma
            ],
            key=lambda r: float(r["noise_level"]),
        )
        x = [
            100.0 * float(r["noise_level"])
            for r in rows
        ]
        y = [
            float(
                r["median_gamma_factor_error"]
            )
            for r in rows
        ]
        ax.plot(
            x,
            y,
            marker="o",
            label=f"true gamma={gamma:g}",
        )

    ax.set_xscale(
        "symlog",
        linthresh=0.02,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Noise level (%)")
    ax.set_ylabel(
        "Median gamma factor error"
    )
    ax.set_title(
        "Exp18 full variable-projection gamma error"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir
        / "full_varpro_median_gamma_factor_error.png",
        dpi=180,
    )
    plt.close(fig)


def main():
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # OnlinePhysics requires a scalar args.noise_level. Exp18 sweeps
    # multiple noise levels externally, so keep OnlinePhysics internally
    # noise-free and add each requested noise level later with
    # fixed_rms_noise_numpy().
    args.noise_level = 0.0
    physics = OnlinePhysics(args, device)
    true_gammas = parse_float_list(
        args.true_gammas
    )
    noise_levels = parse_float_list(
        args.noise_levels
    )

    print("=" * 104)
    print(
        "Exp18A: pure-physics variable projection"
    )
    print("NO neural-network training")
    print(
        "PHYSICAL FORWARD FORMULA: unchanged"
    )
    print(f"device          : {device}")
    print(
        f"q^2             : "
        f"[{args.q2_min}, {args.q2_max}] "
        f"x {args.input_points}"
    )
    print(
        f"true gammas     : {true_gammas}"
    )
    print(
        f"noise levels    : "
        f"{[100*x for x in noise_levels]} %"
    )
    print(
        f"samples/gamma   : "
        f"{args.samples_per_gamma}"
    )
    print(
        "linear params   : a1,a2,a3 "
        "solved by bounded least squares "
        "at every (m,gamma) grid point"
    )
    print("=" * 104)

    start_time = time.time()

    # Same physical truths are reused for every noise level.
    params_np = build_test_parameters(args)
    f_true_np, g_clean_np = (
        forward_clean_in_batches(
            physics,
            params_np,
            args.sample_batch,
        )
    )

    # Build exact bases and verify the decomposition.
    b2, b3 = make_background_bases(physics)
    max_linearity_error = verify_linearity(
        physics,
        args,
        b2,
        b3,
    )
    print(
        "Linearity check max relative error: "
        f"{max_linearity_error:.8g}"
    )

    if (
        not np.isfinite(max_linearity_error)
        or max_linearity_error
        > args.linearity_tolerance
    ):
        raise RuntimeError(
            "Current forward implementation is not "
            "sufficiently linear in [a1,a2,a3] "
            "for Exp18 variable projection: "
            f"max relative error="
            f"{max_linearity_error:.6g}, "
            f"tolerance="
            f"{args.linearity_tolerance:.6g}."
        )

    (
        m_grid,
        gamma_grid,
        candidate_m,
        candidate_gamma,
    ) = build_scan_grid(args)

    print(
        f"Actual candidate grid: "
        f"{len(m_grid)} x {len(gamma_grid)} "
        f"= {len(candidate_m)} pairs"
    )

    print(
        "Precomputing resonance basis "
        "for the full (m,gamma) grid ..."
    )
    resonance_basis = resonance_basis_for_pairs(
        physics,
        candidate_m,
        candidate_gamma,
        chunk=args.basis_chunk,
    )
    gram = prepare_varpro_gram(
        resonance_basis,
        b2,
        b3,
    )

    print(
        f"Basis ready: "
        f"resonance={tuple(resonance_basis.shape)}, "
        f"background={tuple(b2.shape)}"
    )

    truth_all = torch.from_numpy(
        params_np
    ).to(
        device=physics.device,
        dtype=torch.float32,
    )
    f_true_all = torch.from_numpy(
        f_true_np
    ).to(
        device=physics.device,
        dtype=torch.float32,
    )
    g_clean_all = torch.from_numpy(
        g_clean_np
    ).to(
        device=physics.device,
        dtype=torch.float32,
    )

    candidate_m_t = torch.from_numpy(
        candidate_m
    ).to(physics.device)
    candidate_gamma_t = torch.from_numpy(
        candidate_gamma
    ).to(physics.device)

    sample_rows: List[Dict[str, object]] = []

    for noise_index, noise_level in enumerate(
        noise_levels
    ):
        print("-" * 104)
        print(
            f"Noise = "
            f"{100.0*noise_level:.6g}%"
        )

        if noise_level > 0:
            g_obs_np = fixed_rms_noise_numpy(
                g_clean_np,
                float(noise_level),
                args.seed + 20000 + noise_index,
            )
        else:
            g_obs_np = g_clean_np.astype(
                np.float32,
                copy=True,
            )

        g_obs_all = torch.from_numpy(
            g_obs_np
        ).to(
            device=physics.device,
            dtype=torch.float32,
        )

        for start in range(
            0,
            len(params_np),
            args.sample_batch,
        ):
            stop = min(
                start + args.sample_batch,
                len(params_np),
            )

            truth = truth_all[start:stop]
            f_true = f_true_all[start:stop]
            g_clean = g_clean_all[start:stop]
            g_obs = g_obs_all[start:stop]

            # Physical upper-bound control.
            oracle_gamma = oracle_gamma_profile_batch(
                physics,
                g_obs,
                truth,
                gamma_grid,
                chunk=min(
                    args.basis_chunk,
                    len(gamma_grid),
                ),
            )
            oracle_params = truth.clone()
            oracle_params[:, 4] = oracle_gamma

            oracle_eval = evaluate_best_parameters(
                physics,
                oracle_params,
                truth,
                g_obs,
                g_clean,
                f_true,
            )

            # Main Exp18 method.
            best_idx, best_x, _ = (
                solve_full_varpro_batch(
                    g_obs,
                    resonance_basis,
                    b2,
                    b3,
                    gram,
                    coordinate_descent_iters=(
                        args.coordinate_descent_iters
                    ),
                )
            )

            varpro_params = torch.empty_like(truth)
            varpro_params[:, 0:3] = best_x
            varpro_params[:, 3] = (
                candidate_m_t[best_idx]
            )
            varpro_params[:, 4] = (
                candidate_gamma_t[best_idx]
            )

            varpro_eval = evaluate_best_parameters(
                physics,
                varpro_params,
                truth,
                g_obs,
                g_clean,
                f_true,
            )

            for local_i in range(stop - start):
                global_i = start + local_i

                for regime, ev in (
                    (
                        "oracle_nuisance_1d",
                        oracle_eval,
                    ),
                    (
                        "full_varpro_2d",
                        varpro_eval,
                    ),
                ):
                    pred = ev["pred"][local_i]
                    tr = ev["true"][local_i]

                    gf = float(
                        factor_error(
                            np.array([pred[4]]),
                            np.array([tr[4]]),
                        )[0]
                    )

                    sample_rows.append(
                        {
                            "sample_index": int(
                                global_i
                            ),
                            "noise_level": float(
                                noise_level
                            ),
                            "noise_percent": float(
                                100.0 * noise_level
                            ),
                            "true_gamma": float(
                                tr[4]
                            ),
                            "regime": regime,
                            "true_a1": float(tr[0]),
                            "true_a2": float(tr[1]),
                            "true_a3": float(tr[2]),
                            "true_m": float(tr[3]),
                            "pred_a1": float(pred[0]),
                            "pred_a2": float(pred[1]),
                            "pred_a3": float(pred[2]),
                            "pred_m": float(pred[3]),
                            "pred_gamma": float(
                                pred[4]
                            ),
                            "gamma_factor_error": gf,
                            "a1_normalized_abs_error": float(
                                ev["abs_norm"][
                                    local_i, 0
                                ]
                            ),
                            "a2_normalized_abs_error": float(
                                ev["abs_norm"][
                                    local_i, 1
                                ]
                            ),
                            "a3_normalized_abs_error": float(
                                ev["abs_norm"][
                                    local_i, 2
                                ]
                            ),
                            "m_normalized_abs_error": float(
                                ev["abs_norm"][
                                    local_i, 3
                                ]
                            ),
                            "f_relative_l2": float(
                                ev["f_rel"][local_i]
                            ),
                            "g_clean_relative_l2": float(
                                ev["g_clean_rel"][
                                    local_i
                                ]
                            ),
                            "g_observed_relative_l2": float(
                                ev["g_obs_rel"][
                                    local_i
                                ]
                            ),
                        }
                    )

    summary_rows: List[Dict[str, object]] = []
    for noise_level in noise_levels:
        for true_gamma in true_gammas:
            for regime in (
                "oracle_nuisance_1d",
                "full_varpro_2d",
            ):
                summary_rows.append(
                    summarize_group(
                        sample_rows,
                        noise_level=float(
                            noise_level
                        ),
                        true_gamma=float(
                            true_gamma
                        ),
                        regime=regime,
                    )
                )

    rows_to_csv(
        output_dir / "exp18_summary.csv",
        summary_rows,
    )
    rows_to_csv(
        output_dir / "exp18_samples.csv",
        sample_rows,
    )

    metadata = {
        "physical_formula_changed": False,
        "neural_network_used": False,
        "method": (
            "2D scan over (m,gamma) + "
            "bounded variable projection "
            "for [a1,a2,a3]"
        ),
        "linearity_check_max_relative_error": (
            max_linearity_error
        ),
        "linearity_tolerance": (
            args.linearity_tolerance
        ),
        "candidate_pairs": int(
            len(candidate_m)
        ),
        "m_grid_points": int(len(m_grid)),
        "gamma_grid_points": int(
            len(gamma_grid)
        ),
        "true_gammas": [
            float(x) for x in true_gammas
        ],
        "noise_levels": [
            float(x) for x in noise_levels
        ],
        "samples_per_gamma": int(
            args.samples_per_gamma
        ),
        "elapsed_seconds": float(
            time.time() - start_time
        ),
        "args": vars(args),
    }

    with (
        output_dir / "exp18_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    make_plots(
        output_dir,
        summary_rows,
    )

    print("=" * 104)
    print("Exp18 finished")
    print(f"output: {output_dir}")
    print("Read first:")
    print("  exp18_summary.csv")
    print("  exp18_samples.csv")
    print("")
    print("Key comparison:")
    print(
        "  oracle_nuisance_1d "
        "vs full_varpro_2d"
    )
    print("=" * 104)
    print(
        "noise% | gamma | regime               | "
        "med factor | <=x2  | med f L2 | med g(clean)"
    )
    print("-" * 104)

    for r in summary_rows:
        print(
            f"{r['noise_percent']:6.3g} | "
            f"{r['true_gamma']:5.3g} | "
            f"{r['regime']:20s} | "
            f"{r['median_gamma_factor_error']:10.4f} | "
            f"{r['within_factor_2']:5.3f} | "
            f"{r['median_f_relative_l2']:8.4f} | "
            f"{r['median_g_clean_relative_l2']:12.5g}"
        )


if __name__ == "__main__":
    main()
