#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Coverage-first, g-diverse sampler for the large Monte-Carlo parameter pool.

This module does not train a network and never edits g by hand.  It only:
- reads continuous parameter rows from parameters.dat without replacement;
- applies a curriculum stage by fixing non-free parameters;
- preserves marginal coverage of every free parameter;
- within the same coverage priority, prefers physically generated clean-g
  curves that are farther apart;
- treats the requested g gap as a soft target, never as a reason to delete a
  parameter bin.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from train_mc_parameter_pool_transformer_loss import ShuffledNoReplacementSampler

PARAMETER_NAMES = ("a1", "a2", "a3", "m", "gamma")
PARAMETER_LOWER = np.array([0.0, 0.0, -0.05, 0.0, 0.0], dtype=np.float64)
PARAMETER_UPPER = np.array([0.2, 0.05, 0.05, 2.0, 1.0], dtype=np.float64)

STAGE_SPECS: dict[str, dict[str, Any]] = {
    "gamma": {
        "free": (4,),
        "fixed": {0: 0.10, 1: 0.025, 2: 0.0, 3: 0.8},
        "a1_min": None,
        "default_bins": 16,
        "description": "gamma free; a1=0.10, a2=0.025, a3=0, m=0.8",
    },
    "a1gamma": {
        "free": (0, 4),
        "fixed": {1: 0.025, 2: 0.0, 3: 0.8},
        "a1_min": 0.05,
        "default_bins": 8,
        "description": "a1,gamma free; a1>=0.05; a2=0.025, a3=0, m=0.8",
    },
    "a1mgamma": {
        "free": (0, 3, 4),
        "fixed": {1: 0.025, 2: 0.0},
        "a1_min": 0.05,
        "default_bins": 6,
        "description": "a1,m,gamma free; a1>=0.05; background fixed",
    },
    "all5_strong": {
        "free": (0, 1, 2, 3, 4),
        "fixed": {},
        "a1_min": 0.05,
        "default_bins": 4,
        "description": "all five free; a1>=0.05",
    },
    "all5_full": {
        "free": (0, 1, 2, 3, 4),
        "fixed": {},
        "a1_min": 0.0,
        "default_bins": 4,
        "description": "all five free over the original pool range",
    },
}


class PoolCursor:
    """Deterministic block-shuffled no-replacement cursor over a memmap pool."""

    def __init__(
        self,
        pool: np.memmap,
        usable_rows: int,
        block_size: int,
        seed: int,
        *,
        cycle: int = 0,
        progress: int = 0,
    ) -> None:
        self.pool = pool
        self.usable_rows = int(usable_rows)
        self.sampler = ShuffledNoReplacementSampler(
            pool=pool,
            usable_rows=usable_rows,
            block_size=block_size,
            seed=seed,
        )
        self.cycle = int(cycle)
        self.progress = int(progress)
        self.rows_read = 0

    def draw(self, n: int) -> np.ndarray:
        if n <= 0:
            raise ValueError("draw n must be positive")
        pieces: list[np.ndarray] = []
        remaining = int(n)
        while remaining > 0:
            if self.progress >= self.usable_rows:
                self.cycle += 1
                self.progress = 0
            available = self.usable_rows - self.progress
            take = min(remaining, available)
            pieces.append(
                self.sampler.sample_chunk(
                    cycle=self.cycle,
                    progress=self.progress,
                    chunk_size=take,
                )
            )
            self.progress += take
            self.rows_read += take
            remaining -= take
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)


def apply_stage(rows: np.ndarray, stage: str) -> np.ndarray:
    spec = STAGE_SPECS[stage]
    p = np.asarray(rows, dtype=np.float32).copy()
    a1_min = spec["a1_min"]
    if a1_min is not None:
        p = p[p[:, 0] >= float(a1_min)]
    for idx, value in spec["fixed"].items():
        p[:, int(idx)] = float(value)
    return p


def collect_candidates(cursor: PoolCursor, stage: str, target: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    total = 0
    while total < target:
        read_n = max(target - total, 256)
        if STAGE_SPECS[stage]["a1_min"] is not None:
            read_n = int(math.ceil(read_n * 1.5))
        rows = apply_stage(cursor.draw(read_n), stage)
        if len(rows) == 0:
            continue
        take = min(target - total, len(rows))
        chunks.append(rows[:take])
        total += take
    return np.concatenate(chunks, axis=0)


def parameter_bin_ids(params: torch.Tensor, free_idx: tuple[int, ...], bins: int, stage: str) -> torch.Tensor:
    """Marginal continuous-range bins used only for sampling bookkeeping."""
    ids = []
    a1_min = STAGE_SPECS[stage]["a1_min"]
    for idx in free_idx:
        lo = float(PARAMETER_LOWER[idx])
        hi = float(PARAMETER_UPPER[idx])
        if idx == 0 and a1_min is not None:
            lo = float(a1_min)
        x = ((params[:, idx] - lo) / max(hi - lo, 1e-12)).clamp(0.0, 1.0 - 1e-7)
        ids.append(torch.floor(x * bins).long())
    return torch.stack(ids, dim=1)


def relative_gap_to_one(
    g_all: torch.Tensor,
    rms_all: torch.Tensor,
    g_one: torch.Tensor,
    rms_one: torch.Tensor,
) -> torch.Tensor:
    rms_diff = torch.sqrt(torch.mean((g_all - g_one.unsqueeze(0)).square(), dim=1))
    denom = 0.5 * (rms_all + rms_one)
    return rms_diff / denom.clamp_min(1e-12)


def batch_gap_stats(g: torch.Tensor) -> tuple[float, float]:
    if g.shape[0] < 2:
        return float("nan"), float("nan")
    with torch.no_grad():
        l2 = torch.cdist(g.float(), g.float(), p=2)
        rms_diff = l2 / math.sqrt(g.shape[1])
        rms = torch.sqrt(torch.mean(g.float().square(), dim=1)).clamp_min(1e-12)
        denom = 0.5 * (rms[:, None] + rms[None, :])
        rel = rms_diff / denom
        rel.fill_diagonal_(float("inf"))
        nearest = torch.min(rel, dim=1).values
        return float(torch.min(nearest).item()), float(torch.median(nearest).item())


def select_batch(
    params: torch.Tensor,
    g_clean: torch.Tensor,
    *,
    stage: str,
    free_idx: tuple[int, ...],
    batch_size: int,
    bins: int,
    strategy: str,
    target_gap_fraction: float,
    global_bin_counts: torch.Tensor,
    global_joint_counts: torch.Tensor,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Coverage first, g separation second; no bin is deleted for a g-gap shortfall."""
    n = params.shape[0]
    if n < batch_size:
        raise ValueError("candidate count must be >= batch size")
    bin_ids = parameter_bin_ids(params, free_idx, bins, stage)
    local_counts = torch.zeros((len(free_idx), bins), dtype=torch.long, device=params.device)
    # Coarse JOINT cells preserve parameter-combination diversity in addition to
    # marginal coverage.  Even all5 with 4 bins/parameter is only 4^5=1024 cells.
    powers = torch.as_tensor([bins ** j for j in range(len(free_idx))], dtype=torch.long, device=params.device)
    joint_ids = torch.sum(bin_ids * powers.view(1, -1), dim=1)
    joint_total = int(bins ** len(free_idx))
    if global_joint_counts.numel() != joint_total:
        raise ValueError(f"global_joint_counts has {global_joint_counts.numel()} entries; expected {joint_total}")
    local_joint_counts = torch.zeros(joint_total, dtype=torch.long, device=params.device)
    available = torch.ones(n, dtype=torch.bool, device=params.device)
    selected: list[int] = []

    rms_all = torch.sqrt(torch.mean(g_clean.square(), dim=1)).clamp_min(1e-12)
    min_gap = torch.full((n,), float("inf"), dtype=g_clean.dtype, device=g_clean.device)
    tie = torch.from_numpy(rng.random(n).astype(np.float32)).to(params.device)

    for step in range(batch_size):
        # Marginal coverage has first priority. Joint-cell novelty is the next
        # priority, so the sampler does not collapse into a few combinations.
        marginal_bonus = torch.zeros(n, dtype=torch.int16, device=params.device)
        balance_score = torch.zeros(n, dtype=torch.float32, device=params.device)
        for j in range(len(free_idx)):
            ids = bin_ids[:, j]
            marginal_bonus += (local_counts[j, ids] == 0).to(torch.int16)
            balance_score += 1.0 / (1.0 + global_bin_counts[j, ids].float())
        joint_bonus = (local_joint_counts[joint_ids] == 0).to(torch.int16)
        coverage_bonus = 2 * marginal_bonus + joint_bonus
        balance_score += 0.5 / (1.0 + global_joint_counts[joint_ids].float())

        coverage_bonus[~available] = -1
        max_bonus = int(torch.max(coverage_bonus).item())
        eligible = available & (coverage_bonus == max_bonus)

        if strategy == "coverage_gdiverse" and step > 0:
            if target_gap_fraction > 0:
                hit = eligible & (min_gap >= target_gap_fraction)
                if torch.any(hit):
                    eligible = hit
            score = min_gap.float() + 1e-4 * balance_score + 1e-7 * tie
        else:
            score = balance_score + 1e-4 * tie

        score = torch.where(eligible, score, torch.full_like(score, -float("inf")))
        chosen = int(torch.argmax(score).item())
        selected.append(chosen)
        available[chosen] = False

        for j in range(len(free_idx)):
            bid = int(bin_ids[chosen, j].item())
            local_counts[j, bid] += 1
            global_bin_counts[j, bid] += 1
        jid = int(joint_ids[chosen].item())
        local_joint_counts[jid] += 1
        global_joint_counts[jid] += 1

        if strategy == "coverage_gdiverse":
            gap = relative_gap_to_one(g_clean, rms_all, g_clean[chosen], rms_all[chosen])
            min_gap = torch.minimum(min_gap, gap)
            min_gap[~available] = -float("inf")

    indices = torch.as_tensor(selected, dtype=torch.long, device=params.device)
    min_rel, median_rel = batch_gap_stats(g_clean[indices])

    coverage = {}
    for j, idx in enumerate(free_idx):
        hit = int(torch.count_nonzero(local_counts[j] > 0).item())
        coverage[PARAMETER_NAMES[idx]] = {
            "bins_hit": hit,
            "bins_total": int(bins),
            "coverage_fraction": hit / float(bins),
        }

    joint_hit = int(torch.count_nonzero(local_joint_counts > 0).item())
    global_joint_hit = int(torch.count_nonzero(global_joint_counts > 0).item())
    return indices, {
        "batch_min_g_gap_percent": 100.0 * min_rel,
        "batch_median_nearest_g_gap_percent": 100.0 * median_rel,
        "target_g_gap_percent": 100.0 * target_gap_fraction,
        "target_met": bool(np.isfinite(min_rel) and min_rel >= target_gap_fraction),
        "coverage": coverage,
        "joint_bins_hit_in_batch": joint_hit,
        "joint_bins_total": joint_total,
        "unique_joint_fraction_of_batch": joint_hit / float(batch_size),
        "global_joint_bins_coverage_fraction": global_joint_hit / float(joint_total),
    }


def collect_validation_rows(
    pool: np.memmap,
    usable_rows: int,
    stage: str,
    n: int,
    seed: int,
) -> np.ndarray:
    """Separate-pool validation: use every eligible row once before repeating."""
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    total = 0
    while total < n:
        order = rng.permutation(usable_rows).astype(np.int64, copy=False)
        for start in range(0, usable_rows, 8192):
            if total >= n:
                break
            ids = order[start : start + 8192]
            read_order = np.argsort(ids, kind="stable")
            rows_sorted = np.asarray(pool[ids[read_order]], dtype=np.float32)
            restore = np.empty_like(read_order)
            restore[read_order] = np.arange(len(ids), dtype=read_order.dtype)
            rows = apply_stage(rows_sorted[restore], stage)
            if len(rows) == 0:
                continue
            take = min(n - total, len(rows))
            chunks.append(rows[:take])
            total += take
    return np.concatenate(chunks, axis=0)


def add_fractional_noise(g: torch.Tensor, noise_level: float, generator: torch.Generator) -> torch.Tensor:
    if noise_level <= 0:
        return g
    rms = torch.sqrt(torch.mean(g.square(), dim=1, keepdim=True).clamp_min(1e-24))
    noise = torch.randn(g.shape, device=g.device, dtype=g.dtype, generator=generator)
    return g + float(noise_level) * rms * noise
