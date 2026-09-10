#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Observation-grid construction for Exp38.

The spectral density / forward kernel are unchanged. Exp38 changes only which
Euclidean q^2 locations are observed and, for hybrid designs, concatenates
multiple q^2 windows into one response vector g.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def _segment(seg: dict) -> np.ndarray:
    kind = str(seg["kind"])
    if kind == "linear":
        q = np.linspace(float(seg["q2_min"]), float(seg["q2_max"]), int(seg["points"]), dtype=np.float64)
    elif kind == "geom_abs":
        amax = float(seg["abs_max"]); amin = float(seg["abs_min"])
        if not (amax > amin > 0):
            raise ValueError("geom_abs requires abs_max > abs_min > 0")
        q = -np.geomspace(amax, amin, int(seg["points"]), dtype=np.float64)
    else:
        raise ValueError(f"unknown q2 segment kind: {kind}")
    return q


def make_q2(design: dict) -> np.ndarray:
    if design.get("kind") == "hybrid":
        parts = [_segment(s) for s in design["segments"]]
        q = np.concatenate(parts)
    else:
        q = _segment(design)
    # Sort from most negative to least negative and remove near-duplicates.
    q = np.sort(np.asarray(q, dtype=np.float64))
    keep = np.ones(len(q), dtype=bool)
    if len(q) > 1:
        keep[1:] = np.diff(q) > 1e-12
    q = q[keep]
    if len(q) < 8 or np.any(~np.isfinite(q)):
        raise ValueError("observation design must contain at least 8 finite q2 points")
    if np.any(q >= 0.0):
        raise ValueError("Exp38 q2 designs must stay in the Euclidean q2<0 domain")
    return q


def design_by_id(cfg: dict, design_id: str) -> dict:
    matches = [d for d in cfg["observation_designs"] if d["id"] == design_id]
    if len(matches) != 1:
        raise KeyError(f"design id {design_id!r} not found uniquely")
    return matches[0]


def describe_design(cfg: dict, design: dict) -> dict:
    q = make_q2(design)
    oq0, oq1 = map(float, cfg["original_q2_domain"])
    return {
        "design_id": design["id"],
        "q2_points": int(len(q)),
        "q2_min": float(q.min()),
        "q2_max": float(q.max()),
        "extends_below_original_q2_min": int(float(q.min()) < oq0 - 1e-12),
        "extends_above_original_q2_max": int(float(q.max()) > oq1 + 1e-12),
        "uses_nonuniform_spacing": int(np.std(np.diff(q)) > 1e-8 * max(abs(float(np.mean(np.diff(q)))), 1.0)),
        "median_abs_q2_spacing": float(np.median(np.abs(np.diff(q)))) if len(q)>1 else 0.0,
    }


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
