#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small CPU-only tests for analyze_exp7_visibility.py helper logic."""
from __future__ import annotations

import numpy as np

from analyze_exp7_visibility import (
    parse_visibility_bins,
    summarize_bins,
    visibility_bin_indices,
)


def main() -> None:
    edges = parse_visibility_bins("0,0.05,0.20,1.000001")
    visibility = np.array([0.00, 0.01, 0.05, 0.19, 0.20, 0.80, 1.0])
    ids = visibility_bin_indices(visibility, edges)
    expected = np.array([0, 0, 1, 1, 2, 2, 2])
    if not np.array_equal(ids, expected):
        raise AssertionError(f"bin ids mismatch: {ids} != {expected}")

    n = len(visibility)
    norm_abs = np.zeros((n, 5), dtype=np.float64)
    norm_abs[:, 3] = np.array([0.25, 0.24, 0.20, 0.18, 0.10, 0.05, 0.02])
    norm_abs[:, 4] = np.array([0.26, 0.25, 0.21, 0.19, 0.11, 0.06, 0.03])

    rows = summarize_bins(
        edges=edges,
        true_visibility=visibility,
        pred_visibility=np.clip(visibility + 0.01, 0.0, 1.0),
        norm_abs=norm_abs,
        log_gamma_abs=np.linspace(1.0, 0.1, n),
        width_rel=np.linspace(2.0, 0.2, n),
        f_rel=np.linspace(0.5, 0.1, n),
        g_rel=np.linspace(0.05, 0.01, n),
    )
    if sum(row["count"] for row in rows) != n:
        raise AssertionError("bin counts do not add up")
    if not (rows[0]["m_norm_abs_mean"] > rows[-1]["m_norm_abs_mean"]):
        raise AssertionError("synthetic visibility trend was not preserved")

    print("EXP7 VISIBILITY ANALYSIS TEST PASSED")


if __name__ == "__main__":
    main()
