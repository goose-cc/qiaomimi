#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fast numerical self-check for Exp37 before the full experiment."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from mc_physics import scaled_forward_observation_numpy
from exp37_core import (
    forward64_batched, make_profile_bank, profile_alias_margins,
    refine_alias_ids, alias_margin_direct, unacceptable, varpro_predict_one,
)
from analyze_exp37_jacobian import finite_diff_scaled_jacobian


def main():
    cfg=json.loads(Path("exp37_3p_config.json").read_text(encoding="utf-8"))
    cfg=json.loads(json.dumps(cfg))
    cfg["profile_grid"]={"m_points":31,"log_gamma_points":31}
    cfg["continuous_prebank"]["continuous_maxiter"]=24
    cfg["continuous_prebank"]["continuous_multistart"]=2
    integ=int(cfg["physics_integration_points"])
    fixed=cfg["fixed_parameters"]
    params=np.array([
        [0.07,fixed["a2"],fixed["a3"],0.48,0.018],
        [0.11,fixed["a2"],fixed["a3"],0.63,0.065],
        [0.16,fixed["a2"],fixed["a3"],0.82,0.18],
        [0.19,fixed["a2"],fixed["a3"],1.05,0.55],
        [0.09,fixed["a2"],fixed["a3"],1.17,0.85],
        [0.14,fixed["a2"],fixed["a3"],0.43,0.32],
    ],dtype=np.float64)

    project=scaled_forward_observation_numpy(params,integration_points=integ)
    g64=forward64_batched(params,integ)
    rel=float(np.linalg.norm(g64.astype(np.float32).astype(np.float64)-project.astype(np.float64))/
              max(np.linalg.norm(project.astype(np.float64)),1e-30))
    if rel>1e-7:
        raise RuntimeError(f"forward64/project mismatch {rel}")

    mgrid,lggrid,bg,R,r2=make_profile_bank(cfg,integ)
    diag=profile_alias_margins(g64,params,cfg,bg,mgrid,lggrid,R,r2,cfg["recovery_tolerances"],chunk_size=6)
    rows=refine_alias_ids(
        np.arange(len(params)),params,g64,cfg,bg,mgrid,lggrid,R,r2,diag,integ,
        maxiter=24,multistart=2,progress_every=6,
    )
    for r in rows:
        sid=int(r["state_id"])
        if float(r["recovery_alias_margin_refined"]) > float(r["recovery_alias_margin_grid_verified"])+1e-9:
            raise RuntimeError("refined margin exceeds verified grid margin")
        alias=params[sid].copy()
        alias[0]=float(r["alias_a1_refined"]); alias[3]=float(r["alias_m_refined"]); alias[4]=float(r["alias_gamma_refined"])
        if not unacceptable(params[sid],alias,cfg["recovery_tolerances"]):
            raise RuntimeError("refined alias violates unacceptable-alternative definition")
        direct=alias_margin_direct(params[sid],alias,cfg,integ)
        if not np.isclose(direct,float(r["recovery_alias_margin_refined"]),rtol=2e-6,atol=2e-9):
            raise RuntimeError("direct alias margin verification failed")

    # VarPro clean-fit check on three states. Parameter uniqueness is not assumed;
    # only the direct physical fit itself must be nearly exact.
    fit_rel=[]
    for i in range(3):
        pred=varpro_predict_one(
            g64[i],cfg,bg,mgrid,lggrid,R,r2,integ,seed_count=3,maxiter=30
        )
        fit_rel.append(float(pred["fit_relL2"]))
    if max(fit_rel)>5e-5:
        raise RuntimeError(f"VarPro clean fit unexpectedly poor: {fit_rel}")

    # Jacobian must be finite and have three nonnegative singular values.
    _,J=finite_diff_scaled_jacobian(params[2],cfg,integ)
    S=np.linalg.svd(J,compute_uv=False)
    if not np.isfinite(J).all() or not np.isfinite(S).all() or np.any(S<0):
        raise RuntimeError("Jacobian numerical check failed")

    print("="*100)
    print("EXP37 NUMERICAL SELF-CHECK PASS")
    print("forward64 roundtrip relL2:",rel)
    print("refined<=grid checked states:",len(rows))
    print("VarPro clean fit max relL2:",max(fit_rel))
    print("Jacobian singular values:",S.tolist())
    print("="*100)


if __name__=="__main__":
    main()
