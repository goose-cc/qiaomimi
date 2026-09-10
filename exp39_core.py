#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Numerical core for Exp39 targeted observation design.

Exp39 does not change any parameter definition domain. It diagnoses fixed-m
(a1,gamma) slices under the current best Exp38 observation design and then
constructs q2 designs that maximize worst-slice/tolerance-scaled information.

The final authority remains the direct-verified continuous 3P alias audit from
exp38_core; the greedy design objective is only a search heuristic.
"""
from __future__ import annotations

import json, math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

from mc_physics import valid_parameter_mask_numpy
from exp38_core import forward64_batched, forward64, rms_rows, best_a1_and_sse
from exp38_observation import make_q2


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_frozen_parameter_domain(cfg: dict[str, Any]) -> None:
    pr = cfg["parameter_ranges"]
    expected = {"a1": [0.05, 0.20], "m": [0.40, 1.20], "gamma": [0.01, 1.0]}
    for k, v in expected.items():
        got = list(map(float, pr[k]))
        if not np.allclose(got, v, rtol=0.0, atol=1e-14):
            raise RuntimeError(f"Exp39 forbids changing {k} domain: got {got}, expected {v}")
    fx = cfg["fixed_parameters"]
    if abs(float(fx["a2"]) - 0.025) > 1e-14 or abs(float(fx["a3"])) > 1e-14:
        raise RuntimeError("Exp39 expects a2=0.025 and a3=0 fixed")


def current_q2(cfg: dict[str, Any]) -> np.ndarray:
    q = make_q2(cfg["current_best_design"])
    if np.any(q >= 0):
        raise RuntimeError("q2 must remain in q2<0")
    return q


def candidate_q2(cfg: dict[str, Any]) -> np.ndarray:
    td = cfg["targeted_design"]
    parts = []
    for seg in td["candidate_q2_segments"]:
        parts.append(make_q2(seg))
    # Guarantee exact inclusion of current-best and seed points.
    parts.append(current_q2(cfg))
    parts.append(np.asarray(td["seed_q2"], dtype=np.float64))
    q = np.unique(np.concatenate(parts).astype(np.float64))
    q.sort()
    if len(q) < max(int(d["points"]) for d in td["designs"]):
        raise RuntimeError("candidate q2 pool is smaller than requested design size")
    if np.any(~np.isfinite(q)) or np.any(q >= 0):
        raise RuntimeError("invalid candidate q2 pool")
    return q


def generate_slice_anchors(cfg: dict[str, Any], m_values: np.ndarray,
                           anchors_per_slice: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Same number of full-domain (a1,gamma) anchors at every fixed m slice."""
    pr = cfg["parameter_ranges"]; fx = cfg["fixed_parameters"]
    n = int(anchors_per_slice); all_p = []; slice_id = []
    lo, hi = math.log10(pr["gamma"][0]), math.log10(pr["gamma"][1])
    for si, mv in enumerate(np.asarray(m_values, dtype=np.float64)):
        sampler = qmc.Sobol(d=2, scramble=True, seed=int(seed) + 997 * si)
        m2 = int(math.ceil(math.log2(max(n, 2))))
        u = sampler.random_base2(m2)[:n]
        a1 = pr["a1"][0] + u[:, 0] * (pr["a1"][1] - pr["a1"][0])
        gam = 10.0 ** (lo + u[:, 1] * (hi - lo))
        p = np.column_stack([
            a1, np.full(n, fx["a2"]), np.full(n, fx["a3"]),
            np.full(n, float(mv)), gam
        ]).astype(np.float64)
        if not np.all(valid_parameter_mask_numpy(p)):
            raise RuntimeError(f"invalid anchors at m={mv}")
        all_p.append(p); slice_id.extend([si] * n)
    return np.concatenate(all_p, axis=0), np.asarray(slice_id, dtype=np.int64)


def _forward_shifted(params: np.ndarray, q2: np.ndarray, cfg: dict[str, Any], col: int,
                     plus: np.ndarray, minus: np.ndarray | None, log_mode: bool = False):
    """Finite difference helper with explicit one-sided handling at domain boundaries."""
    integ = int(cfg["physics_integration_points"])
    base = np.asarray(params, dtype=np.float64)
    pp = base.copy(); pm = base.copy()
    if log_mode:
        pp[:, col] *= np.exp(plus)
        gp = forward64_batched(pp, q2, integ)
        if minus is None:
            g0 = forward64_batched(base, q2, integ)
            return gp, g0
        pm[:, col] *= np.exp(-minus)
        gm = forward64_batched(pm, q2, integ)
        return gp, gm
    pp[:, col] += plus
    gp = forward64_batched(pp, q2, integ)
    if minus is None:
        g0 = forward64_batched(base, q2, integ)
        return gp, g0
    pm[:, col] -= minus
    gm = forward64_batched(pm, q2, integ)
    return gp, gm


def scaled_jacobian(params: np.ndarray, q2: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Tolerance-scaled J=[da1,dm,dloggamma], no noise normalization yet.

    A unit movement in each column corresponds to the recovery tolerance:
    da1=0.005, dm=0.03, dlog(gamma)=log(1.2).
    """
    p = np.asarray(params, dtype=np.float64); q2 = np.asarray(q2, dtype=np.float64)
    pr = cfg["parameter_ranges"]; tol = cfg["recovery_tolerances"]
    integ = int(cfg["physics_integration_points"]); frac = 0.04
    base = forward64_batched(p, q2, integ)
    cols = []

    # a1 derivative, central unless close to a boundary.
    unit = float(tol["a1_abs"]); h = frac * unit
    out = np.empty_like(base)
    interior = (p[:, 0] - h >= pr["a1"][0]) & (p[:, 0] + h <= pr["a1"][1])
    if np.any(interior):
        pp = p[interior].copy(); pm = p[interior].copy(); pp[:,0]+=h; pm[:,0]-=h
        out[interior] = (forward64_batched(pp,q2,integ)-forward64_batched(pm,q2,integ))/(2*h/unit)
    lo = ~interior & (p[:,0] - h < pr["a1"][0]); hi = ~interior & ~lo
    if np.any(lo):
        pp=p[lo].copy(); pp[:,0]+=h
        out[lo]=(forward64_batched(pp,q2,integ)-base[lo])/(h/unit)
    if np.any(hi):
        pm=p[hi].copy(); pm[:,0]-=h
        out[hi]=(base[hi]-forward64_batched(pm,q2,integ))/(h/unit)
    cols.append(out)

    # m derivative.
    unit = float(tol["m_abs"]); h = frac * unit
    out = np.empty_like(base)
    interior = (p[:, 3] - h >= pr["m"][0]) & (p[:, 3] + h <= pr["m"][1])
    if np.any(interior):
        pp=p[interior].copy();pm=p[interior].copy();pp[:,3]+=h;pm[:,3]-=h
        out[interior]=(forward64_batched(pp,q2,integ)-forward64_batched(pm,q2,integ))/(2*h/unit)
    lo = ~interior & (p[:,3] - h < pr["m"][0]); hi = ~interior & ~lo
    if np.any(lo):
        pp=p[lo].copy();pp[:,3]+=h
        out[lo]=(forward64_batched(pp,q2,integ)-base[lo])/(h/unit)
    if np.any(hi):
        pm=p[hi].copy();pm[:,3]-=h
        out[hi]=(base[hi]-forward64_batched(pm,q2,integ))/(h/unit)
    cols.append(out)

    # log(gamma) derivative.
    unit = math.log(float(tol["gamma_factor"])); h = frac * unit
    out = np.empty_like(base); lg = np.log(p[:,4]); lmin=math.log(pr["gamma"][0]); lmax=math.log(pr["gamma"][1])
    interior = (lg-h >= lmin) & (lg+h <= lmax)
    if np.any(interior):
        pp=p[interior].copy();pm=p[interior].copy();pp[:,4]*=math.exp(h);pm[:,4]*=math.exp(-h)
        out[interior]=(forward64_batched(pp,q2,integ)-forward64_batched(pm,q2,integ))/(2*h/unit)
    lo = ~interior & (lg-h < lmin); hi = ~interior & ~lo
    if np.any(lo):
        pp=p[lo].copy();pp[:,4]*=math.exp(h)
        out[lo]=(forward64_batched(pp,q2,integ)-base[lo])/(h/unit)
    if np.any(hi):
        pm=p[hi].copy();pm[:,4]*=math.exp(-h)
        out[hi]=(base[hi]-forward64_batched(pm,q2,integ))/(h/unit)
    cols.append(out)
    return np.stack(cols, axis=2), base


def jacobian_metrics(params: np.ndarray, q2: np.ndarray, cfg: dict[str, Any]):
    J, base = scaled_jacobian(params, q2, cfg)
    sigma = float(cfg["reference_noise"]) * rms_rows(base)
    Jn = J / np.maximum(sigma[:, None, None], 1e-30)
    sv = np.linalg.svd(Jn, compute_uv=False); cond = sv[:,0] / np.maximum(sv[:,-1], 1e-30)
    J2 = Jn[:, :, [0,2]]; sv2 = np.linalg.svd(J2, compute_uv=False); cond2 = sv2[:,0] / np.maximum(sv2[:,-1],1e-30)
    def cos(a,b):
        return np.sum(a*b,axis=1)/np.maximum(np.sqrt(np.sum(a*a,axis=1)*np.sum(b*b,axis=1)),1e-30)
    return {
        "sigma_min_3p": sv[:,-1], "condition_3p": cond,
        "sigma_min_agamma": sv2[:,-1], "condition_agamma": cond2,
        "cos_a1_m": cos(Jn[:,:,0],Jn[:,:,1]),
        "cos_a1_loggamma": cos(Jn[:,:,0],Jn[:,:,2]),
        "cos_m_loggamma": cos(Jn[:,:,1],Jn[:,:,2]),
    }


def _background(cfg: dict[str, Any], q2: np.ndarray) -> np.ndarray:
    pr=cfg["parameter_ranges"];fx=cfg["fixed_parameters"]
    pp=np.array([[0.0,fx["a2"],fx["a3"],0.5*(pr["m"][0]+pr["m"][1]),0.5]],dtype=np.float64)
    return forward64(pp,q2,int(cfg["physics_integration_points"]))[0]


def fixed_m_alias_dense(params: np.ndarray, q2: np.ndarray, cfg: dict[str, Any], gamma_points: int = 321) -> np.ndarray:
    """Direct-stable dense-grid 2P alias margin with m held exactly fixed.

    This is a slice-ranking diagnostic, not the final 3P continuous audit.
    It searches the OR failure set: |da1|>=tol OR gamma-factor>=tol.
    """
    p=np.asarray(params,dtype=np.float64); tol=cfg["recovery_tolerances"];pr=cfg["parameter_ranges"];fx=cfg["fixed_parameters"]
    uniq=np.unique(np.round(p[:,3],12));
    if len(uniq)!=1: raise ValueError("fixed_m_alias_dense expects one m slice at a time")
    mv=float(uniq[0]); q2=np.asarray(q2,dtype=np.float64); integ=int(cfg["physics_integration_points"])
    bg=_background(cfg,q2); lg=np.linspace(math.log10(pr["gamma"][0]),math.log10(pr["gamma"][1]),int(gamma_points)); gam=10**lg
    bankp=np.column_stack([np.ones(len(gam)),np.full(len(gam),fx["a2"]),np.full(len(gam),fx["a3"]),np.full(len(gam),mv),gam])
    R=forward64_batched(bankp,q2,integ)-bg[None,:]; r2=np.einsum('ij,ij->i',R,R)
    clean=forward64_batched(p,q2,integ); ref=float(cfg["reference_noise"]); out=np.full(len(p),np.inf)
    amin,amax=map(float,pr["a1"]); dA=float(tol["a1_abs"]); dLG=math.log10(float(tol["gamma_factor"]))
    for k in range(len(p)):
        y=clean[k]-bg; truea=float(p[k,0]); truelg=math.log10(float(p[k,4])); best=math.inf
        # a1-low/high regions across all gamma grid points.
        for ab in ((amin,min(amax,truea-dA)),(max(amin,truea+dA),amax)):
            if ab[0] <= ab[1]+1e-14:
                a=np.clip((R@y)/r2,ab[0],ab[1]); resid=y[None,:]-a[:,None]*R; sse=np.einsum('ij,ij->i',resid,resid); best=min(best,float(sse.min()))
        # gamma-low/high regions with free a1.
        afree=np.clip((R@y)/r2,amin,amax); resid=y[None,:]-afree[:,None]*R; sse=np.einsum('ij,ij->i',resid,resid)
        low=lg <= truelg-dLG+1e-14; high=lg >= truelg+dLG-1e-14
        if np.any(low): best=min(best,float(sse[low].min()))
        if np.any(high): best=min(best,float(sse[high].min()))
        denom=max(ref*float(rms_rows(clean[k:k+1])[0]),1e-30)
        out[k]=math.sqrt(max(best,0.0)/len(q2))/denom
    return out


def make_candidate_information(cfg: dict[str, Any], qpool: np.ndarray, m_values: np.ndarray,
                               anchors_per_slice: int, seed: int):
    """Per-slice Fisher-like information contribution of every candidate q2 point."""
    params,sid=generate_slice_anchors(cfg,m_values,anchors_per_slice,seed)
    J, _ = scaled_jacobian(params,qpool,cfg)
    # Fixed normalization: sigma from the current best design for each anchor.
    base_current=forward64_batched(params,current_q2(cfg),int(cfg["physics_integration_points"]))
    sigma=float(cfg["reference_noise"])*rms_rows(base_current)
    V=J/np.maximum(sigma[:,None,None],1e-30)
    S=len(m_values);Q=len(qpool);B3=np.empty((S,Q,3,3),dtype=np.float64);B2=np.empty((S,Q,2,2),dtype=np.float64)
    for s in range(S):
        vv=V[sid==s]
        B3[s]=np.einsum('nqi,nqj->qij',vv,vv)/max(len(vv),1)
        ww=vv[:,:,[0,2]]
        B2[s]=np.einsum('nqi,nqj->qij',ww,ww)/max(len(ww),1)
    return B3,B2,params,sid


def _map_points_to_pool(qpool: np.ndarray, values: np.ndarray) -> list[int]:
    q=np.asarray(qpool,float); out=[]
    for x in np.asarray(values,float):
        j=int(np.argmin(np.abs(q-x)))
        if j not in out: out.append(j)
    return out


def _robust_score(mats: np.ndarray, quantile: float) -> np.ndarray:
    """mats shape [slice,candidate,d,d] -> robust min-eigen score per candidate."""
    ev=np.linalg.eigvalsh(mats)[...,0]
    score=np.quantile(ev,float(quantile),axis=0)
    # Tiny tie-break with the mean, irrelevant once scores are nonzero.
    return score + 1e-12*np.mean(ev,axis=0)


def greedy_design(qpool: np.ndarray, B3: np.ndarray, B2: np.ndarray, total_points: int,
                  mode: str, robust_quantile: float, target_slices: np.ndarray,
                  seed_points: np.ndarray, mixed_target_fraction: float = 0.45,
                  initial_indices: list[int] | None = None, progress_every: int = 25) -> np.ndarray:
    Q=len(qpool); selected=[] if initial_indices is None else list(dict.fromkeys(map(int,initial_indices)))
    for j in _map_points_to_pool(qpool,seed_points):
        if j not in selected and len(selected)<int(total_points): selected.append(j)
    chosen=np.zeros(Q,dtype=bool);chosen[selected]=True
    A3=np.sum(B3[:,selected],axis=1) if selected else np.zeros((B3.shape[0],3,3))
    A2=np.sum(B2[:,selected],axis=1) if selected else np.zeros((B2.shape[0],2,2))
    target_slices=np.asarray(target_slices,dtype=int)
    target_steps=int(round(float(mixed_target_fraction)*int(total_points)))

    while len(selected)<int(total_points):
        rem=np.flatnonzero(~chosen)
        if mode=="robust3p": which="3p"
        elif mode=="worstslice_agamma": which="ag"
        elif mode in ("mixed","augment_current"):
            which="ag" if len(selected)<target_steps and mode=="mixed" else ("ag" if mode=="augment_current" else "3p")
        else: raise ValueError(mode)
        if which=="ag":
            mats=A2[target_slices,None,:,:]+B2[target_slices][:,rem,:,:]
        else:
            mats=A3[:,None,:,:]+B3[:,rem,:,:]
        score=_robust_score(mats,robust_quantile)
        j=int(rem[int(np.argmax(score))]);selected.append(j);chosen[j]=True;A3+=B3[:,j];A2+=B2[:,j]
        if len(selected)%int(progress_every)==0 or len(selected)==int(total_points):
            print(f"    {mode}: {len(selected)}/{total_points} q2 points")
    return np.sort(np.asarray(selected,dtype=int))


def design_objective_summary(indices: np.ndarray, B3: np.ndarray, B2: np.ndarray,
                             target_slices: np.ndarray) -> dict[str,float]:
    A3=np.sum(B3[:,indices],axis=1);A2=np.sum(B2[:,indices],axis=1)
    e3=np.linalg.eigvalsh(A3)[:,0];e2=np.linalg.eigvalsh(A2)[:,0]
    return {
        "fim3_lambda_min_min_slice":float(np.min(e3)),
        "fim3_lambda_min_p10_slice":float(np.quantile(e3,.1)),
        "fim3_lambda_min_median_slice":float(np.median(e3)),
        "agamma_lambda_min_min_target_slice":float(np.min(e2[target_slices])),
        "agamma_lambda_min_median_target_slice":float(np.median(e2[target_slices])),
    }
