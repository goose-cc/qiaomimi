#!/usr/bin/env python3
"""Five-parameter V2, phase 1: physics candidate/alias audit and paired q2 pilot.

This file deliberately DOES NOT produce a final 12k-state dataset and never
screens states by TEST g. It uses existing source-of-truth physics/data writers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from build_fivep_dataset import (
    BOUNDS, NAMES, RECOVERY, REF_NOISE, FORMULA, INTEGRATION_POINTS,
    assert_source_contract, create_states, generate_curves, fingerprint,
    write_examples, write_manifest, write_normalization, state_identity,
)
from mc_physics import valid_parameter_mask_numpy, scaled_forward_observation_at_q2_numpy

LOW = np.array([BOUNDS[k][0] for k in NAMES], dtype=np.float64)
HIGH = np.array([BOUNDS[k][1] for k in NAMES], dtype=np.float64)
AUDIT_K = (1., 2., 3., 4., 6.)


def sha_raw(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def q2_fingerprint(q2: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(q2, '<f8').tobytes()).hexdigest()


def load_historical_q2(path: Path) -> tuple[np.ndarray, dict]:
    """Use original .npy bytes. Do not reconstruct or force nominal '240'."""
    path = path.resolve(strict=True)
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 1 or arr.dtype.kind != 'f':
        raise ValueError('Historical q2.npy must be a one-dimensional floating-point array')
    q = np.asarray(arr, np.float64)
    if not (len(q) >= 100 and np.isfinite(q).all() and
            np.all(np.diff(q) > 0.) and np.all(q < 0.)):
        raise ValueError('Historical q2 is nonfinite, nonascending, or not Euclidean')
    # Historical hybrid label is nominal. The reference file is authoritative.
    if len(q) not in (239, 240):
        raise ValueError(f'Expected original hybrid 239/240 entries, got {len(q)}: verify original CP02 dataset')
    # The old CP02 generator recorded a hash of raw float64 q2 VALUES (not
    # the .npy file). Check it when its original metadata accompanies q2.npy.
    meta_file = path.parent/'metadata.json'
    meta_sha_checked = False
    if meta_file.is_file():
        saved = json.loads(meta_file.read_text(encoding='utf-8'))
        stored = saved.get('q2_sha256')
        if stored is not None:
            if stored != q2_fingerprint(q):
                raise ValueError('Saved CP02 metadata q2_sha256 disagrees with original q2.npy')
            meta_sha_checked = True
        stated_count = saved.get('q2_points')
        if stated_count is not None and int(stated_count) != len(q):
            raise ValueError('Saved CP02 metadata point count disagrees with q2.npy')
    return q, {'source_path': str(path), 'source_file_sha256': sha_raw(path),
               'q2_values_sha256': q2_fingerprint(q), 'historical_metadata_checked': meta_sha_checked,
               'point_count': int(len(q)),
               'q2_min': float(q[0]), 'q2_max': float(q[-1]),
               'design_label': 'hybrid_ultranear_240 (historical reference)'}


def bins_for(p: np.ndarray, nbin: int) -> np.ndarray:
    z = (p - LOW) / (HIGH - LOW)
    z[:, 4] = np.log(p[:, 4] / LOW[4]) / np.log(HIGH[4] / LOW[4])
    return np.minimum((np.clip(z, 0., 1.) * nbin).astype(np.int32), nbin-1)


def coverage_counts(groups: dict[str, np.ndarray], nbin: int) -> list[dict]:
    results = []
    for name, p in groups.items():
        b = bins_for(np.asarray(p, np.float64).reshape(-1, 5), nbin)
        for j, param in enumerate(NAMES):
            for k in range(nbin):
                lo = LOW[j] + (HIGH[j] - LOW[j]) * k / nbin
                hi = LOW[j] + (HIGH[j] - LOW[j]) * (k + 1) / nbin
                if j == 4:
                    lo = np.exp(np.log(LOW[4]) + np.log(HIGH[4]/LOW[4]) * k / nbin)
                    hi = np.exp(np.log(LOW[4]) + np.log(HIGH[4]/LOW[4]) * (k+1) / nbin)
                results.append(dict(group=name, parameter=param, bin=k,
                                    bin_lower=f'{lo:.16g}', bin_upper=f'{hi:.16g}',
                                    count=int(np.sum(b[:, j] == k))))
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Empty rows')
    with path.open('w', newline='', encoding='utf-8-sig') as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)



def generate_candidate_pool(target: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Scrambled Sobol in full box (log gamma), then original rho>=0 gate.

    Returns raw candidate states, ALL physically accepted candidates, and
    selected accepted unique pool of the requested size (boundary anchors first).
    """
    anchors, anch_log = create_states(60, seed, 5)
    anchors = anchors[:int(anch_log['physical_endpoint_anchors'])]
    raw_parts = [anchors]
    good_parts = [anchors]
    selected = [row.copy() for row in anchors]
    selected_keys = {row.astype('<f8').tobytes() for row in selected}
    rounds = 0
    # Full-power-of-2 scrambled Sobol blocks, not multiple random sample() calls.
    while len(selected) < target and rounds < 25:
        remaining = target - len(selected)
        power = max(8, int(math.ceil(math.log2(max(256,remaining*2)))))
        u = qmc.Sobol(d=5, scramble=True, seed=seed+7919*rounds).random_base2(power)
        p = LOW+u*(HIGH-LOW)
        p[:,4] = np.exp(np.log(LOW[4]) + u[:,4]*np.log(HIGH[4]/LOW[4]))
        valid = valid_parameter_mask_numpy(p)
        raw_parts.append(p)
        good_parts.append(p[valid])
        for row in p[valid]:
            k=row.astype('<f8').tobytes()
            if k not in selected_keys:
                selected.append(row.copy()); selected_keys.add(k)
                if len(selected)==target: break
        rounds += 1
    if len(selected) != target:
        raise RuntimeError(f'Only {len(selected)} distinct valid states found; increase sampling rounds')
    raw=np.vstack(raw_parts)
    physical=np.vstack(good_parts)
    chosen=np.stack(selected)
    if not valid_parameter_mask_numpy(chosen).all():
        raise RuntimeError('Candidate physical constraint bypassed')
    return raw, physical, chosen, {
       'raw_candidate_count':len(raw),'physically_accepted_count':len(physical),
       'selected_valid_unique_count':len(chosen),'endpoint_anchor_count':len(anchors),
       'rounds':rounds,'gamma_sampler':'scrambled Sobol uniform in log(gamma)',
       'sobol_note':'Physical filtering changes exact low-discrepancy distribution',
    }


def sigma_of(g: np.ndarray) -> np.ndarray:
    return REF_NOISE * np.sqrt(np.mean(np.asarray(g, np.float64)**2, axis=1))


def pair_info(p: np.ndarray, i: np.ndarray, j: np.ndarray, d: np.ndarray) -> dict:
    delta = np.abs(p[i] - p[j])
    factors = np.maximum(p[i, 4] / p[j, 4], p[j, 4] / p[i, 4])
    # Informal physical proximity, including a log-scale gamma coordinate.
    scaled = delta / (HIGH - LOW)
    logscaled = np.abs(np.log(p[i, 4]/p[j, 4])) / np.log(HIGH[4]/LOW[4])
    scaled[:, 4] = logscaled
    near_local = np.max(scaled, axis=1) <= .05
    incompatible = ((delta[:, 0] > RECOVERY['a1_abs']) |
                    (delta[:, 3] > RECOVERY['m_abs']) |
                    (factors > RECOVERY['gamma_factor']))
    disjoint = ((delta[:, 0] > 2*RECOVERY['a1_abs']) |
                (delta[:, 3] > 2*RECOVERY['m_abs']) |
                (factors > RECOVERY['gamma_factor']**2))
    return dict(deltas=delta, gamma_factor=factors, near_local=near_local,
                incompatible=incompatible, disjoint=disjoint,
                potential_alias=(d < 6) & incompatible)


def audit_nearest(p: np.ndarray, g: np.ndarray, n_neighbors: int = 48,
                  projection_dim: int = 10, basis_size: int = 768, block: int = 128):
    """Shortlisted kNN in an SVD projection, then EXACT physics-distance rerank.

    This is an audit of probed neighbors, NOT an all-domain/no-alias proof.
    """
    n, nq = g.shape
    if n < 3 or p.shape != (n, 5):
        raise ValueError('Need >=3 physical states of shape [N,5]')
    sig = sigma_of(g)
    if np.any(sig <= 0.) or not np.isfinite(g).all():
        raise ValueError('Invalid candidate g or sigma')
    # Fit a low-dimensional search index from a deterministic subsample;
    # exact 240/100-dimensional D is recomputed for EVERY shortlisted pair.
    sample_idx = np.unique(np.linspace(0, n-1, min(n, basis_size)).astype(int))
    center = np.mean(g[sample_idx], axis=0)
    mat = g[sample_idx] - center
    _, _, vt = np.linalg.svd(mat, full_matrices=False)
    rank = min(int(projection_dim), len(vt))
    basis = vt[:rank].T
    coords = np.asarray((g-center) @ basis, np.float64)
    tree = cKDTree(coords)
    k = min(n, max(2, int(n_neighbors) + 1))
    _, nearest = tree.query(coords, k=k, workers=1)
    nearest = np.asarray(nearest, dtype=np.int64)
    if nearest.ndim == 1:
        nearest = nearest[:, None]
    chosen_idx = np.empty(n, dtype=np.int64)
    nearest_d = np.full(n, np.inf, dtype=np.float64)
    nearest_count = np.zeros((len(AUDIT_K),), dtype=np.int64)
    conflict_count = np.zeros((len(AUDIT_K),), dtype=np.int64)
    local_count = np.zeros((len(AUDIT_K),), dtype=np.int64)
    # One nearest row per state and ALL shortlisted neighbors evaluated at
    # the exact full observation length. This avoids O(N^2) in 50k/100k.
    for left in range(0, n, block):
        right = min(n, left+block)
        ix = np.arange(left, right)
        j_all = nearest[left:right]
        den = np.maximum(sig[left:right, None], sig[j_all])
        dd = np.sqrt(np.sum((g[left:right, None, :] - g[j_all])**2, axis=2)) / den
        dd[j_all == ix[:, None]] = np.inf
        z = np.argmin(dd, axis=1)
        chosen_idx[left:right] = j_all[np.arange(len(ix)), z]
        nearest_d[left:right] = dd[np.arange(len(ix)), z]
        pp = p[ix[:, None]]
        oth = p[j_all]
        delta = np.abs(pp - oth)
        fac = np.maximum(pp[...,4]/oth[...,4], oth[...,4]/pp[...,4])
        conflict = ((delta[...,0] > RECOVERY['a1_abs']) |
                    (delta[...,3] > RECOVERY['m_abs']) |
                    (fac > RECOVERY['gamma_factor']))
        relative = delta/(HIGH-LOW)
        relative[...,4] = np.abs(np.log(pp[...,4]/oth[...,4]))/np.log(HIGH[4]/LOW[4])
        local = np.max(relative,axis=2) <= .05
        for t, threshold in enumerate(AUDIT_K):
            flag = dd < threshold
            nearest_count[t] += int(np.sum(np.any(flag, axis=1)))
            conflict_count[t] += int(np.sum(np.any(flag & conflict, axis=1)))
            local_count[t] += int(np.sum(np.any(flag & local, axis=1)))
    target = pair_info(p, np.arange(n), chosen_idx, nearest_d)
    rows = []
    for ix in range(n):
        rows.append(dict(candidate_index=ix, neighbor_index=int(chosen_idx[ix]),
                         D_g=f'{nearest_d[ix]:.10g}',
                         delta_a1=f'{target["deltas"][ix,0]:.10g}',
                         delta_a2=f'{target["deltas"][ix,1]:.10g}',
                         delta_a3=f'{target["deltas"][ix,2]:.10g}',
                         delta_m=f'{target["deltas"][ix,3]:.10g}',
                         gamma_factor=f'{target["gamma_factor"][ix]:.10g}',
                         physical_near_5pct=bool(target['near_local'][ix]),
                         recovery_gate_conflict=bool(target['incompatible'][ix]),
                         recovery_acceptance_windows_disjoint=bool(target['disjoint'][ix]),
                         potential_alias=bool(target['potential_alias'][ix])))
    threshold_rows = []
    for ix, threshold in enumerate(AUDIT_K):
        threshold_rows.append(dict(D_less_than=threshold,
                                   states_with_probed_neighbor=int(nearest_count[ix]),
                                   states_with_probed_recovery_conflict=int(conflict_count[ix]),
                                   states_with_probed_physical_local_neighbor=int(local_count[ix]),
                                   method='projected shortlist; exact D per probed pair; lower bounds on global counts'))
    # Projection diagnostic: if truncation is too severe, increase dimensions/k.
    residual = (g - center) - coords @ basis.T
    residual_rms = np.sqrt(np.mean(residual**2, axis=1))
    projection_noise_ratio = residual_rms/np.maximum(sig,1e-300)
    diag = {'search_method':'SVD-projected kNN shortlist, then full-q2 exact rerank',
            'search_is_exhaustive': False, 'projection_dim': rank,
            'neighbor_shortlist': k-1, 'candidate_count': n,
            'projection_residual_rms_over_noise_p50': float(np.median(projection_noise_ratio)),
            'projection_residual_rms_over_noise_p99': float(np.quantile(projection_noise_ratio,.99)),
            'closest_probed_D_g': float(np.min(nearest_d)),
            'normal_local_definition': 'max normalized parameter coord distance <=0.05; gamma in log range',
            'potential_alias_definition': 'D_g<6 and at least one 3P recovery truth difference exceeds tolerance',
            'disjoint_windows_definition': 'a1 diff>0.010 or m diff>0.060 or gamma ratio>1.44',
            'warning':'Not exhaustive: a true competing alias can be outside nearest k projected points; no recovery guarantee.'}
    return rows, threshold_rows, diag


def run_audit(out: Path, q2_file: Path, pool_size: int, seed: int,
              bins: int, neighbors: int, projection_dim: int, basis_size: int) -> None:
    assert_source_contract()
    q2, q_info = load_historical_q2(q2_file)
    if pool_size < 60 or bins < 2 or bins > 20:
        raise ValueError('pool-size >=60 and bins 2..20 required')
    out = out.resolve()
    if out.exists(): raise FileExistsError(f'Output exists, refusing overwrite: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='._fivep_v2_audit_', dir=out.parent))
    try:
        raw, accepted, p, sampling = generate_candidate_pool(pool_size, seed)
        if not valid_parameter_mask_numpy(p).all():
            raise RuntimeError('Sampled physically invalid five-parameter state')
        np.save(tmp/'candidate_parameters.npy', p)
        np.save(tmp/'raw_candidate_parameters.npy', raw)
        np.save(tmp/'physical_accepted_parameters.npy', accepted)
        g = generate_curves(p, q2)
        if g.shape != (pool_size, len(q2)):
            raise RuntimeError('Wrong physical clean g shape')
        np.save(tmp/'candidate_g_clean.npy', g.astype(np.float32))
        rows, thresholds, diag = audit_nearest(p, g, neighbors, projection_dim, basis_size)
        write_csv(tmp/'nearest_g_neighbors.csv', rows)
        write_csv(tmp/'alias_threshold_counts.csv', thresholds)
        write_csv(tmp/'coverage_candidate.csv',coverage_counts({'candidate':raw,'accepted_physical':accepted}, bins))
        (tmp/'q2.npy').write_bytes(q2_file.resolve(strict=True).read_bytes())
        payload = dict(kind='FIVEP_V2_CANDIDATE_ALIAS_AUDIT_NOT_FROZEN',
                       sample_count=pool_size, names=list(NAMES), parameters={k:list(v) for k,v in BOUNDS.items()},
                       recovery=RECOVERY, reference_noise=REF_NOISE, q2=q_info,
                       q2_saved_sha256=sha_raw(tmp/'q2.npy'),
                       seed=seed, sobol=sampling, nearest_audit=diag,
                       threshold_counts=thresholds,
                       binning='gamma log-space; a1,a2,a3,m linear',
                       hard_filter='physical rho(s)>=0, no train D_g pairwise threshold or alias rejection yet',
                       historical_reference_must_be_validated_against_original_successful_artifact=True)
        (tmp/'audit_report.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
        tmp.rename(out)
        print(f'V2 AUDIT CREATED {out} | valid states={pool_size} | actual q2 points={len(q2)}')
        print(f'q2 file SHA256={q_info["source_file_sha256"]}')
        print('NO global pairwise D>=6 screening; no training or final Test generation.')
        print('Neighbor search is a shortlist, NOT a full proof against off-shortlist aliases.')
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def pick_pilot_parameters(pool: np.ndarray, states: int, seed: int, bins: int) -> np.ndarray:
    if not (60 <= states <= len(pool)):
        raise ValueError('Pilot states must be >=60 and <=candidate states')
    rng = np.random.default_rng(seed)
    bin_ids = bins_for(pool, bins)
    selected = []
    chosen = np.zeros(len(pool), dtype=bool)
    # Include actual parameter endpoints, entirely parameter-only.
    for j in (2, 4, 0, 3, 1):
        for end in (LOW[j], HIGH[j]):
            where = np.flatnonzero((pool[:,j] == end) & ~chosen)
            if len(where):
                ix = int(where[0]); selected.append(ix); chosen[ix] = True
    missing = {(j,k) for j in range(5) for k in range(bins)}
    for ix in selected:
        missing.difference_update((j,int(bin_ids[ix,j])) for j in range(5))
    while missing:
        potential = np.flatnonzero(~chosen)
        if len(potential) == 0 or len(selected) >= states:
            raise RuntimeError('Cannot get marginal pilot coverage; increase states/pool')
        gain = np.zeros(len(potential),dtype=np.int32)
        for j,k in missing:
            gain += (bin_ids[potential,j] == k)
        best = int(potential[np.flatnonzero(gain==np.max(gain))[0]])
        selected.append(best); chosen[best] = True
        missing.difference_update((j,int(bin_ids[best,j])) for j in range(5))
    remaining = rng.permutation(np.flatnonzero(~chosen))
    selected.extend(map(int,remaining[:states-len(selected)]))
    return pool[np.asarray(selected,dtype=int)].copy()


def split_pilot_train_val(p:np.ndarray, seed:int, val_fraction:float=.1) -> np.ndarray:
    rng=np.random.default_rng(seed)
    bins=np.clip((np.log(p[:,4]/LOW[4])/np.log(HIGH[4]/LOW[4])*5).astype(int),0,4)
    split=np.full(len(p),'train',dtype='<U5')
    for k in range(5):
        ix=np.flatnonzero(bins == k)
        if len(ix) < 3: raise ValueError('Too few states in gamma log-bin for pilot val split')
        rng.shuffle(ix)
        n=max(1,int(round(len(ix)*val_fraction)))
        split[ix[:n]]='val'
    return split


def run_pilot(out:Path, audit_dir:Path, q2_file:Path, states:int, seed:int,
              bins:int, replicas:int) -> None:
    assert_source_contract()
    q_h, q_info=load_historical_q2(q2_file)
    with (audit_dir/'audit_report.json').open(encoding='utf-8') as f:
        audit=json.load(f)
    if audit['q2']['source_file_sha256'] != q_info['source_file_sha256']:
        raise ValueError('Historical q2 file SHA differs from the candidate audit: STOP')
    p_pool=np.load(audit_dir/'candidate_parameters.npy',allow_pickle=False)
    raw_pool=np.load(audit_dir/'raw_candidate_parameters.npy',allow_pickle=False)
    accepted_pool=np.load(audit_dir/'physical_accepted_parameters.npy',allow_pickle=False)
    p=pick_pilot_parameters(p_pool, states, seed, bins)
    if not valid_parameter_mask_numpy(p).all():raise RuntimeError('Pilot physical invalidity')
    split=split_pilot_train_val(p, seed)
    pairs=[state_identity(row) for row in p]
    pid=np.asarray([z[0] for z in pairs],dtype='<U35')
    sid=np.asarray([z[1] for z in pairs],dtype=np.int64)
    if len(set(pid)) != states:raise RuntimeError('Duplicate pilot truth')
    q100=np.linspace(-100.,-6.,100,dtype=np.float64)
    designs={'uniform_100':q100,'hybrid_ultranear_240':q_h}
    out=out.resolve()
    if out.exists():raise FileExistsError(f'Output exists: {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='._fivep_v2_pilot_',dir=out.parent))
    try:
        write_manifest(tmp/'state_manifest.csv',p,split,pid,sid)
        write_csv(tmp/'coverage_pilot.csv',coverage_counts({'candidate':raw_pool,
                                      'accepted_physical':accepted_pool,
                                      'train':p[split=='train'],
                                      'val':p[split=='val']},bins))
        for name,q in designs.items():
            dir=tmp/name
            dir.mkdir(parents=True,exist_ok=True)
            if name=='hybrid_ultranear_240':
                (dir/'q2.npy').write_bytes(q2_file.resolve(strict=True).read_bytes())
            else:
                np.save(dir/'q2.npy',q)
            g=generate_curves(p,q)
            for s,offset in (('train',11),('val',23)):
                use=split==s
                write_examples(dir/'noise_0p2pct'/f'{s}.npz',p[use],g[use],pid[use],sid[use],s,q,
                               REF_NOISE,replicas if s=='train' else 1,seed+offset)
            write_normalization(dir)
            meta=dict(kind='FIVEP_V2_PAIRED_PILOT_NO_FINAL_TEST',design_label=name,
                      stage='5P',parameter_names=list(NAMES),
                      parameter_ranges={k:list(v) for k,v in BOUNDS.items()},
                      recovery_tolerances=RECOVERY,reference_noise=REF_NOISE,
                      train_replicas=replicas,states=int(states),
                      split_counts={s:int(np.sum(split==s)) for s in ('train','val')},
                      q2_points=int(len(q)),q2_min=float(q[0]),q2_max=float(q[-1]),
                      q2_file_sha256=sha_raw(dir/'q2.npy'),
                      q2_values_sha256=q2_fingerprint(q),
                      historical_reference=q_info if name=='hybrid_ultranear_240' else None,
                      source_sha256={'mc_physics.py':fingerprint(Path(__file__).resolve().parent/'mc_physics.py')},
                      warning='Pilot 100/239 grid comparison only, no model training and no recovered-parameter PASS')
            (dir/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        overall=dict(kind='FIVEP_V2_PAIRED_PILOT',seed=seed,
                     physical_states=states,distinct_train=int(np.sum(split=='train')),
                     distinct_val=int(np.sum(split=='val')),
                     same_physical_state_truth_and_split_across_designs=True,
                     q2_historical_reference=q_info,
                     q2_uniform_count=100,q2_hybrid_count=int(len(q_h)),
                     candidate_audit_sha256=sha_raw(audit_dir/'audit_report.json'),
                     note='No model training; compare with teammate TCN on Train/Val, NOT final Test')
        (tmp/'pilot_manifest.json').write_text(json.dumps(overall,indent=2,ensure_ascii=False),encoding='utf-8')
        tmp.rename(out)
        print(f'PAIRED PILOT CREATED: {out} | states={states} train={overall["distinct_train"]} val={overall["distinct_val"]}')
        print(f'Same parameter truths, two observation designs: uniform 100 / original hybrid {len(q_h)}')
        print('Pilot has no final Test. Network fitting and Recovery still UNTESTED.')
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True)
        raise


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='mode',required=True)
    audit=sub.add_parser('audit',help='sample states, compute clean g, kNN alias/coverage reports')
    audit.add_argument('--q2-file',required=True,type=Path)
    audit.add_argument('--states',type=int,default=50000)
    audit.add_argument('--seed',type=int,default=20261012)
    audit.add_argument('--bins',type=int,default=10)
    audit.add_argument('--neighbors',type=int,default=48)
    audit.add_argument('--projection-dim',type=int,default=10)
    audit.add_argument('--basis-size',type=int,default=768)
    audit.add_argument('--out',required=True,type=Path)
    pilot=sub.add_parser('pilot',help='make same physical truths at uniform and saved hybrid designs')
    pilot.add_argument('--audit-dir',required=True,type=Path)
    pilot.add_argument('--q2-file',required=True,type=Path)
    pilot.add_argument('--states',type=int,default=2000)
    pilot.add_argument('--seed',type=int,default=20261013)
    pilot.add_argument('--bins',type=int,default=10)
    pilot.add_argument('--train-replicas',type=int,default=3)
    pilot.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    if a.mode=='audit':
        run_audit(a.out,a.q2_file,a.states,a.seed,a.bins,a.neighbors,a.projection_dim,a.basis_size)
    else:
        run_pilot(a.out,a.audit_dir,a.q2_file,a.states,a.seed,a.bins,a.train_replicas)

if __name__=='__main__':
    main()
