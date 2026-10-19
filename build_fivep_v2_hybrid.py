#!/usr/bin/env python3
"""Build the 5P V2 continuous-regression dataset on the EXACT historical hybrid q2 grid.

New additive tool. Does not modify existing V1 files or the phase-1 audit.
Train-only alias-risk screens reject only *disjoint recovery windows* with Dg<6,
not ordinary close physical neighbors. Nearest-neighbor probes are not exhaustive.
Val/Test candidates are independently sampled using parameters/physical legality
only; their g and recovery outcomes cannot influence split selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from build_fivep_dataset import (
    BOUNDS, NAMES, RECOVERY, REF_NOISE, INTEGRATION_POINTS, FORMULA,
    assert_source_contract, generate_curves, write_examples, write_manifest,
    write_normalization, state_identity, fingerprint,
)
from mc_physics import valid_parameter_mask_numpy
from fivep_v2_prepare import (
    LOW, HIGH, bins_for, coverage_counts, load_historical_q2,
    sha_raw, q2_fingerprint, sigma_of,
)

# From the exact historical data_cp02_ml_hybrid_domain_v2/q2.npy supplied by
# the data constructor (SHA-256 of .npy FILE bytes, not just float64 values).
EXPECTED_HISTORICAL_Q2_SHA256 = '2624d6cf24873a0e4c60f7ba19e2239a0a5ec961c04badf8056e1f1568dc653d'
ALIAS_D_THRESHOLD = 6.0
LOCAL_AUDIT_NEIGHBORS = 64
LOCAL_AUDIT_PROJECTION = 16
DEFAULT_BINS = 10
DEFAULT_MIN_BIN_FRACTION = .02


def keys_of(p: np.ndarray) -> list[bytes]:
    return [np.asarray(row, '<f8').tobytes() for row in p]


def read_and_check_audit(audit_dir: Path, q2_sha: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Reuse the physically valid 65k candidate pool, never regenerate a lookalike q2."""
    meta = json.loads((audit_dir/'audit_report.json').read_text(encoding='utf-8'))
    if (meta.get('q2', {}).get('source_file_sha256', '').lower() != q2_sha.lower()
            or meta.get('kind') != 'FIVEP_V2_CANDIDATE_ALIAS_AUDIT_NOT_FROZEN'):
        raise ValueError('Candidate audit is not tied to the exact supplied historical q2.npy')
    if (meta.get('parameters') != {k:list(v) for k,v in BOUNDS.items()}
            or meta.get('recovery') != RECOVERY or meta.get('reference_noise') != REF_NOISE):
        raise ValueError('Candidate audit physics/noise/recovery contract changed')
    if sha_raw(audit_dir/'q2.npy').lower()!=q2_sha.lower():
        raise ValueError('Audit q2.npy differs from frozen historical q2 source')
    p = np.load(audit_dir/'candidate_parameters.npy', mmap_mode='r', allow_pickle=False)
    g = np.load(audit_dir/'candidate_g_clean.npy', mmap_mode='r', allow_pickle=False)
    if not (p.ndim==2 and p.shape[1]==5 and g.shape==(len(p),meta['q2']['point_count'])
            and len(p)==meta['sample_count'] and np.isfinite(p).all() and np.isfinite(g).all()):
        raise ValueError('Candidate pool is missing, inconsistent or nonfinite')
    if not valid_parameter_mask_numpy(p).all():
        raise ValueError('Candidate pool contains physically invalid truths')
    found=np.zeros(len(p),dtype=bool)
    neighbors=np.empty(len(p),dtype=np.int64)
    observed_d=np.empty(len(p),dtype=np.float64)
    report_disjoint=np.zeros(len(p),dtype=bool)
    with (audit_dir/'nearest_g_neighbors.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            i=int(row['candidate_index']); j=int(row['neighbor_index'])
            if i<0 or i>=len(p) or j<0 or j>=len(p) or i==j or found[i]:
                raise ValueError('Duplicated, self, or invalid nearest-g audit indices')
            found[i]=True;neighbors[i]=j
            observed_d[i]=float(row['D_g'])
            report_disjoint[i]=row['recovery_acceptance_windows_disjoint'].lower()=='true'
    if not found.all(): raise ValueError('Incomplete nearest-g audit')
    blocked=np.zeros(len(p),dtype=bool)
    # Independently cross-check audit flags with 5P truths and cached full-q2 g,
    # rather than blindly trusting an editable CSV's "True/False" values.
    sigma=sigma_of(g)
    for start in range(0,len(p),256):
        ix=np.arange(start,min(len(p),start+256))
        jj=neighbors[ix]
        delta=np.abs(np.asarray(p[ix])-np.asarray(p[jj]))
        fac=np.maximum(p[ix,4]/p[jj,4],p[jj,4]/p[ix,4])
        disjoint=(delta[:,0]>2*RECOVERY['a1_abs']) | (delta[:,3]>2*RECOVERY['m_abs']) | (fac>RECOVERY['gamma_factor']**2)
        if not np.array_equal(disjoint,report_disjoint[ix]):
            raise ValueError('Nearest-g CSV Recovery disjoint flag inconsistent with true labels')
        measured=np.linalg.norm(np.asarray(g[ix],np.float64)-np.asarray(g[jj],np.float64),axis=1)/np.maximum(sigma[ix],sigma[jj])
        if not np.all(np.isfinite(measured)) or np.any(np.abs(measured-observed_d[ix])>np.maximum(.02,.003*measured)):
            raise ValueError('Nearest-g CSV distances inconsistent with clean observations')
        blocked[ix]=disjoint & (measured<ALIAS_D_THRESHOLD)
    # Independently recompute some saved physics to detect accidental cache drift.
    probe=np.unique(np.linspace(0,len(p)-1,min(len(p),8)).astype(int))
    if not np.allclose(generate_curves(np.asarray(p[probe],np.float64),
                       np.load(audit_dir/'q2.npy',allow_pickle=False)),
                       np.asarray(g[probe],np.float64),rtol=3e-5,atol=2e-7):
        raise ValueError('Cached candidate g does not match actual five-parameter physics')
    return p,g,blocked,meta



def extend_missing_train_endpoints(pool: np.ndarray, g: np.ndarray,
                                   blocked: np.ndarray, q2: np.ndarray,
                                   seed: int, count_per_endpoint: int = 3):
    """Add physically valid, train-only endpoint alternatives when the phase-1
    candidate's one canonical endpoint anchor had a strong disjoint alias.

    Unlike overriding a hard alias rejection, replacement anchors undergo
    their own kNN+exact-full-q2 g-risk screening against the whole candidate
    TRAIN pool. No Val/Test g is involved, and no risk threshold is relaxed.
    """
    pool=np.asarray(pool,dtype=np.float64)
    base_g=np.asarray(g,dtype=np.float32)
    if len(pool)!=len(blocked) or base_g.shape!=(len(pool),len(q2)):
        raise ValueError('Wrong candidates for endpoint augmentation')
    missing=[]
    for j in (2,4,0,3,1):
        for end in (LOW[j],HIGH[j]):
            if not np.any((pool[:,j]==end) & ~blocked):
                missing.append((j,float(end)))
    if not missing:
        return pool,base_g,blocked.copy(),{'extra_endpoint_states':0,'missing_endpoints_repaired':[]}
    rng=np.random.default_rng(seed+778123)
    # As in the original finite audit, candidate retrieval is projected;
    # the actual g distance and parameter-window test are performed at FULL q2.
    n=len(pool)
    basis_idx=np.unique(np.linspace(0,n-1,min(n,768)).astype(int))
    center=np.mean(np.asarray(base_g[basis_idx],np.float64),axis=0)
    _,_,vt=np.linalg.svd(np.asarray(base_g[basis_idx],np.float64)-center,full_matrices=False)
    basis=vt[:min(16,len(vt))].T
    coords=(np.asarray(base_g,np.float64)-center)@basis
    tree=cKDTree(coords)
    sig=sigma_of(base_g)
    known=set(keys_of(pool))
    extra_p=[];extra_g=[];log=[]
    for j,endpoint in missing:
        accepted=0
        candidates=rng.permutation(n)
        # Ensure we can find multiple DIFFERENT physically valid variants.
        for start in range(0,len(candidates),256):
            batch=pool[candidates[start:start+256]].copy()
            batch[:,j]=endpoint
            good=valid_parameter_mask_numpy(batch)
            if not np.any(good):continue
            props=batch[good]
            curves=generate_curves(props,q2)
            z=(curves-center)@basis
            neighbor_idx=tree.query(z,k=min(n,96),workers=1)[1]
            if neighbor_idx.ndim==1:neighbor_idx=neighbor_idx[:,None]
            for proposal,curve,near in zip(props,curves,neighbor_idx):
                key=proposal.astype('<f8').tobytes()
                if key in known: continue
                competitor=pool[near]
                delta=np.abs(competitor-proposal)
                factor=np.maximum(competitor[:,4]/proposal[4],proposal[4]/competitor[:,4])
                window_disjoint=(delta[:,0]>2*RECOVERY['a1_abs']) | (delta[:,3]>2*RECOVERY['m_abs']) | (factor>RECOVERY['gamma_factor']**2)
                sd=REF_NOISE*np.sqrt(np.mean(curve*curve))
                distance=np.linalg.norm(np.asarray(base_g[near],np.float64)-curve,axis=1)/np.maximum(sd,sig[near])
                if np.any((distance<ALIAS_D_THRESHOLD) & window_disjoint):
                    continue
                known.add(key)
                extra_p.append(proposal.copy());extra_g.append(curve.astype(np.float32))
                accepted+=1
                if accepted>=count_per_endpoint:break
            if accepted>=count_per_endpoint:break
        if accepted<count_per_endpoint:
            raise RuntimeError(f'No alias-screened alternate Train endpoint {NAMES[j]}={endpoint}; '
                               'expand audited candidate pool, do NOT waive alias rule or bounds')
        log.append({'parameter':NAMES[j],'value':endpoint,'extra_valid_states':accepted})
    enlarged_p=np.vstack([pool,np.asarray(extra_p,dtype=np.float64)])
    enlarged_g=np.vstack([base_g,np.asarray(extra_g,dtype=np.float32)])
    enlarged_bad=np.concatenate([blocked,np.zeros(len(extra_p),dtype=bool)])
    return enlarged_p,enlarged_g,enlarged_bad,{
        'extra_endpoint_states':len(extra_p),'missing_endpoints_repaired':log,
        'reference_pool_size':n,'screening_method':'projected 96-neighbor shortlist, full-q2 D and disjoint-window filter',
        'screening_is_exhaustive':False,
    }


def select_parameter_coverage(pool: np.ndarray, count: int, bins: int, min_fraction: float,
                              seed: int, eligible: np.ndarray,
                              require_endpoints: bool=True) -> np.ndarray:
    """Train or held-out selection, based ONLY on parameter bins and eligibility.

    'eligible' may include pre-frozen TRAIN alias masks; held-out eligibility must
    include ONLY physical validity and noncollision with already frozen states.
    """
    pool=np.asarray(pool,np.float64)
    if not (0 < min_fraction <= 1/bins and count>=5*bins and len(pool)==len(eligible)):
        raise ValueError('Invalid coverage quota/selection size')
    cand=np.flatnonzero(eligible)
    if len(cand)<count: raise RuntimeError('Not enough eligible distinct physical candidates')
    codes=bins_for(pool,bins)
    rng=np.random.default_rng(seed)
    used=np.zeros(len(pool),dtype=bool)
    picked:list[int]=[]
    totals=np.zeros((5,bins),dtype=np.int64)

    def insert(index:int) -> None:
        if used[index]: return
        if len(picked)>=count: raise RuntimeError('Coverage requires more states; enlarge target/candidate pool')
        used[index]=True; picked.append(index)
        totals[np.arange(5),codes[index]]+=1

    if require_endpoints:
        for j in (2,4,0,3,1):
            for endpoint in (LOW[j],HIGH[j]):
                found=cand[pool[cand,j]==endpoint]
                if not len(found):
                    raise RuntimeError(f'Cannot preserve exact {NAMES[j]} endpoint {endpoint}; expand candidate pool')
                insert(int(found[0]))
    quota=max(1,int(math.ceil(count*min_fraction)))
    # Rare bins first; filling one bin also fills four others.
    availability=np.array([[np.sum(codes[cand,j]==k) for k in range(bins)] for j in range(5)])
    if np.any(availability<quota):
        miss=np.argwhere(availability<quota)[0]
        raise RuntimeError(f'Insufficient physically valid states in {NAMES[miss[0]]} bin {miss[1]}: '
                           f'{availability[tuple(miss)]} available, {quota} required')
    order=sorted(((j,k) for j in range(5) for k in range(bins)),
                 key=lambda jk:(availability[jk],jk))
    for j,k in order:
        need=max(0,quota-int(totals[j,k]))
        if need==0: continue
        choices=cand[(codes[cand,j]==k) & ~used[cand]]
        if len(choices)<need:
            raise RuntimeError('Not enough unused candidates for fixed coverage quota')
        take=rng.choice(choices,size=need,replace=False)
        for ix in take: insert(int(ix))
    if np.any(totals<quota):
        raise RuntimeError('Marginal coverage floor not reached')
    if len(picked)<count:
        remaining=cand[~used[cand]]
        take=rng.choice(remaining,size=count-len(picked),replace=False)
        for ix in take: insert(int(ix))
    result=np.asarray(picked,dtype=np.int64)
    # Randomize order so nested 1k/2k/... prefixes are not quota-first biased.
    result=result[rng.permutation(len(result))]
    if require_endpoints:
        for j in range(5):
            if not (np.any(pool[result,j]==LOW[j]) and np.any(pool[result,j]==HIGH[j])):
                raise RuntimeError('Missing exact parameter endpoint')
    return result


def independent_holdout_candidates(count: int, seed: int, previously_used: set[bytes],
                                   bins: int, min_fraction: float) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict]:
    """Independent Sobol parameter-only sampling. No g evaluation or label recovery.

    For true endpoint coverage, mutate other independently sampled valid states
    and reapply rho(s)>=0. Never reuse training's canonical endpoint anchors.
    """
    if count<5*bins: raise ValueError('Too few held-out states for requested coverage')
    rng=np.random.default_rng(seed+1337)
    all_raw=[]; all_phys=[]; extras=[]
    valid_keys=set(previously_used)
    for round_ in range(8):
        block_len=1<<int(math.ceil(math.log2(max(1024, 8*count))))
        u=qmc.Sobol(d=5,scramble=True,seed=seed+7919*round_).random_base2(int(math.log2(block_len)))
        raw=LOW+u*(HIGH-LOW)
        raw[:,4]=np.exp(np.log(LOW[4])+u[:,4]*np.log(HIGH[4]/LOW[4]))
        ok=valid_parameter_mask_numpy(raw)
        all_raw.append(raw)
        all_phys.append(raw[ok])
        # Independent endpoint variants: vary all remaining coordinates,
        # enforce physical validity after fixing the one desired coordinate.
        if round_==0:
            for j in (2,4,0,3,1):
                for edge in (LOW[j],HIGH[j]):
                    mod=raw.copy()
                    mod[:,j]=edge
                    good=valid_parameter_mask_numpy(mod)
                    cand=mod[good]
                    rng.shuffle(cand)
                    for row in cand:
                        key=row.astype('<f8').tobytes()
                        if key not in valid_keys:
                            extras.append(row.copy());valid_keys.add(key);break
                    else:
                        raise RuntimeError(f'Cannot sample valid independent endpoint {NAMES[j]}={edge}')
        phys=np.vstack(all_phys)
        raw_all=np.vstack(all_raw)
        # Force anchors into pool; skip all previously frozen state truths.
        unique={}
        for row in [*extras,*phys]:
            key=np.asarray(row,'<f8').tobytes()
            if key not in previously_used and key not in unique:
                unique[key]=row
        pool=np.stack(list(unique.values()))
        if len(pool)>count*3:
            try:
                indices=select_parameter_coverage(pool,count,bins,min_fraction,seed+901, np.ones(len(pool),bool))
            except RuntimeError as e:
                if round_==7: raise
                continue
            return pool[indices],raw_all,phys,dict(seed=seed,raw_candidates=len(raw_all),
                    physical_candidates=len(phys),endpoint_variants=len(extras),
                    chosen_unique=count,selection_uses_g=False)
    raise RuntimeError('Cannot cover independent held-out parameter bins; increase Sobol pool, not scientific tolerances')


def selected_train_conflicts(p: np.ndarray, g: np.ndarray, n_neighbors: int = LOCAL_AUDIT_NEIGHBORS,
                             project: int = LOCAL_AUDIT_PROJECTION) -> tuple[set[int],dict]:
    """Probe candidate pairs in selected Train; return later endpoint of each risky pair.

    Shortlist based on SVD projection, EXACT g separation recomputed for all
    shortlisted candidates. Only strong disjoint Recovery windows cause pruning.
    """
    n=len(p)
    if n<3: return set(), {'pairs_probed':0,'disjoint_under_6':0,'normal_close_under_6':0}
    sig=sigma_of(g)
    reference=np.unique(np.linspace(0,n-1,min(n,768)).astype(int))
    center=np.mean(g[reference],axis=0)
    _,_,vt=np.linalg.svd(g[reference]-center,full_matrices=False)
    basis=vt[:min(project,len(vt))].T
    coords=(g-center)@basis
    neighbors=cKDTree(coords).query(coords,k=min(n,n_neighbors+1),workers=1)[1]
    if neighbors.ndim==1: neighbors=neighbors[:,None]
    bad:set[int]=set(); numrisk=0; numlocal=0; numprobed=0; min_d=math.inf
    for begin in range(0,n,32):
        end=min(n,begin+32)
        ii=np.arange(begin,end)
        jj=np.asarray(neighbors[begin:end],dtype=np.int64)
        distance=np.linalg.norm(g[ii[:,None]]-g[jj],axis=2)/np.maximum(sig[ii,None],sig[jj])
        notself=ii[:,None] != jj
        distance[~notself]=np.inf
        min_d=min(min_d,float(np.min(distance)))
        delta=np.abs(p[ii[:,None]]-p[jj])
        fac=np.maximum(p[ii,None,4]/p[jj,4],p[jj,4]/p[ii,None,4])
        disjoint=(delta[...,0]>2*RECOVERY['a1_abs']) | (delta[...,3]>2*RECOVERY['m_abs']) | (fac>RECOVERY['gamma_factor']**2)
        close=(distance<ALIAS_D_THRESHOLD)
        numprobed+=int(np.count_nonzero(notself))
        risks=close & disjoint
        numrisk+=int(np.count_nonzero(risks))
        numlocal+=int(np.count_nonzero(close & ~disjoint))
        for local_i, js in enumerate(jj):
            for j in js[risks[local_i]]:
                # Remove the larger index; protect early anchors where possible.
                bad.add(max(int(ii[local_i]),int(j)))
    return bad, {'pairs_probed_directed':numprobed,'disjoint_under_6_directed':numrisk,
                 'close_compatible_under_6_directed':numlocal,
                 'minimum_probed_train_D_g':min_d,
                 'search_exhaustive':False,'neighbors':n_neighbors,'projection_dim':project}


def build(out:Path, audit_dir:Path, q2_file:Path,
          train_states:int=9600,val_states:int=1200,test_states:int=1200,
          seed:int=20261019, bins:int=DEFAULT_BINS,
          min_bin_fraction:float=DEFAULT_MIN_BIN_FRACTION,
          train_replicas:int=3, expected_q2_sha:str=EXPECTED_HISTORICAL_Q2_SHA256,
          local_probe:bool=True) -> dict:
    assert_source_contract()
    if min(train_states,val_states,test_states)<5*bins:
        raise ValueError('Each split needs at least 5*bins independent states')
    if not (bins==10 and min_bin_fraction==.02 and train_replicas==3):
        raise ValueError('Frozen V2 coverage is 10 bins, 2% per marginal bin, 3 Train noise replicas')
    q, qinfo=load_historical_q2(q2_file)
    if qinfo['source_file_sha256'].lower()!=expected_q2_sha.lower():
        raise ValueError('Historical hybrid q2 FILE SHA-256 mismatch: refuse substituted/recreated grid')
    if not (len(q)==239 and float(q[0])==-150. and float(q[-1])==-.05):
        raise ValueError('Historical hybrid observation grid must use saved 239 values in [-150,-.05]')
    pool, cache_g, prescreen_bad, audit=read_and_check_audit(audit_dir,qinfo['source_file_sha256'])
    if cache_g.shape[1]!=len(q): raise ValueError('Candidate clean g length differs from historical hybrid')
    if len(pool)<train_states:
        raise RuntimeError('Candidate audit has fewer states than target Train; enlarge candidate pool')
    if len(set(keys_of(pool)))!=len(pool):
        raise ValueError('Duplicate physical truth in candidate audit')
    original_candidate_count=len(pool)
    original_prescreen_bad_count=int(prescreen_bad.sum())
    pool,cache_g,prescreen_bad,endpoint_repair=extend_missing_train_endpoints(
        pool,cache_g,prescreen_bad,q,seed)
    out=out.resolve()
    if out.exists(): raise FileExistsError(f'Output exists; no overwrite: {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='._fivep_v2_formal_',dir=out.parent))
    try:
        blacklist=prescreen_bad.copy()
        review=[]
        # Train must freeze BEFORE Val/Test candidate generation.
        for iteration in range(6):
            train_idx=select_parameter_coverage(pool,train_states,bins,min_bin_fraction,
                     seed+iteration,np.asarray(~blacklist),True)
            p_train=np.asarray(pool[train_idx],np.float64)
            g_cached=np.asarray(cache_g[train_idx],np.float64)
            if local_probe:
                bad,diag=selected_train_conflicts(p_train,g_cached)
            else:
                bad,diag=set(),{'skipped':True,'search_exhaustive':False}
            review.append(dict(iteration=iteration, detected_selected_conflict_indices=len(bad),
                               diagnostic=diag))
            if not bad: break
            blacklist[train_idx[np.asarray(sorted(bad),dtype=int)]]=True
        else:
            raise RuntimeError('Repeated local train alias probes still find strongly conflicting neighbors; expand pool/audit')
        # Freeze Train state identities now; no holdout g has been evaluated.
        used=set(keys_of(p_train))
        p_val,raw_val,phys_val,val_log=independent_holdout_candidates(val_states,seed+100003,used,bins,min_bin_fraction)
        used.update(keys_of(p_val))
        p_test,raw_test,phys_test,test_log=independent_holdout_candidates(test_states,seed+200003,used,bins,min_bin_fraction)
        all_p=np.vstack([p_train,p_val,p_test])
        labels=np.concatenate([np.full(train_states,'train',dtype='<U5'),
                               np.full(val_states,'val',dtype='<U5'),
                               np.full(test_states,'test',dtype='<U5')])
        if not (valid_parameter_mask_numpy(all_p).all() and len(set(keys_of(all_p)))==len(all_p)):
            raise RuntimeError('Physical bounds or state independence compromised')
        identity=[state_identity(row) for row in all_p]
        pid=np.asarray([i[0] for i in identity],dtype='<U35')
        sid=np.asarray([i[1] for i in identity],dtype=np.int64)
        if len(set(pid))!=len(pid) or len(set(map(int,sid)))!=len(sid):
            raise RuntimeError('State-ID hash collision')
        write_manifest(tmp/'state_manifest.csv',all_p,labels,pid,sid)
        # Preserve actual original NPY bytes, not a recomputed near-identical grid.
        (tmp/'q2.npy').write_bytes(q2_file.resolve(strict=True).read_bytes())
        coverage=coverage_counts({'candidate':np.load(audit_dir/'raw_candidate_parameters.npy',mmap_mode='r',allow_pickle=False),
                     'accepted_physical':np.load(audit_dir/'physical_accepted_parameters.npy',mmap_mode='r',allow_pickle=False),
                     'candidate_val':raw_val,'accepted_val':phys_val,
                     'candidate_test':raw_test,'accepted_test':phys_test,
                     'train':p_train,'val':p_val,'test':p_test},bins)
        from fivep_v2_prepare import write_csv
        write_csv(tmp/'coverage_v2.csv',coverage)
        # Genuine forward physics AFTER split freeze. These are exact float64 results
        # with the original integration and the historical fixed q2 grid.
        for split,truth,offset in [('train',p_train,11),('val',p_val,23),('test',p_test,37)]:
            clean=generate_curves(truth,q)
            if not np.isfinite(clean).all(): raise RuntimeError(f'Nonfinite {split} forward observation')
            use=np.flatnonzero(labels==split)
            write_examples(tmp/'noise_0p2pct'/f'{split}.npz',truth,clean,pid[use],sid[use],
                           split,q,REF_NOISE,train_replicas if split=='train' else 1,seed+offset)
            if split=='test':
                for lev,dirname,add in [(0.,'noise_0pct',41),(.01,'noise_1pct',43)]:
                    write_examples(tmp/dirname/'test.npz',truth,clean,pid[use],sid[use],
                                   'test',q,lev,1,seed+add)
        write_normalization(tmp)
        nested={}
        for size in [1000,2000,5000,9600,15000,20000]:
            if size<=train_states:
                # IDs are unique & train indices are randomized AFTER the coverage floor.
                np.save(tmp/f'train_state_indices_{size}.npy',np.arange(size,dtype=np.int32))
                nested[str(size)]={'state_indices_file':f'train_state_indices_{size}.npy',
                                   'noise_row_expansion':'repeat each physical-state index 3 times'}
        payload=dict(kind='FIVEP_V2_HYBRID_FORMAL_DATASET',stage='5P',names=list(NAMES),
            parameter_ranges={k:list(v) for k,v in BOUNDS.items()},recovery_tolerances=RECOVERY,
            reference_noise=REF_NOISE,noise_rule='sigma=0.002*RMS(g_clean); independent N(0,1) per noise replica',
            q2_design='hybrid_ultranear_240 (historical file actual 239 distinct observations)',
            q2_points=len(q),q2_domain=[float(q[0]),float(q[-1])],
            q2_source_file_sha256=qinfo['source_file_sha256'],q2_values_sha256=q2_fingerprint(q),
            q2_locked_to_historical=True,physics_integration_points=INTEGRATION_POINTS,
            source_sha256={p.name:fingerprint(Path(__file__).parent/p.name) for p in
                           [Path('mc_physics.py'),Path('mc_pool_config.py')]},
            split_counts={'train':train_states,'val':val_states,'test':test_states},
            sample_counts={'train':train_states*train_replicas,'val':val_states,'test':test_states},
            train_replicas=train_replicas,seed=seed,
            physical_screen='rho(s)>=0 continuously on s in [.1764,6]; original bounds',
            alias_screening={'train_candidate_pool_size':original_candidate_count,'train_prescreen_disjoint_D_below_6':original_prescreen_bad_count,
                             'train_endpoint_repair':endpoint_repair,
                             'rejected_total_after_selected_probe':int(blacklist.sum()),
                             'only_disjoint_recovery_windows_under_D6_removed':True,
                             'global_pairwise_D6_required':False,
                             'nearest_probe':'SVD shortlist 64, exact full-q2 distance, not exhaustive',
                             'local_train_probe_iterations':review,
                             'candidate_audit_sha256':sha_raw(audit_dir/'audit_report.json'),
                             'probe_does_not_guarantee_continuous_space_identifiability':True},
            coverage={'bin_count':bins,'minimum_fraction_per_bin':min_bin_fraction,
                      'gamma_log_bins':True,'a1_a2_a3_m_linear_bins':True},
            blind_split={'selection_order':['train','val','test'],
                         'train_selected_before_holdouts_exist':True,
                         'holdout_selection_uses_g':False,
                         'holdout_candidate_seeds':{'val':val_log['seed'],'test':test_log['seed']},
                         'no_state_overlap':True},
            holdout_sampling={'val':val_log,'test':test_log},
            nested_train_subsets=nested,
            scientific_status='PHYSICS_AND_SPLIT_INTEGRITY_ONLY__RECOVERY_NOT_CERTIFIED',
            caveat='Train-only nearest-neighbor alias screen is nonexhaustive; near-alias can exist outside shortlist.')
        (tmp/'metadata.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
        tmp.rename(out)
        print(f'FORMAL 5P V2 CREATED: {out} | states Train={train_states} Val={val_states} Test={test_states}')
        print(f'TRAIN SAMPLES={train_states*train_replicas} | historical q2={len(q)} points, SHA256={qinfo["source_file_sha256"]}')
        print(f'Alias train candidate screen: {original_prescreen_bad_count} strong conflict flags; '
              f'final excluded {int(blacklist.sum())}; NOT all-close-neighbors screening')
        print('BLIND: train fixed before independently sampling Val/Test. No Test-g screening; no network training.')
        print('NOT RECOVERY CERTIFIED.')
        return payload
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True)
        raise


def main() -> None:
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--audit-dir',type=Path,required=True,
                   help='v2_candidate_audit_65536 produced with the actual historical q2.npy')
    a.add_argument('--q2-file',type=Path,required=True,
                   help='EXACT historical data_cp02_ml_hybrid_domain_v2/q2.npy')
    a.add_argument('--out',type=Path,required=True)
    a.add_argument('--train-states',type=int,default=9600)
    a.add_argument('--val-states',type=int,default=1200)
    a.add_argument('--test-states',type=int,default=1200)
    a.add_argument('--seed',type=int,default=20261019)
    args=a.parse_args()
    build(args.out,args.audit_dir,args.q2_file,args.train_states,args.val_states,args.test_states,seed=args.seed)


if __name__=='__main__':main()
