#!/usr/bin/env python3
"""Read-only V2 historical-hybrid physics, grouped-split, coverage and noise verifier.

Does NOT fit any model and does not imply numerical inverse Recovery success.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from build_fivep_dataset import (
    BOUNDS, NAMES, RECOVERY, REF_NOISE, state_identity, assert_source_contract,
    fingerprint,
)
from mc_physics import valid_parameter_mask_numpy, scaled_forward_observation_at_q2_numpy
from fivep_v2_prepare import bins_for, load_historical_q2, sha_raw, q2_fingerprint
from build_fivep_v2_hybrid import (
    EXPECTED_HISTORICAL_Q2_SHA256, DEFAULT_BINS, DEFAULT_MIN_BIN_FRACTION,
    ALIAS_D_THRESHOLD,
)


def read_manifest(path: Path) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    with path.open(encoding='utf-8',newline='') as fp:
        rows=list(csv.DictReader(fp))
    if not rows: raise ValueError('Empty physical-state manifest')
    p=np.asarray([[float(row[k]) for k in NAMES] for row in rows],dtype=np.float64)
    split=np.asarray([r['split'] for r in rows],dtype='<U5')
    ids=np.asarray([r['physical_state_id'] for r in rows],dtype='<U35')
    sids=np.asarray([int(r['state_id']) for r in rows],dtype=np.int64)
    return p,split,ids,sids


def validate(out:Path, q2_file:Path, expected_q2_sha:str=EXPECTED_HISTORICAL_Q2_SHA256,
             forward_count:int=12) -> dict:
    assert_source_contract()
    if forward_count<0: raise ValueError('forward_count must be nonnegative')
    meta=json.loads((out/'metadata.json').read_text(encoding='utf-8'))
    if meta.get('kind')!='FIVEP_V2_HYBRID_FORMAL_DATASET':
        raise ValueError('Not the formal V2 hybrid dataset')
    if (meta.get('parameter_ranges')!={k:list(v) for k,v in BOUNDS.items()}
            or meta.get('recovery_tolerances')!=RECOVERY or meta.get('reference_noise')!=REF_NOISE):
        raise ValueError('Original parameter bounds/noise/Recovery policy changed')
    if (meta.get('coverage',{}).get('bin_count')!=10 or
            meta['coverage'].get('minimum_fraction_per_bin') != DEFAULT_MIN_BIN_FRACTION or
            meta.get('train_replicas')!=3):
        raise ValueError('Train replicas or coverage policy changed')
    qq,info=load_historical_q2(q2_file)
    q_saved=np.load(out/'q2.npy',allow_pickle=False)
    if (info['source_file_sha256'].lower()!=expected_q2_sha.lower()
            or sha_raw(out/'q2.npy').lower()!=expected_q2_sha.lower()
            or not np.array_equal(q_saved,qq)
            or meta['q2_source_file_sha256'].lower()!=expected_q2_sha.lower()
            or meta['q2_values_sha256'] != q2_fingerprint(qq)
            or meta['q2_points']!=239 or meta['q2_domain'] != [-150.,-.05]):
        raise ValueError('V2 historical q2 file/hash/point-count/origin changed')
    source=Path(__file__).resolve().parent
    for code,original in meta['source_sha256'].items():
        if fingerprint(source/code)!=original:
            raise ValueError(f'Original physics dependency modified after construction: {code}')
    p,split,ids,sids=read_manifest(out/'state_manifest.csv')
    if len(p)!=sum(meta['split_counts'].values()):
        raise ValueError('Wrong physical-state count')
    if len(set(ids))!=len(ids) or len(set(map(int,sids)))!=len(sids):
        raise ValueError('Duplicated physical_state_id across Train/Val/Test')
    if not np.isfinite(p).all() or not valid_parameter_mask_numpy(p).all():
        raise ValueError('Invalid or nonfinite physical truth')
    for i,row in enumerate(p):
        if state_identity(row)!=(str(ids[i]),int(sids[i])):
            raise ValueError('State hash/manifest truth mismatch')
    if set(split)!={'train','val','test'}:
        raise ValueError('Missing Train/Val/Test split')
    for s in ('train','val','test'):
        if int(np.sum(split==s))!=meta['split_counts'][s]:
            raise ValueError(f'Wrong {s} physical state count')
        part=p[split==s]
        b=bins_for(part,10)
        min_count=int(np.ceil(len(part)*DEFAULT_MIN_BIN_FRACTION))
        for j in range(5):
            counts=np.bincount(b[:,j],minlength=10)
            if int(counts.min()) < min_count:
                raise ValueError(f'{s}: marginal coverage below 2% in {NAMES[j]}')
            if not (np.any(part[:,j]==BOUNDS[NAMES[j]][0]) and
                    np.any(part[:,j]==BOUNDS[NAMES[j]][1])):
                raise ValueError(f'{s}: missing exact {NAMES[j]} endpoint')
    # Check coverage report produced from the same split truths.
    with (out/'coverage_v2.csv').open(encoding='utf-8-sig',newline='') as fp:
        cov=list(csv.DictReader(fp))
    for s in ('train','val','test'):
        b=bins_for(p[split==s],10)
        for j,param in enumerate(NAMES):
            for k in range(10):
                r=[item for item in cov if item['group']==s and item['parameter']==param and int(item['bin'])==k]
                if len(r)!=1 or int(r[0]['count'])!=int(np.sum(b[:,j]==k)):
                    raise ValueError(f'Coverage CSV/physical truth mismatch {s} {param} {k}')
    for split_name in ('train','val','test'):
        use=split==split_name
        truth=p[use]
        clean_shape=(len(truth)* (3 if split_name=='train' else 1), len(qq))
        with np.load(out/'noise_0p2pct'/f'{split_name}.npz',allow_pickle=False) as d:
            x=np.asarray(d['gy'],dtype=np.float64)
            clean=np.asarray(d['gy_clean'],dtype=np.float64)
            lbl=np.asarray(d['parameters'],dtype=np.float64)
            record_id=d['physical_state_id'].astype(str)
            state_number=np.asarray(d['state_id'],dtype=np.int64)
            reps=np.asarray(d['replica_id'],dtype=np.int32)
            sigma=np.asarray(d['noise_sigma'],dtype=np.float64)
            level=np.asarray(d['noise_level'],dtype=np.float64)
            if (x.shape!=clean_shape or clean.shape!=clean_shape or lbl.shape!=(len(x),5)
                    or not np.all(np.isfinite(x)) or not np.all(np.isfinite(clean))
                    or not np.all(np.isfinite(lbl))):
                raise ValueError(f'Invalid {split_name} arrays or shape')
            repeats=3 if split_name=='train' else 1
            if (not np.array_equal(record_id,np.repeat(ids[use],repeats))
                    or not np.array_equal(state_number,np.repeat(sids[use],repeats))
                    or not np.array_equal(reps,np.tile(np.arange(repeats),len(truth)))
                    or not np.array_equal(lbl,np.repeat(truth.astype(np.float32).astype(np.float64),repeats,axis=0))
                    or not np.array_equal(d['q2'],qq)):
                raise ValueError(f'State, truth, replica or q2 leakage in {split_name}')
            if not np.allclose(level,REF_NOISE,rtol=5e-7,atol=1e-10):
                raise ValueError('Noise-level header changed')
            sigma_expected=REF_NOISE*np.sqrt(np.mean(clean*clean,axis=1))
            if not np.allclose(sigma,sigma_expected,rtol=2e-6,atol=1e-10):
                raise ValueError('Noise sigma incorrect for 0.2% RMS')
            if split_name=='train':
                with np.load(out/'normalization_train_only.npz',allow_pickle=False) as norm:
                    if (not np.allclose(norm['input_mean'],x.mean(axis=0),rtol=1e-6,atol=1e-6)
                            or not np.allclose(norm['input_std'],x.std(axis=0),rtol=1e-6,atol=1e-6)):
                        raise ValueError('Input normalization was not computed from Train only')
                    target=np.column_stack([lbl[:,:4],np.log(lbl[:,4])])
                    if (not np.allclose(norm['target_mean'],target.mean(axis=0),rtol=1e-6,atol=1e-6)
                            or not np.allclose(norm['target_std'],target.std(axis=0),rtol=1e-6,atol=1e-6)):
                        raise ValueError('Target normalization was not computed from Train only')
            if forward_count:
                ix=np.linspace(0,len(truth)-1,min(len(truth),forward_count),dtype=int)
                physical=scaled_forward_observation_at_q2_numpy(truth[ix],qq,
                                             integration_points=512,output_dtype=np.float64)
                saved=clean[ix*repeats]
                if not np.allclose(physical,saved,rtol=1e-4,atol=2e-7):
                    raise ValueError(f'Independent high-order physical integral failed for {split_name}')
    # Verify same test truths across noise variants, no separate test states.
    for dirname in ('noise_0pct','noise_1pct'):
        file=out/dirname/'test.npz'
        if not file.exists(): raise ValueError(f'Missing {dirname} independent noise test')
        with np.load(file,allow_pickle=False) as z:
            if not (np.array_equal(z['physical_state_id'].astype(str),ids[split=='test'])
                    and np.array_equal(z['q2'],qq)):
                raise ValueError('Noise test variants changed held-out Test truths')
    if meta['alias_screening']['global_pairwise_D6_required'] is not False:
        raise ValueError('Global pairwise D>=6 screening was silently reintroduced')
    if (meta['blind_split']['holdout_selection_uses_g'] is not False or
            not meta['blind_split']['train_selected_before_holdouts_exist']):
        raise ValueError('Blind split protocol corrupted')
    for n, rec in meta.get('nested_train_subsets', {}).items():
        indices=np.load(out/rec['state_indices_file'],allow_pickle=False)
        if len(indices)!=int(n) or len(np.unique(indices))!=len(indices) or np.any(indices<0) or np.any(indices>=meta['split_counts']['train']):
            raise ValueError('Invalid nested train state indices')
    print('V2 HYBRID DATA INTEGRITY PASS | historical q2=239; exact original SHA-256 matched')
    print('Split states:',meta['split_counts'],'| training samples:',meta['sample_counts']['train'])
    print('COVERAGE/BLIND SPLIT/PHYSICS/0.2% RMS/RECOVERY CONTRACT PASS')
    print('NOT RECOVERY CERTIFIED: training and held-out parameter prediction are separate.')
    return meta


def main() -> None:
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data',type=Path,required=True)
    a.add_argument('--q2-file',type=Path,required=True)
    a.add_argument('--forward-count',type=int,default=12)
    opt=a.parse_args()
    validate(opt.data,opt.q2_file,forward_count=opt.forward_count)

if __name__=='__main__':main()
