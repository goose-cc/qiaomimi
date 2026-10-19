#!/usr/bin/env python3
"""Read-only validation of paired V2 Train/Val pilot datasets; no network training."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from mc_physics import valid_parameter_mask_numpy, scaled_forward_observation_at_q2_numpy
from build_fivep_dataset import BOUNDS, NAMES, RECOVERY, REF_NOISE
from fivep_v2_prepare import load_historical_q2, q2_fingerprint, sha_raw


def validate(root:Path, reference_q2:Path, forward_count:int=12) -> dict:
    root=root.resolve(strict=True)
    manifest=json.loads((root/'pilot_manifest.json').read_text(encoding='utf-8'))
    if manifest['kind'] != 'FIVEP_V2_PAIRED_PILOT':
        raise ValueError('Not a V2 paired pilot')
    qref, qmeta=load_historical_q2(reference_q2)
    if manifest['q2_historical_reference']['source_file_sha256'] != qmeta['source_file_sha256']:
        raise ValueError('Historical q2 hash differs: STOP')
    if not (0 <= forward_count <= 100000):raise ValueError('Bad forward-count')
    with (root/'state_manifest.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    p=np.asarray([[float(row[k]) for k in NAMES] for row in rows],dtype=np.float64)
    split=np.asarray([row['split'] for row in rows])
    ids=np.asarray([row['physical_state_id'] for row in rows])
    if len(p)!=manifest['physical_states'] or not valid_parameter_mask_numpy(p).all():
        raise ValueError('Manifest size/physical validity failure')
    for j,k in enumerate(NAMES):
        lo,hi=BOUNDS[k]
        if np.any(p[:,j] < lo-1e-12) or np.any(p[:,j]>hi+1e-12):
            raise ValueError('Physical parameter ranges changed')
    if set(split) != {'train','val'} or len(set(ids)) != len(p):
        raise ValueError('Manifest split/state ID duplication')
    if np.sum(split=='train') != manifest['distinct_train'] or np.sum(split=='val') != manifest['distinct_val']:
        raise ValueError('Wrong pilot split sizes')
    descriptions={}
    for name,length in (('uniform_100',100),('hybrid_ultranear_240',len(qref))):
        folder=root/name
        meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        if (meta['kind']!='FIVEP_V2_PAIRED_PILOT_NO_FINAL_TEST' or
            meta['parameter_ranges']!={k:list(v) for k,v in BOUNDS.items()} or
            meta['recovery_tolerances'] != RECOVERY or
            meta['reference_noise'] != REF_NOISE):
            raise ValueError(f'{name}: physics/Recovery contract tampered')
        qfile=folder/'q2.npy'
        q=np.load(qfile,allow_pickle=False)
        if not (len(q)==length and np.all(np.diff(q)>0) and q2_fingerprint(q)==meta['q2_values_sha256'] and
                sha_raw(qfile)==meta['q2_file_sha256']):
            raise ValueError(f'{name}: q2 mismatch')
        if name=='hybrid_ultranear_240':
            if not np.array_equal(q,qref) or sha_raw(qfile)!=qmeta['source_file_sha256']:
                raise ValueError('Hybrid q2 has changed from historical reference')
        else:
            if not np.array_equal(q,np.linspace(-100.,-6.,100)):
                raise ValueError('Uniform control q2 changed')
        with np.load(folder/'normalization_train_only.npz',allow_pickle=False) as d:
            norm={k:d[k] for k in d.files}
        for s in ('train','val'):
            file=folder/'noise_0p2pct'/f'{s}.npz'
            with np.load(file,allow_pickle=False) as d:
                x=np.asarray(d['gy'],np.float64)
                clean=np.asarray(d['gy_clean'],np.float64)
                p_label=np.asarray(d['parameters'],np.float64)
                state_label=d['physical_state_id'].astype(str)
                rep=np.asarray(d['replica_id'],np.int32)
                sig=np.asarray(d['noise_sigma'],np.float64)
                if not (x.shape==(len(state_label),length)==clean.shape and
                        p_label.shape==(len(state_label),5) and
                        np.isfinite(x).all() and np.isfinite(p_label).all()):
                    raise ValueError(f'{name}/{s}: wrong shape or NaN')
                expected_ids=ids[split==s]
                expected_truth=p[split==s].astype(np.float32).astype(np.float64)
                replicas=meta['train_replicas'] if s=='train' else 1
                if not (len(x)==len(expected_ids)*replicas and
                        np.array_equal(state_label,np.repeat(expected_ids,replicas)) and
                        np.array_equal(p_label,np.repeat(expected_truth,replicas,axis=0)) and
                        np.array_equal(rep,np.tile(np.arange(replicas),len(expected_ids)))):
                    raise ValueError(f'{name}/{s}: state/replica leakage or truth mismatch')
                expected_sig=REF_NOISE*np.sqrt(np.mean(clean**2,axis=1))
                if not np.allclose(sig,expected_sig,rtol=2e-6,atol=1e-10):
                    raise ValueError(f'{name}/{s}: sigma violates 0.2% RMS')
                if s=='train':
                    if not np.allclose(norm['input_mean'],x.mean(axis=0),rtol=1e-6,atol=1e-6):
                        raise ValueError('Normalization fitted on non-train data')
                    if not np.allclose(norm['input_std'],x.std(axis=0),rtol=1e-6,atol=1e-6):
                        raise ValueError('Train input standard deviation inconsistent')
                # Recompute limited unnoised physics with 512 Gauss-Legendre points.
                if forward_count:
                    idx=np.linspace(0,len(expected_ids)-1,min(forward_count,len(expected_ids))).astype(int)
                    physical=scaled_forward_observation_at_q2_numpy(p[split==s][idx],q,
                                            integration_points=512,output_dtype=np.float64)
                    original=clean[idx*replicas]
                    if not np.allclose(physical,original,rtol=1e-4,atol=2e-7):
                        raise ValueError(f'{name}/{s}: high-precision physics mismatch')
        if (folder/'noise_0p2pct'/'test.npz').exists():
            raise ValueError('Paired pilot should not include final Test')
        descriptions[name]=dict(q2_points=length,train_states=int(np.sum(split=='train')),
                                val_states=int(np.sum(split=='val')))
    # Ensure data for each design reflects the same unique physical-state truths.
    print('V2 PAIRED PILOT DATA PASS | matched Train/Val truth and split; no Test; no network training')
    print(f'hybrid q2 points={len(qref)} source sha256={qmeta["source_file_sha256"]}')
    return descriptions


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data',type=Path,required=True)
    a.add_argument('--q2-file',type=Path,required=True)
    a.add_argument('--forward-count',type=int,default=12)
    args=a.parse_args()
    validate(args.data,args.q2_file,args.forward_count)

if __name__=='__main__':main()
