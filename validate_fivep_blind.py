#!/usr/bin/env python3
"""Verify source contract, blind state splits and TRAIN-only g separation.

Does not train a network. It does not require TEST g to be separated from
TRAIN g; that would bias an unseen-state blind test.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist, pdist
from build_fivep_dataset import NAMES, BOUNDS, RECOVERY, REF_NOISE, generate_curves
from build_fivep_codebook import LOW, HIGH, bin_assignments
from build_fivep_blind import KIND, SPLITS
from validate_fivep_dataset import run as base_run
from mc_physics import scaled_forward_observation_at_q2_numpy


def validate(root: Path, forward_count: int = 0) -> dict:
    root = root.resolve()
    meta = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    if meta.get('dataset_kind') != KIND:
        raise ValueError('Not a blind held-out five-parameter dataset')
    if meta.get('recovery_tolerances') != RECOVERY or meta.get('parameter_ranges') != {k: list(v) for k,v in BOUNDS.items()}:
        raise ValueError('Hard parameter/recovery contract changed')
    proto = meta.get('blind_protocol', {})
    threshold = float(proto.get('train_g_threshold', -1))
    bins = int(proto.get('coverage_bins', 0))
    if not np.isfinite(threshold) or threshold < 6 or not 2 <= bins <= 10:
        raise ValueError('Train g separation policy or bins were relaxed')
    if not proto.get('validation_test_g_not_used_to_select_states') or not proto.get('train_test_g_separation_is_not_a_selection_gate'):
        raise ValueError('Blind selection protocol flags absent or changed')
    for split in SPLITS:
        if meta['sampler'][split]['selection_uses_candidate_g'] != (split == 'train'):
            raise ValueError('Split selection claims g leakage')
    baseline = base_run(root, forward_count, audit_alias=False)
    if baseline['integrity_status'] != 'PASS':
        raise RuntimeError('Base physical integrity failed')
    with (root/'state_manifest.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    p = np.array([[float(r[k]) for k in NAMES] for r in rows], dtype=np.float64)
    split = np.asarray([r['split'] for r in rows])
    physical_ids = np.asarray([r['physical_state_id'] for r in rows])
    for i, a in enumerate(SPLITS):
        mask = split == a
        if np.count_nonzero(mask) != meta['sample_counts'][a]:
            raise ValueError(f'{a} wrong count')
        if any(set(physical_ids[mask]).intersection(physical_ids[split == b]) for b in SPLITS[i+1:]):
            raise ValueError(f'{a} physical state leaked into another split')
        if len(set(map(tuple, p[mask]))) != int(np.count_nonzero(mask)):
            raise ValueError(f'{a} duplicated parameter truth')
        group_bins = bin_assignments(p[mask], bins)
        if any(np.any(np.bincount(group_bins[:,j], minlength=bins) == 0) for j in range(5)):
            raise ValueError(f'{a} missing predeclared parameter coverage bin')
        if a != 'val':
            if any(not (np.any(p[mask,j] == LOW[j]) and np.any(p[mask,j] == HIGH[j])) for j in range(5)):
                raise ValueError(f'{a} missing marginal endpoints')
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            if set(map(tuple,p[split == a])) & set(map(tuple,p[split == b])):
                raise ValueError('Cross-split parameter truth duplicated')
    q2 = np.load(root/'q2.npy', allow_pickle=False)
    # Independent 512-node integration to check train distinguishability.
    clean_high = scaled_forward_observation_at_q2_numpy(p, q2, integration_points=512, output_dtype=np.float64)
    group = clean_high[split == 'train']
    sigma = REF_NOISE*np.sqrt(np.mean(group**2, axis=1))
    d = pdist(group)
    upper = np.triu_indices(len(group), 1)
    den = np.maximum(sigma[upper[0]], sigma[upper[1]])
    train_min = float(np.min(d/den))
    if train_min < threshold*(1-1e-8):
        raise ValueError(f'Train codebook Dmin={train_min} fell below {threshold}')
    # Holdout proximity is DIAGNOSTIC ONLY, never used to reject test states.
    test = clean_high[split == 'test']
    test_sig = REF_NOISE*np.sqrt(np.mean(test**2, axis=1))
    cross = cdist(group, test) / np.maximum(sigma[:,None], test_sig[None,:])
    cross_min = float(np.min(cross))
    report = {
        'status': 'BLIND_HOLDOUT_DATA_INTEGRITY_PASS_NOT_RECOVERY_CERTIFIED',
        'train_g_min_distance_512': train_min,
        'train_test_g_min_distance_512_DIAGNOSTIC_ONLY': cross_min,
        'train_g_threshold': threshold,
        'split_states': {s: int(np.count_nonzero(split == s)) for s in SPLITS},
        'no_network_training': True,
        'warning': 'Unseen-state test is held out from g-based selection, so low train-test g distances must be reported, not screened away.',
    }
    (root/'blind_validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'BLIND HOLDOUT DATA PASS | train Dmin512={train_min:.5g}; train-test min={cross_min:.5g} (diagnostic only)')
    print('NOT RECOVERY CERTIFIED. DO NOT tune/further screen using held-out test labels.')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--forward-count', type=int, default=0)
    a = ap.parse_args()
    if a.forward_count < 0:
        ap.error('--forward-count must be nonnegative')
    validate(a.data, a.forward_count)


if __name__ == '__main__':
    main()
