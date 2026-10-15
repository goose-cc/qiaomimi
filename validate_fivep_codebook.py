#!/usr/bin/env python3
"""Validate finite selected-codebook g separation and full marginal coverage.

Runs existing dataset integrity validator first. Does NOT train any model.
Finite codebook PASS never certifies continuous-domain identifiability/recovery.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from mc_physics import scaled_forward_observation_at_q2_numpy
from build_fivep_dataset import BOUNDS, NAMES, RECOVERY, REF_NOISE, generate_curves
from build_fivep_codebook import LOW, HIGH, SPLITS, bin_assignments
from validate_fivep_dataset import run as validate_existing


def validate(folder: Path, forward_count: int = 0) -> dict:
    folder = folder.resolve()
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('dataset_kind') != 'FINITE_G_SEPARATED_FIVE_PARAMETER_CODEBOOK':
        raise ValueError('Not the new finite 5P codebook dataset')
    if metadata.get('recovery_tolerances') != RECOVERY:
        raise ValueError('Hard recovery limits changed')
    if metadata.get('parameter_ranges') != {k: list(v) for k, v in BOUNDS.items()}:
        raise ValueError('Parameter domain changed')
    config = metadata['codebook']
    threshold = float(config['threshold'])
    nbins = int(config['coverage_bins_per_parameter_per_split'])
    if not np.isfinite(threshold) or threshold < 6:
        raise ValueError('Selected codebook separation weaker than supplied baseline 6')
    if not 2 <= nbins <= 10:
        raise ValueError('Invalid codebook coverage bins')
    # Reuse old thorough checks (writes only into this NEW dataset).
    base = validate_existing(folder, forward_count, audit_alias=False)
    if base['integrity_status'] != 'PASS':
        raise RuntimeError('Base physical data failed integrity verification')

    with (folder/'state_manifest.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    params = np.array([[float(row[k]) for k in NAMES] for row in rows], dtype=np.float64)
    split = np.asarray([row['split'] for row in rows])
    q2 = np.load(folder/'q2.npy', allow_pickle=False)
    b = bin_assignments(params, nbins)
    report = {}
    for name in SPLITS:
        mask = split == name
        subset = params[mask]
        if len(subset) == 0:
            raise ValueError(f'Empty {name} split')
        counts = np.array([np.bincount(b[mask, j], minlength=nbins)
                           for j in range(5)])
        if np.any(counts == 0):
            raise ValueError(f'{name} missing a parameter coverage bin')
        if name in ('train', 'test'):
            for j, label in enumerate(NAMES):
                if not (np.any(subset[:, j] == LOW[j]) and
                        np.any(subset[:, j] == HIGH[j])):
                    raise ValueError(f'{name} lacks exact {label} endpoints')
        if not np.array_equal(counts, np.asarray(config['coverage_bin_counts'][name])):
            raise ValueError(f'{name} bin-counts do not match metadata')
        report[name] = {'states': int(mask.sum()),
                        'min': dict(zip(NAMES, map(float, subset.min(axis=0)))),
                        'max': dict(zip(NAMES, map(float, subset.max(axis=0)))),
                        'coverage_bins': counts.tolist()}

    def pair_distance(g: np.ndarray):
        sig = REF_NOISE*np.sqrt(np.mean(g*g, axis=1))
        if not np.all(np.isfinite(g)) or np.any(sig <= 0):
            raise ValueError('Invalid forward g/noise scale')
        diff = squareform(pdist(g, metric='euclidean'))
        d = diff / np.maximum(sig[:, None], sig[None, :])
        np.fill_diagonal(d, np.inf)
        return d

    # Independent high-quadrature audit; cannot rely on generated metadata.
    signal_128 = generate_curves(params, q2)
    signal_512 = scaled_forward_observation_at_q2_numpy(
        params, q2, integration_points=512, output_dtype=np.float64)
    min_128 = float(pair_distance(signal_128).min())
    d_high = pair_distance(signal_512)
    min_512 = float(d_high.min())
    if min(min_128, min_512) < threshold * (1 - 1e-8):
        raise ValueError(f'Observed codebook near-alias: Dmin={min_512} < {threshold}')
    if not np.isclose(config['min_pairwise_normalized_g'], min_128, atol=1e-5, rtol=1e-5):
        raise ValueError('Pairwise separation does not match metadata')
    train_test_d = d_high[np.ix_(split == 'train', split == 'test')]
    if train_test_d.size == 0:
        raise ValueError('Empty train-test cross product')
    train_test_min = float(train_test_d.min())
    # Non-training nearest-neighbor sanity baseline on clean g only.
    # This is NOT a neural model, and its accuracy is NOT an upper bound.
    d_tt = np.linalg.norm(signal_512[split == 'test', None, :] -
                          signal_512[None, split == 'train', :], axis=2)
    nearest_train = np.argmin(d_tt, axis=1)
    y_test = params[split == 'test']
    y_near = params[split == 'train'][nearest_train]
    gate_a1 = np.abs(y_test[:, 0] - y_near[:, 0]) <= RECOVERY['a1_abs']
    gate_m = np.abs(y_test[:, 3] - y_near[:, 3]) <= RECOVERY['m_abs']
    gate_gamma = np.maximum(y_test[:, 4]/y_near[:, 4], y_near[:, 4]/y_test[:, 4]) <= RECOVERY['gamma_factor']
    nearest_proxy = {
        'a1_fraction': float(np.mean(gate_a1)),
        'm_fraction': float(np.mean(gate_m)),
        'gamma_fraction': float(np.mean(gate_gamma)),
        'joint_fraction': float(np.mean(gate_a1 & gate_m & gate_gamma)),
        'interpretation': 'Clean-g nearest TRAIN state baseline; does NOT train a network or certify feasibility.',
    }
    report_final = {
        'status': 'FINITE_CODEBOOK_INTEGRITY_PASS_NOT_RECOVERY_CERTIFIED',
        'original_recovery': RECOVERY,
        'original_parameter_ranges': {k: list(v) for k, v in BOUNDS.items()},
        'original_q2_domain': [-100., -6.],
        'original_noise': REF_NOISE,
        'selected_states': len(params),
        'min_pairwise_g_128': min_128,
        'min_pairwise_g_512': min_512,
        'min_train_test_g_512': train_test_min,
        'required_min_g_distance': threshold,
        'by_split': report,
        'nearest_train_no_model_baseline': nearest_proxy,
        'limitations': [
            'Only SELECTED states are pairwise g-separated; the full continuous physical domain can still contain near aliases.',
            'Parameter marginal endpoints/bins do not imply all joint parameter combinations are represented.',
            'A separated codebook does not establish generalization or the original recovery acceptance rate.',
        ],
    }
    (folder/'codebook_validation_report.json').write_text(
        json.dumps(report_final, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'FINITE CODEBOOK PASS | states={len(params)} | '
          f'Dmin128={min_128:.6g} | Dmin512={min_512:.6g} | '
          f'train-test Dmin512={train_test_min:.6g}')
    print('Train/Test: exact endpoints for all 5 parameters; all splits: full marginal bin coverage.')
    print(f'NO-TRAINING nearest-TRAIN joint gate fraction: {nearest_proxy["joint_fraction"]:.3f} (diagnostic only)')
    print('NO NETWORK TRAINING. NOT CONTINUOUS IDENTIFIABILITY OR RECOVERY CERTIFICATION.')
    return report_final


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data', type=Path, default=Path('data_fivep_codebook_v1'))
    cli.add_argument('--forward-count', type=int, default=0)
    args = cli.parse_args()
    if args.forward_count < 0:
        cli.error('--forward-count must be nonnegative')
    validate(args.data, args.forward_count)


if __name__ == '__main__':
    main()
