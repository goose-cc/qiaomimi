#!/usr/bin/env python3
"""Independent integrity checks for build_fivep_dataset.py output; never certifies recovery."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from mc_physics import scaled_forward_observation_at_q2_numpy, valid_parameter_mask_numpy
from build_fivep_dataset import (BOUNDS, NAMES, RECOVERY, REF_NOISE,
                                 FIXED, INTEGRATION_POINTS, FORMULA,
                                 assert_source_contract, q2_grid,
                                 state_identity, generate_curves, fingerprint)


def alias_witness(q2: np.ndarray) -> dict:
    """One deterministic, bounded 5P compensation check; NOT a global certificate."""
    truth = np.asarray([.125, .025, .0, .8, .2], dtype=np.float64)
    altered = truth.copy()
    altered[0] = .14  # > twice the a1 tolerance: no estimate can satisfy both
    target = scaled_forward_observation_at_q2_numpy(truth[None, :], q2, 512)[0]
    free = [1, 2, 3, 4]
    low = np.asarray([BOUNDS[NAMES[i]][0] for i in free])
    high = np.asarray([BOUNDS[NAMES[i]][1] for i in free])

    def resid(z):
        altered[free] = z
        return (scaled_forward_observation_at_q2_numpy(altered[None, :], q2, 128)[0]
                - target)

    fit = least_squares(resid, truth[free], bounds=(low, high),
                        max_nfev=250, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    altered[free] = fit.x
    physical = bool(valid_parameter_mask_numpy(altered[None, :])[0])
    estimate = scaled_forward_observation_at_q2_numpy(altered[None, :], q2, 512)[0]
    sigma = REF_NOISE * float(np.sqrt(np.mean(target**2)))
    return {
        'truth': truth.tolist(), 'alternative': altered.tolist(),
        'both_physically_valid': bool(valid_parameter_mask_numpy(truth[None, :])[0] and physical),
        'a1_recovery_windows_disjoint': bool(abs(altered[0]-truth[0]) > 2*RECOVERY['a1_abs']),
        'observation_difference_over_noise_rms': float(np.sqrt(np.mean((estimate-target)**2)) / sigma),
        'meaning': 'Single continuous alias witness; NOT a proof of dataset recoverability or a global impossibility theorem',
    }


def run(root: Path, forward_count: int, audit_alias: bool) -> dict:
    assert_source_contract()
    root = root.resolve()
    errors: list[str] = []
    results: list[dict] = []

    def check(name: str, condition: bool, detail=None) -> None:
        ok = bool(condition)
        results.append({'check': name, 'status': 'PASS' if ok else 'FAIL', 'detail': detail})
        if not ok:
            errors.append(name)

    meta = json.loads((root/'metadata.json').read_text('utf-8'))
    check('hard_recovery_immutable', meta.get('recovery_tolerances') == RECOVERY)
    check('hard_a1_m_gamma_ranges', all(meta.get('parameter_ranges', {}).get(k) == list(BOUNDS[k])
            for k in ('a1', 'm', 'gamma')))
    check('a2_a3_source_ranges', all(meta.get('parameter_ranges', {}).get(k) == list(BOUNDS[k])
            for k in ('a2', 'a3')))
    check('reference_noise_fixed', meta.get('reference_noise') == REF_NOISE)
    check('source_formula_fixed', meta.get('formula_version') == FORMULA
          and meta.get('physics_integration_points') == INTEGRATION_POINTS)
    src = Path(__file__).resolve().parent
    for file in ('mc_physics.py', 'mc_pool_config.py'):
        check('physics_source_hash_'+file, fingerprint(src/file) == meta['source_sha256'][file])

    q2 = np.load(root/'q2.npy', allow_pickle=False)
    check('locked_q2_grid', np.array_equal(q2_grid(None if not (root/'q2.npy').exists() else str(root/'q2.npy')), q2))
    check('locked_q2_hash', hashlib.sha256(q2.astype('<f8').tobytes()).hexdigest() == meta['q2_sha256'])

    with (root/'state_manifest.csv').open(newline='', encoding='utf-8') as f:
        records = list(csv.DictReader(f))
    ids = np.asarray([r['physical_state_id'] for r in records])
    numeric_ids = np.asarray([int(r['state_id']) for r in records], dtype=np.int64)
    p = np.asarray([[float(r[k]) for k in NAMES] for r in records], dtype=np.float64)
    split = np.asarray([r['split'] for r in records])
    check('manifest_nonempty_unique_states', len(ids) >= 60 and len(set(ids)) == len(ids)
          and len(set(map(int, numeric_ids))) == len(ids), len(ids))
    check('manifest_sha_id_consistency', all(state_identity(row) == (pid, int(sid))
          for row, pid, sid in zip(p, ids, numeric_ids)))
    check('physical_nonnegative_and_finite', bool(valid_parameter_mask_numpy(p).all()))
    check('all_locked_parameter_ranges', all(np.all((p[:, j] >= BOUNDS[k][0]) &
          (p[:, j] <= BOUNDS[k][1])) for j, k in enumerate(NAMES)))
    check('manifest_three_splits', set(split) == {'train', 'val', 'test'})
    check('manifest_expected_split_counts', all(int(np.count_nonzero(split == k)) ==
          meta['sample_counts'][k] for k in ('train', 'val', 'test')))
    stage = int(meta['stage'][0])
    check('correct_stage_fixed_parameters',
          (stage != 3 or np.all(p[:, 1] == FIXED['a2']))
          and (stage == 5 or np.all(p[:, 2] == FIXED['a3'])))

    levels = [('noise_0p2pct', 'train', REF_NOISE),
              ('noise_0p2pct', 'val', REF_NOISE),
              ('noise_0p2pct', 'test', REF_NOISE),
              ('noise_0pct', 'test', 0.),
              ('noise_1pct', 'test', .01)]
    manifest_lookup = dict(zip(ids.tolist(), p))
    clean_by_id = {}
    sample_total = 0
    for directory, spl, level in levels:
        path = root / directory / f'{spl}.npz'
        with np.load(path, allow_pickle=False) as z:
            mandatory = {'q2', 'gy', 'gy_clean', 'gy_noisy', 'g_clean', 'noise',
                         'parameters', 'a1', 'a2', 'a3', 'm', 'gamma', 'physical_state_id',
                         'state_id', 'sample_id', 'noise_level', 'noise_sigma',
                         'replica_id', 'q2_observed', 'observation_indices', 'split'}
            check(f'{directory}_{spl}_fields', mandatory.issubset(z.files))
            x = np.asarray(z['gy'], np.float64)
            gp = np.asarray(z['g_clean'], np.float64)
            pp = np.asarray(z['parameters'], np.float64)
            noise = np.asarray(z['noise'], np.float64)
            sigma = np.asarray(z['noise_sigma'], np.float64)
            l = np.asarray(z['noise_level'], np.float64)
            sid = np.asarray(z['physical_state_id']).astype(str)
            sample_id = np.asarray(z['sample_id']).astype(str)
            reps = np.asarray(z['replica_id'])
            sample_total += len(sid)
            check(f'{directory}_{spl}_shapes_finite', x.shape == gp.shape == noise.shape ==
                  (len(sid), len(q2)) and pp.shape == (len(sid), 5)
                  and np.isfinite(x).all() and np.isfinite(pp).all())
            check(f'{directory}_{spl}_q2', np.array_equal(z['q2'], q2)
                  and np.array_equal(z['q2_observed'], q2)
                  and np.array_equal(z['observation_indices'], np.arange(len(q2))))
            check(f'{directory}_{spl}_unique_sample_id', len(set(sample_id)) == len(sample_id))
            check(f'{directory}_{spl}_state_ownership', set(sid) == set(ids[split == spl])
                  and all(s in manifest_lookup for s in sid))
            check(f'{directory}_{spl}_noise_level', bool(np.allclose(l, level, rtol=0, atol=1e-9)))
            check(f'{directory}_{spl}_noise_definition',
                  np.allclose(x-gp, noise, rtol=2e-3, atol=2e-6)
                  and np.allclose(sigma, level*np.sqrt(np.mean(gp*gp, axis=1)),
                                  rtol=1e-5, atol=1e-8))
            check(f'{directory}_{spl}_zero_noise' if level == 0 else f'{directory}_{spl}_nonzero_noise',
                  bool(np.all(noise == 0)) if level == 0 else bool(np.any(noise != 0)))
            check(f'{directory}_{spl}_replica_counts', all(int(np.sum(sid == state)) ==
                  int(meta['noise_splits'][directory][spl]) for state in set(sid)))
            check(f'{directory}_{spl}_parameter_labels',
                  np.allclose(pp, np.stack([manifest_lookup[s] for s in sid]), atol=6e-8, rtol=0.)
                  and all(np.allclose(z[k], pp[:, j], rtol=0, atol=0)
                          for j, k in enumerate(NAMES))
                  and np.all(np.asarray(z['split']).astype(str) == spl))
            check(f'{directory}_{spl}_numeric_id', np.array_equal(z['state_id'],
                  np.asarray([numeric_ids[np.flatnonzero(ids == s)[0]] for s in sid])))
            for s, g in zip(sid, gp):
                if s in clean_by_id:
                    if not np.array_equal(clean_by_id[s], g):
                        errors.append('inconsistent_clean_curve_'+s)
                else:
                    clean_by_id[s] = g.copy()
    check('all_physical_states_have_clean_curve', set(clean_by_id) == set(ids))

    rng = np.random.default_rng(19931)
    if forward_count == 0 or forward_count >= len(ids):
        selected = np.arange(len(ids))
    else:
        selected = np.unique(np.r_[0, len(ids)-1, rng.choice(len(ids), forward_count, replace=False)])
    direct = generate_curves(p[selected], q2)
    stored = np.stack([clean_by_id[ids[i]] for i in selected])
    rel_error = float(np.max(np.abs(direct-stored)/np.maximum(np.abs(direct), 1e-9)))
    check('direct_physics_forward_recomputation', np.allclose(direct, stored, rtol=3e-6, atol=3e-7), rel_error)
    high_order = scaled_forward_observation_at_q2_numpy(
        p[selected], q2, integration_points=512, output_dtype=np.float64)
    quadrature_error = float(np.max(np.abs(direct-high_order) / np.maximum(np.abs(high_order), 1e-9)))
    check('128_vs_512_point_quadrature_convergence',
          np.allclose(direct, high_order, rtol=5e-5, atol=1e-7), quadrature_error)

    with np.load(root/'normalization_train_only.npz', allow_pickle=False) as norm:
        with np.load(root/'noise_0p2pct'/'train.npz', allow_pickle=False) as z:
            train_gy = z['gy'].astype(np.float64)
            pp = z['parameters'].astype(np.float64)
        target = np.column_stack([pp[:, :4], np.log(pp[:, 4])])
        check('train_only_input_normalization',
              np.allclose(norm['input_mean'], train_gy.mean(axis=0), atol=1e-12)
              and np.allclose(norm['input_std'], train_gy.std(axis=0), atol=1e-12))
        expected_std = target.std(axis=0)
        fixed_mask = expected_std <= 1e-12
        expected_std[fixed_mask] = 1.0
        check('train_only_target_normalization',
              np.allclose(norm['target_mean'], target.mean(axis=0), atol=1e-12)
              and np.allclose(norm['target_std'], expected_std, atol=1e-12)
              and np.array_equal(norm['target_fixed_mask'], fixed_mask)
              and norm['target_mean'].shape == (5,)
              and list(norm['target_names']) == ['a1', 'a2', 'a3', 'm', 'log_gamma'])

    coverage = {}
    for j, k in enumerate(NAMES):
        h = np.histogram(p[:, j] if k != 'gamma' else np.log10(p[:, j]), bins=10)[0]
        coverage[k] = {'min': float(np.min(p[:, j])), 'max': float(np.max(p[:, j])),
                       'ten_bin_counts': h.tolist()}
    check('five_parameter_variation', stage != 5 or all(float(np.ptp(p[:,j])) > 1e-6 for j in range(5)))
    free_names = meta['free_parameters']
    check('closed_boundaries_present_for_free_params', all(
          np.any(p[:, NAMES.index(k)] == BOUNDS[k][0]) and
          np.any(p[:, NAMES.index(k)] == BOUNDS[k][1]) for k in free_names))
    witness = alias_witness(q2) if audit_alias else None
    report = {'integrity_status': 'PASS' if not errors else 'FAIL',
              'recovery_status': 'NOT_TESTED_REQUIRES_MODEL_PREDICTIONS',
              'recovery_gates_unchanged': RECOVERY,
              'stage': meta['stage'], 'physical_states': len(ids),
              'sample_count_all_files': sample_total,
              'forward_states_checked': len(selected),
              'max_forward_relative_error': rel_error,
              'max_quadrature_relative_error': quadrature_error,
              'split_counts': meta['sample_counts'],
              'parameter_coverage': coverage, 'alias_witness': witness,
              'checks': results, 'errors': errors}
    (root/'validation_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'DATA INTEGRITY {report["integrity_status"]} | stage {meta["stage"]} | '
          f'physical states {len(ids)} | forward checked {len(selected)}')
    print('RECOVERY NOT CERTIFIED; requires independent 5-output training and holdout evaluation.')
    if witness is not None:
        print('Forced wrong-a1 alias RMS/noise:', witness['observation_difference_over_noise_rms'])
    if errors:
        print('FAILED CHECKS:', errors)
        raise SystemExit(2)
    return report


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data', default='data_fivep_smoke')
    cli.add_argument('--forward-count', type=int, default=64,
                     help='Recompute N distinct states; use 0 for all states')
    cli.add_argument('--alias-witness', action='store_true',
                     help='Extra deterministic 5P near-alias search with locked recovery gates')
    args = cli.parse_args()
    if args.forward_count < 0:
        raise ValueError('forward-count must be nonnegative')
    run(Path(args.data), args.forward_count, args.alias_witness)


if __name__ == '__main__':
    main()
