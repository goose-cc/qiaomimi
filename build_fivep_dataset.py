#!/usr/bin/env python3
"""Additive 3P/4P/5P physical-state dataset builder; original physics unchanged."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from mc_physics import (scaled_forward_observation_at_q2_numpy,
                        valid_parameter_mask_numpy)
from mc_pool_config import DEFAULT_PHYSICS

NAMES = ('a1', 'a2', 'a3', 'm', 'gamma')
BOUNDS = {'a1': (0.05, 0.20), 'a2': (0.0, 0.05),
          'a3': (-0.05, 0.05), 'm': (0.40, 1.20),
          'gamma': (0.01, 1.00)}
RECOVERY = {'a1_abs': 0.005, 'm_abs': 0.03, 'gamma_factor': 1.2}
REF_NOISE = 0.002
FIXED = {'a2': 0.025, 'a3': 0.0}
FORMULA = 'rho-m2-center-v1'
INTEGRATION_POINTS = 128
Q2_COUNT = 100
SEED_DEFAULT = 20261012


def assert_source_contract() -> None:
    c = DEFAULT_PHYSICS
    expected = {'a2_min': 0., 'a2_max': .05, 'a3_min': -.05,
                'a3_max': .05, 's_min': .1764, 's_max': 6.,
                'q2_min': -100., 'q2_max': -6., 'q2_points': Q2_COUNT,
                'shift': 400., 'data_scale': 160000.}
    changed = {k: (getattr(c, k), v) for k, v in expected.items()
               if getattr(c, k) != v}
    if changed:
        raise RuntimeError(f'Existing physics source differs: {changed}')
    for p, (lo, hi) in BOUNDS.items():
        if p in ('a1', 'm', 'gamma'):
            src_lo = getattr(c, p + ('_min_open' if p != 'a1' else '_min'))
            src_hi = getattr(c, p + '_max')
            if not (src_lo <= lo < hi <= src_hi):
                raise RuntimeError(f'Locked {p} range incompatible with physics')


def q2_grid(path: str | None) -> np.ndarray:
    q = (np.asarray(np.load(path), dtype=np.float64).reshape(-1)
         if path else np.linspace(-100., -6., Q2_COUNT, dtype=np.float64))
    if (q.shape != (Q2_COUNT,) or not np.isfinite(q).all()
            or np.any(np.diff(q) <= 0.)
            or q[0] != -100. or q[-1] != -6.
            or np.any(q < -100.) or np.any(q > -6.)):
        raise ValueError('q2 must be 100 strictly increasing points in [-100,-6], including both endpoints')
    return q


def _stage_bounds(stage: int) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    free = {3: (0, 3, 4), 4: (0, 1, 3, 4),
            5: (0, 1, 2, 3, 4)}[stage]
    low = np.array([BOUNDS[k][0] for k in NAMES], dtype=np.float64)
    high = np.array([BOUNDS[k][1] for k in NAMES], dtype=np.float64)
    return low, high, free


def create_states(n: int, seed: int, stage: int) -> tuple[np.ndarray, dict]:
    if n < 60:
        raise ValueError('at least 60 physical states needed for grouped train/val/test')
    lo, hi, free = _stage_bounds(stage)
    found: list[np.ndarray] = []
    seen: set[bytes] = set()
    # Deterministic physical boundary anchors: keep the actual closed endpoints.
    # The a3=-0.05 endpoint needs a different background/resonance combination
    # to respect continuous rho(s)>=0, rather than weakening that condition.
    for index in free:
        for endpoint in (lo[index], hi[index]):
            anchor = np.asarray([.125, FIXED['a2'], FIXED['a3'], .8, .2], dtype=np.float64)
            anchor[index] = endpoint
            if index == 2 and endpoint < 0:
                anchor = np.asarray([.125, .05, -.05, .6, .2], dtype=np.float64)
            if not valid_parameter_mask_numpy(anchor[None, :])[0]:
                raise RuntimeError(f'Cannot construct physical endpoint anchor for {NAMES[index]}={endpoint}')
            key = anchor.astype('<f8').tobytes()
            if key not in seen:
                found.append(anchor)
                seen.add(key)
    anchor_count = len(found)
    attempted = 0
    physically_rejected = 0
    rounds = 0
    # Each round uses a full 2^m scrambled Sobol block, never partial sample calls.
    while len(found) < n and rounds < 32:
        block_power = max(7, int(math.ceil(math.log2(n - len(found)))))
        block = qmc.Sobol(d=len(free), scramble=True, seed=seed + 7919*rounds).random_base2(block_power)
        p = np.tile([.125, FIXED['a2'], FIXED['a3'], .8, .2], (len(block), 1)).astype(np.float64)
        for j, index in enumerate(free):
            if index == 4:
                p[:, index] = np.exp(np.log(lo[index]) + block[:, j] * (np.log(hi[index]) - np.log(lo[index])))
            else:
                p[:, index] = lo[index] + block[:, j] * (hi[index] - lo[index])
        # Explicit locked-box assertion. Real physical validity also needs rho(s)>=0.
        if np.any(p < lo - 1e-12) or np.any(p > hi + 1e-12):
            raise RuntimeError('Generated states leave the frozen physical bounds')
        ok = valid_parameter_mask_numpy(p)
        attempted += len(p)
        physically_rejected += int(np.count_nonzero(~ok))
        for row in p[ok]:
            key = row.astype('<f8').tobytes()
            if key not in seen:
                seen.add(key)
                found.append(row)
                if len(found) == n:
                    break
        rounds += 1
    if len(found) != n:
        raise RuntimeError(f'Only {len(found)} valid states after {attempted} attempts: increase sampler budget')
    p = np.stack(found)
    if not valid_parameter_mask_numpy(p).all():
        raise RuntimeError('Physical validity not satisfied')
    return p, {'attempted': attempted, 'physically_rejected': physically_rejected,
               'accepted': n, 'physical_endpoint_anchors': anchor_count,
               'sobol_rounds': rounds,
               'sobol_note': 'Accepted sample is filtered; Sobol balance is not guaranteed after rejection'}


def split_states(p: np.ndarray, seed: int) -> np.ndarray:
    # Group by the *true* gamma state before expanding noise; covers all width scales.
    gamma_bins = np.clip(np.floor((np.log10(p[:, 4]) + 2.) / 2. * 5).astype(int), 0, 4)
    split = np.full(len(p), '', dtype='<U5')
    rng = np.random.default_rng(seed + 701)
    for band in range(5):
        ids = np.flatnonzero(gamma_bins == band)
        if len(ids) < 3:
            raise RuntimeError(f'gamma band {band} has fewer than 3 states; do not relax domain')
        rng.shuffle(ids)
        n_test = max(1, int(round(.15 * len(ids))))
        n_val = max(1, int(round(.15 * len(ids))))
        if n_test + n_val >= len(ids):
            raise RuntimeError('Cannot split a gamma stratum into 3 sets')
        split[ids[:n_test]] = 'test'
        split[ids[n_test:n_test+n_val]] = 'val'
        split[ids[n_test+n_val:]] = 'train'
    if np.any(split == '') or set(split) != {'train', 'val', 'test'}:
        raise RuntimeError('Physical-state split incomplete')
    return split


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_identity(p: np.ndarray) -> tuple[str, int]:
    digest = hashlib.sha256(FORMULA.encode('ascii') + np.asarray(p, '<f8').tobytes()).digest()
    return 'f5_' + digest.hex()[:32], int.from_bytes(digest[:8], 'little') & ((1 << 63) - 1)


def generate_curves(p: np.ndarray, q2: np.ndarray, batch: int = 256) -> np.ndarray:
    return np.concatenate([
        scaled_forward_observation_at_q2_numpy(p[i:i+batch], q2,
            integration_points=INTEGRATION_POINTS, output_dtype=np.float64)
        for i in range(0, len(p), batch)], axis=0)


def write_examples(path: Path, p: np.ndarray, clean: np.ndarray, ids: np.ndarray,
                   numeric_ids: np.ndarray, split_name: str, q2: np.ndarray,
                   noise_level: float, replicas: int, seed: int) -> None:
    if replicas < 1:
        raise ValueError('replicas must be positive')
    indices = np.repeat(np.arange(len(p)), replicas)
    params = p[indices]
    g = clean[indices]
    sigma = noise_level * np.sqrt(np.mean(g*g, axis=1))
    rng = np.random.default_rng(seed)
    noise = sigma[:, None] * rng.standard_normal(g.shape)
    gy = g + noise
    ids_repeated = ids[indices]
    repl = np.tile(np.arange(replicas, dtype=np.int32), len(p))
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray([f'{pid}:{split_name}:{noise_level:g}:r{r}'
                          for pid, r in zip(ids_repeated, repl)], dtype='<U96')
    np.savez_compressed(path,
        gy=gy.astype(np.float32), gy_noisy=gy.astype(np.float32),
        gy_clean=g.astype(np.float32), g_clean=g.astype(np.float32),
        noise=noise.astype(np.float32),
        parameters=params.astype(np.float32), params=params.astype(np.float32),
        params3=params[:, [0, 3, 4]].astype(np.float32),
        a1=params[:, 0].astype(np.float32), a2=params[:, 1].astype(np.float32),
        a3=params[:, 2].astype(np.float32), m=params[:, 3].astype(np.float32),
        gamma=params[:, 4].astype(np.float32),
        physical_state_id=ids_repeated, state_id=numeric_ids[indices],
        sample_id=samples, replica_id=repl, noise_replica_id=repl,
        noise_level=np.full(len(params), noise_level, np.float32),
        noise_sigma=sigma.astype(np.float32),
        split=np.full(len(params), split_name, dtype='<U5'),
        q2=q2, q2_observed=q2,
        observation_indices=np.arange(len(q2), dtype=np.int32))


def write_manifest(path: Path, p: np.ndarray, splits: np.ndarray,
                   pid: np.ndarray, sid: np.ndarray) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['physical_state_id', 'state_id', 'split', *NAMES])
        for i in range(len(p)):
            writer.writerow([pid[i], int(sid[i]), splits[i],
                             *(format(float(v), '.17g') for v in p[i])])


def write_normalization(path: Path) -> None:
    with np.load(path / 'noise_0p2pct' / 'train.npz', allow_pickle=False) as z:
        x = np.asarray(z['gy'], np.float64)
        p = np.asarray(z['parameters'], np.float64)
    target = np.column_stack([p[:, :4], np.log(p[:, 4])])
    xm, xs = x.mean(axis=0), x.std(axis=0)
    ym, ys = target.mean(axis=0), target.std(axis=0)
    if not np.all(xs > 0):
        raise RuntimeError('Degenerate train-only input normalization')
    fixed_target = ys <= 1e-12
    ys[fixed_target] = 1.0  # 3P/4P nuisance constants, never divide by zero
    np.savez_compressed(path / 'normalization_train_only.npz',
        input_mean=xm, input_std=xs,
        target_mean=ym, target_std=ys, target_fixed_mask=fixed_target,
        target_names=np.asarray(['a1', 'a2', 'a3', 'm', 'log_gamma']),
        target_transform=np.asarray(['identity']*4 + ['natural_log']))


def make_dataset(output: Path, n: int, seed: int, stage: int, q2_path: str | None,
                 train_replicas: int) -> None:
    assert_source_contract()
    if train_replicas < 1 or train_replicas > 100:
        raise ValueError('train_replicas must be 1..100')
    q2 = q2_grid(q2_path)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'{output} already exists; choose a NEW directory, never overwrite')
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='._building_f5_', dir=str(output.parent)))
    try:
        p, sampling = create_states(n, seed, stage)
        split = split_states(p, seed)
        pid_sid = [state_identity(row) for row in p]
        pid = np.asarray([v[0] for v in pid_sid], dtype='<U35')
        sid = np.asarray([v[1] for v in pid_sid], dtype=np.int64)
        if len(set(pid)) != n or len(set(map(int, sid))) != n:
            raise RuntimeError('Physical-state hash collision')
        clean = generate_curves(p, q2)
        if not np.isfinite(clean).all():
            raise RuntimeError('Non-finite physical curves')
        write_manifest(tmp / 'state_manifest.csv', p, split, pid, sid)
        np.save(tmp / 'q2.npy', q2)
        for name, s in [('train', seed+11), ('val', seed+23), ('test', seed+37)]:
            mask = split == name
            write_examples(tmp/'noise_0p2pct'/f'{name}.npz', p[mask], clean[mask],
                           pid[mask], sid[mask], name, q2, REF_NOISE,
                           train_replicas if name == 'train' else 1, s)
        # Separate testing challenges; *same* held-out states, never extra training.
        test_mask = split == 'test'
        for label, level, shift in [('noise_0pct', 0., 41), ('noise_1pct', .01, 43)]:
            write_examples(tmp/label/'test.npz', p[test_mask], clean[test_mask],
                           pid[test_mask], sid[test_mask], 'test', q2, level, 1, seed+shift)
        write_normalization(tmp)
        source = Path(__file__).resolve().parent
        metadata = {
            'dataset_kind': 'UNFILTERED_FULL_DOMAIN_CANDIDATES_NOT_RECOVERY_CERTIFIED',
            'stage': f'{stage}P', 'parameter_names': list(NAMES),
            'free_parameters': [NAMES[i] for i in _stage_bounds(stage)[2]],
            'fixed_parameters': {'a2': FIXED['a2'] if stage == 3 else None,
                                 'a3': FIXED['a3'] if stage != 5 else None},
            'parameter_ranges': {k: list(v) for k,v in BOUNDS.items()},
            'q2_domain': [-100., -6.], 'q2_points': Q2_COUNT,
            'q2_sha256': hashlib.sha256(q2.astype('<f8').tobytes()).hexdigest(),
            'formula_version': FORMULA, 's_domain': [.1764, 6.],
            'shift': 400., 'data_scale': 160000.,
            'physics_integration_points': INTEGRATION_POINTS,
            'reference_noise': REF_NOISE,
            'noise_rule': 'sigma=level*RMS(g_clean); gy=g_clean+sigma*N(0,1)',
            'noise_splits': {'noise_0p2pct': {'train': train_replicas, 'val': 1, 'test': 1},
                             'noise_0pct': {'test': 1}, 'noise_1pct': {'test': 1}},
            'recovery_tolerances': RECOVERY,
            'split_rule': 'by physical_state_id before noise; stratify log10(gamma) in five bands',
            'sample_counts': {name: int(np.count_nonzero(split == name))
                              for name in ('train', 'val', 'test')},
            'sampler': sampling, 'seed': seed,
            'q2_design': 'user_supplied' if q2_path else 'uniform_100',
            'source_sha256': {'mc_physics.py': fingerprint(source/'mc_physics.py'),
                              'mc_pool_config.py': fingerprint(source/'mc_pool_config.py')},
            'scientific_status': 'INTEGRITY_ONLY__RECOVERY_NOT_PROVEN',
        }
        (tmp/'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        tmp.rename(output)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(f'Created {output} | stage {stage}P | physical states {n} | splits {metadata["sample_counts"]}')
    print('NOT RECOVERY CERTIFIED. Validate separately; train 5-output models for 5P.')


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--stage', type=int, choices=(3, 4, 5), default=5)
    a.add_argument('--states', type=int, default=256)
    a.add_argument('--seed', type=int, default=SEED_DEFAULT)
    a.add_argument('--train-replicas', type=int, default=3)
    a.add_argument('--q2-file', default=None, help='Optional NPY design; must retain exact domain and 100 points')
    a.add_argument('--out', default='data_fivep_smoke')
    args = a.parse_args()
    make_dataset(Path(args.out), args.states, args.seed, args.stage,
                 args.q2_file, args.train_replicas)


if __name__ == '__main__':
    main()
