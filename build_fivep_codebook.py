#!/usr/bin/env python3
"""Make a 5P finite codebook with unchanged physics, bounds, q2 and recovery gates.

Only the SELECTED finite support is required to be pairwise distinguishable.
This is NOT continuous-domain identifiability or neural-network recovery proof.
Existing datasets and physics files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from mc_physics import valid_parameter_mask_numpy, scaled_forward_observation_at_q2_numpy
from build_fivep_dataset import (
    BOUNDS, NAMES, RECOVERY, REF_NOISE, FORMULA, FIXED,
    INTEGRATION_POINTS, Q2_COUNT, assert_source_contract, q2_grid,
    create_states, generate_curves, state_identity, write_examples,
    write_manifest, write_normalization, fingerprint,
)

LOW = np.asarray([BOUNDS[k][0] for k in NAMES], dtype=np.float64)
HIGH = np.asarray([BOUNDS[k][1] for k in NAMES], dtype=np.float64)
SPLITS = ('train', 'val', 'test')


def candidates(pool_size: int, boundary_each: int, seed: int) -> tuple[np.ndarray, dict]:
    base, old_log = create_states(pool_size, seed, 5)
    rng = np.random.default_rng(seed + 1977)
    seen = {p.astype('<f8').tobytes() for p in base}
    additions = []
    generated = 0
    for j in range(5):
        for end in (LOW[j], HIGH[j]):
            accepted = 0
            for _ in range(30):
                z = rng.random((512, 5))
                p = LOW + z*(HIGH-LOW)
                p[:, 4] = np.exp(np.log(LOW[4]) + z[:, 4] * np.log(HIGH[4]/LOW[4]))
                p[:, j] = end  # Exact endpoint: unchanged physics and domain
                generated += len(p)
                for row in p[valid_parameter_mask_numpy(p)]:
                    key = row.astype('<f8').tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    additions.append(row.copy())
                    accepted += 1
                    if accepted >= boundary_each:
                        break
                if accepted >= boundary_each:
                    break
            if accepted < boundary_each:
                raise RuntimeError(f'Could not construct enough physical endpoint states for {NAMES[j]}={end}')
    return np.vstack([base, additions]), {
        'base': old_log, 'extra_endpoints': len(additions),
        'extra_candidates_drawn': generated,
    }


def bin_assignments(p: np.ndarray, nbins: int) -> np.ndarray:
    z = (p - LOW) / (HIGH - LOW)
    z[:, 4] = np.log(p[:, 4] / LOW[4]) / np.log(HIGH[4] / LOW[4])
    return np.minimum((np.clip(z, 0, 1)*nbins).astype(np.int32), nbins-1)


def normalized_separations(curves: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """All pairwise Euclidean distances in units of the larger per-state noise SD."""
    norms = np.sum(curves * curves, axis=1)
    gram = curves @ curves.T
    square = np.maximum(0., norms[:, None] + norms[None, :] - 2*gram)
    den = np.maximum(sigmas[:, None], sigmas[None, :])
    dist = np.sqrt(square) / den
    np.fill_diagonal(dist, np.inf)
    return dist


def select_codebook(p: np.ndarray, g: np.ndarray, count: int,
                    threshold: float, nbins: int, seed: int):
    if count < 60:
        raise ValueError('At least 60 states required; no relaxation of train/test coverage')
    if not (np.isfinite(threshold) and threshold >= 6.):
        raise ValueError('min-distance must be finite and >=6 (new codebook screening floor, not a recovery gate)')
    if not (2 <= nbins <= 10):
        raise ValueError('coverage-bins must be between 2 and 10')
    if len(p) < count:
        raise RuntimeError('Insufficient candidate pool')
    nval = max(12, int(round(.15*count)))
    ntest = nval
    ntrain = count - nval - ntest
    if ntrain < 12:
        raise ValueError('Insufficient train allocation')
    quotas = {'train': ntrain, 'val': nval, 'test': ntest}
    noise = REF_NOISE * np.sqrt(np.mean(g*g, axis=1))
    if np.any(noise <= 0) or not np.isfinite(noise).all():
        raise ValueError('Nonpositive/nonfinite noise scale')
    bins = bin_assignments(p, nbins)
    rng = np.random.default_rng(seed + 787)
    used = np.zeros(len(p), dtype=bool)
    nearest = np.full(len(p), np.inf)
    chosen = {name: [] for name in SPLITS}
    counts = {name: np.zeros((5, nbins), dtype=np.int32) for name in SPLITS}

    def add(name, idx):
        if used[idx] or nearest[idx] < threshold:
            raise RuntimeError('Internal g-separation constraint violation')
        used[idx] = True
        chosen[name].append(int(idx))
        counts[name][np.arange(5), bins[idx]] += 1
        d = np.linalg.norm(g - g[idx], axis=1) / np.maximum(noise, noise[idx])
        nearest[:] = np.minimum(nearest, d)
        nearest[used] = -np.inf

    def pick(name, options, missing=None):
        allowed = np.flatnonzero(options & ~used & (nearest >= threshold))
        if len(allowed) == 0:
            raise RuntimeError(f'Not enough distinguishable physical states for {name}; '
                               'increase candidate pool or report infeasible, do NOT relax constraints')
        hist = counts[name]
        scarcity = np.sum(1./(1. + hist[np.arange(5)[:, None], bins[allowed].T]), axis=0)
        novelty = np.minimum(nearest[allowed], 100.) / 100.
        score = 10*scarcity + novelty + rng.uniform(0, .0001, len(allowed))
        if missing is not None:
            gain = np.array([sum((j, int(bins[i, j])) in missing for j in range(5))
                             for i in allowed], dtype=np.int32)
            score += 100*gain
        best = int(allowed[np.argmax(score)])
        add(name, best)

    # Both TRAIN and TEST contain exact endpoints of every free parameter.
    # VAL is kept independent and binned but exact endpoints aren't mandatory.
    for name in ('test', 'train'):
        for j in (2, 4, 0, 3, 1):  # rare physical endpoint a3=-.05 first
            for endpoint in (LOW[j], HIGH[j]):
                selected = p[chosen[name]] if chosen[name] else np.empty((0, 5))
                if len(selected) and np.any(selected[:, j] == endpoint):
                    continue
                if len(chosen[name]) >= quotas[name]:
                    raise RuntimeError(f'Quota exhausted before exact range coverage for {name}')
                pick(name, p[:, j] == endpoint)

    # Enforce space coverage independently, not just global min/max.
    for name in ('test', 'val', 'train'):
        need = {(j, b) for j in range(5) for b in range(nbins)}
        for idx in chosen[name]:
            need.difference_update((j, int(bins[idx, j])) for j in range(5))
        while need:
            if len(chosen[name]) >= quotas[name]:
                raise RuntimeError(f'{name}: insufficient quota for {nbins}-bin coverage; '
                                   'increase --states (no forced PASS)')
            mask = np.array([any((j, int(bins[i, j])) in need for j in range(5))
                             for i in range(len(p))], dtype=bool)
            pick(name, mask, need)
            idx = chosen[name][-1]
            need.difference_update((j, int(bins[idx, j])) for j in range(5))

    # Balanced fill by least-covered parameter bins, always checking all splits.
    for name in ('train', 'val', 'test'):
        while len(chosen[name]) < quotas[name]:
            pick(name, np.ones(len(p), dtype=bool))

    indices = np.array(sum((chosen[name] for name in SPLITS), []), dtype=int)
    names = np.array(sum(([name]*len(chosen[name]) for name in SPLITS), []), dtype='<U5')
    # Final all-pairs check, independently recomputed at 512 nodes by validator.
    full = normalized_separations(g[indices], noise[indices])
    worst = float(np.min(full))
    if worst < threshold * (1.-1e-9):
        raise RuntimeError('Final selected codebook has an observation collision')
    for name in ('train', 'test'):
        x = p[indices[names == name]]
        if not all(np.any(x[:, j] == LOW[j]) and np.any(x[:, j] == HIGH[j]) for j in range(5)):
            raise RuntimeError(f'{name} lost exact endpoint coverage')
    return indices, names, {
        'min_pairwise_normalized_g': worst,
        'threshold': threshold, 'coverage_bins_per_parameter_per_split': nbins,
        'quota': quotas,
        'excluded_candidate_alias_conflicts': int(np.sum((~used) & (nearest < threshold))),
        'excluded_candidate_other_than_alias': int(np.sum((~used) & (nearest >= threshold))),
        'coverage_bin_counts': {name: counts[name].tolist() for name in SPLITS},
        'exact_endpoint_coverage_splits': ['train', 'test'],
        'distance_definition': 'L2(g_i-g_j) / max(0.002*RMS(g_i),0.002*RMS(g_j))',
        'scientific_caveat': 'FINITE SELECTED CODEBOOK ONLY; does not resolve continuous-domain aliases',
    }


def build(out: Path, n: int, pool_size: int, boundary_each: int,
          threshold: float, bins: int, seed: int, replicas: int,
          q2_file: str | None):
    assert_source_contract()
    if pool_size < n or boundary_each < 2 or not (1 <= replicas <= 100):
        raise ValueError('Need pool-size >= states, boundary-each >=2, replicas in 1..100')
    q2 = q2_grid(q2_file)
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing directory: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='._fivep_codebook_', dir=out.parent))
    try:
        pool, sampling = candidates(pool_size, boundary_each, seed)
        g = generate_curves(pool, q2)
        ids, split, selection = select_codebook(pool, g, n, threshold, bins, seed)
        p, clean = pool[ids], g[ids]
        split_count = {s: int(np.sum(split == s)) for s in SPLITS}
        pid_sid = [state_identity(row) for row in p]
        pid = np.array([it[0] for it in pid_sid], dtype='<U35')
        sid = np.array([it[1] for it in pid_sid], dtype=np.int64)
        if len(set(pid)) != len(p):
            raise RuntimeError('Repeated physical-state ID')
        write_manifest(temp/'state_manifest.csv', p, split, pid, sid)
        np.save(temp/'q2.npy', q2)
        for name, noise_seed in (('train', seed+11), ('val', seed+23), ('test', seed+37)):
            use = split == name
            write_examples(temp/'noise_0p2pct'/f'{name}.npz', p[use], clean[use],
                           pid[use], sid[use], name, q2, REF_NOISE,
                           replicas if name == 'train' else 1, noise_seed)
        test = split == 'test'
        for noise_dir, lev, no_seed in (('noise_0pct', 0., seed+41),
                                         ('noise_1pct', .01, seed+43)):
            write_examples(temp/noise_dir/'test.npz', p[test], clean[test],
                           pid[test], sid[test], 'test', q2, lev, 1, no_seed)
        write_normalization(temp)
        folder = Path(__file__).resolve().parent
        meta = {
            'dataset_kind': 'FINITE_G_SEPARATED_FIVE_PARAMETER_CODEBOOK',
            'stage': '5P', 'parameter_names': list(NAMES),
            'free_parameters': list(NAMES), 'fixed_parameters': {'a2': None, 'a3': None},
            'parameter_ranges': {key: list(value) for key, value in BOUNDS.items()},
            'q2_domain': [-100., -6.], 'q2_points': Q2_COUNT,
            'q2_sha256': hashlib.sha256(q2.astype('<f8').tobytes()).hexdigest(),
            'formula_version': FORMULA, 's_domain': [.1764, 6.],
            'shift': 400., 'data_scale': 160000.,
            'physics_integration_points': INTEGRATION_POINTS,
            'reference_noise': REF_NOISE,
            'noise_rule': 'sigma=level*RMS(g_clean); gy=g_clean+sigma*N(0,1)',
            'noise_splits': {'noise_0p2pct': {'train': replicas, 'val': 1, 'test': 1},
                             'noise_0pct': {'test': 1}, 'noise_1pct': {'test': 1}},
            'recovery_tolerances': RECOVERY,
            'split_rule': 'global g-separation first; exact endpoints for train/test; independent state IDs before noise',
            'sample_counts': split_count, 'sampler': sampling, 'seed': seed,
            'q2_design': 'user_supplied' if q2_file else 'uniform_100',
            'codebook': selection,
            'source_sha256': {'mc_physics.py': fingerprint(folder/'mc_physics.py'),
                              'mc_pool_config.py': fingerprint(folder/'mc_pool_config.py')},
            'scientific_status': 'FINITE_CODEBOOK_INTEGRITY_ONLY__NOT_CONTINUOUS_OR_RECOVERY_CERTIFIED',
        }
        (temp/'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.rename(out)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    print(f'CREATED: {out} | states={len(p)} | splits={split_count}')
    print(f'CODEBOOK Dmin={selection["min_pairwise_normalized_g"]:.6g} >= {threshold:g}; '
          f'train/test exact endpoints; each split {bins}-bin coverage.')
    print('NOT RECOVERY CERTIFIED; continuous valid parameter domain remains non-identifiable in general.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=int, default=120)
    parser.add_argument('--pool-size', type=int, default=2048)
    parser.add_argument('--boundary-each', type=int, default=24)
    parser.add_argument('--min-distance', type=float, default=6.)
    parser.add_argument('--coverage-bins', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--train-replicas', type=int, default=3)
    parser.add_argument('--q2-file', default=None)
    parser.add_argument('--out', default='data_fivep_codebook_v1')
    args = parser.parse_args()
    build(Path(args.out), args.states, args.pool_size, args.boundary_each,
          args.min_distance, args.coverage_bins, args.seed, args.train_replicas,
          args.q2_file)


if __name__ == '__main__':
    main()
