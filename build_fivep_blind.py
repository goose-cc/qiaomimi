#!/usr/bin/env python3
"""Build a genuinely held-out 5P dataset, without test-g-informed train selection.

The training codebook is selected using TRAIN candidates only. Validation and
Test have independent candidate seeds and are chosen using PHYSICAL PARAMETERS
for predeclared marginal coverage, never their g values or recovery outcomes.
All physical bounds, q2, noise and recovery gates are inherited unchanged.
This does NOT imply recovery performance on unseen states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from build_fivep_codebook import candidates, bin_assignments, LOW, HIGH
from build_fivep_dataset import (
    BOUNDS, NAMES, RECOVERY, REF_NOISE, FORMULA, INTEGRATION_POINTS,
    Q2_COUNT, assert_source_contract, q2_grid, generate_curves,
    state_identity, write_examples, write_manifest, write_normalization,
    fingerprint,
)

SPLITS = ('train', 'val', 'test')
KIND = 'FIVEP_INDEPENDENT_HELDOUT_V1'


def quotas_for(total: int) -> dict[str, int]:
    if total < 60:
        raise ValueError('At least 60 states required')
    hold = max(12, int(round(0.15 * total)))
    return {'train': total - 2*hold, 'val': hold, 'test': hold}


def _rows_key(p: np.ndarray) -> list[bytes]:
    return [row.astype('<f8').tobytes() for row in p]


def pick_parameters(pool: np.ndarray, count: int, bins: int, seed: int,
                    *, exact_endpoints: bool, curves: np.ndarray | None,
                    threshold: float) -> np.ndarray:
    """Select by p coverage; TRAIN also enforces g-separation inside TRAIN.

    CRITICAL: For validation and test, curves is None. No g is evaluated or
    consulted before those states and their split assignments are locked.
    """
    if count <= 0 or len(pool) < count or bins < 2 or bins > 10:
        raise ValueError('Invalid quota/pool or bins')
    if curves is not None and curves.shape != (len(pool), Q2_COUNT):
        raise ValueError('Wrong TRAIN physical observation shape')
    groups = bin_assignments(pool, bins)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    used = np.zeros(len(pool), dtype=bool)
    counts = np.zeros((5, bins), dtype=np.int32)
    noise = (REF_NOISE*np.sqrt(np.mean(curves**2, axis=1))) if curves is not None else None
    nearest = np.full(len(pool), np.inf)
    if noise is not None and (np.any(noise <= 0) or not np.isfinite(noise).all()):
        raise RuntimeError('TRAIN has invalid noise scales')

    def pick(mask: np.ndarray, missing: set[tuple[int,int]] | None = None) -> None:
        allowed = np.flatnonzero(mask & (~used) & (nearest >= threshold))
        if not len(allowed):
            raise RuntimeError('No state meets frozen coverage/separation policy; increase candidate pool')
        sub = groups[allowed]
        scarcity = np.sum(1. / (1. + counts[np.arange(5)[:, None], sub.T]), axis=0)
        score = 10 * scarcity + rng.random(len(allowed))*1e-6
        if missing:
            gain = np.array([sum((j, int(sub[i,j])) in missing for j in range(5))
                             for i in range(len(allowed))], dtype=np.float64)
            score += 100 * gain
        selected = int(allowed[np.argmax(score)])
        used[selected] = True
        chosen.append(selected)
        counts[np.arange(5), groups[selected]] += 1
        if curves is not None:
            ds = np.linalg.norm(curves - curves[selected], axis=1) / np.maximum(noise, noise[selected])
            nearest[:] = np.minimum(nearest, ds)
            nearest[used] = -np.inf

    if exact_endpoints:
        for j in (2, 4, 0, 3, 1):
            for endpoint in (LOW[j], HIGH[j]):
                if chosen and np.any(pool[np.asarray(chosen), j] == endpoint):
                    continue
                pick(pool[:, j] == endpoint)
    need = {(j, b) for j in range(5) for b in range(bins)}
    for idx in chosen:
        need.difference_update((j, int(groups[idx, j])) for j in range(5))
    while need:
        if len(chosen) >= count:
            raise RuntimeError('Quota exhausted before parameter bin coverage')
        mask = np.array([any((j, int(groups[i, j])) in need for j in range(5))
                         for i in range(len(pool))], dtype=bool)
        pick(mask, need)
        idx = chosen[-1]
        need.difference_update((j, int(groups[idx, j])) for j in range(5))
    while len(chosen) < count:
        pick(np.ones(len(pool), dtype=bool))
    result = np.asarray(chosen, dtype=int)
    if len(result) != count:
        raise RuntimeError('Wrong selected state count')
    if exact_endpoints:
        if any(not (np.any(pool[result,j] == LOW[j]) and np.any(pool[result,j] == HIGH[j]))
               for j in range(5)):
            raise RuntimeError('Missing locked marginal parameter endpoint')
    if curves is not None:
        # Verify selected TRAIN g separation; TEST g is deliberately absent here.
        from scipy.spatial.distance import pdist, squareform
        dg = squareform(pdist(curves[result]))
        scale = np.maximum(noise[result,None], noise[None,result])
        np.fill_diagonal(dg, np.inf)
        if float(np.min(dg/scale)) < threshold*(1-1e-8):
            raise RuntimeError('Insufficient training-codebook separation')
    return result


def build(out: Path, states: int = 256, pool_size: int = 4096,
          boundary_each: int = 24, coverage_bins: int = 4,
          min_distance: float = 6.0, seed: int = 20261012,
          train_replicas: int = 3, q2_file: str | None = None) -> dict:
    assert_source_contract()
    quotas = quotas_for(states)
    if (pool_size < quotas['train'] or boundary_each < 2 or
            not (2 <= coverage_bins <= 10) or
            not (np.isfinite(min_distance) and min_distance >= 6.) or
            not (1 <= train_replicas <= 100)):
        raise ValueError('Invalid configuration; cannot relax frozen noise or g-separation floor')
    q2 = q2_grid(q2_file)
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f'Output exists: {out}; will not overwrite')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='._fivep_blind_', dir=out.parent))
    try:
        selected: dict[str, np.ndarray] = {}
        sampling = {}
        used_keys: set[bytes] = set()
        pool_seeds = {'train': seed, 'val': seed + 100003, 'test': seed + 200003}
        # The train codebook is selected and frozen BEFORE val/test pools exist.
        for name in SPLITS:
            p, log = candidates(pool_size if name == 'train' else max(512, 8*quotas[name]),
                                boundary_each, pool_seeds[name])
            original_len = len(p)
            keep = np.asarray([key not in used_keys for key in _rows_key(p)], dtype=bool)
            p = p[keep]
            if len(p) < quotas[name]:
                raise RuntimeError(f'Insufficient unique {name} candidates')
            # No evaluation of g for Val or Test until AFTER parameter selection.
            curves = generate_curves(p, q2) if name == 'train' else None
            indices = pick_parameters(
                p, quotas[name], coverage_bins, pool_seeds[name] + 57,
                exact_endpoints=name in ('train', 'test'),
                curves=curves, threshold=min_distance,
            )
            selected[name] = p[indices]
            used_keys.update(_rows_key(selected[name]))
            sampling[name] = {'candidate_pool_before_dedup': original_len,
                              'candidate_pool_after_dedup': len(p),
                              'seed': pool_seeds[name],
                              'selection_uses_candidate_g': name == 'train',
                              'sampler_log': log}
        p = np.vstack([selected[name] for name in SPLITS])
        split = np.concatenate([np.full(len(selected[name]), name, dtype='<U5') for name in SPLITS])
        assert len(p) == states and len(set(_rows_key(p))) == states
        clean = generate_curves(p, q2)  # AFTER all split selections are frozen
        identities = [state_identity(row) for row in p]
        ids = np.asarray([it[0] for it in identities], dtype='<U35')
        number_ids = np.asarray([it[1] for it in identities], dtype=np.int64)
        if len(set(ids)) != states or len(set(map(int, number_ids))) != states:
            raise RuntimeError('Physical-state-ID collision')
        write_manifest(tmp/'state_manifest.csv', p, split, ids, number_ids)
        np.save(tmp/'q2.npy', q2)
        for name, shift in (('train', 11), ('val', 23), ('test', 37)):
            mask = split == name
            write_examples(tmp/'noise_0p2pct'/f'{name}.npz', p[mask], clean[mask],
                           ids[mask], number_ids[mask], name, q2, REF_NOISE,
                           train_replicas if name == 'train' else 1, seed+shift)
        t = split == 'test'
        for dirname, lev, shift in (('noise_0pct', 0., 41), ('noise_1pct', .01, 43)):
            write_examples(tmp/dirname/'test.npz', p[t], clean[t],
                           ids[t], number_ids[t], 'test', q2, lev, 1, seed+shift)
        write_normalization(tmp)
        # Diagnostics are reported after selection, NEVER used as a rejection criterion.
        from scipy.spatial.distance import cdist, pdist
        gtrain = clean[split == 'train']
        sig = REF_NOISE*np.sqrt(np.mean(gtrain*gtrain, axis=1))
        d_train = pdist(gtrain)
        den = np.maximum(sig[np.triu_indices(len(sig), k=1)[0]],
                         sig[np.triu_indices(len(sig), k=1)[1]])
        dmin_train = float(np.min(d_train/den))
        if dmin_train < min_distance*(1-1e-8):
            raise RuntimeError('TRAIN Dmin failed after final forward integration')
        source = Path(__file__).resolve().parent
        metadata = {
            'dataset_kind': KIND,
            'stage': '5P', 'parameter_names': list(NAMES),
            'free_parameters': list(NAMES), 'fixed_parameters': {'a2': None, 'a3': None},
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
            'sample_counts': quotas,
            'split_rule': 'Train-only g selection FIRST; independent Val/Test candidate seeds; Val/Test predeclared parameter-only marginal coverage; test g NEVER used to select train or test states',
            'sampler': sampling, 'seed': seed,
            'q2_design': 'user_supplied' if q2_file else 'uniform_100',
            'blind_protocol': {
                'train_g_min_distance': dmin_train,
                'train_g_threshold': min_distance,
                'coverage_bins': coverage_bins,
                'train_test_g_separation_is_not_a_selection_gate': True,
                'validation_test_g_not_used_to_select_states': True,
                'frozen_data_note': 'Do not tune generation/reject hard test cases based on test recovery; no continuous-domain identifiability claim',
            },
            'source_sha256': {'mc_physics.py': fingerprint(source/'mc_physics.py'),
                              'mc_pool_config.py': fingerprint(source/'mc_pool_config.py')},
            'scientific_status': 'PHYSICS_AND_GROUPED_HOLDOUT_ONLY__RECOVERY_NOT_CERTIFIED',
        }
        (tmp/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.rename(out)
        print(f'CREATED: {out} | states={states} | splits={quotas} | train Dmin={dmin_train:.5g}')
        print('BLIND-SPLIT: g only used for TRAIN selection. Val/Test chosen by parameter coverage and independent seeds.')
        print('NOT RECOVERY CERTIFIED; no network training.')
        return metadata
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--states', type=int, default=256)
    a.add_argument('--pool-size', type=int, default=4096)
    a.add_argument('--boundary-each', type=int, default=24)
    a.add_argument('--coverage-bins', type=int, default=4)
    a.add_argument('--min-distance', type=float, default=6.)
    a.add_argument('--seed', type=int, default=20261012)
    a.add_argument('--train-replicas', type=int, default=3)
    a.add_argument('--q2-file', default=None)
    a.add_argument('--out', default='data_fivep_blind_v1')
    args = a.parse_args()
    build(Path(args.out), args.states, args.pool_size, args.boundary_each,
          args.coverage_bins, args.min_distance, args.seed, args.train_replicas,
          args.q2_file)


if __name__ == '__main__':
    main()
