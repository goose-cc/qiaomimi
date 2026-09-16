from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _segment(seg: dict) -> np.ndarray:
    kind = str(seg['kind'])
    if kind == 'linear':
        q = np.linspace(float(seg['q2_min']), float(seg['q2_max']), int(seg['points']), dtype=np.float64)
    elif kind == 'geom_abs':
        amax, amin = float(seg['abs_max']), float(seg['abs_min'])
        if not (amax > amin > 0):
            raise ValueError('geom_abs requires abs_max > abs_min > 0')
        q = -np.geomspace(amax, amin, int(seg['points']), dtype=np.float64)
    else:
        raise ValueError(f'unknown observation segment kind: {kind}')
    return q


def make_q2(design: dict) -> np.ndarray:
    if design.get('kind') == 'hybrid':
        q = np.concatenate([_segment(s) for s in design['segments']])
    else:
        q = _segment(design)
    q = np.sort(np.asarray(q, dtype=np.float64))
    keep = np.ones(len(q), dtype=bool)
    if len(q) > 1:
        keep[1:] = np.diff(q) > 1e-12
    q = q[keep]
    if len(q) < 8 or not np.isfinite(q).all():
        raise ValueError('q2 design must contain >=8 finite points')
    if np.any(q >= 0.0):
        raise ValueError('CP02 keeps q2 in the Euclidean q2<0 domain')
    return q


def design_by_id(cfg: dict, design_id: str) -> dict:
    m = [d for d in cfg['observation_designs'] if d['id'] == design_id]
    if len(m) != 1:
        raise KeyError(f'design {design_id!r} not found uniquely')
    return m[0]


def describe(cfg: dict, design: dict) -> dict:
    q = make_q2(design)
    oq0, oq1 = cfg['original_q2_domain']
    return {
        'design_id': design['id'],
        'q2_points': len(q),
        'q2_min': float(q.min()),
        'q2_max': float(q.max()),
        'extends_below_original_q2_min': int(q.min() < float(oq0)-1e-12),
        'extends_above_original_q2_max': int(q.max() > float(oq1)+1e-12),
        'median_abs_q2_spacing': float(np.median(np.abs(np.diff(q)))),
        'nonuniform': int(np.std(np.diff(q)) > 1e-8 * max(abs(np.mean(np.diff(q))), 1.0)),
    }
