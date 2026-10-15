"""Regression tests for independent held-out 5P codebook construction."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from build_fivep_blind import build, KIND, pick_parameters
from build_fivep_dataset import NAMES
from validate_fivep_blind import validate


class BlindFivePTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.path = Path(cls.workspace.name) / 'blind'
        build(cls.path, states=90, pool_size=512, boundary_each=12,
              coverage_bins=3, min_distance=6., seed=20261012,
              train_replicas=2)

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def test_integrity_separation_and_unseen_state(self):
        report = validate(self.path, forward_count=0)
        self.assertIn('PASS', report['status'])
        self.assertGreaterEqual(report['train_g_min_distance_512'], 6.)
        with (self.path/'state_manifest.csv').open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        all_by_split = {s: set() for s in ('train', 'val', 'test')}
        for r in rows:
            all_by_split[r['split']].add(r['physical_state_id'])
        for a in all_by_split:
            for b in all_by_split:
                if a != b:
                    self.assertFalse(all_by_split[a] & all_by_split[b])
        self.assertEqual(len(all_by_split['train']), 62)
        self.assertEqual(len(all_by_split['val']), 14)
        self.assertEqual(len(all_by_split['test']), 14)
        with np.load(self.path/'noise_0p2pct'/'train.npz', allow_pickle=False) as z:
            self.assertEqual(z['gy'].shape, (124, 100))
            self.assertEqual(z['parameters'].shape, (124, 5))
        m = json.loads((self.path/'metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(m['dataset_kind'], KIND)
        self.assertTrue(m['blind_protocol']['validation_test_g_not_used_to_select_states'])
        self.assertFalse(m['sampler']['test']['selection_uses_candidate_g'])
        self.assertFalse(m['sampler']['val']['selection_uses_candidate_g'])

    def test_reject_existing_output(self):
        with self.assertRaises(FileExistsError):
            build(self.path, states=90, pool_size=512, boundary_each=12,
                  coverage_bins=3, min_distance=6.)

    def test_reject_loosened_g_threshold(self):
        impossible = Path(self.workspace.name) / 'rejected'
        with self.assertRaises(ValueError):
            build(impossible, states=90, pool_size=512,
                  boundary_each=12, coverage_bins=3,
                  min_distance=1.)
        self.assertFalse(impossible.exists())

    def test_tampered_recovery_gate_rejected(self):
        f = self.path/'metadata.json'
        original = f.read_bytes()
        try:
            meta = json.loads(original)
            meta['recovery_tolerances']['m_abs'] = 0.3
            f.write_text(json.dumps(meta), encoding='utf-8')
            with self.assertRaises(ValueError):
                validate(self.path, forward_count=2)
        finally:
            f.write_bytes(original)

    def test_protocol_tampering_detected(self):
        f = self.path/'metadata.json'
        original = f.read_bytes()
        try:
            meta = json.loads(original)
            meta['sampler']['test']['selection_uses_candidate_g'] = True
            f.write_text(json.dumps(meta), encoding='utf-8')
            with self.assertRaises(ValueError):
                validate(self.path, forward_count=2)
        finally:
            f.write_bytes(original)

    def test_test_holdout_does_not_require_g_or_use_threshold(self):
        rng = np.random.default_rng(3)
        p = np.tile([.12, .025, 0, .8, .2], (128,1)).astype(float)
        # A heldout candidate filter MUST select without any g and can do so
        # even when a deliberately ridiculous TRAIN-only distance is provided.
        p[:,0] = rng.uniform(.05, .2, len(p))
        p[:,1] = rng.uniform(0, .05, len(p))
        p[:,2] = rng.uniform(-.04, .04, len(p))
        p[:,3] = rng.uniform(.4, 1.2, len(p))
        p[:,4] = rng.uniform(.01, 1., len(p))
        result = pick_parameters(p, 24, 3, seed=4,
                                 exact_endpoints=False, curves=None,
                                 threshold=1e100)
        self.assertEqual(len(result), 24)
        self.assertEqual(len(set(result.tolist())), 24)


if __name__ == '__main__':
    unittest.main(verbosity=2)
