"""Small-scale real-physics V2 phase-one tests. Never train a network."""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from cp02_observation import design_by_id, make_q2
from fivep_v2_prepare import (
    load_historical_q2, generate_candidate_pool, audit_nearest,
    run_audit, run_pilot, pair_info, coverage_counts,
)
from validate_fivep_v2_pilot import validate


class FivepV2PhaseOneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # This is a temporary TEST FIXTURE from historical config. It must not
        # be confused with the original data_cp02_ml_hybrid_domain_v2/q2.npy.
        conf = json.loads((Path(__file__).parent/'cp02_corrected_3p_config.json').read_text())
        q = make_q2(design_by_id(conf,'hybrid_ultranear_240'))
        self.q2 = self.root/'q2_TEST_FIXTURE.npy'
        np.save(self.q2,q)
    def tearDown(self): self.tmp.cleanup()

    def test_historical_q2_dedup_239_and_bounds(self):
        q,m=load_historical_q2(self.q2)
        self.assertEqual(len(q),239)
        self.assertEqual((float(q[0]),float(q[-1])),(-150.,-.05))
        self.assertEqual(m['point_count'],239)
        meta=self.root/'metadata.json'
        meta.write_text(json.dumps({'q2_sha256':m['q2_values_sha256'],'q2_points':239}))
        self.assertTrue(load_historical_q2(self.q2)[1]['historical_metadata_checked'])
        meta.write_text(json.dumps({'q2_sha256':'tampered','q2_points':239}))
        with self.assertRaisesRegex(ValueError,'disagrees'):
            load_historical_q2(self.q2)
        meta.unlink()
        bad=self.root/'bad.npy';np.save(bad,np.linspace(-100,-6,100))
        with self.assertRaisesRegex(ValueError,'239/240'):
            load_historical_q2(bad)

    def test_candidate_sampling_and_alias_diagnostics(self):
        raw,physical,p,log=generate_candidate_pool(70, 20261012)
        self.assertEqual(p.shape,(70,5))
        self.assertGreaterEqual(len(raw),len(physical))
        self.assertGreaterEqual(len(physical),70)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.any(p[:,2] == -.05))
        self.assertTrue(np.any(p[:,4] == .01))
        self.assertTrue(np.any(p[:,4] == 1.))
        counts=coverage_counts({'candidate':raw,'accepted':physical},4)
        self.assertEqual(len(counts),2*5*4)

    def test_nearest_reports_label_conflict(self):
        p=np.array([[.10,.025,0,.8,.2],[.12,.025,0,.8,.2],[.14,.025,0,.8,.2]])
        g=np.array([[.10,.20,.30],[.10001,.20001,.30001],[.9,.8,.7]])
        rows,thresholds,diag=audit_nearest(p,g,n_neighbors=2,projection_dim=2,basis_size=3)
        self.assertEqual(len(rows),3)
        self.assertTrue(rows[0]['recovery_acceptance_windows_disjoint'])
        self.assertTrue(rows[0]['potential_alias'])
        self.assertFalse(diag['search_is_exhaustive'])
        self.assertEqual(len(thresholds),5)

    def test_end_to_end_train_val_pilot_and_tamper(self):
        audit=self.root/'audit'; pilot=self.root/'pilot'
        run_audit(audit,self.q2,96,20261012,4,16,8,96)
        run_pilot(pilot,audit,self.q2,80,20261013,4,3)
        result=validate(pilot,self.q2,forward_count=5)
        self.assertEqual(result['uniform_100']['q2_points'],100)
        self.assertEqual(result['hybrid_ultranear_240']['q2_points'],239)
        self.assertFalse((pilot/'hybrid_ultranear_240'/'noise_0p2pct'/'test.npz').exists())
        from train_fivep_compare_v2 import FivePDataset,load_norm,verify_train_val
        for design,points in [('uniform_100',100),('hybrid_ultranear_240',239)]:
            d=pilot/design
            norm=load_norm(d/'normalization_train_only.npz')
            train=FivePDataset(d/'noise_0p2pct'/'train.npz',norm)
            val=FivePDataset(d/'noise_0p2pct'/'val.npz',norm)
            verify_train_val(train,val)
            self.assertEqual(train.x.shape[1],points)
            self.assertEqual(train.y.shape[1],5)
        (pilot/'hybrid_ultranear_240'/'q2.npy').write_bytes(b'not a valid numpy array')
        with self.assertRaises(Exception): validate(pilot,self.q2,forward_count=0)
        with self.assertRaises(FileExistsError): run_pilot(pilot,audit,self.q2,80,20261013,4,3)


if __name__=='__main__':unittest.main(verbosity=2)
