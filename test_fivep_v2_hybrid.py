#!/usr/bin/env python3
"""V2 test fixture uses a GENERATED q2, not the real historical file.

Production CLI never accepts this fixture SHA: it is locked to the true CP02
historical byte SHA-256. The internal build/validate test API overrides the
expected hash only for unit tests.
"""
from __future__ import annotations
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_fivep_v2_hybrid import (
    ALIAS_D_THRESHOLD, build, read_and_check_audit, selected_train_conflicts,
    independent_holdout_candidates, extend_missing_train_endpoints, select_parameter_coverage,
)
from fivep_v2_prepare import run_audit, sha_raw
from validate_fivep_v2_hybrid import validate


class FormalHybridDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.tmp.name)
        cls.q2=cls.root/'test_fixture_q2.npy'
        q=np.unique(np.concatenate([
            np.linspace(-150.,-20.,48,dtype=np.float64),
            -np.geomspace(20.,.05,192,dtype=np.float64),
        ]))
        assert len(q)==239 and q[0]==-150 and q[-1]==-.05
        np.save(cls.q2,q)
        cls.sha=sha_raw(cls.q2)
        cls.audit=cls.root/'audit'
        run_audit(cls.audit,cls.q2,512,20261019,10,32,12,256)
        cls.dest=cls.root/'dataset'
        build(cls.dest,cls.audit,cls.q2,train_states=210,val_states=60,
              test_states=60,seed=20261021,expected_q2_sha=cls.sha)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_dataset_integrity_and_noise_and_independent_forward(self):
        result=validate(self.dest,self.q2,expected_q2_sha=self.sha,forward_count=8)
        self.assertEqual(result['q2_points'],239)
        self.assertEqual(result['split_counts'],{'train':210,'val':60,'test':60})
        self.assertEqual(result['sample_counts']['train'],630)
        self.assertFalse(result['alias_screening']['global_pairwise_D6_required'])
        self.assertEqual(result['blind_split']['holdout_selection_uses_g'],False)

    def test_no_overwrite_original(self):
        with self.assertRaises(FileExistsError):
            build(self.dest,self.audit,self.q2,train_states=210,val_states=60,
                  test_states=60,expected_q2_sha=self.sha)

    def test_production_q2_is_byte_locked(self):
        # The fixture may have exactly the same bytes as the historical grid
        # on some NumPy/platform versions. Deliberately alter an interior point.
        tampered_q2 = self.root / 'tampered_q2.npy'
        q = np.load(self.q2, allow_pickle=False).copy()
        q[100] = np.nextafter(q[100], np.inf)
        np.save(tampered_q2, q)
        self.assertNotEqual(sha_raw(tampered_q2),
                            '2624d6cf24873a0e4c60f7ba19e2239a0a5ec961c04badf8056e1f1568dc653d')
        with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):
            build(self.root/'bad_sha',self.audit,tampered_q2,train_states=210,val_states=60,
                  test_states=60)
        self.assertFalse((self.root/'bad_sha').exists())

    def test_state_split_and_coverage_report(self):
        from collections import Counter
        with (self.dest/'state_manifest.csv').open(encoding='utf-8', newline='') as f:
            manifest=list(csv.DictReader(f))
        self.assertEqual(len(set(item['physical_state_id'] for item in manifest)),330)
        sizes=Counter(item['split'] for item in manifest)
        self.assertEqual(sizes,{'train':210,'val':60,'test':60})
        with (self.dest/'coverage_v2.csv').open(encoding='utf-8-sig',newline='') as f:
            coverage=list(csv.DictReader(f))
        groups=set(item['group'] for item in coverage)
        self.assertTrue({'candidate','accepted_physical','train','val','test'}.issubset(groups))
        self.assertEqual(sum(1 for c in coverage if c['group']=='test' and c['parameter']=='gamma'),10)
        for split in ('train','val','test'):
            per=[int(c['count']) for c in coverage if c['group']==split]
            self.assertGreaterEqual(min(per),int(np.ceil(sizes[split]*.02)))

    def test_normal_close_neighbors_are_not_mistaken_for_alias(self):
        p=np.array([[.125,.025,0.,.8,.2],[.125000001,.025,0.,.8,.2],
                    [.17,.025,.01,1.1,.7]],dtype=np.float64)
        g=np.array([[.1,.2,.3],[.1+1e-8,.2,.3],[1.,2.,3.]])
        rejects,diag=selected_train_conflicts(p,g,n_neighbors=2,project=2)
        self.assertEqual(rejects,set())
        self.assertGreater(diag['close_compatible_under_6_directed'],0)

    def test_disjoint_alias_identified_without_global_D6_cutoff(self):
        # Two inverse-incompatible truths with almost identical g are flagged,
        # while the close same-truth neighborhood in the previous test is kept.
        p=np.array([[.10,.025,0,.8,.2],[.14,.025,0,.8,.2],
                    [.18,.040,.03,1.15,.9]],dtype=np.float64)
        g=np.array([[.1,.2,.3],[.10000001,.2,.3],[1.,2.,3.]])
        bad,diag=selected_train_conflicts(p,g,n_neighbors=2,project=2)
        self.assertIn(1,bad)
        self.assertGreater(diag['disjoint_under_6_directed'],0)

    def test_alias_screened_endpoint_variants_restore_full_range(self):
        p=np.load(self.audit/'candidate_parameters.npy',allow_pickle=False)
        g=np.load(self.audit/'candidate_g_clean.npy',allow_pickle=False)
        q=np.load(self.q2,allow_pickle=False)
        blocked=np.zeros(len(p),bool)
        # Force risk at the *only* original exact a3=0.05 and gamma=0.01 anchors.
        blocked |= (p[:,2]==.05) | (p[:,4]==.01)
        expanded,curves,ban,log=extend_missing_train_endpoints(p,g,blocked,q,20261019)
        self.assertEqual(log['extra_endpoint_states'],6)
        self.assertTrue(np.any((expanded[:,2]==.05)&~ban))
        self.assertTrue(np.any((expanded[:,4]==.01)&~ban))
        selected=select_parameter_coverage(expanded,120,10,.02,20261021,~ban)
        self.assertTrue(np.any(expanded[selected,2]==.05))
        self.assertTrue(np.any(expanded[selected,4]==.01))

    def test_protects_report_false_negative(self):
        copy=self.root/'tampered_audit'
        copy.mkdir(exist_ok=True)
        for f in self.audit.iterdir():
            (copy/f.name).write_bytes(f.read_bytes())
        file=copy/'nearest_g_neighbors.csv'
        with file.open(encoding='utf-8-sig',newline='') as fp:
            r=list(csv.DictReader(fp))
        r[0]['recovery_acceptance_windows_disjoint']='not True'
        with file.open('w',encoding='utf-8-sig',newline='') as fp:
            wr=csv.DictWriter(fp,fieldnames=list(r[0])); wr.writeheader();wr.writerows(r)
        with self.assertRaisesRegex(ValueError,'disjoint flag'):
            read_and_check_audit(copy,self.sha)

    def test_metadata_recovery_tamper_rejected(self):
        import shutil
        tmp=self.root/'corrupted_metadata'
        shutil.copytree(self.dest,tmp)
        mp=tmp/'metadata.json'
        data=json.loads(mp.read_text(encoding='utf-8'))
        data['recovery_tolerances']['a1_abs']=.05
        mp.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'Recovery policy changed'):
            validate(tmp,self.q2,expected_q2_sha=self.sha,forward_count=0)

    def test_holdout_no_train_key_reuse(self):
        from validate_fivep_v2_hybrid import read_manifest
        p,split,ids,sid=read_manifest(self.dest/'state_manifest.csv')
        self.assertEqual(len(set(ids)),len(ids))
        self.assertEqual(len(set(ids[split=='train']) & set(ids[split=='test'])),0)
        self.assertEqual(len(set(ids[split=='val']) & set(ids[split=='test'])),0)

    def test_requires_pool_large_enough(self):
        with self.assertRaisesRegex(RuntimeError,'fewer states'):
            build(self.root/'too_large',self.audit,self.q2,train_states=1000,val_states=60,
                  test_states=60,expected_q2_sha=self.sha)


if __name__=='__main__': unittest.main(verbosity=2)
