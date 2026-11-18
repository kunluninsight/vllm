import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

import retrain_maxpcp as r


def fixture(stage=0):
    features = [dict(step=7, concurrency=64, **{k: 1.0 for k in r.ALL_FEATURES})]
    features[0]['global_padded_tokens'] = 2048
    events, metadata = {}, {}
    for pcp in range(8):
        # rank 1 starts earlier, takes longer, yet finishes with rank 0.
        start = 100_000_000 if pcp != 1 else 90_000_000
        end = 110_000_000
        events[pcp] = [dict(event=r.endpoint(stage), context=['run', 'scheduler', 7, 'real'],
                           chunk=0, rank=dict(pp=stage, pcp=pcp),
                           data=dict(start_mono_ns=start, observed_mono_ns=end))]
        metadata[pcp] = dict(collection_id='run', clock_domain='node',
                             host_clock=dict(source='time.monotonic_ns', boot_id='boot',
                                             time_namespace='ns'))
    return features, events, metadata


class AggregationTests(unittest.TestCase):
    def test_missing_rank_rejected(self):
        f, e, m = fixture()
        del e[7]
        with self.assertRaisesRegex(ValueError, 'PCP'):
            r.aggregate_stage(f, e, m, 0, 'fixture')

    def test_equal_step_in_another_context_does_not_pair(self):
        f, e, m = fixture()
        e[7][0]['context'][1] = 'other-scheduler'
        with self.assertRaisesRegex(ValueError, 'context'):
            r.aggregate_stage(f, e, m, 0, 'fixture')

    def test_max_local_duration_is_not_envelope(self):
        f, e, m = fixture()
        # rank 2 begins later and ends later, yielding a 25ms envelope,
        # but the longest local interval remains rank 1's 20ms.
        e[2][0]['data'] = dict(start_mono_ns=105_000_000, observed_mono_ns=115_000_000)
        row = r.aggregate_stage(f, e, m, 0, 'fixture')[0]
        self.assertEqual(row['target_ms'], 20.0)
        self.assertEqual(row['envelope_ms'], 25.0)
        self.assertEqual(row['max_pcp'], 1)
        self.assertEqual(row['start_skew_ms'], 15.0)

    def test_unverified_clock_does_not_invent_envelope(self):
        f, e, m = fixture()
        m[7]['host_clock']['boot_id'] = 'different-boot'
        row = r.aggregate_stage(f, e, m, 0, 'fixture')[0]
        self.assertEqual(row['target_ms'], 20.0)
        self.assertIsNone(row['envelope_ms'])
        self.assertIsNone(row['start_skew_ms'])

    def test_temporal_split_keeps_four_step_gap(self):
        rows = [dict(dataset='d', case='c64', stage=0, step=i, reference_start_ns=i)
                for i in range(30)]
        r.assign_splits(rows, holdout_fraction=.2, gap_steps=4)
        self.assertEqual([x['step'] for x in rows if x['split']=='train'], list(range(20)))
        self.assertEqual([x['step'] for x in rows if x['split']=='gap'], list(range(20,24)))
        self.assertEqual([x['step'] for x in rows if x['split']=='holdout'], list(range(24,30)))

    def test_direct_model_reload_matches_runtime(self):
        from sglang_simulator.simulation.vllm.tpu_pp.stage_model import StageModel
        rows = []
        for i in range(50):
            row = dict(dataset='fixture',case='c1',stage=0, split='train', target_ms=30+i/5,
                       **{k:float(i+1) for k in r.ALL_FEATURES})
            rows.append(row)
        bundle = r.fit_stage(rows, 0, n_estimators=5)
        self.assertEqual(len(bundle['models']), 1)
        self.assertEqual(bundle['label_aggregation'], 'max_pcp_local_duration')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'model.pkl'
            joblib.dump(bundle,p)
            loaded=StageModel.from_file(str(p),0,3)
            x=np.array([[row[k] for k in r.FEATURE_NAMES] for row in rows])
            expected=bundle['models'][0].predict(x)
            np.testing.assert_allclose([loaded.predict(row) for row in rows], expected,
                                       rtol=0,atol=1e-12)


if __name__ == '__main__':
    unittest.main()
