import unittest
from types import SimpleNamespace

import numpy as np

import evaluate_baseline as e


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


class BaselineComparisonTests(unittest.TestCase):
    def test_baseline_clips_each_component_then_applies_offset(self):
        model = SimpleNamespace(models=[ConstantModel(-.002), ConstantModel(.0005)])
        bare = e.predict_batched_ms(model, np.zeros((1, 1)))
        self.assertEqual(bare.tolist(), [.5])
        # P3 padding 2048 has -1ms offset; old runtime clips the final result.
        self.assertEqual(e.with_legacy_offsets(bare, [2048], 3).tolist(), [0.])

    def test_main_summary_contains_only_holdout(self):
        rows = [dict(dataset='d', case='c64', stage=0, global_padded_tokens=8192,
                     split=split, target_ms=100., baseline_with_offset_ms=pred,
                     baseline_bare_ms=pred, candidate_ms=pred)
                for split, pred in [('train', 200.), ('holdout', 101.), ('gap', 300.)]]
        table=e.summarize(rows, holdout_only=True)
        self.assertTrue(table)
        self.assertTrue(all(r['n']==1 and r['bias_ms']==1. for r in table))


if __name__ == '__main__':
    unittest.main()
