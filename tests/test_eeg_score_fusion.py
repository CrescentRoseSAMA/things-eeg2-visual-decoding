import unittest

from eeg_csls_replication import PAIRS
from eeg_score_fusion import fusion_gate


class ScoreFusionTest(unittest.TestCase):
    def test_gate_requires_top5_gain_and_broad_improvement_without_top1_regression(self):
        rows = [dict(subject=subject, seed=seed, queries=1000, top1_hits=80, control_top1_hits=80,
                     top5_hits=295, control_top5_hits=280) for subject, seed in PAIRS]
        self.assertTrue(fusion_gate(rows))
        with self.assertRaises(ValueError):
            fusion_gate(rows[:-1])
        rows[0]["top5_hits"] -= 1
        self.assertFalse(fusion_gate(rows))
        rows[0]["top5_hits"] += 1
        rows[0]["top1_hits"] -= 1
        self.assertFalse(fusion_gate(rows))
        rows[0]["top1_hits"] += 1
        for row in rows:
            row["top5_hits"] = 280 + (30 if row["subject"] <= 7 else 0)
        self.assertFalse(fusion_gate(rows))
