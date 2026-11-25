import unittest
from itertools import product

from eeg_gemini_iterations import PILOT, contrast


def paired_rows(pairs, gain1, gain5):
    return [dict(subject=subject, seed=seed, queries=1000, top1_hits=70 + gain1,
                 control_top1_hits=70, top5_hits=220 + gain5, control_top5_hits=220,
                 parameters=493545) for subject, seed in pairs]


class GeminiGateTest(unittest.TestCase):
    def test_pilot_requires_gain_consistency_and_non_decreasing_top5(self):
        rows = paired_rows(PILOT, 5, 0)
        self.assertTrue(contrast(rows)["pilot_pass"])
        rows[0]["top1_hits"] -= 1
        self.assertFalse(contrast(rows)["pilot_pass"])
        rows[0]["top1_hits"] += 1
        rows[0]["top5_hits"] -= 1
        self.assertFalse(contrast(rows)["pilot_pass"])
        rows[0]["top5_hits"] += 1
        for row, hits in zip(rows, (80, 80, 70, 70), strict=True):
            row["top1_hits"] = hits
        self.assertFalse(contrast(rows)["pilot_pass"])

    def test_formal_requires_both_magnitudes_and_consistency(self):
        rows = paired_rows(product(range(5, 11), (1, 2)), 10, 20)
        self.assertTrue(contrast(rows)["formal_gate_pass"])
        rows[0]["top5_hits"] -= 1
        self.assertFalse(contrast(rows)["formal_gate_pass"])
        rows[0]["top5_hits"] += 1
        rows[0]["top1_hits"] -= 1
        self.assertFalse(contrast(rows)["formal_gate_pass"])
        for row in rows:
            row["top1_hits"] = 90 if row["seed"] == 1 else 70
        self.assertFalse(contrast(rows)["formal_gate_pass"])
        for row in rows:
            row["top1_hits"] = 90 if row["subject"] <= 7 else 70
        self.assertFalse(contrast(rows)["formal_gate_pass"])
