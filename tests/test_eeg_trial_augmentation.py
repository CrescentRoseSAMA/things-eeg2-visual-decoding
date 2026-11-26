import unittest

from eeg_trial_augmentation import gate
from eeg_gemini_iterations import PILOT
from eeg_csls_replication import PAIRS
from eeg_transfer_pilot import replication_gate


def rows_for(pairs, top1, top5):
    return [dict(subject=subject, seed=seed, queries=1000, top1_hits=70 + top1,
                 control_top1_hits=70, top5_hits=230 + top5, control_top5_hits=230,
                 parameters=889353) for subject, seed in pairs]


class TrialAugmentationGateTest(unittest.TestCase):
    def test_transfer_replication_gate_uses_both_seeds_and_four_subjects(self):
        rows = rows_for(PAIRS, 0, 12)
        self.assertTrue(replication_gate(rows))
        with self.assertRaises(ValueError):
            replication_gate(rows[:-1])
        rows[0]["top5_hits"] -= 1
        self.assertFalse(replication_gate(rows))
        rows[0]["top5_hits"] += 1
        rows[0]["top1_hits"] -= 1
        rows[1]["top1_hits"] += 1
        self.assertFalse(replication_gate(rows))
        rows[0]["top1_hits"] += 1
        rows[1]["top1_hits"] -= 1
        for row in rows:
            row["top5_hits"] = 230 + (24 if row["subject"] <= 7 else 0)
        self.assertFalse(replication_gate(rows))

    def test_soft_clip_pilot_allows_exact_top1_tolerance(self):
        rows = rows_for(PILOT, -2, 10)
        self.assertTrue(gate("pilot", rows, soft_clip=True))
        rows[0]["top1_hits"] -= 1
        self.assertFalse(gate("pilot", rows, soft_clip=True))
        rows[0]["top1_hits"] += 1
        rows[-1]["top5_hits"] -= 1
        self.assertFalse(gate("pilot", rows, soft_clip=True))
        for row, gain in zip(rows, (20, 20, 0, 0), strict=True):
            row["top5_hits"] = 230 + gain
        self.assertFalse(gate("pilot", rows, soft_clip=True))

    def test_pilot_allows_zero_top1_gain_and_requires_complete_pairs(self):
        rows = rows_for(PILOT, 0, 10)
        self.assertTrue(gate("pilot", rows))
        with self.assertRaises(ValueError):
            gate("pilot", rows[:-1])
        rows[0]["top1_hits"] -= 1
        self.assertFalse(gate("pilot", rows))
        rows[0]["top1_hits"] += 1
        for row, gain in zip(rows, (20, 20, 0, 0), strict=True):
            row["top5_hits"] = 230 + gain
        self.assertFalse(gate("pilot", rows))

    def test_replication_requires_top1_gain_and_no_losing_seed(self):
        rows = rows_for(PAIRS, 2, 10)
        self.assertTrue(gate("replication", rows))
        rows[0]["top1_hits"] -= 1
        self.assertFalse(gate("replication", rows))
        for row in rows:
            row["top1_hits"] = 69 if row["seed"] == 1 else 75
        self.assertFalse(gate("replication", rows))
