import unittest

from eeg_class_alignment import gate


class AdvancementTest(unittest.TestCase):
    def test_pilot_requires_each_top1_and_allows_equal_mean_top5(self):
        rows = [dict(gain1=1, gain5=-1), dict(gain1=1, gain5=1)]
        self.assertTrue(gate("pilot", rows))
        rows[0]["gain1"] = 0
        self.assertFalse(gate("pilot", rows))

    def test_replication_threshold_and_consistency(self):
        rows = [dict(subject=s, seed=t, gain1=5, gain5=0)
                for s in range(3, 11) for t in (1, 2)]
        self.assertTrue(gate("replication", rows))
        rows[0]["gain1"] -= 1
        self.assertFalse(gate("replication", rows))
        for row in rows:
            row["gain1"] = 10 if row["seed"] == 1 else 0
        self.assertFalse(gate("replication", rows))
        for row in rows:
            row["gain1"] = 10 if row["subject"] <= 6 else 0
        self.assertFalse(gate("replication", rows))
        for row in rows:
            row["gain1"] = 5
        rows[0]["gain5"] = -1
        self.assertFalse(gate("replication", rows))
