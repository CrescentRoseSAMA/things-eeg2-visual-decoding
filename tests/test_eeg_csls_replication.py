import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import numpy as np

import eeg_csls_replication as experiment
from eeg_csls_replication import (ADJUSTED, NORMALIZED, PAIRS, REFERENCE,
                                  csls_scores, formal_gate)


class CslsReplicationTest(unittest.TestCase):
    def test_formal_scoring_writes_the_paired_table_once(self):
        gallery = np.eye(200, dtype=np.float32)
        design = dict(train_images=gallery, test_centers=gallery, test_labels=np.arange(200))
        stats = dict(parameters=1, train_seconds=0, peak_gpu_mib=0)

        def load(path):
            return nullcontext(design) if path.name == "design.npz" else gallery

        def score_arms(*args):
            return {name: np.ones(200, dtype=int) for name in (experiment.CANDIDATE, NORMALIZED, ADJUSTED)}

        writer = Mock()
        with patch("builtins.print"), patch.object(experiment.np, "load", side_effect=load), patch.multiple(
                experiment, train=Mock(), freeze=Mock(), reload_and_score=Mock(), checked_metrics=Mock(),
                load_subject=Mock(), training_stats=Mock(return_value=stats), scoring_arms=score_arms,
                write_log=writer, save_json=Mock()):
            experiment.run_formal()
        self.assertEqual(writer.call_count, 1)
        self.assertEqual(len(writer.call_args.args[1]), 200)

    def test_csls_uses_training_neighbors_and_independent_query_offsets(self):
        query = np.array([[0.9, 0.8, 0.7], [0.1, 0.2, 0.3]])
        bank = np.array([[0.8, 0.1, 0.6], [0.9, 0.2, 0.1], [0.7, 0.3, 0.3]])
        expected = 2 * query - np.array([[0.85], [0.25]]) - np.array([0.85, 0.25, 0.45])
        np.testing.assert_allclose(csls_scores(query, bank, 2), expected)
        np.testing.assert_allclose(csls_scores(query[:1], bank, 2), expected[:1])

    def test_formal_gate_requires_complete_pairs_and_gain_over_normalization(self):
        rows = [dict(subject=s, seed=t, candidate=ADJUSTED, control=control, queries=1000,
                     top1_hits=80, control_top1_hits=70 if control == REFERENCE else 79,
                     top5_hits=260, control_top5_hits=240, parameters=889353)
                for s, t in PAIRS for control in (REFERENCE, NORMALIZED)]
        self.assertTrue(formal_gate(rows))
        with self.assertRaises(ValueError):
            formal_gate(rows[:-1])
        rows[0]["top5_hits"] -= 1
        self.assertFalse(formal_gate(rows))
        rows[0]["top5_hits"] += 1
        for row in rows[1::2]:
            row["control_top1_hits"] = 80
        self.assertFalse(formal_gate(rows))
