import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from eeg_evaluate import aggregate, collect, ranks_and_accuracy


class EvaluationTests(unittest.TestCase):
    def test_ranking_and_permuted_template_labels(self):
        scores = np.tile([6, 5, 4, 3, 2, 1], (3, 1))
        labels = np.array([0, 4, 5])
        ranks, top1, top5 = ranks_and_accuracy(scores, labels)
        np.testing.assert_array_equal(ranks, [1, 5, 6])
        self.assertAlmostEqual(top1, 1 / 3)
        self.assertAlmostEqual(top5, 2 / 3)
        perm = np.array([1, 2, 4, 0, 5, 3])
        ranks, _, _ = ranks_and_accuracy(scores[:, perm], np.argsort(perm)[labels])
        np.testing.assert_array_equal(ranks, [1, 5, 6])

    def test_invalid_scores_and_labels(self):
        with self.assertRaises(ValueError):
            ranks_and_accuracy(np.array([[0, 1, np.nan, 4, 5]]), [3])
        with self.assertRaises(ValueError):
            ranks_and_accuracy(np.ones((1, 5)), [5])

    def test_partial_directory_does_not_count(self):
        with TemporaryDirectory() as temp:
            run = Path(temp) / "ab/sub-01/seed-0"
            run.mkdir(parents=True)
            np.save(run / "scores.npy", np.eye(200, dtype=np.float32))
            rows, pending = collect(Path(temp))
            self.assertFalse(rows)
            self.assertEqual(len(pending), 200)

    def test_paired_effect_and_seed_variation(self):
        rows = []
        for group, k in (("baseline", 0), ("a", 1), ("b", 2), ("ab", 3)):
            for subject in range(1, 11):
                for seed in range(5):
                    rows.append(dict(config=group, subject=subject, seed=seed,
                                     top1=(subject + seed + k) / 100,
                                     top5=(20 + seed + k) / 100, parameters=200,
                                     train_seconds=10, peak_gpu_mib=100))
        overall, subjects, seeds, deltas = aggregate(rows)
        self.assertEqual(len(overall), 4)
        self.assertEqual(len(subjects), 40)
        self.assertEqual(len(seeds), 20)
        self.assertEqual(len(deltas), 100)
        self.assertAlmostEqual(overall[-1]["top1_pct"], 10.5)
        self.assertAlmostEqual(overall[-1]["top1_seed_sd_pp"], np.std(range(5), ddof=1))
        self.assertTrue(all(np.isclose(row["top1_delta_pp"], 1)
                            for row in deltas if row["comparison"] == "ab-b"))
        self.assertFalse(aggregate(rows[:-1])[0])


if __name__ == "__main__":
    unittest.main()
