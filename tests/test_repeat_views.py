import unittest

import numpy as np
import torch

from eeg_repeat_views import omitted_means, sample_epoch, sample_subset_epoch


class RepeatViewTest(unittest.TestCase):
    def test_subset_means_are_uniform_by_size_and_replay_by_epoch(self):
        from itertools import combinations

        values = (1., 2., 4., 8.)
        raw = torch.tensor(values).reshape(1, 4, 1, 1).expand(4096, -1, -1, -1)
        full = torch.full((4096, 1, 1), 100.)
        initial = torch.get_rng_state()
        result = sample_subset_epoch(full, raw, seed=2, epoch=6)
        torch.testing.assert_close(initial, torch.get_rng_state())
        torch.testing.assert_close(result, sample_subset_epoch(full, raw, seed=2, epoch=6))
        self.assertFalse(torch.equal(result, sample_subset_epoch(full, raw, seed=2, epoch=5)))
        for size in (2, 3, 4):
            means = (100.,) if size == 4 else (sum(c) / size for c in combinations(values, size))
            matches = torch.isclose(result.flatten()[:, None], torch.tensor(tuple(means))).any(1)
            self.assertAlmostEqual(float(matches.float().mean()), 1 / 3, delta=0.03)

    def test_each_omission_matches_three_raw_trials(self):
        raw = np.random.default_rng(1).normal(size=(7, 4, 3, 8))
        got = omitted_means(raw)
        for omitted in range(4):
            expected = np.delete(raw, omitted, axis=1).mean(axis=1).astype(np.float32)
            np.testing.assert_array_equal(got[:, omitted], expected)

    def test_choices_resume_from_epoch_and_leave_other_rngs_untouched(self):
        full = torch.full((128, 1, 1), 4.)
        views = torch.arange(4.).reshape(1, 4, 1, 1).expand(128, -1, -1, -1)
        initial = torch.get_rng_state()
        result = sample_epoch(full, views, seed=2, epoch=6)
        torch.testing.assert_close(initial, torch.get_rng_state())
        self.assertEqual(set(result.flatten().tolist()), {0., 1., 2., 3., 4.})
        self.assertFalse(torch.equal(result, sample_epoch(full, views, seed=2, epoch=5)))
        torch.testing.assert_close(result, sample_epoch(full, views, seed=2, epoch=6))
