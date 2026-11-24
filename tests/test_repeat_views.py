import unittest

import numpy as np
import torch

from eeg_repeat_views import omitted_means, sample_epoch


class RepeatViewTest(unittest.TestCase):
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
