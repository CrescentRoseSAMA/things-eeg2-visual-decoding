import unittest

import numpy as np
import torch

from eeg_repeat_views import omitted_means, perturb_trial_epoch, sample_epoch, sample_subset_epoch
from eeg_train import make_weight_ema, train


class RepeatViewTest(unittest.TestCase):
    def test_single_trials_and_perturbations_are_epoch_replayable(self):
        raw = torch.tensor([1., 2., 4., 8.]).reshape(1, 4, 1, 1).expand(4096, -1, 3, 8)
        full = raw.mean(1)
        state = torch.get_rng_state()
        single_enabled = sample_subset_epoch(full, raw, 3, 8, min_trials=1)
        single_trial = (single_enabled[:, :1, :1].flatten()[:, None] == torch.tensor([1., 2., 4., 8.])).any(1)
        self.assertAlmostEqual(float(single_trial.float().mean()), .25, delta=.03)
        perturbed = perturb_trial_epoch(single_enabled, 3, 8)
        torch.testing.assert_close(perturbed, perturb_trial_epoch(single_enabled, 3, 8), atol=0, rtol=0)
        removed = (perturbed == 0).all(2)
        self.assertAlmostEqual(float(removed.float().mean()), .05, delta=.01)
        self.assertAlmostEqual(float((perturbed - single_enabled)[~removed].std()), .10, delta=.005)
        milder = perturb_trial_epoch(single_enabled, 3, 8, noise_std=.05)
        torch.testing.assert_close((milder == 0).all(2), removed, atol=0, rtol=0)
        torch.testing.assert_close((milder - single_enabled)[~removed],
                                   (perturbed - single_enabled)[~removed] / 2, atol=1e-6, rtol=0)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_ema_averages_weights_keeps_latest_buffers_and_resumes(self):
        model = torch.nn.BatchNorm1d(2)
        ema = make_weight_ema(model)
        ema.update_parameters(model)
        with torch.no_grad():
            model.weight.fill_(3)
            model.running_mean.fill_(5)
        ema.update_parameters(model)
        torch.testing.assert_close(ema.module.weight, torch.full((2,), 1.2))
        torch.testing.assert_close(ema.module.running_mean, torch.full((2,), 5.))
        restored = make_weight_ema(model)
        restored.load_state_dict(ema.state_dict(), strict=True)
        ema.update_parameters(model)
        restored.update_parameters(model)
        torch.testing.assert_close(restored.module.weight, ema.module.weight, atol=0, rtol=0)

    def test_trial_regularization_requires_raw_trials_and_enough_epochs_for_ema(self):
        for options in ({"single_trials": True}, {"perturb_trials": True}, {"weight_ema": True}):
            with self.assertRaises(ValueError):
                train("baseline", 1, 0, "unused", 5, 128, {}, **options)

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
