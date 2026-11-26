import unittest

import numpy as np
import torch

from eeg_cross_subject import fresh_split
from eeg_model import COSINE_REGRESSION_CONFIG, TRANSFER_CONFIG, EEGEncoder
from eeg_train import seed_run, source_indices
from eeg_transfer_pilot import source_batches, target_state


class CrossSubjectTest(unittest.TestCase):
    def test_shared_pretraining_visits_each_image_once_per_subject(self):
        first = list(source_batches(11, 3, 4, 0, 5))
        replay = list(source_batches(11, 3, 4, 0, 5))
        self.assertEqual(len(first), 9)
        for (a, rows), (b, repeated) in zip(first, replay, strict=True):
            self.assertEqual(a, b)
            self.assertLessEqual(len(rows), 4)
            np.testing.assert_array_equal(rows, repeated)
        for subject in range(3):
            np.testing.assert_array_equal(np.sort(np.concatenate([rows for idx, rows in first if idx == subject])),
                                          np.arange(11))

    def test_identity_adapters_and_strict_target_checkpoint(self):
        eeg = torch.randn(2, 17, 80)
        seed_run(4)
        original = EEGEncoder(COSINE_REGRESSION_CONFIG).eval()
        seed_run(4)
        shared = EEGEncoder(TRANSFER_CONFIG, adapter_count=3).eval()
        target = EEGEncoder(TRANSFER_CONFIG).eval()
        with torch.no_grad():
            torch.testing.assert_close(shared(eeg, 1), original(eeg), atol=0, rtol=0)
            shared.electrode_adapters[1].weight[0, 2] = 0.3
            target.load_state_dict(target_state(shared.state_dict(), 1), strict=True)
            torch.testing.assert_close(target(eeg), shared(eeg, 1), atol=0, rtol=0)
        with self.assertRaises(KeyError):
            target_state(shared.state_dict(), 3)

    def test_balanced_source_schedule_and_resumption(self):
        for start in (1, 9, 17, 25, 33):
            block = np.stack([source_indices(14540, 8, 0, e) for e in range(start, start+8)])
            self.assertTrue(np.all(np.sort(block, axis=0) == np.arange(8)[:, None]))
            for epoch, row in enumerate(block, start):
                self.assertLessEqual(np.ptp(np.bincount(row)), 1)
                np.testing.assert_array_equal(row, source_indices(14540, 8, 0, epoch))

    def test_fresh_split_is_disjoint_and_templates_use_only_support(self):
        labels = np.repeat(np.arange(1654), 10)
        design = dict(train_labels=labels, train_images=np.arange(len(labels))[:, None],
                      dev_classes=np.arange(200))
        fresh = fresh_split(design)
        self.assertFalse(np.intersect1d(fresh["dev_classes"], design["dev_classes"]).size)
        self.assertFalse(np.intersect1d(labels[fresh["dev_train"]], fresh["dev_classes"]).size)
        np.testing.assert_array_equal(fresh["dev_centers"].ravel(), 10*fresh["dev_classes"]+2)
        third = fresh_split(design, 2028, exclude=fresh["dev_classes"])
        self.assertEqual(len(third["dev_classes"]), 200)
        self.assertEqual(len(third["dev_train"]), 14540)
        self.assertFalse(np.intersect1d(third["dev_classes"], np.r_[design["dev_classes"], fresh["dev_classes"]]).size)
        np.testing.assert_array_equal(third["dev_query"].reshape(200, 5),
                                      10 * third["dev_classes"][:, None] + np.arange(5, 10))
