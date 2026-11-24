import unittest

import numpy as np

from eeg_cross_subject import fresh_split
from eeg_train import source_indices


class CrossSubjectTest(unittest.TestCase):
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
