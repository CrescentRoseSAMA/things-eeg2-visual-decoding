import unittest

import numpy as np
import torch
import torch.nn.functional as F

from eeg_model import contrastive_loss
from eeg_visual_negatives import ImageNegatives, neighbors


class VisualNegativesTest(unittest.TestCase):
    def test_distinct_eligible_nearby_classes_and_epoch_replay(self):
        rng = np.random.default_rng(13)
        labels = np.tile(np.arange(50), 10)
        images = rng.normal(size=(len(labels), 8))
        order = rng.permutation(len(labels))
        nearest = neighbors(np.stack([images[labels == i].mean(0) for i in range(50)]))
        for mode in ("near", "random"):
            bank = ImageNegatives(images, labels, mode)
            ids = bank.sample(order, 32, 1, 6)
            np.testing.assert_array_equal(ids, bank.sample(order, 32, 1, 6))
            for start in range(0, len(order), 32):
                batch = order[start:start+32]
                self.assertFalse(np.isin(ids[batch], batch).any())
                for i in batch:
                    classes = labels[ids[i]]
                    self.assertEqual(len(np.unique(classes)), 16)
                    self.assertNotIn(labels[i], classes)
                    if mode == "near":
                        self.assertTrue(np.isin(classes, nearest[labels[i], :20]).all())

    def test_extra_logits_only_change_eeg_to_image(self):
        torch.manual_seed(3)
        eeg = F.normalize(torch.randn(4, 8), dim=1).requires_grad_()
        paired = F.normalize(torch.randn(4, 8), dim=1)
        extra = F.normalize(torch.randn(4, 16, 8), dim=2)
        logits = eeg @ paired.T / .07
        losses = []
        for i in range(4):
            candidates = torch.cat((paired, extra[i]))
            losses.append(torch.logsumexp(candidates @ eeg[i] / .07, 0) - logits[i,i])
        expected = (torch.stack(losses).mean() + F.cross_entropy(logits.T, torch.arange(4))) / 2
        actual = contrastive_loss(eeg, paired, extra)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual,eeg)[0], torch.autograd.grad(expected,eeg)[0])
