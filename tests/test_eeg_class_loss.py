import unittest

import torch
import torch.nn.functional as F

from eeg_model import ClassPrototypes


class ClassPrototypesTest(unittest.TestCase):
    def test_top5_ranking_uses_the_fifth_wrong_class_and_excludes_paired_image(self):
        torch.manual_seed(11)
        images = F.normalize(torch.randn(80, 8, dtype=torch.double), dim=1)
        labels = torch.arange(8).repeat_interleave(10)
        bank = ClassPrototypes(images, labels)
        ids = torch.tensor([2, 22, 72])
        gallery = torch.tensor([2, 1, 3, 0, 4, 5, 6])
        eeg = F.normalize(torch.randn(3, 8, dtype=torch.double), dim=1).requires_grad_()
        actual = bank.top5_loss(eeg, images[ids], ids, gallery)
        expected = []
        for vector, index in zip(eeg, ids, strict=True):
            target = labels[index]
            others = [c for c in gallery if c != target][:len(gallery) - 1]
            negatives = torch.stack([F.normalize(images[labels == c].mean(0), dim=0) for c in others]) @ vector
            positive = F.normalize(images[(labels == target) & (torch.arange(len(labels)) != index)].mean(0),
                                   dim=0) @ vector
            expected.append(((negatives.sort(descending=True).values[4] - positive) / .07).relu())
        expected = torch.stack(expected).mean()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, eeg)[0], torch.autograd.grad(expected, eeg)[0])

    def test_loss_and_gradient_against_explicit_leave_one_out_means(self):
        torch.manual_seed(17)
        images = F.normalize(torch.randn(30, 8, dtype=torch.double), dim=1)
        labels = torch.tensor([4, 8, 12] * 10)
        bank = ClassPrototypes(images, labels)
        ids = torch.tensor([0, 1, 4, 11])
        eeg = torch.randn(4, 8, dtype=torch.double, requires_grad=True)
        actual = bank.loss(eeg, images[ids], ids)
        rows = []
        for vector, index in zip(eeg, ids):
            means = [images[(labels == label) & (torch.arange(30) != index)].mean(0)
                     for label in labels.unique()]
            rows.append(torch.stack(means) @ vector)
        expected = F.cross_entropy(torch.stack(rows) / 0.07, torch.searchsorted(labels.unique(), labels[ids]))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, eeg)[0], torch.autograd.grad(expected, eeg)[0])

    def test_incomplete_class_rejected(self):
        with self.assertRaisesRegex(ValueError, "ten"):
            ClassPrototypes(torch.ones(19, 8), torch.arange(19) // 10)
