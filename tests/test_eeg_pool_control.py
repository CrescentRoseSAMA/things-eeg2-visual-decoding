import unittest

import torch

from eeg_model import EEGEncoder


class PoolControlTest(unittest.TestCase):
    def test_residual_projection_starts_as_the_control_and_can_learn(self):
        torch.manual_seed(7)
        control = EEGEncoder("ab_pool4").eval()
        control_rng = torch.get_rng_state()
        torch.manual_seed(7)
        candidate = EEGEncoder("ab_pool4_resproj").eval()
        self.assertTrue(torch.equal(torch.get_rng_state(), control_rng))
        self.assertEqual(sum(p.numel() for p in candidate.parameters()), 889353)
        for name, value in control.state_dict().items():
            torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)
        eeg = torch.randn(4, 17, 80)
        embeddings = candidate(eeg)
        torch.testing.assert_close(embeddings, control(eeg), rtol=0, atol=0)
        embeddings[:, 0].sum().backward()
        self.assertGreater(candidate.residual_projection[-1].weight.grad.abs().sum().item(), 0)

    def test_pointwise_mixer_starts_as_the_control_and_can_learn(self):
        torch.manual_seed(7)
        control = EEGEncoder("ab_pool4").eval()
        control_rng = torch.get_rng_state()
        torch.manual_seed(7)
        candidate = EEGEncoder("ab_pool4_pwmix").eval()
        self.assertTrue(torch.equal(torch.get_rng_state(), control_rng))
        self.assertEqual(sum(p.numel() for p in candidate.parameters()), 494697)
        torch.testing.assert_close(candidate.readout[-1].weight, control.readout[-1].weight, rtol=0, atol=0)
        eeg = torch.randn(4, 17, 80)
        embeddings = candidate(eeg)
        torch.testing.assert_close(embeddings, control(eeg), rtol=0, atol=0)
        embeddings[:, 0].sum().backward()
        self.assertGreater(candidate.readout[3].alpha.grad.abs().sum().item(), 0)

    def test_b_pool4_only_changes_pooling_and_head_width(self):
        torch.manual_seed(7)
        parent = EEGEncoder("b").eval()
        torch.manual_seed(7)
        short = EEGEncoder("b_pool4").eval()
        self.assertEqual(len(short.temporal), 1)
        self.assertEqual(short.readout[-1].in_features, 640)
        self.assertEqual(sum(p.numel() for p in short.parameters()), 493545)
        torch.testing.assert_close(short.temporal[0].weight, parent.temporal[0].weight)
        eeg = torch.randn(3, 17, 80)
        with torch.no_grad():
            torch.testing.assert_close(short(eeg), parent(eeg), rtol=2e-5, atol=2e-6)

    def test_average_windows_and_wide_head_match(self):
        torch.manual_seed(7)
        short = EEGEncoder("ab_pool4")
        torch.manual_seed(7)
        long = EEGEncoder("ab_pool8_stride4")
        for a, b in zip(short.parameters(), long.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        x = torch.arange(80.).reshape(1, 1, 1, 80)
        expected = torch.stack([x[..., max(0, 4*i-2):min(80, 4*i+6)].mean(-1) for i in range(20)], -1)
        torch.testing.assert_close(long.readout[3](x), expected)
        self.assertEqual(short.readout[3](x).shape, expected.shape)


if __name__ == "__main__":
    unittest.main()
