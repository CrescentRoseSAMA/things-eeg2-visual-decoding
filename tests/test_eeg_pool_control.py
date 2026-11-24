import unittest

import torch

from eeg_model import EEGEncoder


class PoolControlTest(unittest.TestCase):
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
