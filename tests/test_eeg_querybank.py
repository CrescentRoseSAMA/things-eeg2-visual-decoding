import unittest

import numpy as np

from eeg_querybank import qb_scores, querybank_statistics


class QuerybankScoresTest(unittest.TestCase):
    def test_bank_statistics_and_dynamic_gate_are_per_query(self):
        bank = np.array([[0.30, 0.12, 0.01], [0.15, 0.26, 0.00], [0.25, 0.16, 0.04]])
        query = np.array([[0.35, 0.34, 0.01], [0.12, 0.10, 0.20]])
        stats = querybank_statistics(bank)
        np.testing.assert_array_equal(stats["training_top1"], [2, 1, 0])
        expected_denominator = np.log(np.exp(20 * bank).sum(axis=0))
        np.testing.assert_allclose(stats["log_denominator"], expected_denominator)
        adjusted, used = qb_scores(query, stats)
        np.testing.assert_array_equal(used, [True, False])
        np.testing.assert_allclose(adjusted[0], 20 * query[0] - expected_denominator)
        np.testing.assert_array_equal(adjusted[1], query[1])
        np.testing.assert_array_equal(qb_scores(query[:1], stats)[0][0], adjusted[0])
