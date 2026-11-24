import unittest

import numpy as np

from eeg_data import condition_rows, development_split


class DataProtocolTests(unittest.TestCase):
    def test_rejects_row_permutation_even_when_lengths_match(self):
        with self.assertRaises(ValueError):
            condition_rows(["a", "a"], ["2.jpg", "1.jpg"], 1, 2)

    def test_rejects_duplicate_image(self):
        with self.assertRaises(ValueError):
            condition_rows(["a", "a"], ["1.jpg", "1.jpg"], 1, 2)

    def test_dev_templates_exclude_query_images_and_keep_mean_magnitude(self):
        classes = np.repeat(np.arange(1654), 10)
        images = np.zeros((len(classes), 2), np.float32)
        images[:, 0] = 1
        images[::10] = [0, 1]
        split = development_split(classes, images)
        self.assertEqual(len(split["dev_train"]), 14540)
        self.assertEqual(len(set(classes[split["dev_query"]])), 200)
        self.assertEqual(len(split["dev_query"]), 1000)
        self.assertFalse(set(classes[split["dev_train"]]) & set(classes[split["dev_query"]]))
        self.assertFalse(set(split["dev_support"]) & set(split["dev_query"]))
        np.testing.assert_allclose(split["dev_centers"][0], [0.8, 0.2])


if __name__ == "__main__":
    unittest.main()
