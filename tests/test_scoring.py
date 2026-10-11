import unittest
from src.verifier.scorer import calculate_score, meets_threshold, WEIGHTS


class ScoringTests(unittest.TestCase):
    def test_all_ones(self):
        self.assertAlmostEqual(calculate_score(1, 1, 1, 1, 1), 1.0)

    def test_all_zeros(self):
        self.assertAlmostEqual(calculate_score(0, 0, 0, 0, 0), 0.0)

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0)

    def test_threshold_boundary(self):
        self.assertTrue(meets_threshold(0.80, 0.80))
        self.assertFalse(meets_threshold(0.79, 0.80))

    def test_rejects_out_of_range_factor(self):
        with self.assertRaises(ValueError):
            calculate_score(1.1, 1, 1, 1, 1)


if __name__ == "__main__":
    unittest.main()
