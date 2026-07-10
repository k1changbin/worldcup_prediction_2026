import unittest

from src.forecast import wilson_interval


class WilsonIntervalTests(unittest.TestCase):
    def test_interval_contains_observed_proportion(self):
        low, high = wilson_interval(35, 100)
        self.assertLess(low, 0.35)
        self.assertGreater(high, 0.35)

    def test_boundaries_stay_inside_unit_interval(self):
        self.assertEqual(wilson_interval(0, 100)[0], 0.0)
        self.assertEqual(wilson_interval(100, 100)[1], 1.0)

    def test_invalid_trial_count_is_rejected(self):
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)


if __name__ == "__main__":
    unittest.main()
