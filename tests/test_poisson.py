import math
import unittest

import numpy as np

from src.poisson import (
    elo_expected_score_to_lambdas,
    match_probabilities,
    modal_scoreline,
    modal_scorelines,
    poisson_modes,
    poisson_prob,
    simulate_match_score,
    win_prob_to_lambda,
)


class WinProbabilityConversionTests(unittest.TestCase):
    def test_balanced_match_uses_baseline_for_both_teams(self):
        self.assertEqual(win_prob_to_lambda(0.5), (1.35, 1.35))

    def test_elo_probability_endpoints_stay_finite(self):
        for probability in (0.0, 1.0):
            with self.subTest(probability=probability):
                lambda_a, lambda_b = win_prob_to_lambda(probability)
                self.assertTrue(math.isfinite(lambda_a))
                self.assertTrue(math.isfinite(lambda_b))
                self.assertGreater(lambda_a, 0.0)
                self.assertGreater(lambda_b, 0.0)

    def test_primary_elo_api_and_compatibility_wrapper_agree(self):
        expected = elo_expected_score_to_lambdas(0.7, exponent=0.42)
        compatibility = win_prob_to_lambda(0.7, exponent=0.42)
        self.assertEqual(expected, compatibility)

    def test_exponent_is_calibratable(self):
        shallow = elo_expected_score_to_lambdas(0.75, exponent=0.1)
        steep = elo_expected_score_to_lambdas(0.75, exponent=0.8)
        self.assertGreater(steep[0] / steep[1], shallow[0] / shallow[1])

    def test_invalid_probability_and_baseline_are_rejected(self):
        for probability in (-0.1, 1.1, math.nan, math.inf):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError):
                    win_prob_to_lambda(probability)
        with self.assertRaises(TypeError):
            win_prob_to_lambda("0.5")
        with self.assertRaises(ValueError):
            win_prob_to_lambda(0.5, base_goals=-1.0)
        for exponent in (0.0, -0.1, math.nan, math.inf):
            with self.subTest(exponent=exponent):
                with self.assertRaises(ValueError):
                    elo_expected_score_to_lambdas(0.5, exponent=exponent)
        with self.assertRaises(TypeError):
            elo_expected_score_to_lambdas(0.5, exponent="0.376")


class PoissonProbabilityTests(unittest.TestCase):
    def assert_normalized(self, outcomes):
        self.assertEqual(set(outcomes), {"win", "draw", "lose"})
        self.assertTrue(all(0.0 <= value <= 1.0 for value in outcomes.values()))
        self.assertAlmostEqual(sum(outcomes.values()), 1.0, places=15)

    def test_known_pmf_values(self):
        self.assertEqual(poisson_prob(0.0, 0), 1.0)
        self.assertEqual(poisson_prob(0.0, 3), 0.0)
        self.assertAlmostEqual(poisson_prob(2.0, 3), 4 * math.exp(-2) / 3)

    def test_balanced_match_is_symmetric(self):
        outcomes = match_probabilities(1.35, 1.35)
        self.assert_normalized(outcomes)
        self.assertAlmostEqual(outcomes["win"], outcomes["lose"], places=14)

    def test_extreme_elo_outputs_remain_normalized(self):
        for elo_probability in (0.0, 1e-15, 0.5, 1.0 - 1e-15, 1.0):
            with self.subTest(elo_probability=elo_probability):
                lambdas = win_prob_to_lambda(elo_probability)
                outcomes = match_probabilities(*lambdas)
                self.assert_normalized(outcomes)

    def test_score_cutoff_no_longer_discards_probability_mass(self):
        outcomes = match_probabilities(30.0, 25.0, max_goals=10)
        self.assert_normalized(outcomes)
        self.assertGreater(outcomes["win"], outcomes["lose"])

    def test_zero_lambda_cases_are_exact(self):
        self.assertEqual(
            match_probabilities(0.0, 0.0),
            {"win": 0.0, "draw": 1.0, "lose": 0.0},
        )
        outcomes = match_probabilities(0.0, 2.0)
        self.assert_normalized(outcomes)
        self.assertEqual(outcomes["win"], 0.0)
        self.assertAlmostEqual(outcomes["draw"], math.exp(-2.0))

    def test_large_lambdas_use_stable_fallback(self):
        outcomes = match_probabilities(10_000_000.0, 10_000_000.0)
        self.assert_normalized(outcomes)
        self.assertAlmostEqual(outcomes["win"], outcomes["lose"], places=14)

    def test_invalid_probability_inputs_are_rejected(self):
        for invalid in (-1.0, math.nan, math.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    match_probabilities(invalid, 1.0)
        with self.assertRaises(TypeError):
            match_probabilities(1.0, "1.0")
        with self.assertRaises(ValueError):
            match_probabilities(1.0, 1.0, max_goals=-1)
        with self.assertRaises(TypeError):
            match_probabilities(1.0, 1.0, max_goals=10.5)


class ModalScorelineTests(unittest.TestCase):
    def test_non_integer_lambdas_have_one_analytical_mode(self):
        self.assertEqual(modal_scoreline(1.8, 0.9), (1, 0))
        self.assertEqual(modal_scorelines(1.8, 0.9), ((1, 0),))

    def test_integer_lambda_exposes_both_tied_modes(self):
        self.assertEqual(poisson_modes(2.0), (2, 1))
        self.assertEqual(
            set(modal_scorelines(2.0, 1.0)),
            {(2, 1), (2, 0), (1, 1), (1, 0)},
        )
        self.assertEqual(modal_scoreline(2.0, 1.0), (2, 1))


class RandomGeneratorInjectionTests(unittest.TestCase):
    def test_injected_generator_reproduces_standalone_score(self):
        first = simulate_match_score(2.2, 0.8, rng=np.random.default_rng(2026))
        second = simulate_match_score(2.2, 0.8, rng=np.random.default_rng(2026))
        self.assertEqual(first, second)
        self.assertTrue(all(isinstance(value, int) for value in first))

    def test_injected_generators_reproduce_sequence(self):
        first_rng = np.random.default_rng(42)
        second_rng = np.random.default_rng(42)
        first = [simulate_match_score(1.4, 1.1, rng=first_rng) for _ in range(8)]
        second = [simulate_match_score(1.4, 1.1, rng=second_rng) for _ in range(8)]
        self.assertEqual(first, second)

    def test_invalid_rng_is_rejected(self):
        with self.assertRaises(TypeError):
            simulate_match_score(1.0, 1.0, rng=object())


if __name__ == "__main__":
    unittest.main()
