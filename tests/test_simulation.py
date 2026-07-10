import json
import random
import shutil
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest import mock

import numpy as np

from src.bracket import (
    resolve_match_teams,
    validate_knockout_bracket,
    winner_sources_for_match,
)
from src.elo import EloSystem
from src.model_config import (
    HOST_ADVANTAGE_ELO,
    NEAR_REGION_TRAVEL_PENALTY,
    REST_ELO_PER_DAY,
)
from src.schedule import CITY_REGIONS, TournamentSchedule, date_to_day_num
from src.simulation import KNOCKOUT_MATCH_INFO, WorldCupSimulation


ROOT = Path(__file__).resolve().parents[1]


def make_simulation(*, actual_results=None, rng=None, np_rng=None):
    actual_results_file = None
    temp_directory = None
    if actual_results is not None:
        temp_directory = Path(tempfile.mkdtemp())
        path = temp_directory / "actual_results.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(actual_results, handle)
        actual_results_file = str(path)

    elo = EloSystem()
    elo.load_ratings(str(ROOT / "data" / "elo_ratings.json"))
    if rng is None:
        rng = random.Random(2026)
    if np_rng is None and not callable(getattr(rng, "poisson", None)):
        np_rng = np.random.default_rng(2026)
    simulation = WorldCupSimulation(
        elo,
        actual_results_file=actual_results_file,
        rng=rng,
        np_rng=np_rng,
    )
    if temp_directory is not None:
        simulation._test_temp_cleanup = weakref.finalize(
            simulation,
            shutil.rmtree,
            temp_directory,
            True,
        )
    return simulation


class ScheduleSourceOfTruthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schedule = TournamentSchedule.from_file(ROOT / "data" / "schedule.json")

    def test_knockout_metadata_is_derived_from_schedule(self):
        for match in self.schedule.knockout_matches():
            with self.subTest(match_number=match.match_number):
                metadata = KNOCKOUT_MATCH_INFO[match.match_number]
                self.assertEqual(metadata["date"], match.date)
                self.assertEqual(metadata["region"], CITY_REGIONS[match.host_city])
                self.assertEqual(metadata["host_city"], match.host_city)

    def test_official_round_of_16_winner_sources_include_m89_and_m90(self):
        self.assertEqual(winner_sources_for_match(self.schedule, 89), (74, 77))
        self.assertEqual(winner_sources_for_match(self.schedule, 90), (73, 75))
        self.assertEqual(winner_sources_for_match(self.schedule, 97), (89, 90))
        validate_knockout_bracket(self.schedule)

    def test_last_group_match_uses_each_teams_actual_date_and_city(self):
        simulation = make_simulation()
        state = simulation.build_group_stage_team_states(["Mexico"])["Mexico"]
        self.assertEqual(state["last_match_number"], 53)
        self.assertEqual(state["last_date"], date_to_day_num("2026-06-24"))
        self.assertEqual(state["last_city"], "mexico-city")
        self.assertEqual(state["last_region"], CITY_REGIONS["mexico-city"])


class ActualResultLookupTests(unittest.TestCase):
    def test_match_number_takes_priority_over_legacy_pair_fallback(self):
        results = [
            {
                "team_a": "Brazil",
                "team_b": "Morocco",
                "score_a": 0,
                "score_b": 1,
                "stage": "knockout",
                "winner": "Morocco",
            },
            {
                "match_number": 73,
                "team_a": "Brazil",
                "team_b": "Morocco",
                "score_a": 2,
                "score_b": 0,
                "stage": "knockout",
                "winner": "Brazil",
            },
        ]
        simulation = make_simulation(actual_results=results)

        outcome = simulation.simulate_knockout_match(
            "Brazil", "Morocco", match_number=73
        )

        self.assertEqual(outcome, ("Brazil", 2, 0, False))

    def test_legacy_result_without_match_number_remains_supported(self):
        results = [{
            "team_a": "Morocco",
            "team_b": "Brazil",
            "score_a": 1,
            "score_b": 1,
            "stage": "knockout",
            "winner": "Morocco",
        }]
        simulation = make_simulation(actual_results=results)

        outcome = simulation.simulate_knockout_match(
            "Brazil", "Morocco", match_number=73
        )

        self.assertEqual(outcome, ("Morocco", 1, 1, True))

    def test_numbered_result_with_wrong_teams_is_rejected(self):
        results = [{
            "match_number": 73,
            "team_a": "Spain",
            "team_b": "France",
            "score_a": 1,
            "score_b": 0,
            "stage": "knockout",
            "winner": "Spain",
        }]
        simulation = make_simulation(actual_results=results)

        with self.assertRaisesRegex(ValueError, "M73 has teams"):
            simulation.simulate_knockout_match(
                "Brazil", "Morocco", match_number=73
            )

    def test_consensus_does_not_resimulate_a_locked_match(self):
        results = [{
            "match_number": 73,
            "team_a": "Brazil",
            "team_b": "Morocco",
            "score_a": 2,
            "score_b": 0,
            "stage": "knockout",
            "winner": "Brazil",
        }]
        simulation = make_simulation(actual_results=results)
        simulation.np_rng = mock.Mock()

        winner, score_a, score_b, is_pk, counts = (
            simulation.simulate_knockout_match_consensus(
                "Brazil", "Morocco", runs=10000, match_number=73
            )
        )

        self.assertEqual((winner, score_a, score_b, is_pk), ("Brazil", 2, 0, False))
        self.assertEqual(counts["Brazil"], 10000)
        simulation.np_rng.poisson.assert_not_called()

    def test_tied_actual_knockout_result_requires_explicit_winner(self):
        results = [{
            "match_number": 73,
            "team_a": "Brazil",
            "team_b": "Morocco",
            "score_a": 1,
            "score_b": 1,
            "stage": "knockout",
        }]
        simulation = make_simulation(actual_results=results)

        with self.assertRaisesRegex(ValueError, "missing an explicit winner"):
            simulation.simulate_knockout_match(
                "Brazil", "Morocco", match_number=73
            )


class ModelConsistencyTests(unittest.TestCase):
    def test_group_simulation_always_enables_host_country_advantage(self):
        simulation = make_simulation(actual_results=[])
        simulation.simulate_match = mock.Mock(return_value=(0, 0))

        simulation.simulate_group_stage()

        self.assertEqual(simulation.simulate_match.call_count, 72)
        for call in simulation.simulate_match.call_args_list:
            self.assertTrue(call.kwargs["home_advantage"])

    def test_group_schedule_context_uses_real_rest_and_travel(self):
        simulation = make_simulation(actual_results=[])
        contexts = simulation.build_group_stage_match_contexts()

        self.assertEqual(contexts[1]["rest_days_diff"], 0)
        self.assertEqual(contexts[1]["travel_fatigue_a"], 0.0)
        self.assertEqual(contexts[25]["rest_days_a"], 7)
        self.assertEqual(contexts[25]["rest_days_b"], 7)
        self.assertEqual(
            contexts[25]["travel_fatigue_a"],
            NEAR_REGION_TRAVEL_PENALTY,
        )
        self.assertEqual(
            contexts[25]["travel_fatigue_b"],
            NEAR_REGION_TRAVEL_PENALTY,
        )

    def test_group_simulation_passes_schedule_travel_to_match_model(self):
        simulation = make_simulation(actual_results=[])
        simulation.simulate_match = mock.Mock(return_value=(0, 0))

        simulation.simulate_group_stage()

        self.assertTrue(
            any(
                call.kwargs["travel_fatigue_a"] > 0
                or call.kwargs["travel_fatigue_b"] > 0
                for call in simulation.simulate_match.call_args_list
            )
        )

    def test_injected_generators_reproduce_python_and_numpy_sequences(self):
        first = make_simulation(
            rng=random.Random(42),
            np_rng=np.random.default_rng(42),
        )
        second = make_simulation(
            rng=random.Random(42),
            np_rng=np.random.default_rng(42),
        )

        first_scores = [first.simulate_match("Brazil", "Morocco") for _ in range(8)]
        second_scores = [second.simulate_match("Brazil", "Morocco") for _ in range(8)]

        self.assertEqual(first_scores, second_scores)
        self.assertEqual(first.rng.random(), second.rng.random())

    def test_injected_random_generators_are_used(self):
        first = make_simulation(
            rng=random.Random(7),
            np_rng=np.random.default_rng(11),
        )
        second = make_simulation(
            rng=random.Random(7),
            np_rng=np.random.default_rng(11),
        )

        self.assertEqual(
            first.simulate_knockout_match("Brazil", "Morocco"),
            second.simulate_knockout_match("Brazil", "Morocco"),
        )

    def test_one_numpy_generator_can_drive_all_randomness(self):
        first = make_simulation(rng=np.random.default_rng(19))
        second = make_simulation(rng=np.random.default_rng(19))

        self.assertEqual(
            [first.simulate_knockout_match("Brazil", "Morocco") for _ in range(5)],
            [second.simulate_knockout_match("Brazil", "Morocco") for _ in range(5)],
        )

    def test_public_expected_goals_matches_score_sampler_context(self):
        simulation = make_simulation()
        lambdas = simulation.get_expected_goals(
            "USA",
            "Brazil",
            home_advantage=True,
            rest_days_diff=2,
            travel_fatigue_a=0.015,
            travel_fatigue_b=0.03,
        )
        neutral_lambdas = simulation.get_expected_goals(
            "USA", "Brazil", home_advantage=False
        )

        self.assertGreater(lambdas[0], 0.0)
        self.assertGreater(lambdas[1], 0.0)
        self.assertGreater(lambdas[0] / lambdas[1], neutral_lambdas[0] / neutral_lambdas[1])

    def test_public_adjusted_ratings_uses_shared_model_constants(self):
        simulation = make_simulation()
        base_a = simulation.elo_system.get_rating("USA")
        base_b = simulation.elo_system.get_rating("Brazil")

        adjusted_a, adjusted_b = simulation.get_adjusted_ratings(
            "USA", "Brazil", home_advantage=True, rest_days_diff=2
        )

        self.assertEqual(
            adjusted_a,
            base_a + HOST_ADVANTAGE_ELO + 2 * REST_ELO_PER_DAY,
        )
        self.assertEqual(adjusted_b, base_b)


class CurrentBracketStateTests(unittest.TestCase):
    @staticmethod
    def fake_standings(simulation):
        standings = {}
        for index, (group_name, teams) in enumerate(simulation.groups.items()):
            rows = []
            for position, team in enumerate(teams):
                points = 30 - index if position == 2 and index < 8 else 0
                rows.append((team, {
                    "pts": points,
                    "gd": 0,
                    "gf": 0,
                    "ga": 0,
                    "w": 0,
                    "d": 0,
                    "l": 0,
                }))
            standings[group_name] = rows
        return standings

    def test_current_state_resolves_actual_winners_and_future_context(self):
        simulation = make_simulation(actual_results=[])
        standings = self.fake_standings(simulation)
        team_by_code, third_assignment, _ = simulation._knockout_bracket_inputs(
            standings
        )
        winners = {}
        match_74 = simulation.schedule.match(74)
        match_77 = simulation.schedule.match(77)
        teams_74 = resolve_match_teams(
            match_74,
            team_by_code=team_by_code,
            third_assignment=third_assignment,
            winners=winners,
        )
        teams_77 = resolve_match_teams(
            match_77,
            team_by_code=team_by_code,
            third_assignment=third_assignment,
            winners=winners,
        )
        simulation.actual_results = [
            {
                "match_number": 74,
                "team_a": teams_74[0],
                "team_b": teams_74[1],
                "score_a": 1,
                "score_b": 0,
                "stage": "knockout",
                "winner": teams_74[0],
            },
            {
                "match_number": 77,
                "team_a": teams_77[0],
                "team_b": teams_77[1],
                "score_a": 2,
                "score_b": 0,
                "stage": "knockout",
                "winner": teams_77[0],
            },
        ]

        state = simulation.build_current_knockout_state(standings)
        match_89 = state["Round of 16"][0]
        match_90 = state["Round of 16"][1]

        self.assertEqual(match_89["team_a"], teams_74[0])
        self.assertEqual(match_89["team_b"], teams_77[0])
        self.assertEqual(match_89["rest_days_a"], 5)
        self.assertEqual(match_89["rest_days_b"], 4)
        self.assertEqual(match_89["rest_days_diff"], 1)
        self.assertEqual(match_90["team_a"], "Winner M73")
        self.assertEqual(match_90["team_b"], "Winner M75")
        self.assertIsNone(match_90["rest_days_diff"])

    def test_full_simulation_follows_every_schedule_winner_source(self):
        simulation = make_simulation(
            actual_results=[],
            rng=random.Random(99),
            np_rng=np.random.default_rng(99),
        )
        simulation.last_standings = self.fake_standings(simulation)

        results = simulation.simulate_knockout_stage()
        all_matches = {
            match["match_number"]: match
            for label in (
                "Round of 32",
                "Round of 16",
                "Quarter-finals",
                "Semi-finals",
                "Third-place",
                "Final",
            )
            for match in results[label]
        }

        self.assertEqual(len(all_matches), 32)
        for match_number in range(89, 103):
            source_a, source_b = winner_sources_for_match(
                simulation.schedule, match_number
            )
            self.assertEqual(
                all_matches[match_number]["team_a"],
                all_matches[source_a]["winner"],
            )
            self.assertEqual(
                all_matches[match_number]["team_b"],
                all_matches[source_b]["winner"],
            )
        source_a, source_b = winner_sources_for_match(simulation.schedule, 104)
        self.assertEqual(all_matches[104]["team_a"], all_matches[source_a]["winner"])
        self.assertEqual(all_matches[104]["team_b"], all_matches[source_b]["winner"])
        semi_101 = all_matches[101]
        semi_102 = all_matches[102]
        loser_101 = (
            semi_101["team_b"]
            if semi_101["winner"] == semi_101["team_a"]
            else semi_101["team_a"]
        )
        loser_102 = (
            semi_102["team_b"]
            if semi_102["winner"] == semi_102["team_a"]
            else semi_102["team_a"]
        )
        self.assertEqual(all_matches[103]["team_a"], loser_101)
        self.assertEqual(all_matches[103]["team_b"], loser_102)
        self.assertEqual(results["Champion"], all_matches[104]["winner"])


if __name__ == "__main__":
    unittest.main()
