import json
import unittest
from pathlib import Path

from fetch_calibration_data import parse_world_cup_group_matches
from src.evaluation import build_current_group_records, evaluate_parameters
from src.simulation import HOST_COUNTRIES


ROOT = Path(__file__).resolve().parents[1]


def load_data(name):
    with (ROOT / "data" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


class CalibrationDataTests(unittest.TestCase):
    def test_pre_match_ratings_are_reconstructed_without_leakage(self):
        row = "2022\t11\t20\tQA\tEC\t0\t2\tWC\t\t-38\t1642\t1870\t-7\t+1\t53\t17"
        record = parse_world_cup_group_matches(row, 2022, "QA", group_matches=1)[0]
        self.assertEqual(record["team_a"], "EC")
        self.assertEqual(record["elo_a"], 1832)
        self.assertEqual(record["elo_b"], 1680)
        self.assertTrue(record["host_b"])

    def test_metrics_are_finite_and_probabilistic(self):
        records = [
            {
                "elo_a": 1800,
                "elo_b": 1700,
                "score_a": 2,
                "score_b": 1,
                "host_a": False,
                "host_b": False,
            },
            {
                "elo_a": 1700,
                "elo_b": 1700,
                "score_a": 0,
                "score_b": 0,
                "host_a": False,
                "host_b": False,
            },
        ]
        metrics = evaluate_parameters(records)
        self.assertEqual(metrics["matches"], 2)
        self.assertGreater(metrics["log_loss"], 0.0)
        self.assertGreater(metrics["scoreline_log_loss"], 0.0)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_current_records_use_pre_tournament_ratings_and_canonical_sides(self):
        results = [{
            "team_a": "Zulu",
            "team_b": "Alpha",
            "score_a": 2,
            "score_b": 0,
            "date": "2026-06-11",
            "stage": "group",
        }]
        records = build_current_group_records(
            results,
            {"Zulu": 1800, "Alpha": 1700},
            {"Zulu"},
        )
        self.assertEqual(records[0]["team_a"], "Alpha")
        self.assertEqual(records[0]["score_a"], 0)
        self.assertTrue(records[0]["host_b"])

    def test_production_exponent_improves_over_previous_snapshot(self):
        historical = load_data("model_calibration_matches.json")
        current = build_current_group_records(
            load_data("actual_results.json"),
            load_data("elo_ratings_pre_tournament.json"),
            HOST_COUNTRIES,
        )

        for records in (historical, current):
            with self.subTest(matches=len(records)):
                production = evaluate_parameters(records)
                previous = evaluate_parameters(records, exponent=0.376)
                self.assertLess(production["log_loss"], previous["log_loss"])
                self.assertLess(production["brier_score"], previous["brier_score"])
                self.assertLess(
                    production["scoreline_log_loss"],
                    previous["scoreline_log_loss"],
                )


if __name__ == "__main__":
    unittest.main()
