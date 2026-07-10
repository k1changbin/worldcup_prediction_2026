import json
import unittest
from pathlib import Path

from src.data_validation import (
    validate_actual_results,
    validate_dataset,
    validate_groups,
    validate_schedule,
    validate_third_place_annex,
)


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    with (ROOT / "data" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


class DataValidationTests(unittest.TestCase):
    def test_checked_in_dataset_is_valid(self):
        errors = validate_dataset(
            load("groups.json"),
            load("elo_ratings.json"),
            load("fifa_rankings.json"),
            load("team_conduct_scores.json"),
            load("schedule.json"),
            load("actual_results.json"),
            load("squads.json"),
            load("absences.json"),
            load("elo_ratings_pre_tournament.json"),
            load("model_calibration_matches.json"),
            load("third_place_annex_c.json"),
        )
        self.assertEqual(errors, [])

    def test_duplicate_schedule_number_is_rejected(self):
        schedule = load("schedule.json")
        schedule[1]["matchNumber"] = schedule[0]["matchNumber"]
        self.assertTrue(validate_schedule(schedule))

    def test_incomplete_annex_c_is_rejected(self):
        annex = load("third_place_annex_c.json")
        annex.pop(next(iter(annex)))
        self.assertTrue(validate_third_place_annex(annex))

    def test_tied_knockout_result_without_winner_is_rejected(self):
        schedule = load("schedule.json")
        teams = {team for group in load("groups.json").values() for team in group}
        result = [{
            "match_number": 73,
            "team_a": "Brazil",
            "team_b": "Morocco",
            "score_a": 1,
            "score_b": 1,
            "date": schedule[72]["date"],
            "stage": "knockout",
            "winner": None,
        }]
        errors = validate_actual_results(result, teams, schedule)
        self.assertTrue(any("knockout winner" in error for error in errors))

    def test_malformed_nested_values_return_errors_instead_of_crashing(self):
        groups = load("groups.json")
        groups["Group A"][0] = []
        self.assertTrue(validate_groups(groups))

        schedule = load("schedule.json")
        schedule[0]["homeTeam"] = []
        self.assertTrue(validate_schedule(schedule))


if __name__ == "__main__":
    unittest.main()
