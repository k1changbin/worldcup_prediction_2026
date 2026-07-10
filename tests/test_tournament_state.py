import json
import unittest
from pathlib import Path

from src.tournament_state import (
    calculate_group_standings,
    get_active_teams,
    group_stage_is_complete,
    rank_group_stats,
)


ROOT = Path(__file__).resolve().parents[1]


def load_data(name):
    with (ROOT / "data" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


class TournamentStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = load_data("groups.json")
        cls.results = load_data("actual_results.json")
        cls.ratings = load_data("elo_ratings.json")
        cls.fifa_rankings = load_data("fifa_rankings.json")
        cls.conduct = load_data("team_conduct_scores.json")

    def test_current_group_stage_is_complete(self):
        self.assertTrue(group_stage_is_complete(self.groups, self.results))

    def test_official_standings_include_each_team_once(self):
        standings = calculate_group_standings(
            self.groups,
            self.results,
            self.fifa_rankings,
            self.conduct,
        )
        ranked = [team for group in standings.values() for team, _ in group]
        self.assertEqual(len(ranked), 48)
        self.assertEqual(len(set(ranked)), 48)

    def test_active_teams_exclude_recorded_knockout_losers(self):
        active = get_active_teams(
            self.groups,
            self.results,
            elo_ratings=self.ratings,
        )
        knockout_losers = set()
        for match in self.results:
            if match.get("stage") != "knockout" or not match.get("winner"):
                continue
            knockout_losers.add(
                match["team_b"]
                if match["winner"] == match["team_a"]
                else match["team_a"]
            )
        self.assertTrue(active.isdisjoint(knockout_losers))

    def test_head_to_head_precedes_overall_goal_difference(self):
        stats = {
            "Alpha": {"pts": 6, "gd": 1, "gf": 3},
            "Beta": {"pts": 6, "gd": 10, "gf": 12},
            "Gamma": {"pts": 3, "gd": 0, "gf": 2},
            "Delta": {"pts": 0, "gd": -11, "gf": 0},
        }
        results = {
            ("Alpha", "Beta"): (1, 0),
            ("Beta", "Alpha"): (0, 1),
        }

        ranked = rank_group_stats(stats, results)

        self.assertEqual([team for team, _ in ranked[:2]], ["Alpha", "Beta"])

    def test_conduct_then_fifa_ranking_breaks_a_complete_performance_tie(self):
        stats = {
            "Alpha": {"pts": 4, "gd": 0, "gf": 2},
            "Beta": {"pts": 4, "gd": 0, "gf": 2},
        }
        results = {
            ("Alpha", "Beta"): (1, 1),
            ("Beta", "Alpha"): (1, 1),
        }

        ranked_by_conduct = rank_group_stats(
            stats,
            results,
            {"Alpha": 20, "Beta": 10},
            {"Alpha": -1, "Beta": -5},
        )
        self.assertEqual(ranked_by_conduct[0][0], "Alpha")

        ranked_by_fifa = rank_group_stats(
            stats,
            results,
            {"Alpha": 20, "Beta": 10},
            {"Alpha": -1, "Beta": -1},
        )
        self.assertEqual(ranked_by_fifa[0][0], "Beta")


if __name__ == "__main__":
    unittest.main()
