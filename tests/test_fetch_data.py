import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fetch_data import (
    apply_espn_knockout_decisions,
    assign_schedule_match_numbers,
    build_round_of_32_schedule_pairs,
    extract_espn_knockout_decisions,
    merge_result_snapshots,
    validate_actual_results,
    fetch_live_world_cup_data,
)
from src.io_utils import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]


def load_data(name):
    with (ROOT / "data" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def schedule_match(number, date, home, away, kickoff, stage="group-stage"):
    return {
        "matchNumber": number,
        "date": date,
        "kickoffUtc": kickoff,
        "stage": stage,
        "homeTeam": home,
        "awayTeam": away,
    }


def result(number, date, team_a, team_b, score_a, score_b, stage="group"):
    winner = team_a if score_a > score_b else team_b if score_b > score_a else None
    return {
        "match_number": number,
        "team_a": team_a,
        "team_b": team_b,
        "score_a": score_a,
        "score_b": score_b,
        "date": date,
        "stage": stage,
        "winner": winner,
    }


class MatchNumberBackfillTests(unittest.TestCase):
    def test_checked_in_results_backfill_to_contiguous_schedule_numbers(self):
        schedule = load_data("schedule.json")
        current = load_data("actual_results.json")
        known_pairs = build_round_of_32_schedule_pairs(
            current,
            schedule,
            load_data("groups.json"),
            load_data("fifa_rankings.json"),
            load_data("team_conduct_scores.json"),
            load_data("third_place_annex_c.json"),
        )

        backfilled = assign_schedule_match_numbers(current, schedule, known_pairs)

        self.assertTrue(
            validate_actual_results(
                backfilled,
                schedule,
                known_match_pairs=known_pairs,
            )
        )
        self.assertEqual(
            [item["match_number"] for item in backfilled],
            list(range(1, len(backfilled) + 1)),
        )
        by_number = {item["match_number"]: item for item in backfilled}
        self.assertEqual(
            {by_number[74]["team_a"], by_number[74]["team_b"]},
            {"Germany", "Paraguay"},
        )
        self.assertEqual(
            {by_number[75]["team_a"], by_number[75]["team_b"]},
            {"Netherlands", "Morocco"},
        )
        self.assertEqual(
            {by_number[89]["team_a"], by_number[89]["team_b"]},
            {"France", "Paraguay"},
        )
        self.assertEqual(
            {by_number[90]["team_a"], by_number[90]["team_b"]},
            {"Morocco", "Canada"},
        )

        corrupted = [dict(item) for item in backfilled]
        corrupted[88]["team_a"] = "England"
        corrupted[88]["winner"] = "England"
        with self.assertRaisesRegex(ValueError, "bracket dependencies"):
            validate_actual_results(
                corrupted,
                schedule,
                known_match_pairs=known_pairs,
            )

    def test_same_day_knockout_uses_kickoff_order_not_match_number_order(self):
        schedule = [
            schedule_match(
                89,
                "2026-07-04",
                "Winner Match 74",
                "Winner Match 77",
                "2026-07-04T21:00:00Z",
                "round-of-16",
            ),
            schedule_match(
                90,
                "2026-07-04",
                "Winner Match 73",
                "Winner Match 75",
                "2026-07-04T17:00:00Z",
                "round-of-16",
            ),
        ]
        feed_order = [
            {
                "team_a": "Morocco",
                "team_b": "Canada",
                "score_a": 3,
                "score_b": 0,
                "date": "2026-07-04",
                "stage": "knockout",
                "winner": "Morocco",
            },
            {
                "team_a": "France",
                "team_b": "Paraguay",
                "score_a": 1,
                "score_b": 0,
                "date": "2026-07-04",
                "stage": "knockout",
                "winner": "France",
            },
        ]

        backfilled = assign_schedule_match_numbers(feed_order, schedule)
        by_number = {item["match_number"]: item for item in backfilled}

        self.assertEqual(
            {by_number[89]["team_a"], by_number[89]["team_b"]},
            {"France", "Paraguay"},
        )
        self.assertEqual(
            {by_number[90]["team_a"], by_number[90]["team_b"]},
            {"Morocco", "Canada"},
        )

    def test_later_round_is_matched_from_winner_dependencies_when_available(self):
        schedule = [
            schedule_match(
                1,
                "2026-07-01",
                "Alpha",
                "Beta",
                "2026-07-01T17:00:00Z",
                "round-of-32",
            ),
            schedule_match(
                2,
                "2026-07-01",
                "Gamma",
                "Delta",
                "2026-07-01T21:00:00Z",
                "round-of-32",
            ),
            schedule_match(
                3,
                "2026-07-04",
                "Winner Match 1",
                "Winner Match 2",
                "2026-07-04T21:00:00Z",
                "round-of-16",
            ),
            schedule_match(
                4,
                "2026-07-04",
                "Winner Match 8",
                "Winner Match 9",
                "2026-07-04T17:00:00Z",
                "round-of-16",
            ),
        ]
        previous = [
            result(1, "2026-07-01", "Alpha", "Beta", 2, 0, "knockout"),
            result(2, "2026-07-01", "Gamma", "Delta", 0, 1, "knockout"),
        ]
        later = {
            "team_a": "Alpha",
            "team_b": "Delta",
            "score_a": 1,
            "score_b": 0,
            "date": "2026-07-04",
            "stage": "knockout",
            "winner": "Alpha",
        }

        backfilled = assign_schedule_match_numbers(previous + [later], schedule)

        self.assertEqual(backfilled[-1]["match_number"], 3)


class EspnDecisionTests(unittest.TestCase):
    def test_tied_semifinal_never_uses_later_third_place_appearance(self):
        semifinal = {
            "team_a": "Alpha",
            "team_b": "Beta",
            "score_a": 1,
            "score_b": 1,
            "date": "2026-07-14",
            "stage": "knockout",
            "winner": None,
        }
        # Beta appearing in the third-place match must not influence this result.
        later_matches = [
            {
                "team_a": "Beta",
                "team_b": "Delta",
                "score_a": 2,
                "score_b": 0,
                "date": "2026-07-18",
                "stage": "knockout",
                "winner": "Beta",
            },
            {
                "team_a": "Alpha",
                "team_b": "Gamma",
                "score_a": 1,
                "score_b": 0,
                "date": "2026-07-19",
                "stage": "knockout",
                "winner": "Alpha",
            },
        ]

        unresolved = apply_espn_knockout_decisions([semifinal] + later_matches, {})
        self.assertIsNone(unresolved[0]["winner"])

        decisions = {
            ("2026-07-14", frozenset(("Alpha", "Beta"))): "Alpha"
        }
        resolved = apply_espn_knockout_decisions([semifinal] + later_matches, decisions)
        self.assertEqual(resolved[0]["winner"], "Alpha")

    def test_extracts_only_explicit_espn_winner_or_advance_flag(self):
        payload = {
            "events": [
                {
                    "competitions": [
                        {
                            "competitors": [
                                {
                                    "team": {"displayName": "Alpha"},
                                    "winner": False,
                                },
                                {
                                    "team": {"displayName": "Beta"},
                                    "advance": True,
                                },
                            ]
                        }
                    ]
                }
            ]
        }

        decisions = extract_espn_knockout_decisions(
            payload,
            "2026-07-14",
            {"Alpha", "Beta"},
        )

        self.assertEqual(
            decisions[("2026-07-14", frozenset(("Alpha", "Beta")))],
            "Beta",
        )


class RefreshFailureTests(unittest.TestCase):
    def test_network_failure_is_reported_to_callers(self):
        with mock.patch(
            "fetch_data.httpx.get",
            side_effect=RuntimeError("offline"),
        ):
            with self.assertRaisesRegex(RuntimeError, "team mapping"):
                fetch_live_world_cup_data()


class ResultPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.schedule = [
            schedule_match(
                1,
                "2026-06-11",
                "Alpha",
                "Beta",
                "2026-06-11T17:00:00Z",
            ),
            schedule_match(
                2,
                "2026-06-11",
                "Gamma",
                "Delta",
                "2026-06-11T21:00:00Z",
            ),
        ]

    def test_partial_feed_preserves_locked_result_and_grows_monotonically(self):
        existing = [result(1, "2026-06-11", "Alpha", "Beta", 1, 0)]
        incoming = [result(2, "2026-06-11", "Gamma", "Delta", 2, 1)]

        merged = merge_result_snapshots(existing, incoming, self.schedule)

        self.assertEqual([item["match_number"] for item in merged], [1, 2])
        self.assertEqual(merged[0], existing[0])

    def test_conflicting_locked_score_is_rejected(self):
        existing = [result(1, "2026-06-11", "Alpha", "Beta", 1, 0)]
        changed = [result(1, "2026-06-11", "Alpha", "Beta", 0, 1)]

        with self.assertRaisesRegex(ValueError, "conflicts with locked fields"):
            merge_result_snapshots(existing, changed, self.schedule)

    def test_atomic_json_write_replaces_valid_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.json"
            path.write_text('{"old": true}\n', encoding="utf-8")

            atomic_write_json(path, {"new": [1, 2, 3]})

            with path.open(encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), {"new": [1, 2, 3]})
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_failed_atomic_json_write_preserves_previous_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.json"
            previous = '{"locked": true}\n'
            path.write_text(previous, encoding="utf-8")

            with mock.patch("src.io_utils.json.dump", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    atomic_write_json(path, {"new": True})

            self.assertEqual(path.read_text(encoding="utf-8"), previous)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
