import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fetch_suspensions
from src.absences import clean_served_suspensions, upsert_suspension


def england_match(number):
    return {
        "match_number": number,
        "team_a": "England",
        "team_b": "Opponent",
        "score_a": 1,
        "score_b": 0,
        "stage": "knockout",
    }


class SuspensionRecordTests(unittest.TestCase):
    def test_yellow_card_accumulation_cannot_be_registered_as_two_match_ban(self):
        with self.assertRaisesRegex(ValueError, "one-match ban"):
            upsert_suspension(
                {},
                "England",
                "Player",
                "yellow_cards",
                served_at_count=7,
                suspension_length=2,
            )

    def test_extended_ban_replaces_existing_suspension_and_last_two_matches(self):
        absences = {
            "England": [
                {
                    "name": "Jarell Quansah",
                    "type": "suspension",
                    "reason": "yellow_cards",
                    "served_at_count": 6,
                    "suspension_length": 1,
                }
            ]
        }

        absences, changed = upsert_suspension(
            absences,
            "England",
            "Jarell Quansah",
            "red_card",
            served_at_count=7,
            suspension_length=2,
        )

        self.assertTrue(changed)
        self.assertEqual(
            absences["England"],
            [
                {
                    "name": "Jarell Quansah",
                    "type": "suspension",
                    "reason": "red_card",
                    "served_at_count": 7,
                    "suspension_length": 2,
                }
            ],
        )

        # England has completed five matches.  The suspension remains through
        # matches six and seven, then clears before a potential final.
        for completed in (5, 6):
            cleaned, updated = clean_served_suspensions(
                absences,
                [england_match(number) for number in range(1, completed + 1)],
            )
            self.assertFalse(updated)
            self.assertIn("England", cleaned)

        cleaned, updated = clean_served_suspensions(
            absences,
            [england_match(number) for number in range(1, 8)],
        )
        self.assertTrue(updated)
        self.assertNotIn("England", cleaned)

    def test_sync_refreshes_an_existing_record_when_ban_is_extended(self):
        html = """
        <table>
          <tr><th>Player</th><th>Team</th><th>Offense</th><th>Suspension</th></tr>
          <tr><td>Jarell Quansah</td><td>England</td><td>Red card</td><td>Two matches</td></tr>
        </table>
        """
        response = mock.Mock(status_code=200, text=html)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            squads_path = root / "squads.json"
            results_path = root / "actual_results.json"
            absences_path = root / "absences.json"
            squads_path.write_text(
                json.dumps(
                    {
                        "England": [
                            {
                                "name": "Jarell Quansah",
                                "position": "Defender",
                                "value_eur": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            results_path.write_text(
                json.dumps([england_match(number) for number in range(1, 6)]),
                encoding="utf-8",
            )
            absences_path.write_text(
                json.dumps(
                    {
                        "England": [
                            {
                                "name": "Jarell Quansah",
                                "type": "suspension",
                                "reason": "yellow_cards",
                                "served_at_count": 6,
                                "suspension_length": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(fetch_suspensions, "SQUADS_PATH", squads_path),
                mock.patch.object(fetch_suspensions, "ACTUAL_RESULTS_PATH", results_path),
                mock.patch.object(fetch_suspensions, "ABSENCES_PATH", absences_path),
                mock.patch.object(fetch_suspensions, "load_active_teams", return_value={"England"}),
                mock.patch.object(fetch_suspensions.httpx, "get", return_value=response),
            ):
                self.assertTrue(fetch_suspensions.main())

            saved = json.loads(absences_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["England"][0]["reason"], "red_card")
            self.assertEqual(saved["England"][0]["suspension_length"], 2)
            self.assertEqual(saved["England"][0]["served_at_count"], 7)


if __name__ == "__main__":
    unittest.main()
