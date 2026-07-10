import unittest

from fetch_injuries import merge_active_squad_snapshot


class SquadSnapshotTests(unittest.TestCase):
    def test_partial_parse_preserves_previous_active_squad(self):
        parsed = {"Alpha": [{"name": "New"}]}
        existing = {
            "Alpha": [{"name": "Old"}],
            "Beta": [{"name": "Preserved"}],
            "Eliminated": [{"name": "Dropped"}],
        }

        merged, missing = merge_active_squad_snapshot(
            {"Alpha", "Beta"}, parsed, existing
        )

        self.assertEqual(merged["Alpha"], [{"name": "New"}])
        self.assertEqual(merged["Beta"], [{"name": "Preserved"}])
        self.assertNotIn("Eliminated", merged)
        self.assertEqual(missing, [])

    def test_missing_current_and_previous_squad_is_reported(self):
        merged, missing = merge_active_squad_snapshot(
            {"Alpha", "Beta"}, {"Alpha": []}, {}
        )
        self.assertEqual(merged, {"Alpha": []})
        self.assertEqual(missing, ["Beta"])


if __name__ == "__main__":
    unittest.main()
