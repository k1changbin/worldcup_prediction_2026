import json
import tempfile
import unittest
from pathlib import Path

from src.elo import EloSystem


class EloSystemTests(unittest.TestCase):
    def test_expected_score_is_symmetric(self):
        first = EloSystem.expected_score(1800, 1700)
        second = EloSystem.expected_score(1700, 1800)
        self.assertAlmostEqual(first + second, 1.0)

    def test_extreme_rating_gap_remains_finite(self):
        self.assertGreaterEqual(EloSystem.expected_score(1e6, -1e6), 0.0)
        self.assertLessEqual(EloSystem.expected_score(1e6, -1e6), 1.0)

    def test_invalid_rating_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ratings.json"
            path.write_text(json.dumps({"Alpha": "high"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                EloSystem().load_ratings(path)


if __name__ == "__main__":
    unittest.main()
