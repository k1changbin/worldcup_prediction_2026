import os
import json
import math
from numbers import Real

class EloSystem:
    def __init__(self, k_factor=20):
        self.k_factor = k_factor
        self.ratings = {}

    def load_ratings(self, path: str = None):
        """Load initial Elo ratings from JSON, resolving the default path automatically."""
        if path is None:
            # Resolve data/elo_ratings.json relative to the project root.
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base_dir, "data", "elo_ratings.json")

        with open(path, "r", encoding="utf-8") as f:
            ratings = json.load(f)
        if not isinstance(ratings, dict) or not ratings:
            raise ValueError("Elo ratings must be a non-empty JSON object")
        invalid = [
            team
            for team, rating in ratings.items()
            if not isinstance(team, str)
            or not team
            or isinstance(rating, bool)
            or not isinstance(rating, Real)
            or not math.isfinite(float(rating))
        ]
        if invalid:
            raise ValueError(f"Invalid Elo rating entries: {', '.join(map(str, invalid))}")
        self.ratings = {team: float(rating) for team, rating in ratings.items()}

    def get_rating(self, team):
        return self.ratings.get(team, 1500)

    @staticmethod
    def expected_score(rating_a, rating_b):
        for label, rating in (("rating_a", rating_a), ("rating_b", rating_b)):
            if (
                isinstance(rating, bool)
                or not isinstance(rating, Real)
                or not math.isfinite(float(rating))
            ):
                raise ValueError(f"{label} must be a finite real number")
        difference = (float(rating_b) - float(rating_a)) / 400.0
        if difference >= 0:
            power = 10.0 ** min(difference, 308.0)
            return 1.0 / (1.0 + power)
        power = 10.0 ** min(-difference, 308.0)
        return power / (1.0 + power)

    def update_rating(self, team_a, team_b, score_a, score_b, k_factor=None):
        ra = self.get_rating(team_a)
        rb = self.get_rating(team_b)

        ea = self.expected_score(ra, rb)
        eb = self.expected_score(rb, ra)

        if score_a > score_b:
            sa, sb = 1, 0
        elif score_a < score_b:
            sa, sb = 0, 1
        else:
            sa, sb = 0.5, 0.5

        k = self.k_factor if k_factor is None else k_factor
        self.ratings[team_a] = ra + k * (sa - ea)
        self.ratings[team_b] = rb + k * (sb - eb)

    def export_ratings(self):
        return dict(sorted(self.ratings.items(), key=lambda item: item[1], reverse=True))


if __name__ == "__main__":
    elo = EloSystem()
    elo.load_ratings()

    team_a, team_b = "Brazil", "South Korea"
    expected = elo.expected_score(
        elo.get_rating(team_a),
        elo.get_rating(team_b)
    )

    print(f"{team_a} vs {team_b}")
    print(f"{team_a} Elo expected score: {expected:.1%}")
    print(f"{team_b} Elo expected score: {1 - expected:.1%}")
