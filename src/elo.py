import os
import json

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
            self.ratings = json.load(f)

    def get_rating(self, team):
        return self.ratings.get(team, 1500)

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

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
    prob = elo.expected_score(
        elo.get_rating(team_a),
        elo.get_rating(team_b)
    )

    print(f"{team_a} vs {team_b}")
    print(f"{team_a} win probability: {prob:.1%}")
    print(f"{team_b} win probability: {1 - prob:.1%}")
