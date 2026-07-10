import json
from pathlib import Path

from src.data_validation import validate_dataset


ROOT = Path(__file__).resolve().parent


def load(name):
    with (ROOT / "data" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def main():
    errors = validate_dataset(
        groups=load("groups.json"),
        ratings=load("elo_ratings.json"),
        fifa_rankings=load("fifa_rankings.json"),
        conduct=load("team_conduct_scores.json"),
        schedule=load("schedule.json"),
        results=load("actual_results.json"),
        squads=load("squads.json"),
        absences=load("absences.json"),
        pre_tournament_ratings=load("elo_ratings_pre_tournament.json"),
        calibration_matches=load("model_calibration_matches.json"),
        third_place_annex=load("third_place_annex_c.json"),
    )
    if errors:
        for error in errors:
            print(f"[Data validation error] {error}")
        raise SystemExit(1)
    print("Data validation passed.")


if __name__ == "__main__":
    main()
