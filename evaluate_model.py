import argparse
import json

from src.evaluation import (
    build_current_group_records,
    evaluate_parameters,
    grid_search,
)
from src.paths import data_path
from src.model_config import BASE_GOALS
from src.simulation import HOST_COUNTRIES


PREVIOUS_EXPONENT = 0.376


def print_metrics(label, metrics):
    print(label)
    print(f"  Matches: {metrics['matches']}")
    print(f"  Multiclass log loss: {metrics['log_loss']:.4f}")
    print(f"  Multiclass Brier score: {metrics['brier_score']:.4f}")
    print(f"  Poisson scoreline log loss: {metrics['scoreline_log_loss']:.4f}")
    print(
        f"  Mean predicted/observed total goals: "
        f"{metrics['mean_predicted_goals']:.2f}/{metrics['mean_observed_goals']:.2f}"
    )
    print(f"  Most-likely-outcome accuracy: {metrics['accuracy']:.1%}")
    predicted = metrics["mean_predicted"]
    observed = metrics["observed_frequency"]
    print(
        "  Predicted W/D/L: "
        f"{predicted['win']:.1%}/{predicted['draw']:.1%}/{predicted['lose']:.1%}"
    )
    print(
        "  Observed W/D/L:  "
        f"{observed['win']:.1%}/{observed['draw']:.1%}/{observed['lose']:.1%}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-search", action="store_true")
    args = parser.parse_args()

    with data_path("model_calibration_matches.json").open(encoding="utf-8") as handle:
        records = json.load(handle)

    historical_metrics = evaluate_parameters(records)
    print_metrics(
        "Current model: retrospective 2018 and 2022 group stages",
        historical_metrics,
    )
    for year in sorted({record["year"] for record in records}):
        print_metrics(
            f"Retrospective tournament slice: {year}",
            evaluate_parameters(record for record in records if record["year"] == year),
        )
    historical_previous = evaluate_parameters(records, exponent=PREVIOUS_EXPONENT)
    print(
        f"Previous exponent comparison ({PREVIOUS_EXPONENT:.3f} -> production):\n"
        f"  Retrospective log loss: {historical_previous['log_loss']:.4f} -> "
        f"{historical_metrics['log_loss']:.4f}\n"
        f"  Retrospective Brier:    {historical_previous['brier_score']:.4f} -> "
        f"{historical_metrics['brier_score']:.4f}\n"
        f"  Scoreline log loss:     {historical_previous['scoreline_log_loss']:.4f} -> "
        f"{historical_metrics['scoreline_log_loss']:.4f}"
    )

    with data_path("elo_ratings_pre_tournament.json").open(encoding="utf-8") as handle:
        pre_tournament_ratings = json.load(handle)
    with data_path("actual_results.json").open(encoding="utf-8") as handle:
        actual_results = json.load(handle)
    current_records = build_current_group_records(
        actual_results,
        pre_tournament_ratings,
        HOST_COUNTRIES,
    )
    current_metrics = evaluate_parameters(current_records)
    print_metrics(
        "Current-tournament diagnostic: 2026 group stage",
        current_metrics,
    )
    current_previous = evaluate_parameters(
        current_records,
        exponent=PREVIOUS_EXPONENT,
    )
    print(
        f"  Previous exponent log loss: {current_previous['log_loss']:.4f} -> "
        f"{current_metrics['log_loss']:.4f}\n"
        f"  Previous exponent Brier:    {current_previous['brier_score']:.4f} -> "
        f"{current_metrics['brier_score']:.4f}\n"
        f"  Previous scoreline log loss: {current_previous['scoreline_log_loss']:.4f} -> "
        f"{current_metrics['scoreline_log_loss']:.4f}"
    )

    if args.grid_search:
        candidates = grid_search(
            records,
            base_goals_values=[BASE_GOALS],
            exponent_values=[value / 100 for value in range(10, 46)],
            host_bonus_values=range(0, 81, 10),
        )
        best = candidates[0]
        print("Best in-sample diagnostic parameters")
        print(
            f"  base_goals={best['base_goals']:.2f}, "
            f"exponent={best['exponent']:.3f}, "
            f"host_bonus={best['host_bonus']:.0f}, "
            f"log_loss={best['log_loss']:.4f}, "
            f"brier={best['brier_score']:.4f}"
        )
        print(
            "  This W/D/L grid keeps the scoring baseline fixed and is an "
            "in-sample diagnostic over 96 matches; do not use it as an "
            "automatic production update."
        )


if __name__ == "__main__":
    main()
