"""Command-line full-tournament Monte Carlo forecast."""

import argparse
import json

from src.factory import create_world_cup_simulation
from src.forecast import run_tournament_forecast
from src.paths import data_path
from src.tournament_state import get_active_teams


def create_simulation():
    simulation = create_world_cup_simulation()
    return simulation.elo_system, simulation


def run_monte_carlo(iterations=10000, show_confidence=True):
    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    elo, simulation = create_simulation()
    results_path = data_path("actual_results.json")
    with results_path.open(encoding="utf-8") as handle:
        actual_results = json.load(handle)
    if actual_results:
        print(f"[Actual results locked] ({len(actual_results)} matches)")

    active_teams = get_active_teams(
        simulation.groups,
        simulation.actual_results,
        elo_ratings=elo.ratings,
    )
    if not active_teams:
        active_teams = set(elo.ratings)

    print(f"[Simulation started] Running {iterations:,} Monte Carlo simulations")
    progress_step = max(1, iterations // 10)

    def report_progress(completed, total):
        if completed == total or completed % progress_step == 0:
            print(f"Progress: {completed / total * 100:.0f}% complete")

    forecast = run_tournament_forecast(
        simulation,
        iterations,
        teams=sorted(active_teams),
        progress_callback=report_progress,
    )

    print("\n[Monte Carlo Simulation Results] (sorted by championship probability)")
    header = (
        f"{'Rank':<4} {'Team':<24} {'Champion':<9} {'Final':<7} "
        f"{'SF':<7} {'QF':<7} {'R16':<7} {'R32':<7}"
    )
    if show_confidence:
        header += " Champion 95% CI"
    print(header)
    print("-" * len(header))

    for rank, row in enumerate(forecast, 1):
        probabilities = row["probabilities"]
        line = (
            f"{rank:<4} {row['team']:<24} "
            f"{probabilities['Champion'] * 100:>5.1f}% "
            f"{probabilities['F'] * 100:>6.1f}% "
            f"{probabilities['SF'] * 100:>6.1f}% "
            f"{probabilities['QF'] * 100:>6.1f}% "
            f"{probabilities['R16'] * 100:>6.1f}% "
            f"{probabilities['R32'] * 100:>6.1f}%"
        )
        if show_confidence:
            low, high = row["confidence_intervals"]["Champion"]
            line += f" {low * 100:>6.2f}%–{high * 100:.2f}%"
        print(line)
    if show_confidence:
        print(
            "\n95% intervals quantify Monte Carlo sampling uncertainty only; "
            "they do not include model or input-data uncertainty."
        )
    return forecast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--no-confidence", action="store_true")
    args = parser.parse_args()
    run_monte_carlo(
        iterations=args.iterations,
        show_confidence=not args.no_confidence,
    )


if __name__ == "__main__":
    main()
