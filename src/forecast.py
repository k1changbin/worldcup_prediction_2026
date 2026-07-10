"""Shared full-tournament Monte Carlo aggregation utilities."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable


STAGES = ("R32", "R16", "QF", "SF", "F", "Champion")


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054):
    """Return a Wilson score confidence interval for a binomial proportion."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")

    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    centre = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    low = max(0.0, centre - margin)
    high = min(1.0, centre + margin)
    if successes == 0:
        low = 0.0
    if successes == trials:
        high = 1.0
    return low, high


def run_tournament_forecast(
    simulation,
    iterations: int,
    teams: Iterable[str] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
):
    """Run a tournament simulation repeatedly and aggregate stage probabilities.

    ``simulation`` is a configured ``WorldCupSimulation`` instance. Completed
    matches remain fixed by that instance, while all remaining matches are
    sampled independently on every iteration.
    """
    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    if teams is None:
        team_order = [
            team
            for group_teams in simulation.groups.values()
            for team in group_teams
        ]
    else:
        team_order = list(dict.fromkeys(teams))

    counts = {
        team: {stage: 0 for stage in STAGES}
        for team in team_order
    }

    for completed in range(1, iterations + 1):
        standings = simulation.simulate_group_stage()
        advancing = simulation.get_advancing_teams(standings)
        for team in advancing:
            if team in counts:
                counts[team]["R32"] += 1

        knockout = simulation.simulate_knockout_stage()
        for match in knockout["Round of 32"]:
            if match["winner"] in counts:
                counts[match["winner"]]["R16"] += 1
        for match in knockout["Round of 16"]:
            if match["winner"] in counts:
                counts[match["winner"]]["QF"] += 1
        for match in knockout["Quarter-finals"]:
            if match["winner"] in counts:
                counts[match["winner"]]["SF"] += 1
        for match in knockout["Semi-finals"]:
            if match["winner"] in counts:
                counts[match["winner"]]["F"] += 1
        champion = knockout["Champion"]
        if champion in counts:
            counts[champion]["Champion"] += 1

        if progress_callback is not None:
            progress_callback(completed, iterations)

    records = []
    for team in team_order:
        stage_values = {}
        intervals = {}
        for stage in STAGES:
            stage_values[stage] = counts[team][stage] / iterations
            intervals[stage] = wilson_interval(counts[team][stage], iterations)
        records.append(
            {
                "team": team,
                "counts": counts[team],
                "probabilities": stage_values,
                "confidence_intervals": intervals,
            }
        )

    records.sort(
        key=lambda row: tuple(row["counts"][stage] for stage in reversed(STAGES)),
        reverse=True,
    )
    return records
