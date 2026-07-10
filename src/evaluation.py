"""Retrospective and current diagnostics for the Elo-to-Poisson model."""

from __future__ import annotations

import math

from src.elo import EloSystem
from src.model_config import BASE_GOALS, ELO_LAMBDA_EXPONENT, HOST_ADVANTAGE_ELO
from src.poisson import (
    elo_expected_score_to_lambdas,
    match_probabilities,
    poisson_prob,
)


def predict_lambdas(
    record,
    base_goals=BASE_GOALS,
    exponent=ELO_LAMBDA_EXPONENT,
    host_bonus=HOST_ADVANTAGE_ELO,
):
    rating_a = float(record["elo_a"])
    rating_b = float(record["elo_b"])
    if record.get("host_a") and not record.get("host_b"):
        rating_a += host_bonus
    elif record.get("host_b") and not record.get("host_a"):
        rating_b += host_bonus

    expected_score = EloSystem.expected_score(rating_a, rating_b)
    lambda_a, lambda_b = elo_expected_score_to_lambdas(
        expected_score,
        base_goals,
        exponent=exponent,
    )
    return lambda_a, lambda_b


def predict_outcomes(
    record,
    base_goals=BASE_GOALS,
    exponent=ELO_LAMBDA_EXPONENT,
    host_bonus=HOST_ADVANTAGE_ELO,
):
    lambda_a, lambda_b = predict_lambdas(
        record,
        base_goals=base_goals,
        exponent=exponent,
        host_bonus=host_bonus,
    )
    return match_probabilities(lambda_a, lambda_b)


def outcome_label(record):
    if record["score_a"] > record["score_b"]:
        return "win"
    if record["score_a"] < record["score_b"]:
        return "lose"
    return "draw"


def build_current_group_records(results, pre_tournament_ratings, host_teams):
    """Build leakage-free records from the current tournament group stage."""
    records = []
    for match in results:
        if match.get("stage", "group") != "group":
            continue
        team_a = match["team_a"]
        team_b = match["team_b"]
        if team_a not in pre_tournament_ratings or team_b not in pre_tournament_ratings:
            raise KeyError(f"Pre-tournament Elo is missing for {team_a} or {team_b}")
        record = {
            "year": 2026,
            "date": match["date"],
            "team_a": team_a,
            "team_b": team_b,
            "score_a": match["score_a"],
            "score_b": match["score_b"],
            "elo_a": pre_tournament_ratings[team_a],
            "elo_b": pre_tournament_ratings[team_b],
            "host_a": team_a in host_teams,
            "host_b": team_b in host_teams,
        }
        # The live Elo feed may put a winner first. Canonical ordering keeps
        # side assignment independent of the observed result.
        if record["team_a"] > record["team_b"]:
            for left, right in (
                ("team_a", "team_b"),
                ("score_a", "score_b"),
                ("elo_a", "elo_b"),
                ("host_a", "host_b"),
            ):
                record[left], record[right] = record[right], record[left]
        records.append(record)
    return records


def evaluate_parameters(
    records,
    base_goals=BASE_GOALS,
    exponent=ELO_LAMBDA_EXPONENT,
    host_bonus=HOST_ADVANTAGE_ELO,
):
    records = list(records)
    if not records:
        raise ValueError("at least one calibration match is required")

    log_loss = 0.0
    scoreline_log_loss = 0.0
    brier_score = 0.0
    correct = 0
    predicted_totals = {key: 0.0 for key in ("win", "draw", "lose")}
    observed_totals = {key: 0 for key in ("win", "draw", "lose")}
    predicted_goal_total = 0.0
    observed_goal_total = 0

    for record in records:
        lambda_a, lambda_b = predict_lambdas(
            record,
            base_goals=base_goals,
            exponent=exponent,
            host_bonus=host_bonus,
        )
        probabilities = match_probabilities(lambda_a, lambda_b)
        observed = outcome_label(record)
        log_loss -= math.log(max(probabilities[observed], 1e-15))
        score_probability = poisson_prob(
            lambda_a,
            record["score_a"],
        ) * poisson_prob(lambda_b, record["score_b"])
        scoreline_log_loss -= math.log(max(score_probability, 1e-15))
        predicted_goal_total += lambda_a + lambda_b
        observed_goal_total += record["score_a"] + record["score_b"]
        for label, probability in probabilities.items():
            target = 1.0 if label == observed else 0.0
            brier_score += (probability - target) ** 2
            predicted_totals[label] += probability
        observed_totals[observed] += 1
        correct += max(probabilities, key=probabilities.get) == observed

    count = len(records)
    return {
        "matches": count,
        "log_loss": log_loss / count,
        "scoreline_log_loss": scoreline_log_loss / count,
        "brier_score": brier_score / count,
        "accuracy": correct / count,
        "mean_predicted": {
            label: total / count for label, total in predicted_totals.items()
        },
        "observed_frequency": {
            label: total / count for label, total in observed_totals.items()
        },
        "mean_predicted_goals": predicted_goal_total / count,
        "mean_observed_goals": observed_goal_total / count,
    }


def grid_search(
    records,
    base_goals_values,
    exponent_values,
    host_bonus_values,
):
    """Return candidate parameters sorted by multiclass log loss."""
    candidates = []
    for base_goals in base_goals_values:
        for exponent in exponent_values:
            for host_bonus in host_bonus_values:
                metrics = evaluate_parameters(
                    records,
                    base_goals=base_goals,
                    exponent=exponent,
                    host_bonus=host_bonus,
                )
                candidates.append(
                    {
                        "base_goals": float(base_goals),
                        "exponent": float(exponent),
                        "host_bonus": float(host_bonus),
                        **metrics,
                    }
                )
    candidates.sort(key=lambda row: (row["log_loss"], row["brier_score"]))
    return candidates
