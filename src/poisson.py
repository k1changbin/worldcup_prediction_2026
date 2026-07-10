"""Poisson score model utilities.

The public functions in this module deliberately keep the original two-team
API small. Probability calculations use an adaptive score support instead of
discarding everything above a fixed score, while tests can inject an explicit
NumPy random generator when they need controlled draws.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

# pyrefly: ignore [missing-import]
import numpy as np

from src.elo import EloSystem
from src.model_config import BASE_GOALS, ELO_LAMBDA_EXPONENT


_MIN_WIN_PROBABILITY = 1e-6
_TAIL_STANDARD_DEVIATIONS = 12.0
_TAIL_PADDING_GOALS = 12
_NORMAL_APPROXIMATION_VARIANCE = 10_000_000.0
_SQRT_TWO = math.sqrt(2.0)


def _finite_real(value: Real, name: str) -> float:
    """Return *value* as a finite float or raise a helpful exception."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")

    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _expected_goals(value: Real, name: str) -> float:
    converted = _finite_real(value, name)
    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return converted


def elo_expected_score_to_lambdas(
    expected_score: float,
    base_goals: float = BASE_GOALS,
    *,
    exponent: float = ELO_LAMBDA_EXPONENT,
) -> tuple[float, float]:
    """Convert an Elo expected score into expected goals for both teams.

    ``expected_score`` must be in the inclusive interval ``[0, 1]``.  Elo's
    expected score is not a literal win probability: it is the expected share
    of match points before draws are modelled.  The endpoints are clipped
    internally so that an Elo calculation that underflows to exactly zero or
    one still produces finite expected goals.

    ``base_goals`` is the average goals per team and may be zero.  ``exponent``
    controls how strongly the Elo odds ratio changes expected goals and is
    exposed so it can be calibrated by the backtest framework.
    """
    probability = _finite_real(expected_score, "expected_score")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("expected_score must be between 0 and 1")

    baseline = _expected_goals(base_goals, "base_goals")
    calibrated_exponent = _finite_real(exponent, "exponent")
    if calibrated_exponent <= 0.0:
        raise ValueError("exponent must be greater than zero")
    probability = min(
        1.0 - _MIN_WIN_PROBABILITY,
        max(_MIN_WIN_PROBABILITY, probability),
    )

    # Power transform based on the Elo expected-score odds ratio.  This aligns
    # the expected-goal ratio with the Elo strength ratio.
    ratio = probability / (1.0 - probability)
    try:
        lambda_a = baseline * (ratio**calibrated_exponent)
        lambda_b = baseline * ((1.0 / ratio) ** calibrated_exponent)
    except OverflowError as exc:
        raise ValueError("exponent is too large to produce finite lambdas") from exc

    if not math.isfinite(lambda_a) or not math.isfinite(lambda_b):
        raise ValueError("base_goals or exponent is too large to produce finite lambdas")
    return lambda_a, lambda_b


def win_prob_to_lambda(
    win_prob: float,
    base_goals: float = BASE_GOALS,
    *,
    exponent: float = ELO_LAMBDA_EXPONENT,
) -> tuple[float, float]:
    """Backward-compatible wrapper for :func:`elo_expected_score_to_lambdas`."""
    return elo_expected_score_to_lambdas(
        win_prob,
        base_goals,
        exponent=exponent,
    )


def poisson_prob(lam: float, k: int) -> float:
    """Return the probability of scoring exactly ``k`` goals.

    The log-PMF form avoids the intermediate overflow caused by ``lam ** k``
    in the direct formula.
    """
    expected = _expected_goals(lam, "lam")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral):
        raise TypeError("k must be an integer")
    goals = int(k)
    if goals < 0:
        raise ValueError("k must be non-negative")

    if expected == 0.0:
        return 1.0 if goals == 0 else 0.0

    log_probability = (
        -expected
        + goals * math.log(expected)
        - math.lgamma(goals + 1.0)
    )
    return math.exp(log_probability)


def _poisson_support(lam: float, minimum_max_goals: int) -> tuple[int, np.ndarray]:
    """Return a normalized, negligible-tail PMF and its first goal value."""
    if lam == 0.0:
        return 0, np.array([1.0], dtype=float)

    # Twelve standard deviations plus a small fixed margin makes the omitted
    # mass negligible for the football-scale lambdas produced by this model.
    # Starting near the mode also keeps very large means from allocating all
    # the impossible low-score entries between zero and the mode.
    padding = math.ceil(
        _TAIL_STANDARD_DEVIATIONS * math.sqrt(lam) + _TAIL_PADDING_GOALS
    )
    mode = int(math.floor(lam))
    lower = max(0, mode - padding)
    upper = max(minimum_max_goals, mode + padding)

    probabilities = np.empty(upper - lower + 1, dtype=float)
    mode_index = mode - lower
    probabilities[mode_index] = poisson_prob(lam, mode)

    for goals in range(mode, lower, -1):
        index = goals - lower
        probabilities[index - 1] = probabilities[index] * goals / lam

    for goals in range(mode, upper):
        index = goals - lower
        probabilities[index + 1] = probabilities[index] * lam / (goals + 1)

    total = float(np.sum(probabilities, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ArithmeticError("could not construct a stable Poisson distribution")
    probabilities /= total
    return lower, probabilities


def _normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / _SQRT_TWO)


def _large_lambda_probabilities(lambda_a: float, lambda_b: float) -> tuple[float, float, float]:
    """Approximate a very large Skellam distribution with continuity correction."""
    scale = max(lambda_a, lambda_b)
    scaled_a = lambda_a / scale
    scaled_b = lambda_b / scale
    scaled_variance = scaled_a + scaled_b

    mean_over_sigma = (scaled_a - scaled_b) * math.sqrt(scale / scaled_variance)
    half_over_sigma = 0.5 / (math.sqrt(scale) * math.sqrt(scaled_variance))
    lower_z = -mean_over_sigma - half_over_sigma
    upper_z = -mean_over_sigma + half_over_sigma

    lose = _normal_cdf(lower_z)
    draw = max(0.0, _normal_cdf(upper_z) - lose)
    win = max(0.0, 1.0 - _normal_cdf(upper_z))
    return win, draw, lose


def _normalise_outcomes(win: float, draw: float, lose: float) -> dict[str, float]:
    """Normalize three finite outcome masses and correct their rounding residue."""
    outcomes = np.array([win, draw, lose], dtype=float)
    outcomes = np.maximum(outcomes, 0.0)
    total = float(np.sum(outcomes, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ArithmeticError("match outcome probabilities could not be normalized")

    outcomes /= total
    # Put the sub-ULP correction into the largest bucket so small but valid
    # probabilities are not rounded away unnecessarily.
    largest = int(np.argmax(outcomes))
    outcomes[largest] += 1.0 - float(np.sum(outcomes, dtype=np.float64))
    return {
        "win": float(outcomes[0]),
        "draw": float(outcomes[1]),
        "lose": float(outcomes[2]),
    }


def match_probabilities(
    lambda_a: float,
    lambda_b: float,
    max_goals: int = 10,
) -> dict[str, float]:
    """Calculate team-A win, draw, and loss probabilities.

    ``max_goals`` is retained for backward compatibility and acts as a minimum
    score support.  The support expands automatically until the remaining tail
    is negligible, so the returned probabilities are always normalized instead
    of silently losing the mass above a fixed score cutoff.
    """
    expected_a = _expected_goals(lambda_a, "lambda_a")
    expected_b = _expected_goals(lambda_b, "lambda_b")
    if isinstance(max_goals, (bool, np.bool_)) or not isinstance(max_goals, Integral):
        raise TypeError("max_goals must be an integer")
    minimum_max_goals = int(max_goals)
    if minimum_max_goals < 0:
        raise ValueError("max_goals must be non-negative")

    if expected_a == 0.0 and expected_b == 0.0:
        return {"win": 0.0, "draw": 1.0, "lose": 0.0}
    if expected_a == 0.0:
        draw = math.exp(-expected_b)
        return _normalise_outcomes(0.0, draw, 1.0 - draw)
    if expected_b == 0.0:
        draw = math.exp(-expected_a)
        return _normalise_outcomes(1.0 - draw, draw, 0.0)

    if expected_a + expected_b >= _NORMAL_APPROXIMATION_VARIANCE:
        return _normalise_outcomes(
            *_large_lambda_probabilities(expected_a, expected_b)
        )

    lower_a, probability_a = _poisson_support(expected_a, minimum_max_goals)
    lower_b, probability_b = _poisson_support(expected_b, minimum_max_goals)
    upper_a = lower_a + len(probability_a) - 1
    upper_b = lower_b + len(probability_b) - 1

    cdf_a = np.cumsum(probability_a)
    cdf_b = np.cumsum(probability_b)

    goals_a = lower_a + np.arange(len(probability_a))
    b_indices = goals_a - 1 - lower_b
    chance_b_is_lower = np.zeros_like(probability_a)
    chance_b_is_lower[b_indices >= len(probability_b)] = 1.0
    inside_b = (b_indices >= 0) & (b_indices < len(probability_b))
    chance_b_is_lower[inside_b] = cdf_b[b_indices[inside_b]]
    win = float(np.dot(probability_a, chance_b_is_lower))

    goals_b = lower_b + np.arange(len(probability_b))
    a_indices = goals_b - 1 - lower_a
    chance_a_is_lower = np.zeros_like(probability_b)
    chance_a_is_lower[a_indices >= len(probability_a)] = 1.0
    inside_a = (a_indices >= 0) & (a_indices < len(probability_a))
    chance_a_is_lower[inside_a] = cdf_a[a_indices[inside_a]]
    lose = float(np.dot(probability_b, chance_a_is_lower))

    overlap_lower = max(lower_a, lower_b)
    overlap_upper = min(upper_a, upper_b)
    if overlap_lower <= overlap_upper:
        a_start = overlap_lower - lower_a
        b_start = overlap_lower - lower_b
        overlap_length = overlap_upper - overlap_lower + 1
        draw = float(
            np.dot(
                probability_a[a_start : a_start + overlap_length],
                probability_b[b_start : b_start + overlap_length],
            )
        )
    else:
        draw = 0.0

    return _normalise_outcomes(win, draw, lose)


def poisson_modes(lam: float) -> tuple[int, ...]:
    """Return every analytical mode of a Poisson distribution."""
    expected = _expected_goals(lam, "lam")
    if expected == 0.0:
        return (0,)

    upper_mode = int(math.floor(expected))
    if expected.is_integer():
        # Put floor(lambda) first so ``modal_scoreline`` follows the common
        # single-mode convention while still exposing the exact lower tie.
        return (upper_mode, upper_mode - 1)
    return (upper_mode,)


def modal_scorelines(lambda_a: float, lambda_b: float) -> tuple[tuple[int, int], ...]:
    """Return every equally most-likely scoreline analytically."""
    modes_a = poisson_modes(lambda_a)
    modes_b = poisson_modes(lambda_b)
    return tuple((goals_a, goals_b) for goals_a in modes_a for goals_b in modes_b)


def modal_scoreline(lambda_a: float, lambda_b: float) -> tuple[int, int]:
    """Return one deterministic analytical mode for the match scoreline.

    Positive integer lambdas have two equally likely Poisson modes.  In that
    uncommon tie, this convenience function chooses ``floor(lambda)`` for a
    stable result; use :func:`modal_scorelines` when every tied mode is needed.
    """
    return modal_scorelines(lambda_a, lambda_b)[0]


def simulate_match_score(
    lambda_a: float,
    lambda_b: float,
    *,
    rng: Any | None = None,
) -> tuple[int, int]:
    """Sample a football score from each team's Poisson expected goals.

    An existing NumPy-compatible ``rng`` can be injected by tests. Supplying
    none creates a fresh generator for an ordinary non-deterministic draw.
    """
    expected_a = _expected_goals(lambda_a, "lambda_a")
    expected_b = _expected_goals(lambda_b, "lambda_b")

    random_source = rng if rng is not None else np.random.default_rng()
    poisson = getattr(random_source, "poisson", None)
    if not callable(poisson):
        raise TypeError("rng must provide a callable poisson method")

    try:
        score_a = poisson(expected_a)
        score_b = poisson(expected_b)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"could not sample Poisson score: {exc}") from exc
    return int(score_a), int(score_b)


if __name__ == "__main__":
    elo = EloSystem()
    elo.load_ratings()

    team_a, team_b = "Brazil", "South Korea"
    expected_score = elo.expected_score(
        elo.get_rating(team_a),
        elo.get_rating(team_b),
    )

    lambda_a, lambda_b = elo_expected_score_to_lambdas(expected_score)
    result = match_probabilities(lambda_a, lambda_b)

    print(f"{team_a} vs {team_b}")
    print(f"Expected goals - {team_a}: {lambda_a:.2f}, {team_b}: {lambda_b:.2f}")
    print(f"{team_a} win: {result['win']:.1%}")
    print(f"Draw: {result['draw']:.1%}")
    print(f"{team_b} win: {result['lose']:.1%}")

    # Example simulated match result.
    sim_score_a, sim_score_b = simulate_match_score(lambda_a, lambda_b)
    print("\n--- [One-Match Simulation] ---")
    print(f"Final score: {team_a} {sim_score_a} : {sim_score_b} {team_b}")
