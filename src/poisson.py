import math
# pyrefly: ignore [missing-import]
import numpy as np
from src.elo import EloSystem


def win_prob_to_lambda(win_prob: float, base_goals: float = 1.35) -> tuple:
    """
    Convert Elo win probability into expected goals for both teams.
    base_goals: average goals per team, defaulting to about 1.35 for World Cup matches.
    """
    eps = 1e-6
    p = max(eps, min(1.0 - eps, win_prob))
    
    # Power transform based on the Elo expected-score odds ratio.
    # This aligns the expected-goal ratio with the Elo strength ratio.
    ratio = p / (1.0 - p)
    lambda_a = base_goals * (ratio ** 0.376)
    lambda_b = base_goals * ((1.0 / ratio) ** 0.376)
    return lambda_a, lambda_b


def poisson_prob(lam: float, k: int) -> float:
    """Return the probability of scoring exactly k goals."""
    return (lam ** k * math.exp(-lam)) / math.factorial(k)


def match_probabilities(lambda_a: float, lambda_b: float, max_goals: int = 10) -> dict:
    """
    Calculate win/draw/loss probabilities from both teams' expected goals.
    max_goals: maximum scoreline considered for each team.
    """
    win, draw, lose = 0.0, 0.0, 0.0

    for g_a in range(max_goals + 1):
        for g_b in range(max_goals + 1):
            prob = poisson_prob(lambda_a, g_a) * poisson_prob(lambda_b, g_b)
            if g_a > g_b:
                win += prob
            elif g_a == g_b:
                draw += prob
            else:
                lose += prob

    return {"win": win, "draw": draw, "lose": lose}


def simulate_match_score(lambda_a: float, lambda_b: float) -> tuple:
    """
    Sample a football score from each team's Poisson expected goals.
    Returns: (team A goals, team B goals)
    """
    score_a = np.random.poisson(lambda_a)
    score_b = np.random.poisson(lambda_b)
    return score_a, score_b


if __name__ == "__main__":
    elo = EloSystem()
    elo.load_ratings()

    team_a, team_b = "Brazil", "South Korea"
    win_prob = elo.expected_score(
        elo.get_rating(team_a),
        elo.get_rating(team_b)
    )

    lambda_a, lambda_b = win_prob_to_lambda(win_prob)
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
