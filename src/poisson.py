import math
# pyrefly: ignore [missing-import]
import numpy as np
from src.elo import EloSystem


def win_prob_to_lambda(win_prob: float, base_goals: float = 1.35) -> tuple:
    """
    Elo 승리 확률을 양 팀 예상 득점(λ)으로 변환
    base_goals: 평균 득점 기준값 (월드컵 평균 약 1.35골, 경기당 총합 2.7골)
    """
    eps = 1e-6
    p = max(eps, min(1.0 - eps, win_prob))
    
    # ELO 기대 승률 비율 기반의 거듭제곱 공식 적용 (지수 = 0.376)
    # 두 팀의 expected goals 비율이 ELO 기대 승률 비율과 매칭되도록 보정
    ratio = p / (1.0 - p)
    lambda_a = base_goals * (ratio ** 0.376)
    lambda_b = base_goals * ((1.0 / ratio) ** 0.376)
    return lambda_a, lambda_b


def poisson_prob(lam: float, k: int) -> float:
    """팀이 정확히 k골 넣을 확률"""
    return (lam ** k * math.exp(-lam)) / math.factorial(k)


def match_probabilities(lambda_a: float, lambda_b: float, max_goals: int = 10) -> dict:
    """
    두 팀의 예상 득점으로 승/무/패 확률 계산
    max_goals: 고려할 최대 득점 수
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
    푸아송 분포를 바탕으로 양 팀의 예상 득점(람다)을 실제 스코어로 무작위 추출합니다.
    반환값: (팀 A 득점, 팀 B 득점)
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
    print(f"예상 득점 — {team_a}: {lambda_a:.2f}골, {team_b}: {lambda_b:.2f}골")
    print(f"{team_a} 승리: {result['win']:.1%}")
    print(f"무승부: {result['draw']:.1%}")
    print(f"{team_b} 승리: {result['lose']:.1%}")

    # 실제 경기 시뮬레이션 결과 예시
    sim_score_a, sim_score_b = simulate_match_score(lambda_a, lambda_b)
    print("\n--- [시뮬레이션 1경기 진행] ---")
    print(f"최종 스코어: {team_a} {sim_score_a} : {sim_score_b} {team_b}")