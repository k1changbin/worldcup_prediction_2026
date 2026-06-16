import json
import os
import sys
from collections import Counter
import numpy as np
from src.elo import EloSystem
from src.poisson import win_prob_to_lambda, match_probabilities, simulate_match_score

def find_team(query, valid_teams):
    query = query.strip().lower()
    # 1. 완전 일치
    for team in valid_teams:
        if team.lower() == query:
            return team
    # 2. 부분 일치
    matches = [team for team in valid_teams if query in team.lower()]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        return matches
    return None

def draw_bar(pct, width=20):
    filled = int(round(width * (pct / 100)))
    return "█" * filled + "░" * (width - filled)

def predict_match():
    # ELO 레이팅 파일 로드
    elo_path = "data/elo_ratings.json"
    if not os.path.exists(elo_path):
        print("[에러] data/elo_ratings.json 파일을 찾을 수 없습니다.")
        return
        
    with open(elo_path, "r", encoding="utf-8") as f:
        ratings = json.load(f)
        valid_teams = set(ratings.keys())
        
    # 커맨드라인 인자 확인 또는 입력 받기
    team_a_query = ""
    team_b_query = ""
    
    if len(sys.argv) >= 3:
        team_a_query = sys.argv[1]
        team_b_query = sys.argv[2]
    else:
        print("[2026 월드컵 매치 승부예측기]")
        print("참가국 중 두 팀을 입력해 승률을 예측해보세요.")
        team_a_query = input("첫 번째 팀명 입력 (영어): ").strip()
        team_b_query = input("두 번째 팀명 입력 (영어): ").strip()
        
    if not team_a_query or not team_b_query:
        print("[에러] 두 팀의 이름을 모두 입력해야 합니다.")
        return
        
    team_a = find_team(team_a_query, valid_teams)
    team_b = find_team(team_b_query, valid_teams)
    
    # 예외 처리: 다중 매칭 혹은 매칭 실패
    if isinstance(team_a, list):
        print(f"[경고] 첫 번째 팀명이 모호합니다. 다음 중 하나인가요? {team_a}")
        return
    if isinstance(team_b, list):
        print(f"[경고] 두 번째 팀명이 모호합니다. 다음 중 하나인가요? {team_b}")
        return
    if not team_a:
        print(f"[에러] 첫 번째 팀 '{team_a_query}'을(를) 찾을 수 없습니다. 참가국 스펠링을 확인하세요.")
        return
    if not team_b:
        print(f"[에러] 두 번째 팀 '{team_b_query}'을(를) 찾을 수 없습니다. 참가국 스펠링을 확인하세요.")
        return
    if team_a == team_b:
        print("[에러] 서로 다른 두 팀을 입력해야 합니다.")
        return

    # ELO 조회 및 계산
    elo = EloSystem()
    elo.load_ratings(elo_path)
    
    rating_a = elo.get_rating(team_a)
    rating_b = elo.get_rating(team_b)
    
    # 기대 승률 계산 (동일 전력 시 0.5)
    win_prob_a = elo.expected_score(rating_a, rating_b)
    
    # 푸아송 람다 계산 (득점력)
    lambda_a, lambda_b = win_prob_to_lambda(win_prob_a)
    
    # 승무패 확률 계산
    result = match_probabilities(lambda_a, lambda_b)
    win_pct = result["win"] * 100
    draw_pct = result["draw"] * 100
    lose_pct = result["lose"] * 100
    
    # 화면 출력
    print("\n" + "=" * 55)
    print(f"[승부 예측 결과] {team_a} vs {team_b}")
    print("=" * 55)
    print(f"[Elo Rating] {team_a} ({rating_a:.1f}) vs {team_b} ({rating_b:.1f})")
    print(f"[평균 예상 득점] {team_a}: {lambda_a:.2f}골 | {team_b}: {lambda_b:.2f}골")
    print("-" * 55)
    
    print(f"[{team_a} 승리] {win_pct:>5.1f}%  {draw_bar(win_pct)}")
    print(f"[ 무  승  부 ] {draw_pct:>5.1f}%  {draw_bar(draw_pct)}")
    print(f"[{team_b} 승리] {lose_pct:>5.1f}%  {draw_bar(lose_pct)}")
    print("-" * 55)
    
    # 모의 경기 1,000,000회 시뮬레이션을 통해 가장 확률 높은(최빈) 스코어 산출
    sim_runs = 1000000
    sa_samples = np.random.poisson(lambda_a, sim_runs)
    sb_samples = np.random.poisson(lambda_b, sim_runs)
    
    score_counts = Counter(zip(sa_samples, sb_samples))
    (best_sa, best_sb), count = max(score_counts.items(), key=lambda x: x[1])
    score_prob = (count / sim_runs) * 100
    
    print(f"[가장 확률 높은 예상 스코어 (1,000,000회 ELO 시뮬레이션 기준)]")
    print(f"   * {team_a} {best_sa} - {best_sb} {team_b}  (약 {score_prob:.1f}% 확률)")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    predict_match()
