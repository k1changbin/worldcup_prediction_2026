import json
import os
import sys
from collections import Counter
import math
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

def get_injury_multipliers(team, injuries_dict):
    squads_path = "data/squads.json"
    squads = {}
    if os.path.exists(squads_path):
        with open(squads_path, "r", encoding="utf-8") as f:
            try:
                squads = json.load(f)
            except json.JSONDecodeError:
                pass
                
    raw_injuries = injuries_dict.get(team, [])
    team_injuries = []
    if isinstance(raw_injuries, list):
        for item in raw_injuries:
            if isinstance(item, dict):
                team_injuries.append(item.get("name"))
            else:
                team_injuries.append(str(item))
    else:
        team_injuries = [str(raw_injuries)]
    team_injuries = [n for n in team_injuries if n]

    if not team_injuries or team not in squads:
        return 1.0, 1.0, []

    players = squads[team]
    total_value = sum(p["value_eur"] for p in players)
    if total_value == 0:
        return 1.0, 1.0, []

    # HHI
    hhi = sum((p["value_eur"] / total_value) ** 2 for p in players)
    min_hhi = 0.0385
    max_hhi = 0.3000
    norm_hhi = max(0.0, min(1.0, (hhi - min_hhi) / (max_hhi - min_hhi)))
    depth_factor = 0.2 + 0.8 * norm_hhi

    attack_total = 0.0
    defense_total = 0.0
    for p in players:
        pos = p["position"]
        val = p["value_eur"]
        if pos in ["Goalkeeper", "Defender"]:
            defense_total += val
        else:
            attack_total += val

    attack_reduction = 0.0
    defense_reduction = 0.0
    details = []

    for player_name in team_injuries:
        matched_player = None
        for p in players:
            if p["name"].strip().lower() == player_name.strip().lower():
                matched_player = p
                break
        
        if matched_player:
            pos = matched_player["position"]
            val = matched_player["value_eur"]
            pos_str = "수비" if pos in ["Goalkeeper", "Defender"] else "공격"
            
            if pos in ["Goalkeeper", "Defender"]:
                if defense_total > 0:
                    share = val / defense_total
                    reduction = share * depth_factor
                    defense_reduction += reduction
                    details.append(f"{matched_player['name']} ({pos_str}, 가치: €{val/1000000:.1f}M, 포지션 비중: {share*100:.1f}%, 누수율: {reduction*100:.1f}%)")
            else:
                if attack_total > 0:
                    share = val / attack_total
                    reduction = share * depth_factor
                    attack_reduction += reduction
                    details.append(f"{matched_player['name']} ({pos_str}, 가치: €{val/1000000:.1f}M, 포지션 비중: {share*100:.1f}%, 누수율: {reduction*100:.1f}%)")

    attack_multiplier = max(0.5, 1.0 - attack_reduction)
    defense_multiplier = min(2.0, 1.0 + defense_reduction)
    return attack_multiplier, defense_multiplier, details

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
    
    rest_days_diff = 0
    travel_fatigue_a = 0.0
    travel_fatigue_b = 0.0
    
    if len(sys.argv) >= 3:
        team_a_query = sys.argv[1]
        team_b_query = sys.argv[2]
        
        # 추가 인자 파싱 (휴식일 격차, 피로도 A, 피로도 B)
        # 형식: predict_match.py "Korea" "Spain" [rest_days_diff] [fatigue_a] [fatigue_b]
        if len(sys.argv) >= 4:
            try:
                rest_days_diff = int(sys.argv[3])
            except ValueError:
                pass
        if len(sys.argv) >= 6:
            try:
                travel_fatigue_a = float(sys.argv[4])
                travel_fatigue_b = float(sys.argv[5])
            except ValueError:
                pass
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
    
    # 1. 개최국 홈 이점 적용 (+40 ELO)
    from src.simulation import HOST_COUNTRIES
    is_host_a = team_a in HOST_COUNTRIES
    is_host_b = team_b in HOST_COUNTRIES
    home_adv_msg = ""
    
    if is_host_a and not is_host_b:
        rating_a += 40
        home_adv_msg = f" * {team_a} 개최국 홈 우위 버프 적용 (ELO +40)"
    elif is_host_b and not is_host_a:
        rating_b += 40
        home_adv_msg = f" * {team_b} 개최국 홈 우위 버프 적용 (ELO +40)"
        
    # 2. 휴식일 체력 격차 보정 적용 (하루당 +5, 최대 +30 ELO)
    rest_bonus = min(abs(rest_days_diff) * 5, 30)
    if rest_days_diff >= 1:
        rating_a += rest_bonus
    elif rest_days_diff <= -1:
        rating_b += rest_bonus
        
    # 기대 승률 계산
    win_prob_a = elo.expected_score(rating_a, rating_b)
    
    # 푸아송 람다 계산 (기본 득점력)
    lambda_a, lambda_b = win_prob_to_lambda(win_prob_a)
    
    # 부상자 로드
    injuries_path = "data/absences.json"
    injuries = {}
    if os.path.exists(injuries_path):
        with open(injuries_path, "r", encoding="utf-8") as f:
            try:
                injuries = json.load(f)
            except json.JSONDecodeError:
                pass
                
    att_mult_a, def_mult_a, details_a = get_injury_multipliers(team_a, injuries)
    att_mult_b, def_mult_b, details_b = get_injury_multipliers(team_b, injuries)
    
    # 3. 부상 배율 및 4. 이동 피로도 적용
    final_lambda_a = lambda_a * att_mult_a * def_mult_b * (1.0 - travel_fatigue_a)
    final_lambda_b = lambda_b * att_mult_b * def_mult_a * (1.0 - travel_fatigue_b)
    
    # 보정된 평균 예상 득점으로 승무패 확률 계산
    result = match_probabilities(final_lambda_a, final_lambda_b)
    win_pct = result["win"] * 100
    draw_pct = result["draw"] * 100
    lose_pct = result["lose"] * 100
    
    # 화면 출력
    print("\n" + "=" * 55)
    print(f"[승부 예측 결과] {team_a} vs {team_b}")
    print("=" * 55)
    print(f"[Elo Rating] {team_a} ({elo.get_rating(team_a):.1f}) vs {team_b} ({elo.get_rating(team_b):.1f})")
    
    # 환경 변수 보정 현황 출력
    if home_adv_msg or rest_days_diff != 0 or travel_fatigue_a > 0 or travel_fatigue_b > 0:
        print("-" * 55)
        print("[환경 변수 및 일정 보정 현황]")
        if home_adv_msg:
            print(home_adv_msg)
        if rest_days_diff > 0:
            print(f" * {team_a} 체력 우위 적용 (상대보다 휴식일 +{rest_days_diff}일, ELO +{rest_bonus})")
        elif rest_days_diff < 0:
            print(f" * {team_b} 체력 우위 적용 (상대보다 휴식일 +{-rest_days_diff}일, ELO +{rest_bonus})")
        if travel_fatigue_a > 0:
            print(f" * {team_a} 이동 피로도/로테이션 적용 (득점력 감쇄: -{travel_fatigue_a*100:.1f}%)")
        if travel_fatigue_b > 0:
            print(f" * {team_b} 이동 피로도/로테이션 적용 (득점력 감쇄: -{travel_fatigue_b*100:.1f}%)")
            
    # 부상자 정보 섹션 출력
    if details_a or details_b:
        print("-" * 55)
        print("[부상 정보 및 전력 누수 현황]")
        squads_path = "data/squads.json"
        squads = {}
        if os.path.exists(squads_path):
            with open(squads_path, "r", encoding="utf-8") as f:
                try:
                    squads = json.load(f)
                except:
                    pass
        
        if details_a:
            players_a = squads.get(team_a, [])
            total_raw_val_a = sum(p["value_eur"] for p in players_a)
            total_val_a = total_raw_val_a / 1000000
            hhi_a = sum((p["value_eur"] / total_raw_val_a) ** 2 for p in players_a) if total_raw_val_a > 0 else 0
            norm_hhi_a = max(0.0, min(1.0, (hhi_a - 0.0385) / (0.3 - 0.0385)))
            dependency_a = 0.2 + 0.8 * norm_hhi_a
            print(f" * {team_a} (스쿼드 가치: €{total_val_a:.1f}M, 에이스 의존도 계수: {dependency_a:.2f}):")
            for detail in details_a:
                print(f"   - {detail}")
        if details_b:
            players_b = squads.get(team_b, [])
            total_raw_val_b = sum(p["value_eur"] for p in players_b)
            total_val_b = total_raw_val_b / 1000000
            hhi_b = sum((p["value_eur"] / total_raw_val_b) ** 2 for p in players_b) if total_raw_val_b > 0 else 0
            norm_hhi_b = max(0.0, min(1.0, (hhi_b - 0.0385) / (0.3 - 0.0385)))
            dependency_b = 0.2 + 0.8 * norm_hhi_b
            print(f" * {team_b} (스쿼드 가치: €{total_val_b:.1f}M, 에이스 의존도 계수: {dependency_b:.2f}):")
            for detail in details_b:
                print(f"   - {detail}")
                
    print("-" * 55)
    print(f"[평균 예상 득점] {team_a}: {final_lambda_a:.2f}골 | {team_b}: {final_lambda_b:.2f}골")
    
    # 보정 전 ELO 득점력이랑 차이가 나는 경우에만 출력
    is_modified = (
        not math.isclose(rating_a, elo.get_rating(team_a)) or 
        not math.isclose(rating_b, elo.get_rating(team_b)) or
        not math.isclose(final_lambda_a, lambda_a) or 
        not math.isclose(final_lambda_b, lambda_b)
    )
    if is_modified:
        base_win_prob = elo.expected_score(elo.get_rating(team_a), elo.get_rating(team_b))
        base_lam_a, base_lam_b = win_prob_to_lambda(base_win_prob)
        print(f"   (보정 전 순수 ELO 기준 - {team_a}: {base_lam_a:.2f}골 | {team_b}: {base_lam_b:.2f}골)")
        
    print("-" * 55)
    print(f"[{team_a} 승리] {win_pct:>5.1f}%  {draw_bar(win_pct)}")
    print(f"[ 무  승  부 ] {draw_pct:>5.1f}%  {draw_bar(draw_pct)}")
    print(f"[{team_b} 승리] {lose_pct:>5.1f}%  {draw_bar(lose_pct)}")
    print("-" * 55)
    
    # 모의 경기 1,000,000회 시뮬레이션을 통해 가장 확률 높은(최빈) 스코어 산출
    sim_runs = 1000000
    sa_samples = np.random.poisson(final_lambda_a, sim_runs)
    sb_samples = np.random.poisson(final_lambda_b, sim_runs)
    
    score_counts = Counter(zip(sa_samples, sb_samples))
    (best_sa, best_sb), count = max(score_counts.items(), key=lambda x: x[1])
    score_prob = (count / sim_runs) * 100
    
    print(f"[가장 확률 높은 예상 스코어 (1,000,000회 ELO 시뮬레이션 기준)]")
    print(f"   * {team_a} {best_sa} - {best_sb} {team_b}  (약 {score_prob:.1f}% 확률)")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    predict_match()
