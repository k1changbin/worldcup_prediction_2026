import sys
import os
import json
from collections import defaultdict
from src.elo import EloSystem
from src.simulation import WorldCupSimulation

def run_monte_carlo(iterations=10000):
    elo = EloSystem()
    elo.load_ratings("data/elo_ratings.json")
    
    actual_results_path = "data/actual_results.json"
    if os.path.exists(actual_results_path):
        try:
            with open(actual_results_path, "r", encoding="utf-8") as f:
                actual_results = json.load(f)
                if actual_results:
                    print(f"[실제 경기 결과 고정] ({len(actual_results)}경기)...")
        except json.JSONDecodeError as e:
            print(f"[오류] 실제 경기 결과 로드 중 오류 발생: {e}")
            
    sim = WorldCupSimulation(elo, "data/groups.json", actual_results_path)
    
    # 통계 저장용 딕셔너리 (48개 참가국 전체로 초기화)
    stats = {}
    for group_teams in sim.groups.values():
        for team in group_teams:
            stats[team] = {"R32": 0, "R16": 0, "QF": 0, "SF": 0, "F": 0, "Champion": 0}
    
    print(f"[시뮬레이션 시작] {iterations}번의 몬테카를로 시뮬레이션을 시작합니다... (약 10~30초 소요 예상)")
    
    for i in range(iterations):
        # 1. 조별 리그 진행
        standings = sim.simulate_group_stage()
        
        # 2. 32강 진출팀 확정
        advancing_teams = sim.get_advancing_teams(standings)
        for team in advancing_teams:
            stats[team]["R32"] += 1
            
        # 3. 토너먼트 진행
        knockout_results = sim.simulate_knockout_stage(advancing_teams)
        
        # 4. 결과 집계
        for match in knockout_results["Round of 32"]:
            stats[match["winner"]]["R16"] += 1
            
        for match in knockout_results["Round of 16"]:
            stats[match["winner"]]["QF"] += 1
            
        for match in knockout_results["Quarter-finals"]:
            stats[match["winner"]]["SF"] += 1
            
        for match in knockout_results["Semi-finals"]:
            stats[match["winner"]]["F"] += 1
            
        champion = knockout_results["Champion"]
        stats[champion]["Champion"] += 1
        
        # 진행 상황 출력 (10% 단위)
        if (i + 1) % (iterations // 10) == 0:
            print(f"진행도: {(i + 1) / iterations * 100:.0f}% 완료")
            
    # 확률 계산 및 정렬 (우승 확률 -> 결승 -> 4강 -> 8강 -> 16강 -> 32강 순으로 정렬)
    print("\n[몬테카를로 시뮬레이션 결과] (우승 확률 순)")
    print(f"{'순위':<4} {'팀명':<15} {'우승':<7} {'결승':<7} {'4강':<7} {'8강':<7} {'16강':<7} {'32강':<7}")
    print("-" * 70)
    
    sorted_teams = sorted(
        stats.items(), 
        key=lambda x: (x[1]["Champion"], x[1]["F"], x[1]["SF"], x[1]["QF"], x[1]["R16"], x[1]["R32"]), 
        reverse=True
    )
    
    for rank, (team, s) in enumerate(sorted_teams, 1):
        champ_pct = s["Champion"] / iterations * 100
        f_pct = s["F"] / iterations * 100
        sf_pct = s["SF"] / iterations * 100
        qf_pct = s["QF"] / iterations * 100
        r16_pct = s["R16"] / iterations * 100
        r32_pct = s["R32"] / iterations * 100
        
        print(f"{rank:<4} {team:<15} {champ_pct:>5.1f}% {f_pct:>6.1f}% {sf_pct:>6.1f}% {qf_pct:>6.1f}% {r16_pct:>6.1f}% {r32_pct:>6.1f}%")

if __name__ == "__main__":
    run_monte_carlo(10000)
