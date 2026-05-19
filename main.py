import sys
import os
from collections import defaultdict
from src.elo import EloSystem
from src.simulation import WorldCupSimulation

def run_monte_carlo(iterations=10000):
    # PYTHONPATH 문제 해결을 위해 명시적으로 설정이 필요할 수 있지만
    # 프로젝트 최상단에서 실행한다고 가정합니다.
    
    elo = EloSystem()
    elo.load_ratings("data/elo_ratings.json")
    
    sim = WorldCupSimulation(elo, "data/groups.json")
    
    # 통계 저장용 딕셔너리
    stats = defaultdict(lambda: {"R32": 0, "R16": 0, "QF": 0, "SF": 0, "F": 0, "Champion": 0})
    
    print(f"🔥 {iterations}번의 몬테카를로 시뮬레이션을 시작합니다... (약 10~30초 소요 예상)")
    
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
            
    # 확률 계산 및 정렬 (우승 확률 순)
    print("\n📊 몬테카를로 시뮬레이션 결과 (우승 확률 순 TOP 20)")
    print(f"{'순위':<4} {'팀명':<15} {'우승':<7} {'결승':<7} {'4강':<7} {'8강':<7} {'16강':<7} {'32강':<7}")
    print("-" * 70)
    
    sorted_teams = sorted(stats.items(), key=lambda x: x[1]["Champion"], reverse=True)
    
    for rank, (team, s) in enumerate(sorted_teams[:20], 1):
        champ_pct = s["Champion"] / iterations * 100
        f_pct = s["F"] / iterations * 100
        sf_pct = s["SF"] / iterations * 100
        qf_pct = s["QF"] / iterations * 100
        r16_pct = s["R16"] / iterations * 100
        r32_pct = s["R32"] / iterations * 100
        
        print(f"{rank:<4} {team:<15} {champ_pct:>5.1f}% {f_pct:>6.1f}% {sf_pct:>6.1f}% {qf_pct:>6.1f}% {r16_pct:>6.1f}% {r32_pct:>6.1f}%")

if __name__ == "__main__":
    run_monte_carlo(10000)
