import os
import json
import itertools
import random
from collections import defaultdict
from src.elo import EloSystem
from src.poisson import win_prob_to_lambda, simulate_match_score


class WorldCupSimulation:
    def __init__(self, elo_system: EloSystem, groups_file: str = "data/groups.json", actual_results_file: str = None):
        self.elo_system = elo_system
        with open(groups_file, "r", encoding="utf-8") as f:
            self.groups = json.load(f)

        self.actual_results = {}
        if actual_results_file and os.path.exists(actual_results_file):
            with open(actual_results_file, "r", encoding="utf-8") as f:
                try:
                    results_list = json.load(f)
                    for match in results_list:
                        team_a = match["team_a"]
                        team_b = match["team_b"]
                        score_a = match["score_a"]
                        score_b = match["score_b"]
                        key = tuple(sorted([team_a, team_b]))
                        self.actual_results[key] = {
                            team_a: score_a,
                            team_b: score_b
                        }
                except json.JSONDecodeError:
                    pass

    def simulate_match(self, team_a: str, team_b: str):
        """두 팀의 1경기를 시뮬레이션하고 결과를 반환합니다."""
        rating_a = self.elo_system.get_rating(team_a)
        rating_b = self.elo_system.get_rating(team_b)

        win_prob_a = self.elo_system.expected_score(rating_a, rating_b)
        lambda_a, lambda_b = win_prob_to_lambda(win_prob_a)
        
        score_a, score_b = simulate_match_score(lambda_a, lambda_b)
        return score_a, score_b

    def simulate_group_stage(self):
        """
        모든 조별 리그를 시뮬레이션합니다.
        반환값: 각 조의 최종 순위 딕셔너리
        """
        group_standings = {}

        for group_name, teams in self.groups.items():
            # 팀별 성적 저장: [승점, 득실차, 다득점]
            # 편의를 위해 딕셔너리로 관리
            stats = {
                team: {"pts": 0, "gd": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0} 
                for team in teams
            }

            # 조 내의 모든 팀끼리 1번씩 맞붙음 (총 6경기)
            matchups = list(itertools.combinations(teams, 2))
            
            for team_a, team_b in matchups:
                key = tuple(sorted([team_a, team_b]))
                if key in self.actual_results:
                    score_a = self.actual_results[key][team_a]
                    score_b = self.actual_results[key][team_b]
                else:
                    score_a, score_b = self.simulate_match(team_a, team_b)
                
                # A팀 기록
                stats[team_a]["gf"] += score_a
                stats[team_a]["ga"] += score_b
                stats[team_a]["gd"] += (score_a - score_b)
                
                # B팀 기록
                stats[team_b]["gf"] += score_b
                stats[team_b]["ga"] += score_a
                stats[team_b]["gd"] += (score_b - score_a)
                
                # 승무패 및 승점 기록
                if score_a > score_b:
                    stats[team_a]["pts"] += 3
                    stats[team_a]["w"] += 1
                    stats[team_b]["l"] += 1
                elif score_a < score_b:
                    stats[team_b]["pts"] += 3
                    stats[team_b]["w"] += 1
                    stats[team_a]["l"] += 1
                else:
                    stats[team_a]["pts"] += 1
                    stats[team_b]["pts"] += 1
                    stats[team_a]["d"] += 1
                    stats[team_b]["d"] += 1

            # 순위 정렬 (1. 승점 -> 2. 골득실 -> 3. 다득점)
            sorted_teams = sorted(
                stats.items(), 
                key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"]), 
                reverse=True
            )
            
            group_standings[group_name] = sorted_teams

        return group_standings

    def get_advancing_teams(self, group_standings):
        """조별 리그 결과에서 32강 진출 팀을 추립니다."""
        first_places = []
        second_places = []
        third_places = []
        
        for group, teams in group_standings.items():
            first_places.append(teams[0])
            second_places.append(teams[1])
            third_places.append(teams[2])
            
        # 3위 팀 중 상위 8팀 추출 (승점 -> 골득실 -> 다득점 순)
        third_places.sort(key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"]), reverse=True)
        top_8_thirds = third_places[:8]
        
        # 전체 32팀 시드 배정 (1위 -> 2위 -> 3위 순)
        first_places.sort(key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"]), reverse=True)
        second_places.sort(key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"]), reverse=True)
        
        seeded_teams = [team[0] for team in first_places] + \
                       [team[0] for team in second_places] + \
                       [team[0] for team in top_8_thirds]
                       
        return seeded_teams

    def simulate_knockout_match(self, team_a, team_b):
        """단판 승부 시뮬레이션 (무승부 시 승부차기)"""
        score_a, score_b = self.simulate_match(team_a, team_b)
        
        if score_a != score_b:
            winner = team_a if score_a > score_b else team_b
            return winner, score_a, score_b, False # False는 승부차기 아님을 의미
            
        # 무승부일 경우 승부차기 (간단히 Elo 승률을 가중치로 랜덤 승자 결정)
        rating_a = self.elo_system.get_rating(team_a)
        rating_b = self.elo_system.get_rating(team_b)
        win_prob_a = self.elo_system.expected_score(rating_a, rating_b)
        
        if random.random() < win_prob_a:
            return team_a, score_a, score_b, True
        else:
            return team_b, score_a, score_b, True

    def simulate_knockout_stage(self, seeded_teams):
        """32강부터 결승까지 토너먼트 진행"""
        # 시드 기반 매치업 (1번 vs 32번, 2번 vs 31번...)
        current_round = []
        for i in range(16):
            current_round.append((seeded_teams[i], seeded_teams[31 - i]))
            
        rounds = ["Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "Final"]
        results = {}
        
        for round_name in rounds:
            next_round = []
            results[round_name] = []
            
            for match in current_round:
                team_a, team_b = match
                winner, score_a, score_b, is_pk = self.simulate_knockout_match(team_a, team_b)
                results[round_name].append({
                    "team_a": team_a, "team_b": team_b, 
                    "score_a": score_a, "score_b": score_b, 
                    "winner": winner, "is_pk": is_pk
                })
                next_round.append(winner)
                
            # 다음 라운드 매치업 생성 (순차적으로 2팀씩)
            if len(next_round) > 1:
                current_round = [(next_round[i], next_round[i+1]) for i in range(0, len(next_round), 2)]
            else:
                results["Champion"] = next_round[0]
                
        return results


if __name__ == "__main__":
    import sys
    import os
    
    # 상위 경로를 sys.path에 추가하여 src 모듈을 정상적으로 임포트
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    elo = EloSystem()
    elo.load_ratings()
    
    sim = WorldCupSimulation(elo)
    
    print("조별 리그 시뮬레이션 중...")
    standings = sim.simulate_group_stage()
    
    print("32강 진출 팀 확정 중...")
    seeded_teams = sim.get_advancing_teams(standings)
    
    print("토너먼트 시뮬레이션 중...\n")
    knockout_results = sim.simulate_knockout_stage(seeded_teams)
    
    print("--- [결승전 결과] ---")
    final_match = knockout_results["Final"][0]
    pk_str = " (승부차기)" if final_match["is_pk"] else ""
    print(f"{final_match['team_a']} {final_match['score_a']} : {final_match['score_b']} {final_match['team_b']}{pk_str}")
    print(f"우승: {knockout_results['Champion']}")
