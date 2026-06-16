import os
import json
import itertools
import random
from collections import defaultdict
from functools import cmp_to_key
from src.elo import EloSystem
from src.poisson import win_prob_to_lambda, simulate_match_score

# 국가별 스쿼드 뎁스 지수 (기본값은 1.0)
# 백업 선수의 수준과 에이스 의존도를 반영 (낮을수록 백업이 두터워 전력 누수가 적음)
SQUAD_DEPTH_INDEX = {
    # 0.2: 초강팀 (로드리/음바페 등이 결장해도 백업이 월드클래스)
    "Spain": 0.2,
    "France": 0.2,
    "England": 0.2,
    "Germany": 0.2,
    "Portugal": 0.2,
    "Brazil": 0.2,
    "Argentina": 0.2,
    # 0.4: 상급 뎁스 (준월드클래스 백업 보유)
    "Netherlands": 0.4,
    "Italy": 0.4,
    "Belgium": 0.4,
    "Croatia": 0.4,
    "Uruguay": 0.4,
    "Colombia": 0.4,
    # 0.6: 중급 뎁스 (유럽 주요 리거 수준 백업 보유)
    "Japan": 0.6,
    "Mexico": 0.6,
    "USA": 0.6,
    "Morocco": 0.6,
    "Switzerland": 0.6,
    "Denmark": 0.6,
    "Senegal": 0.6,
    # 0.8: 하급 뎁스 (에이스 의존도 높음, 백업과의 기량 격차가 큼)
    "South Korea": 0.8,
    "Australia": 0.8,
    "Canada": 0.8,
    "Türkiye": 0.8,
    "Sweden": 0.8,
    "Austria": 0.8,
    "Ecuador": 0.8,
}


class WorldCupSimulation:
    def __init__(self, elo_system: EloSystem, groups_file: str = "data/groups.json", actual_results_file: str = None, injuries_file: str = "data/injuries.json"):
        self.elo_system = elo_system
        with open(groups_file, "r", encoding="utf-8") as f:
            self.groups = json.load(f)

        self.actual_results = []
        if actual_results_file and os.path.exists(actual_results_file):
            with open(actual_results_file, "r", encoding="utf-8") as f:
                try:
                    self.actual_results = json.load(f)
                except json.JSONDecodeError:
                    pass

        self.injuries = {}
        if injuries_file and os.path.exists(injuries_file):
            with open(injuries_file, "r", encoding="utf-8") as f:
                try:
                    self.injuries = json.load(f)
                except json.JSONDecodeError:
                    pass

        self.last_standings = None

    def get_injury_multipliers(self, team: str):
        """부상 선수 목록과 스쿼드 뎁스를 고려한 공격/수비 보정 배율 계산"""
        team_injuries = self.injuries.get(team, [])
        if not team_injuries:
            return 1.0, 1.0

        depth = SQUAD_DEPTH_INDEX.get(team, 1.0)
        attack_reduction = 0.0
        defense_reduction = 0.0

        for injury in team_injuries:
            tier = injury.get("tier", "A")
            pos = injury.get("position", "attack")
            # S급: 15%, A급: 8%, B급: 4%
            base = 0.15 if tier == "S" else (0.08 if tier == "A" else 0.04)
            
            if pos == "attack":
                attack_reduction += base * depth
            elif pos == "defense":
                defense_reduction += base * depth

        attack_multiplier = max(0.5, 1.0 - attack_reduction)
        defense_multiplier = min(2.0, 1.0 + defense_reduction)
        return attack_multiplier, defense_multiplier

    def simulate_match(self, team_a: str, team_b: str):
        """두 팀의 1경기를 시뮬레이션하고 결과를 반환합니다."""
        rating_a = self.elo_system.get_rating(team_a)
        rating_b = self.elo_system.get_rating(team_b)

        win_prob_a = self.elo_system.expected_score(rating_a, rating_b)
        lambda_a, lambda_b = win_prob_to_lambda(win_prob_a)
        
        # 부상 보정 배율 적용
        att_mult_a, def_mult_a = self.get_injury_multipliers(team_a)
        att_mult_b, def_mult_b = self.get_injury_multipliers(team_b)
        
        final_lambda_a = lambda_a * att_mult_a * def_mult_b
        final_lambda_b = lambda_b * att_mult_b * def_mult_a
        
        score_a, score_b = simulate_match_score(final_lambda_a, final_lambda_b)
        return score_a, score_b

    def simulate_group_stage(self):
        """
        모든 조별 리그를 시뮬레이션합니다.
        반환값: 각 조의 최종 순위 딕셔너리
        """
        group_standings = {}

        for group_name, teams in self.groups.items():
            # 팀별 성적 저장: [승점, 득실차, 다득점]
            stats = {
                team: {"pts": 0, "gd": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0} 
                for team in teams
            }

            # 조별 리그 모든 경기 결과 저장 (승자승 타이브레이커용)
            group_match_results = {}

            # 조 내의 모든 팀끼리 1번씩 맞붙음 (총 6경기)
            matchups = list(itertools.combinations(teams, 2))
            
            for team_a, team_b in matchups:
                actual_match = None
                for m in self.actual_results:
                    if {m["team_a"], m["team_b"]} == {team_a, team_b} and m.get("stage", "group") == "group":
                        actual_match = m
                        break
                
                if actual_match:
                    score_a = actual_match["score_a"] if actual_match["team_a"] == team_a else actual_match["score_b"]
                    score_b = actual_match["score_b"] if actual_match["team_a"] == team_a else actual_match["score_a"]
                else:
                    score_a, score_b = self.simulate_match(team_a, team_b)
                
                # 승자승 기록
                group_match_results[(team_a, team_b)] = (score_a, score_b)
                group_match_results[(team_b, team_a)] = (score_b, score_a)

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

            # 커스텀 승자 정렬 (승점 -> 골득실 -> 다득점 -> 승자승 -> ELO -> 알파벳 순)
            def compare_teams(x, y):
                # x, y: (team_name, stats_dict)
                if x[1]["pts"] != y[1]["pts"]:
                    return -1 if x[1]["pts"] > y[1]["pts"] else 1
                if x[1]["gd"] != y[1]["gd"]:
                    return -1 if x[1]["gd"] > y[1]["gd"] else 1
                if x[1]["gf"] != y[1]["gf"]:
                    return -1 if x[1]["gf"] > y[1]["gf"] else 1
                
                # 승자승
                x_name, y_name = x[0], y[0]
                match_key = (x_name, y_name)
                if match_key in group_match_results:
                    score_x, score_y = group_match_results[match_key]
                    if score_x != score_y:
                        return -1 if score_x > score_y else 1
                
                # ELO 레이팅
                elo_x = self.elo_system.get_rating(x_name)
                elo_y = self.elo_system.get_rating(y_name)
                if elo_x != elo_y:
                    return -1 if elo_x > elo_y else 1
                
                # 알파벳
                if x_name != y_name:
                    return -1 if x_name < y_name else 1
                return 0

            sorted_teams = sorted(
                stats.items(), 
                key=cmp_to_key(compare_teams)
            )
            
            group_standings[group_name] = sorted_teams

        self.last_standings = group_standings
        return group_standings

    def get_advancing_teams(self, group_standings):
        """조별 리그 결과에서 32강 진출 팀을 추립니다."""
        first_places = []
        second_places = []
        third_places = []
        
        for group, teams in group_standings.items():
            first_places.append(teams[0])
            second_places.append(teams[1])
            # 3위 팀 이름과 통계, 그리고 조 구분 저장
            group_letter = group.split(" ")[1]
            third_places.append({
                "team_name": teams[2][0],
                "group": group_letter,
                "stats": teams[2][1]
            })
            
        # 3위 팀 중 상위 8팀 추출 (승점 -> 골득실 -> 다득점 -> ELO -> 알파벳 순)
        def compare_third_places(x, y):
            if x["stats"]["pts"] != y["stats"]["pts"]:
                return -1 if x["stats"]["pts"] > y["stats"]["pts"] else 1
            if x["stats"]["gd"] != y["stats"]["gd"]:
                return -1 if x["stats"]["gd"] > y["stats"]["gd"] else 1
            if x["stats"]["gf"] != y["stats"]["gf"]:
                return -1 if x["stats"]["gf"] > y["stats"]["gf"] else 1
            
            elo_x = self.elo_system.get_rating(x["team_name"])
            elo_y = self.elo_system.get_rating(y["team_name"])
            if elo_x != elo_y:
                return -1 if elo_x > elo_y else 1
                
            if x["team_name"] != y["team_name"]:
                return -1 if x["team_name"] < y["team_name"] else 1
            return 0

        third_places.sort(key=cmp_to_key(compare_third_places))
        top_8_thirds = third_places[:8]
        
        seeded_teams = [team[0] for team in first_places] + \
                       [team[0] for team in second_places] + \
                       [t["team_name"] for t in top_8_thirds]
                       
        return seeded_teams

    def match_thirds(self, third_place_teams):
        """
        이분 매칭 알고리즘을 사용해 8개의 3위 팀을 대진표 슬롯에 할당합니다.
        third_place_teams: list of tuples (team_name, group_letter)
        """
        slots = [
            {"id": 74, "allowed": {"A", "B", "C", "D", "F"}},
            {"id": 77, "allowed": {"C", "D", "F", "G", "H"}},
            {"id": 79, "allowed": {"C", "E", "F", "H", "I"}},
            {"id": 80, "allowed": {"E", "H", "I", "J", "K"}},
            {"id": 81, "allowed": {"B", "E", "F", "I", "J"}},
            {"id": 82, "allowed": {"A", "E", "H", "I", "J"}},
            {"id": 85, "allowed": {"E", "F", "G", "I", "J"}},
            {"id": 87, "allowed": {"D", "E", "I", "J", "L"}}
        ]
        
        assignment = {}
        used_teams = set()
        
        def dfs(slot_idx):
            if slot_idx == len(slots):
                return True
            slot = slots[slot_idx]
            slot_id = slot["id"]
            allowed = slot["allowed"]
            
            for team_name, group_letter in third_place_teams:
                if team_name in used_teams:
                    continue
                if group_letter in allowed:
                    assignment[slot_id] = team_name
                    used_teams.add(team_name)
                    if dfs(slot_idx + 1):
                        return True
                    used_teams.remove(team_name)
                    del assignment[slot_id]
            return False
            
        if dfs(0):
            return assignment
        
        # fallback: 매칭 실패 시 그냥 순서대로 강제 할당
        for idx, slot in enumerate(slots):
            slot_id = slot["id"]
            if idx < len(third_place_teams):
                assignment[slot_id] = third_place_teams[idx][0]
        return assignment

    def simulate_knockout_match(self, team_a, team_b):
        """단판 승부 시뮬레이션 (무승부 시 승부차기)"""
        # 실제 완료된 경기 결과 고정 체크
        actual_match = None
        for m in self.actual_results:
            if {m["team_a"], m["team_b"]} == {team_a, team_b} and m.get("stage") == "knockout":
                actual_match = m
                break
                
        if actual_match:
            score_a = actual_match["score_a"] if actual_match["team_a"] == team_a else actual_match["score_b"]
            score_b = actual_match["score_b"] if actual_match["team_a"] == team_a else actual_match["score_a"]
            winner = actual_match["winner"]
            
            is_pk = (score_a == score_b)
            if winner:
                return winner, score_a, score_b, is_pk
            else:
                if not is_pk:
                    winner = team_a if score_a > score_b else team_b
                    return winner, score_a, score_b, False
                else:
                    # 실제 결과에 승자가 안 적혀있는 경우 (예: final 경기 결과가 draw로만 있는 등), Elo 기반 승부차기 결정
                    rating_a = self.elo_system.get_rating(team_a)
                    rating_b = self.elo_system.get_rating(team_b)
                    win_prob_a = self.elo_system.expected_score(rating_a, rating_b)
                    winner = team_a if random.random() < win_prob_a else team_b
                    return winner, score_a, score_b, True

        score_a, score_b = self.simulate_match(team_a, team_b)
        
        if score_a != score_b:
            winner = team_a if score_a > score_b else team_b
            return winner, score_a, score_b, False # False는 승부차기 아님
            
        # 무승부일 경우 승부차기
        rating_a = self.elo_system.get_rating(team_a)
        rating_b = self.elo_system.get_rating(team_b)
        win_prob_a = self.elo_system.expected_score(rating_a, rating_b)
        
        if random.random() < win_prob_a:
            return team_a, score_a, score_b, True
        else:
            return team_b, score_a, score_b, True

    def simulate_knockout_stage(self, seeded_teams):
        """32강부터 결승까지 토너먼트 진행"""
        if not self.last_standings:
            # 안전장치: last_standings가 없는 경우 seeded_teams 기반으로 groups 모킹하여 동작
            raise ValueError("조별 리그 standings 데이터가 존재하지 않습니다.")
            
        # 1. 32강 진출 3위 팀 매칭 연산
        first_places = []
        second_places = []
        third_places = []
        for group, teams in self.last_standings.items():
            first_places.append(teams[0])
            second_places.append(teams[1])
            group_letter = group.split(" ")[1]
            third_places.append({
                "team_name": teams[2][0],
                "group": group_letter,
                "stats": teams[2][1]
            })
            
        def compare_third_places(x, y):
            if x["stats"]["pts"] != y["stats"]["pts"]:
                return -1 if x["stats"]["pts"] > y["stats"]["pts"] else 1
            if x["stats"]["gd"] != y["stats"]["gd"]:
                return -1 if x["stats"]["gd"] > y["stats"]["gd"] else 1
            if x["stats"]["gf"] != y["stats"]["gf"]:
                return -1 if x["stats"]["gf"] > y["stats"]["gf"] else 1
            elo_x = self.elo_system.get_rating(x["team_name"])
            elo_y = self.elo_system.get_rating(y["team_name"])
            if elo_x != elo_y:
                return -1 if elo_x > elo_y else 1
            if x["team_name"] != y["team_name"]:
                return -1 if x["team_name"] < y["team_name"] else 1
            return 0

        third_places.sort(key=cmp_to_key(compare_third_places))
        top_8_thirds = third_places[:8]
        
        # 팀 코드 매핑 (A1 -> team_name, etc.)
        team_by_code = {}
        for group_name, group_teams in self.last_standings.items():
            group_letter = group_name.split(" ")[1]
            team_by_code[f"{group_letter}1"] = group_teams[0][0]
            team_by_code[f"{group_letter}2"] = group_teams[1][0]
            team_by_code[f"{group_letter}3"] = group_teams[2][0]
            
        third_place_inputs = [(t["team_name"], t["group"]) for t in top_8_thirds]
        third_assignment = self.match_thirds(third_place_inputs)
        
        # 2. R32 공식 매치업 구성
        r32_slots = [
            {"match_id": 73, "team_a": "A2", "team_b": "B2"},
            {"match_id": 74, "team_a": "E1", "team_b": "3rd_74"},
            {"match_id": 75, "team_a": "F1", "team_b": "C2"},
            {"match_id": 76, "team_a": "C1", "team_b": "F2"},
            {"match_id": 77, "team_a": "I1", "team_b": "3rd_77"},
            {"match_id": 78, "team_a": "E2", "team_b": "I2"},
            {"match_id": 79, "team_a": "A1", "team_b": "3rd_79"},
            {"match_id": 80, "team_a": "L1", "team_b": "3rd_80"},
            {"match_id": 81, "team_a": "D1", "team_b": "3rd_81"},
            {"match_id": 82, "team_a": "G1", "team_b": "3rd_82"},
            {"match_id": 83, "team_a": "K2", "team_b": "L2"},
            {"match_id": 84, "team_a": "H1", "team_b": "J2"},
            {"match_id": 85, "team_a": "B1", "team_b": "3rd_85"},
            {"match_id": 86, "team_a": "J1", "team_b": "H2"},
            {"match_id": 87, "team_a": "K1", "team_b": "3rd_87"},
            {"match_id": 88, "team_a": "D2", "team_b": "G2"}
        ]
        
        r32_matches = []
        for slot in r32_slots:
            m_id = slot["match_id"]
            code_a = slot["team_a"]
            code_b = slot["team_b"]
            
            team_a = team_by_code[code_a]
            if code_b.startswith("3rd_"):
                team_b = third_assignment[m_id]
            else:
                team_b = team_by_code[code_b]
            r32_matches.append((team_a, team_b))
            
        results = {}
        
        # 3. 32강 시뮬레이션
        results["Round of 32"] = []
        r32_winners = []
        for team_a, team_b in r32_matches:
            winner, score_a, score_b, is_pk = self.simulate_knockout_match(team_a, team_b)
            results["Round of 32"].append({
                "team_a": team_a, "team_b": team_b,
                "score_a": score_a, "score_b": score_b,
                "winner": winner, "is_pk": is_pk
            })
            r32_winners.append(winner)
            
        # 4. 16강 시뮬레이션
        r16_pairings = [
            (r32_winners[0], r32_winners[2]),
            (r32_winners[1], r32_winners[4]),
            (r32_winners[3], r32_winners[5]),
            (r32_winners[6], r32_winners[7]),
            (r32_winners[10], r32_winners[11]),
            (r32_winners[8], r32_winners[9]),
            (r32_winners[13], r32_winners[15]),
            (r32_winners[12], r32_winners[14])
        ]
        
        results["Round of 16"] = []
        r16_winners = []
        for team_a, team_b in r16_pairings:
            winner, score_a, score_b, is_pk = self.simulate_knockout_match(team_a, team_b)
            results["Round of 16"].append({
                "team_a": team_a, "team_b": team_b,
                "score_a": score_a, "score_b": score_b,
                "winner": winner, "is_pk": is_pk
            })
            r16_winners.append(winner)
            
        # 5. 8강 시뮬레이션
        qf_pairings = [
            (r16_winners[0], r16_winners[1]),
            (r16_winners[2], r16_winners[3]),
            (r16_winners[4], r16_winners[5]),
            (r16_winners[6], r16_winners[7])
        ]
        
        results["Quarter-finals"] = []
        qf_winners = []
        for team_a, team_b in qf_pairings:
            winner, score_a, score_b, is_pk = self.simulate_knockout_match(team_a, team_b)
            results["Quarter-finals"].append({
                "team_a": team_a, "team_b": team_b,
                "score_a": score_a, "score_b": score_b,
                "winner": winner, "is_pk": is_pk
            })
            qf_winners.append(winner)
            
        # 6. 4강 시뮬레이션
        sf_pairings = [
            (qf_winners[0], qf_winners[1]),
            (qf_winners[2], qf_winners[3])
        ]
        
        results["Semi-finals"] = []
        sf_winners = []
        for team_a, team_b in sf_pairings:
            winner, score_a, score_b, is_pk = self.simulate_knockout_match(team_a, team_b)
            results["Semi-finals"].append({
                "team_a": team_a, "team_b": team_b,
                "score_a": score_a, "score_b": score_b,
                "winner": winner, "is_pk": is_pk
            })
            sf_winners.append(winner)
            
        # 7. 결승전 시뮬레이션
        final_team_a, final_team_b = sf_winners[0], sf_winners[1]
        winner, score_a, score_b, is_pk = self.simulate_knockout_match(final_team_a, final_team_b)
        results["Final"] = [{
            "team_a": final_team_a, "team_b": final_team_b,
            "score_a": score_a, "score_b": score_b,
            "winner": winner, "is_pk": is_pk
        }]
        
        results["Champion"] = winner
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
