import os
import json
import random
import numpy as np
from collections import Counter
from itertools import combinations
from src.elo import EloSystem
from src.poisson import win_prob_to_lambda, simulate_match_score
from src.absences import build_squad_stats, calculate_absence_multipliers, load_absences

# 공동 개최국 정의
HOST_COUNTRIES = {"USA", "Mexico", "Canada"}

# 2026 월드컵 개최지 5대 권역 분류
# Region 1: West Coast
# Region 2: Central-West / Mexico
# Region 3: Central
# Region 4: Southeast
# Region 5: East Coast
GROUP_REGIONS = {
    "A": 1, "B": 1, "C": 1,
    "D": 2, "E": 2, "F": 2,
    "G": 3, "H": 3, "I": 3,
    "J": 5, "K": 5, "L": 5
}

# 토너먼트 경기 일정 및 개최 권역 정보
KNOCKOUT_MATCH_INFO = {
    # 32강 (Matches 73-88)
    73: {"date": "2026-06-28", "region": 5},
    74: {"date": "2026-06-29", "region": 1},
    75: {"date": "2026-06-29", "region": 5},
    76: {"date": "2026-06-30", "region": 5},
    77: {"date": "2026-06-30", "region": 3},
    78: {"date": "2026-07-01", "region": 5},
    79: {"date": "2026-07-01", "region": 1},
    80: {"date": "2026-07-02", "region": 1},
    81: {"date": "2026-07-02", "region": 3},
    82: {"date": "2026-07-03", "region": 2},
    83: {"date": "2026-07-03", "region": 5},
    84: {"date": "2026-07-04", "region": 5},
    85: {"date": "2026-07-04", "region": 1},
    86: {"date": "2026-07-05", "region": 5},
    87: {"date": "2026-07-05", "region": 3},
    88: {"date": "2026-07-06", "region": 3},
    # 16강 (Matches 89-96)
    89: {"date": "2026-07-04", "region": 1},
    90: {"date": "2026-07-05", "region": 3},
    91: {"date": "2026-07-05", "region": 5},
    92: {"date": "2026-07-06", "region": 1},
    93: {"date": "2026-07-07", "region": 5},
    94: {"date": "2026-07-07", "region": 3},
    95: {"date": "2026-07-08", "region": 5},
    96: {"date": "2026-07-08", "region": 5},
    # 8강 (Matches 97-100)
    97: {"date": "2026-07-09", "region": 3},
    98: {"date": "2026-07-10", "region": 5},
    99: {"date": "2026-07-11", "region": 1},
    100: {"date": "2026-07-11", "region": 5},
    # 4강 (Matches 101-102)
    101: {"date": "2026-07-14", "region": 3},
    102: {"date": "2026-07-15", "region": 4},
    # 결승 (Match 104)
    104: {"date": "2026-07-19", "region": 5}
}

def date_to_day_num(date_str: str) -> int:
    """날짜 문자열('YYYY-MM-DD')을 정수형 기준일로 변환"""
    parts = date_str.split("-")
    month = int(parts[1])
    day = int(parts[2])
    if month == 6:
        return day
    elif month == 7:
        return 30 + day
    return day


class WorldCupSimulation:
    def __init__(
        self,
        elo_system: EloSystem,
        groups_file: str = "data/groups.json",
        actual_results_file: str = None,
        absences_file: str = "data/absences.json",
        squads_file: str = "data/squads.json",
        fifa_rankings_file: str = "data/fifa_rankings.json",
        team_conduct_file: str = "data/team_conduct_scores.json",
        third_place_annex_file: str = "data/third_place_annex_c.json",
    ):
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

        self.injuries = load_absences(absences_file) if absences_file else {}

        self.squads = {}
        if os.path.exists(squads_file):
            with open(squads_file, "r", encoding="utf-8") as f:
                try:
                    self.squads = json.load(f)
                except json.JSONDecodeError:
                    pass

        # 스쿼드별 총 가치, 포지션별 가치 및 HHI 집중도 지수 사전 계산
        self.team_squad_stats = build_squad_stats(self.squads)
        self.fifa_rankings = self._load_optional_json(fifa_rankings_file, {})
        self.team_conduct_scores = self._load_optional_json(team_conduct_file, {})
        self.third_place_annex = self._load_optional_json(third_place_annex_file, {})

        self.last_standings = None

    def _load_optional_json(self, path: str, default):
        if not path or not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default

    def get_team_conduct_score(self, team: str) -> int:
        return int(self.team_conduct_scores.get(team, 0))

    def get_fifa_ranking(self, team: str) -> int:
        ranking = self.fifa_rankings.get(team)
        if ranking is None:
            raise KeyError(f"FIFA ranking data is missing for {team}")
        return int(ranking)

    def get_injury_multipliers(self, team: str):
        """스쿼드 가치 비중과 HHI 의존도를 고려한 동적 공격/수비 결장 보정 배율 계산"""
        attack_multiplier, defense_multiplier, _ = calculate_absence_multipliers(
            team,
            self.injuries,
            self.squads,
            self.team_squad_stats,
        )
        return attack_multiplier, defense_multiplier

    def simulate_match(self, team_a: str, team_b: str, home_advantage: bool = True, rest_days_diff: int = 0, travel_fatigue_a: float = 0.0, travel_fatigue_b: float = 0.0):
        """두 팀의 1경기를 시뮬레이션하고 결과를 반환합니다."""
        rating_a = self.elo_system.get_rating(team_a)
        rating_b = self.elo_system.get_rating(team_b)

        # 1. 개최국 홈 우위 적용 (+40 ELO)
        if home_advantage:
            is_host_a = team_a in HOST_COUNTRIES
            is_host_b = team_b in HOST_COUNTRIES
            if is_host_a and not is_host_b:
                rating_a += 40
            elif is_host_b and not is_host_a:
                rating_b += 40

        # 2. 휴식일 체력 격차 보정 적용 (하루당 +5, 최대 +30 ELO)
        rest_bonus = min(abs(rest_days_diff) * 5, 30)
        if rest_days_diff >= 1:
            rating_a += rest_bonus
        elif rest_days_diff <= -1:
            rating_b += rest_bonus

        win_prob_a = self.elo_system.expected_score(rating_a, rating_b)
        lambda_a, lambda_b = win_prob_to_lambda(win_prob_a)
        
        # 3. 부상 보정 배율 적용 (get_injury_multipliers: attack_multiplier <= 1.0, defense_multiplier >= 1.0)
        att_mult_a, def_mult_a = self.get_injury_multipliers(team_a)
        att_mult_b, def_mult_b = self.get_injury_multipliers(team_b)
        
        # 4. 이동 피로도 및 로테이션 보정 적용
        # A팀의 공격진이 다치면 A팀의 득점이 줄고(att_mult_a <= 1.0), B팀의 수비진이 다치면 A팀의 득점이 늘어납니다(def_mult_b >= 1.0).
        final_lambda_a = lambda_a * att_mult_a * def_mult_b * (1.0 - travel_fatigue_a)
        final_lambda_b = lambda_b * att_mult_b * def_mult_a * (1.0 - travel_fatigue_b)
        
        score_a, score_b = simulate_match_score(final_lambda_a, final_lambda_b)
        return score_a, score_b

    def _update_group_stats(self, team_a, team_b, score_a, score_b, stats, group_match_results):
        group_match_results[(team_a, team_b)] = (score_a, score_b)
        group_match_results[(team_b, team_a)] = (score_b, score_a)

        stats[team_a]["gf"] += score_a
        stats[team_a]["ga"] += score_b
        stats[team_a]["gd"] += (score_a - score_b)
        stats[team_b]["gf"] += score_b
        stats[team_b]["ga"] += score_a
        stats[team_b]["gd"] += (score_b - score_a)
        
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

    def _head_to_head_stats(self, team_names, group_match_results):
        h2h = {
            team: {"pts": 0, "gd": 0, "gf": 0}
            for team in team_names
        }
        for team_a, team_b in combinations(team_names, 2):
            score_a, score_b = group_match_results[(team_a, team_b)]
            h2h[team_a]["gf"] += score_a
            h2h[team_a]["gd"] += score_a - score_b
            h2h[team_b]["gf"] += score_b
            h2h[team_b]["gd"] += score_b - score_a
            if score_a > score_b:
                h2h[team_a]["pts"] += 3
            elif score_a < score_b:
                h2h[team_b]["pts"] += 3
            else:
                h2h[team_a]["pts"] += 1
                h2h[team_b]["pts"] += 1
        return h2h

    def _rank_equal_points_teams(self, team_names, stats, group_match_results):
        if len(team_names) <= 1:
            return team_names

        h2h = self._head_to_head_stats(team_names, group_match_results)
        for key in ("pts", "gd", "gf"):
            grouped = {}
            for team in team_names:
                grouped.setdefault(h2h[team][key], []).append(team)
            if len(grouped) > 1:
                ranked = []
                for value in sorted(grouped.keys(), reverse=True):
                    tied = grouped[value]
                    ranked.extend(self._rank_equal_points_teams(tied, stats, group_match_results))
                return ranked

        return sorted(
            team_names,
            key=lambda team: (
                -stats[team]["gd"],
                -stats[team]["gf"],
                -self.get_team_conduct_score(team),
                self.get_fifa_ranking(team),
            ),
        )

    def _sort_group_standings(self, stats, group_match_results):
        grouped_by_points = {}
        for team in stats:
            grouped_by_points.setdefault(stats[team]["pts"], []).append(team)

        ranked_teams = []
        for points in sorted(grouped_by_points.keys(), reverse=True):
            tied = grouped_by_points[points]
            ranked_teams.extend(self._rank_equal_points_teams(tied, stats, group_match_results))

        return [(team, stats[team]) for team in ranked_teams]

    def _sort_third_places(self, third_places):
        return sorted(
            third_places,
            key=lambda team: (
                -team["stats"]["pts"],
                -team["stats"]["gd"],
                -team["stats"]["gf"],
                -self.get_team_conduct_score(team["team_name"]),
                self.get_fifa_ranking(team["team_name"]),
            ),
        )

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

            # 조별 리그 모든 경기 결과 저장 (맞대결 타이브레이커용)
            group_match_results = {}

            # 라운드별 경기 페어링 (T0, T1, T2, T3)
            r1_pairings = [(teams[0], teams[1]), (teams[2], teams[3])]
            r2_pairings = [(teams[0], teams[2]), (teams[1], teams[3])]
            r3_pairings = [(teams[0], teams[3]), (teams[1], teams[2])]

            # 1라운드 및 2라운드 시뮬레이션 선 진행
            for team_a, team_b in r1_pairings + r2_pairings:
                actual_match = None
                for m in self.actual_results:
                    if {m["team_a"], m["team_b"]} == {team_a, team_b} and m.get("stage", "group") == "group":
                        actual_match = m
                        break
                
                if actual_match:
                    score_a = actual_match["score_a"] if actual_match["team_a"] == team_a else actual_match["score_b"]
                    score_b = actual_match["score_b"] if actual_match["team_a"] == team_a else actual_match["score_a"]
                else:
                    # 조별 리그 단계에서는 홈 이점 미적용 (중립국 구장)
                    score_a, score_b = self.simulate_match(team_a, team_b, home_advantage=False)
                
                # 기록 업데이트
                self._update_group_stats(team_a, team_b, score_a, score_b, stats, group_match_results)

            # 2라운드 끝난 후 진출 확정팀 식별 (승점 6점 획득 팀)
            rotation_teams = set()
            for team, s in stats.items():
                if s["pts"] == 6:
                    rotation_teams.add(team)

            # 3라운드 진행
            for team_a, team_b in r3_pairings:
                actual_match = None
                for m in self.actual_results:
                    if {m["team_a"], m["team_b"]} == {team_a, team_b} and m.get("stage", "group") == "group":
                        actual_match = m
                        break
                
                if actual_match:
                    score_a = actual_match["score_a"] if actual_match["team_a"] == team_a else actual_match["score_b"]
                    score_b = actual_match["score_b"] if actual_match["team_a"] == team_a else actual_match["score_a"]
                else:
                    # 3차전 로테이션 보정 배율 적용 (진출 확정팀은 예상 득점 20% 페널티)
                    fatigue_a = 0.2 if team_a in rotation_teams else 0.0
                    fatigue_b = 0.2 if team_b in rotation_teams else 0.0
                    score_a, score_b = self.simulate_match(team_a, team_b, home_advantage=False, travel_fatigue_a=fatigue_a, travel_fatigue_b=fatigue_b)
                
                # 기록 업데이트
                self._update_group_stats(team_a, team_b, score_a, score_b, stats, group_match_results)

            sorted_teams = self._sort_group_standings(stats, group_match_results)
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
            group_letter = group.split(" ")[1]
            third_places.append({
                "team_name": teams[2][0],
                "group": group_letter,
                "stats": teams[2][1]
            })
            
        top_8_thirds = self._sort_third_places(third_places)[:8]
        
        seeded_teams = [team[0] for team in first_places] + \
                       [team[0] for team in second_places] + \
                       [t["team_name"] for t in top_8_thirds]
                       
        return seeded_teams

    def match_thirds(self, third_place_teams):
        """FIFA World Cup 2026 Regulations Annex C에 따라 3위 팀을 32강 슬롯에 배정합니다."""
        teams_by_group = {group_letter: team_name for team_name, group_letter in third_place_teams}
        key = "".join(sorted(teams_by_group))
        if key not in self.third_place_annex:
            raise ValueError(f"Annex C third-place assignment is missing for groups: {key}")

        return {
            int(match_id): teams_by_group[group_letter]
            for match_id, group_letter in self.third_place_annex[key].items()
        }

    def simulate_knockout_match(self, team_a, team_b, rest_days_diff: int = 0, travel_fatigue_a: float = 0.0, travel_fatigue_b: float = 0.0):
        """단판 승부 시뮬레이션 (무승부 시 승부차기, 환경 변수 반영)"""
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
                    rating_a = self.elo_system.get_rating(team_a)
                    rating_b = self.elo_system.get_rating(team_b)
                    win_prob_a = self.elo_system.expected_score(rating_a, rating_b)
                    winner = team_a if random.random() < win_prob_a else team_b
                    return winner, score_a, score_b, True

        score_a, score_b = self.simulate_match(
            team_a, team_b, 
            home_advantage=True, 
            rest_days_diff=rest_days_diff, 
            travel_fatigue_a=travel_fatigue_a, 
            travel_fatigue_b=travel_fatigue_b
        )
        
        if score_a != score_b:
            winner = team_a if score_a > score_b else team_b
            return winner, score_a, score_b, False
            
        # 90분 무승부인 경우 연장전(Extra Time) 시뮬레이션 (30분 추가)
        rating_a = self.elo_system.get_rating(team_a)
        rating_b = self.elo_system.get_rating(team_b)
        
        is_host_a = team_a in HOST_COUNTRIES
        is_host_b = team_b in HOST_COUNTRIES
        if is_host_a and not is_host_b:
            rating_a += 40
        elif is_host_b and not is_host_a:
            rating_b += 40
            
        rest_bonus = min(abs(rest_days_diff) * 5, 30)
        if rest_days_diff >= 1:
            rating_a += rest_bonus
        elif rest_days_diff <= -1:
            rating_b += rest_bonus
            
        win_prob_a = self.elo_system.expected_score(rating_a, rating_b)
        lambda_a, lambda_b = win_prob_to_lambda(win_prob_a)
        
        att_mult_a, def_mult_a = self.get_injury_multipliers(team_a)
        att_mult_b, def_mult_b = self.get_injury_multipliers(team_b)
        
        final_lambda_a = lambda_a * att_mult_a * def_mult_b * (1.0 - travel_fatigue_a)
        final_lambda_b = lambda_b * att_mult_b * def_mult_a * (1.0 - travel_fatigue_b)
        
        # 30분 연장전 골 수 산출 (90분 평균 대비 1/3 비율)
        goals_a_et = np.random.poisson(final_lambda_a / 3.0)
        goals_b_et = np.random.poisson(final_lambda_b / 3.0)
        
        score_a += goals_a_et
        score_b += goals_b_et
        
        if score_a != score_b:
            winner = team_a if score_a > score_b else team_b
            return winner, score_a, score_b, False
            
        # 연장전도 무승부인 경우 승부차기 진행
        win_prob_pk_a = self.elo_system.expected_score(rating_a, rating_b)
        if random.random() < win_prob_pk_a:
            return team_a, score_a, score_b, True
        else:
            return team_b, score_a, score_b, True

    def simulate_knockout_match_consensus(self, team_a, team_b, rest_days_diff: int = 0, travel_fatigue_a: float = 0.0, travel_fatigue_b: float = 0.0, runs: int = 10000):
        """여러 번의 단판 시뮬레이션에서 가장 자주 진출한 팀과 대표 스코어를 반환합니다."""
        runs = max(1, int(runs))
        winner_counts = Counter()
        score_counts = Counter()

        for _ in range(runs):
            winner, score_a, score_b, is_pk = self.simulate_knockout_match(
                team_a,
                team_b,
                rest_days_diff=rest_days_diff,
                travel_fatigue_a=travel_fatigue_a,
                travel_fatigue_b=travel_fatigue_b,
            )
            winner_counts[winner] += 1
            score_counts[(winner, score_a, score_b, is_pk)] += 1

        winner = winner_counts.most_common(1)[0][0]
        winner_score_counts = Counter({
            (score_a, score_b, is_pk): count
            for (score_winner, score_a, score_b, is_pk), count in score_counts.items()
            if score_winner == winner
        })
        score_a, score_b, is_pk = winner_score_counts.most_common(1)[0][0]
        return winner, score_a, score_b, is_pk, winner_counts

    def simulate_knockout_round(self, pairings, match_start_id, team_states, consensus_runs=None):
        """동적 변수(휴식일 격차, 이동 피로도) 계산기를 내장한 한 라운드 시뮬레이션 일괄 수행기"""
        round_results = []
        winners = []
        for idx, (team_a, team_b) in enumerate(pairings):
            match_id = match_start_id + idx
            m_info = KNOCKOUT_MATCH_INFO[match_id]
            match_date = date_to_day_num(m_info["date"])
            match_region = m_info["region"]
            
            # 피로도 및 휴식일 파악을 위한 각 팀의 직전 상태 확인
            state_a = team_states[team_a]
            state_b = team_states[team_b]
            
            # 1. 휴식일 계산
            rest_a = match_date - state_a["last_date"]
            rest_b = match_date - state_b["last_date"]
            rest_days_diff = rest_a - rest_b
            
            # 2. 대륙 권역간 이동에 의한 피로도(Travel Fatigue) 계산
            diff_a = abs(match_region - state_a["last_region"])
            fatigue_a = 0.03 if diff_a >= 3 else (0.015 if diff_a > 0 else 0.0)
            
            diff_b = abs(match_region - state_b["last_region"])
            fatigue_b = 0.03 if diff_b >= 3 else (0.015 if diff_b > 0 else 0.0)
            
            if consensus_runs and consensus_runs > 1:
                winner, score_a, score_b, is_pk, winner_counts = self.simulate_knockout_match_consensus(
                    team_a,
                    team_b,
                    rest_days_diff=rest_days_diff,
                    travel_fatigue_a=fatigue_a,
                    travel_fatigue_b=fatigue_b,
                    runs=consensus_runs,
                )
                advance_prob_a = winner_counts[team_a] / consensus_runs
                advance_prob_b = winner_counts[team_b] / consensus_runs
            else:
                winner, score_a, score_b, is_pk = self.simulate_knockout_match(
                    team_a,
                    team_b,
                    rest_days_diff=rest_days_diff,
                    travel_fatigue_a=fatigue_a,
                    travel_fatigue_b=fatigue_b,
                )
                advance_prob_a = None
                advance_prob_b = None
            
            # 진출한 승리팀의 최종 일정(날짜 및 권역) 업데이트
            team_states[winner] = {
                "last_date": match_date,
                "last_region": match_region
            }
            
            round_results.append({
                "team_a": team_a, "team_b": team_b,
                "score_a": score_a, "score_b": score_b,
                "winner": winner, "is_pk": is_pk,
                "rest_days_diff": rest_days_diff,
                "travel_fatigue_a": fatigue_a,
                "travel_fatigue_b": fatigue_b,
                "advance_prob_a": advance_prob_a,
                "advance_prob_b": advance_prob_b,
                "consensus_runs": consensus_runs if consensus_runs and consensus_runs > 1 else None,
            })
            winners.append(winner)
            
        return round_results, winners

    def simulate_knockout_stage(self, consensus_runs=None):
        """32강부터 결승까지 토너먼트 진행 (휴식일 및 피로도 누적 시뮬레이션)"""
        if not self.last_standings:
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
            
        top_8_thirds = self._sort_third_places(third_places)[:8]
        
        # 팀 코드 매핑 (A1 -> team_name, etc.)
        team_by_code = {}
        for group_name, group_teams in self.last_standings.items():
            group_letter = group_name.split(" ")[1]
            team_by_code[f"{group_letter}1"] = group_teams[0][0]
            team_by_code[f"{group_letter}2"] = group_teams[1][0]
            team_by_code[f"{group_letter}3"] = group_teams[2][0]
            
        third_place_inputs = [(t["team_name"], t["group"]) for t in top_8_thirds]
        third_assignment = self.match_thirds(third_place_inputs)
        
        # 32강에 올라간 각 팀의 마지막 경기 날짜와 지역 추적 데이터 초기화
        # (조별 리그 최종 종료일인 6월 27일, day 27을 기준점으로 시작)
        team_states = {}
        for group_name, group_teams in self.last_standings.items():
            group_letter = group_name.split(" ")[1]
            reg = GROUP_REGIONS[group_letter]
            team_states[group_teams[0][0]] = {"last_date": 27, "last_region": reg}
            team_states[group_teams[1][0]] = {"last_date": 27, "last_region": reg}
        for t in top_8_thirds:
            reg = GROUP_REGIONS[t["group"]]
            team_states[t["team_name"]] = {"last_date": 27, "last_region": reg}
        
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
        results["Round of 32"], r32_winners = self.simulate_knockout_round(r32_matches, 73, team_states, consensus_runs=consensus_runs)
            
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
        results["Round of 16"], r16_winners = self.simulate_knockout_round(r16_pairings, 89, team_states, consensus_runs=consensus_runs)
            
        # 5. 8강 시뮬레이션
        qf_pairings = [
            (r16_winners[0], r16_winners[1]),
            (r16_winners[2], r16_winners[3]),
            (r16_winners[4], r16_winners[5]),
            (r16_winners[6], r16_winners[7])
        ]
        results["Quarter-finals"], qf_winners = self.simulate_knockout_round(qf_pairings, 97, team_states, consensus_runs=consensus_runs)
            
        # 6. 4강 시뮬레이션
        sf_pairings = [
            (qf_winners[0], qf_winners[1]),
            (qf_winners[2], qf_winners[3])
        ]
        results["Semi-finals"], sf_winners = self.simulate_knockout_round(sf_pairings, 101, team_states, consensus_runs=consensus_runs)
            
        # 7. 결승전 시뮬레이션
        final_pairings = [(sf_winners[0], sf_winners[1])]
        results["Final"], final_winners = self.simulate_knockout_round(final_pairings, 104, team_states, consensus_runs=consensus_runs)
        
        results["Champion"] = final_winners[0]
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
    knockout_results = sim.simulate_knockout_stage()
    
    print("--- [결승전 결과] ---")
    final_match = knockout_results["Final"][0]
    pk_str = " (승부차기)" if final_match["is_pk"] else ""
    print(f"{final_match['team_a']} {final_match['score_a']} : {final_match['score_b']} {final_match['team_b']}{pk_str}")
    print(f"우승: {knockout_results['Champion']}")
