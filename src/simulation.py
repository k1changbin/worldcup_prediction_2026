import os
import json
import random
import numpy as np
from collections import Counter
from itertools import combinations
from src.elo import EloSystem
from src.model_config import (
    BASE_GOALS,
    ELO_LAMBDA_EXPONENT,
    HOST_ADVANTAGE_ELO,
    LONG_REGION_TRAVEL_PENALTY,
    NEAR_REGION_TRAVEL_PENALTY,
    REST_ADVANTAGE_CAP,
    REST_ELO_PER_DAY,
    ROTATION_ATTACK_PENALTY,
)
from src.paths import data_path
from src.poisson import elo_expected_score_to_lambdas
from src.absences import build_squad_stats, calculate_absence_multipliers, load_absences
from src.bracket import (
    SIMULATED_KNOCKOUT_STAGES,
    resolve_match_team_labels,
    resolve_match_teams,
    team_codes_from_standings,
    validate_knockout_bracket,
)
from src.schedule import (
    DEFAULT_SCHEDULE_PATH,
    date_to_day_num as date_to_day_num,
    load_schedule,
)
from src.tournament_state import rank_group_stats, rank_third_places

# Co-host countries.
HOST_COUNTRIES = {"USA", "Mexico", "Canada"}

# Backwards-compatible metadata view.  Values are derived from schedule.json;
# the simulator itself uses its injected ``TournamentSchedule`` instance.
_DEFAULT_SCHEDULE = load_schedule(DEFAULT_SCHEDULE_PATH)
KNOCKOUT_MATCH_INFO = {
    match.match_number: {
        "date": match.date,
        "region": match.region,
        "host_city": match.host_city,
        "stage": match.stage,
    }
    for match in _DEFAULT_SCHEDULE.knockout_matches()
}


class WorldCupSimulation:
    def __init__(
        self,
        elo_system: EloSystem,
        groups_file: str = data_path("groups.json"),
        actual_results_file: str = None,
        absences_file: str = data_path("absences.json"),
        squads_file: str = data_path("squads.json"),
        fifa_rankings_file: str = data_path("fifa_rankings.json"),
        team_conduct_file: str = data_path("team_conduct_scores.json"),
        third_place_annex_file: str = data_path("third_place_annex_c.json"),
        schedule_file: str = str(DEFAULT_SCHEDULE_PATH),
        rng=None,
        np_rng=None,
    ):
        self.elo_system = elo_system
        self.schedule = load_schedule(schedule_file)

        if rng is not None and np_rng is None and callable(
            getattr(rng, "poisson", None)
        ):
            # A NumPy Generator can drive both score and penalty sampling.
            self.rng = rng
            self.np_rng = rng
        else:
            self.rng = rng if rng is not None else random.Random()
            self.np_rng = (
                np_rng if np_rng is not None else np.random.default_rng()
            )
        if not callable(getattr(self.rng, "random", None)):
            raise TypeError("rng must provide a random() method")
        if not callable(getattr(self.np_rng, "poisson", None)):
            raise TypeError("np_rng must provide a poisson() method")

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

        # Precompute squad total value, position totals, and HHI concentration.
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

    @staticmethod
    def _actual_match_number(match):
        value = match.get("match_number", match.get("matchNumber"))
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalized_result_stage(match):
        stage = str(match.get("stage", "group")).lower()
        return "group" if stage in {"group", "group-stage"} else "knockout"

    def _find_actual_match(
        self,
        team_a: str = None,
        team_b: str = None,
        *,
        stage: str,
        match_number: int = None,
    ):
        """Find a recorded result, preferring its stable match number.

        Older ``actual_results.json`` files did not contain match numbers, so
        an unordered team-pair lookup remains as a compatibility fallback.
        """

        if match_number is not None:
            numbered = [
                match
                for match in self.actual_results
                if self._actual_match_number(match) == int(match_number)
            ]
            if len(numbered) > 1:
                raise ValueError(f"Multiple actual results are recorded for M{match_number}")
            if numbered:
                actual = numbered[0]
                if team_a is not None and team_b is not None:
                    actual_teams = {actual.get("team_a"), actual.get("team_b")}
                    if actual_teams != {team_a, team_b}:
                        raise ValueError(
                            f"Actual result for M{match_number} has teams "
                            f"{actual.get('team_a')} / {actual.get('team_b')}, expected "
                            f"{team_a} / {team_b}"
                        )
                return actual

        if team_a is None or team_b is None:
            return None

        expected_stage = "group" if stage in {"group", "group-stage"} else "knockout"
        for match in self.actual_results:
            if self._actual_match_number(match) is not None:
                continue
            if self._normalized_result_stage(match) != expected_stage:
                continue
            if {match.get("team_a"), match.get("team_b")} == {team_a, team_b}:
                return match
        return None

    @staticmethod
    def _orient_actual_score(actual_match, team_a: str, team_b: str):
        actual_teams = {actual_match.get("team_a"), actual_match.get("team_b")}
        if actual_teams != {team_a, team_b}:
            raise ValueError(
                f"Actual result teams do not match {team_a} / {team_b}: "
                f"{actual_match.get('team_a')} / {actual_match.get('team_b')}"
            )
        if actual_match["team_a"] == team_a:
            return int(actual_match["score_a"]), int(actual_match["score_b"])
        return int(actual_match["score_b"]), int(actual_match["score_a"])

    @staticmethod
    def _actual_winner(actual_match, team_a: str, team_b: str, score_a, score_b):
        winner = actual_match.get("winner")
        if winner in {team_a, team_b}:
            return winner
        if score_a > score_b:
            return team_a
        if score_b > score_a:
            return team_b
        return None

    def get_injury_multipliers(self, team: str):
        """Calculate dynamic attack/defense absence multipliers from squad value and HHI."""
        attack_multiplier, defense_multiplier, _ = calculate_absence_multipliers(
            team,
            self.injuries,
            self.squads,
            self.team_squad_stats,
        )
        return attack_multiplier, defense_multiplier

    def get_expected_goals(
        self,
        team_a: str,
        team_b: str,
        home_advantage: bool = True,
        rest_days_diff: int = 0,
        travel_fatigue_a: float = 0.0,
        travel_fatigue_b: float = 0.0,
    ) -> tuple[float, float]:
        """Return context-adjusted expected goals for both teams.

        This is the shared model entry point for the simulator, dashboard, and
        command-line predictor.  ``home_advantage`` means host-country
        advantage; all tournament venues are otherwise neutral.
        """

        for label, fatigue in (
            ("travel_fatigue_a", travel_fatigue_a),
            ("travel_fatigue_b", travel_fatigue_b),
        ):
            if not 0.0 <= fatigue <= 1.0:
                raise ValueError(f"{label} must be between 0 and 1")

        rating_a, rating_b = self.get_adjusted_ratings(
            team_a,
            team_b,
            home_advantage=home_advantage,
            rest_days_diff=rest_days_diff,
        )

        expected_score_a = self.elo_system.expected_score(rating_a, rating_b)
        lambda_a, lambda_b = elo_expected_score_to_lambdas(
            expected_score_a,
            base_goals=BASE_GOALS,
            exponent=ELO_LAMBDA_EXPONENT,
        )

        att_mult_a, def_mult_a = self.get_injury_multipliers(team_a)
        att_mult_b, def_mult_b = self.get_injury_multipliers(team_b)

        final_lambda_a = (
            lambda_a
            * att_mult_a
            * def_mult_b
            * (1.0 - travel_fatigue_a)
        )
        final_lambda_b = (
            lambda_b
            * att_mult_b
            * def_mult_a
            * (1.0 - travel_fatigue_b)
        )
        return float(final_lambda_a), float(final_lambda_b)

    def get_adjusted_ratings(
        self,
        team_a: str,
        team_b: str,
        home_advantage: bool = True,
        rest_days_diff: int = 0,
    ) -> tuple[float, float]:
        """Return Elo ratings after host-country and rest adjustments."""

        rating_a = float(self.elo_system.get_rating(team_a))
        rating_b = float(self.elo_system.get_rating(team_b))

        if home_advantage:
            is_host_a = team_a in HOST_COUNTRIES
            is_host_b = team_b in HOST_COUNTRIES
            if is_host_a and not is_host_b:
                rating_a += HOST_ADVANTAGE_ELO
            elif is_host_b and not is_host_a:
                rating_b += HOST_ADVANTAGE_ELO

        rest_bonus = min(
            abs(rest_days_diff) * REST_ELO_PER_DAY,
            REST_ADVANTAGE_CAP,
        )
        if rest_days_diff >= 1:
            rating_a += rest_bonus
        elif rest_days_diff <= -1:
            rating_b += rest_bonus
        return rating_a, rating_b

    def simulate_match(self, team_a: str, team_b: str, home_advantage: bool = True, rest_days_diff: int = 0, travel_fatigue_a: float = 0.0, travel_fatigue_b: float = 0.0):
        """Simulate one match between two teams and return the score."""
        lambda_a, lambda_b = self.get_expected_goals(
            team_a,
            team_b,
            home_advantage=home_advantage,
            rest_days_diff=rest_days_diff,
            travel_fatigue_a=travel_fatigue_a,
            travel_fatigue_b=travel_fatigue_b,
        )
        return int(self.np_rng.poisson(lambda_a)), int(self.np_rng.poisson(lambda_b))

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

    def _sort_group_standings(self, stats, group_match_results):
        return rank_group_stats(
            stats,
            group_match_results,
            self.fifa_rankings,
            self.team_conduct_scores,
        )

    def _sort_third_places(self, third_places):
        return rank_third_places(
            third_places,
            self.fifa_rankings,
            self.team_conduct_scores,
        )

    def _group_round_matches(self, group_name, teams):
        """Return three rounds of pairings with optional schedule match IDs."""

        group_letter = group_name.split()[-1]
        scheduled = self.schedule.group_matches(group_letter)
        expected_pairs = {
            frozenset(pair) for pair in combinations(teams, 2)
        }
        scheduled_pairs = {
            frozenset((match.home_team, match.away_team)) for match in scheduled
        }
        if len(scheduled) == 6 and scheduled_pairs == expected_pairs:
            rounds = []
            for start in (0, 2, 4):
                round_matches = scheduled[start : start + 2]
                round_teams = {
                    team
                    for match in round_matches
                    for team in (match.home_team, match.away_team)
                }
                if round_teams != set(teams):
                    raise ValueError(
                        f"Schedule group {group_letter} does not contain "
                        "each team exactly once per round"
                    )
                rounds.append(
                    [
                        (match.home_team, match.away_team, match.match_number)
                        for match in round_matches
                    ]
                )
            return rounds

        # Preserve support for callers that inject small synthetic group files
        # without also providing a matching tournament schedule.
        return [
            [(teams[0], teams[1], None), (teams[2], teams[3], None)],
            [(teams[0], teams[2], None), (teams[1], teams[3], None)],
            [(teams[0], teams[3], None), (teams[1], teams[2], None)],
        ]

    def simulate_group_stage(self):
        """
        Simulate all group-stage matches.
        Returns a dictionary of final standings by group.
        """
        group_standings = {}
        match_contexts = self.build_group_stage_match_contexts()

        for group_name, teams in self.groups.items():
            # Store team records.
            stats = {
                team: {"pts": 0, "gd": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0} 
                for team in teams
            }

            # Store all group match results for head-to-head tiebreakers.
            group_match_results = {}

            rounds = self._group_round_matches(group_name, teams)

            # Simulate rounds 1 and 2 first.
            for team_a, team_b, match_number in rounds[0] + rounds[1]:
                actual_match = self._find_actual_match(
                    team_a,
                    team_b,
                    stage="group",
                    match_number=match_number,
                )
                
                if actual_match:
                    score_a, score_b = self._orient_actual_score(
                        actual_match, team_a, team_b
                    )
                else:
                    context = match_contexts.get(match_number, {})
                    score_a, score_b = self.simulate_match(
                        team_a,
                        team_b,
                        home_advantage=True,
                        rest_days_diff=context.get("rest_days_diff", 0),
                        travel_fatigue_a=context.get("travel_fatigue_a", 0.0),
                        travel_fatigue_b=context.get("travel_fatigue_b", 0.0),
                    )
                
                # Update records.
                self._update_group_stats(team_a, team_b, score_a, score_b, stats, group_match_results)

            # Identify teams that have effectively qualified after round 2.
            rotation_teams = set()
            for team, s in stats.items():
                if s["pts"] == 6:
                    rotation_teams.add(team)

            # Simulate round 3.
            for team_a, team_b, match_number in rounds[2]:
                actual_match = self._find_actual_match(
                    team_a,
                    team_b,
                    stage="group",
                    match_number=match_number,
                )
                
                if actual_match:
                    score_a, score_b = self._orient_actual_score(
                        actual_match, team_a, team_b
                    )
                else:
                    # Apply a round-3 rotation penalty to teams already on six points.
                    rotation_a = (
                        ROTATION_ATTACK_PENALTY
                        if team_a in rotation_teams
                        else 0.0
                    )
                    rotation_b = (
                        ROTATION_ATTACK_PENALTY
                        if team_b in rotation_teams
                        else 0.0
                    )
                    context = match_contexts.get(match_number, {})
                    travel_a = context.get("travel_fatigue_a", 0.0)
                    travel_b = context.get("travel_fatigue_b", 0.0)
                    # Combine independent percentage reductions without
                    # double-counting the same share of attacking output.
                    fatigue_a = 1.0 - (1.0 - travel_a) * (1.0 - rotation_a)
                    fatigue_b = 1.0 - (1.0 - travel_b) * (1.0 - rotation_b)
                    score_a, score_b = self.simulate_match(
                        team_a,
                        team_b,
                        home_advantage=True,
                        rest_days_diff=context.get("rest_days_diff", 0),
                        travel_fatigue_a=fatigue_a,
                        travel_fatigue_b=fatigue_b,
                    )
                
                # Update records.
                self._update_group_stats(team_a, team_b, score_a, score_b, stats, group_match_results)

            sorted_teams = self._sort_group_standings(stats, group_match_results)
            group_standings[group_name] = sorted_teams

        self.last_standings = group_standings
        return group_standings

    def get_advancing_teams(self, group_standings):
        """Return teams advancing to the Round of 32 from group standings."""
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
        """Assign third-place teams to Round of 32 slots using FIFA World Cup 2026 Annex C."""
        teams_by_group = {group_letter: team_name for team_name, group_letter in third_place_teams}
        key = "".join(sorted(teams_by_group))
        if key not in self.third_place_annex:
            raise ValueError(f"Annex C third-place assignment is missing for groups: {key}")

        return {
            int(match_id): teams_by_group[group_letter]
            for match_id, group_letter in self.third_place_annex[key].items()
        }

    def simulate_knockout_match(
        self,
        team_a,
        team_b,
        rest_days_diff: int = 0,
        travel_fatigue_a: float = 0.0,
        travel_fatigue_b: float = 0.0,
        match_number: int = None,
    ):
        """Simulate a single knockout match, including penalties and contextual adjustments."""
        actual_match = self._find_actual_match(
            team_a,
            team_b,
            stage="knockout",
            match_number=match_number,
        )
                
        if actual_match:
            score_a, score_b = self._orient_actual_score(
                actual_match, team_a, team_b
            )
            winner = self._actual_winner(
                actual_match, team_a, team_b, score_a, score_b
            )
            is_pk = score_a == score_b
            if winner is None:
                match_label = f"M{match_number}" if match_number else "knockout match"
                raise ValueError(
                    f"Tied actual result for {match_label} is missing an explicit winner"
                )
            return winner, score_a, score_b, is_pk

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
            
        # If the match is tied after 90 minutes, simulate 30 minutes of extra time.
        final_lambda_a, final_lambda_b = self.get_expected_goals(
            team_a,
            team_b,
            home_advantage=True,
            rest_days_diff=rest_days_diff,
            travel_fatigue_a=travel_fatigue_a,
            travel_fatigue_b=travel_fatigue_b,
        )
        
        # Extra-time goals use one third of the 90-minute expected goals.
        goals_a_et = int(self.np_rng.poisson(final_lambda_a / 3.0))
        goals_b_et = int(self.np_rng.poisson(final_lambda_b / 3.0))
        
        score_a += goals_a_et
        score_b += goals_b_et
        
        if score_a != score_b:
            winner = team_a if score_a > score_b else team_b
            return winner, score_a, score_b, False
            
        # If extra time is still tied, decide the match on penalties.
        rating_a, rating_b = self.get_adjusted_ratings(
            team_a,
            team_b,
            home_advantage=True,
            rest_days_diff=rest_days_diff,
        )
        win_prob_pk_a = self.elo_system.expected_score(rating_a, rating_b)
        if self.rng.random() < win_prob_pk_a:
            return team_a, score_a, score_b, True
        else:
            return team_b, score_a, score_b, True

    def simulate_knockout_match_consensus(
        self,
        team_a,
        team_b,
        rest_days_diff: int = 0,
        travel_fatigue_a: float = 0.0,
        travel_fatigue_b: float = 0.0,
        runs: int = 10000,
        match_number: int = None,
    ):
        """Return the most frequent advancing team and representative score across many simulations."""
        runs = max(1, int(runs))

        actual_match = self._find_actual_match(
            team_a,
            team_b,
            stage="knockout",
            match_number=match_number,
        )
        if actual_match is not None:
            winner, score_a, score_b, is_pk = self.simulate_knockout_match(
                team_a,
                team_b,
                rest_days_diff=rest_days_diff,
                travel_fatigue_a=travel_fatigue_a,
                travel_fatigue_b=travel_fatigue_b,
                match_number=match_number,
            )
            return (
                winner,
                score_a,
                score_b,
                is_pk,
                Counter({winner: runs}),
            )

        winner_counts = Counter()
        score_counts = Counter()

        for _ in range(runs):
            winner, score_a, score_b, is_pk = self.simulate_knockout_match(
                team_a,
                team_b,
                rest_days_diff=rest_days_diff,
                travel_fatigue_a=travel_fatigue_a,
                travel_fatigue_b=travel_fatigue_b,
                match_number=match_number,
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

    @staticmethod
    def _travel_fatigue(from_region: int, to_region: int) -> float:
        difference = abs(to_region - from_region)
        if difference >= 3:
            return LONG_REGION_TRAVEL_PENALTY
        if difference > 0:
            return NEAR_REGION_TRAVEL_PENALTY
        return 0.0

    def build_group_stage_team_states(self, teams=None):
        """Build each team's latest date/venue state from its real schedule."""

        if teams is None:
            teams = [team for group in self.groups.values() for team in group]
        states = {}
        for team in teams:
            last_match = self.schedule.last_group_match(team)
            states[team] = {
                "last_date": last_match.day_number,
                "last_region": last_match.region,
                "last_match_number": last_match.match_number,
                "last_city": last_match.host_city,
            }
        return states

    def build_group_stage_match_contexts(self):
        """Return rest and travel inputs for every scheduled group match."""

        team_states = {}
        contexts = {}
        matches = sorted(
            self.schedule.stage_matches("group-stage"),
            key=lambda match: (
                match.day_number,
                match.kickoff_utc or "",
                match.match_number,
            ),
        )
        for match in matches:
            state_a = team_states.get(match.home_team)
            state_b = team_states.get(match.away_team)
            rest_a = match.day_number - state_a["last_date"] if state_a else 0
            rest_b = match.day_number - state_b["last_date"] if state_b else 0
            contexts[match.match_number] = {
                "match_number": match.match_number,
                "date": match.date,
                "host_city": match.host_city,
                "region": match.region,
                "rest_days_a": rest_a,
                "rest_days_b": rest_b,
                "rest_days_diff": rest_a - rest_b if state_a and state_b else 0,
                "travel_fatigue_a": (
                    self._travel_fatigue(state_a["last_region"], match.region)
                    if state_a
                    else 0.0
                ),
                "travel_fatigue_b": (
                    self._travel_fatigue(state_b["last_region"], match.region)
                    if state_b
                    else 0.0
                ),
            }
            state = {
                "last_date": match.day_number,
                "last_region": match.region,
                "last_match_number": match.match_number,
                "last_city": match.host_city,
            }
            team_states[match.home_team] = dict(state)
            team_states[match.away_team] = dict(state)
        return contexts

    def get_match_context(self, match_number, team_a, team_b, team_states):
        """Return schedule-derived rest and travel context for a match."""

        match = self.schedule.match(match_number)
        state_a = team_states[team_a]
        state_b = team_states[team_b]
        rest_a = match.day_number - state_a["last_date"]
        rest_b = match.day_number - state_b["last_date"]
        fatigue_a = self._travel_fatigue(state_a["last_region"], match.region)
        fatigue_b = self._travel_fatigue(state_b["last_region"], match.region)
        return {
            "match_number": match.match_number,
            "date": match.date,
            "host_city": match.host_city,
            "region": match.region,
            "rest_days_a": rest_a,
            "rest_days_b": rest_b,
            "rest_days_diff": rest_a - rest_b,
            "travel_fatigue_a": fatigue_a,
            "travel_fatigue_b": fatigue_b,
        }

    def simulate_knockout_round(
        self,
        pairings,
        match_start_id,
        team_states,
        consensus_runs=None,
        match_numbers=None,
    ):
        """Simulate one knockout round with schedule-derived context."""

        if match_numbers is None:
            match_numbers = [match_start_id + idx for idx in range(len(pairings))]
        if len(match_numbers) != len(pairings):
            raise ValueError("match_numbers and pairings must have the same length")

        round_results = []
        winners = []
        for match_id, (team_a, team_b) in zip(match_numbers, pairings):
            context = self.get_match_context(
                match_id, team_a, team_b, team_states
            )
            rest_days_diff = context["rest_days_diff"]
            fatigue_a = context["travel_fatigue_a"]
            fatigue_b = context["travel_fatigue_b"]

            if consensus_runs and consensus_runs > 1:
                effective_runs = max(1, int(consensus_runs))
                winner, score_a, score_b, is_pk, winner_counts = self.simulate_knockout_match_consensus(
                    team_a,
                    team_b,
                    rest_days_diff=rest_days_diff,
                    travel_fatigue_a=fatigue_a,
                    travel_fatigue_b=fatigue_b,
                    runs=effective_runs,
                    match_number=match_id,
                )
                advance_prob_a = winner_counts[team_a] / effective_runs
                advance_prob_b = winner_counts[team_b] / effective_runs
            else:
                effective_runs = None
                winner, score_a, score_b, is_pk = self.simulate_knockout_match(
                    team_a,
                    team_b,
                    rest_days_diff=rest_days_diff,
                    travel_fatigue_a=fatigue_a,
                    travel_fatigue_b=fatigue_b,
                    match_number=match_id,
                )
                advance_prob_a = None
                advance_prob_b = None

            match = self.schedule.match(match_id)
            latest_state = {
                "last_date": match.day_number,
                "last_region": match.region,
                "last_match_number": match.match_number,
                "last_city": match.host_city,
            }
            # Both teams played this match. Keeping the loser's state is
            # required when the semifinal losers feed the third-place match.
            team_states[team_a] = dict(latest_state)
            team_states[team_b] = dict(latest_state)

            round_results.append({
                "match_number": match_id,
                "match_id": match_id,
                "date": match.date,
                "host_city": match.host_city,
                "region": match.region,
                "team_a": team_a,
                "team_b": team_b,
                "score_a": score_a,
                "score_b": score_b,
                "winner": winner,
                "is_pk": is_pk,
                "rest_days_a": context["rest_days_a"],
                "rest_days_b": context["rest_days_b"],
                "rest_days_diff": rest_days_diff,
                "travel_fatigue_a": fatigue_a,
                "travel_fatigue_b": fatigue_b,
                "advance_prob_a": advance_prob_a,
                "advance_prob_b": advance_prob_b,
                "consensus_runs": effective_runs,
            })
            winners.append(winner)

        return round_results, winners

    def _knockout_bracket_inputs(self, group_standings):
        third_places = []
        team_by_code = team_codes_from_standings(group_standings)
        advancing_teams = []
        for group_name, group_teams in group_standings.items():
            group_letter = group_name.split()[-1]
            advancing_teams.extend((group_teams[0][0], group_teams[1][0]))
            third_places.append({
                "team_name": group_teams[2][0],
                "group": group_letter,
                "stats": group_teams[2][1],
            })

        top_8_thirds = self._sort_third_places(third_places)[:8]
        advancing_teams.extend(team["team_name"] for team in top_8_thirds)
        third_assignment = self.match_thirds(
            [(team["team_name"], team["group"]) for team in top_8_thirds]
        )
        return team_by_code, third_assignment, advancing_teams

    def simulate_knockout_stage(self, consensus_runs=None):
        """Simulate the schedule-defined knockout bracket through the final."""

        if not self.last_standings:
            raise ValueError("Group-stage standings are not available.")
        validate_knockout_bracket(self.schedule)

        team_by_code, third_assignment, advancing_teams = (
            self._knockout_bracket_inputs(self.last_standings)
        )
        team_states = self.build_group_stage_team_states(advancing_teams)
        winners = {}
        losers = {}
        results = {}

        for schedule_stage, result_label in SIMULATED_KNOCKOUT_STAGES:
            scheduled_matches = self.schedule.stage_matches(schedule_stage)
            pairings = [
                resolve_match_teams(
                    match,
                    team_by_code=team_by_code,
                    third_assignment=third_assignment,
                    winners=winners,
                    losers=losers,
                )
                for match in scheduled_matches
            ]
            match_numbers = [match.match_number for match in scheduled_matches]
            round_results, round_winners = self.simulate_knockout_round(
                pairings,
                match_numbers[0],
                team_states,
                consensus_runs=consensus_runs,
                match_numbers=match_numbers,
            )
            results[result_label] = round_results
            for match, pairing, winner in zip(
                scheduled_matches, pairings, round_winners
            ):
                winners[match.match_number] = winner
                losers[match.match_number] = (
                    pairing[1] if winner == pairing[0] else pairing[0]
                )

        results["Champion"] = winners[104]
        return results

    def build_current_knockout_state(self, group_standings=None):
        """Build a non-simulated bracket from completed actual results.

        Known future participants are resolved, while unavailable sources are
        returned as ``Winner Mxx`` labels.  Confirmed future matchups also
        include the same schedule-derived rest/travel context used by the
        Monte Carlo engine.
        """

        standings = group_standings or self.last_standings
        if not standings:
            standings = self.simulate_group_stage()
        validate_knockout_bracket(self.schedule)

        team_by_code, third_assignment, advancing_teams = (
            self._knockout_bracket_inputs(standings)
        )
        team_states = self.build_group_stage_team_states(advancing_teams)
        winners = {}
        losers = {}
        results = {}

        for schedule_stage, result_label in SIMULATED_KNOCKOUT_STAGES:
            round_results = []
            for match in self.schedule.stage_matches(schedule_stage):
                team_a, team_b = resolve_match_team_labels(
                    match,
                    team_by_code=team_by_code,
                    third_assignment=third_assignment,
                    winners=winners,
                    losers=losers,
                )
                known_teams = team_a in team_states and team_b in team_states
                actual_match = self._find_actual_match(
                    team_a if known_teams else None,
                    team_b if known_teams else None,
                    stage="knockout",
                    match_number=match.match_number,
                )

                if actual_match and not known_teams:
                    team_a = actual_match.get("team_a")
                    team_b = actual_match.get("team_b")
                    known_teams = team_a in team_states and team_b in team_states

                context = None
                if known_teams:
                    context = self.get_match_context(
                        match.match_number, team_a, team_b, team_states
                    )

                score_a = score_b = winner = None
                is_pk = False
                if actual_match:
                    score_a, score_b = self._orient_actual_score(
                        actual_match, team_a, team_b
                    )
                    winner = self._actual_winner(
                        actual_match, team_a, team_b, score_a, score_b
                    )
                    is_pk = winner is not None and score_a == score_b

                result = {
                    "match_number": match.match_number,
                    "match_id": match.match_number,
                    "date": match.date,
                    "host_city": match.host_city,
                    "region": match.region,
                    "team_a": team_a,
                    "team_b": team_b,
                    "score_a": score_a,
                    "score_b": score_b,
                    "winner": winner,
                    "is_pk": is_pk,
                    "played": winner is not None,
                    "rest_days_a": context["rest_days_a"] if context else None,
                    "rest_days_b": context["rest_days_b"] if context else None,
                    "rest_days_diff": context["rest_days_diff"] if context else None,
                    "travel_fatigue_a": context["travel_fatigue_a"] if context else None,
                    "travel_fatigue_b": context["travel_fatigue_b"] if context else None,
                }
                round_results.append(result)

                if winner is not None:
                    winners[match.match_number] = winner
                    losers[match.match_number] = (
                        team_b if winner == team_a else team_a
                    )
                    latest_state = {
                        "last_date": match.day_number,
                        "last_region": match.region,
                        "last_match_number": match.match_number,
                        "last_city": match.host_city,
                    }
                    team_states[team_a] = dict(latest_state)
                    team_states[team_b] = dict(latest_state)

            results[result_label] = round_results

        results["Champion"] = winners.get(104, "TBD")
        return results


if __name__ == "__main__":
    elo = EloSystem()
    elo.load_ratings()
    
    sim = WorldCupSimulation(elo)
    
    print("Simulating the group stage...")
    standings = sim.simulate_group_stage()
    
    print("Resolving Round of 32 teams...")
    print("Simulating the knockout stage...\n")
    knockout_results = sim.simulate_knockout_stage()
    
    print("--- [Final Result] ---")
    final_match = knockout_results["Final"][0]
    pk_str = " (penalties)" if final_match["is_pk"] else ""
    print(f"{final_match['team_a']} {final_match['score_a']} : {final_match['score_b']} {final_match['team_b']}{pk_str}")
    print(f"Champion: {knockout_results['Champion']}")
