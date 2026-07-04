import json
import os
from itertools import combinations


def load_json(path, default_val=None):
    if default_val is None:
        default_val = {}
    if not path or not os.path.exists(path):
        return default_val
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return default_val


def flatten_group_teams(groups):
    if not isinstance(groups, dict):
        return set()

    teams = set()
    for group_teams in groups.values():
        teams.update(group_teams)
    return teams


def _group_match_keys(actual_results, group_teams):
    group_set = set(group_teams)
    keys = set()
    for match in actual_results:
        if match.get("stage", "group") != "group":
            continue
        team_a = match.get("team_a")
        team_b = match.get("team_b")
        if team_a in group_set and team_b in group_set:
            keys.add(frozenset((team_a, team_b)))
    return keys


def group_stage_is_complete(groups, actual_results):
    if not groups:
        return False

    for group_teams in groups.values():
        expected_keys = {
            frozenset(pair)
            for pair in combinations(group_teams, 2)
        }
        if not expected_keys.issubset(_group_match_keys(actual_results, group_teams)):
            return False

    return True


def get_knockout_losers(actual_results):
    losers = set()
    for match in actual_results:
        if match.get("stage", "group") == "group":
            continue

        winner = match.get("winner")
        team_a = match.get("team_a")
        team_b = match.get("team_b")
        if not winner or winner not in {team_a, team_b}:
            continue

        loser = team_b if winner == team_a else team_a
        if loser:
            losers.add(loser)

    return losers


def _update_group_stats(team_a, team_b, score_a, score_b, stats, group_match_results):
    group_match_results[(team_a, team_b)] = (score_a, score_b)
    group_match_results[(team_b, team_a)] = (score_b, score_a)

    stats[team_a]["gf"] += score_a
    stats[team_a]["ga"] += score_b
    stats[team_a]["gd"] += score_a - score_b
    stats[team_b]["gf"] += score_b
    stats[team_b]["ga"] += score_a
    stats[team_b]["gd"] += score_b - score_a

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
        stats[team_a]["d"] += 1
        stats[team_b]["pts"] += 1
        stats[team_b]["d"] += 1


def _head_to_head_stats(team_names, group_match_results):
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


def _ranking_value(team, fifa_rankings):
    try:
        return int(fifa_rankings.get(team, 999))
    except (TypeError, ValueError):
        return 999


def _conduct_value(team, team_conduct_scores):
    try:
        return int(team_conduct_scores.get(team, 0))
    except (TypeError, ValueError):
        return 0


def _rank_equal_points_teams(team_names, stats, group_match_results, fifa_rankings, team_conduct_scores):
    if len(team_names) <= 1:
        return team_names

    h2h = _head_to_head_stats(team_names, group_match_results)
    for key in ("pts", "gd", "gf"):
        grouped = {}
        for team in team_names:
            grouped.setdefault(h2h[team][key], []).append(team)
        if len(grouped) > 1:
            ranked = []
            for value in sorted(grouped.keys(), reverse=True):
                ranked.extend(
                    _rank_equal_points_teams(
                        grouped[value],
                        stats,
                        group_match_results,
                        fifa_rankings,
                        team_conduct_scores,
                    )
                )
            return ranked

    return sorted(
        team_names,
        key=lambda team: (
            -stats[team]["gd"],
            -stats[team]["gf"],
            -_conduct_value(team, team_conduct_scores),
            _ranking_value(team, fifa_rankings),
        ),
    )


def _sort_group_standings(stats, group_match_results, fifa_rankings, team_conduct_scores):
    grouped_by_points = {}
    for team in stats:
        grouped_by_points.setdefault(stats[team]["pts"], []).append(team)

    ranked_teams = []
    for points in sorted(grouped_by_points.keys(), reverse=True):
        ranked_teams.extend(
            _rank_equal_points_teams(
                grouped_by_points[points],
                stats,
                group_match_results,
                fifa_rankings,
                team_conduct_scores,
            )
        )

    return [(team, stats[team]) for team in ranked_teams]


def _sort_third_places(third_places, fifa_rankings, team_conduct_scores):
    return sorted(
        third_places,
        key=lambda team: (
            -team["stats"]["pts"],
            -team["stats"]["gd"],
            -team["stats"]["gf"],
            -_conduct_value(team["team_name"], team_conduct_scores),
            _ranking_value(team["team_name"], fifa_rankings),
        ),
    )


def _actual_group_standings(groups, actual_results, fifa_rankings, team_conduct_scores):
    standings = {}

    for group_name, group_teams in groups.items():
        stats = {
            team: {"pts": 0, "gd": 0, "gf": 0, "ga": 0, "w": 0, "d": 0, "l": 0}
            for team in group_teams
        }
        group_match_results = {}

        for team_a, team_b in combinations(group_teams, 2):
            actual_match = None
            for match in actual_results:
                if (
                    match.get("stage", "group") == "group"
                    and {match.get("team_a"), match.get("team_b")} == {team_a, team_b}
                ):
                    actual_match = match
                    break
            if not actual_match:
                raise ValueError(f"Missing group result for {team_a} vs {team_b}")

            score_a = actual_match["score_a"] if actual_match["team_a"] == team_a else actual_match["score_b"]
            score_b = actual_match["score_b"] if actual_match["team_a"] == team_a else actual_match["score_a"]
            _update_group_stats(team_a, team_b, score_a, score_b, stats, group_match_results)

        standings[group_name] = _sort_group_standings(
            stats,
            group_match_results,
            fifa_rankings,
            team_conduct_scores,
        )

    return standings


def _advancing_group_teams(standings, fifa_rankings, team_conduct_scores):
    first_places = []
    second_places = []
    third_places = []

    for group, teams in standings.items():
        first_places.append(teams[0])
        second_places.append(teams[1])
        group_letter = group.split(" ")[1]
        third_places.append({
            "team_name": teams[2][0],
            "group": group_letter,
            "stats": teams[2][1],
        })

    top_8_thirds = _sort_third_places(third_places, fifa_rankings, team_conduct_scores)[:8]
    return {
        team[0]
        for team in first_places + second_places
    } | {
        team["team_name"]
        for team in top_8_thirds
    }


def get_active_teams(
    groups,
    actual_results,
    elo_ratings=None,
    groups_file="data/groups.json",
    fifa_rankings_file="data/fifa_rankings.json",
    team_conduct_file="data/team_conduct_scores.json",
    third_place_annex_file="data/third_place_annex_c.json",
):
    """Return teams that can still affect future tournament predictions."""
    if not isinstance(groups, dict):
        groups = {}
    if not isinstance(actual_results, list):
        actual_results = []
    if not isinstance(elo_ratings, dict):
        elo_ratings = {}

    tournament_teams = flatten_group_teams(groups)
    if elo_ratings:
        tournament_teams.update(elo_ratings.keys())

    active_teams = set(tournament_teams)
    if not active_teams:
        return set()

    if group_stage_is_complete(groups, actual_results):
        try:
            fifa_rankings = load_json(fifa_rankings_file, {})
            team_conduct_scores = load_json(team_conduct_file, {})
            standings = _actual_group_standings(
                groups,
                actual_results,
                fifa_rankings,
                team_conduct_scores,
            )
            active_teams = _advancing_group_teams(standings, fifa_rankings, team_conduct_scores)
        except Exception:
            # Missing tiebreak data should not cause data refreshes to drop teams.
            active_teams = set(tournament_teams)

    active_teams -= get_knockout_losers(actual_results)
    return active_teams


def load_active_teams(
    groups_path="data/groups.json",
    actual_results_path="data/actual_results.json",
    ratings_path="data/elo_ratings.json",
):
    groups = load_json(groups_path, {})
    actual_results = load_json(actual_results_path, [])
    elo_ratings = load_json(ratings_path, {})
    return get_active_teams(
        groups,
        actual_results,
        elo_ratings=elo_ratings,
        groups_file=groups_path,
    )


def filter_team_map(data, teams):
    team_set = set(teams)
    return {
        team: items
        for team, items in data.items()
        if team in team_set
    }
