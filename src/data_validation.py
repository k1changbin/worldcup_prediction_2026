"""Schema and cross-file validation for tournament data snapshots."""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

from src.bracket import validate_knockout_bracket
from src.schedule import TournamentSchedule


REQUIRED_SCHEDULE_FIELDS = {
    "matchNumber",
    "date",
    "stage",
    "homeTeam",
    "awayTeam",
    "hostCity",
}
VALID_POSITIONS = {"Goalkeeper", "Defender", "Midfielder", "Forward"}


def validate_groups(groups):
    errors = []
    if not isinstance(groups, dict) or len(groups) != 12:
        return ["groups.json must contain exactly 12 groups"]

    all_teams = []
    for group_name, teams in groups.items():
        if not isinstance(teams, list) or len(teams) != 4:
            errors.append(f"{group_name} must contain exactly four teams")
            continue
        for index, team in enumerate(teams):
            if not isinstance(team, str) or not team.strip():
                errors.append(f"{group_name} team {index} has an invalid name")
            else:
                all_teams.append(team)

    duplicates = sorted(
        team for team, count in Counter(all_teams).items() if count > 1
    )
    if len(all_teams) != 48:
        errors.append("groups.json must contain 48 team slots")
    if duplicates:
        errors.append(f"teams appear in multiple groups: {', '.join(duplicates)}")
    return errors


def validate_schedule(schedule):
    errors = []
    if not isinstance(schedule, list):
        return ["schedule.json must contain a list"]
    if len(schedule) != 104:
        errors.append("schedule.json must contain exactly 104 matches")

    numbers = []
    stages = []
    for index, match in enumerate(schedule):
        if not isinstance(match, dict):
            errors.append(f"schedule entry {index} is not an object")
            continue
        missing = REQUIRED_SCHEDULE_FIELDS - set(match)
        if missing:
            errors.append(
                f"schedule entry {index} is missing: {', '.join(sorted(missing))}"
            )
        for field in ("date", "stage", "homeTeam", "awayTeam", "hostCity"):
            if not isinstance(match.get(field), str) or not match.get(field).strip():
                errors.append(f"schedule entry {index} has an invalid {field}")
        number = match.get("matchNumber")
        if isinstance(number, int) and not isinstance(number, bool):
            numbers.append(number)
        else:
            errors.append(f"schedule entry {index} has an invalid matchNumber")
        stages.append(match.get("stage"))

    if sorted(numbers) != list(range(1, 105)):
        errors.append("schedule match numbers must be unique and cover 1 through 104")
    expected_stages = {
        "group-stage": 72,
        "round-of-32": 16,
        "round-of-16": 8,
        "quarter-finals": 4,
        "semi-finals": 2,
        "third-place": 1,
        "final": 1,
    }
    if Counter(stages) != Counter(expected_stages):
        errors.append("schedule stage counts do not match the 104-match format")
    try:
        normalized = TournamentSchedule.from_mappings(schedule)
        validate_knockout_bracket(normalized)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"schedule structure is invalid: {exc}")
    return errors


def validate_group_schedule(groups, schedule):
    """Validate all 72 group fixtures against the declared groups."""
    if not isinstance(groups, dict) or not isinstance(schedule, list):
        return []

    errors = []
    expected_letters = {str(name).split()[-1] for name in groups}
    for index, match in enumerate(schedule):
        if not isinstance(match, dict) or match.get("stage") != "group-stage":
            continue
        if match.get("group") not in expected_letters:
            errors.append(f"group-stage schedule entry {index} has an invalid group")

    for group_name, teams in groups.items():
        if (
            not isinstance(teams, list)
            or len(teams) != 4
            or any(not isinstance(team, str) or not team for team in teams)
        ):
            continue
        letter = str(group_name).split()[-1]
        matches = [
            match
            for match in schedule
            if isinstance(match, dict)
            and match.get("stage") == "group-stage"
            and match.get("group") == letter
        ]
        expected_pairs = {
            frozenset(pair) for pair in combinations(teams, 2)
        }
        actual_pairs = []
        for index, match in enumerate(matches):
            home = match.get("homeTeam")
            away = match.get("awayTeam")
            if isinstance(home, str) and isinstance(away, str):
                actual_pairs.append(frozenset((home, away)))
            else:
                actual_pairs.append(frozenset((f"invalid-{index}",)))
        if len(matches) != 6 or set(actual_pairs) != expected_pairs:
            errors.append(
                f"schedule fixtures for {group_name} must contain each pairing once"
            )
        duplicates = [
            pair for pair, count in Counter(actual_pairs).items() if count > 1
        ]
        if duplicates:
            errors.append(f"schedule fixtures for {group_name} contain duplicates")
    return errors


def validate_third_place_annex(annex):
    """Validate all 495 FIFA Annex C third-place combinations."""
    if not isinstance(annex, dict):
        return ["third_place_annex_c.json must contain an object"]

    letters = "ABCDEFGHIJKL"
    expected_combinations = {
        "".join(selection) for selection in combinations(letters, 8)
    }
    errors = []
    if set(annex) != expected_combinations:
        errors.append(
            "third_place_annex_c.json must contain all 495 eight-group combinations"
        )

    expected_match_ids = {74, 77, 79, 80, 81, 82, 85, 87}
    for combination_key, assignment in annex.items():
        if (
            not isinstance(combination_key, str)
            or len(combination_key) != 8
            or any(letter not in letters for letter in combination_key)
        ):
            errors.append(f"Annex C has an invalid combination key: {combination_key!r}")
            continue
        if not isinstance(assignment, dict):
            errors.append(f"Annex C combination {combination_key} is not an object")
            continue
        try:
            match_ids = {int(match_id) for match_id in assignment}
        except (TypeError, ValueError):
            errors.append(f"Annex C combination {combination_key} has invalid match IDs")
            continue
        if match_ids != expected_match_ids:
            errors.append(
                f"Annex C combination {combination_key} has invalid R32 slots"
            )
        assigned_groups = list(assignment.values())
        if (
            len(assigned_groups) != 8
            or set(assigned_groups) != set(combination_key)
        ):
            errors.append(
                f"Annex C combination {combination_key} must assign each group once"
            )
    return errors


def validate_actual_results(
    results,
    tournament_teams,
    schedule=None,
    require_match_numbers=False,
):
    errors = []
    if not isinstance(results, list):
        return ["actual_results.json must contain a list"]

    valid_teams = set(tournament_teams)
    schedule_by_number = {
        match.get("matchNumber"): match
        for match in (schedule or [])
        if isinstance(match, dict)
    }
    seen_numbers = set()
    seen_legacy_keys = set()

    for index, match in enumerate(results):
        if not isinstance(match, dict):
            errors.append(f"actual result {index} is not an object")
            continue

        team_a = match.get("team_a")
        team_b = match.get("team_b")
        teams_valid = (
            isinstance(team_a, str)
            and isinstance(team_b, str)
            and team_a in valid_teams
            and team_b in valid_teams
            and team_a != team_b
        )
        if not teams_valid:
            errors.append(f"actual result {index} has invalid teams")

        scores_valid = True
        for score_key in ("score_a", "score_b"):
            score = match.get(score_key)
            if not isinstance(score, int) or isinstance(score, bool) or score < 0:
                errors.append(f"actual result {index} has invalid {score_key}")
                scores_valid = False

        stage = match.get("stage", "group")
        if stage not in {"group", "knockout"}:
            errors.append(f"actual result {index} has invalid stage")

        winner = match.get("winner")
        valid_winners = {team_a, team_b} if teams_valid else set()
        if winner is not None and (
            not isinstance(winner, str) or winner not in valid_winners
        ):
            errors.append(f"actual result {index} has invalid winner")
        if stage == "group" and scores_valid:
            expected = None
            if match.get("score_a") > match.get("score_b"):
                expected = team_a
            elif match.get("score_b") > match.get("score_a"):
                expected = team_b
            if winner != expected:
                errors.append(f"actual result {index} has inconsistent group winner")
        if stage == "knockout":
            if not isinstance(winner, str) or winner not in valid_winners:
                errors.append(f"actual result {index} is missing a knockout winner")
            elif scores_valid and match.get("score_a") != match.get("score_b"):
                expected = (
                    team_a
                    if match.get("score_a") > match.get("score_b")
                    else team_b
                )
                if winner != expected:
                    errors.append(
                        f"actual result {index} has inconsistent knockout winner"
                    )

        match_number = match.get("match_number")
        if match_number is not None:
            if isinstance(match_number, bool) or not isinstance(match_number, int):
                errors.append(f"actual result {index} has invalid match_number")
            elif match_number not in schedule_by_number:
                errors.append(f"actual result {index} has unknown match_number")
            elif match_number in seen_numbers:
                errors.append(f"match_number {match_number} is duplicated")
            else:
                scheduled = schedule_by_number[match_number]
                expected_stage = (
                    "group"
                    if scheduled.get("stage") == "group-stage"
                    else "knockout"
                )
                if stage != expected_stage:
                    errors.append(
                        f"actual result {index} stage does not match M{match_number}"
                    )
                if match.get("date") != scheduled.get("date"):
                    errors.append(
                        f"actual result {index} date does not match M{match_number}"
                    )
                if expected_stage == "group" and teams_valid and {
                    team_a,
                    team_b,
                } != {
                    scheduled.get("homeTeam"),
                    scheduled.get("awayTeam"),
                }:
                    errors.append(
                        f"actual result {index} teams do not match M{match_number}"
                    )
            if (
                isinstance(match_number, int)
                and not isinstance(match_number, bool)
                and match_number in schedule_by_number
            ):
                seen_numbers.add(match_number)
        else:
            if require_match_numbers:
                errors.append(f"actual result {index} is missing match_number")
            legacy_pair = (
                frozenset((team_a, team_b))
                if teams_valid
                else ("invalid", index)
            )
            legacy_key = (match.get("date"), legacy_pair)
            if legacy_key in seen_legacy_keys:
                errors.append(f"actual result {index} duplicates a date/matchup")
            seen_legacy_keys.add(legacy_key)

    if require_match_numbers and seen_numbers:
        expected_prefix = set(range(1, max(seen_numbers) + 1))
        if seen_numbers != expected_prefix:
            errors.append("actual result match numbers must form a contiguous prefix")

    return errors


def validate_squads(squads, tournament_teams):
    errors = []
    if not isinstance(squads, dict):
        return ["squads.json must contain an object"]
    unknown = sorted(set(squads) - set(tournament_teams))
    if unknown:
        errors.append(f"squads.json has unknown teams: {', '.join(unknown)}")
    for team, players in squads.items():
        if not isinstance(players, list) or not players:
            errors.append(f"{team} squad must contain players")
            continue
        names = set()
        for index, player in enumerate(players):
            if not isinstance(player, dict):
                errors.append(f"{team} player {index} is not an object")
                continue
            name = player.get("name")
            position = player.get("position")
            value = player.get("value_eur")
            if not isinstance(name, str) or not name.strip():
                errors.append(f"{team} player {index} has an invalid name")
            elif name.casefold() in names:
                errors.append(f"{team} has duplicate player {name}")
            else:
                names.add(name.casefold())
            if position not in VALID_POSITIONS:
                errors.append(f"{team} player {index} has an invalid position")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                errors.append(f"{team} player {index} has an invalid value_eur")
    return errors


def validate_absences(absences, tournament_teams):
    errors = []
    if not isinstance(absences, dict):
        return ["absences.json must contain an object"]
    unknown = sorted(set(absences) - set(tournament_teams))
    if unknown:
        errors.append(f"absences.json has unknown teams: {', '.join(unknown)}")
    for team, items in absences.items():
        if not isinstance(items, list):
            errors.append(f"{team} absences must contain a list")
            continue
        seen = set()
        for index, item in enumerate(items):
            if isinstance(item, str):
                name = item.strip()
                absence_type = "injury"
            elif isinstance(item, dict):
                name = item.get("name")
                absence_type = item.get("type", "injury")
                if absence_type == "suspension":
                    served = item.get("served_at_count")
                    if (
                        not isinstance(served, int)
                        or isinstance(served, bool)
                        or served < 1
                    ):
                        errors.append(
                            f"{team} absence {index} has invalid served_at_count"
                        )
                    reason = item.get("reason")
                    if reason not in {"yellow_cards", "red_card", "disciplinary"}:
                        errors.append(
                            f"{team} absence {index} has an invalid suspension reason"
                        )
                    length = item.get("suspension_length", 1)
                    if (
                        not isinstance(length, int)
                        or isinstance(length, bool)
                        or length < 1
                    ):
                        errors.append(
                            f"{team} absence {index} has invalid suspension_length"
                        )
                    elif reason == "yellow_cards" and length != 1:
                        errors.append(
                            f"{team} absence {index} has an invalid yellow-card suspension length"
                        )
            else:
                errors.append(f"{team} absence {index} is invalid")
                continue
            if not isinstance(name, str) or not name.strip():
                errors.append(f"{team} absence {index} has an invalid name")
                continue
            key = (name.casefold(), absence_type)
            if key in seen:
                errors.append(f"{team} has duplicate absence {name}")
            seen.add(key)
    return errors


def validate_calibration_matches(records):
    errors = []
    if not isinstance(records, list):
        return ["model_calibration_matches.json must contain a list"]
    year_counts = Counter()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"calibration record {index} is not an object")
            continue
        missing = {
            "year", "date", "team_a", "team_b", "score_a", "score_b",
            "elo_a", "elo_b", "host_a", "host_b",
        } - set(record)
        if missing:
            errors.append(
                f"calibration record {index} is missing: {', '.join(sorted(missing))}"
            )
            continue
        if not isinstance(record["year"], int) or isinstance(record["year"], bool):
            errors.append(f"calibration record {index} has an invalid year")
        else:
            year_counts[record["year"]] += 1
        if not isinstance(record["date"], str):
            errors.append(f"calibration record {index} has an invalid date")
        if (
            not isinstance(record["team_a"], str)
            or not isinstance(record["team_b"], str)
            or not record["team_a"]
            or not record["team_b"]
        ):
            errors.append(f"calibration record {index} has invalid teams")
        for key in ("score_a", "score_b"):
            if (
                not isinstance(record[key], int)
                or isinstance(record[key], bool)
                or record[key] < 0
            ):
                errors.append(f"calibration record {index} has invalid {key}")
        for key in ("elo_a", "elo_b"):
            if (
                isinstance(record[key], bool)
                or not isinstance(record[key], (int, float))
                or not math.isfinite(float(record[key]))
                or record[key] <= 0
            ):
                errors.append(f"calibration record {index} has invalid {key}")
        for key in ("host_a", "host_b"):
            if not isinstance(record[key], bool):
                errors.append(f"calibration record {index} has invalid {key}")
        if (
            isinstance(record["team_a"], str)
            and isinstance(record["team_b"], str)
            and record["team_a"] >= record["team_b"]
        ):
            errors.append(f"calibration record {index} is not canonically ordered")
    if year_counts != Counter({2018: 48, 2022: 48}):
        errors.append("calibration data must contain 48 group matches for 2018 and 2022")
    return errors


def validate_dataset(
    groups,
    ratings,
    fifa_rankings,
    conduct,
    schedule,
    results,
    squads=None,
    absences=None,
    pre_tournament_ratings=None,
    calibration_matches=None,
    third_place_annex=None,
):
    errors = []
    errors.extend(validate_groups(groups))
    errors.extend(validate_schedule(schedule))
    errors.extend(validate_group_schedule(groups, schedule))
    if third_place_annex is not None:
        errors.extend(validate_third_place_annex(third_place_annex))

    tournament_teams = {
        team
        for group_teams in groups.values()
        if isinstance(group_teams, list)
        for team in group_teams
        if isinstance(team, str)
    } if isinstance(groups, dict) else set()
    for label, mapping in (
        ("elo_ratings.json", ratings),
        ("fifa_rankings.json", fifa_rankings),
        ("team_conduct_scores.json", conduct),
    ):
        if not isinstance(mapping, dict):
            errors.append(f"{label} must contain an object")
            continue
        missing = sorted(tournament_teams - set(mapping))
        extra = sorted(set(mapping) - tournament_teams)
        if missing:
            errors.append(f"{label} is missing teams: {', '.join(missing)}")
        if extra:
            errors.append(f"{label} has unknown teams: {', '.join(extra)}")
        for team, value in mapping.items():
            if label == "team_conduct_scores.json":
                valid = (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value <= 0
                )
            elif label == "fifa_rankings.json":
                valid = (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value > 0
                )
            else:
                valid = (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and value > 0
                )
            if not valid:
                errors.append(f"{label} has an invalid value for {team}")

    errors.extend(
        validate_actual_results(
            results,
            tournament_teams,
            schedule,
            require_match_numbers=True,
        )
    )
    if squads is not None:
        errors.extend(validate_squads(squads, tournament_teams))
    if absences is not None:
        errors.extend(validate_absences(absences, tournament_teams))
    if pre_tournament_ratings is not None:
        if not isinstance(pre_tournament_ratings, dict):
            errors.append("elo_ratings_pre_tournament.json must contain an object")
        elif set(pre_tournament_ratings) != tournament_teams:
            errors.append(
                "elo_ratings_pre_tournament.json must contain exactly the tournament teams"
            )
        else:
            for team, value in pre_tournament_ratings.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or value <= 0
                ):
                    errors.append(
                        f"elo_ratings_pre_tournament.json has an invalid value for {team}"
                    )
    if calibration_matches is not None:
        errors.extend(validate_calibration_matches(calibration_matches))
    return errors
