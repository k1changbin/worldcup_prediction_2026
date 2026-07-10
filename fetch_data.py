import argparse
import json
import math
import os
from datetime import date

import httpx

from src.bracket import build_round_of_32_matchups, team_codes_from_standings
from src.io_utils import atomic_write_json
from src.paths import data_path
from src.schedule import TournamentSchedule
from src.tournament_state import calculate_group_standings, load_active_teams

# Mapping between website team names and project team names.
TEAM_NAME_MAP = {
    "United States": "USA",
    "Turkey": "Türkiye",
    "Turkiye": "Türkiye",
    "Czech Republic": "Czechia",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Cote d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",
}

def normalize_team_name(name, valid_teams):
    if not name:
        return None
    name = name.strip()
    
    # 1. Check manual mapping first.
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
        
    # 2. Check case-insensitive exact matches.
    for valid in valid_teams:
        if valid.lower() == name.lower():
            return valid
            
    return None

GROUP_PAGE_URLS = {
    group: f"https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_{group}"
    for group in "ABCDEFGHIJKL"
}

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"

ACTUAL_RESULT_REQUIRED_FIELDS = {
    "match_number",
    "team_a",
    "team_b",
    "score_a",
    "score_b",
    "date",
    "stage",
    "winner",
}
SCHEDULE_STAGES = {
    "group-stage",
    "round-of-32",
    "round-of-16",
    "quarter-finals",
    "semi-finals",
    "third-place",
    "final",
}
def normalized_schedule_stage(stage):
    """Return the compact stage name used by ``actual_results.json``."""
    return "group" if stage == "group-stage" else "knockout"


def _schedule_team_is_concrete(team_name):
    if not isinstance(team_name, str):
        return False
    return not (
        team_name.startswith("Group ")
        or team_name.startswith("Winner Match ")
        or team_name.startswith("Loser Match ")
    )


def validate_schedule(schedule_data, *, require_full=False):
    """Validate the schedule fields needed for result reconciliation."""
    if not isinstance(schedule_data, list) or not schedule_data:
        raise ValueError("schedule.json must contain a non-empty list")

    seen_numbers = set()
    for index, match in enumerate(schedule_data):
        if not isinstance(match, dict):
            raise ValueError(f"schedule entry {index} must be an object")
        for field in ("matchNumber", "date", "stage", "homeTeam", "awayTeam"):
            if field not in match:
                raise ValueError(f"schedule entry {index} is missing {field}")

        match_number = match["matchNumber"]
        if isinstance(match_number, bool) or not isinstance(match_number, int) or match_number <= 0:
            raise ValueError(f"invalid schedule matchNumber: {match_number!r}")
        if match_number in seen_numbers:
            raise ValueError(f"duplicate schedule matchNumber: {match_number}")
        seen_numbers.add(match_number)

        try:
            date.fromisoformat(match["date"])
        except (TypeError, ValueError):
            raise ValueError(f"invalid schedule date for match {match_number}") from None

        if match["stage"] not in SCHEDULE_STAGES:
            raise ValueError(f"invalid schedule stage for match {match_number}")
        if not isinstance(match["homeTeam"], str) or not isinstance(match["awayTeam"], str):
            raise ValueError(f"invalid schedule teams for match {match_number}")

    if require_full and seen_numbers != set(range(1, 105)):
        raise ValueError("schedule match numbers must cover 1 through 104")
    return {match["matchNumber"]: match for match in schedule_data}


def _result_pair(result):
    return frozenset((result.get("team_a"), result.get("team_b")))


def _schedule_pair(match):
    if not (
        _schedule_team_is_concrete(match.get("homeTeam"))
        and _schedule_team_is_concrete(match.get("awayTeam"))
    ):
        return None
    return frozenset((match["homeTeam"], match["awayTeam"]))


def _schedule_time_key(match):
    """Order same-day fixtures by kickoff rather than match number."""
    return (match.get("kickoffUtc") or match["date"], match["matchNumber"])


def _resolve_bracket_team(label, results_by_number):
    prefix_to_field = {
        "Winner Match ": "winner",
        "Loser Match ": "loser",
    }
    for prefix, field in prefix_to_field.items():
        if not isinstance(label, str) or not label.startswith(prefix):
            continue
        try:
            source_number = int(label[len(prefix):])
        except ValueError:
            return None
        source = results_by_number.get(source_number)
        if source is None or source.get("winner") not in {
            source.get("team_a"),
            source.get("team_b"),
        }:
            return None
        if field == "winner":
            return source["winner"]
        return (
            source["team_b"]
            if source["winner"] == source["team_a"]
            else source["team_a"]
        )
    return None


def _resolved_schedule_pair(match, results_by_number, known_match_pairs=None):
    if known_match_pairs and match.get("matchNumber") in known_match_pairs:
        return known_match_pairs[match["matchNumber"]]
    concrete_pair = _schedule_pair(match)
    if concrete_pair is not None:
        return concrete_pair
    home = _resolve_bracket_team(match.get("homeTeam"), results_by_number)
    away = _resolve_bracket_team(match.get("awayTeam"), results_by_number)
    if home and away:
        return frozenset((home, away))
    return None


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


def build_round_of_32_schedule_pairs(
    actual_results,
    schedule_data,
    groups,
    fifa_rankings,
    team_conduct_scores,
    third_place_annex,
):
    """Resolve M73-M88 participants from standings and FIFA Annex C.

    This is the authoritative bridge between concrete group results and the
    placeholder labels in ``schedule.json``.  Returning exact pairs here lets
    later rounds be resolved recursively through their Winner/Loser Match
    dependencies.
    """
    validate_schedule(schedule_data)
    if not any(result.get("stage") == "knockout" for result in actual_results):
        return {}
    if not all(
        isinstance(value, dict)
        for value in (groups, fifa_rankings, team_conduct_scores, third_place_annex)
    ):
        raise ValueError("group standings and Annex C data must be objects")

    standings = calculate_group_standings(
        groups,
        actual_results,
        fifa_rankings,
        team_conduct_scores,
        require_complete=True,
    )
    third_places = []
    for group_name, ranked_teams in standings.items():
        parts = group_name.split()
        if len(parts) != 2 or len(parts[1]) != 1:
            raise ValueError(f"invalid group name: {group_name!r}")
        group_letter = parts[1].upper()
        if len(ranked_teams) < 3:
            raise ValueError(f"incomplete standings for {group_name}")
        third_places.append(
            {
                "team": ranked_teams[2][0],
                "group": group_letter,
                "stats": ranked_teams[2][1],
            }
        )

    third_places.sort(
        key=lambda item: (
            -item["stats"]["pts"],
            -item["stats"]["gd"],
            -item["stats"]["gf"],
            -_conduct_value(item["team"], team_conduct_scores),
            _ranking_value(item["team"], fifa_rankings),
        )
    )
    advancing_thirds = third_places[:8]
    annex_key = "".join(sorted(item["group"] for item in advancing_thirds))
    annex_assignment = third_place_annex.get(annex_key)
    if not isinstance(annex_assignment, dict):
        raise ValueError(f"Annex C assignment is missing for groups: {annex_key}")

    team_by_code = team_codes_from_standings(standings)
    third_assignment = {}
    for match_number, group_letter in annex_assignment.items():
        try:
            third_assignment[int(match_number)] = team_by_code[f"{group_letter}3"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid Annex C third-place slot for M{match_number}"
            ) from exc

    tournament_schedule = TournamentSchedule.from_mappings(schedule_data)
    matchups = build_round_of_32_matchups(
        tournament_schedule,
        team_by_code=team_by_code,
        third_assignment=third_assignment,
    )
    resolved_pairs = {
        match_number: frozenset(teams)
        for match_number, teams in matchups.items()
    }

    if len(resolved_pairs) != 16:
        raise ValueError(
            f"resolved {len(resolved_pairs)} Round-of-32 matches; expected 16"
        )
    return resolved_pairs


def classify_result_stage(result, schedule_data):
    """Classify one result using the authoritative schedule, never the calendar alone."""
    exact_matches = [
        match
        for match in schedule_data
        if match.get("date") == result.get("date")
        and _schedule_pair(match) == _result_pair(result)
    ]
    if len(exact_matches) == 1:
        return normalized_schedule_stage(exact_matches[0]["stage"])
    if len(exact_matches) > 1:
        raise ValueError(
            f"ambiguous schedule entries for {result.get('team_a')} vs {result.get('team_b')}"
        )

    date_matches = [match for match in schedule_data if match.get("date") == result.get("date")]
    stages = {normalized_schedule_stage(match.get("stage")) for match in date_matches}
    if len(stages) == 1:
        return stages.pop()

    raise ValueError(
        f"could not classify {result.get('team_a')} vs {result.get('team_b')} "
        f"on {result.get('date')} from schedule.json"
    )


def build_unmapped_actual_results(parsed_matches, schedule_data):
    """Convert parsed Elo rows to result records without guessing tied winners."""
    validate_schedule(schedule_data)
    results = []
    for match in parsed_matches:
        date_str = (
            f"{int(match['year']):04d}-{int(match['month']):02d}-{int(match['day']):02d}"
        )
        result = {
            "team_a": match["team_a"],
            "team_b": match["team_b"],
            "score_a": match["score_a"],
            "score_b": match["score_b"],
            "date": date_str,
        }
        result["stage"] = classify_result_stage(result, schedule_data)

        if result["score_a"] > result["score_b"]:
            result["winner"] = result["team_a"]
        elif result["score_b"] > result["score_a"]:
            result["winner"] = result["team_b"]
        else:
            # A tied knockout result must be resolved from an authoritative
            # advancement flag (currently ESPN), not a later appearance.
            result["winner"] = None
        results.append(result)
    return results


def extract_espn_knockout_decisions(payload, query_date, valid_teams):
    """Extract authoritative advancement decisions from one ESPN response."""
    decisions = {}
    if not isinstance(payload, dict):
        return decisions

    events = payload.get("events", [])
    if not isinstance(events, list):
        return decisions
    for event in events:
        if not isinstance(event, dict):
            continue
        competitions = event.get("competitions", [])
        if not isinstance(competitions, list):
            continue
        for competition in competitions:
            if not isinstance(competition, dict):
                continue
            competitors = competition.get("competitors", [])
            if not isinstance(competitors, list) or len(competitors) != 2:
                continue

            teams = []
            flagged_winners = []
            for competitor in competitors:
                if not isinstance(competitor, dict):
                    break
                team_data = competitor.get("team", {})
                if not isinstance(team_data, dict):
                    break
                raw_name = (
                    team_data.get("displayName")
                    or team_data.get("name")
                    or team_data.get("location")
                )
                team_name = normalize_team_name(raw_name, valid_teams)
                if not team_name:
                    break
                teams.append(team_name)
                if competitor.get("advance") is True or competitor.get("winner") is True:
                    flagged_winners.append(team_name)

            if len(teams) == 2 and len(flagged_winners) == 1:
                decisions[(query_date, frozenset(teams))] = flagged_winners[0]

    return decisions


def apply_espn_knockout_decisions(results, decisions):
    """Apply ESPN decisions only to tied knockout matches."""
    resolved = []
    for original in results:
        result = dict(original)
        if (
            result.get("stage") == "knockout"
            and result.get("score_a") == result.get("score_b")
        ):
            winner = decisions.get((result.get("date"), _result_pair(result)))
            if winner in {result.get("team_a"), result.get("team_b")}:
                result["winner"] = winner
            else:
                result["winner"] = None
        resolved.append(result)
    return resolved


def assign_schedule_match_numbers(results, schedule_data, known_match_pairs=None):
    """Backfill ``match_number`` from concrete teams, bracket sources, or kickoff.

    Group fixtures have concrete team names and are matched exactly.  Knockout
    fixtures first resolve winner/loser dependencies.  Any still-unresolved
    same-day fixtures follow ``kickoffUtc`` because match-number order can differ
    from chronological feed order (notably M89/M90).  Existing match numbers are
    verified rather than silently trusted.  The function is pure and can be used
    by a migration or by the live refresh path.
    """
    schedule_by_number = validate_schedule(schedule_data)
    known_match_pairs = known_match_pairs or {}
    schedule_sorted = sorted(schedule_data, key=lambda match: match["matchNumber"])
    assigned = []
    used_numbers = set()

    for original in results:
        result = dict(original)
        match_number = result.get("match_number")
        if match_number is not None:
            schedule_match = schedule_by_number.get(match_number)
            if schedule_match is None:
                raise ValueError(f"unknown match_number: {match_number}")
            if schedule_match["date"] != result.get("date"):
                raise ValueError(f"date does not match schedule for match {match_number}")
            if normalized_schedule_stage(schedule_match["stage"]) != result.get("stage"):
                raise ValueError(f"stage does not match schedule for match {match_number}")
            expected_pair = _resolved_schedule_pair(
                schedule_match,
                {},
                known_match_pairs,
            )
            if expected_pair is not None and expected_pair != _result_pair(result):
                raise ValueError(f"teams do not match schedule for match {match_number}")
            if match_number in used_numbers:
                raise ValueError(f"duplicate result match_number: {match_number}")
            used_numbers.add(match_number)
        assigned.append(result)

    # Concrete group fixtures can be mapped independent of source row order.
    for result in assigned:
        if result.get("match_number") is not None:
            continue
        candidates = [
            match
            for match in schedule_sorted
            if match["matchNumber"] not in used_numbers
            and match["date"] == result.get("date")
            and normalized_schedule_stage(match["stage"]) == result.get("stage")
            and _resolved_schedule_pair(match, {}, known_match_pairs) == _result_pair(result)
        ]
        if len(candidates) > 1:
            raise ValueError(
                f"ambiguous match number for {result.get('team_a')} vs {result.get('team_b')}"
            )
        if len(candidates) == 1:
            match_number = candidates[0]["matchNumber"]
            result["match_number"] = match_number
            used_numbers.add(match_number)

    # Resolve later-round Winner/Loser Match placeholders whenever earlier
    # results make the participant pair unambiguous.
    while True:
        progress = False
        results_by_number = {
            result["match_number"]: result
            for result in assigned
            if result.get("match_number") is not None
        }
        for result in assigned:
            if result.get("match_number") is not None:
                continue
            candidates = [
                match
                for match in schedule_sorted
                if match["matchNumber"] not in used_numbers
                and match["date"] == result.get("date")
                and normalized_schedule_stage(match["stage"]) == result.get("stage")
                and _resolved_schedule_pair(
                    match,
                    results_by_number,
                    known_match_pairs,
                )
                == _result_pair(result)
            ]
            if len(candidates) > 1:
                raise ValueError(
                    f"ambiguous bracket match for {result.get('team_a')} "
                    f"vs {result.get('team_b')}"
                )
            if len(candidates) == 1:
                match_number = candidates[0]["matchNumber"]
                result["match_number"] = match_number
                used_numbers.add(match_number)
                progress = True
        if not progress:
            break

    # Round-of-32 participant labels cannot be resolved without standings data,
    # and an external feed can be partial.  Map remaining rows by kickoff time,
    # never by the numeric match id.
    pending_groups = {}
    for result in assigned:
        if result.get("match_number") is None:
            key = (result.get("date"), result.get("stage"))
            pending_groups.setdefault(key, []).append(result)

    for (date_str, stage), pending in pending_groups.items():
        candidates = sorted([
            match
            for match in schedule_sorted
            if match["matchNumber"] not in used_numbers
            and match["date"] == date_str
            and normalized_schedule_stage(match["stage"]) == stage
        ], key=_schedule_time_key)
        if len(pending) > len(candidates):
            raise ValueError(f"more results than scheduled matches on {date_str}")
        if stage == "group":
            raise ValueError(f"group result teams did not match schedule on {date_str}")

        for result, schedule_match in zip(pending, candidates):
            match_number = schedule_match["matchNumber"]
            result["match_number"] = match_number
            used_numbers.add(match_number)

    unresolved = [result for result in assigned if result.get("match_number") is None]
    if unresolved:
        first = unresolved[0]
        raise ValueError(
            f"could not assign match_number for {first.get('team_a')} vs {first.get('team_b')}"
        )

    return sorted(assigned, key=lambda result: result["match_number"])


def validate_actual_results(
    results,
    schedule_data,
    *,
    known_match_pairs=None,
    require_knockout_winner=True,
    require_contiguous=True,
):
    """Validate the persisted result schema and its schedule invariants."""
    if not isinstance(results, list):
        raise ValueError("actual_results.json must contain a list")
    schedule_by_number = validate_schedule(schedule_data)
    known_match_pairs = known_match_pairs or {}
    numbers = []

    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"result {index} must be an object")
        missing = ACTUAL_RESULT_REQUIRED_FIELDS - set(result)
        if missing:
            raise ValueError(f"result {index} is missing: {', '.join(sorted(missing))}")

        match_number = result["match_number"]
        if isinstance(match_number, bool) or not isinstance(match_number, int):
            raise ValueError(f"result {index} has invalid match_number")
        if match_number in numbers:
            raise ValueError(f"duplicate result match_number: {match_number}")
        numbers.append(match_number)

        schedule_match = schedule_by_number.get(match_number)
        if schedule_match is None:
            raise ValueError(f"result {index} references unknown match_number {match_number}")
        if result["date"] != schedule_match["date"]:
            raise ValueError(f"result date differs from schedule for match {match_number}")
        expected_stage = normalized_schedule_stage(schedule_match["stage"])
        if result["stage"] != expected_stage:
            raise ValueError(f"result stage differs from schedule for match {match_number}")

        team_a = result["team_a"]
        team_b = result["team_b"]
        if not isinstance(team_a, str) or not team_a or not isinstance(team_b, str) or not team_b:
            raise ValueError(f"result {index} has invalid team names")
        if team_a == team_b:
            raise ValueError(f"result {index} contains the same team twice")
        expected_pair = _resolved_schedule_pair(
            schedule_match,
            {},
            known_match_pairs,
        )
        if expected_pair is not None and expected_pair != _result_pair(result):
            raise ValueError(f"result teams differ from schedule for match {match_number}")

        score_a = result["score_a"]
        score_b = result["score_b"]
        if (
            isinstance(score_a, bool)
            or not isinstance(score_a, int)
            or score_a < 0
            or isinstance(score_b, bool)
            or not isinstance(score_b, int)
            or score_b < 0
        ):
            raise ValueError(f"result {index} has invalid scores")

        try:
            date.fromisoformat(result["date"])
        except (TypeError, ValueError):
            raise ValueError(f"result {index} has an invalid date") from None

        winner = result["winner"]
        score_winner = team_a if score_a > score_b else team_b if score_b > score_a else None
        if score_winner is not None and winner != score_winner:
            raise ValueError(f"result {index} winner conflicts with the score")
        if score_winner is None:
            if result["stage"] == "group" and winner is not None:
                raise ValueError(f"tied group result {index} cannot have a winner")
            if result["stage"] == "knockout":
                if winner is not None and winner not in {team_a, team_b}:
                    raise ValueError(f"result {index} has an invalid knockout winner")
                if require_knockout_winner and winner is None:
                    raise ValueError(f"tied knockout result {index} has no ESPN decision")

    results_by_number = {result["match_number"]: result for result in results}
    for result in results:
        schedule_match = schedule_by_number[result["match_number"]]
        expected_pair = _resolved_schedule_pair(
            schedule_match,
            results_by_number,
            known_match_pairs,
        )
        if expected_pair is not None and expected_pair != _result_pair(result):
            raise ValueError(
                f"result teams violate bracket dependencies for match "
                f"{result['match_number']}"
            )

    if len(set(numbers)) != len(numbers):
        raise ValueError("result match numbers must be unique")
    if require_contiguous and numbers:
        expected = list(range(1, max(numbers) + 1))
        if numbers != expected:
            raise ValueError(
                "completed match numbers must be ordered and increase contiguously from 1"
            )
    return True


def merge_result_snapshots(
    existing_results,
    incoming_results,
    schedule_data,
    known_match_pairs=None,
):
    """Merge a possibly partial feed without dropping previously locked matches."""
    validate_actual_results(
        existing_results,
        schedule_data,
        known_match_pairs=known_match_pairs,
        require_knockout_winner=False,
        require_contiguous=True,
    )
    validate_actual_results(
        incoming_results,
        schedule_data,
        known_match_pairs=known_match_pairs,
        require_knockout_winner=False,
        require_contiguous=False,
    )

    merged = {result["match_number"]: dict(result) for result in existing_results}
    stable_fields = ("team_a", "team_b", "score_a", "score_b", "date", "stage")

    for incoming in incoming_results:
        match_number = incoming["match_number"]
        previous = merged.get(match_number)
        if previous is not None:
            conflicts = [field for field in stable_fields if previous.get(field) != incoming.get(field)]
            if conflicts:
                raise ValueError(
                    f"incoming match {match_number} conflicts with locked fields: "
                    f"{', '.join(conflicts)}"
                )
            # A new ESPN flag may correct a winner inferred by older versions.
            if incoming.get("winner") is not None:
                previous["winner"] = incoming["winner"]
        else:
            merged[match_number] = dict(incoming)

    combined = [merged[number] for number in sorted(merged)]
    validate_actual_results(
        combined,
        schedule_data,
        known_match_pairs=known_match_pairs,
        require_knockout_winner=True,
        require_contiguous=True,
    )

    if len(combined) < len(existing_results):
        raise ValueError("result refresh would reduce the number of completed matches")
    return combined


def backfill_actual_results_match_numbers(*, write=False):
    """Validate and optionally persist match numbers for the local result file."""
    input_names = (
        "schedule.json",
        "actual_results.json",
        "groups.json",
        "fifa_rankings.json",
        "team_conduct_scores.json",
        "third_place_annex_c.json",
    )
    loaded = {}
    for input_name in input_names:
        with data_path(input_name).open(encoding="utf-8") as handle:
            loaded[input_name] = json.load(handle)
    validate_schedule(loaded["schedule.json"], require_full=True)

    known_match_pairs = build_round_of_32_schedule_pairs(
        loaded["actual_results.json"],
        loaded["schedule.json"],
        loaded["groups.json"],
        loaded["fifa_rankings.json"],
        loaded["team_conduct_scores.json"],
        loaded["third_place_annex_c.json"],
    )
    backfilled = assign_schedule_match_numbers(
        loaded["actual_results.json"],
        loaded["schedule.json"],
        known_match_pairs,
    )
    validate_actual_results(
        backfilled,
        loaded["schedule.json"],
        known_match_pairs=known_match_pairs,
    )
    if write:
        atomic_write_json(data_path("actual_results.json"), backfilled)
    return backfilled

def get_match_teams_from_fevent(fevent, valid_teams):
    home_el = fevent.select_one(".fhome [itemprop='name']")
    away_el = fevent.select_one(".faway [itemprop='name']")
    home = normalize_team_name(home_el.get_text(" ", strip=True) if home_el else None, valid_teams)
    away = normalize_team_name(away_el.get_text(" ", strip=True) if away_el else None, valid_teams)
    return home, away

def has_completed_score(fevent):
    score_el = fevent.select_one(".fscore")
    if not score_el:
        return False
    score = score_el.get_text(" ", strip=True)
    return any(ch.isdigit() for ch in score)

def iter_top_level_cells(table):
    tbody = table.find("tbody", recursive=False)
    rows = tbody.find_all("tr", recursive=False) if tbody else table.find_all("tr", recursive=False)
    if not rows:
        return []
    return rows[0].find_all("td", recursive=False)

def find_lineup_table_after(fevent):
    for table in fevent.find_all_next("table"):
        if "fevent" in (table.get("class") or []):
            return None
        cells = [cell for cell in iter_top_level_cells(table) if cell.get_text(" ", strip=True)]
        if len(cells) < 2:
            continue
        if any("card" in img.get("alt", "").lower() for img in table.find_all("img")):
            return table
    return None

def conduct_score_for_player_row(row):
    alts = [
        img.get("alt", "").strip().lower()
        for img in row.find_all("img")
    ]
    has_yellow_red = any("yellow-red card" in alt for alt in alts)
    has_red = any(alt == "red card" for alt in alts)
    has_yellow = any(alt == "yellow card" for alt in alts)

    if has_yellow_red:
        return -3
    if has_red and has_yellow:
        return -5
    if has_red:
        return -4
    if has_yellow:
        return -1
    return 0

def conduct_score_for_team_cell(cell):
    score = 0
    for row in cell.find_all("tr"):
        score += conduct_score_for_player_row(row)
    return score

def fetch_espn_knockout_decisions(dates, valid_teams, headers):
    """Fetch knockout advancement decisions from the ESPN scoreboard, including penalties."""
    decisions = {}
    for date_str in sorted(dates):
        espn_date = date_str.replace("-", "")
        try:
            response = httpx.get(
                ESPN_SCOREBOARD_URL,
                params={"dates": espn_date},
                headers=headers,
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"[Match results] Failed to query ESPN scoreboard ({date_str}): {e}")
            continue

        decisions.update(
            extract_espn_knockout_decisions(data, date_str, valid_teams)
        )

    return decisions

def update_team_conduct_scores(valid_teams, headers):
    from bs4 import BeautifulSoup

    conduct_path = data_path("team_conduct_scores.json")
    conduct_scores = {team: 0 for team in valid_teams}
    parsed_matches = 0
    matches_with_cards = 0

    print("[Conduct] Recalculating team conduct scores from Wikipedia match records...")
    for group, url in GROUP_PAGE_URLS.items():
        try:
            response = httpx.get(url, headers=headers, timeout=15.0)
            response.raise_for_status()
        except Exception as e:
            print(f"[Conduct] Failed to request Group {group} page: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        for fevent in soup.find_all("table", class_="fevent"):
            if not has_completed_score(fevent):
                continue

            home, away = get_match_teams_from_fevent(fevent, valid_teams)
            if not home or not away:
                continue

            parsed_matches += 1
            lineup_table = find_lineup_table_after(fevent)
            if not lineup_table:
                continue

            cells = [cell for cell in iter_top_level_cells(lineup_table) if cell.get_text(" ", strip=True)]
            if len(cells) < 2:
                continue

            home_score = conduct_score_for_team_cell(cells[0])
            away_score = conduct_score_for_team_cell(cells[-1])
            if home_score or away_score:
                matches_with_cards += 1
            conduct_scores[home] += home_score
            conduct_scores[away] += away_score

    if parsed_matches != 72:
        raise RuntimeError(
            f"Conduct refresh parsed {parsed_matches} of 72 group matches; "
            "the previous snapshot was preserved"
        )

    atomic_write_json(conduct_path, dict(sorted(conduct_scores.items())))

    print(
        f"[Conduct] Update complete: checked {parsed_matches} matches, "
        f"applied card records from {matches_with_cards} matches -> {conduct_path}"
    )

def fetch_live_world_cup_data():
    elo_ratings_path = data_path("elo_ratings.json")
    actual_results_path = data_path("actual_results.json")
    
    if not os.path.exists(elo_ratings_path):
        raise FileNotFoundError(f"{elo_ratings_path} was not found")
        
    # Load the 48 participating teams.
    with open(elo_ratings_path, "r", encoding="utf-8") as f:
        local_ratings = json.load(f)
        valid_teams = set(local_ratings.keys())
    active_teams = load_active_teams(
        groups_path=data_path("groups.json"),
        actual_results_path=actual_results_path,
        ratings_path=elo_ratings_path,
    )
    if not active_teams:
        active_teams = set(valid_teams)
        
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # 1. Download team-name/code mapping file.
    print("[ELO] Downloading team-name/code mapping from eloratings.net...")
    try:
        r_teams = httpx.get("https://www.eloratings.net/en.teams.tsv", headers=headers, timeout=15.0)
        r_teams.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to request team mapping file: {e}") from e
        
    code_to_name = {}
    for line in r_teams.text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            code_to_name[parts[0]] = parts[1]
            
    # 2. Download and update Elo ratings.
    print("[ELO] Downloading live world Elo ratings from eloratings.net...")
    try:
        r_world = httpx.get("https://www.eloratings.net/World.tsv", headers=headers, timeout=15.0)
        r_world.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to request Elo ratings file: {e}") from e
        
    updated_ratings = dict(local_ratings)
    updated_active_teams = set()
    
    for line in r_world.text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 4:
            country_code = parts[2]
            try:
                elo_val = float(parts[3])
            except ValueError:
                continue
            if not math.isfinite(elo_val) or elo_val <= 0:
                continue
                
            raw_name = code_to_name.get(country_code)
            normalized_name = normalize_team_name(raw_name, valid_teams)
            
            if normalized_name in active_teams:
                updated_ratings[normalized_name] = elo_val
                updated_active_teams.add(normalized_name)

    missing_ratings = sorted(active_teams - updated_active_teams)
    if missing_ratings:
        raise RuntimeError(
            "Elo feed is missing active teams; preserving the previous snapshot: "
            + ", ".join(missing_ratings)
        )
    ratings_updated_count = len(updated_active_teams)
                
    # 3. Download and validate 2026 match results before changing either local
    # data snapshot. This avoids a half-refreshed Elo/results state.
    print("[Match results] Downloading 2026 match-result data from eloratings.net...")
    try:
        r_results = httpx.get("https://www.eloratings.net/2026_results.tsv", headers=headers, timeout=15.0)
        r_results.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to request match results file: {e}") from e
        
    parsed_matches = []
    
    for line in r_results.text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 8:
            tournament_code = parts[7]
            # Keep only World Cup final-tournament matches.
            if tournament_code != "WC":
                continue
                
            home_code = parts[3]
            away_code = parts[4]
            
            try:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                score_a = int(parts[5])
                score_b = int(parts[6])
            except ValueError:
                # Non-numeric scores are treated as unplayed matches.
                continue
                
            home_name = code_to_name.get(home_code)
            away_name = code_to_name.get(away_code)
            
            home_team = normalize_team_name(home_name, valid_teams)
            away_team = normalize_team_name(away_name, valid_teams)
            
            if home_team and away_team and home_team != away_team:
                parsed_matches.append({
                    "year": year,
                    "month": month,
                    "day": day,
                    "team_a": home_team,
                    "team_b": away_team,
                    "score_a": score_a,
                    "score_b": score_b
                })
                
    # Reconcile the result feed against the authoritative 104-match schedule.
    schedule_path = data_path("schedule.json")
    try:
        with open(schedule_path, "r", encoding="utf-8") as f:
            schedule_data = json.load(f)
        validate_schedule(schedule_data, require_full=True)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(
            f"Invalid schedule; preserving local results: {e}"
        ) from e

    try:
        incoming_results = build_unmapped_actual_results(parsed_matches, schedule_data)
    except (KeyError, TypeError, ValueError) as e:
        raise RuntimeError(
            f"Invalid result feed; preserving local results: {e}"
        ) from e

    tied_knockout_dates = {
        result["date"]
        for result in incoming_results
        if result["stage"] == "knockout"
        and result["score_a"] == result["score_b"]
    }
    espn_decisions = fetch_espn_knockout_decisions(
        tied_knockout_dates,
        valid_teams,
        headers,
    )
    incoming_results = apply_espn_knockout_decisions(
        incoming_results,
        espn_decisions,
    )
    applied_decisions = sum(
        1
        for result in incoming_results
        if result["stage"] == "knockout"
        and result["score_a"] == result["score_b"]
        and result["winner"] is not None
    )
    if applied_decisions:
        print(
            f"[Match results] Applied {applied_decisions} ESPN "
            "penalty/advancement decisions."
        )

    try:
        known_match_pairs = {}
        if any(result["stage"] == "knockout" for result in incoming_results):
            standings_inputs = {}
            for input_name in (
                "groups.json",
                "fifa_rankings.json",
                "team_conduct_scores.json",
                "third_place_annex_c.json",
            ):
                with data_path(input_name).open(encoding="utf-8") as f:
                    standings_inputs[input_name] = json.load(f)
            known_match_pairs = build_round_of_32_schedule_pairs(
                incoming_results,
                schedule_data,
                standings_inputs["groups.json"],
                standings_inputs["fifa_rankings.json"],
                standings_inputs["team_conduct_scores.json"],
                standings_inputs["third_place_annex_c.json"],
            )

        incoming_results = assign_schedule_match_numbers(
            incoming_results,
            schedule_data,
            known_match_pairs,
        )
        validate_actual_results(
            incoming_results,
            schedule_data,
            known_match_pairs=known_match_pairs,
            require_knockout_winner=False,
            require_contiguous=False,
        )

        if os.path.exists(actual_results_path):
            with open(actual_results_path, "r", encoding="utf-8") as f:
                local_results = json.load(f)
            # This also provides the one-time backfill path for pre-migration files.
            local_results = assign_schedule_match_numbers(
                local_results,
                schedule_data,
                known_match_pairs,
            )
            actual_results = merge_result_snapshots(
                local_results,
                incoming_results,
                schedule_data,
                known_match_pairs,
            )
        else:
            actual_results = incoming_results
            validate_actual_results(
                actual_results,
                schedule_data,
                known_match_pairs=known_match_pairs,
            )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        raise RuntimeError(
            f"Result refresh failed validation; preserving local data: {e}"
        ) from e

    atomic_write_json(elo_ratings_path, updated_ratings)
    print(
        f"[ELO] Elo ratings update complete: applied ratings for "
        f"{ratings_updated_count} active teams ({len(active_teams)} considered)"
    )
    try:
        with open(actual_results_path, "r", encoding="utf-8") as f:
            current_file_results = json.load(f)
    except (OSError, json.JSONDecodeError):
        current_file_results = None

    if current_file_results == actual_results:
        print("[Match results] No new completed matches were added.")
    else:
        atomic_write_json(actual_results_path, actual_results)
        print(
            f"[Match results] Update complete: locked {len(actual_results)} "
            f"completed matches in {actual_results_path}."
        )
        for result in actual_results:
            winner_str = (
                f" (winner: {result['winner']})" if result["winner"] else ""
            )
            print(
                f"   - [M{result['match_number']} {result['stage'].upper()}] "
                f"{result['team_a']} {result['score_a']} : {result['score_b']} "
                f"{result['team_b']}{winner_str}"
            )
        
    # 4. Sync live suspension data from Wikipedia.
    suspensions_updated = True
    try:
        import fetch_suspensions
        print("\n[ELO & suspensions] Syncing live suspension data after Elo/result updates...")
        fetch_suspensions.main()
    except Exception as e:
        suspensions_updated = False
        print(f"\n[Warning] Failed to sync live suspension data. Elo/results were still updated: {e}")

    # 5. Conduct scores are no longer updated after the group stage.
    #    The function remains available for standings recalculation or verification.
    print("[Conduct] Skipping team_conduct_scores.json update because the group stage is complete.")
    return {
        "elo_results_updated": True,
        "suspensions_updated": suspensions_updated,
    }

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Refresh World Cup data or backfill schedule match numbers."
    )
    parser.add_argument(
        "--backfill-match-numbers",
        action="store_true",
        help="Backfill and atomically save match_number in actual_results.json without network access.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the backfill without writing actual_results.json.",
    )
    args = parser.parse_args(argv)

    if args.check_only and not args.backfill_match_numbers:
        parser.error("--check-only requires --backfill-match-numbers")
    if args.backfill_match_numbers:
        results = backfill_actual_results_match_numbers(write=not args.check_only)
        action = "validated" if args.check_only else "updated"
        print(
            f"[Match results] {action} {len(results)} matches with contiguous "
            "schedule match numbers."
        )
        return
    fetch_live_world_cup_data()


if __name__ == "__main__":
    main()
