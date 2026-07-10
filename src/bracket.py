"""Schedule-driven 2026 World Cup bracket resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from src.schedule import MatchInfo, TournamentSchedule


SIMULATED_KNOCKOUT_STAGES: tuple[tuple[str, str], ...] = (
    ("round-of-32", "Round of 32"),
    ("round-of-16", "Round of 16"),
    ("quarter-finals", "Quarter-finals"),
    ("semi-finals", "Semi-finals"),
    ("third-place", "Third-place"),
    ("final", "Final"),
)


_WINNER_RE = re.compile(r"^Winner Match (\d+)$", re.IGNORECASE)
_LOSER_RE = re.compile(r"^Loser Match (\d+)$", re.IGNORECASE)
_GROUP_WINNER_RE = re.compile(r"^Group ([A-L]) winners$", re.IGNORECASE)
_GROUP_RUNNER_UP_RE = re.compile(
    r"^Group ([A-L]) runners-up$", re.IGNORECASE
)
_THIRD_PLACE_RE = re.compile(
    r"^Group [A-L](?:/[A-L])+ third place$", re.IGNORECASE
)


@dataclass(frozen=True)
class BracketSource:
    """One participant source encoded in the official schedule."""

    kind: str
    value: str | int


def parse_bracket_source(label: str) -> BracketSource:
    """Parse an official schedule participant label."""

    if match := _WINNER_RE.fullmatch(label):
        return BracketSource("winner", int(match.group(1)))
    if match := _LOSER_RE.fullmatch(label):
        return BracketSource("loser", int(match.group(1)))
    if match := _GROUP_WINNER_RE.fullmatch(label):
        return BracketSource("group", f"{match.group(1).upper()}1")
    if match := _GROUP_RUNNER_UP_RE.fullmatch(label):
        return BracketSource("group", f"{match.group(1).upper()}2")
    if _THIRD_PLACE_RE.fullmatch(label):
        return BracketSource("third-place", label)
    raise ValueError(f"Unsupported bracket source label: {label!r}")


def sources_for_match(
    schedule: TournamentSchedule, match_number: int
) -> tuple[BracketSource, BracketSource]:
    match = schedule.match(match_number)
    return (
        parse_bracket_source(match.home_team),
        parse_bracket_source(match.away_team),
    )


def winner_sources_for_match(
    schedule: TournamentSchedule, match_number: int
) -> tuple[int, int]:
    """Return the two earlier match numbers feeding a winner-v-winner match."""

    sources = sources_for_match(schedule, match_number)
    if any(source.kind != "winner" for source in sources):
        raise ValueError(f"M{match_number} is not fed by two match winners")
    return int(sources[0].value), int(sources[1].value)


def _resolve_source(
    source: BracketSource,
    *,
    match_number: int,
    team_by_code: Mapping[str, str],
    third_assignment: Mapping[int, str],
    winners: Mapping[int, str],
    losers: Mapping[int, str] | None = None,
) -> str:
    if source.kind == "group":
        code = str(source.value)
        try:
            return team_by_code[code]
        except KeyError as exc:
            raise KeyError(f"No team is assigned to bracket slot {code}") from exc
    if source.kind == "third-place":
        try:
            return third_assignment[match_number]
        except KeyError as exc:
            raise KeyError(
                f"No third-place team is assigned to M{match_number}"
            ) from exc
    if source.kind == "winner":
        source_match = int(source.value)
        try:
            return winners[source_match]
        except KeyError as exc:
            raise KeyError(
                f"Winner of M{source_match} is unavailable for M{match_number}"
            ) from exc
    if source.kind == "loser":
        source_match = int(source.value)
        if losers is None:
            raise KeyError(
                f"Loser of M{source_match} is unavailable for M{match_number}"
            )
        try:
            return losers[source_match]
        except KeyError as exc:
            raise KeyError(
                f"Loser of M{source_match} is unavailable for M{match_number}"
            ) from exc
    raise ValueError(f"Unsupported bracket source kind: {source.kind!r}")


def resolve_match_teams(
    match: MatchInfo,
    *,
    team_by_code: Mapping[str, str],
    third_assignment: Mapping[int, str],
    winners: Mapping[int, str],
    losers: Mapping[int, str] | None = None,
) -> tuple[str, str]:
    """Resolve both teams for a scheduled knockout match."""

    source_a = parse_bracket_source(match.home_team)
    source_b = parse_bracket_source(match.away_team)
    common = {
        "match_number": match.match_number,
        "team_by_code": team_by_code,
        "third_assignment": third_assignment,
        "winners": winners,
        "losers": losers,
    }
    return (
        _resolve_source(source_a, **common),
        _resolve_source(source_b, **common),
    )


def resolve_match_team_labels(
    match: MatchInfo,
    *,
    team_by_code: Mapping[str, str],
    third_assignment: Mapping[int, str],
    winners: Mapping[int, str],
    losers: Mapping[int, str] | None = None,
) -> tuple[str, str]:
    """Resolve known participants and label unresolved result sources."""

    labels = []
    for source in (
        parse_bracket_source(match.home_team),
        parse_bracket_source(match.away_team),
    ):
        try:
            labels.append(
                _resolve_source(
                    source,
                    match_number=match.match_number,
                    team_by_code=team_by_code,
                    third_assignment=third_assignment,
                    winners=winners,
                    losers=losers,
                )
            )
        except KeyError:
            if source.kind == "winner":
                labels.append(f"Winner M{source.value}")
            elif source.kind == "loser":
                labels.append(f"Loser M{source.value}")
            else:
                raise
    return labels[0], labels[1]


def team_codes_from_standings(group_standings) -> dict[str, str]:
    """Convert standings rows into bracket codes such as ``A1`` and ``B2``."""

    team_by_code = {}
    for group_name, rows in group_standings.items():
        group_letter = str(group_name).split()[-1].upper()
        if len(rows) < 3:
            raise ValueError(f"Standings for {group_name} have fewer than three teams")
        for position, row in enumerate(rows[:3], start=1):
            team = row[0] if isinstance(row, (list, tuple)) else row
            team_by_code[f"{group_letter}{position}"] = team
    return team_by_code


def build_round_of_32_matchups(
    schedule: TournamentSchedule,
    *,
    team_by_code: Mapping[str, str],
    third_assignment: Mapping[int, str],
) -> dict[int, tuple[str, str]]:
    """Resolve the complete R32 matchup map from standings and Annex C."""

    return {
        match.match_number: resolve_match_teams(
            match,
            team_by_code=team_by_code,
            third_assignment=third_assignment,
            winners={},
        )
        for match in schedule.stage_matches("round-of-32")
    }


def validate_knockout_bracket(schedule: TournamentSchedule) -> None:
    """Validate all 32-team bracket sources in schedule order."""

    expected_by_stage = {
        "round-of-32": set(range(73, 89)),
        "round-of-16": set(range(89, 97)),
        "quarter-finals": set(range(97, 101)),
        "semi-finals": {101, 102},
        "third-place": {103},
        "final": {104},
    }
    for stage, expected_numbers in expected_by_stage.items():
        matches = schedule.stage_matches(stage)
        actual_numbers = {match.match_number for match in matches}
        if actual_numbers != expected_numbers:
            raise ValueError(
                f"Schedule stage {stage!r} has match numbers "
                f"{sorted(actual_numbers)}; expected {sorted(expected_numbers)}"
            )

    for match in schedule.knockout_matches():
        for source in sources_for_match(schedule, match.match_number):
            if source.kind in {"winner", "loser"}:
                source_match = int(source.value)
                if source_match >= match.match_number:
                    raise ValueError(
                        f"M{match.match_number} depends on non-earlier M{source_match}"
                    )
