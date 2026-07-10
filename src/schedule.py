"""Tournament schedule access and venue-region metadata.

The checked-in ``data/schedule.json`` file is the source of truth for match
numbers, dates, stages, bracket source labels, and host cities.  This module
keeps parsing and validation in one place so the simulator and dashboard do
not maintain competing copies of the tournament calendar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEDULE_PATH = PROJECT_ROOT / "data" / "schedule.json"


# Five coarse travel regions used by the existing fatigue model.  The values
# are ordered west-to-east so distance between buckets remains meaningful.
CITY_REGIONS: dict[str, int] = {
    # Pacific coast
    "vancouver": 1,
    "seattle": 1,
    "san-francisco": 1,
    "los-angeles": 1,
    # Mexico
    "guadalajara": 2,
    "mexico-city": 2,
    "monterrey": 2,
    # Central United States
    "dallas": 3,
    "houston": 3,
    "kansas-city": 3,
    # Southeast
    "atlanta": 4,
    "miami": 4,
    # Northeast
    "toronto": 5,
    "boston": 5,
    "new-york": 5,
    "philadelphia": 5,
}


def date_to_day_num(value: str) -> int:
    """Convert an ISO date to the legacy tournament-day number.

    June 1 is day 1 and July 1 is day 31.  Keeping this representation
    preserves the public values previously emitted by the simulator while
    using real calendar arithmetic instead of a hard-coded month branch.
    """

    parsed = date.fromisoformat(value)
    origin = date(parsed.year, 5, 31)
    return (parsed - origin).days


def region_for_city(host_city: str) -> int:
    """Return the travel region for a schedule host-city slug."""

    try:
        return CITY_REGIONS[host_city]
    except KeyError as exc:
        raise ValueError(f"Unknown World Cup host city: {host_city!r}") from exc


@dataclass(frozen=True)
class MatchInfo:
    """Normalized representation of one entry in ``schedule.json``."""

    match_number: int
    date: str
    stage: str
    home_team: str
    away_team: str
    host_city: str
    stadium: str = ""
    group: str | None = None
    kickoff_utc: str | None = None

    @property
    def day_number(self) -> int:
        return date_to_day_num(self.date)

    @property
    def region(self) -> int:
        return region_for_city(self.host_city)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "MatchInfo":
        required = (
            "matchNumber",
            "date",
            "stage",
            "homeTeam",
            "awayTeam",
            "hostCity",
        )
        missing = [key for key in required if raw.get(key) in (None, "")]
        if missing:
            raise ValueError(
                "Schedule entry is missing required fields: " + ", ".join(missing)
            )

        try:
            match_number = int(raw["matchNumber"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid schedule match number: {raw.get('matchNumber')!r}"
            ) from exc

        match = cls(
            match_number=match_number,
            date=str(raw["date"]),
            stage=str(raw["stage"]),
            home_team=str(raw["homeTeam"]),
            away_team=str(raw["awayTeam"]),
            host_city=str(raw["hostCity"]),
            stadium=str(raw.get("stadium") or ""),
            group=str(raw["group"]) if raw.get("group") else None,
            kickoff_utc=str(raw["kickoffUtc"]) if raw.get("kickoffUtc") else None,
        )

        # Validate derived fields at load time so bad schedule data cannot
        # silently fall back to an arbitrary date or travel region.
        _ = match.day_number
        _ = match.region
        return match


class TournamentSchedule:
    """Validated, indexed tournament schedule."""

    def __init__(self, matches: Iterable[MatchInfo]):
        normalized = sorted(matches, key=lambda match: match.match_number)
        by_number: dict[int, MatchInfo] = {}
        for match in normalized:
            if match.match_number in by_number:
                raise ValueError(
                    f"Duplicate schedule match number: {match.match_number}"
                )
            by_number[match.match_number] = match

        if not normalized:
            raise ValueError("Schedule must contain at least one match")

        self.matches: tuple[MatchInfo, ...] = tuple(normalized)
        self._by_number = by_number

    @classmethod
    def from_file(
        cls, path: str | Path = DEFAULT_SCHEDULE_PATH
    ) -> "TournamentSchedule":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, list):
            raise ValueError("Schedule JSON must contain a list of matches")
        return cls(MatchInfo.from_mapping(item) for item in raw)

    @classmethod
    def from_mappings(
        cls, entries: Sequence[Mapping[str, object]]
    ) -> "TournamentSchedule":
        return cls(MatchInfo.from_mapping(item) for item in entries)

    def match(self, match_number: int) -> MatchInfo:
        try:
            return self._by_number[int(match_number)]
        except (KeyError, TypeError, ValueError) as exc:
            raise KeyError(f"Schedule has no match M{match_number}") from exc

    def stage_matches(self, stage: str) -> list[MatchInfo]:
        return [match for match in self.matches if match.stage == stage]

    def group_matches(self, group_letter: str) -> list[MatchInfo]:
        return [
            match
            for match in self.matches
            if match.stage == "group-stage" and match.group == group_letter
        ]

    def last_group_match(self, team: str) -> MatchInfo:
        candidates = [
            match
            for match in self.matches
            if match.stage == "group-stage"
            and team in {match.home_team, match.away_team}
        ]
        if not candidates:
            raise KeyError(f"Schedule has no group-stage match for {team}")
        return max(
            candidates,
            key=lambda match: (match.day_number, match.match_number),
        )

    def knockout_matches(self) -> list[MatchInfo]:
        return [
            match for match in self.matches if match.stage != "group-stage"
        ]


def load_schedule(
    path: str | Path = DEFAULT_SCHEDULE_PATH,
) -> TournamentSchedule:
    """Load a validated tournament schedule from disk."""

    return TournamentSchedule.from_file(path)

