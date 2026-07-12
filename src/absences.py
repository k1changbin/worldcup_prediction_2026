import json
import os
from copy import deepcopy

from src.io_utils import atomic_write_json


RESERVED_ABSENCE_KEYS = {"injuries", "suspensions"}
SUSPENSION_REASONS = {"yellow_cards", "red_card", "disciplinary"}


def load_json(path, default_val=None):
    if default_val is None:
        default_val = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default_val
    return default_val


def save_json(path, data):
    atomic_write_json(path, data)


def _absence_name(item):
    if isinstance(item, dict):
        return item.get("name")
    return str(item) if item is not None else None


def _normalize_item(item, default_type="injury"):
    if isinstance(item, dict):
        name = item.get("name")
        if not name:
            return None
        normalized = deepcopy(item)
        normalized.setdefault("type", default_type)
        return normalized

    if item is None:
        return None

    name = str(item).strip()
    return name if name else None


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _add_absence(result, team, item, default_type="injury"):
    normalized = _normalize_item(item, default_type)
    if not team or normalized is None:
        return

    result.setdefault(team, [])
    name = _absence_name(normalized)
    item_type = normalized.get("type", default_type) if isinstance(normalized, dict) else default_type

    for idx, existing in enumerate(result[team]):
        existing_name = _absence_name(existing)
        existing_type = existing.get("type", "injury") if isinstance(existing, dict) else "injury"
        if existing_name == name and existing_type == item_type:
            if isinstance(normalized, dict) and not isinstance(existing, dict):
                result[team][idx] = normalized
            return

    result[team].append(normalized)


def normalize_absences(raw_absences):
    """Return the canonical team-keyed absence map.

    Canonical shape:
    {
      "South Korea": ["Player Name", {"name": "Other Player", "type": "suspension"}]
    }

    Older nested shapes such as {"injuries": {...}, "suspensions": {...}} are
    accepted and merged into the same team-keyed result.
    """
    if not isinstance(raw_absences, dict):
        return {}

    result = {}

    for team, items in raw_absences.items():
        if team in RESERVED_ABSENCE_KEYS:
            continue
        for item in _as_list(items):
            _add_absence(result, team, item, "injury")

    nested_injuries = raw_absences.get("injuries")
    if isinstance(nested_injuries, dict):
        for team, items in nested_injuries.items():
            for item in _as_list(items):
                _add_absence(result, team, item, "injury")

    nested_suspensions = raw_absences.get("suspensions")
    if isinstance(nested_suspensions, dict):
        for team, items in nested_suspensions.items():
            for item in _as_list(items):
                _add_absence(result, team, item, "suspension")

    return result


def load_absences(path):
    return normalize_absences(load_json(path, {}))


def save_absences(path, absences):
    save_json(path, normalize_absences(absences))


def upsert_suspension(
    absences,
    team,
    player_name,
    reason,
    served_at_count,
    suspension_length=1,
):
    """Add or refresh one player's active suspension.

    A disciplinary decision may be amended after the initial match report.  The
    latest source must therefore replace an existing record rather than leave a
    stale one-match suspension in place.
    """
    if reason not in SUSPENSION_REASONS:
        raise ValueError(f"unsupported suspension reason: {reason!r}")
    if (
        not isinstance(served_at_count, int)
        or isinstance(served_at_count, bool)
        or served_at_count < 1
    ):
        raise ValueError("served_at_count must be a positive integer")
    if (
        not isinstance(suspension_length, int)
        or isinstance(suspension_length, bool)
        or suspension_length < 1
    ):
        raise ValueError("suspension_length must be a positive integer")
    if reason == "yellow_cards" and suspension_length != 1:
        raise ValueError("yellow-card accumulation must result in a one-match ban")

    normalized = normalize_absences(absences)
    record = {
        "name": player_name,
        "type": "suspension",
        "reason": reason,
        "served_at_count": served_at_count,
        "suspension_length": suspension_length,
    }
    team_records = normalized.setdefault(team, [])

    for index, item in enumerate(team_records):
        if _absence_name(item) == player_name:
            changed = item != record
            team_records[index] = record
            return normalized, changed

    team_records.append(record)
    return normalized, True


def get_absence_names(raw_list):
    names = []
    for item in _as_list(raw_list):
        name = _absence_name(item)
        if name:
            names.append(name)
    return names


def format_absence_list_to_str_list(raw_list):
    formatted = []
    reason_map = {
        "red_card": "red-card suspension",
        "yellow_cards": "yellow-card accumulation suspension",
        "disciplinary": "additional disciplinary suspension",
    }

    for item in _as_list(raw_list):
        if isinstance(item, dict):
            name = item.get("name")
            if not name:
                continue
            if item.get("type", "injury") == "suspension":
                reason_text = reason_map.get(item.get("reason"), "suspension")
                formatted.append(f"{name} ({reason_text})")
            else:
                formatted.append(f"{name} (injury)")
        else:
            name = _absence_name(item)
            if name:
                formatted.append(f"{name} (injury)")

    return formatted


def count_team_matches(team, actual_results):
    return sum(
        1
        for match in actual_results
        if match.get("team_a") == team or match.get("team_b") == team
    )


def clean_served_suspensions(absences, actual_results):
    cleaned = normalize_absences(absences)
    updated = False

    for team, players_list in list(cleaned.items()):
        new_list = []
        matches_played = count_team_matches(team, actual_results)

        for item in players_list:
            if (
                isinstance(item, dict)
                and item.get("type") == "suspension"
                and matches_played >= item.get("served_at_count", 0)
            ):
                updated = True
                continue
            new_list.append(item)

        if len(new_list) != len(players_list):
            updated = True
            if new_list:
                cleaned[team] = new_list
            else:
                del cleaned[team]

    return cleaned, updated


def build_squad_stats(squads):
    stats = {}
    for team, players in squads.items():
        total_value = sum(p["value_eur"] for p in players)
        if total_value == 0:
            continue

        hhi = sum((p["value_eur"] / total_value) ** 2 for p in players)
        attack_total = sum(
            p["value_eur"]
            for p in players
            if p["position"] not in ["Goalkeeper", "Defender"]
        )
        defense_total = sum(
            p["value_eur"]
            for p in players
            if p["position"] in ["Goalkeeper", "Defender"]
        )

        stats[team] = {
            "total_value": total_value,
            "hhi": hhi,
            "attack_total": attack_total,
            "defense_total": defense_total,
        }

    return stats


def calculate_absence_multipliers(team, absences, squads, squad_stats=None, include_values=False):
    if isinstance(absences, dict) and RESERVED_ABSENCE_KEYS.intersection(absences):
        absence_map = normalize_absences(absences)
    else:
        absence_map = absences if isinstance(absences, dict) else {}

    team_injuries = get_absence_names(absence_map.get(team, []))
    if not team_injuries or team not in squads:
        return 1.0, 1.0, []

    if squad_stats is None:
        squad_stats = build_squad_stats(squads)

    if team not in squad_stats:
        return 1.0, 1.0, []

    stats = squad_stats[team]
    players = squads[team]

    hhi = stats["hhi"]
    min_hhi = 0.0385
    max_hhi = 0.3000
    norm_hhi = max(0.0, min(1.0, (hhi - min_hhi) / (max_hhi - min_hhi)))
    depth_factor = 0.2 + 0.8 * norm_hhi

    attack_reduction = 0.0
    defense_reduction = 0.0
    details = []

    for player_name in team_injuries:
        matched_player = None
        for player in players:
            if player["name"].strip().lower() == player_name.strip().lower():
                matched_player = player
                break

        if not matched_player:
            continue

        pos = matched_player["position"]
        val = matched_player["value_eur"]
        pos_str = "defense" if pos in ["Goalkeeper", "Defender"] else "attack"

        if pos in ["Goalkeeper", "Defender"]:
            if stats["defense_total"] > 0:
                share = val / stats["defense_total"]
                reduction = share * depth_factor
                defense_reduction += reduction
                if include_values:
                    details.append(
                        f"{matched_player['name']} ({pos_str}, value: €{val/1000000:.1f}M, "
                        f"position share: {share*100:.1f}%, reduction: {reduction*100:.1f}%)"
                    )
                else:
                    details.append(f"{matched_player['name']} ({pos_str}, reduction: {reduction*100:.1f}%)")
        elif stats["attack_total"] > 0:
            share = val / stats["attack_total"]
            reduction = share * depth_factor
            attack_reduction += reduction
            if include_values:
                details.append(
                    f"{matched_player['name']} ({pos_str}, value: €{val/1000000:.1f}M, "
                    f"position share: {share*100:.1f}%, reduction: {reduction*100:.1f}%)"
                )
            else:
                details.append(f"{matched_player['name']} ({pos_str}, reduction: {reduction*100:.1f}%)")

    attack_multiplier = max(0.5, 1.0 - attack_reduction)
    defense_multiplier = min(2.0, 1.0 + defense_reduction)
    return attack_multiplier, defense_multiplier, details
