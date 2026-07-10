import json
import os
import re
import unicodedata
import httpx
from bs4 import BeautifulSoup
from src.absences import clean_served_suspensions, load_absences, load_json, save_absences
from src.tournament_state import filter_team_map, load_active_teams
from src.paths import data_path
from src.io_utils import atomic_write_json

# Mapping between Elo team names and Wikipedia headings.
TEAM_ALIASES = {
    "United States": "USA",
    "Czech Republic": "Czechia",
    "Turkey": "Türkiye"
}

# Star-player database used for fixed market-value inputs.
STAR_PLAYERS = {
    "South Korea": [
        {"name": "Son Heung-min", "position": "Forward", "value": 50000000},
        {"name": "Kim Min-jae", "position": "Defender", "value": 60000000},
        {"name": "Lee Kang-in", "position": "Midfielder", "value": 25000000},
        {"name": "Hwang Hee-chan", "position": "Forward", "value": 25000000}
    ],
    "Argentina": [
        {"name": "Lionel Messi", "position": "Forward", "value": 30000000},
        {"name": "Lautaro Martinez", "position": "Forward", "value": 110000000},
        {"name": "Enzo Fernandez", "position": "Midfielder", "value": 75000000},
        {"name": "Alexis Mac Allister", "position": "Midfielder", "value": 75000000},
        {"name": "Julian Alvarez", "position": "Forward", "value": 90000000},
        {"name": "Emiliano Martinez", "position": "Goalkeeper", "value": 28000000},
        {"name": "Cristian Romero", "position": "Defender", "value": 60000000}
    ],
    "France": [
        {"name": "Kylian Mbappe", "position": "Forward", "value": 180000000},
        {"name": "Antoine Griezmann", "position": "Forward", "value": 25000000},
        {"name": "Eduardo Camavinga", "position": "Midfielder", "value": 90000000},
        {"name": "Aurelien Tchouameni", "position": "Midfielder", "value": 90000000},
        {"name": "William Saliba", "position": "Defender", "value": 80000000},
        {"name": "Ousmane Dembele", "position": "Forward", "value": 60000000}
    ],
    "Spain": [
        {"name": "Rodri", "position": "Midfielder", "value": 120000000},
        {"name": "Lamine Yamal", "position": "Forward", "value": 90000000},
        {"name": "Pedri", "position": "Midfielder", "value": 80000000},
        {"name": "Gavi", "position": "Midfielder", "value": 90000000},
        {"name": "Nico Williams", "position": "Forward", "value": 60000000},
        {"name": "Dani Carvajal", "position": "Defender", "value": 120000000}
    ],
    "England": [
        {"name": "Harry Kane", "position": "Forward", "value": 100000000},
        {"name": "Jude Bellingham", "position": "Midfielder", "value": 180000000},
        {"name": "Bukayo Saka", "position": "Forward", "value": 140000000},
        {"name": "Declan Rice", "position": "Midfielder", "value": 120000000},
        {"name": "John Stones", "position": "Defender", "value": 38000000}
    ],
    "Norway": [
        {"name": "Erling Haaland", "position": "Forward", "value": 180000000},
        {"name": "Martin Odegaard", "position": "Midfielder", "value": 95000000}
    ],
    "Portugal": [
        {"name": "Cristiano Ronaldo", "position": "Forward", "value": 15000000},
        {"name": "Bruno Fernandes", "position": "Midfielder", "value": 70000000},
        {"name": "Bernardo Silva", "position": "Midfielder", "value": 70000000},
        {"name": "Ruben Dias", "position": "Defender", "value": 80000000},
        {"name": "Rafael Leao", "position": "Forward", "value": 90000000}
    ],
    "Brazil": [
        {"name": "Vinicius Junior", "position": "Forward", "value": 150000000},
        {"name": "Rodrygo", "position": "Forward", "value": 100000000},
        {"name": "Bruno Guimaraes", "position": "Midfielder", "value": 85000000},
        {"name": "Gabriel Magalhaes", "position": "Defender", "value": 70000000},
        {"name": "Alisson", "position": "Goalkeeper", "value": 28000000}
    ],
    "Egypt": [
        {"name": "Mohamed Salah", "position": "Forward", "value": 55000000}
    ],
    "Belgium": [
        {"name": "Kevin De Bruyne", "position": "Midfielder", "value": 50000000},
        {"name": "Romelu Lukaku", "position": "Forward", "value": 30000000}
    ],
    "Croatia": [
        {"name": "Luka Modric", "position": "Midfielder", "value": 60000000},
        {"name": "Josko Gvardiol", "position": "Defender", "value": 75000000}
    ],
    "Uruguay": [
        {"name": "Federico Valverde", "position": "Midfielder", "value": 120000000},
        {"name": "Darwin Nunez", "position": "Forward", "value": 70000000},
        {"name": "Ronald Araujo", "position": "Defender", "value": 70000000}
    ],
    "Germany": [
        {"name": "Florian Wirtz", "position": "Midfielder", "value": 110000000},
        {"name": "Jamal Musiala", "position": "Midfielder", "value": 110000000},
        {"name": "Kai Havertz", "position": "Forward", "value": 75000000},
        {"name": "Antonio Rudiger", "position": "Defender", "value": 25000000},
        {"name": "Manuel Neuer", "position": "Goalkeeper", "value": 4000000}
    ],
    "USA": [
        {"name": "Christian Pulisic", "position": "Forward", "value": 35000000},
        {"name": "Weston McKennie", "position": "Midfielder", "value": 25000000},
        {"name": "Folarin Balogun", "position": "Forward", "value": 30000000}
    ],
    "Canada": [
        {"name": "Alphonso Davies", "position": "Defender", "value": 50000000},
        {"name": "Jonathan David", "position": "Forward", "value": 50000000}
    ]
}

def normalize_team(heading):
    heading = heading.strip()
    return TEAM_ALIASES.get(heading, heading)

def clean_player_name(name):
    # Remove citation brackets like [a] or [12]
    name = re.sub(r"\[[^\]]+\]", "", name)
    # Remove text in parentheses (e.g. (captain))
    name = re.sub(r"\([^)]+\)", "", name)
    # Remove asterisks or other punctuation
    name = name.replace("*", "")
    return name.strip()

def normalize_for_matching(name):
    # Normalize unicode to decompose characters (remove accents)
    name = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in name if not unicodedata.combining(c))
    # Replace hyphens/special spaces with normal spaces
    ascii_name = ascii_name.replace("-", " ").replace("_", " ")
    return " ".join(ascii_name.lower().split())

def get_deterministic_squad_value(rating):
    # Estimate total squad market value from Elo with a deterministic formula.
    if rating >= 2100:
        return int(800000000 + (rating - 2100) * 30000000)
    elif rating >= 2000:
        return int(500000000 + (rating - 2000) * 30000000)
    elif rating >= 1900:
        return int(300000000 + (rating - 1900) * 20000000)
    elif rating >= 1800:
        return int(150000000 + (rating - 1800) * 15000000)
    elif rating >= 1700:
        return int(50000000 + (rating - 1700) * 10000000)
    else:
        return int(10000000 + max(0.0, rating - 1200) * 800000)


def merge_active_squad_snapshot(active_teams, parsed_squads, existing_squads):
    """Return a complete active-team snapshot without erasing good old data."""
    merged = dict(parsed_squads)
    for team in set(active_teams) - set(merged):
        if existing_squads.get(team):
            merged[team] = existing_squads[team]
    missing = sorted(set(active_teams) - set(merged))
    return merged, missing

def fetch_live_injuries_and_squads():
    squads_path = data_path("squads.json")
    injuries_path = data_path("absences.json")
    elo_path = data_path("elo_ratings.json")
    actual_results_path = data_path("actual_results.json")

    if not os.path.exists(elo_path):
        raise FileNotFoundError(f"{elo_path} does not exist")

    with open(elo_path, "r", encoding="utf-8") as f:
        ratings = json.load(f)
    existing_squads = load_json(squads_path, {})
    active_teams = load_active_teams(ratings_path=elo_path)
    if not active_teams:
        active_teams = set(ratings.keys())
    print(f"[Squad & injury fetch] Considering {len(active_teams)} active teams.")

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
    print(f"[Squad & injury fetch] Fetching data from Wikipedia... ({url})")

    try:
        r = httpx.get(url, headers=headers, timeout=15.0)
        if r.status_code != 200:
            raise RuntimeError(f"Wikipedia returned HTTP {r.status_code}")
    except Exception as e:
        raise RuntimeError(f"Wikipedia request failed: {e}") from e

    soup = BeautifulSoup(r.text, "html.parser")
    headings = soup.find_all(["h2", "h3", "h4"])
    
    parsed_teams_data = {}
    live_injuries = {}
    total_players_count = 0
    total_injuries_count = 0

    print("Parsing Wikipedia page...")

    for h in headings:
        heading_text = h.text.strip()
        normalized_heading = normalize_team(heading_text)

        # Keep only teams still active in the tournament.
        if normalized_heading not in active_teams:
            continue

        # Support both wrapped heading divs and normal heading sibling structures.
        p = h.parent
        is_wrapped = p and p.name == "div" and any(cls.startswith("mw-heading") for cls in p.get("class", []))
        start_el = p if is_wrapped else h

        # 1. Collect descriptive paragraphs and the squad table.
        paragraphs = []
        table = None
        curr = start_el.find_next_sibling()
        while curr:
            if curr.name in ["h2", "h3", "h4"]:
                break
            if curr.name == "div" and any(cls.startswith("mw-heading") for cls in curr.get("class", [])):
                break
            if curr.name == "p":
                paragraphs.append(curr.text.strip())
            if curr.name == "table" and not table:
                table = curr
            curr = curr.find_next_sibling()

        text_block = " ".join(paragraphs)

        # 2. Parse squad table.
        players_list = []
        if table:
            rows = table.find_all("tr")[1:]  # Skip header row.
            for row in rows:
                cols = row.find_all(["th", "td"])
                if len(cols) < 7:
                    continue
                
                raw_no = cols[0].text.strip()
                raw_pos = cols[1].text.strip()
                raw_name = cols[2].text.strip()
                raw_caps = cols[4].text.strip()

                # Map positions.
                pos_str = "Forward"
                if "GK" in raw_pos:
                    pos_str = "Goalkeeper"
                elif "DF" in raw_pos:
                    pos_str = "Defender"
                elif "MF" in raw_pos:
                    pos_str = "Midfielder"
                elif "FW" in raw_pos:
                    pos_str = "Forward"

                # Clean player name.
                name_clean = clean_player_name(raw_name)

                # Clean and parse caps.
                caps_val = 0
                try:
                    caps_val = int(re.sub(r"\D", "", raw_caps))
                except ValueError:
                    pass

                players_list.append({
                    "name": name_clean,
                    "position": pos_str,
                    "caps": caps_val,
                    "value_eur": 0  # Temporary value.
                })

        if not players_list:
            print(f"   - [Warning] Could not parse squad table for {normalized_heading}.")
            continue

        # 3. Analyze injury/replacement information in paragraphs.
        team_injured_names = []
        p_clean = re.sub(r"\[\d+\]", "", text_block)
        
        # Find sentences that include both "withdrew" and "replaced by".
        sentences = re.split(r"(?<=[.!?])\s+", p_clean)
        for sentence in sentences:
            if "withdrew" in sentence and "replaced by" in sentence:
                parts = sentence.split("withdrew")
                before = parts[0].strip()
                if "." in before:
                    before = before.split(".")[-1].strip()
                
                # Extract withdrawn player names.
                withdrawn_raw = re.split(r"\b(?:and|,)\b", before)
                withdrawn_players = [w.strip() for w in withdrawn_raw if w.strip()]

                # Extract replacement player names.
                after = parts[1]
                replaced_players = []
                if "replaced by" in after:
                    after_part = after.split("replaced by")[1].strip()
                    after_clean = re.split(r"\b(?:on|at|with|due|in|respectively)\b|[,.]", after_part)[0].strip()
                    replaced_raw = re.split(r"\b(?:and|,)\b", after_clean)
                    replaced_players = [r.strip() for r in replaced_raw if r.strip()]

                # Match replacements and restore the withdrawn player in the squad list.
                # This lets the simulation calculate value loss from the missing original player.
                for w_name, r_name in zip(withdrawn_players, replaced_players):
                    w_norm = normalize_for_matching(w_name)
                    r_norm = normalize_for_matching(r_name)

                    # Find the replacement player in the table.
                    matched_idx = -1
                    for idx, p_item in enumerate(players_list):
                        if normalize_for_matching(p_item["name"]) == r_norm:
                            matched_idx = idx
                            break

                    if matched_idx != -1:
                        # Restore the original withdrawn player over the replacement.
                        players_list[matched_idx]["name"] = w_name
                        team_injured_names.append(w_name)
                    else:
                        # If the replacement is not found, still add the withdrawn player as absent.
                        team_injured_names.append(w_name)

        parsed_teams_data[normalized_heading] = players_list
        total_players_count += len(players_list)

        if team_injured_names:
            live_injuries[normalized_heading] = team_injured_names
            total_injuries_count += len(team_injured_names)
            print(f"   - {normalized_heading}: parsed {len(players_list)} players (withdrawn: {team_injured_names})")
        else:
            print(f"   - {normalized_heading}: parsed {len(players_list)} players")

    # 4. Assign market values from Elo-based totals plus manually specified star players.
    final_squads = {}
    for team, players in parsed_teams_data.items():
        rating = ratings.get(team, 1700.0)
        total_squad_value = get_deterministic_squad_value(rating)

        stars = STAR_PLAYERS.get(team, [])
        star_values_sum = 0
        star_names_lower = {}
        for s in stars:
            star_names_lower[normalize_for_matching(s["name"])] = s

        # Assign star-player values first.
        assigned_players = []
        regular_players = []

        for p in players:
            p_norm = normalize_for_matching(p["name"])
            if p_norm in star_names_lower:
                star_data = star_names_lower[p_norm]
                p["value_eur"] = star_data["value"]
                # Correct the position if it differs from the star-player database.
                p["position"] = star_data["position"]
                assigned_players.append(p)
                star_values_sum += star_data["value"]
            else:
                regular_players.append(p)

        # Distribute remaining value.
        remaining_value = max(total_squad_value * 0.2, total_squad_value - star_values_sum)
        
        # Sort regular players by caps and allocate value with a Pareto-style distribution.
        regular_players.sort(key=lambda x: x["caps"], reverse=True)
        
        num_regulars = len(regular_players)
        if num_regulars > 0:
            weights = [1.0 / (i + 1) for i in range(num_regulars)]
            sum_weights = sum(weights)
            
            for idx, p in enumerate(regular_players):
                w = weights[idx]
                val = int((w / sum_weights) * remaining_value)
                p["value_eur"] = max(100000, val)
                assigned_players.append(p)

        # Store assigned players as parsed.
        final_squads[team] = assigned_players

    # A transient parser failure must not erase a previously valid active
    # squad. Keep the last known snapshot for teams that were not parsed.
    final_squads, uncovered_teams = merge_active_squad_snapshot(
        active_teams,
        final_squads,
        existing_squads,
    )
    if uncovered_teams:
        raise RuntimeError(
            "No current or previous squad data for: "
            + ", ".join(uncovered_teams)
            + ". Existing files were not changed."
        )

    # 5. Save files.
    atomic_write_json(squads_path, final_squads)

    absences_data = load_absences(injuries_path)
    actual_results = load_json(actual_results_path, [])
    absences_data, _ = clean_served_suspensions(absences_data, actual_results)
    preserved_suspensions = {}
    for team, items in filter_team_map(absences_data, active_teams).items():
        suspensions = [
            item
            for item in items
            if isinstance(item, dict) and item.get("type") == "suspension"
        ]
        if suspensions:
            preserved_suspensions[team] = suspensions

    canonical_absences = preserved_suspensions
    # Preserve non-suspension absences for teams whose page section could not
    # be parsed; a successful parse with no injury intentionally clears them.
    for team in active_teams - set(parsed_teams_data):
        previous_injuries = [
            item
            for item in absences_data.get(team, [])
            if not (isinstance(item, dict) and item.get("type") == "suspension")
        ]
        if previous_injuries:
            canonical_absences.setdefault(team, []).extend(previous_injuries)
    for team, players in live_injuries.items():
        canonical_absences.setdefault(team, [])
        canonical_absences[team].extend(players)

    save_absences(injuries_path, canonical_absences)

    print(f"\n[Success] Data update complete.")
    print(f"   - Squad data saved to: {squads_path} ({len(final_squads)} teams, {total_players_count} players)")
    print(f"   - Absence data saved to: {injuries_path} ({len(live_injuries)} teams, {total_injuries_count} players)")
    return True

if __name__ == "__main__":
    fetch_live_injuries_and_squads()
