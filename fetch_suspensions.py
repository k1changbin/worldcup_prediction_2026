import httpx
from bs4 import BeautifulSoup
import json
import os
import re
from src.absences import clean_served_suspensions, load_absences, save_absences
from src.tournament_state import filter_team_map, load_active_teams

# Mapping between website team names and project team names.
TEAM_NAME_MAP = {
    "United States": "USA",
    "Turkey": "Türkiye",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Cote d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "Turkiye": "Türkiye"
}

def normalize_team_name(name, valid_teams):
    if not name:
        return None
    name = name.strip()
    
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
        
    for valid in valid_teams:
        if valid.lower() == name.lower():
            return valid
            
    return None

URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"
ABSENCES_PATH = "data/absences.json"
SQUADS_PATH = "data/squads.json"
ACTUAL_RESULTS_PATH = "data/actual_results.json"

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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_actual_match_count(team_name, actual_results):
    count = 0
    for match in actual_results:
        # Check if the team was in this match
        # stage can be group or knockout
        if match.get("team_a") == team_name or match.get("team_b") == team_name:
            count += 1
    return count

def main():
    print("[Suspension sync] Fetching live suspended-player data from Wikipedia...")
    
    # Load data.
    squads = load_json(SQUADS_PATH)
    actual_results = load_json(ACTUAL_RESULTS_PATH, [])
    all_injuries = load_absences(ABSENCES_PATH)
    active_teams = load_active_teams()
    if active_teams:
        squads = filter_team_map(squads, active_teams)
        injuries = filter_team_map(all_injuries, active_teams)
        pruned_absence_count = sum(
            len(items)
            for team, items in all_injuries.items()
            if team not in active_teams
        )
        print(f"[Suspension sync] Considering {len(active_teams)} active teams.")
    else:
        injuries = all_injuries
        pruned_absence_count = 0
    
    html_content = None
    
    # 1. Try fetching from Wikipedia. A User-Agent is required.
    try:
        response = httpx.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10.0)
        if response.status_code == 200:
            html_content = response.text
            print("[Suspension sync] Loaded the official Wikipedia page.")
        else:
            print(f"[Suspension sync] Wikipedia page is unavailable or failed to load. (HTTP {response.status_code})")
    except Exception as e:
        print(f"[Suspension sync] Wikipedia request failed: {e}")
        
    # 2. If the live page is unavailable, fall back to a local mock file.
    if not html_content:
        mock_path = "scratch/mock_disciplinary_record.html"
        if os.path.exists(mock_path):
            print(f"[Suspension sync] Parsing local mock file: {mock_path}")
            with open(mock_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        else:
            print("[Suspension sync] No usable data source is available. Exiting.")
            return

    # 3. Parse HTML.
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    
    suspensions_table = None
    table_type = None # "main_page" or "disciplinary_page"
    
    for table in tables:
        # Inspect header rows.
        headers = [th.text.strip().lower() for th in table.find_all("th")]
        # Check whether Player and Suspension are present in the headers.
        has_player = any("player" in h for h in headers)
        has_suspension = any("suspension" in h or "served" in h or "match" in h for h in headers)
        has_offense = any("offense" in h or "booking" in h or "card" in h for h in headers)
        
        if has_player and has_suspension:
            has_team = any("team" in h or "country" in h or "association" in h for h in headers)
            if has_team:
                suspensions_table = table
                table_type = "disciplinary_page"
                break
            elif has_offense:
                suspensions_table = table
                table_type = "main_page"
                break

    if not suspensions_table:
        print("[Suspension sync] Could not find a Suspensions table.")
        return

    # 4. Extract and map table data.
    rows = suspensions_table.find_all("tr")
    added_count = 0
    
    for row in rows:
        tds = row.find_all("td")
        if len(tds) < 3:
            continue
            
        if table_type == "disciplinary_page":
            player_raw = tds[0].text.strip()
            team_raw = tds[1].text.strip()
            offense_raw = tds[2].text.strip() if len(tds) > 2 else ""
            suspension_raw = tds[3].text.strip() if len(tds) > 3 else ""
        else: # "main_page"
            player_raw = tds[0].text.strip()
            
            # Extract the country from the flag icon image or link title.
            flagicon = tds[0].find(class_="flagicon")
            team_raw = ""
            if flagicon:
                a_tag = flagicon.find("a")
                if a_tag and "title" in a_tag.attrs:
                    team_raw = a_tag["title"]
                    for suffix in ["national soccer team", "national football team", "national team"]:
                        team_raw = team_raw.replace(suffix, "").strip()
            
            if not team_raw:
                team_raw = tds[0].text.strip()
                
            offense_raw = tds[1].text.strip()
            suspension_raw = tds[2].text.strip()
        
        # Map the team.
        matched_team = normalize_team_name(team_raw, squads.keys())
                
        if not matched_team:
            continue
            
        # Map the player.
        matched_player = None
        team_players = squads[matched_team]
        for p in team_players:
            p_name = p["name"]
            # Use stricter matching: exact name, or every player-name token is present in the raw text.
            if p_name.lower() == player_raw.lower() or all(word.lower() in player_raw.lower() for word in p_name.split()):
                matched_player = p_name
                break
                
        if not matched_player:
            continue
            
        # Determine match count and suspension length.
        N = get_actual_match_count(matched_team, actual_results)
        
        # Parse suspension length. Default to one match.
        suspension_length = 1
        combined_text = (offense_raw + " " + suspension_raw).lower()
        if "2 matches" in combined_text or "two matches" in combined_text:
            suspension_length = 2
        elif "3 matches" in combined_text or "three matches" in combined_text:
            suspension_length = 3
            
        matchday_nums = [int(x) for x in re.findall(r"matchday\s*(\d+)", combined_text)]
        if matchday_nums:
            served_at_count = max(matchday_nums)
        else:
            served_at_count = N + suspension_length

        if N >= served_at_count:
            continue
            
        reason = "red_card" if "red" in combined_text else "yellow_cards"
        
        # Add the suspension record to the absences database.
        if matched_team not in injuries:
            injuries[matched_team] = []
            
        # Check whether this player is already registered by name.
        already_exists = False
        for idx, item in enumerate(injuries[matched_team]):
            # Support both legacy string records and structured dictionary records.
            item_name = item if isinstance(item, str) else item.get("name")
            if item_name == matched_player:
                already_exists = True
                # Upgrade or refresh legacy/non-suspension records to structured suspensions.
                if isinstance(item, str) or item.get("type") != "suspension":
                    injuries[matched_team][idx] = {
                        "name": matched_player,
                        "type": "suspension",
                        "reason": reason,
                        "served_at_count": served_at_count
                    }
                    added_count += 1
                break
                
        if not already_exists:
            injuries[matched_team].append({
                "name": matched_player,
                "type": "suspension",
                "reason": reason,
                "served_at_count": served_at_count
            })
            added_count += 1
            print(f"[Suspension sync] Registered {matched_player} for {matched_team} (returns after match count: {served_at_count})")

    injuries, cleaned_served = clean_served_suspensions(injuries, actual_results)

    if added_count > 0 or pruned_absence_count > 0 or cleaned_served:
        save_absences(ABSENCES_PATH, injuries)
        if added_count > 0:
            print(f"[Suspension sync] Synced {added_count} suspended players into the database.")
        if pruned_absence_count > 0:
            print(f"[Suspension sync] Removed {pruned_absence_count} absence records for eliminated teams.")
        if cleaned_served:
            print("[Suspension sync] Removed served suspension records.")
    else:
        print("[Suspension sync] No new suspended players were added.")

if __name__ == "__main__":
    main()
