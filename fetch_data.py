import json
import os
import httpx

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

        for event in data.get("events", []):
            for competition in event.get("competitions", []):
                competitors = competition.get("competitors", [])
                if len(competitors) != 2:
                    continue

                teams = []
                winner = None
                for competitor in competitors:
                    team_data = competitor.get("team", {})
                    raw_name = (
                        team_data.get("displayName")
                        or team_data.get("name")
                        or team_data.get("location")
                    )
                    team_name = normalize_team_name(raw_name, valid_teams)
                    if not team_name:
                        break
                    teams.append(team_name)
                    if competitor.get("advance") or competitor.get("winner"):
                        winner = team_name

                if len(teams) == 2 and winner:
                    decisions[frozenset(teams)] = winner

    return decisions

def update_team_conduct_scores(valid_teams, headers):
    from bs4 import BeautifulSoup

    conduct_path = "data/team_conduct_scores.json"
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

    with open(conduct_path, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(conduct_scores.items())), f, ensure_ascii=False, indent=2)

    print(
        f"[Conduct] Update complete: checked {parsed_matches} matches, "
        f"applied card records from {matches_with_cards} matches -> {conduct_path}"
    )

def fetch_live_world_cup_data():
    elo_ratings_path = "data/elo_ratings.json"
    actual_results_path = "data/actual_results.json"
    
    if not os.path.exists(elo_ratings_path):
        print(f"Error: {elo_ratings_path} was not found. Run this from the project root.")
        return
        
    # Load the 48 participating teams.
    with open(elo_ratings_path, "r", encoding="utf-8") as f:
        local_ratings = json.load(f)
        valid_teams = set(local_ratings.keys())
        
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # 1. Download team-name/code mapping file.
    print("[ELO] Downloading team-name/code mapping from eloratings.net...")
    try:
        r_teams = httpx.get("https://www.eloratings.net/en.teams.tsv", headers=headers, timeout=15.0)
        r_teams.raise_for_status()
    except Exception as e:
        print(f"Failed to request team mapping file: {e}")
        return
        
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
        print(f"Failed to request Elo ratings file: {e}")
        return
        
    updated_ratings = dict(local_ratings)
    ratings_updated_count = 0
    
    for line in r_world.text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 4:
            country_code = parts[2]
            try:
                elo_val = float(parts[3])
            except ValueError:
                continue
                
            raw_name = code_to_name.get(country_code)
            normalized_name = normalize_team_name(raw_name, valid_teams)
            
            if normalized_name in valid_teams:
                updated_ratings[normalized_name] = elo_val
                ratings_updated_count += 1
                
    with open(elo_ratings_path, "w", encoding="utf-8") as f:
        json.dump(updated_ratings, f, ensure_ascii=False, indent=2)
        
    print(f"[ELO] Elo ratings update complete: applied ratings for {ratings_updated_count} teams")
    
    # 3. Download and parse 2026 match results.
    print("[Match results] Downloading 2026 match-result data from eloratings.net...")
    try:
        r_results = httpx.get("https://www.eloratings.net/2026_results.tsv", headers=headers, timeout=15.0)
        r_results.raise_for_status()
    except Exception as e:
        print(f"Failed to request match results file: {e}")
        return
        
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
                
    # Load schedule data to identify each match stage.
    schedule_data = []
    schedule_path = "data/schedule.json"
    if os.path.exists(schedule_path):
        with open(schedule_path, "r", encoding="utf-8") as f:
            try:
                schedule_data = json.load(f)
            except json.JSONDecodeError:
                pass
                
    def get_stage(team_a, team_b, date_str, month, day):
        for s in schedule_data:
            if s.get("date") == date_str:
                if (s.get("homeTeam") in (team_a, team_b)) and (s.get("awayTeam") in (team_a, team_b)):
                    return "knockout" if s.get("stage") != "group-stage" else "group"
        # fallback
        return "knockout" if (month == 6 and day >= 28) or (month >= 7) else "group"

    actual_results = []
    tied_knockout_dates = set()
    for idx, match in enumerate(parsed_matches):
        team_a = match["team_a"]
        team_b = match["team_b"]
        score_a = match["score_a"]
        score_b = match["score_b"]
        month = match["month"]
        day = match["day"]
        
        date_str = f"{match['year']}-{month:02d}-{day:02d}"
        stage = get_stage(team_a, team_b, date_str, month, day)
        
        winner = None
        if score_a > score_b:
            winner = team_a
        elif score_b > score_a:
            winner = team_b
        else:
            if stage == "knockout":
                # Search later matches to infer which tied team advanced.
                # Note: a third-place playoff can appear before the final.
                found_next = False
                for next_match in parsed_matches[idx + 1:]:
                    next_teams = {next_match["team_a"], next_match["team_b"]}
                    if team_a in next_teams and not found_next:
                        winner = team_a
                        found_next = True
                    elif team_b in next_teams and not found_next:
                        winner = team_b
                        found_next = True
                        
                # If there is no later match, as with the final, keep winner as None.
                if winner is None:
                    tied_knockout_dates.add(date_str)
                        
        actual_results.append({
            "team_a": team_a,
            "team_b": team_b,
            "score_a": score_a,
            "score_b": score_b,
            "date": f"{match['year']}-{month:02d}-{day:02d}",
            "stage": stage,
            "winner": winner
        })

    if tied_knockout_dates:
        espn_decisions = fetch_espn_knockout_decisions(tied_knockout_dates, valid_teams, headers)
        applied_decisions = 0
        for result in actual_results:
            if (
                result["stage"] == "knockout"
                and result["score_a"] == result["score_b"]
                and not result["winner"]
            ):
                winner = espn_decisions.get(frozenset((result["team_a"], result["team_b"])))
                if winner:
                    result["winner"] = winner
                    applied_decisions += 1
        if applied_decisions:
            print(f"[Match results] Applied {applied_decisions} ESPN penalty/advancement decisions.")
                
    # Load and compare existing local results.
    local_results = []
    if os.path.exists(actual_results_path):
        try:
            with open(actual_results_path, "r", encoding="utf-8") as f:
                local_results = json.load(f)
        except Exception:
            local_results = []

    is_same = False
    if len(local_results) == len(actual_results):
        is_same = True
        for lr, ar in zip(local_results, actual_results):
            if (lr.get("team_a") != ar.get("team_a") or
                lr.get("team_b") != ar.get("team_b") or
                lr.get("score_a") != ar.get("score_a") or
                lr.get("score_b") != ar.get("score_b") or
                lr.get("date") != ar.get("date") or
                lr.get("stage") != ar.get("stage") or
                lr.get("winner") != ar.get("winner")):
                is_same = False
                break

    if is_same:
        print("[Match results] No new completed matches were added.")
    else:
        with open(actual_results_path, "w", encoding="utf-8") as f:
            json.dump(actual_results, f, ensure_ascii=False, indent=2)
        print(f"Match results update complete: locked {len(actual_results)} completed matches in {actual_results_path}.")
        for res in actual_results:
            winner_str = f" (winner: {res['winner']})" if res['winner'] else ""
            print(f"   - [{res['stage'].upper()}] {res['team_a']} {res['score_a']} : {res['score_b']} {res['team_b']}{winner_str}")
        
    # 4. Sync live suspension data from Wikipedia.
    try:
        import fetch_suspensions
        print("\n[ELO & suspensions] Syncing live suspension data after Elo/result updates...")
        fetch_suspensions.main()
    except Exception as e:
        print(f"\n[Warning] Failed to sync live suspension data. Elo/results were still updated: {e}")

    # 5. Conduct scores are no longer updated after the group stage.
    #    The function remains available for standings recalculation or verification.
    print("[Conduct] Skipping team_conduct_scores.json update because the group stage is complete.")

if __name__ == "__main__":
    fetch_live_world_cup_data()
