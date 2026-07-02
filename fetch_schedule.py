import json
import os
import httpx

# Mapping between Elo team names and API team names.
TEAM_NAME_MAP = {
    "United States": "USA",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Cote d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "Turkiye": "Türkiye"
}

def map_team_name(name):
    if not name:
        return name
    return TEAM_NAME_MAP.get(name, name)

def fetch_schedule():
    url = "https://www.thestatsapi.com/world-cup/data/fixtures.json"
    schedules_path = "data/schedule.json"
    
    print(f"[Schedule fetch] Fetching World Cup schedule from {url}...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        r = httpx.get(url, headers=headers, timeout=15.0)
        if r.status_code != 200:
            print(f"[Error] Received HTTP {r.status_code}.")
            return False
            
        data = r.json()
        fixtures = data.get("fixtures", [])
        
        mapped_fixtures = []
        for f in fixtures:
            # Apply team-name mapping.
            home = map_team_name(f.get("homeTeam"))
            away = map_team_name(f.get("awayTeam"))
            
            mapped_fixtures.append({
                "matchNumber": f.get("matchNumber"),
                "date": f.get("date"),
                "kickoffUtc": f.get("kickoffUtc"),
                "stage": f.get("stage"),
                "group": f.get("group"),
                "homeTeam": home,
                "awayTeam": away,
                "stadium": f.get("stadium"),
                "hostCity": f.get("hostCity")
            })
            
        # Create the directory and save the file.
        os.makedirs(os.path.dirname(schedules_path), exist_ok=True)
        with open(schedules_path, "w", encoding="utf-8") as file:
            json.dump(mapped_fixtures, file, ensure_ascii=False, indent=2)
            
        print(f"[Success] Fetched {len(mapped_fixtures)} World Cup matches -> {schedules_path}")
        return True
        
    except Exception as e:
        print(f"[Error] Failed to fetch schedule: {e}")
        return False

if __name__ == "__main__":
    fetch_schedule()
