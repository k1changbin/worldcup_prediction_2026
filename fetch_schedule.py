import json
import os
import httpx

# ELO 국가명과 API 국가명 간 매핑 사전
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
    
    print(f"[일정 수집] {url} 에서 월드컵 일정 수집 중...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        r = httpx.get(url, headers=headers, timeout=15.0)
        if r.status_code != 200:
            print(f"[에러] HTTP {r.status_code} 응답을 받았습니다.")
            return False
            
        data = r.json()
        fixtures = data.get("fixtures", [])
        
        mapped_fixtures = []
        for f in fixtures:
            # 팀 이름 매핑 적용
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
            
        # 디렉터리 생성 및 저장
        os.makedirs(os.path.dirname(schedules_path), exist_ok=True)
        with open(schedules_path, "w", encoding="utf-8") as file:
            json.dump(mapped_fixtures, file, ensure_ascii=False, indent=2)
            
        print(f"[성공] 월드컵 일정 {len(mapped_fixtures)}개 경기 수집 완료 ➡️ {schedules_path}")
        return True
        
    except Exception as e:
        print(f"[에러] 일정 수집 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    fetch_schedule()
