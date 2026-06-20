import httpx
from bs4 import BeautifulSoup
import json
import os
import sys

URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"
ABSENCES_PATH = "data/absences.json"
SQUADS_PATH = "data/squads.json"
ACTUAL_RESULTS_PATH = "data/actual_results.json"

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

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
    print("[징계 동기화] Wikipedia에서 실시간 출장정지 선수 명단을 수집합니다...")
    
    # 데이터 로드
    squads = load_json(SQUADS_PATH)
    actual_results = load_json(ACTUAL_RESULTS_PATH)
    injuries = load_json(ABSENCES_PATH)
    
    html_content = None
    
    # 1. Wikipedia에서 가져오기 시도 (User-Agent 필수)
    try:
        response = httpx.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10.0)
        if response.status_code == 200:
            html_content = response.text
            print("[징계 동기화] Wikipedia 공식 페이지 로드 성공.")
        else:
            print(f"[징계 동기화] Wikipedia 페이지가 존재하지 않거나 로드에 실패했습니다. (HTTP {response.status_code})")
    except Exception as e:
        print(f"[징계 동기화] Wikipedia 연결 중 오류 발생: {e}")
        
    # 2. 로드 실패 시 로컬 모크 파일 체크
    if not html_content:
        mock_path = "scratch/mock_disciplinary_record.html"
        if os.path.exists(mock_path):
            print(f"[징계 동기화] 로컬 테스트 모크 파일({mock_path})을 사용하여 파싱을 진행합니다.")
            with open(mock_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        else:
            print("[징계 동기화] 가동 가능한 데이터 소스가 없어 종료합니다.")
            return

    # 3. HTML 파싱
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.find_all("table")
    
    suspensions_table = None
    table_type = None # "main_page" or "disciplinary_page"
    
    for table in tables:
        # 헤더 행 검사
        headers = [th.text.strip().lower() for th in table.find_all("th")]
        # Player와 Suspension이 헤더에 포함되어 있는지 확인
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
        print("[징계 동기화] Suspensions 테이블을 찾을 수 없습니다.")
        return

    # 4. 테이블 데이터 추출 및 매핑
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
            
            # flagicon 내부의 깃발 이미지 또는 링크 타이틀에서 국가 추출
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
        
        # 국가(Team) 매핑
        matched_team = None
        for team_name in squads.keys():
            if team_name.lower() in team_raw.lower() or team_raw.lower() in team_name.lower():
                matched_team = team_name
                break
                
        if not matched_team:
            continue
            
        # 선수(Player) 매핑
        matched_player = None
        team_players = squads[matched_team]
        for p in team_players:
            p_name = p["name"]
            # 부분 일치 혹은 완전 일치 검사
            if p_name.lower() in player_raw.lower() or player_raw.lower() in p_name.lower():
                matched_player = p_name
                break
                
        if not matched_player:
            continue
            
        # 경기 수 및 정지 기간 산정
        N = get_actual_match_count(matched_team, actual_results)
        
        # 징계 기간 파싱 (기본 1경기, '2 matches' 등이 보이면 2경기)
        suspension_length = 1
        combined_text = (offense_raw + " " + suspension_raw).lower()
        if "2 matches" in combined_text or "two matches" in combined_text or "2경기" in combined_text:
            suspension_length = 2
        elif "3 matches" in combined_text or "three matches" in combined_text or "3경기" in combined_text:
            suspension_length = 3
            
        import re
        matchday_nums = [int(x) for x in re.findall(r"matchday\s*(\d+)", combined_text)]
        if matchday_nums:
            served_at_count = max(matchday_nums)
        else:
            served_at_count = N + suspension_length
            
        reason = "red_card" if "red" in combined_text or "퇴장" in combined_text else "yellow_cards"
        
        # injuries 데이터베이스에 징계 기록 추가
        if matched_team not in injuries:
            injuries[matched_team] = []
            
        # 이미 등록되어 있는지 체크 (이름 기준)
        already_exists = False
        for idx, item in enumerate(injuries[matched_team]):
            # 기존 레코드가 문자열인 경우와 딕셔너리인 경우 모두 대응
            item_name = item if isinstance(item, str) else item.get("name")
            if item_name == matched_player:
                already_exists = True
                # 기존이 단순 문자열이거나 징계가 만료된 경우 딕셔너리형으로 고도화/갱신
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
            print(f"[징계 동기화] {matched_team}의 {matched_player} 등록 완료 (출장정지 적용 실제경기수: {served_at_count}회차)")

    if added_count > 0:
        save_json(ABSENCES_PATH, injuries)
        print(f"[징계 동기화] 총 {added_count}명의 징계 선수가 데이터베이스에 동기화되었습니다.")
    else:
        print("[징계 동기화] 새로 추가된 징계 선수가 없습니다.")

if __name__ == "__main__":
    main()
