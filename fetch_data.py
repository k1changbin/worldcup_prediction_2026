import json
import os
import httpx

# 웹사이트 팀명과 프로젝트 데이터 팀명의 불일치 매핑 사전
TEAM_NAME_MAP = {
    "United States": "USA",
    "Turkey": "Türkiye",
}

def normalize_team_name(name, valid_teams):
    if not name:
        return None
    name = name.strip()
    
    # 1. 수동 매핑 사전 확인
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
        
    # 2. 대소문자 무관 완전 일치 확인
    for valid in valid_teams:
        if valid.lower() == name.lower():
            return valid
            
    # 3. 부분 일치 확인 (예: Curaçao 대응)
    for valid in valid_teams:
        if valid.lower() in name.lower() or name.lower() in valid.lower():
            return valid
            
    return None

def fetch_live_world_cup_data():
    elo_ratings_path = "data/elo_ratings.json"
    actual_results_path = "data/actual_results.json"
    
    if not os.path.exists(elo_ratings_path):
        print(f"에러: {elo_ratings_path} 파일을 찾을 수 없습니다. 프로젝트 루트에서 실행해 주세요.")
        return
        
    # 48개 참가국 정보 로드
    with open(elo_ratings_path, "r", encoding="utf-8") as f:
        local_ratings = json.load(f)
        valid_teams = set(local_ratings.keys())
        
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # 1. 팀명/코드 매핑 파일 다운로드
    print("[ELO] eloratings.net에서 팀명/코드 매핑 파일 다운로드 중...")
    try:
        r_teams = httpx.get("https://www.eloratings.net/en.teams.tsv", headers=headers, timeout=15.0)
        r_teams.raise_for_status()
    except Exception as e:
        print(f"팀 매핑 파일 요청 실패: {e}")
        return
        
    code_to_name = {}
    for line in r_teams.text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            code_to_name[parts[0]] = parts[1]
            
    # 2. ELO 레이팅 파일 다운로드 및 갱신
    print("[ELO] eloratings.net에서 전 세계 실시간 ELO 레이팅 파일 다운로드 중...")
    try:
        r_world = httpx.get("https://www.eloratings.net/World.tsv", headers=headers, timeout=15.0)
        r_world.raise_for_status()
    except Exception as e:
        print(f"ELO 레이팅 파일 요청 실패: {e}")
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
        
    print(f"[ELO] ELO 레이팅 갱신 완료: {ratings_updated_count}개국 ELO 반영")
    
    # 3. 2026 경기 결과(스코어) 다운로드 및 파싱
    print("[경기 결과] eloratings.net에서 2026년 경기 결과 데이터 다운로드 중...")
    try:
        r_results = httpx.get("https://www.eloratings.net/2026_results.tsv", headers=headers, timeout=15.0)
        r_results.raise_for_status()
    except Exception as e:
        print(f"경기 결과 파일 요청 실패: {e}")
        return
        
    actual_results = []
    seen_matchups = set()
    
    for line in r_results.text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 8:
            tournament_code = parts[7]
            # 월드컵 본선 경기(WC)만 추출
            if tournament_code != "WC":
                continue
                
            home_code = parts[3]
            away_code = parts[4]
            
            try:
                score_a = int(parts[5])
                score_b = int(parts[6])
            except ValueError:
                # 숫자가 아니면 경기 전으로 취급하고 패스
                continue
                
            home_name = code_to_name.get(home_code)
            away_name = code_to_name.get(away_code)
            
            home_team = normalize_team_name(home_name, valid_teams)
            away_team = normalize_team_name(away_name, valid_teams)
            
            if home_team and away_team and home_team != away_team:
                match_key = tuple(sorted([home_team, away_team]))
                if match_key in seen_matchups:
                    continue
                    
                seen_matchups.add(match_key)
                actual_results.append({
                    "team_a": home_team,
                    "team_b": away_team,
                    "score_a": score_a,
                    "score_b": score_b
                })
                
    with open(actual_results_path, "w", encoding="utf-8") as f:
        json.dump(actual_results, f, ensure_ascii=False, indent=2)
        
    print(f"경기 결과 갱신 완료: 총 {len(actual_results)}개의 종료된 경기 결과를 {actual_results_path}에 고정 저장했습니다.")
    for res in actual_results:
        print(f"   - {res['team_a']} {res['score_a']} : {res['score_b']} {res['team_b']}")

if __name__ == "__main__":
    fetch_live_world_cup_data()
