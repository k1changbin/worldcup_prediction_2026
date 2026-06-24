import json
import os
import re
import unicodedata
import httpx
from bs4 import BeautifulSoup
from src.absences import load_absences, save_absences

# ELO 국가명과 Wikipedia 헤딩명 간 매핑 사전
TEAM_ALIASES = {
    "United States": "USA",
    "Czech Republic": "Czechia",
    "Turkey": "Türkiye"
}

# 주요 에이스 선수 데이터베이스 (시장 가치 고정 입력용)
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
    # ELO 기반으로 스쿼드 총 가치 산출 (백만 유로 단위, 결정론적 수식 적용)
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

def fetch_live_injuries_and_squads():
    squads_path = "data/squads.json"
    injuries_path = "data/absences.json"
    elo_path = "data/elo_ratings.json"

    if not os.path.exists(elo_path):
        print(f"[에러] {elo_path} 파일이 존재하지 않습니다.")
        return

    with open(elo_path, "r", encoding="utf-8") as f:
        ratings = json.load(f)

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
    print(f"[스쿼드 & 부상자 수집] Wikipedia에서 데이터 수집 시작... ({url})")

    try:
        r = httpx.get(url, headers=headers, timeout=15.0)
        if r.status_code != 200:
            print(f"[에러] HTTP {r.status_code} 응답을 받았습니다.")
            return
    except Exception as e:
        print(f"[에러] Wikipedia 요청 실패: {e}")
        return

    soup = BeautifulSoup(r.text, "html.parser")
    headings = soup.find_all(["h2", "h3", "h4"])
    
    parsed_teams_data = {}
    live_injuries = {}
    total_players_count = 0
    total_injuries_count = 0

    print("Wikipedia 페이지 파싱 중...")

    for h in headings:
        heading_text = h.text.strip()
        normalized_heading = normalize_team(heading_text)

        # ELO 레이팅에 존재하는 국가인지 확인
        if normalized_heading not in ratings:
            continue

        # 헤딩 div로 감싸져 있는 구조 및 일반 구조 모두 대응하여 sibling 검색
        p = h.parent
        is_wrapped = p and p.name == "div" and any(cls.startswith("mw-heading") for cls in p.get("class", []))
        start_el = p if is_wrapped else h

        # 1. 조별 설명 단락 및 테이블 수집
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

        # 2. 스쿼드 테이블 파싱
        players_list = []
        if table:
            rows = table.find_all("tr")[1:]  # 헤더 제외
            for row in rows:
                cols = row.find_all(["th", "td"])
                if len(cols) < 7:
                    continue
                
                raw_no = cols[0].text.strip()
                raw_pos = cols[1].text.strip()
                raw_name = cols[2].text.strip()
                raw_caps = cols[4].text.strip()

                # 포지션 매핑
                pos_str = "Forward"
                if "GK" in raw_pos:
                    pos_str = "Goalkeeper"
                elif "DF" in raw_pos:
                    pos_str = "Defender"
                elif "MF" in raw_pos:
                    pos_str = "Midfielder"
                elif "FW" in raw_pos:
                    pos_str = "Forward"

                # 선수 이름 정제
                name_clean = clean_player_name(raw_name)

                # 출장 수(Caps) 정제 및 파싱
                caps_val = 0
                try:
                    caps_val = int(re.sub(r"\D", "", raw_caps))
                except ValueError:
                    pass

                players_list.append({
                    "name": name_clean,
                    "position": pos_str,
                    "caps": caps_val,
                    "value_eur": 0  # 임시값
                })

        if not players_list:
            print(f"   - [경고] {normalized_heading}의 스쿼드 테이블을 파싱하지 못했습니다.")
            continue

        # 3. 단락 내 부상/대체 정보 분석
        team_injured_names = []
        p_clean = re.sub(r"\[\d+\]", "", text_block)
        
        # 'withdrew' 및 'replaced by' 키워드가 들어있는 문장 탐색
        sentences = re.split(r"(?<=[.!?])\s+", p_clean)
        for sentence in sentences:
            if "withdrew" in sentence and "replaced by" in sentence:
                parts = sentence.split("withdrew")
                before = parts[0].strip()
                if "." in before:
                    before = before.split(".")[-1].strip()
                
                # 부상 결장 선수명 추출
                withdrawn_raw = re.split(r"\b(?:and|,)\b", before)
                withdrawn_players = [w.strip() for w in withdrawn_raw if w.strip()]

                # 대체 선수명 추출
                after = parts[1]
                replaced_players = []
                if "replaced by" in after:
                    after_part = after.split("replaced by")[1].strip()
                    after_clean = re.split(r"\b(?:on|at|with|due|in|respectively)\b|[,.]", after_part)[0].strip()
                    replaced_raw = re.split(r"\b(?:and|,)\b", after_clean)
                    replaced_players = [r.strip() for r in replaced_raw if r.strip()]

                # 매칭 및 스쿼드 복원 처리
                # (스쿼드 테이블에 있는 대체 선수를 부상 선수로 대체하여, 시뮬레이션에서 가치 손실을 계산하도록 함)
                for w_name, r_name in zip(withdrawn_players, replaced_players):
                    w_norm = normalize_for_matching(w_name)
                    r_norm = normalize_for_matching(r_name)

                    # 테이블에서 대체 선수 탐색
                    matched_idx = -1
                    for idx, p_item in enumerate(players_list):
                        if normalize_for_matching(p_item["name"]) == r_norm:
                            matched_idx = idx
                            break

                    if matched_idx != -1:
                        # 복원: 대체선수를 부상당한 원조 선수로 덮어쓰기
                        players_list[matched_idx]["name"] = w_name
                        team_injured_names.append(w_name)
                    else:
                        # 대체 선수가 테이블에서 확인되지 않는 경우, 그냥 부상자 명단에만 우선 추가
                        team_injured_names.append(w_name)

        parsed_teams_data[normalized_heading] = players_list
        total_players_count += len(players_list)

        if team_injured_names:
            live_injuries[normalized_heading] = team_injured_names
            total_injuries_count += len(team_injured_names)
            print(f"   - {normalized_heading}: {len(players_list)}명 파싱 완료 (부상 이탈: {team_injured_names})")
        else:
            print(f"   - {normalized_heading}: {len(players_list)}명 파싱 완료")

    # 4. 각 팀별 ELO 레이팅 기반 및 에이스 수동 지정을 통한 선수 가치(Market Value) 동적 책정
    final_squads = {}
    for team, players in parsed_teams_data.items():
        rating = ratings.get(team, 1700.0)
        total_squad_value = get_deterministic_squad_value(rating)

        stars = STAR_PLAYERS.get(team, [])
        star_values_sum = 0
        star_names_lower = {}
        for s in stars:
            star_names_lower[normalize_for_matching(s["name"])] = s

        # 스타 플레이어 값 우선 적용
        assigned_players = []
        regular_players = []

        for p in players:
            p_norm = normalize_for_matching(p["name"])
            if p_norm in star_names_lower:
                star_data = star_names_lower[p_norm]
                p["value_eur"] = star_data["value"]
                # 만약 포지션 불일치가 있으면 고침
                p["position"] = star_data["position"]
                assigned_players.append(p)
                star_values_sum += star_data["value"]
            else:
                regular_players.append(p)

        # 잔여 가치 분배
        remaining_value = max(total_squad_value * 0.2, total_squad_value - star_values_sum)
        
        # 일반 선수들은 Caps(출장수) 기준 내림차순 정렬하여 Pareto(지프) 분포로 가치 배분
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

        # 원래 인덱스 순으로 정렬하거나 그대로 저장
        final_squads[team] = assigned_players

    # 5. 파일 저장
    # data 폴더가 없을 경우 생성
    os.makedirs(os.path.dirname(squads_path), exist_ok=True)

    with open(squads_path, "w", encoding="utf-8") as f:
        json.dump(final_squads, f, ensure_ascii=False, indent=2)

    absences_data = load_absences(injuries_path)
    preserved_suspensions = {}
    for team, items in absences_data.items():
        suspensions = [
            item
            for item in items
            if isinstance(item, dict) and item.get("type") == "suspension"
        ]
        if suspensions:
            preserved_suspensions[team] = suspensions

    canonical_absences = preserved_suspensions
    for team, players in live_injuries.items():
        canonical_absences.setdefault(team, [])
        canonical_absences[team].extend(players)

    save_absences(injuries_path, canonical_absences)

    print(f"\n[성공] 데이터 업데이트 완료!")
    print(f"   - 스쿼드 데이터 저장 경로: {squads_path} (총 {len(final_squads)}개국, {total_players_count}명)")
    print(f"   - 부상자 데이터 저장 경로: {injuries_path} (총 {len(live_injuries)}개국, {total_injuries_count}명)")

if __name__ == "__main__":
    fetch_live_injuries_and_squads()
