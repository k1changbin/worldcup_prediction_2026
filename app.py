import streamlit as st
import json
import os
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from src.elo import EloSystem
from src.poisson import win_prob_to_lambda, match_probabilities, simulate_match_score
from src.simulation import WorldCupSimulation, HOST_COUNTRIES, GROUP_REGIONS

# 페이지 기본 설정
st.set_page_config(
    page_title="2026 FIFA 월드컵 AI 예측 대시보드",
    page_icon="cup",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 도시별 기후/지리적 권역 매핑 (simulation.py의 KNOCKOUT_MATCH_INFO 및 피로도 산출용)
CITY_REGIONS = {
    "los-angeles": 1, "san-francisco": 1, "seattle": 1, "vancouver": 1,
    "guadalajara": 2, "mexico-city": 2, "monterrey": 2,
    "dallas": 3, "houston": 3, "kansas-city": 3,
    "atlanta": 4, "miami": 4,
    "boston": 5, "new-york": 5, "philadelphia": 5, "toronto": 5
}

# 데이터 경로 정의
ELO_PATH = "data/elo_ratings.json"
SQUADS_PATH = "data/squads.json"
ABSENCES_PATH = "data/absences.json"
SCHEDULE_PATH = "data/schedule.json"
ACTUAL_RESULTS_PATH = "data/actual_results.json"
GROUPS_PATH = "data/groups.json"

# 국가별 국기 이모티콘 매핑 사전
FLAG_MAP = {
    "Mexico": "🇲🇽",
    "South Korea": "🇰🇷",
    "South Africa": "🇿🇦",
    "Czechia": "🇨🇿",
    "Canada": "🇨🇦",
    "Switzerland": "🇨🇭",
    "Qatar": "🇶🇦",
    "Bosnia and Herzegovina": "🇧🇦",
    "Brazil": "🇧🇷",
    "Morocco": "🇲🇦",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Haiti": "🇭🇹",
    "USA": "🇺🇸",
    "Australia": "🇦🇺",
    "Paraguay": "🇵🇾",
    "Türkiye": "🇹🇷",
    "Germany": "🇩🇪",
    "Ecuador": "🇪🇨",
    "Ivory Coast": "🇨🇮",
    "Curaçao": "🇨🇼",
    "Netherlands": "🇳🇱",
    "Japan": "🇯🇵",
    "Tunisia": "🇹🇳",
    "Sweden": "🇸🇪",
    "Belgium": "🇧🇪",
    "Iran": "🇮🇷",
    "Egypt": "🇪🇬",
    "New Zealand": "🇳🇿",
    "Spain": "🇪🇸",
    "Uruguay": "🇺🇾",
    "Saudi Arabia": "🇸🇦",
    "Cape Verde": "🇨🇻",
    "France": "🇫🇷",
    "Senegal": "🇸🇳",
    "Norway": "🇳🇴",
    "Iraq": "🇮🇶",
    "Argentina": "🇦🇷",
    "Austria": "🇦🇹",
    "Algeria": "🇩🇿",
    "Jordan": "🇯🇴",
    "Portugal": "🇵🇹",
    "Colombia": "🇨🇴",
    "Uzbekistan": "🇺🇿",
    "DR Congo": "🇨🇩",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Croatia": "🇭🇷",
    "Panama": "🇵🇦",
    "Ghana": "🇬🇭"
}

def get_flag(team_name):
    return FLAG_MAP.get(team_name, "")

@st.cache_data
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

def clean_served_suspensions(injuries_dict, actual_results_list):
    updated = False
    for team, players_list in list(injuries_dict.items()):
        new_list = []
        for p in players_list:
            if isinstance(p, dict) and p.get("type") == "suspension":
                # Count actual matches played by this team
                N = 0
                for match in actual_results_list:
                    if match.get("team_a") == team or match.get("team_b") == team:
                        N += 1
                
                served_at = p.get("served_at_count", 0)
                if N >= served_at:
                    # Suspension served! Mark updated to save
                    updated = True
                    continue # Skip adding it to new list (removes it)
            new_list.append(p)
        
        if len(new_list) != len(players_list):
            updated = True
            injuries_dict[team] = new_list
            if not new_list:
                del injuries_dict[team]
                
    if updated:
        save_json(ABSENCES_PATH, injuries_dict)
    return injuries_dict

def get_absence_names(raw_list):
    names = []
    if isinstance(raw_list, list):
        for item in raw_list:
            if isinstance(item, dict):
                names.append(item.get("name"))
            else:
                names.append(str(item))
    return [n for n in names if n]

def format_absence_list_to_str_list(raw_list):
    res = []
    if isinstance(raw_list, list):
        for p in raw_list:
            if isinstance(p, dict):
                p_name = p.get("name")
                p_type = p.get("type", "injury")
                if p_type == "suspension":
                    reason_map = {
                        "red_card": "퇴장 징계",
                        "yellow_cards": "경고 누적 징계"
                    }
                    reason_text = reason_map.get(p.get("reason"), "출장 정지")
                    res.append(f"{p_name} ({reason_text})")
                else:
                    res.append(f"{p_name} (부상)")
            else:
                res.append(f"{p} (부상)")
    return res

# 데이터 로드
elo_ratings = load_json(ELO_PATH)
squads = load_json(SQUADS_PATH)
injuries_raw = load_json(ABSENCES_PATH)
schedule = load_json(SCHEDULE_PATH)
actual_results = load_json(ACTUAL_RESULTS_PATH)
groups_dict = load_json(GROUPS_PATH)

# 자동 복귀 처리 실행
injuries_raw = clean_served_suspensions(injuries_raw, actual_results)

# 데이터 유효성 검증
if not elo_ratings or not groups_dict:
    st.error("데이터 파일(data/elo_ratings.json, data/groups.json)이 누락되었거나 비어 있습니다. 프로젝트 설정을 확인하세요.")
    st.stop()

# ----------------- 부상 선수 관리 로직 (사이드바 연동을 위한 도우미) -----------------
def get_injury_multipliers(team, injuries_dict):
    raw_injuries = injuries_dict.get(team, [])
    team_injuries = []
    if isinstance(raw_injuries, list):
        for item in raw_injuries:
            if isinstance(item, dict):
                team_injuries.append(item.get("name"))
            else:
                team_injuries.append(str(item))
    else:
        team_injuries = [str(raw_injuries)]
    team_injuries = [n for n in team_injuries if n]

    if not team_injuries or team not in squads:
        return 1.0, 1.0, []

    players = squads[team]
    total_value = sum(p["value_eur"] for p in players)
    if total_value == 0:
        return 1.0, 1.0, []

    hhi = sum((p["value_eur"] / total_value) ** 2 for p in players)
    min_hhi = 0.0385
    max_hhi = 0.3000
    norm_hhi = max(0.0, min(1.0, (hhi - min_hhi) / (max_hhi - min_hhi)))
    depth_factor = 0.2 + 0.8 * norm_hhi

    attack_total = sum(p["value_eur"] for p in players if p["position"] not in ["Goalkeeper", "Defender"])
    defense_total = sum(p["value_eur"] for p in players if p["position"] in ["Goalkeeper", "Defender"])

    attack_reduction = 0.0
    defense_reduction = 0.0
    details = []

    for player_name in team_injuries:
        matched_player = None
        for p in players:
            if p["name"].strip().lower() == player_name.strip().lower():
                matched_player = p
                break
        
        if matched_player:
            pos = matched_player["position"]
            val = matched_player["value_eur"]
            pos_str = "수비" if pos in ["Goalkeeper", "Defender"] else "공격"
            
            if pos in ["Goalkeeper", "Defender"]:
                if defense_total > 0:
                    share = val / defense_total
                    reduction = share * depth_factor
                    defense_reduction += reduction
                    details.append(f"{matched_player['name']} ({pos_str}, 누수율: {reduction*100:.1f}%)")
            else:
                if attack_total > 0:
                    share = val / attack_total
                    reduction = share * depth_factor
                    attack_reduction += reduction
                    details.append(f"{matched_player['name']} ({pos_str}, 누수율: {reduction*100:.1f}%)")

    attack_multiplier = max(0.5, 1.0 - attack_reduction)
    defense_multiplier = min(2.0, 1.0 + defense_reduction)
    return attack_multiplier, defense_multiplier, details

# ----------------- 전체 일정 기반 피로도/휴식일 상태 동적 계산 -----------------
def compute_schedule_states(schedule_list):
    def get_day_num(date_str):
        parts = date_str.split("-")
        month = int(parts[1])
        day = int(parts[2])
        return day if month == 6 else 30 + day

    team_states = {}
    for grp, teams in groups_dict.items():
        group_letter = grp.split(" ")[1]
        reg = GROUP_REGIONS.get(group_letter, 3)
        for t in teams:
            team_states[t] = {"last_date": None, "last_region": reg}

    match_states = {}
    for match in sorted(schedule_list, key=lambda x: x["matchNumber"]):
        m_num = match["matchNumber"]
        m_date = match["date"]
        day_num = get_day_num(m_date)
        city = match["hostCity"]
        region = CITY_REGIONS.get(city, 3)
        
        home = match["homeTeam"]
        away = match["awayTeam"]
        
        # 홈팀 계산
        rest_home = 0
        fatigue_home = 0.0
        if home in team_states and team_states[home]["last_date"] is not None:
            prev_date = team_states[home]["last_date"]
            prev_reg = team_states[home]["last_region"]
            rest_home = day_num - prev_date
            diff = abs(region - prev_reg)
            fatigue_home = 0.03 if diff >= 3 else (0.015 if diff > 0 else 0.0)
            
        # 원정팀 계산
        rest_away = 0
        fatigue_away = 0.0
        if away in team_states and team_states[away]["last_date"] is not None:
            prev_date = team_states[away]["last_date"]
            prev_reg = team_states[away]["last_region"]
            rest_away = day_num - prev_date
            diff = abs(region - prev_reg)
            fatigue_away = 0.03 if diff >= 3 else (0.015 if diff > 0 else 0.0)

        rest_days_diff = 0
        if (home in team_states and team_states[home]["last_date"] is not None and 
            away in team_states and team_states[away]["last_date"] is not None):
            rest_days_diff = rest_home - rest_away

        match_states[m_num] = {
            "rest_days_diff": rest_days_diff,
            "travel_fatigue_a": fatigue_home,
            "travel_fatigue_b": fatigue_away
        }

        # 상태 업데이트 (예측 매치더라도 일정 진행에 맞춰 위치 이동 반영)
        if home in team_states:
            team_states[home] = {"last_date": day_num, "last_region": region}
        if away in team_states:
            team_states[away] = {"last_date": day_num, "last_region": region}

    return match_states

match_states_cache = compute_schedule_states(schedule)

# ----------------- 사이드바 (부상/결장 관리 패널) -----------------
st.sidebar.markdown("## 실시간 부상 및 징계 관리")
st.sidebar.info("선수의 부상 또는 징계(출장정지) 상태를 등록하면 전체 예측 모델에 실시간으로 반영됩니다.")

# Wikipedia 동기화 버튼
import fetch_suspensions
if st.sidebar.button("실시간 징계 정보 동기화 (Wikipedia)"):
    with st.sidebar.spinner("Wikipedia에서 징계 기록 수집 중..."):
        fetch_suspensions.main()
    st.sidebar.success("동기화가 완료되었습니다.")
    st.rerun()

# 현재 부상 및 징계 명단 출력
st.sidebar.markdown("### 현재 결장 선수 목록")
active_injuries = load_json(ABSENCES_PATH)

# 자동 복귀 처리 재실행 (안전 장치)
active_injuries = clean_served_suspensions(active_injuries, actual_results)

if not active_injuries or all(len(v) == 0 for v in active_injuries.values()):
    st.sidebar.text("등록된 결장 선수가 없습니다.")
else:
    for team, players_list in list(active_injuries.items()):
        if not players_list:
            continue
        st.sidebar.markdown(f"**{get_flag(team)} {team}**")
        for p in players_list:
            if isinstance(p, dict):
                p_name = p.get("name")
                p_type = p.get("type", "injury")
                if p_type == "suspension":
                    reason_map = {
                        "red_card": "퇴장 징계",
                        "yellow_cards": "경고 누적 징계"
                    }
                    reason_text = reason_map.get(p.get("reason"), "출장 정지")
                    st.sidebar.text(f"  • {p_name} ({reason_text}, {p.get('served_at_count')}회차 복귀)")
                else:
                    st.sidebar.text(f"  • {p_name} (부상)")
            else:
                st.sidebar.text(f"  • {p} (부상)")

st.sidebar.markdown("---")
st.sidebar.markdown("### 새 결장 선수 등록")

# 국가 선택
all_teams = sorted(list(elo_ratings.keys()))
selected_team = st.sidebar.selectbox("국가 선택", all_teams, format_func=lambda x: f"{get_flag(x)} {x}")

# 선수 선택
if selected_team in squads:
    team_players = [p["name"] for p in squads[selected_team]]
    # 현재 이미 결장 등록되어 있는 선수 제외
    already_injured_names = get_absence_names(active_injuries.get(selected_team, []))
    available_players = [p for p in team_players if p not in already_injured_names]
    
    if available_players:
        selected_player = st.sidebar.selectbox("선수 선택", available_players)
        absence_reason = st.sidebar.selectbox("결장 사유", ["부상 (Injury)", "퇴장 (1경기 정지)", "경고 누적 (1경기 정지)", "추가 징계 (2경기 정지)"])
        
        if st.sidebar.button("결장 목록에 추가"):
            if selected_team not in active_injuries:
                active_injuries[selected_team] = []
                
            if "부상" in absence_reason:
                active_injuries[selected_team].append({
                    "name": selected_player,
                    "type": "injury"
                })
            else:
                # 징계 등록
                N = 0
                for match in actual_results:
                    if match.get("team_a") == selected_team or match.get("team_b") == selected_team:
                        N += 1
                
                suspension_length = 2 if "2경기" in absence_reason else 1
                served_at = N + suspension_length
                reason = "red_card" if "퇴장" in absence_reason else "yellow_cards"
                
                active_injuries[selected_team].append({
                    "name": selected_player,
                    "type": "suspension",
                    "reason": reason,
                    "served_at_count": served_at
                })
                
            save_json(ABSENCES_PATH, active_injuries)
            st.rerun()
    else:
        st.sidebar.warning("이 팀의 모든 선수가 결장 상태이거나 선택 가능한 선수가 없습니다.")
else:
    st.sidebar.warning(f"{selected_team}의 스쿼드 데이터가 존재하지 않습니다.")

# 결장 복귀 처리 (Multiselect)
st.sidebar.markdown("---")
st.sidebar.markdown("### 결장 선수 복귀")

# 결장 등록된 선수들을 드롭다운 리스트용 문자열로 수집
injured_options = []
for team, players_list in active_injuries.items():
    for p in players_list:
        p_name = p.get("name") if isinstance(p, dict) else p
        injured_options.append(f"{get_flag(team)} {team} - {p_name}")

if injured_options:
    selected_to_recover = st.sidebar.multiselect("복귀할 선수 선택", injured_options)
    if st.sidebar.button("선택 선수 복귀 처리"):
        for opt in selected_to_recover:
            # 매칭되는 팀과 선수 찾아 삭제
            for team, players_list in list(active_injuries.items()):
                for p in list(players_list):
                    p_name = p.get("name") if isinstance(p, dict) else p
                    if f"{get_flag(team)} {team} - {p_name}" == opt:
                        players_list.remove(p)
                if not active_injuries[team]:
                    del active_injuries[team]
        save_json(ABSENCES_PATH, active_injuries)
        st.rerun()
else:
    st.sidebar.text("복귀 처리할 결장 선수가 없습니다.")


# ----------------- 메인 대시보드 화면 구성 -----------------
st.title("2026 FIFA 월드컵 AI 실시간 예측 대시보드")
st.markdown("본 대시보드는 실시간 부상 상태, ELO 레이팅, 경기 간 휴식 일정 및 대륙 이동 피로도를 Poisson 확률 공식에 대입하여 경기 결과를 정밀 예측합니다.")

# 메인 탭 구조 설정
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "경기 일정 및 승무패 예측", 
    "현재 리그 진행도", 
    "토너먼트 대진표",
    "1 대 1 가상 매치 시뮬레이터",
    "전체 예측"
])

# ----------------- 탭 1: 경기 일정 및 승무패 예측 -----------------
with tab1:
    st.header("전체 104경기 날짜별 일정 및 예측")
    st.caption("※ 모든 경기 일정 및 날짜는 경기가 치러지는 개최 도시의 **현지 날짜 기준**으로 표시됩니다.")
    
    # 실제 치러진 경기 캐싱용 매핑 구축
    actual_map = {}
    for res in actual_results:
        key = frozenset([res["team_a"], res["team_b"]])
        actual_map[key] = res

    # 일정 필터링 옵션
    import datetime
    today = datetime.date.today()
    start_wc = datetime.date(2026, 6, 11)
    end_wc = datetime.date(2026, 7, 19)
    
    # 오늘이 월드컵 기간 중이라면 오늘 경기를 디폴트로 보여주기 위해 토글을 끔(False), 아니면 전체보기(True)
    is_during_wc = start_wc <= today <= end_wc
    default_view_all = not is_during_wc

    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        stage_filter = st.selectbox("대회 단계", ["전체보기", "조별리그", "토너먼트"])
    with col_f2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        view_all = st.toggle("모든 날짜 보기", value=default_view_all)

    # 선택된 대회 단계에 맞는 기본 경기 리스트 추출
    if stage_filter == "조별리그":
        stage_matches = [f for f in schedule if f["stage"] == "group-stage"]
    elif stage_filter == "토너먼트":
        stage_matches = [f for f in schedule if f["stage"] != "group-stage"]
    else:
        stage_matches = schedule

    if not view_all:
        dates_str = sorted(list(set(f["date"] for f in stage_matches)))
        if dates_str:
            dates_obj = [datetime.date.fromisoformat(d) for d in dates_str]
            # 오늘 날짜가 해당 단계 기간 내에 있으면 오늘 날짜를 디폴트로, 없으면 첫 경기 날짜를 디폴트로 설정
            default_date = today if dates_obj[0] <= today <= dates_obj[-1] else dates_obj[0]
            selected_date = st.date_input(
                "달력에서 날짜 선택",
                value=default_date,
                min_value=dates_obj[0],
                max_value=dates_obj[-1]
            )
            date_filter = selected_date.strftime("%Y-%m-%d")
        else:
            date_filter = "전체 일자"
    else:
        date_filter = "전체 일자"

    # 필터링 적용
    filtered_schedule = stage_matches
    if date_filter != "전체 일자":
        filtered_schedule = [f for f in filtered_schedule if f["date"] == date_filter]

    st.markdown(f"**총 {len(filtered_schedule)}개의 경기가 매칭되었습니다.**")
    if len(filtered_schedule) == 0 and date_filter != "전체 일자":
        st.info("선택하신 날짜에는 예정된 월드컵 경기가 없습니다. 달력에서 다른 날짜를 선택해 주세요.")
    
    # 매치별 카드 리스트 렌더링
    for match in filtered_schedule:
        m_num = match["matchNumber"]
        m_date = match["date"]
        stage = "조별리그" if match["stage"] == "group-stage" else "토너먼트"
        group_info = f"Group {match['group']}" if match["group"] else ""
        stadium = match["stadium"]
        city = match["hostCity"].replace("-", " ").title()
        
        home = match["homeTeam"]
        away = match["awayTeam"]

        # 실제 이미 치러진 경기 여부 확인
        match_key = frozenset([home, away])
        has_actual = match_key in actual_map
        
        with st.container():
            st.markdown(
                f"""
                <div style="background-color: #1e293b; padding: 15px; border-radius: 10px; margin-bottom: 15px; border: 1px solid #334155;">
                    <span style="color: #38bdf8; font-weight: bold;">[Match {m_num}]</span> 
                    <span style="color: #94a3b8; font-size: 0.9em;"> {m_date} | {stage} {group_info} | {stadium} ({city})</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            c1, c2, c3 = st.columns([2, 5, 2])
            
            with c1:
                st.markdown(f"<div style='font-size: 1.35em; font-weight: bold; text-align: right; margin-bottom: 2px;'>{get_flag(home)} {home}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: right; color: #94a3b8; font-size: 0.9em;'>ELO: {elo_ratings.get(home, 1500.0):.0f}</div>", unsafe_allow_html=True)
                # 부상자 출력
                team_inj = active_injuries.get(home, [])
                if team_inj:
                    formatted_inj = format_absence_list_to_str_list(team_inj)
                    st.markdown(f"<div style='text-align: right; color: #f43f5e; font-size: 0.8em; margin-top: 2px;'>결장: {', '.join(formatted_inj)}</div>", unsafe_allow_html=True)
            
            with c3:
                st.markdown(f"<div style='font-size: 1.35em; font-weight: bold; text-align: left; margin-bottom: 2px;'>{get_flag(away)} {away}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: left; color: #94a3b8; font-size: 0.9em;'>ELO: {elo_ratings.get(away, 1500.0):.0f}</div>", unsafe_allow_html=True)
                # 부상자 출력
                team_inj = active_injuries.get(away, [])
                if team_inj:
                    formatted_inj = format_absence_list_to_str_list(team_inj)
                    st.markdown(f"<div style='text-align: left; color: #f43f5e; font-size: 0.8em; margin-top: 2px;'>결장: {', '.join(formatted_inj)}</div>", unsafe_allow_html=True)

            with c2:
                if has_actual:
                    # 실제 경기 결과 표시
                    actual = actual_map[match_key]
                    s_home = actual["score_a"] if actual["team_a"] == home else actual["score_b"]
                    s_away = actual["score_b"] if actual["team_a"] == home else actual["score_a"]
                    w_team = actual["winner"]
                    winner_text = f"승자: {get_flag(w_team)} {w_team}" if w_team else "무승부"
                    st.markdown(
                        f"""
                        <div style='text-align: center;'>
                            <div style='color: #10b981; font-size: 1.8em; font-weight: bold; margin-bottom: 2px;'>{s_home} : {s_away}</div>
                            <p style='color: #64748b; font-size: 0.85em; font-weight: bold; margin: 0;'>실제 경기 완료 ({winner_text})</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    # ELO 예측 확률 계산
                    r_home = elo_ratings.get(home, 1700.0)
                    r_away = elo_ratings.get(away, 1700.0)
                    
                    # 1. 홈 우위 적용
                    if home in HOST_COUNTRIES and away not in HOST_COUNTRIES:
                        r_home += 40
                    elif away in HOST_COUNTRIES and home not in HOST_COUNTRIES:
                        r_away += 40
                        
                    # 2. 휴식일 보정
                    states = match_states_cache.get(m_num, {"rest_days_diff": 0, "travel_fatigue_a": 0.0, "travel_fatigue_b": 0.0})
                    rest_diff = states["rest_days_diff"]
                    rest_bonus = min(abs(rest_diff) * 5, 30)
                    if rest_diff >= 1:
                        r_home += rest_bonus
                    elif rest_diff <= -1:
                        r_away += rest_bonus
                        
                    # 3. Poisson 람다 변환
                    elo = EloSystem()
                    expected_home = elo.expected_score(r_home, r_away)
                    l_home, l_away = win_prob_to_lambda(expected_home)
                    
                    # 4. 부상 보정 및 5. 이동 피로도 적용
                    att_m_home, def_m_home, _ = get_injury_multipliers(home, active_injuries)
                    att_m_away, def_m_away, _ = get_injury_multipliers(away, active_injuries)
                    fat_home = states["travel_fatigue_a"]
                    fat_away = states["travel_fatigue_b"]
                    
                    final_l_home = l_home * att_m_home * def_m_away * (1.0 - fat_home)
                    final_l_away = l_away * att_m_away * def_m_home * (1.0 - fat_away)
                    
                    # 승률 분석
                    prob = match_probabilities(final_l_home, final_l_away)
                    p_win = prob["win"] * 100
                    p_draw = prob["draw"] * 100
                    p_lose = prob["lose"] * 100

                    # 1,000,000회 시뮬레이션을 통해 최빈 예상 스코어 도출
                    sa_s = np.random.poisson(final_l_home, 10000)
                    sb_s = np.random.poisson(final_l_away, 10000)
                    sc_counts = Counter(zip(sa_s, sb_s))
                    (b_sa, b_sb), _ = max(sc_counts.items(), key=lambda x: x[1])

                    # 게이지바 디자인
                    st.markdown(
                        f"""
                        <div style='text-align: center; margin-bottom: 5px; font-weight: bold;'>
                            AI 예상: {get_flag(home)} {home} 승 {p_win:.1f}% | 무승부 {p_draw:.1f}% | {get_flag(away)} {away} 승 {p_lose:.1f}%
                        </div>
                        <div style="display: flex; height: 20px; border-radius: 10px; overflow: hidden; border: 1px solid #475569; margin-bottom: 5px;">
                            <div style="width: {p_win}%; background-color: #0ea5e9; text-align: center; color: white; font-size: 0.8em; line-height: 18px;">{p_win:.0f}%</div>
                            <div style="width: {p_draw}%; background-color: #64748b; text-align: center; color: white; font-size: 0.8em; line-height: 18px;">{p_draw:.0f}%</div>
                            <div style="width: {p_lose}%; background-color: #ec4899; text-align: center; color: white; font-size: 0.8em; line-height: 18px;">{p_lose:.0f}%</div>
                        </div>
                        <div style='text-align: center; color: #38bdf8; font-size: 0.85em; font-weight: bold;'>
                            최빈 예상 스코어: {b_sa} - {b_sb}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        st.markdown("<hr style='border: 0.5px solid #334155; margin-top:20px; margin-bottom:20px;'>", unsafe_allow_html=True)


# ----------------- 탭 2: 현재 리그 진행도 -----------------
with tab2:
    st.header("조별 리그 실제 진행도 및 현재 순위")
    st.markdown("현재까지 완료된 실제 경기 결과를 바탕으로 집계된 조별 리그 순위표입니다.")
    
    # 실제 경기 결과 및 그룹 데이터 로드
    groups = load_json(GROUPS_PATH)
    actual_results = load_json(ACTUAL_RESULTS_PATH)
    
    # 각 팀별 실제 성적 초기화
    standings = {}
    for group_name, teams in groups.items():
        standings[group_name] = {
            team: {"pts": 0, "w": 0, "d": 0, "l": 0, "gd": 0, "gf": 0, "ga": 0}
            for team in teams
        }
    
    # 실제 완료된 경기 집계
    for match in actual_results:
        if match.get("stage", "group") == "group":
            team_a = match["team_a"]
            team_b = match["team_b"]
            score_a = match["score_a"]
            score_b = match["score_b"]
            
            # 어느 조인지 찾기
            g_name_a = None
            for g_name, teams in groups.items():
                if team_a in teams:
                    g_name_a = g_name
                    break
            
            g_name_b = None
            for g_name, teams in groups.items():
                if team_b in teams:
                    g_name_b = g_name
                    break
            
            if g_name_a and g_name_b and g_name_a == g_name_b:
                group_name = g_name_a
                # A팀 업데이트
                standings[group_name][team_a]["gf"] += score_a
                standings[group_name][team_a]["ga"] += score_b
                standings[group_name][team_a]["gd"] += (score_a - score_b)
                
                # B팀 업데이트
                standings[group_name][team_b]["gf"] += score_b
                standings[group_name][team_b]["ga"] += score_a
                standings[group_name][team_b]["gd"] += (score_b - score_a)
                
                if score_a > score_b:
                    standings[group_name][team_a]["pts"] += 3
                    standings[group_name][team_a]["w"] += 1
                    standings[group_name][team_b]["l"] += 1
                elif score_a < score_b:
                    standings[group_name][team_b]["pts"] += 3
                    standings[group_name][team_b]["w"] += 1
                    standings[group_name][team_a]["l"] += 1
                else:
                    standings[group_name][team_a]["pts"] += 1
                    standings[group_name][team_a]["d"] += 1
                    standings[group_name][team_b]["pts"] += 1
                    standings[group_name][team_b]["d"] += 1

    # 각 조별로 정렬 및 시각화
    groups_list = sorted(list(groups.keys()))
    
    for row_idx in range(4): # 4개 행 (행마다 3개 조)
        cols = st.columns(3)
        for col_idx in range(3):
            g_idx = row_idx * 3 + col_idx
            if g_idx < len(groups_list):
                group_name = groups_list[g_idx]
                teams_dict = standings[group_name]
                
                # 정렬 기준: pts -> gd -> gf
                sorted_teams = sorted(
                    teams_dict.items(),
                    key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"]),
                    reverse=True
                )
                
                # 데이터프레임용 데이터 구축
                rows = []
                for rank, (team, stats) in enumerate(sorted_teams, 1):
                    rows.append({
                        "순위": rank,
                        "팀": f"{get_flag(team)} {team}",
                        "승점": stats["pts"],
                        "승": stats["w"],
                        "무": stats["d"],
                        "패": stats["l"],
                        "득실": stats["gd"],
                        "득": stats["gf"]
                    })
                df_g = pd.DataFrame(rows)
                
                with cols[col_idx]:
                    st.subheader(group_name)
                    st.dataframe(
                        df_g.style.hide(axis="index"),
                        width="stretch",
                        height=180
                    )



# ----------------- 탭 3: 토너먼트 대진표 시각화 -----------------
with tab3:
    st.header("가상 월드컵 토너먼트 대진표 시각화")
    st.markdown("현재 부상 선수 명단과 ELO 레이팅이 반영된 1회성 토너먼트 시뮬레이션을 실행하여, 32강부터 결승전 및 우승팀까지 이어지는 대진표를 한 화면에 트리 형태로 확인합니다.")
    
    if st.button("가상 토너먼트 시뮬레이션 및 대진표 생성"):
        with st.spinner("가상 토너먼트 매치 연산 중..."):
            # ELO 객체 초기화
            elo_sys = EloSystem()
            elo_sys.load_ratings(ELO_PATH)
            
            # 시뮬레이터 실행
            sim = WorldCupSimulation(
                elo_system=elo_sys,
                groups_file=GROUPS_PATH,
                actual_results_file=ACTUAL_RESULTS_PATH,
                absences_file=ABSENCES_PATH,
                squads_file=SQUADS_PATH
            )
            
            standings = sim.simulate_group_stage()
            r32_teams = sim.get_advancing_teams(standings)
            ko_results = sim.simulate_knockout_stage(r32_teams)
            
            r32 = ko_results["Round of 32"]
            r16 = ko_results["Round of 16"]
            qf = ko_results["Quarter-finals"]
            sf = ko_results["Semi-finals"]
            final = ko_results["Final"]
            champion = ko_results["Champion"]
            
            def gen_card_html(m, round_name, m_id):
                win_a = "winner" if m["winner"] == m["team_a"] else ""
                win_b = "winner" if m["winner"] == m["team_b"] else ""
                pk_a = " (PK)" if m["is_pk"] and m["winner"] == m["team_a"] else ""
                pk_b = " (PK)" if m["is_pk"] and m["winner"] == m["team_b"] else ""
                
                # 모달에서 사용할 양팀 상세 데이터 산출
                team_a = m["team_a"]
                team_b = m["team_b"]
                r_a = elo_ratings.get(team_a, 1500.0)
                r_b = elo_ratings.get(team_b, 1500.0)
                
                # 부상 전력 누수 산출
                att_m_a, def_m_a, _ = get_injury_multipliers(team_a, active_injuries)
                att_m_b, def_m_b, _ = get_injury_multipliers(team_b, active_injuries)
                
                # 개최국 홈 우위 버프 적용
                r_a_adj = r_a + (40 if team_a in HOST_COUNTRIES else 0)
                r_b_adj = r_b + (40 if team_b in HOST_COUNTRIES else 0)
                
                # ELO 기반 예상 득점(람다) 산출
                elo_sys = EloSystem()
                expected_a = elo_sys.expected_score(r_a_adj, r_b_adj)
                l_a, l_b = win_prob_to_lambda(expected_a)
                
                final_l_a = l_a * att_m_a * def_m_b
                final_l_b = l_b * att_m_b * def_m_a
                
                # 승무패 확률 계산
                prob = match_probabilities(final_l_a, final_l_b)
                p_win = prob["win"] * 100
                p_draw = prob["draw"] * 100
                p_lose = prob["lose"] * 100
                
                # 부상 명단 가공
                inj_a_str = ",".join(format_absence_list_to_str_list(active_injuries.get(team_a, [])))
                inj_b_str = ",".join(format_absence_list_to_str_list(active_injuries.get(team_b, [])))
                
                # JS 파라미터 따옴표 이스케이프 처리
                t_a_esc = team_a.replace("'", "\\'")
                t_b_esc = team_b.replace("'", "\\'")
                inj_a_esc = inj_a_str.replace("'", "\\'")
                inj_b_esc = inj_b_str.replace("'", "\\'")
                w_esc = m["winner"].replace("'", "\\'") if m["winner"] else ""
                
                onclick_attr = f"onclick=\"showMatchDetail({m_id}, '{t_a_esc}', {m['score_a']}, '{t_b_esc}', {m['score_b']}, {p_win:.2f}, {p_draw:.2f}, {p_lose:.2f}, {r_a_adj:.1f}, {r_b_adj:.1f}, {final_l_a:.2f}, {final_l_b:.2f}, '{inj_a_esc}', '{inj_b_esc}', {str(m['is_pk']).lower()}, '{w_esc}')\""
                
                return f"""
                <div class="match-card" {onclick_attr}>
                    <div class="match-title">M{m_id} {round_name}</div>
                    <div class="team-row {win_a}">
                        <span class="team-name">{team_a}</span>
                        <span class="score">{m['score_a']}{pk_a}</span>
                    </div>
                    <div class="team-row {win_b}">
                        <span class="team-name">{team_b}</span>
                        <span class="score">{m['score_b']}{pk_b}</span>
                    </div>
                </div>
                """
            
            # Left Bracket Column HTML
            col1_html = "".join([gen_card_html(r32[i], "32강", 73 + i) for i in [0, 2, 1, 4, 3, 5, 6, 7]])
            col2_html = "".join([gen_card_html(r16[i], "16강", 89 + i) for i in [0, 1, 2, 3]])
            col3_html = "".join([gen_card_html(qf[i], "8강", 97 + i) for i in [0, 1]])
            col4_html = gen_card_html(sf[0], "4강", 101)
            
            # Center Column (Final & Champ)
            col5_html = f"""
            {gen_card_html(final[0], "결승", 104)}
            <div class="champion-card">
                <div class="champion-title">CHAMPION</div>
                <div class="champion-name">{champion}</div>
            </div>
            """
            
            # Right Bracket Column HTML
            col6_html = gen_card_html(sf[1], "4강", 102)
            col7_html = "".join([gen_card_html(qf[i], "8강", 97 + i) for i in [2, 3]])
            col8_html = "".join([gen_card_html(r16[i], "16강", 89 + i) for i in [4, 5, 6, 7]])
            col9_html = "".join([gen_card_html(r32[i], "32강", 73 + i) for i in [10, 11, 8, 9, 13, 15, 12, 14]])
            
            html_code = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{
                        background-color: #0f172a;
                        color: #f1f5f9;
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                        margin: 0;
                        padding: 10px;
                        overflow: auto;
                        height: 100vh;
                        box-sizing: border-box;
                    }}
                    .bracket-container {{
                        display: grid;
                        grid-template-columns: repeat(9, minmax(110px, 1fr));
                        gap: 8px;
                        height: 580px;
                        min-width: 1080px;
                    }}
                    .round-col {{
                        display: flex;
                        flex-direction: column;
                        justify-content: space-around;
                        height: 100%;
                    }}
                    .match-card {{
                        background-color: #1e293b;
                        border: 1px solid #334155;
                        border-radius: 6px;
                        padding: 6px;
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                        display: flex;
                        flex-direction: column;
                        gap: 2px;
                        font-size: 8.5px;
                        cursor: pointer;
                        transition: background-color 0.2s, transform 0.1s;
                    }}
                    .match-card:hover {{
                        background-color: #334155;
                        transform: scale(1.03);
                        border-color: #38bdf8;
                    }}
                    .match-title {{
                        font-size: 8px;
                        font-weight: 600;
                        color: #38bdf8;
                        border-bottom: 1px solid #334155;
                        padding-bottom: 2px;
                        margin-bottom: 4px;
                        text-align: center;
                    }}
                    .team-row {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 1px 2px;
                        border-radius: 3px;
                    }}
                    .team-row.winner {{
                        font-weight: bold;
                        color: #10b981;
                        background-color: rgba(16, 185, 129, 0.08);
                    }}
                    .team-name {{
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        max-width: 80px;
                    }}
                    .score {{
                        font-weight: bold;
                        font-size: 8.5px;
                    }}
                    .champion-card {{
                        background: linear-gradient(135deg, #1e293b, #0f172a);
                        border: 2px solid #10b981;
                        border-radius: 8px;
                        padding: 8px;
                        text-align: center;
                        box-shadow: 0 0 15px rgba(16, 185, 129, 0.25);
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        gap: 4px;
                    }}
                    .champion-title {{
                        font-size: 9px;
                        font-weight: 800;
                        color: #10b981;
                        letter-spacing: 0.5px;
                    }}
                    .champion-name {{
                        font-weight: bold;
                        font-size: 11px;
                        color: #ffffff;
                    }}
                    /* Modal overlay styling */
                    .modal-overlay {{
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background-color: rgba(15, 23, 42, 0.75);
                        display: none;
                        justify-content: center;
                        align-items: center;
                        z-index: 1000;
                    }}
                    .modal-content {{
                        background-color: #1e293b;
                        border: 2px solid #334155;
                        border-radius: 12px;
                        padding: 20px;
                        width: 380px;
                        max-width: 90%;
                        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                        position: relative;
                        color: #f1f5f9;
                    }}
                    .close-btn {{
                        position: absolute;
                        top: 8px;
                        right: 12px;
                        font-size: 20px;
                        font-weight: bold;
                        color: #94a3b8;
                        cursor: pointer;
                        transition: color 0.2s;
                    }}
                    .close-btn:hover {{
                        color: #f1f5f9;
                    }}
                    .modal-teams {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 15px;
                        margin-top: 10px;
                    }}
                    .modal-team-box {{
                        width: 40%;
                        text-align: center;
                    }}
                    .modal-team-name {{
                        font-size: 13px;
                        font-weight: bold;
                        margin-bottom: 4px;
                    }}
                    .modal-score {{
                        font-size: 26px;
                        font-weight: bold;
                        color: #10b981;
                        width: 20%;
                        text-align: center;
                    }}
                    .modal-stats {{
                        background-color: #0f172a;
                        border-radius: 8px;
                        padding: 10px;
                        margin-bottom: 12px;
                        font-size: 10.5px;
                    }}
                    .stat-row {{
                        display: flex;
                        justify-content: space-between;
                        padding: 5px 0;
                        border-bottom: 1px solid #1e293b;
                    }}
                    .stat-row:last-child {{
                        border-bottom: none;
                    }}
                    .stat-label {{
                        color: #94a3b8;
                    }}
                    .gauge-container {{
                        height: 16px;
                        border-radius: 8px;
                        overflow: hidden;
                        display: flex;
                        border: 1px solid #475569;
                        margin-top: 8px;
                    }}
                    .gauge-bar {{
                        text-align: center;
                        color: white;
                        font-size: 9px;
                        line-height: 14px;
                        font-weight: bold;
                    }}
                </style>
            </head>
            <body>
                <div class="bracket-container">
                    <div class="round-col">{col1_html}</div>
                    <div class="round-col">{col2_html}</div>
                    <div class="round-col">{col3_html}</div>
                    <div class="round-col">{col4_html}</div>
                    <div class="round-col" style="justify-content: center; gap: 30px;">{col5_html}</div>
                    <div class="round-col">{col6_html}</div>
                    <div class="round-col">{col7_html}</div>
                    <div class="round-col">{col8_html}</div>
                    <div class="round-col">{col9_html}</div>
                </div>

                <!-- 모달 레이어 -->
                <div id="match-modal" class="modal-overlay" onclick="closeModal()">
                    <div class="modal-content" onclick="event.stopPropagation()">
                        <span class="close-btn" onclick="closeModal()">&times;</span>
                        <div id="modal-body"></div>
                    </div>
                </div>

                <script>
                    function showMatchDetail(matchId, teamA, scoreA, teamB, scoreB, winProbA, winProbDraw, winProbB, eloA, eloB, lambdaA, lambdaB, injuriesA, injuriesB, isPk, winner) {{
                        const modal = document.getElementById('match-modal');
                        const body = document.getElementById('modal-body');
                        
                        const pkSuffix = isPk ? " (PK)" : "";
                        const winnerText = winner ? "시뮬레이션 승리: " + winner + pkSuffix : "무승부";
                        
                        const injAList = injuriesA ? injuriesA.split(',') : [];
                        const injBList = injuriesB ? injuriesB.split(',') : [];
                        const injAHtml = injAList.length > 0 ? injAList.map(n => `<span style="color:#f43f5e">${{n}}</span>`).join(', ') : '없음';
                        const injBHtml = injBList.length > 0 ? injBList.map(n => `<span style="color:#f43f5e">${{n}}</span>`).join(', ') : '없음';

                        body.innerHTML = `
                            <div style="font-size: 11px; font-weight: bold; color: #38bdf8; margin-bottom: 12px; text-align: center;">[MATCH ${{matchId}}] 시뮬레이션 예측 분석 리포트</div>
                            <div class="modal-teams">
                                <div class="modal-team-box">
                                    <div class="modal-team-name">${{teamA}}</div>
                                    <div style="font-size: 9.5px; color: #94a3b8;">보정 ELO: ${{Math.round(eloA)}}</div>
                                </div>
                                <div class="modal-score">${{scoreA}} : ${{scoreB}}</div>
                                <div class="modal-team-box">
                                    <div class="modal-team-name">${{teamB}}</div>
                                    <div style="font-size: 9.5px; color: #94a3b8;">보정 ELO: ${{Math.round(eloB)}}</div>
                                </div>
                            </div>
                            <div style="text-align: center; font-size: 11.5px; font-weight: bold; color: #10b981; margin-bottom: 15px;">
                                ${{winnerText}}
                            </div>
                            
                            <div class="modal-stats">
                                <div class="stat-row">
                                    <span class="stat-label">평균 예상 득점 (람다)</span>
                                    <span>${{teamA}} <b>${{lambdaA}}골</b> vs <b>${{lambdaB}}골</b> ${{teamB}}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-label">부상 결장 자원</span>
                                    <span style="text-align: right; max-width: 65%; word-break: break-all; font-size: 9.5px; line-height: 1.3;">
                                        ${{teamA}}: ${{injAHtml}}<br>
                                        ${{teamB}}: ${{injBHtml}}
                                    </span>
                                </div>
                            </div>
                            
                            <div style="font-size: 9.5px; font-weight: bold; color: #cbd5e1; margin-bottom: 4px; text-align: center;">AI 승 / 무 / 패 예측 확률</div>
                            <div class="gauge-container">
                                <div class="gauge-bar" style="width: ${{winProbA}}%; background-color: #0ea5e9;">${{Math.round(winProbA)}}%</div>
                                <div class="gauge-bar" style="width: ${{winProbDraw}}%; background-color: #64748b;">${{Math.round(winProbDraw)}}%</div>
                                <div class="gauge-bar" style="width: ${{winProbB}}%; background-color: #ec4899;">${{Math.round(winProbB)}}%</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 8.5px; color: #94a3b8; margin-top: 4px; padding: 0 4px;">
                                <span>${{teamA}} 승</span>
                                <span>무승부</span>
                                <span>${{teamB}} 승</span>
                            </div>
                        `;
                        modal.style.display = 'flex';
                    }}

                    function closeModal() {{
                        document.getElementById('match-modal').style.display = 'none';
                    }}
                </script>
            </body>
            </html>
            """
            st.iframe(html_code, height=600)


# ----------------- 탭 4: 1대1 가상 매치 시뮬레이터 -----------------
with tab4:
    st.header("1대1 가상 매치 시뮬레이터 (커스텀 변수 입력)")
    st.markdown("임의의 두 국가를 선택하고, 경기 환경 변수(홈 우위, 휴식 일정 격차, 대륙 이동 피로도)를 조정해 즉각적인 AI 승률 분석을 실행합니다.")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        team_a = st.selectbox("첫 번째 팀 (Home)", sorted(list(elo_ratings.keys())), index=selected_team_idx if 'selected_team_idx' in locals() else 0, format_func=lambda x: f"{get_flag(x)} {x}")
    with col_t2:
        team_b = st.selectbox("두 번째 팀 (Away)", sorted(list(elo_ratings.keys())), index=1, format_func=lambda x: f"{get_flag(x)} {x}")
        
    st.markdown("### 경기 환경 보정 설정")
    
    c_env1, c_env2, c_env3 = st.columns(3)
    with c_env1:
        home_advantage_opt = st.selectbox("개최국 홈 우위 버프 적용", ["None", f"Team A ({get_flag(team_a)} {team_a})가 개최국", f"Team B ({get_flag(team_b)} {team_b})가 개최국"])
    with c_env2:
        rest_days_diff = st.slider("체력 우위 (Team A 휴식일 - Team B 휴식일)", min_value=-10, max_value=10, value=0, help="양수일 경우 Team A가 더 쉼, 음수일 경우 Team B가 더 쉼")
    with c_env3:
        st.markdown("**대륙 이동 피로도 감쇄 (0% ~ 10%)**")
        fatigue_a = st.slider(f"{team_a} 피로도", min_value=0.0, max_value=0.10, value=0.0, step=0.015, format="%.3f")
        fatigue_b = st.slider(f"{team_b} 피로도", min_value=0.0, max_value=0.10, value=0.0, step=0.015, format="%.3f")
        
    if team_a == team_b:
        st.warning("경고: 서로 다른 국가를 선택해야 정상적인 시뮬레이션이 가능합니다.")
    else:
        if st.button("전력 비교 및 AI 승률 예측 실행"):
            # 기본 ELO 점수 로드
            r_a = elo_ratings.get(team_a, 1700.0)
            r_b = elo_ratings.get(team_b, 1700.0)
            
            # ELO 가중치 보정
            if home_advantage_opt == f"Team A ({team_a})가 개최국":
                r_a += 40
            elif home_advantage_opt == f"Team B ({team_b})가 개최국":
                r_b += 40
                
            rest_bonus = min(abs(rest_days_diff) * 5, 30)
            if rest_days_diff >= 1:
                r_a += rest_bonus
            elif rest_days_diff <= -1:
                r_b += rest_bonus

            # ELO Win probability
            elo = EloSystem()
            expected_win_a = elo.expected_score(r_a, r_b)
            
            # Poisson lambda 변환
            l_a, l_b = win_prob_to_lambda(expected_win_a)
            
            # 부상자 로드 및 보정
            att_m_a, def_m_a, details_a = get_injury_multipliers(team_a, active_injuries)
            att_m_b, def_m_b, details_b = get_injury_multipliers(team_b, active_injuries)
            
            final_l_a = l_a * att_m_a * def_m_b * (1.0 - fatigue_a)
            final_l_b = l_b * att_m_b * def_m_a * (1.0 - fatigue_b)
            
            # 승무패 도출
            prob = match_probabilities(final_l_a, final_l_b)
            p_win = prob["win"] * 100
            p_draw = prob["draw"] * 100
            p_lose = prob["lose"] * 100

            # 100,000회 시뮬레이션을 통해 가장 확률 높은 예상 스코어 도출
            sa_samples = np.random.poisson(final_l_a, 100000)
            sb_samples = np.random.poisson(final_l_b, 100000)
            score_counts = Counter(zip(sa_samples, sb_samples))
            (best_sa, best_sb), count = max(score_counts.items(), key=lambda x: x[1])
            score_prob = (count / 100000) * 100

            st.markdown("---")
            st.markdown(f"### {get_flag(team_a)} {team_a} vs {get_flag(team_b)} {team_b} 예측 결과 리포트")
            
            col_res1, col_res2 = st.columns([1, 1])
            with col_res1:
                st.markdown(
                    f"""
                    <div style="background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155;">
                        <div style="font-size: 1.15em; font-weight: bold; color: #38bdf8; margin-bottom: 10px;">보정 후 ELO 레이팅</div>
                        <p>{team_a}: <b>{r_a:.1f}</b> (순수 ELO: {elo_ratings.get(team_a):.0f})</p>
                        <p>{team_b}: <b>{r_b:.1f}</b> (순수 ELO: {elo_ratings.get(team_b):.0f})</p>
                        <div style="font-size: 1.15em; font-weight: bold; color: #38bdf8; margin-top: 15px; margin-bottom: 10px;">예상 득점(람다)</div>
                        <p>{team_a} 평균 예상 득점: <b>{final_l_a:.2f}골</b></p>
                        <p>{team_b} 평균 예상 득점: <b>{final_l_b:.2f}골</b></p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            with col_res2:
                # 최빈 스코어 및 승무패 확률
                st.markdown(
                    f"""
                    <div style="background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155; height: 100%;">
                        <div style="font-size: 1.15em; font-weight: bold; color: #38bdf8; margin-bottom: 10px;">AI 예측 요약</div>
                        <p style="font-size: 1.1em;">가장 확률 높은 스코어: <b style="color: #38bdf8; font-size: 1.3em;">{team_a} {best_sa} : {best_sb} {team_b}</b> (약 {score_prob:.1f}% 확률)</p>
                        <p>{team_a} 승리 확률: <b>{p_win:.1f}%</b></p>
                        <p>무승부 확률: <b>{p_draw:.1f}%</b></p>
                        <p>{team_b} 승리 확률: <b>{p_lose:.1f}%</b></p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
            # 부상자 누수 리포트 출력
            if details_a or details_b:
                st.markdown("#### 적용된 부상 전력 감쇄 정보")
                if details_a:
                    st.markdown(f"**{team_a}** 결장 명단:")
                    for d in details_a:
                        st.markdown(f"  * {d}")
                if details_b:
                    st.markdown(f"**{team_b}** 결장 명단:")
                    for d in details_b:
                        st.markdown(f"  * {d}")

            # 차트 그리기
            st.markdown("#### 승부 예측 게이지")
            st.markdown(
                f"""
                <div style="display: flex; height: 30px; border-radius: 15px; overflow: hidden; border: 1px solid #475569; margin-top: 10px;">
                    <div style="width: {p_win}%; background-color: #0ea5e9; text-align: center; color: white; font-size: 1em; line-height: 28px; font-weight: bold;">{get_flag(team_a)} {team_a} 승 ({p_win:.1f}%)</div>
                    <div style="width: {p_draw}%; background-color: #64748b; text-align: center; color: white; font-size: 1em; line-height: 28px; font-weight: bold;">무승부 ({p_draw:.1f}%)</div>
                    <div style="width: {p_lose}%; background-color: #ec4899; text-align: center; color: white; font-size: 1em; line-height: 28px; font-weight: bold;">{get_flag(team_b)} {team_b} 승 ({p_lose:.1f}%)</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ----------------- 탭 5: 전체 예측 -----------------
with tab5:
    st.header("몬테카를로 시뮬레이터 (전체 대회 시뮬레이션)")
    st.markdown(
        "본 시뮬레이터는 각 국가의 ELO 레이팅, 실시간 부상 상태, 경기간 일정 및 피로도를 결합하여 "
        "설정된 횟수만큼 2026 월드컵을 통째로 모의 진행한 뒤 각 팀의 단계별 생존 확률 통계를 연산합니다."
    )
    
    # UI 개선: 슬라이더와 실행 버튼을 가로로 배치하여 한 눈에 보기 편하게 구성
    col1, col2 = st.columns([3, 1])
    with col1:
        sim_runs = st.slider(
            "시뮬레이션 반복 횟수 선택", 
            min_value=500, 
            max_value=10000, 
            value=2000, 
            step=500, 
            key="sim_runs_tab5"
        )
    with col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True) # 레이블 공간 비우기
        run_btn = st.button(
            f"시뮬레이션 실행 ({sim_runs:,}회)", 
            key="run_sim_tab5",
            use_container_width=True
        )
    
    if run_btn:
        with st.spinner("월드컵 시뮬레이션 진행 중... (약 2~5초 소요)"):
            # ELO 객체 초기화
            elo_sys = EloSystem()
            elo_sys.load_ratings(ELO_PATH)
            
            # 시뮬레이터 실행
            sim = WorldCupSimulation(
                elo_system=elo_sys,
                groups_file=GROUPS_PATH,
                actual_results_file=ACTUAL_RESULTS_PATH,
                absences_file=ABSENCES_PATH,
                squads_file=SQUADS_PATH
            )
            
            # 통계 트래커 생성
            r32_counts = defaultdict(int)
            r16_counts = defaultdict(int)
            qf_counts = defaultdict(int)
            sf_counts = defaultdict(int)
            final_counts = defaultdict(int)
            champion_counts = defaultdict(int)
            
            # 시뮬레이션 루프 진행
            for _ in range(sim_runs):
                standings = sim.simulate_group_stage()
                r32_teams = sim.get_advancing_teams(standings)
                for t in r32_teams:
                    r32_counts[t] += 1
                    
                ko_results = sim.simulate_knockout_stage(r32_teams)
                
                # 16강 진출팀 기록
                for m in ko_results["Round of 16"]:
                    r16_counts[m["winner"]] += 1
                # 8강 진출팀 기록
                for m in ko_results["Quarter-finals"]:
                    qf_counts[m["winner"]] += 1
                # 4강 진출팀 기록
                for m in ko_results["Semi-finals"]:
                    sf_counts[m["winner"]] += 1
                # 결승 진출팀 기록
                for m in ko_results["Final"]:
                    final_counts[m["winner"]] += 1
                # 우승팀 기록
                champion_counts[ko_results["Champion"]] += 1
                
            # 데이터프레임 빌드
            all_countries = list(elo_ratings.keys())
            sim_records = []
            for t in all_countries:
                sim_records.append({
                    "국가": f"{get_flag(t)} {t}",
                    "기본 ELO": elo_ratings.get(t, 1500.0),
                    "32강 진출률": (r32_counts.get(t, 0) / sim_runs) * 100,
                    "16강 진출률": (r16_counts.get(t, 0) / sim_runs) * 100,
                    "8강 진출률": (qf_counts.get(t, 0) / sim_runs) * 100,
                    "4강 진출률": (sf_counts.get(t, 0) / sim_runs) * 100,
                    "결승 진출률": (final_counts.get(t, 0) / sim_runs) * 100,
                    "우승 확률": (champion_counts.get(t, 0) / sim_runs) * 100,
                })
                
            df_sim = pd.DataFrame(sim_records)
            df_sim = df_sim.sort_values(by="우승 확률", ascending=False).reset_index(drop=True)
            
            # 소수점 포맷팅
            formatted_df = df_sim.style.format({
                "기본 ELO": "{:.0f}",
                "32강 진출률": "{:.1f}%",
                "16강 진출률": "{:.1f}%",
                "8강 진출률": "{:.1f}%",
                "4강 진출률": "{:.1f}%",
                "결승 진출률": "{:.1f}%",
                "우승 확률": "{:.2f}%"
            })
            
            st.dataframe(formatted_df, width="stretch", height=600)

