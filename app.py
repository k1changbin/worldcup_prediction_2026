import streamlit as st
import html
import json
import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from src.elo import EloSystem
from src.poisson import win_prob_to_lambda, match_probabilities
from src.simulation import WorldCupSimulation, HOST_COUNTRIES, GROUP_REGIONS
from src.absences import (
    calculate_absence_multipliers,
    clean_served_suspensions as clean_served_suspensions_data,
    format_absence_list_to_str_list,
    get_absence_names,
    load_absences,
    save_absences,
)

# City-to-region mapping used for travel-fatigue calculations.
CITY_REGIONS = {
    "los-angeles": 1, "san-francisco": 1, "seattle": 1, "vancouver": 1,
    "guadalajara": 2, "mexico-city": 2, "monterrey": 2,
    "dallas": 3, "houston": 3, "kansas-city": 3,
    "atlanta": 4, "miami": 4,
    "boston": 5, "new-york": 5, "philadelphia": 5, "toronto": 5
}

# Data paths.
ELO_PATH = "data/elo_ratings.json"
SQUADS_PATH = "data/squads.json"
ABSENCES_PATH = "data/absences.json"
SCHEDULE_PATH = "data/schedule.json"
ACTUAL_RESULTS_PATH = "data/actual_results.json"
GROUPS_PATH = "data/groups.json"
KNOCKOUT_CONSENSUS_RUNS = 10000

# Flag emoji mapping by team.
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

def clean_served_suspensions(injuries_dict, actual_results_list):
    cleaned, updated = clean_served_suspensions_data(injuries_dict, actual_results_list)
    if updated:
        save_absences(ABSENCES_PATH, cleaned)
    return cleaned

squads = {}

def get_injury_multipliers(team, injuries_dict, squads_dict=None):
    source_squads = squads if squads_dict is None else squads_dict
    if squads_dict is None and not source_squads:
        source_squads = load_json(SQUADS_PATH)
    return calculate_absence_multipliers(team, injuries_dict, source_squads)

def run_app():
    global squads

    def clear_projected_knockout_cache():
        for key in (
            "projected_knockout_results",
            "projected_knockout_matches",
            "projected_knockout_error",
            "show_projected_knockout_bracket",
        ):
            st.session_state.pop(key, None)

    st.set_page_config(
        page_title="World Cup 2026 AI Simulator",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Load data.
    elo_ratings = load_json(ELO_PATH)
    squads = load_json(SQUADS_PATH)
    injuries_raw = load_absences(ABSENCES_PATH)
    schedule = load_json(SCHEDULE_PATH)
    actual_results = load_json(ACTUAL_RESULTS_PATH)
    groups_dict = load_json(GROUPS_PATH)

    # Automatically restore served suspensions.
    injuries_raw = clean_served_suspensions(injuries_raw, actual_results)

    # Validate required data.
    if not elo_ratings or not groups_dict:
        st.error("Required data files (data/elo_ratings.json, data/groups.json) are missing or empty. Check the project setup.")
        st.stop()

    # ----------------- Absence management helpers for the sidebar -----------------
    # ----------------- Dynamic rest/fatigue state calculation from the full schedule -----------------
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

            # Calculate home-team state.
            rest_home = 0
            fatigue_home = 0.0
            if home in team_states and team_states[home]["last_date"] is not None:
                prev_date = team_states[home]["last_date"]
                prev_reg = team_states[home]["last_region"]
                rest_home = day_num - prev_date
                diff = abs(region - prev_reg)
                fatigue_home = 0.03 if diff >= 3 else (0.015 if diff > 0 else 0.0)

            # Calculate away-team state.
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

            # Update state so predicted matches still move each team through the schedule.
            if home in team_states:
                team_states[home] = {"last_date": day_num, "last_region": region}
            if away in team_states:
                team_states[away] = {"last_date": day_num, "last_region": region}

        return match_states

    match_states_cache = compute_schedule_states(schedule)

    # ----------------- Sidebar absence-management panel -----------------
    st.sidebar.markdown("## Live Injury and Suspension Management")
    st.sidebar.info("Register player injuries or suspensions to apply them immediately across the prediction model.")

    # Wikipedia sync button.
    import fetch_suspensions
    if st.sidebar.button("Sync Live Suspension Data (Wikipedia)"):
        with st.sidebar.spinner("Fetching suspension records from Wikipedia..."):
            fetch_suspensions.main()
        clear_projected_knockout_cache()
        st.sidebar.success("Sync complete.")
        st.rerun()

    # Show current injuries and suspensions.
    st.sidebar.markdown("### Current Absences")
    active_injuries = load_absences(ABSENCES_PATH)

    # Re-run auto-restoration as a guard.
    active_injuries = clean_served_suspensions(active_injuries, actual_results)

    if not active_injuries or all(len(v) == 0 for v in active_injuries.values()):
        st.sidebar.text("No registered absences.")
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
                            "red_card": "red-card suspension",
                            "yellow_cards": "yellow-card accumulation suspension"
                        }
                        reason_text = reason_map.get(p.get("reason"), "suspension")
                        st.sidebar.text(f"  • {p_name} ({reason_text}, returns after match count {p.get('served_at_count')})")
                    else:
                        st.sidebar.text(f"  • {p_name} (injury)")
                else:
                    st.sidebar.text(f"  • {p} (injury)")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Register New Absence")

    # Team selection.
    all_teams = sorted(list(elo_ratings.keys()))
    selected_team = st.sidebar.selectbox("Select Team", all_teams, format_func=lambda x: f"{get_flag(x)} {x}")

    # Player selection.
    if selected_team in squads:
        team_players = [p["name"] for p in squads[selected_team]]
        # Exclude players that are already registered as absent.
        already_injured_names = get_absence_names(active_injuries.get(selected_team, []))
        available_players = [p for p in team_players if p not in already_injured_names]

        if available_players:
            selected_player = st.sidebar.selectbox("Select Player", available_players)
            absence_reason = st.sidebar.selectbox(
                "Absence Reason",
                ["Injury", "Red card (1-match suspension)", "Yellow-card accumulation (1-match suspension)", "Additional suspension (2 matches)"],
            )

            if st.sidebar.button("Add to Absence List"):
                if selected_team not in active_injuries:
                    active_injuries[selected_team] = []

                if absence_reason == "Injury":
                    active_injuries[selected_team].append({
                        "name": selected_player,
                        "type": "injury"
                    })
                else:
                    # Register suspension.
                    N = 0
                    for match in actual_results:
                        if match.get("team_a") == selected_team or match.get("team_b") == selected_team:
                            N += 1

                    suspension_length = 2 if "2 matches" in absence_reason else 1
                    served_at = N + suspension_length
                    reason = "red_card" if "Red card" in absence_reason else "yellow_cards"

                    active_injuries[selected_team].append({
                        "name": selected_player,
                        "type": "suspension",
                        "reason": reason,
                        "served_at_count": served_at
                    })

                save_absences(ABSENCES_PATH, active_injuries)
                clear_projected_knockout_cache()
                st.rerun()
        else:
            st.sidebar.warning("All players on this team are already absent, or no selectable players are available.")
    else:
        st.sidebar.warning(f"Squad data does not exist for {selected_team}.")

    # Restore absences.
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Restore Players")

    # Build dropdown labels for registered absences.
    injured_options = []
    for team, players_list in active_injuries.items():
        for p in players_list:
            p_name = p.get("name") if isinstance(p, dict) else p
            injured_options.append(f"{get_flag(team)} {team} - {p_name}")

    if injured_options:
        selected_to_recover = st.sidebar.multiselect("Select Players to Restore", injured_options)
        if st.sidebar.button("Restore Selected Players"):
            for opt in selected_to_recover:
                # Find and remove the matching team/player record.
                for team, players_list in list(active_injuries.items()):
                    for p in list(players_list):
                        p_name = p.get("name") if isinstance(p, dict) else p
                        if f"{get_flag(team)} {team} - {p_name}" == opt:
                            players_list.remove(p)
                    if not active_injuries[team]:
                        del active_injuries[team]
            save_absences(ABSENCES_PATH, active_injuries)
            clear_projected_knockout_cache()
            st.rerun()
    else:
        st.sidebar.text("No absent players to restore.")


    # ----------------- Main dashboard layout -----------------
    st.title("2026 FIFA World Cup Tournament Prediction Dashboard")
    st.markdown("Explore the current knockout bracket, completed results, and future match predictions based on final group-stage outcomes.")

    refresh_message = st.session_state.pop("data_refresh_message", None)
    if refresh_message:
        st.success(refresh_message)

    refresh_col, _ = st.columns([1, 4])
    with refresh_col:
        if st.button("Refresh All Data", key="refresh_all_data", use_container_width=True):
            try:
                with st.spinner("Refreshing Elo ratings, actual results, and absence data..."):
                    import fetch_data
                    fetch_data.fetch_live_world_cup_data()
                clear_projected_knockout_cache()
                st.session_state["data_refresh_message"] = "All data has been refreshed."
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to refresh all data: {exc}")

    def build_knockout_projection():
        elo_sys = EloSystem()
        elo_sys.load_ratings(ELO_PATH)
        sim = WorldCupSimulation(
            elo_system=elo_sys,
            groups_file=GROUPS_PATH,
            actual_results_file=ACTUAL_RESULTS_PATH,
            absences_file=ABSENCES_PATH,
            squads_file=SQUADS_PATH
        )

        standings = sim.simulate_group_stage()
        sim.get_advancing_teams(standings)
        ko_results = sim.simulate_knockout_stage(consensus_runs=KNOCKOUT_CONSENSUS_RUNS)

        match_map = {}
        for round_key, start_match in [
            ("Round of 32", 73),
            ("Round of 16", 89),
            ("Quarter-finals", 97),
            ("Semi-finals", 101),
            ("Final", 104),
        ]:
            for idx, projected_match in enumerate(ko_results[round_key]):
                match_map[start_match + idx] = projected_match

        return ko_results, match_map

    projected_knockout_results = st.session_state.get("projected_knockout_results")
    projected_knockout_matches = st.session_state.get("projected_knockout_matches", {})
    knockout_projection_error = st.session_state.get("projected_knockout_error")

    def flatten_knockout_matches(ko_results):
        if not ko_results:
            return {}
        match_map = {}
        for round_key in [
            "Round of 32",
            "Round of 16",
            "Quarter-finals",
            "Semi-finals",
            "Final",
        ]:
            for match in ko_results.get(round_key, []):
                match_id = match.get("match_id")
                if match_id:
                    match_map[match_id] = match
        return match_map

    def get_actual_knockout_result(team_a, team_b):
        if not team_a or not team_b:
            return None
        for result in actual_results:
            if (
                result.get("stage") == "knockout"
                and {result.get("team_a"), result.get("team_b")} == {team_a, team_b}
            ):
                return result
        return None

    def make_current_match(match_id, team_a, team_b):
        result = get_actual_knockout_result(team_a, team_b)
        if result:
            score_a = result["score_a"] if result["team_a"] == team_a else result["score_b"]
            score_b = result["score_b"] if result["team_a"] == team_a else result["score_a"]
            winner = result.get("winner")
            if not winner and score_a != score_b:
                winner = team_a if score_a > score_b else team_b
            return {
                "match_id": match_id,
                "team_a": team_a,
                "team_b": team_b,
                "score_a": score_a,
                "score_b": score_b,
                "winner": winner,
                "is_pk": score_a == score_b and bool(winner),
                "played": True,
            }

        return {
            "match_id": match_id,
            "team_a": team_a,
            "team_b": team_b,
            "score_a": None,
            "score_b": None,
            "winner": None,
            "is_pk": False,
            "played": False,
        }

    def build_current_knockout_state():
        elo_sys = EloSystem()
        elo_sys.load_ratings(ELO_PATH)
        sim = WorldCupSimulation(
            elo_system=elo_sys,
            groups_file=GROUPS_PATH,
            actual_results_file=ACTUAL_RESULTS_PATH,
            absences_file=ABSENCES_PATH,
            squads_file=SQUADS_PATH
        )
        standings = sim.simulate_group_stage()

        team_by_code = {}
        third_places = []
        for group_name, group_teams in standings.items():
            group_letter = group_name.split(" ")[1]
            team_by_code[f"{group_letter}1"] = group_teams[0][0]
            team_by_code[f"{group_letter}2"] = group_teams[1][0]
            team_by_code[f"{group_letter}3"] = group_teams[2][0]
            third_places.append({
                "team_name": group_teams[2][0],
                "group": group_letter,
                "stats": group_teams[2][1],
            })

        top_8_thirds = sim._sort_third_places(third_places)[:8]
        third_assignment = sim.match_thirds([(t["team_name"], t["group"]) for t in top_8_thirds])

        r32_slots = [
            {"match_id": 73, "team_a": "A2", "team_b": "B2"},
            {"match_id": 74, "team_a": "E1", "team_b": "3rd_74"},
            {"match_id": 75, "team_a": "F1", "team_b": "C2"},
            {"match_id": 76, "team_a": "C1", "team_b": "F2"},
            {"match_id": 77, "team_a": "I1", "team_b": "3rd_77"},
            {"match_id": 78, "team_a": "E2", "team_b": "I2"},
            {"match_id": 79, "team_a": "A1", "team_b": "3rd_79"},
            {"match_id": 80, "team_a": "L1", "team_b": "3rd_80"},
            {"match_id": 81, "team_a": "D1", "team_b": "3rd_81"},
            {"match_id": 82, "team_a": "G1", "team_b": "3rd_82"},
            {"match_id": 83, "team_a": "K2", "team_b": "L2"},
            {"match_id": 84, "team_a": "H1", "team_b": "J2"},
            {"match_id": 85, "team_a": "B1", "team_b": "3rd_85"},
            {"match_id": 86, "team_a": "J1", "team_b": "H2"},
            {"match_id": 87, "team_a": "K1", "team_b": "3rd_87"},
            {"match_id": 88, "team_a": "D2", "team_b": "G2"},
        ]

        r32 = []
        for slot in r32_slots:
            team_a = team_by_code[slot["team_a"]]
            team_b = third_assignment[slot["match_id"]] if slot["team_b"].startswith("3rd_") else team_by_code[slot["team_b"]]
            r32.append(make_current_match(slot["match_id"], team_a, team_b))

        def winner_or_slot(matches, idx, match_id):
            return matches[idx]["winner"] or f"Winner M{match_id}"

        r16_pairings = [
            (winner_or_slot(r32, 0, 73), winner_or_slot(r32, 2, 75)),
            (winner_or_slot(r32, 1, 74), winner_or_slot(r32, 4, 77)),
            (winner_or_slot(r32, 3, 76), winner_or_slot(r32, 5, 78)),
            (winner_or_slot(r32, 6, 79), winner_or_slot(r32, 7, 80)),
            (winner_or_slot(r32, 10, 83), winner_or_slot(r32, 11, 84)),
            (winner_or_slot(r32, 8, 81), winner_or_slot(r32, 9, 82)),
            (winner_or_slot(r32, 13, 86), winner_or_slot(r32, 15, 88)),
            (winner_or_slot(r32, 12, 85), winner_or_slot(r32, 14, 87)),
        ]
        r16 = [make_current_match(89 + idx, team_a, team_b) for idx, (team_a, team_b) in enumerate(r16_pairings)]

        qf_pairings = [
            (winner_or_slot(r16, 0, 89), winner_or_slot(r16, 1, 90)),
            (winner_or_slot(r16, 2, 91), winner_or_slot(r16, 3, 92)),
            (winner_or_slot(r16, 4, 93), winner_or_slot(r16, 5, 94)),
            (winner_or_slot(r16, 6, 95), winner_or_slot(r16, 7, 96)),
        ]
        qf = [make_current_match(97 + idx, team_a, team_b) for idx, (team_a, team_b) in enumerate(qf_pairings)]

        sf_pairings = [
            (winner_or_slot(qf, 0, 97), winner_or_slot(qf, 1, 98)),
            (winner_or_slot(qf, 2, 99), winner_or_slot(qf, 3, 100)),
        ]
        sf = [make_current_match(101 + idx, team_a, team_b) for idx, (team_a, team_b) in enumerate(sf_pairings)]

        final = [make_current_match(104, winner_or_slot(sf, 0, 101), winner_or_slot(sf, 1, 102))]
        return {
            "Round of 32": r32,
            "Round of 16": r16,
            "Quarter-finals": qf,
            "Semi-finals": sf,
            "Final": final,
            "Champion": final[0]["winner"] or "TBD",
        }

    def render_static_bracket_html(ko_results, champion_label=None):
        r32 = ko_results["Round of 32"]
        r16 = ko_results["Round of 16"]
        qf = ko_results["Quarter-finals"]
        sf = ko_results["Semi-finals"]
        final = ko_results["Final"]
        champion = champion_label if champion_label is not None else ko_results.get("Champion", "TBD")

        def card(match, round_name, fallback_id):
            match_id = match.get("match_id", fallback_id)
            team_a = html.escape(str(match.get("team_a") or "TBD"))
            team_b = html.escape(str(match.get("team_b") or "TBD"))
            winner = match.get("winner")
            win_a = "winner" if winner == match.get("team_a") else ""
            win_b = "winner" if winner == match.get("team_b") else ""
            pk_a = " (PK)" if match.get("is_pk") and winner == match.get("team_a") else ""
            pk_b = " (PK)" if match.get("is_pk") and winner == match.get("team_b") else ""
            score_a = "" if match.get("score_a") is None else f"{match['score_a']}{pk_a}"
            score_b = "" if match.get("score_b") is None else f"{match['score_b']}{pk_b}"
            state_class = "played" if match.get("played") else "scheduled"

            return f"""
            <div class="match-card {state_class}">
                <div class="match-title">M{match_id} {round_name}</div>
                <div class="team-row {win_a}">
                    <span class="team-name">{team_a}</span>
                    <span class="score">{score_a}</span>
                </div>
                <div class="team-row {win_b}">
                    <span class="team-name">{team_b}</span>
                    <span class="score">{score_b}</span>
                </div>
            </div>
            """

        col1_html = "".join([card(r32[i], "R32", 73 + i) for i in [0, 2, 1, 4, 3, 5, 6, 7]])
        col2_html = "".join([card(r16[i], "R16", 89 + i) for i in [0, 1, 2, 3]])
        col3_html = "".join([card(qf[i], "QF", 97 + i) for i in [0, 1]])
        col4_html = card(sf[0], "SF", 101)
        col5_html = f"""
        {card(final[0], "Final", 104)}
        <div class="champion-card">
            <div class="champion-title">CHAMPION</div>
            <div class="champion-name">{html.escape(str(champion))}</div>
        </div>
        """
        col6_html = card(sf[1], "SF", 102)
        col7_html = "".join([card(qf[i], "QF", 97 + i) for i in [2, 3]])
        col8_html = "".join([card(r16[i], "R16", 89 + i) for i in [4, 5, 6, 7]])
        col9_html = "".join([card(r32[i], "R32", 73 + i) for i in [10, 11, 8, 9, 13, 15, 12, 14]])

        return f"""
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
                    display: flex;
                    flex-direction: column;
                    gap: 2px;
                    font-size: 8.5px;
                }}
                .match-card.played {{
                    border-color: #10b981;
                }}
                .match-card.scheduled {{
                    opacity: 0.88;
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
                    gap: 6px;
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
                    max-width: 82px;
                }}
                .score {{
                    min-width: 18px;
                    text-align: right;
                    font-weight: bold;
                    font-size: 8.5px;
                }}
                .champion-card {{
                    background: linear-gradient(135deg, #1e293b, #0f172a);
                    border: 2px solid #10b981;
                    border-radius: 8px;
                    padding: 8px;
                    text-align: center;
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
        </body>
        </html>
        """

    current_knockout_error = None
    try:
        current_knockout_results = build_current_knockout_state()
    except Exception as exc:
        current_knockout_results = None
        current_knockout_error = exc
    current_knockout_matches = flatten_knockout_matches(current_knockout_results)

    # Main tab layout.
    tab3, tab1, tab2, tab4, tab5 = st.tabs([
        "Knockout Bracket",
        "Schedule and Match Predictions",
        "Final Group Standings",
        "Head-to-Head Simulator",
        "Full Tournament Forecast"
    ])

    # ----------------- Tab 1: Schedule and match predictions -----------------
    with tab1:
        st.header("Schedule and Predictions by Date")
        st.caption("All match dates are shown in the host city's local date.")

        # Build a lookup for completed matches.
        actual_map = {}
        for res in actual_results:
            key = frozenset([res["team_a"], res["team_b"]])
            actual_map[key] = res

        # Schedule filters.
        import datetime
        today = datetime.date.today()
        start_wc = datetime.date(2026, 6, 11)
        end_wc = datetime.date(2026, 7, 19)

        # During the World Cup, default to today's matches; otherwise default to all dates.
        is_during_wc = start_wc <= today <= end_wc
        default_view_all = not is_during_wc

        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            stage_filter = st.selectbox("Stage", ["Knockout", "All", "Group Stage"])
        with col_f2:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            view_all = st.toggle("Show All Dates", value=default_view_all)

        # Select base match list for the chosen stage.
        if stage_filter == "Group Stage":
            stage_matches = [f for f in schedule if f["stage"] == "group-stage"]
        elif stage_filter == "Knockout":
            stage_matches = [f for f in schedule if f["stage"] != "group-stage"]
        else:
            stage_matches = schedule

        if not view_all:
            dates_str = sorted(list(set(f["date"] for f in stage_matches)))
            if dates_str:
                dates_obj = [datetime.date.fromisoformat(d) for d in dates_str]
                # Use today when it falls within the selected stage range; otherwise use the first match date.
                default_date = today if dates_obj[0] <= today <= dates_obj[-1] else dates_obj[0]
                selected_date = st.date_input(
                    "Select Date",
                    value=default_date,
                    min_value=dates_obj[0],
                    max_value=dates_obj[-1]
                )
                date_filter = selected_date.strftime("%Y-%m-%d")
            else:
                date_filter = "All Dates"
        else:
            date_filter = "All Dates"

        # Apply filters.
        filtered_schedule = stage_matches
        if date_filter != "All Dates":
            filtered_schedule = [f for f in filtered_schedule if f["date"] == date_filter]

        st.markdown(f"**{len(filtered_schedule)} matches found.**")
        if len(filtered_schedule) == 0 and date_filter != "All Dates":
            st.info("No World Cup matches are scheduled on the selected date. Choose another date.")

        # Render match cards.
        for match in filtered_schedule:
            m_num = match["matchNumber"]
            m_date = match["date"]
            stage = "Group Stage" if match["stage"] == "group-stage" else "Knockout"
            group_info = f"Group {match['group']}" if match["group"] else ""
            stadium = match["stadium"]
            city = match["hostCity"].replace("-", " ").title()

            home = match["homeTeam"]
            away = match["awayTeam"]
            current_match = current_knockout_matches.get(m_num)
            if current_match:
                home = current_match["team_a"]
                away = current_match["team_b"]
            has_known_teams = home in elo_ratings and away in elo_ratings

            # Check whether the match has already been played.
            match_key = frozenset([home, away])
            has_actual = has_known_teams and match_key in actual_map

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
                    if home in elo_ratings:
                        st.markdown(f"<div style='text-align: right; color: #94a3b8; font-size: 0.9em;'>ELO: {elo_ratings[home]:.0f}</div>", unsafe_allow_html=True)
                    # Show absences.
                    team_inj = active_injuries.get(home, [])
                    if team_inj:
                        formatted_inj = format_absence_list_to_str_list(team_inj)
                        st.markdown(f"<div style='text-align: right; color: #f43f5e; font-size: 0.8em; margin-top: 2px;'>Absences: {', '.join(formatted_inj)}</div>", unsafe_allow_html=True)

                with c3:
                    st.markdown(f"<div style='font-size: 1.35em; font-weight: bold; text-align: left; margin-bottom: 2px;'>{get_flag(away)} {away}</div>", unsafe_allow_html=True)
                    if away in elo_ratings:
                        st.markdown(f"<div style='text-align: left; color: #94a3b8; font-size: 0.9em;'>ELO: {elo_ratings[away]:.0f}</div>", unsafe_allow_html=True)
                    # Show absences.
                    team_inj = active_injuries.get(away, [])
                    if team_inj:
                        formatted_inj = format_absence_list_to_str_list(team_inj)
                        st.markdown(f"<div style='text-align: left; color: #f43f5e; font-size: 0.8em; margin-top: 2px;'>Absences: {', '.join(formatted_inj)}</div>", unsafe_allow_html=True)

                with c2:
                    if has_actual:
                        # Show actual match result.
                        actual = actual_map[match_key]
                        s_home = actual["score_a"] if actual["team_a"] == home else actual["score_b"]
                        s_away = actual["score_b"] if actual["team_a"] == home else actual["score_a"]
                        w_team = actual["winner"]
                        winner_text = f"Winner: {get_flag(w_team)} {w_team}" if w_team else "Draw"
                        st.markdown(
                            f"""
                            <div style='text-align: center;'>
                                <div style='color: #10b981; font-size: 1.8em; font-weight: bold; margin-bottom: 2px;'>{s_home} : {s_away}</div>
                                <p style='color: #64748b; font-size: 0.85em; font-weight: bold; margin: 0;'>Completed match ({winner_text})</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    elif has_known_teams:
                        # Calculate Elo prediction probabilities.
                        r_home = elo_ratings.get(home, 1500.0)
                        r_away = elo_ratings.get(away, 1500.0)

                        # 1. Apply host advantage.
                        if home in HOST_COUNTRIES and away not in HOST_COUNTRIES:
                            r_home += 40
                        elif away in HOST_COUNTRIES and home not in HOST_COUNTRIES:
                            r_away += 40

                        # 2. Apply rest-day adjustment.
                        states = match_states_cache.get(m_num, {"rest_days_diff": 0, "travel_fatigue_a": 0.0, "travel_fatigue_b": 0.0})
                        rest_diff = states["rest_days_diff"]
                        rest_bonus = min(abs(rest_diff) * 5, 30)
                        if rest_diff >= 1:
                            r_home += rest_bonus
                        elif rest_diff <= -1:
                            r_away += rest_bonus

                        # 3. Convert to Poisson lambdas.
                        elo = EloSystem()
                        expected_home = elo.expected_score(r_home, r_away)
                        l_home, l_away = win_prob_to_lambda(expected_home)

                        # 4. Apply absence adjustments and 5. travel fatigue.
                        att_m_home, def_m_home, _ = get_injury_multipliers(home, active_injuries)
                        att_m_away, def_m_away, _ = get_injury_multipliers(away, active_injuries)
                        fat_home = states["travel_fatigue_a"]
                        fat_away = states["travel_fatigue_b"]

                        final_l_home = l_home * att_m_home * def_m_away * (1.0 - fat_home)
                        final_l_away = l_away * att_m_away * def_m_home * (1.0 - fat_away)

                        # Analyze match probabilities.
                        prob = match_probabilities(final_l_home, final_l_away)
                        p_win = prob["win"] * 100
                        p_draw = prob["draw"] * 100
                        p_lose = prob["lose"] * 100

                        # Estimate modal scoreline with 10,000 simulations.
                        sa_s = np.random.poisson(final_l_home, 10000)
                        sb_s = np.random.poisson(final_l_away, 10000)
                        sc_counts = Counter(zip(sa_s, sb_s))
                        (b_sa, b_sb), _ = max(sc_counts.items(), key=lambda x: x[1])

                        # Probability gauge.
                        st.markdown(
                            f"""
                            <div style='text-align: center; margin-bottom: 5px; font-weight: bold;'>
                                AI prediction: {get_flag(home)} {home} win {p_win:.1f}% | Draw {p_draw:.1f}% | {get_flag(away)} {away} win {p_lose:.1f}%
                            </div>
                            <div style="display: flex; height: 20px; border-radius: 10px; overflow: hidden; border: 1px solid #475569; margin-bottom: 5px;">
                                <div style="width: {p_win}%; background-color: #0ea5e9; text-align: center; color: white; font-size: 0.8em; line-height: 18px;">{p_win:.0f}%</div>
                                <div style="width: {p_draw}%; background-color: #64748b; text-align: center; color: white; font-size: 0.8em; line-height: 18px;">{p_draw:.0f}%</div>
                                <div style="width: {p_lose}%; background-color: #ec4899; text-align: center; color: white; font-size: 0.8em; line-height: 18px;">{p_lose:.0f}%</div>
                            </div>
                            <div style='text-align: center; color: #38bdf8; font-size: 0.85em; font-weight: bold;'>
                                Most likely score: {b_sa} - {b_sb}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            """
                            <div style='text-align: center; color: #94a3b8; font-size: 0.95em; font-weight: bold; padding: 18px 0;'>
                                Matchup not yet confirmed
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            st.markdown("<hr style='border: 0.5px solid #334155; margin-top:20px; margin-bottom:20px;'>", unsafe_allow_html=True)


    # ----------------- Tab 2: Final group standings -----------------
    with tab2:
        st.header("Final Group Standings")
        st.markdown("Final standings based on completed group-stage results. These are retained as the source data for knockout bracket generation.")

        # Load actual results and group data.
        groups_dict = load_json(GROUPS_PATH)
        actual_results = load_json(ACTUAL_RESULTS_PATH, [])

        # Initialize actual records by team.
        standings = {}
        for group_name, teams in groups_dict.items():
            standings[group_name] = {
                team: {"pts": 0, "w": 0, "d": 0, "l": 0, "gd": 0, "gf": 0, "ga": 0}
                for team in teams
            }

        # Aggregate completed group matches.
        for match in actual_results:
            if match.get("stage", "group") == "group":
                team_a = match["team_a"]
                team_b = match["team_b"]
                score_a = match["score_a"]
                score_b = match["score_b"]

                # Find the group.
                g_name_a = None
                for g_name, teams in groups_dict.items():
                    if team_a in teams:
                        g_name_a = g_name
                        break

                g_name_b = None
                for g_name, teams in groups_dict.items():
                    if team_b in teams:
                        g_name_b = g_name
                        break

                if g_name_a and g_name_b and g_name_a == g_name_b:
                    group_name = g_name_a
                    # Update Team A.
                    standings[group_name][team_a]["gf"] += score_a
                    standings[group_name][team_a]["ga"] += score_b
                    standings[group_name][team_a]["gd"] += (score_a - score_b)

                    # Update Team B.
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

        # Sort and display each group.
        groups_list = sorted(list(groups_dict.keys()))

        for row_idx in range(4): # 4 rows, 3 groups per row.
            cols = st.columns(3)
            for col_idx in range(3):
                g_idx = row_idx * 3 + col_idx
                if g_idx < len(groups_list):
                    group_name = groups_list[g_idx]
                    teams_dict = standings[group_name]

                    # Sort by points, goal difference, then goals for.
                    sorted_teams = sorted(
                        teams_dict.items(),
                        key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"]),
                        reverse=True
                    )

                    # Build dataframe rows.
                    rows = []
                    for rank, (team, stats) in enumerate(sorted_teams, 1):
                        rows.append({
                            "Rank": rank,
                            "Team": f"{get_flag(team)} {team}",
                            "Pts": stats["pts"],
                            "W": stats["w"],
                            "D": stats["d"],
                            "L": stats["l"],
                            "GD": stats["gd"],
                            "GF": stats["gf"]
                        })
                    df_g = pd.DataFrame(rows)

                    with cols[col_idx]:
                        st.subheader(group_name)
                        st.dataframe(
                            df_g.style.hide(axis="index"),
                            width="stretch",
                            height=180
                        )



    # ----------------- Tab 3: Knockout bracket visualization -----------------
    with tab3:
        st.header("World Cup Knockout Bracket")
        st.markdown("The default bracket shows the currently confirmed tournament state. Completed matches display scores and winners; unplayed matches display team names only.")

        if current_knockout_error:
            st.error(f"Failed to calculate the current knockout bracket: {current_knockout_error}")
        else:
            st.iframe(render_static_bracket_html(current_knockout_results), height=620)

        if st.button("Generate Projected Knockout Bracket", key="generate_projected_knockout_bracket"):
            try:
                with st.spinner(f"Running {KNOCKOUT_CONSENSUS_RUNS:,} tournament-projection simulations..."):
                    projected_knockout_results, projected_knockout_matches = build_knockout_projection()
                st.session_state["projected_knockout_results"] = projected_knockout_results
                st.session_state["projected_knockout_matches"] = projected_knockout_matches
                st.session_state.pop("projected_knockout_error", None)
            except Exception as exc:
                projected_knockout_results = None
                projected_knockout_matches = {}
                st.session_state["projected_knockout_error"] = str(exc)
            st.session_state["show_projected_knockout_bracket"] = True

        if st.session_state.get("show_projected_knockout_bracket"):
            st.subheader("Projected Knockout Bracket")
            st.markdown(f"This bracket simulates each remaining knockout match {KNOCKOUT_CONSENSUS_RUNS:,} times with current Elo ratings, absences, rest days, and travel fatigue, then shows the most frequent advancing team.")

            projected_knockout_results = st.session_state.get("projected_knockout_results")
            projected_knockout_matches = st.session_state.get("projected_knockout_matches", {})
            knockout_projection_error = st.session_state.get("projected_knockout_error")

            if knockout_projection_error:
                st.error(f"Failed to calculate the projected knockout bracket: {knockout_projection_error}")
            elif not projected_knockout_results:
                st.info("Click the button above to generate the projected bracket.")
            else:
                ko_results = projected_knockout_results

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

                    # Calculate detailed team data for the modal.
                    team_a = m["team_a"]
                    team_b = m["team_b"]
                    r_a = elo_ratings.get(team_a, 1500.0)
                    r_b = elo_ratings.get(team_b, 1500.0)

                    # Calculate absence-based strength reductions.
                    att_m_a, def_m_a, _ = get_injury_multipliers(team_a, active_injuries)
                    att_m_b, def_m_b, _ = get_injury_multipliers(team_b, active_injuries)

                    # Apply host advantage and rest-day bonus.
                    is_host_a = team_a in HOST_COUNTRIES
                    is_host_b = team_b in HOST_COUNTRIES

                    r_a_adj = r_a + (40 if is_host_a and not is_host_b else 0)
                    r_b_adj = r_b + (40 if is_host_b and not is_host_a else 0)

                    rest_days_diff = m.get("rest_days_diff", 0)
                    rest_bonus = min(abs(rest_days_diff) * 5, 30)
                    if rest_days_diff >= 1:
                        r_a_adj += rest_bonus
                    elif rest_days_diff <= -1:
                        r_b_adj += rest_bonus

                    # Calculate Elo-based expected goals.
                    elo_sys = EloSystem()
                    expected_a = elo_sys.expected_score(r_a_adj, r_b_adj)
                    l_a, l_b = win_prob_to_lambda(expected_a)

                    fatigue_a = m.get("travel_fatigue_a", 0.0)
                    fatigue_b = m.get("travel_fatigue_b", 0.0)
                    final_l_a = l_a * att_m_a * def_m_b * (1.0 - fatigue_a)
                    final_l_b = l_b * att_m_b * def_m_a * (1.0 - fatigue_b)

                    # Calculate win/draw/loss probabilities.
                    prob = match_probabilities(final_l_a, final_l_b)
                    p_win = prob["win"] * 100
                    p_draw = prob["draw"] * 100
                    p_lose = prob["lose"] * 100

                    # Format absence lists.
                    inj_a_str = ",".join(format_absence_list_to_str_list(active_injuries.get(team_a, [])))
                    inj_b_str = ",".join(format_absence_list_to_str_list(active_injuries.get(team_b, [])))

                    # Escape quotes in JavaScript parameters.
                    t_a_esc = team_a.replace("'", "\\'")
                    t_b_esc = team_b.replace("'", "\\'")
                    inj_a_esc = inj_a_str.replace("'", "\\'")
                    inj_b_esc = inj_b_str.replace("'", "\\'")
                    w_esc = m["winner"].replace("'", "\\'") if m["winner"] else ""
                    advance_prob_a = m.get("advance_prob_a")
                    advance_prob_b = m.get("advance_prob_b")
                    winner_prob = 0.0
                    if m["winner"] == team_a and advance_prob_a is not None:
                        winner_prob = advance_prob_a * 100
                    elif m["winner"] == team_b and advance_prob_b is not None:
                        winner_prob = advance_prob_b * 100
                    consensus_runs = m.get("consensus_runs") or 0

                    onclick_attr = f"onclick=\"showMatchDetail({m_id}, '{t_a_esc}', {m['score_a']}, '{t_b_esc}', {m['score_b']}, {p_win:.2f}, {p_draw:.2f}, {p_lose:.2f}, {r_a_adj:.1f}, {r_b_adj:.1f}, {final_l_a:.2f}, {final_l_b:.2f}, '{inj_a_esc}', '{inj_b_esc}', {str(m['is_pk']).lower()}, '{w_esc}', {winner_prob:.2f}, {consensus_runs})\""

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
                col1_html = "".join([gen_card_html(r32[i], "R32", 73 + i) for i in [0, 2, 1, 4, 3, 5, 6, 7]])
                col2_html = "".join([gen_card_html(r16[i], "R16", 89 + i) for i in [0, 1, 2, 3]])
                col3_html = "".join([gen_card_html(qf[i], "QF", 97 + i) for i in [0, 1]])
                col4_html = gen_card_html(sf[0], "SF", 101)

                # Center Column (Final & Champ)
                col5_html = f"""
                {gen_card_html(final[0], "Final", 104)}
                <div class="champion-card">
                    <div class="champion-title">CHAMPION</div>
                    <div class="champion-name">{champion}</div>
                </div>
                """

                # Right Bracket Column HTML
                col6_html = gen_card_html(sf[1], "SF", 102)
                col7_html = "".join([gen_card_html(qf[i], "QF", 97 + i) for i in [2, 3]])
                col8_html = "".join([gen_card_html(r16[i], "R16", 89 + i) for i in [4, 5, 6, 7]])
                col9_html = "".join([gen_card_html(r32[i], "R32", 73 + i) for i in [10, 11, 8, 9, 13, 15, 12, 14]])

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

                    <!-- Modal layer -->
                    <div id="match-modal" class="modal-overlay" onclick="closeModal()">
                        <div class="modal-content" onclick="event.stopPropagation()">
                            <span class="close-btn" onclick="closeModal()">&times;</span>
                            <div id="modal-body"></div>
                        </div>
                    </div>

                    <script>
                        function showMatchDetail(matchId, teamA, scoreA, teamB, scoreB, winProbA, winProbDraw, winProbB, eloA, eloB, lambdaA, lambdaB, injuriesA, injuriesB, isPk, winner, winnerProb, consensusRuns) {{
                            const modal = document.getElementById('match-modal');
                            const body = document.getElementById('modal-body');

                            const pkSuffix = isPk ? " (PK)" : "";
                            const runSuffix = consensusRuns > 0 ? ` (${{Math.round(winnerProb)}}%, ${{consensusRuns.toLocaleString()}} runs)` : "";
                            const winnerText = winner ? "Consensus advancing team: " + winner + pkSuffix + runSuffix : "Draw";

                            const injAList = injuriesA ? injuriesA.split(',') : [];
                            const injBList = injuriesB ? injuriesB.split(',') : [];
                            const injAHtml = injAList.length > 0 ? injAList.map(n => `<span style="color:#f43f5e">${{n}}</span>`).join(', ') : 'None';
                            const injBHtml = injBList.length > 0 ? injBList.map(n => `<span style="color:#f43f5e">${{n}}</span>`).join(', ') : 'None';

                            body.innerHTML = `
                                <div style="font-size: 11px; font-weight: bold; color: #38bdf8; margin-bottom: 12px; text-align: center;">[MATCH ${{matchId}}] Simulation Prediction Report</div>
                                <div class="modal-teams">
                                    <div class="modal-team-box">
                                        <div class="modal-team-name">${{teamA}}</div>
                                        <div style="font-size: 9.5px; color: #94a3b8;">Adjusted Elo: ${{Math.round(eloA)}}</div>
                                    </div>
                                    <div class="modal-score">${{scoreA}} : ${{scoreB}}</div>
                                    <div class="modal-team-box">
                                        <div class="modal-team-name">${{teamB}}</div>
                                        <div style="font-size: 9.5px; color: #94a3b8;">Adjusted Elo: ${{Math.round(eloB)}}</div>
                                    </div>
                                </div>
                                <div style="text-align: center; font-size: 11.5px; font-weight: bold; color: #10b981; margin-bottom: 15px;">
                                    ${{winnerText}}
                                </div>

                                <div class="modal-stats">
                                    <div class="stat-row">
                                        <span class="stat-label">Average Expected Goals</span>
                                        <span>${{teamA}} <b>${{lambdaA}}</b> vs <b>${{lambdaB}}</b> ${{teamB}}</span>
                                    </div>
                                    <div class="stat-row">
                                        <span class="stat-label">Absent Players</span>
                                        <span style="text-align: right; max-width: 65%; word-break: break-all; font-size: 9.5px; line-height: 1.3;">
                                            ${{teamA}}: ${{injAHtml}}<br>
                                            ${{teamB}}: ${{injBHtml}}
                                        </span>
                                    </div>
                                </div>

                                <div style="font-size: 9.5px; font-weight: bold; color: #cbd5e1; margin-bottom: 4px; text-align: center;">90-Minute AI Win / Draw / Loss Probabilities</div>
                                <div class="gauge-container">
                                    <div class="gauge-bar" style="width: ${{winProbA}}%; background-color: #0ea5e9;">${{Math.round(winProbA)}}%</div>
                                    <div class="gauge-bar" style="width: ${{winProbDraw}}%; background-color: #64748b;">${{Math.round(winProbDraw)}}%</div>
                                    <div class="gauge-bar" style="width: ${{winProbB}}%; background-color: #ec4899;">${{Math.round(winProbB)}}%</div>
                                </div>
                                <div style="display: flex; justify-content: space-between; font-size: 8.5px; color: #94a3b8; margin-top: 4px; padding: 0 4px;">
                                    <span>${{teamA}} win</span>
                                    <span>Draw</span>
                                    <span>${{teamB}} win</span>
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
                st.iframe(html_code, height=620)


    # ----------------- Tab 4: Head-to-head match simulator -----------------
    with tab4:
        st.header("Head-to-Head Match Simulator")
        st.markdown("Select any two teams and adjust match-context variables such as host advantage, rest-day gap, and travel fatigue.")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            team_a = st.selectbox("First Team (Home)", sorted(list(elo_ratings.keys())), index=0, format_func=lambda x: f"{get_flag(x)} {x}")
        with col_t2:
            team_b = st.selectbox("Second Team (Away)", sorted(list(elo_ratings.keys())), index=1, format_func=lambda x: f"{get_flag(x)} {x}")

        st.markdown("### Match Context Adjustments")

        c_env1, c_env2, c_env3 = st.columns(3)
        with c_env1:
            home_advantage_opt = st.selectbox("Apply Co-Host Advantage", ["None", f"Team A ({get_flag(team_a)} {team_a}) is a co-host", f"Team B ({get_flag(team_b)} {team_b}) is a co-host"])
        with c_env2:
            rest_days_diff = st.slider("Rest Advantage (Team A rest days - Team B rest days)", min_value=-10, max_value=10, value=0, help="Positive values mean Team A has more rest; negative values mean Team B has more rest.")
        with c_env3:
            st.markdown("**Travel Fatigue Reduction (0% to 10%)**")
            fatigue_a = st.slider(f"{team_a} fatigue", min_value=0.0, max_value=0.10, value=0.0, step=0.015, format="%.3f")
            fatigue_b = st.slider(f"{team_b} fatigue", min_value=0.0, max_value=0.10, value=0.0, step=0.015, format="%.3f")

        if team_a == team_b:
            st.warning("Select two different teams to run a valid simulation.")
        else:
            if st.button("Compare Strength and Run AI Prediction"):
                # Load base Elo ratings.
                r_a = elo_ratings.get(team_a, 1500.0)
                r_b = elo_ratings.get(team_b, 1500.0)

                # Adjust Elo ratings.
                if home_advantage_opt == f"Team A ({get_flag(team_a)} {team_a}) is a co-host":
                    r_a += 40
                elif home_advantage_opt == f"Team B ({get_flag(team_b)} {team_b}) is a co-host":
                    r_b += 40

                rest_bonus = min(abs(rest_days_diff) * 5, 30)
                if rest_days_diff >= 1:
                    r_a += rest_bonus
                elif rest_days_diff <= -1:
                    r_b += rest_bonus

                # ELO Win probability
                elo = EloSystem()
                expected_win_a = elo.expected_score(r_a, r_b)

                # Convert to Poisson lambdas.
                l_a, l_b = win_prob_to_lambda(expected_win_a)

                # Load and apply absence adjustments.
                att_m_a, def_m_a, details_a = get_injury_multipliers(team_a, active_injuries)
                att_m_b, def_m_b, details_b = get_injury_multipliers(team_b, active_injuries)

                final_l_a = l_a * att_m_a * def_m_b * (1.0 - fatigue_a)
                final_l_b = l_b * att_m_b * def_m_a * (1.0 - fatigue_b)

                # Calculate win/draw/loss probabilities.
                prob = match_probabilities(final_l_a, final_l_b)
                p_win = prob["win"] * 100
                p_draw = prob["draw"] * 100
                p_lose = prob["lose"] * 100

                # Estimate the most likely scoreline with 100,000 simulations.
                sa_samples = np.random.poisson(final_l_a, 100000)
                sb_samples = np.random.poisson(final_l_b, 100000)
                score_counts = Counter(zip(sa_samples, sb_samples))
                (best_sa, best_sb), count = max(score_counts.items(), key=lambda x: x[1])
                score_prob = (count / 100000) * 100

                st.markdown("---")
                st.markdown(f"### {get_flag(team_a)} {team_a} vs {get_flag(team_b)} {team_b} Prediction Report")

                col_res1, col_res2 = st.columns([1, 1])
                with col_res1:
                    st.markdown(
                        f"""
                        <div style="background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155;">
                            <div style="font-size: 1.15em; font-weight: bold; color: #38bdf8; margin-bottom: 10px;">Adjusted Elo Ratings</div>
                            <p>{team_a}: <b>{r_a:.1f}</b> (base Elo: {elo_ratings.get(team_a):.0f})</p>
                            <p>{team_b}: <b>{r_b:.1f}</b> (base Elo: {elo_ratings.get(team_b):.0f})</p>
                            <div style="font-size: 1.15em; font-weight: bold; color: #38bdf8; margin-top: 15px; margin-bottom: 10px;">Expected Goals</div>
                            <p>{team_a} average expected goals: <b>{final_l_a:.2f}</b></p>
                            <p>{team_b} average expected goals: <b>{final_l_b:.2f}</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col_res2:
                    # Modal score and win/draw/loss probabilities.
                    st.markdown(
                        f"""
                        <div style="background-color: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155; height: 100%;">
                            <div style="font-size: 1.15em; font-weight: bold; color: #38bdf8; margin-bottom: 10px;">AI Prediction Summary</div>
                            <p style="font-size: 1.1em;">Most likely score: <b style="color: #38bdf8; font-size: 1.3em;">{team_a} {best_sa} : {best_sb} {team_b}</b> (about {score_prob:.1f}% probability)</p>
                            <p>{team_a} win probability: <b>{p_win:.1f}%</b></p>
                            <p>Draw probability: <b>{p_draw:.1f}%</b></p>
                            <p>{team_b} win probability: <b>{p_lose:.1f}%</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Print absence-reduction report.
                if details_a or details_b:
                    st.markdown("#### Applied Absence Strength Reductions")
                    if details_a:
                        st.markdown(f"**{team_a}** absences:")
                        for d in details_a:
                            st.markdown(f"  * {d}")
                    if details_b:
                        st.markdown(f"**{team_b}** absences:")
                        for d in details_b:
                            st.markdown(f"  * {d}")

                # Draw chart.
                st.markdown("#### Match Prediction Gauge")
                st.markdown(
                    f"""
                    <div style="display: flex; height: 30px; border-radius: 15px; overflow: hidden; border: 1px solid #475569; margin-top: 10px;">
                        <div style="width: {p_win}%; background-color: #0ea5e9; text-align: center; color: white; font-size: 1em; line-height: 28px; font-weight: bold;">{get_flag(team_a)} {team_a} win ({p_win:.1f}%)</div>
                        <div style="width: {p_draw}%; background-color: #64748b; text-align: center; color: white; font-size: 1em; line-height: 28px; font-weight: bold;">Draw ({p_draw:.1f}%)</div>
                        <div style="width: {p_lose}%; background-color: #ec4899; text-align: center; color: white; font-size: 1em; line-height: 28px; font-weight: bold;">{get_flag(team_b)} {team_b} win ({p_lose:.1f}%)</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # ----------------- Tab 5: Full tournament forecast -----------------
    with tab5:
        st.header("Monte Carlo Simulator")
        st.markdown(
            "This simulator combines Elo ratings, live absences, schedule context, and travel fatigue to run the full 2026 World Cup "
            "for the selected number of iterations, then calculates each team's survival probability by round."
        )

        # Place the slider and run button side by side.
        col1, col2 = st.columns([3, 1])
        with col1:
            sim_runs = st.slider(
                "Simulation Iterations",
                min_value=500,
                max_value=10000,
                value=2000,
                step=500,
                key="sim_runs_tab5"
            )
        with col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True) # Label spacer.
            run_btn = st.button(
                f"Run Simulation ({sim_runs:,} runs)",
                key="run_sim_tab5",
                use_container_width=True
            )

        if run_btn:
            with st.spinner("Running World Cup simulation... (about 2-5 seconds)"):
                # Initialize Elo object.
                elo_sys = EloSystem()
                elo_sys.load_ratings(ELO_PATH)

                # Initialize simulator.
                sim = WorldCupSimulation(
                    elo_system=elo_sys,
                    groups_file=GROUPS_PATH,
                    actual_results_file=ACTUAL_RESULTS_PATH,
                    absences_file=ABSENCES_PATH,
                    squads_file=SQUADS_PATH
                )

                # Create statistics trackers.
                r32_counts = defaultdict(int)
                r16_counts = defaultdict(int)
                qf_counts = defaultdict(int)
                sf_counts = defaultdict(int)
                final_counts = defaultdict(int)
                champion_counts = defaultdict(int)

                # Run simulation loop.
                for _ in range(sim_runs):
                    standings = sim.simulate_group_stage()
                    r32_teams = sim.get_advancing_teams(standings)
                    for t in r32_teams:
                        r32_counts[t] += 1

                    ko_results = sim.simulate_knockout_stage()

                    # Record Round of 16 teams.
                    for m in ko_results["Round of 32"]:
                        r16_counts[m["winner"]] += 1
                    # Record quarter-final teams.
                    for m in ko_results["Round of 16"]:
                        qf_counts[m["winner"]] += 1
                    # Record semi-final teams.
                    for m in ko_results["Quarter-finals"]:
                        sf_counts[m["winner"]] += 1
                    # Record finalist teams.
                    for m in ko_results["Semi-finals"]:
                        final_counts[m["winner"]] += 1
                    # Record champion.
                    champion_counts[ko_results["Champion"]] += 1

                # Build dataframe.
                all_countries = list(elo_ratings.keys())
                sim_records = []
                for t in all_countries:
                    sim_records.append({
                        "Team": f"{get_flag(t)} {t}",
                        "Base Elo": elo_ratings.get(t, 1500.0),
                        "R32 Probability": (r32_counts.get(t, 0) / sim_runs) * 100,
                        "R16 Probability": (r16_counts.get(t, 0) / sim_runs) * 100,
                        "QF Probability": (qf_counts.get(t, 0) / sim_runs) * 100,
                        "SF Probability": (sf_counts.get(t, 0) / sim_runs) * 100,
                        "Final Probability": (final_counts.get(t, 0) / sim_runs) * 100,
                        "Champion Probability": (champion_counts.get(t, 0) / sim_runs) * 100,
                    })

                df_sim = pd.DataFrame(sim_records)
                df_sim = df_sim.sort_values(by="Champion Probability", ascending=False).reset_index(drop=True)

                # Format decimals.
                formatted_df = df_sim.style.format({
                    "Base Elo": "{:.0f}",
                    "R32 Probability": "{:.1f}%",
                    "R16 Probability": "{:.1f}%",
                    "QF Probability": "{:.1f}%",
                    "SF Probability": "{:.1f}%",
                    "Final Probability": "{:.1f}%",
                    "Champion Probability": "{:.2f}%"
                })

                st.dataframe(formatted_df, width="stretch", height=600)


def _should_run_app():
    if __name__ == "__main__":
        return True
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx(suppress_warning=True) is not None


if _should_run_app():
    run_app()
