import streamlit as st
import html
import json
import os
import pandas as pd

from src.poisson import (
    match_probabilities,
    modal_scoreline,
    poisson_prob,
)
from src.absences import (
    calculate_absence_multipliers,
    clean_served_suspensions as clean_served_suspensions_data,
    format_absence_list_to_str_list,
    get_absence_names,
    load_absences,
    save_absences,
    upsert_suspension,
)
from src.paths import data_path
from src.forecast import run_tournament_forecast
from src.factory import create_world_cup_simulation
from src.model_config import (
    NEAR_REGION_TRAVEL_PENALTY,
)
from src.tournament_state import (
    calculate_group_standings,
    filter_team_map,
    get_active_teams,
)

# Data paths.
ELO_PATH = str(data_path("elo_ratings.json"))
SQUADS_PATH = str(data_path("squads.json"))
ABSENCES_PATH = str(data_path("absences.json"))
SCHEDULE_PATH = str(data_path("schedule.json"))
ACTUAL_RESULTS_PATH = str(data_path("actual_results.json"))
GROUPS_PATH = str(data_path("groups.json"))
FIFA_RANKINGS_PATH = str(data_path("fifa_rankings.json"))
TEAM_CONDUCT_PATH = str(data_path("team_conduct_scores.json"))
THIRD_PLACE_ANNEX_PATH = str(data_path("third_place_annex_c.json"))
KNOCKOUT_CONSENSUS_RUNS = 10000
KNOCKOUT_DATA_PATHS = (
    ELO_PATH,
    GROUPS_PATH,
    ACTUAL_RESULTS_PATH,
    ABSENCES_PATH,
    SQUADS_PATH,
    SCHEDULE_PATH,
    FIFA_RANKINGS_PATH,
    TEAM_CONDUCT_PATH,
    THIRD_PLACE_ANNEX_PATH,
)


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

def get_file_signature(path):
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        return None, None
    return stat.st_mtime_ns, stat.st_size

def data_cache_key(*paths):
    return tuple((path, *get_file_signature(path)) for path in paths)

@st.cache_data(show_spinner=False)
def _load_json_cached(path, mtime_ns, size):
    if mtime_ns is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None

def load_json(path, default_val=None):
    if default_val is None:
        default_val = {}
    data = _load_json_cached(path, *get_file_signature(path))
    return default_val if data is None else data

def clear_data_caches():
    st.cache_data.clear()

def set_active_main_tab(tab_label):
    st.session_state["active_main_tab"] = tab_label

def clean_served_suspensions(injuries_dict, actual_results_list):
    cleaned, updated = clean_served_suspensions_data(injuries_dict, actual_results_list)
    if updated:
        save_absences(ABSENCES_PATH, cleaned)
        clear_data_caches()
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
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Load data.
    elo_ratings = load_json(ELO_PATH)
    squads = load_json(SQUADS_PATH)
    registered_absences = load_absences(ABSENCES_PATH)
    schedule = load_json(SCHEDULE_PATH)
    actual_results = load_json(ACTUAL_RESULTS_PATH)
    groups_dict = load_json(GROUPS_PATH)

    # Automatically restore served suspensions.
    registered_absences = clean_served_suspensions(registered_absences, actual_results)

    # Validate required data.
    if not elo_ratings or not groups_dict:
        st.error("Required data files (data/elo_ratings.json, data/groups.json) are missing or empty. Check the project setup.")
        st.stop()

    active_teams = get_active_teams(groups_dict, actual_results, elo_ratings=elo_ratings)
    if not active_teams:
        active_teams = set(elo_ratings.keys())
    match_model = create_world_cup_simulation()

    # ----------------- Sidebar absence-management panel -----------------
    st.sidebar.markdown("## Live Injury and Suspension Management")
    st.sidebar.info("Register player injuries or suspensions to apply them immediately across the prediction model.")

    # Wikipedia sync button.
    import fetch_suspensions
    if st.sidebar.button("Sync Live Suspension Data (Wikipedia)"):
        try:
            with st.sidebar.spinner("Fetching suspension records from Wikipedia..."):
                fetch_suspensions.main()
            clear_data_caches()
            clear_projected_knockout_cache()
            st.sidebar.success("Sync complete.")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Suspension sync failed: {exc}")

    # Show current injuries and suspensions.
    st.sidebar.markdown("### Current Absences")
    active_injuries = filter_team_map(registered_absences, active_teams)

    if not active_injuries or all(len(v) == 0 for v in active_injuries.values()):
        st.sidebar.text("No registered absences for active teams.")
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
                            "yellow_cards": "yellow-card accumulation suspension",
                            "disciplinary": "additional disciplinary suspension",
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
    all_teams = sorted(team for team in elo_ratings.keys() if team in active_teams)
    if not all_teams:
        all_teams = sorted(list(elo_ratings.keys()))
    selected_team = st.sidebar.selectbox("Select Team", all_teams, format_func=lambda x: f"{get_flag(x)} {x}")

    # Player selection.
    if selected_team in squads:
        team_players = [p["name"] for p in squads[selected_team]]
        # Exclude players that are already registered as absent.
        already_injured_names = get_absence_names(registered_absences.get(selected_team, []))
        available_players = [p for p in team_players if p not in already_injured_names]

        if available_players:
            selected_player = st.sidebar.selectbox("Select Player", available_players)
            absence_reason = st.sidebar.selectbox(
                "Absence Reason",
                [
                    "Injury",
                    "Yellow-card accumulation (1-match suspension)",
                    "Red card (1-match suspension)",
                    "Red card (2-match suspension)",
                ],
            )

            if st.sidebar.button("Add to Absence List"):
                if selected_team not in registered_absences:
                    registered_absences[selected_team] = []

                if absence_reason == "Injury":
                    registered_absences[selected_team].append({
                        "name": selected_player,
                        "type": "injury"
                    })
                else:
                    # Register suspension.
                    N = 0
                    for match in actual_results:
                        if match.get("team_a") == selected_team or match.get("team_b") == selected_team:
                            N += 1

                    suspension_length = 2 if "2-match" in absence_reason else 1
                    served_at = N + suspension_length
                    reason = "red_card" if absence_reason.startswith("Red card") else "yellow_cards"
                    registered_absences, _ = upsert_suspension(
                        registered_absences,
                        selected_team,
                        selected_player,
                        reason,
                        served_at,
                        suspension_length,
                    )

                save_absences(ABSENCES_PATH, registered_absences)
                clear_data_caches()
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
                for team, players_list in list(registered_absences.items()):
                    for p in list(players_list):
                        p_name = p.get("name") if isinstance(p, dict) else p
                        if f"{get_flag(team)} {team} - {p_name}" == opt:
                            players_list.remove(p)
                    if not registered_absences[team]:
                        del registered_absences[team]
            save_absences(ABSENCES_PATH, registered_absences)
            clear_data_caches()
            clear_projected_knockout_cache()
            st.rerun()
    else:
        st.sidebar.text("No absent players to restore.")


    # ----------------- Main dashboard layout -----------------
    st.title("2026 FIFA World Cup Tournament Prediction Dashboard")
    st.markdown("Explore the current knockout bracket, completed results, and future match predictions based on final group-stage outcomes.")

    refresh_message = st.session_state.pop("data_refresh_message", None)
    if refresh_message:
        if isinstance(refresh_message, dict) and refresh_message.get("level") == "warning":
            st.warning(refresh_message["text"])
        else:
            text = refresh_message.get("text") if isinstance(refresh_message, dict) else refresh_message
            st.success(text)

    refresh_col, _ = st.columns([1, 4])
    with refresh_col:
        if st.button("Refresh All Data", key="refresh_all_data", use_container_width=True):
            try:
                with st.spinner("Refreshing Elo ratings, actual results, and suspensions..."):
                    import fetch_data
                    refresh_result = fetch_data.fetch_live_world_cup_data()
                clear_data_caches()
                clear_projected_knockout_cache()
                if refresh_result["suspensions_updated"]:
                    message = {
                        "level": "success",
                        "text": "Elo ratings, results, and suspensions were refreshed.",
                    }
                else:
                    message = {
                        "level": "warning",
                        "text": (
                            "Elo ratings and results were refreshed, but suspension "
                            "sync failed; the previous suspension snapshot was kept."
                        ),
                    }
                st.session_state["data_refresh_message"] = message
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to refresh all data: {exc}")

    @st.cache_data(show_spinner=False)
    def build_knockout_projection(cache_key):
        sim = create_world_cup_simulation()

        standings = sim.simulate_group_stage()
        sim.get_advancing_teams(standings)
        ko_results = sim.simulate_knockout_stage(consensus_runs=KNOCKOUT_CONSENSUS_RUNS)

        match_map = {
            projected_match["match_id"]: projected_match
            for round_key in (
                "Round of 32",
                "Round of 16",
                "Quarter-finals",
                "Semi-finals",
                "Third-place",
                "Final",
            )
            for projected_match in ko_results[round_key]
        }

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
            "Third-place",
            "Final",
        ]:
            for match in ko_results.get(round_key, []):
                match_id = match.get("match_id")
                if match_id:
                    match_map[match_id] = match
        return match_map

    @st.cache_data(show_spinner=False)
    def build_current_knockout_state(cache_key):
        sim = create_world_cup_simulation()
        standings = sim.simulate_group_stage()
        return sim.build_current_knockout_state(standings)

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

    # Track the active tab in session state so button-triggered reruns
    # do not reset the view back to the first tab.
    tab3, tab1, tab2, tab4, tab5 = st.tabs(
        [
            "Knockout Bracket",
            "Schedule and Match Predictions",
            "Final Group Standings",
            "Head-to-Head Simulator",
            "Full Tournament Forecast",
        ],
        key="active_main_tab",
        on_change="rerun",
    )

    current_knockout_results = None
    current_knockout_error = None
    current_knockout_matches = {}
    if tab1.open is not False or tab3.open is not False:
        try:
            current_knockout_results = build_current_knockout_state(
                data_cache_key(*KNOCKOUT_DATA_PATHS)
            )
        except Exception as exc:
            current_knockout_error = exc
        current_knockout_matches = flatten_knockout_matches(current_knockout_results)

    # ----------------- Tab 1: Schedule and match predictions -----------------
    if tab1.open is not False:
        with tab1:
            st.header("Schedule and Predictions by Date")
            st.caption("All match dates are shown in the host city's local date.")

            # Build a lookup for completed matches.
            actual_by_number = {}
            actual_by_matchup = {}
            for res in actual_results:
                key = frozenset([res["team_a"], res["team_b"]])
                actual_by_matchup[key] = res
                if res.get("match_number") is not None:
                    actual_by_number[res["match_number"]] = res

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

            match_states_cache = match_model.build_group_stage_match_contexts()

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
                actual = actual_by_number.get(m_num)
                if actual is None and has_known_teams:
                    actual = actual_by_matchup.get(match_key)
                has_actual = actual is not None
                home_html = html.escape(str(home))
                away_html = html.escape(str(away))

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
                        st.markdown(f"<div style='font-size: 1.35em; font-weight: bold; text-align: right; margin-bottom: 2px;'>{get_flag(home)} {home_html}</div>", unsafe_allow_html=True)
                        if home in elo_ratings:
                            st.markdown(f"<div style='text-align: right; color: #94a3b8; font-size: 0.9em;'>ELO: {elo_ratings[home]:.0f}</div>", unsafe_allow_html=True)
                        # Show absences.
                        team_inj = active_injuries.get(home, [])
                        if team_inj:
                            formatted_inj = format_absence_list_to_str_list(team_inj)
                            injury_html = html.escape(", ".join(formatted_inj))
                            st.markdown(f"<div style='text-align: right; color: #f43f5e; font-size: 0.8em; margin-top: 2px;'>Absences: {injury_html}</div>", unsafe_allow_html=True)

                    with c3:
                        st.markdown(f"<div style='font-size: 1.35em; font-weight: bold; text-align: left; margin-bottom: 2px;'>{get_flag(away)} {away_html}</div>", unsafe_allow_html=True)
                        if away in elo_ratings:
                            st.markdown(f"<div style='text-align: left; color: #94a3b8; font-size: 0.9em;'>ELO: {elo_ratings[away]:.0f}</div>", unsafe_allow_html=True)
                        # Show absences.
                        team_inj = active_injuries.get(away, [])
                        if team_inj:
                            formatted_inj = format_absence_list_to_str_list(team_inj)
                            injury_html = html.escape(", ".join(formatted_inj))
                            st.markdown(f"<div style='text-align: left; color: #f43f5e; font-size: 0.8em; margin-top: 2px;'>Absences: {injury_html}</div>", unsafe_allow_html=True)

                    with c2:
                        if has_actual:
                            # Show actual match result.
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
                            states = current_match or match_states_cache.get(
                                m_num,
                                {
                                    "rest_days_diff": 0,
                                    "travel_fatigue_a": 0.0,
                                    "travel_fatigue_b": 0.0,
                                },
                            )
                            rest_diff = states["rest_days_diff"]
                            fat_home = states["travel_fatigue_a"]
                            fat_away = states["travel_fatigue_b"]
                            final_l_home, final_l_away = match_model.get_expected_goals(
                                home,
                                away,
                                home_advantage=True,
                                rest_days_diff=rest_diff,
                                travel_fatigue_a=fat_home,
                                travel_fatigue_b=fat_away,
                            )

                            # Analyze match probabilities.
                            prob = match_probabilities(final_l_home, final_l_away)
                            p_win = prob["win"] * 100
                            p_draw = prob["draw"] * 100
                            p_lose = prob["lose"] * 100

                            b_sa, b_sb = modal_scoreline(final_l_home, final_l_away)

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
    if tab2.open is not False:
        with tab2:
            st.header("Final Group Standings")
            st.markdown("Final standings based on completed group-stage results. These are retained as the source data for knockout bracket generation.")

            standings = calculate_group_standings(
                groups_dict,
                actual_results,
                load_json(FIFA_RANKINGS_PATH),
                load_json(TEAM_CONDUCT_PATH),
            )

            # Sort and display each group.
            groups_list = sorted(list(groups_dict.keys()))

            for row_idx in range(4): # 4 rows, 3 groups per row.
                cols = st.columns(3)
                for col_idx in range(3):
                    g_idx = row_idx * 3 + col_idx
                    if g_idx < len(groups_list):
                        group_name = groups_list[g_idx]
                        sorted_teams = standings[group_name]

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
    if tab3.open is not False:
        with tab3:
            st.header("World Cup Knockout Bracket")
            st.markdown("The default bracket shows the currently confirmed tournament state. Completed matches display scores and winners; unplayed matches display team names only.")

            if current_knockout_error:
                st.error(f"Failed to calculate the current knockout bracket: {current_knockout_error}")
            else:
                st.iframe(render_static_bracket_html(current_knockout_results), height=620)

            if st.button(
                "Generate Projected Knockout Bracket",
                key="generate_projected_knockout_bracket",
                on_click=set_active_main_tab,
                args=("Knockout Bracket",),
            ):
                try:
                    with st.spinner(f"Running {KNOCKOUT_CONSENSUS_RUNS:,} simulations per projected knockout match..."):
                        projected_knockout_results, projected_knockout_matches = build_knockout_projection(
                            data_cache_key(*KNOCKOUT_DATA_PATHS)
                        )
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
                        rest_days_diff = m.get("rest_days_diff", 0)
                        r_a_adj, r_b_adj = match_model.get_adjusted_ratings(
                            team_a,
                            team_b,
                            home_advantage=True,
                            rest_days_diff=rest_days_diff,
                        )

                        fatigue_a = m.get("travel_fatigue_a", 0.0)
                        fatigue_b = m.get("travel_fatigue_b", 0.0)
                        final_l_a, final_l_b = match_model.get_expected_goals(
                            team_a,
                            team_b,
                            home_advantage=True,
                            rest_days_diff=rest_days_diff,
                            travel_fatigue_a=fatigue_a,
                            travel_fatigue_b=fatigue_b,
                        )

                        # Calculate win/draw/loss probabilities.
                        prob = match_probabilities(final_l_a, final_l_b)
                        p_win = prob["win"] * 100
                        p_draw = prob["draw"] * 100
                        p_lose = prob["lose"] * 100

                        # Format absence lists.
                        inj_a_str = ",".join(format_absence_list_to_str_list(active_injuries.get(team_a, [])))
                        inj_b_str = ",".join(format_absence_list_to_str_list(active_injuries.get(team_b, [])))

                        advance_prob_a = m.get("advance_prob_a")
                        advance_prob_b = m.get("advance_prob_b")
                        winner_prob = 0.0
                        if m["winner"] == team_a and advance_prob_a is not None:
                            winner_prob = advance_prob_a * 100
                        elif m["winner"] == team_b and advance_prob_b is not None:
                            winner_prob = advance_prob_b * 100
                        consensus_runs = m.get("consensus_runs") or 0

                        onclick_script = (
                            "showMatchDetail("
                            f"{m_id}, {json.dumps(team_a)}, {m['score_a']}, "
                            f"{json.dumps(team_b)}, {m['score_b']}, "
                            f"{p_win:.2f}, {p_draw:.2f}, {p_lose:.2f}, "
                            f"{r_a_adj:.1f}, {r_b_adj:.1f}, "
                            f"{final_l_a:.2f}, {final_l_b:.2f}, "
                            f"{json.dumps(inj_a_str)}, {json.dumps(inj_b_str)}, "
                            f"{str(m['is_pk']).lower()}, "
                            f"{json.dumps(m['winner'] or '')}, "
                            f"{winner_prob:.2f}, {consensus_runs})"
                        )
                        onclick_attr = (
                            f'onclick="{html.escape(onclick_script, quote=True)}"'
                        )
                        team_a_html = html.escape(team_a)
                        team_b_html = html.escape(team_b)

                        return f"""
                        <div class="match-card" {onclick_attr}>
                            <div class="match-title">M{m_id} {round_name}</div>
                            <div class="team-row {win_a}">
                                <span class="team-name">{team_a_html}</span>
                                <span class="score">{m['score_a']}{pk_a}</span>
                            </div>
                            <div class="team-row {win_b}">
                                <span class="team-name">{team_b_html}</span>
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
                            <div class="champion-name">{html.escape(str(champion))}</div>
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
    if tab4.open is not False:
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
                apply_host_advantage = st.checkbox(
                    "Apply Official Co-Host Advantage",
                    value=True,
                    help="Applies only when USA, Mexico, or Canada faces a non-host team.",
                )
            with c_env2:
                rest_days_diff = st.slider("Rest Advantage (Team A rest days - Team B rest days)", min_value=-10, max_value=10, value=0, help="Positive values mean Team A has more rest; negative values mean Team B has more rest.")
            with c_env3:
                st.markdown("**Travel Fatigue Reduction (0% to 10%)**")
                fatigue_a = st.slider(f"{team_a} fatigue", min_value=0.0, max_value=0.10, value=0.0, step=NEAR_REGION_TRAVEL_PENALTY, format="%.3f")
                fatigue_b = st.slider(f"{team_b} fatigue", min_value=0.0, max_value=0.10, value=0.0, step=NEAR_REGION_TRAVEL_PENALTY, format="%.3f")

            if team_a == team_b:
                st.warning("Select two different teams to run a valid simulation.")
            else:
                if st.button(
                    "Compare Strength and Run AI Prediction",
                    on_click=set_active_main_tab,
                    args=("Head-to-Head Simulator",),
                ):
                    r_a, r_b = match_model.get_adjusted_ratings(
                        team_a,
                        team_b,
                        home_advantage=apply_host_advantage,
                        rest_days_diff=rest_days_diff,
                    )

                    # Load and apply absence adjustments.
                    _, _, details_a = get_injury_multipliers(team_a, active_injuries)
                    _, _, details_b = get_injury_multipliers(team_b, active_injuries)

                    final_l_a, final_l_b = match_model.get_expected_goals(
                        team_a,
                        team_b,
                        home_advantage=apply_host_advantage,
                        rest_days_diff=rest_days_diff,
                        travel_fatigue_a=fatigue_a,
                        travel_fatigue_b=fatigue_b,
                    )

                    # Calculate win/draw/loss probabilities.
                    prob = match_probabilities(final_l_a, final_l_b)
                    p_win = prob["win"] * 100
                    p_draw = prob["draw"] * 100
                    p_lose = prob["lose"] * 100

                    best_sa, best_sb = modal_scoreline(final_l_a, final_l_b)
                    score_prob = (
                        poisson_prob(final_l_a, best_sa)
                        * poisson_prob(final_l_b, best_sb)
                        * 100
                    )

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
    if tab5.open is not False:
        with tab5:
            st.header("Monte Carlo Simulator")
            st.markdown(
                "This simulator combines Elo ratings, live absences, schedule context, and travel fatigue to run the full 2026 World Cup "
                "for the selected number of iterations, then calculates each team's survival probability by round."
            )

            # Place iteration count and run button side by side.
            col1, col2 = st.columns([2, 1])
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
                    use_container_width=True,
                    on_click=set_active_main_tab,
                    args=("Full Tournament Forecast",),
                )

            if run_btn:
                with st.spinner("Running World Cup simulation... (about 2-5 seconds)"):
                    sim = create_world_cup_simulation()

                    all_countries = [team for team in elo_ratings.keys() if team in active_teams]
                    if not all_countries:
                        all_countries = list(elo_ratings.keys())
                    forecast = run_tournament_forecast(
                        sim,
                        sim_runs,
                        teams=all_countries,
                    )

                    sim_records = []
                    for team_forecast in forecast:
                        t = team_forecast["team"]
                        probabilities = team_forecast["probabilities"]
                        champion_low, champion_high = team_forecast[
                            "confidence_intervals"
                        ]["Champion"]
                        sim_records.append({
                            "Team": f"{get_flag(t)} {t}",
                            "Base Elo": elo_ratings.get(t, 1500.0),
                            "R32 Probability": probabilities["R32"] * 100,
                            "R16 Probability": probabilities["R16"] * 100,
                            "QF Probability": probabilities["QF"] * 100,
                            "SF Probability": probabilities["SF"] * 100,
                            "Final Probability": probabilities["F"] * 100,
                            "Champion Probability": probabilities["Champion"] * 100,
                            "Champion 95% CI": (
                                f"{champion_low * 100:.2f}%–{champion_high * 100:.2f}%"
                            ),
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
                    st.caption(
                        "The 95% intervals measure Monte Carlo sampling uncertainty only; "
                        "they do not include model or input-data uncertainty."
                    )


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
