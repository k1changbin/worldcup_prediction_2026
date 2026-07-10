import json
import sys
import math
from src.poisson import (
    elo_expected_score_to_lambdas,
    match_probabilities,
    modal_scoreline,
    poisson_prob,
)
from src.absences import calculate_absence_multipliers
from src.model_config import (
    HOST_ADVANTAGE_ELO,
    REST_ADVANTAGE_CAP,
    REST_ELO_PER_DAY,
)
from src.factory import create_world_cup_simulation
from src.paths import data_path
from src.simulation import HOST_COUNTRIES

def find_team(query, valid_teams):
    query = query.strip().lower()
    # 1. Exact match.
    for team in valid_teams:
        if team.lower() == query:
            return team
    # 2. Partial match.
    matches = [team for team in valid_teams if query in team.lower()]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        return matches
    return None

def draw_bar(pct, width=20):
    filled = int(round(width * (pct / 100)))
    return "█" * filled + "░" * (width - filled)

def predict_match():
    # Load Elo ratings.
    elo_path = data_path("elo_ratings.json")
    if not elo_path.exists():
        print("[Error] data/elo_ratings.json was not found.")
        return
        
    with open(elo_path, "r", encoding="utf-8") as f:
        ratings = json.load(f)
        valid_teams = set(ratings.keys())
        
    # Read command-line arguments or prompt interactively.
    team_a_query = ""
    team_b_query = ""
    
    rest_days_diff = 0
    travel_fatigue_a = 0.0
    travel_fatigue_b = 0.0
    
    if len(sys.argv) >= 3:
        team_a_query = sys.argv[1]
        team_b_query = sys.argv[2]
        
        # Parse optional arguments: rest-day gap, fatigue A, fatigue B.
        # Format: predict_match.py "Korea" "Spain" [rest_days_diff] [fatigue_a] [fatigue_b]
        if len(sys.argv) >= 4:
            try:
                rest_days_diff = int(sys.argv[3])
            except ValueError:
                pass
        if len(sys.argv) >= 6:
            try:
                travel_fatigue_a = float(sys.argv[4])
                travel_fatigue_b = float(sys.argv[5])
            except ValueError:
                pass
    else:
        print("[2026 World Cup Match Predictor]")
        print("Enter two participating teams to predict match probabilities.")
        team_a_query = input("First team name (English): ").strip()
        team_b_query = input("Second team name (English): ").strip()
        
    if not team_a_query or not team_b_query:
        print("[Error] Both team names are required.")
        return
    if not 0.0 <= travel_fatigue_a <= 1.0 or not 0.0 <= travel_fatigue_b <= 1.0:
        print("[Error] Travel-fatigue values must be between 0 and 1.")
        return
        
    team_a = find_team(team_a_query, valid_teams)
    team_b = find_team(team_b_query, valid_teams)
    
    # Handle ambiguous or failed matching.
    if isinstance(team_a, list):
        print(f"[Warning] The first team name is ambiguous. Did you mean one of these? {team_a}")
        return
    if isinstance(team_b, list):
        print(f"[Warning] The second team name is ambiguous. Did you mean one of these? {team_b}")
        return
    if not team_a:
        print(f"[Error] Could not find first team '{team_a_query}'. Check the participating-team spelling.")
        return
    if not team_b:
        print(f"[Error] Could not find second team '{team_b_query}'. Check the participating-team spelling.")
        return
    if team_a == team_b:
        print("[Error] Select two different teams.")
        return

    # Look up and calculate Elo values.
    simulation = create_world_cup_simulation()
    elo = simulation.elo_system

    is_host_a = team_a in HOST_COUNTRIES
    is_host_b = team_b in HOST_COUNTRIES
    home_adv_msg = ""
    if is_host_a and not is_host_b:
        home_adv_msg = (
            f" * Applied co-host advantage to {team_a} "
            f"(Elo +{HOST_ADVANTAGE_ELO:.0f})"
        )
    elif is_host_b and not is_host_a:
        home_adv_msg = (
            f" * Applied co-host advantage to {team_b} "
            f"(Elo +{HOST_ADVANTAGE_ELO:.0f})"
        )

    rest_bonus = min(
        abs(rest_days_diff) * REST_ELO_PER_DAY,
        REST_ADVANTAGE_CAP,
    )
    rating_a, rating_b = simulation.get_adjusted_ratings(
        team_a,
        team_b,
        home_advantage=True,
        rest_days_diff=rest_days_diff,
    )
    base_expected_score = elo.expected_score(
        elo.get_rating(team_a),
        elo.get_rating(team_b),
    )
    lambda_a, lambda_b = elo_expected_score_to_lambdas(base_expected_score)

    injuries = simulation.injuries
    squads = simulation.squads
                
    _, _, details_a = calculate_absence_multipliers(
        team_a,
        injuries,
        squads,
        include_values=True,
    )
    _, _, details_b = calculate_absence_multipliers(
        team_b,
        injuries,
        squads,
        include_values=True,
    )
    
    final_lambda_a, final_lambda_b = simulation.get_expected_goals(
        team_a,
        team_b,
        home_advantage=True,
        rest_days_diff=rest_days_diff,
        travel_fatigue_a=travel_fatigue_a,
        travel_fatigue_b=travel_fatigue_b,
    )
    
    # Calculate win/draw/loss probabilities from adjusted expected goals.
    result = match_probabilities(final_lambda_a, final_lambda_b)
    win_pct = result["win"] * 100
    draw_pct = result["draw"] * 100
    lose_pct = result["lose"] * 100
    
    # Print output.
    print("\n" + "=" * 55)
    print(f"[Match Prediction Result] {team_a} vs {team_b}")
    print("=" * 55)
    print(f"[Elo Rating] {team_a} ({elo.get_rating(team_a):.1f}) vs {team_b} ({elo.get_rating(team_b):.1f})")
    
    # Print contextual adjustments.
    if home_adv_msg or rest_days_diff != 0 or travel_fatigue_a > 0 or travel_fatigue_b > 0:
        print("-" * 55)
        print("[Context and Schedule Adjustments]")
        if home_adv_msg:
            print(home_adv_msg)
        if rest_days_diff > 0:
            print(f" * Applied rest advantage to {team_a} (+{rest_days_diff} days over opponent, Elo +{rest_bonus})")
        elif rest_days_diff < 0:
            print(f" * Applied rest advantage to {team_b} (+{-rest_days_diff} days over opponent, Elo +{rest_bonus})")
        if travel_fatigue_a > 0:
            print(f" * Applied travel fatigue/rotation to {team_a} (scoring reduction: -{travel_fatigue_a*100:.1f}%)")
        if travel_fatigue_b > 0:
            print(f" * Applied travel fatigue/rotation to {team_b} (scoring reduction: -{travel_fatigue_b*100:.1f}%)")
            
    # Print absence information.
    if details_a or details_b:
        print("-" * 55)
        print("[Absence and Strength Reduction]")

        if details_a:
            players_a = squads.get(team_a, [])
            total_raw_val_a = sum(p["value_eur"] for p in players_a)
            total_val_a = total_raw_val_a / 1000000
            hhi_a = sum((p["value_eur"] / total_raw_val_a) ** 2 for p in players_a) if total_raw_val_a > 0 else 0
            norm_hhi_a = max(0.0, min(1.0, (hhi_a - 0.0385) / (0.3 - 0.0385)))
            dependency_a = 0.2 + 0.8 * norm_hhi_a
            print(f" * {team_a} (squad value: €{total_val_a:.1f}M, star-dependency factor: {dependency_a:.2f}):")
            for detail in details_a:
                print(f"   - {detail}")
        if details_b:
            players_b = squads.get(team_b, [])
            total_raw_val_b = sum(p["value_eur"] for p in players_b)
            total_val_b = total_raw_val_b / 1000000
            hhi_b = sum((p["value_eur"] / total_raw_val_b) ** 2 for p in players_b) if total_raw_val_b > 0 else 0
            norm_hhi_b = max(0.0, min(1.0, (hhi_b - 0.0385) / (0.3 - 0.0385)))
            dependency_b = 0.2 + 0.8 * norm_hhi_b
            print(f" * {team_b} (squad value: €{total_val_b:.1f}M, star-dependency factor: {dependency_b:.2f}):")
            for detail in details_b:
                print(f"   - {detail}")
                
    print("-" * 55)
    print(f"[Average Expected Goals] {team_a}: {final_lambda_a:.2f} | {team_b}: {final_lambda_b:.2f}")
    
    # Print baseline expected goals only when adjustments changed the result.
    is_modified = (
        not math.isclose(rating_a, elo.get_rating(team_a)) or 
        not math.isclose(rating_b, elo.get_rating(team_b)) or
        not math.isclose(final_lambda_a, lambda_a) or
        not math.isclose(final_lambda_b, lambda_b)
    )
    if is_modified:
        base_expected_score = elo.expected_score(
            elo.get_rating(team_a),
            elo.get_rating(team_b),
        )
        base_lam_a, base_lam_b = elo_expected_score_to_lambdas(
            base_expected_score
        )
        print(f"   (baseline pure Elo - {team_a}: {base_lam_a:.2f} | {team_b}: {base_lam_b:.2f})")
        
    print("-" * 55)
    print(f"[{team_a} win] {win_pct:>5.1f}%  {draw_bar(win_pct)}")
    print(f"[Draw] {draw_pct:>5.1f}%  {draw_bar(draw_pct)}")
    print(f"[{team_b} win] {lose_pct:>5.1f}%  {draw_bar(lose_pct)}")
    print("-" * 55)
    
    best_sa, best_sb = modal_scoreline(final_lambda_a, final_lambda_b)
    score_prob = (
        poisson_prob(final_lambda_a, best_sa)
        * poisson_prob(final_lambda_b, best_sb)
        * 100
    )

    print("[Most Likely Scoreline (analytical Poisson mode)]")
    print(f"   * {team_a} {best_sa} - {best_sb} {team_b}  (about {score_prob:.1f}% probability)")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    predict_match()
