import sys
import os
import json
from src.elo import EloSystem
from src.simulation import WorldCupSimulation

def run_monte_carlo(iterations=10000):
    elo = EloSystem()
    elo.load_ratings("data/elo_ratings.json")
    
    actual_results_path = "data/actual_results.json"
    if os.path.exists(actual_results_path):
        try:
            with open(actual_results_path, "r", encoding="utf-8") as f:
                ar = json.load(f)
                if ar:
                    print(f"[Actual results locked] ({len(ar)} matches)...")
        except json.JSONDecodeError as e:
            print(f"[Error] Failed to load actual results: {e}")
            
    sim = WorldCupSimulation(elo, "data/groups.json", actual_results_path)
    
    # Initialize statistics for all 48 participating teams.
    stats = {}
    for group_teams in sim.groups.values():
        for team in group_teams:
            stats[team] = {"R32": 0, "R16": 0, "QF": 0, "SF": 0, "F": 0, "Champion": 0}
    
    print(f"[Simulation started] Running {iterations} Monte Carlo simulations... (estimated 10-30 seconds)")
    
    for i in range(iterations):
        # 1. Simulate group stage.
        standings = sim.simulate_group_stage()
        
        # 2. Resolve Round of 32 teams.
        advancing_teams = sim.get_advancing_teams(standings)
        for team in advancing_teams:
            stats[team]["R32"] += 1
            
        # 3. Simulate knockout stage.
        knockout_results = sim.simulate_knockout_stage()
        
        # 4. Aggregate results.
        for match in knockout_results["Round of 32"]:
            stats[match["winner"]]["R16"] += 1
            
        for match in knockout_results["Round of 16"]:
            stats[match["winner"]]["QF"] += 1
            
        for match in knockout_results["Quarter-finals"]:
            stats[match["winner"]]["SF"] += 1
            
        for match in knockout_results["Semi-finals"]:
            stats[match["winner"]]["F"] += 1
            
        champion = knockout_results["Champion"]
        stats[champion]["Champion"] += 1
        
        # Print progress at 10% intervals.
        if iterations >= 10 and (i + 1) % (iterations // 10) == 0:
            print(f"Progress: {(i + 1) / iterations * 100:.0f}% complete")
            
    # Calculate probabilities and sort by Champion, Final, SF, QF, R16, and R32.
    print("\n[Monte Carlo Simulation Results] (sorted by championship probability)")
    print(f"{'Rank':<4} {'Team':<15} {'Champion':<9} {'Final':<7} {'SF':<7} {'QF':<7} {'R16':<7} {'R32':<7}")
    print("-" * 70)
    
    sorted_teams = sorted(
        stats.items(), 
        key=lambda x: (x[1]["Champion"], x[1]["F"], x[1]["SF"], x[1]["QF"], x[1]["R16"], x[1]["R32"]), 
        reverse=True
    )
    
    for rank, (team, s) in enumerate(sorted_teams, 1):
        champ_pct = s["Champion"] / iterations * 100
        f_pct = s["F"] / iterations * 100
        sf_pct = s["SF"] / iterations * 100
        qf_pct = s["QF"] / iterations * 100
        r16_pct = s["R16"] / iterations * 100
        r32_pct = s["R32"] / iterations * 100
        
        print(f"{rank:<4} {team:<15} {champ_pct:>5.1f}% {f_pct:>6.1f}% {sf_pct:>6.1f}% {qf_pct:>6.1f}% {r16_pct:>6.1f}% {r32_pct:>6.1f}%")

if __name__ == "__main__":
    run_monte_carlo(10000)
