"""Configured object factories shared by the dashboard and CLIs."""

from src.elo import EloSystem
from src.paths import data_path
from src.simulation import WorldCupSimulation


def create_world_cup_simulation():
    elo = EloSystem()
    elo.load_ratings(data_path("elo_ratings.json"))
    return WorldCupSimulation(
        elo_system=elo,
        groups_file=data_path("groups.json"),
        actual_results_file=data_path("actual_results.json"),
        absences_file=data_path("absences.json"),
        squads_file=data_path("squads.json"),
        fifa_rankings_file=data_path("fifa_rankings.json"),
        team_conduct_file=data_path("team_conduct_scores.json"),
        third_place_annex_file=data_path("third_place_annex_c.json"),
        schedule_file=data_path("schedule.json"),
    )
