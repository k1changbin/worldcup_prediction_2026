"""Documented production parameters for the prediction model."""

# The scoring baseline remains 1.35 goals per team. The Elo-to-goals exponent
# was reduced after diagnostics built from pre-match 2018 and 2022 ratings.
# It improves W/D/L log loss over the previous 0.376 on those matches and on
# the separately reported 2026 group-stage snapshot.
BASE_GOALS = 1.35
ELO_LAMBDA_EXPONENT = 0.25

HOST_ADVANTAGE_ELO = 40.0
REST_ELO_PER_DAY = 5.0
REST_ADVANTAGE_CAP = 30.0
ROTATION_ATTACK_PENALTY = 0.20
NEAR_REGION_TRAVEL_PENALTY = 0.015
LONG_REGION_TRAVEL_PENALTY = 0.03
