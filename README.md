# 2026 World Cup Prediction Simulation

[Korean](README.ko.md) | English

An Elo-to-Poisson match model and Monte Carlo simulator for the 48-team FIFA World Cup 2026. The same prediction engine powers the command-line tools and Streamlit dashboard.

This is an analytical project, not betting advice. Context adjustments and player market values remain modeling assumptions and should be interpreted with the calibration diagnostics described below.

## What the project does

- Simulates all 12 groups using the FIFA 2026 head-to-head-first tiebreak sequence.
- Advances 12 group winners, 12 runners-up, and the best eight third-place teams.
- Resolves the official 32-team bracket from `data/schedule.json` and the 495 Annex C third-place combinations.
- Locks completed results by stable `match_number`, including penalty decisions.
- Models extra time, penalties, co-host advantage, rest, travel, rotation, injuries, and suspensions.
- Produces stage probabilities and Wilson 95% Monte Carlo sampling intervals.
- Exposes schedule, results, standings, bracket, head-to-head predictions, and full forecasts in Streamlit.

## Project structure

```text
app.py                         Streamlit dashboard
main.py                        full-tournament Monte Carlo CLI
predict_match.py               head-to-head CLI
fetch_data.py                  Elo/result/suspension refresh
fetch_schedule.py              schedule refresh with schema validation
fetch_injuries.py              squad and injury refresh
fetch_calibration_data.py      2018/2022 leakage-free calibration dataset builder
evaluate_model.py              log-loss and Brier-score diagnostics
validate_data.py               cross-file data validation
src/poisson.py                 normalized Poisson probabilities and score modes
src/simulation.py              shared tournament and match-context engine
src/schedule.py                authoritative schedule/date/city index
src/bracket.py                 schedule-driven bracket source resolution
src/forecast.py                Monte Carlo aggregation and confidence intervals
src/evaluation.py              model evaluation utilities
src/model_config.py            production model parameters
tests/                         regression suite
```

## Data sources and snapshots

- [World Football Elo Ratings](https://www.eloratings.net/) for ratings and completed scores.
- ESPN scoreboard advancement flags for tied knockout matches.
- Wikipedia squad and disciplinary pages for squads, withdrawals, and suspensions.
- `data/schedule.json` as the checked-in 104-match schedule source of truth.
- FIFA World Cup 2026 Regulations Annex C, represented by `data/third_place_annex_c.json`.

Important files:

- `data/actual_results.json`: completed matches M1 onward, with scores, stage, winner, and `match_number`.
- `data/elo_ratings.json`: current ratings used for future predictions.
- `data/elo_ratings_pre_tournament.json`: May 2026 snapshot used only for leakage-free 2026 diagnostics.
- `data/model_calibration_matches.json`: 96 group matches from 2018 and 2022 with reconstructed pre-match Elo.
- `data/squads.json` and `data/absences.json`: active squad and absence inputs.

Data refreshes validate the full incoming snapshot before replacing local JSON. Writes use a temporary file, `fsync`, and `os.replace`; partial result feeds are merged without deleting locked matches, and conflicting past scores are rejected.

## Model

### Elo expected score

For ratings $R_A$ and $R_B$:

$$
E_A = \frac{1}{1 + 10^{(R_B-R_A)/400}}
$$

`E_A` is Elo expected score, not a literal win probability. Draw probability is introduced by the score model.

The project downloads current Elo values after completed matches. It does not apply a second local K-factor update to those downloaded ratings.

### Expected goals

Production parameters live in `src/model_config.py`:

$$
\lambda_A = 1.35 \left(\frac{E_A}{1-E_A}\right)^{0.25}
$$

$$
\lambda_B = 1.35 \left(\frac{1-E_A}{E_A}\right)^{0.25}
$$

The exponent was reduced from `0.376` to `0.25` after retrospective evaluation using reconstructed pre-match ratings from the 2018 and 2022 group stages. Multiclass log loss fell from `1.0709` to `1.0163`, and Brier score fell from `0.6124` to `0.5984`. On the current 2026 group-stage snapshot, the same comparison is `0.9575` to `0.9244` for log loss and `0.5531` to `0.5471` for Brier score. These are model diagnostics, not guarantees of future performance.

The optional grid search keeps the scoring baseline fixed and is deliberately reported as in-sample. With only two historical tournaments, year-specific optima are unstable, so it never rewrites production parameters automatically.

Run the reproducible report yourself:

```bash
python evaluate_model.py
python evaluate_model.py --grid-search
```

### Complete outcome probabilities

`src/poisson.py` expands the Poisson score support adaptively and normalizes the final win/draw/loss vector. Extreme Elo differences therefore still sum to exactly 100%; probability mass above 10 goals is no longer discarded.

The most likely scoreline is calculated analytically from the two Poisson modes, so the CLI and dashboard no longer need 100,000 or 1,000,000 random samples for that value.

### Context adjustments

- Modeled co-host advantage: `+40` Elo for USA, Mexico, or Canada against a non-host.
- Rest: `+5` Elo per additional rest day, capped at `+30`.
- Third-match rotation: a 20% attack reduction for a team already on six points.
- Travel: 1.5% attack reduction between nearby regions and 3% for long region moves.
- Absences: position-value share scaled by squad HHI concentration, with bounded attack/defense multipliers.

The group stage and knockout stage call the same `get_adjusted_ratings` and `get_expected_goals` APIs, so dashboard and Monte Carlo results use identical adjustments.

## Schedule and bracket correctness

`src/schedule.py` loads dates, cities, stages, and match numbers directly from `data/schedule.json`. `src/bracket.py` parses participant sources such as `Winner Match 74` instead of maintaining a second hard-coded bracket.

For example:

- M89 = winner M74 vs winner M77
- M90 = winner M73 vs winner M75
- M97 = winner M89 vs winner M90

Each advancing team's actual previous date and city are carried forward when calculating rest and travel context.

## Installation

Python 3.11 through 3.14 is accepted by the package metadata; CI currently tests 3.11, 3.12, and 3.13.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` contains only direct runtime dependencies and is aligned with `pyproject.toml`.

## Usage

```bash
# Validate every checked-in data file
python validate_data.py

# Refresh current Elo, results, and suspensions
python fetch_data.py

# Validate or migrate legacy results to stable match numbers without network use
python fetch_data.py --backfill-match-numbers --check-only
python fetch_data.py --backfill-match-numbers

# Full forecast
python main.py --iterations 10000

# Head-to-head forecast; optional values are rest gap, fatigue A, fatigue B
python predict_match.py "South Korea" "Mexico" 1 0.015 0

# Dashboard
streamlit run app.py
```

## Verification

The regression suite covers probability normalization, extreme inputs, analytical score modes, injected test RNGs, FIFA tiebreaks, Annex C, the complete bracket graph, M89/M90, schedule metadata, current bracket state, actual-result locking, tied semifinal decisions, atomic writes, partial feeds, data validation, and Monte Carlo confidence intervals.

```bash
python -m unittest discover -s tests -v
python validate_data.py
python -m compileall -q . -x '(^|/)(venv|\.git|scratch)/'
```

GitHub Actions runs data validation, the full test suite, dependency checks, source compilation, and an application import check on Python 3.11, 3.12, and 3.13.

The reported Wilson intervals quantify finite Monte Carlo sampling error only. They do not represent uncertainty in Elo ratings, model parameters, injuries, or other input assumptions.
