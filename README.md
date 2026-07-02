# 2026 World Cup Prediction Simulation

[Korean](README.ko.md) | English

- **Data source**: https://www.eloratings.net
- **Model**: Elo ratings, Poisson scoring model, and Monte Carlo simulation
- **Input data**:
  - `data/elo_ratings.json` initial Elo ratings by team
  - `data/groups.json` 12-group allocation for the 48-team format
  - `data/schedule.json` 104-match schedule and host-city metadata
  - `data/actual_results.json` completed match scores
  - `data/squads.json` squad and player market-value data
  - `data/absences.json` injury and suspension absences

## Overview

This project predicts the 2026 FIFA World Cup by combining Elo ratings with a Poisson scoring model. It can simulate the full 48-team tournament thousands of times, estimate each team's survival probability by round, and expose the same logic through a Streamlit dashboard and a command-line match predictor.

## Key Features

### 1. Single-Match Prediction (`src/poisson.py`)

- Converts the Elo strength gap between two teams into expected goals for each side.
- Uses `numpy.random.poisson` to sample realistic football scores such as `2-1` or `0-0`.

### 2. Group-Stage Simulation (`src/simulation.py`)

- Simulates the 12 groups from Group A to Group L, with four teams per group and six matches per group.
- Applies the FIFA 2026 tiebreaker order: head-to-head points, head-to-head goal difference, head-to-head goals scored, overall goal difference, overall goals scored, team conduct score, and FIFA ranking.

### 3. Round of 32 and Knockout Bracket

- Advances all group winners and runners-up plus the eight best third-place teams.
- Ranks third-place teams by points, goal difference, goals scored, team conduct score, and FIFA ranking.
- Uses the FIFA World Cup 2026 Regulations Annex C mapping in `data/third_place_annex_c.json` to assign third-place teams to Round of 32 slots.
- Simulates single-elimination matches with extra time after 90-minute draws and Elo-weighted penalties if extra time is still tied.

### 4. Live Data Sync (`fetch_data.py`)

- Fetches the latest Elo ratings and completed World Cup match results from eloratings.net.
- Syncs suspension data from Wikipedia into `data/absences.json`.
- Keeps completed group-stage and knockout matches fixed in future simulations, so real results override simulated scores.
- Forces registered knockout winners into the next round when actual results are available.

### 5. Head-to-Head Match Predictor (`predict_match.py`)

- Predicts win, draw, and loss probabilities for any two teams in the dataset.
- Runs a vectorized 1,000,000-sample Poisson simulation to estimate the most likely scoreline.
- Supports optional rest-day and travel-fatigue inputs from the command line.

### 6. Monte Carlo Tournament Simulation (`main.py`)

- Repeats the full tournament simulation, typically 10,000 times.
- Aggregates each team's probability of reaching the Round of 32, Round of 16, quarter-finals, semi-finals, final, and championship.

### 7. Streamlit Dashboard (`app.py`)

- Provides an interactive dashboard for the current tournament state.
- Includes absence management for injuries and suspensions.
- Shows the schedule, completed results, future match predictions, group standings, knockout bracket, head-to-head simulator, and full tournament Monte Carlo results.

## Mathematics and Formulas

The simulator combines two core models: Elo expected score and the Poisson distribution.

### 1. Elo Expected Score

For teams A and B with Elo ratings $R_A$ and $R_B$:

$$E_A = \frac{1}{1 + 10^{(R_B - R_A) / 400}}$$

- $E_A$ is team A's expected score between 0 and 1.
- A 400-point Elo gap gives the stronger team an expected score of about 90.9%.
- When actual World Cup results are imported, the project applies a World Cup K-factor of `60` so tournament results have a clear effect on ratings.

### 2. Expected Goals

The model converts Elo expected score into expected goals:

$$\lambda_A = \text{Base Goals} \times \left( \frac{E_A}{1 - E_A} \right)^{0.376}$$

$$\lambda_B = \text{Base Goals} \times \left( \frac{1 - E_A}{E_A} \right)^{0.376}$$

- $\lambda_A$ and $\lambda_B$ are each team's Poisson scoring parameters.
- `Base Goals` defaults to `1.35`, for an average match total near 2.7 goals.
- The exponent keeps draw rates closer to real World Cup levels while allowing stronger teams to produce higher scorelines.

### 3. Score Probability

For a team with expected goals $\lambda$, the probability of scoring exactly $k$ goals is:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

The simulator samples from this distribution with `numpy.random.poisson(lambda)`.

## Absence and Squad-Value Adjustment

The model adjusts team strength for missing players using squad market value and a concentration index.

### 1. Squad and Absence Data

- `fetch_injuries.py` parses the Wikipedia squad page and records squad data in `data/squads.json`.
- `fetch_suspensions.py` parses suspension records and writes structured absence entries to `data/absences.json`.
- `src/absences.py` normalizes absence data so the dashboard, CLI predictor, and full simulator share the same adjustment logic.
- Suspensions can be restored automatically once a team has played enough actual matches.

Standard `data/absences.json` shape:

```json
{
  "South Korea": ["Cho Yu-min"],
  "Mexico": [
    {
      "name": "César Montes",
      "type": "suspension",
      "reason": "yellow_cards",
      "served_at_count": 2
    }
  ]
}
```

### 2. Position Value Share

Each absent player's value is compared with the squad's total value in the same broad position group:

$$S_p = \frac{\text{Value}_p}{\sum_{i \in \text{Position}} \text{Value}_i}$$

Goalkeepers and defenders affect the defensive multiplier. Midfielders and forwards affect the attacking multiplier.

### 3. Team Concentration Index

The project uses the Herfindahl-Hirschman Index (HHI) to estimate star-player dependency:

$$H_{\text{team}} = \sum_{i=1}^{26} \left( \frac{\text{Value}_i}{\text{Total Value}_{\text{team}}} \right)^2$$

The normalized dependency factor is:

$$D_{\text{team}} = 0.2 + 0.8 \times \text{Normalized } H_{\text{team}}$$

Final position reduction:

$$\text{Reduction}_{\text{pos}} = \sum_{p \in \text{Absent}} S_p \times D_{\text{team}}$$

Teams with deep, balanced squads lose less strength from one missing player. Teams concentrated around a few stars lose more.

### 4. Final Multiplier Rules

- Attack multiplier: $\max(0.5, 1.0 - \text{attack reduction})$
- Defense multiplier: $\min(2.0, 1.0 + \text{defense reduction})$
- Final expected goals cross-apply both teams' attack and defense adjustments.

## Match-Context Adjustments

The simulator also accounts for tournament context.

### 1. Host Advantage

The co-hosts `USA`, `Mexico`, and `Canada` receive a temporary `+40` Elo boost when facing non-host teams.

### 2. Group-Stage Rotation

Teams that reach six points after two group matches receive a temporary attacking penalty in the third group match to model squad rotation.

### 3. Rest-Day Gap

Knockout matches apply a rest bonus of `+5` Elo per extra rest day, capped at `+30` Elo:

```text
rest_bonus = min(abs(rest_days_diff) * 5, 30)
```

### 4. Travel Fatigue

Host cities are grouped into five geographic regions:

- Region 1: Vancouver, Seattle, San Francisco, Los Angeles
- Region 2: Guadalajara, Monterrey, Mexico City
- Region 3: Dallas, Houston, Kansas City
- Region 4: Miami, Atlanta
- Region 5: Toronto, Boston, Philadelphia, New York/New Jersey

Fatigue rules:

- Same region: no penalty
- Nearby region move: attacking lambda reduced by 1.5%
- Cross-continent move: attacking lambda reduced by 3.0%

## Usage

```bash
# 1. Activate the virtual environment, if used
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Sync live Elo ratings, completed results, and suspension data
PYTHONPATH=. python3 fetch_data.py

# 4. Run the Streamlit dashboard
venv/bin/streamlit run app.py

# 5. Run the full Monte Carlo simulation
PYTHONPATH=. python3 main.py

# 6. Predict a single match
PYTHONPATH=. python3 predict_match.py "South Korea" "Mexico"
```

## Example Output

```text
[Monte Carlo Simulation Results] (sorted by championship probability)
Rank Team            Champion Final   SF      QF      R16     R32
----------------------------------------------------------------------
1    Argentina        31.2%   45.2%   69.1%   88.0%   95.6%  100.0%
2    Spain            23.6%   33.6%   59.0%   66.4%   87.5%  100.0%
3    France           16.7%   35.3%   50.0%   66.2%   89.8%  100.0%
4    England           6.7%   16.7%   32.1%   54.5%   80.6%  100.0%
5    Colombia          4.0%    8.6%   20.3%   50.6%   80.3%   99.8%
6    Portugal          3.2%    7.2%   19.6%   42.2%   75.6%  100.0%
7    Brazil            3.0%    8.6%   18.1%   34.8%   58.4%  100.0%
8    Netherlands       2.5%    8.5%   18.5%   41.5%   62.3%  100.0%
9    Norway            2.3%    8.5%   21.9%   42.1%   82.5%  100.0%
10   Germany           2.1%    7.6%   16.7%   29.9%   78.0%  100.0%
```
