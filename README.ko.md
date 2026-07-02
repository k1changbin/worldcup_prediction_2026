# 2026 월드컵 예측 시뮬레이션

[English](README.md) | 한국어

- **데이터 출처**: https://www.eloratings.net
- **모델**: Elo 레이팅, 포아송 득점 모델, 몬테카를로 시뮬레이션
- **입력 데이터**:
  - `data/elo_ratings.json`: 팀별 초기 Elo 레이팅
  - `data/groups.json`: 48개국 체제의 12개 조 편성
  - `data/schedule.json`: 104경기 일정 및 개최 도시 정보
  - `data/actual_results.json`: 완료된 경기 스코어
  - `data/squads.json`: 스쿼드 및 선수 시장가치 데이터
  - `data/absences.json`: 부상 및 징계 결장 데이터

## 개요

이 프로젝트는 Elo 레이팅과 포아송 득점 모델을 결합해 2026 FIFA 월드컵을 예측합니다. 48개국 전체 대회를 수천 번 반복 시뮬레이션하여 각 팀의 라운드별 생존 확률을 계산하고, 같은 로직을 Streamlit 대시보드와 커맨드라인 매치 예측기로 제공합니다.

## 주요 기능

### 1. 단일 경기 예측 (`src/poisson.py`)

- 두 팀의 Elo 전력 차이를 각 팀의 예상 득점으로 변환합니다.
- `numpy.random.poisson`을 사용해 `2-1`, `0-0` 같은 실제 축구 스코어 형태를 샘플링합니다.

### 2. 조별리그 시뮬레이션 (`src/simulation.py`)

- Group A부터 Group L까지 12개 조를 시뮬레이션합니다.
- FIFA 2026 타이브레이커 순서인 맞대결 승점, 맞대결 골득실, 맞대결 다득점, 전체 골득실, 전체 다득점, 팀 conduct 점수, FIFA 랭킹을 적용합니다.

### 3. 32강 및 토너먼트 브래킷

- 각 조 1위와 2위, 그리고 상위 3위 8팀을 32강에 진출시킵니다.
- 3위 팀은 승점, 골득실, 다득점, 팀 conduct 점수, FIFA 랭킹 순으로 정렬합니다.
- `data/third_place_annex_c.json`의 FIFA World Cup 2026 Regulations Annex C 매핑을 사용해 3위 팀을 32강 슬롯에 배정합니다.
- 90분 무승부 시 연장전을 진행하고, 연장 후에도 동점이면 Elo 기반 확률로 승부차기를 처리합니다.

### 4. 실시간 데이터 동기화 (`fetch_data.py`)

- eloratings.net에서 최신 Elo 레이팅과 완료된 월드컵 경기 결과를 가져옵니다.
- Wikipedia에서 징계 데이터를 동기화해 `data/absences.json`에 반영합니다.
- 실제 완료된 조별리그 및 토너먼트 경기는 이후 시뮬레이션에서 실제 스코어로 고정합니다.
- 실제 토너먼트 승자가 등록된 경우 해당 팀을 다음 라운드로 강제 진출시킵니다.

### 5. 1대1 매치 예측기 (`predict_match.py`)

- 데이터셋에 있는 임의의 두 팀에 대해 승리, 무승부, 패배 확률을 예측합니다.
- 1,000,000회 벡터화 포아송 시뮬레이션으로 가장 가능성이 높은 스코어를 추정합니다.
- 커맨드라인에서 휴식일 차이와 이동 피로도 입력을 선택적으로 지원합니다.

### 6. 몬테카를로 전체 대회 시뮬레이션 (`main.py`)

- 전체 대회를 보통 10,000회 반복 시뮬레이션합니다.
- 각 팀의 32강, 16강, 8강, 4강, 결승, 우승 확률을 집계합니다.

### 7. Streamlit 대시보드 (`app.py`)

- 현재 대회 상태를 인터랙티브 대시보드로 제공합니다.
- 부상 및 징계 결장 선수를 관리할 수 있습니다.
- 일정, 실제 결과, 향후 경기 예측, 조별 순위, 토너먼트 브래킷, 1대1 시뮬레이터, 전체 대회 몬테카를로 결과를 확인할 수 있습니다.

## 수학 모델

시뮬레이터는 Elo 기대값과 포아송 분포를 결합합니다.

### 1. Elo 기대값

팀 A와 팀 B의 Elo 레이팅이 각각 $R_A$, $R_B$일 때:

$$E_A = \frac{1}{1 + 10^{(R_B - R_A) / 400}}$$

- $E_A$는 팀 A의 기대값이며 0과 1 사이의 값입니다.
- Elo 차이가 400점이면 강팀의 기대값은 약 90.9%입니다.
- 실제 월드컵 결과를 반영할 때는 `K-factor = 60`을 적용해 대회 결과가 레이팅에 뚜렷하게 반영되도록 합니다.

### 2. 예상 득점

Elo 기대값을 예상 득점으로 변환합니다.

$$\lambda_A = \text{Base Goals} \times \left( \frac{E_A}{1 - E_A} \right)^{0.376}$$

$$\lambda_B = \text{Base Goals} \times \left( \frac{1 - E_A}{E_A} \right)^{0.376}$$

- $\lambda_A$, $\lambda_B$는 각 팀의 포아송 득점 모수입니다.
- `Base Goals`는 기본값 `1.35`를 사용하며, 경기당 총 득점 평균을 약 2.7골로 둡니다.
- 지수값은 무승부 비율을 실제 월드컵 수준에 가깝게 유지하면서 강팀의 다득점 가능성을 반영하기 위한 보정입니다.

### 3. 스코어 확률

예상 득점이 $\lambda$인 팀이 정확히 $k$골을 넣을 확률은 다음과 같습니다.

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

시뮬레이터는 `numpy.random.poisson(lambda)`로 이 분포에서 스코어를 샘플링합니다.

## 결장 및 스쿼드 가치 보정

모델은 선수 결장을 단순 수동 등급이 아니라 스쿼드 시장가치와 집중도 지수를 기반으로 반영합니다.

### 1. 스쿼드 및 결장 데이터

- `fetch_injuries.py`는 Wikipedia 스쿼드 페이지를 파싱해 `data/squads.json`을 생성합니다.
- `fetch_suspensions.py`는 징계 기록을 파싱해 `data/absences.json`에 구조화된 결장 데이터를 기록합니다.
- `src/absences.py`는 결장 데이터를 표준화해 대시보드, CLI 예측기, 전체 시뮬레이터가 같은 보정 로직을 사용하도록 합니다.
- 징계 선수는 실제 경기 수가 충분히 진행되면 자동 복귀될 수 있습니다.

표준 `data/absences.json` 예시:

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

### 2. 포지션별 가치 비중

결장 선수의 가치를 같은 포지션 그룹의 총 가치와 비교합니다.

$$S_p = \frac{\text{Value}_p}{\sum_{i \in \text{Position}} \text{Value}_i}$$

골키퍼와 수비수는 수비 배율에 영향을 주고, 미드필더와 공격수는 공격 배율에 영향을 줍니다.

### 3. 팀 집중도 지수

프로젝트는 허핀달-허쉬만 지수(HHI)를 사용해 스타 플레이어 의존도를 추정합니다.

$$H_{\text{team}} = \sum_{i=1}^{26} \left( \frac{\text{Value}_i}{\text{Total Value}_{\text{team}}} \right)^2$$

정규화된 의존도 계수:

$$D_{\text{team}} = 0.2 + 0.8 \times \text{Normalized } H_{\text{team}}$$

최종 포지션 전력 감소율:

$$\text{Reduction}_{\text{pos}} = \sum_{p \in \text{Absent}} S_p \times D_{\text{team}}$$

스쿼드가 깊고 균형 잡힌 팀은 한 명의 결장 영향이 작고, 가치가 소수 스타에게 몰린 팀은 결장 영향이 크게 반영됩니다.

### 4. 최종 배율 규칙

- 공격 배율: $\max(0.5, 1.0 - \text{attack reduction})$
- 수비 배율: $\min(2.0, 1.0 + \text{defense reduction})$
- 최종 예상 득점은 양 팀의 공격/수비 보정 배율을 교차 적용합니다.

## 경기 환경 보정

시뮬레이터는 대회 환경 변수도 반영합니다.

### 1. 개최국 이점

공동 개최국 `USA`, `Mexico`, `Canada`는 비개최국을 상대할 때 임시로 `+40` Elo 보정을 받습니다.

### 2. 조별리그 로테이션

조별리그 2경기 후 승점 6점으로 사실상 진출을 확정한 팀은 3차전에서 로테이션을 고려한 임시 공격 페널티를 받습니다.

### 3. 휴식일 차이

토너먼트 경기는 상대보다 하루 더 쉰 팀에 `+5` Elo를 부여하며, 최대 `+30` Elo로 제한합니다.

```text
rest_bonus = min(abs(rest_days_diff) * 5, 30)
```

### 4. 이동 피로도

개최 도시는 5개 지리 권역으로 분류됩니다.

- Region 1: Vancouver, Seattle, San Francisco, Los Angeles
- Region 2: Guadalajara, Monterrey, Mexico City
- Region 3: Dallas, Houston, Kansas City
- Region 4: Miami, Atlanta
- Region 5: Toronto, Boston, Philadelphia, New York/New Jersey

피로도 규칙:

- 같은 권역 이동: 페널티 없음
- 인접 권역 이동: 공격 람다 1.5% 감소
- 대륙 횡단 이동: 공격 람다 3.0% 감소

## 실행 방법

```bash
# 1. 가상환경 활성화
source venv/bin/activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 최신 Elo, 완료 경기 결과, 징계 데이터 동기화
PYTHONPATH=. python3 fetch_data.py

# 4. Streamlit 대시보드 실행
venv/bin/streamlit run app.py

# 5. 전체 몬테카를로 시뮬레이션 실행
PYTHONPATH=. python3 main.py

# 6. 단일 경기 예측
PYTHONPATH=. python3 predict_match.py "South Korea" "Mexico"
```

## 출력 예시

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
