# 2026 월드컵 예측 시뮬레이션

[English](README.md) | 한국어

48개국 체제의 2026 FIFA 월드컵을 위한 Elo-포아송 경기 모델 및 몬테카를로 시뮬레이터입니다. 커맨드라인 도구와 Streamlit 대시보드가 동일한 예측 엔진을 사용합니다.

이 프로젝트는 분석 도구이며 베팅 조언이 아닙니다. 경기 환경 보정과 선수 시장가치는 모델링 가정이므로 아래의 보정 진단 결과와 함께 해석해야 합니다.

## 주요 기능

- FIFA 2026의 맞대결 우선 타이브레이커로 12개 조를 시뮬레이션합니다.
- 조 1·2위 24팀과 상위 3위 8팀을 32강에 진출시킵니다.
- `data/schedule.json`과 Annex C의 495개 조합으로 공식 브래킷을 구성합니다.
- 완료 경기를 안정적인 `match_number`로 고정하고 승부차기 승자까지 반영합니다.
- 연장전, 승부차기, 개최국 이점, 휴식, 이동, 로테이션, 부상, 징계를 반영합니다.
- 라운드별 진출 확률과 Wilson 95% 몬테카를로 표본 구간을 제공합니다.
- 일정, 실제 결과, 조 순위, 브래킷, 1대1 예측, 전체 대회 전망을 Streamlit에서 제공합니다.

## 프로젝트 구조

```text
app.py                         Streamlit 대시보드
main.py                        전체 대회 몬테카를로 CLI
predict_match.py               1대1 경기 예측 CLI
fetch_data.py                  Elo·결과·징계 동기화
fetch_schedule.py              검증을 포함한 일정 동기화
fetch_injuries.py              스쿼드·부상 동기화
fetch_calibration_data.py      2018·2022 보정 데이터 생성기
evaluate_model.py              log loss·Brier score 진단
validate_data.py               데이터 교차 검증
src/poisson.py                 정규화된 포아송 확률과 최빈 스코어
src/simulation.py              공통 대회·경기 환경 엔진
src/schedule.py                일정·날짜·도시 단일 인덱스
src/bracket.py                 일정 기반 브래킷 연결
src/forecast.py                몬테카를로 집계와 신뢰구간
src/evaluation.py              모델 평가 도구
src/model_config.py            운영 모델 계수
tests/                         회귀 테스트
```

## 데이터 출처와 스냅샷

- [World Football Elo Ratings](https://www.eloratings.net/): Elo와 완료 경기 스코어
- ESPN scoreboard: 무승부 토너먼트 경기의 명시적 진출 팀 정보
- Wikipedia 스쿼드·징계 페이지: 선수 명단, 교체, 출장 정지
- `data/schedule.json`: 104경기 일정의 저장소 내 단일 기준
- `data/third_place_annex_c.json`: FIFA 2026 규정 Annex C 매핑

중요 데이터 파일:

- `data/actual_results.json`: 경기 번호, 스코어, 단계, 승자를 포함한 완료 경기
- `data/elo_ratings.json`: 향후 경기 예측에 사용하는 현재 Elo
- `data/elo_ratings_pre_tournament.json`: 2026 평가에만 사용하는 2026년 5월 대회 전 Elo
- `data/model_calibration_matches.json`: 경기 전 Elo를 복원한 2018·2022 조별리그 96경기
- `data/squads.json`, `data/absences.json`: 활성 팀의 스쿼드와 결장 정보

데이터 갱신은 전체 입력을 검증한 뒤에만 로컬 JSON을 교체합니다. 임시 파일, `fsync`, `os.replace`를 사용하며, 일부 경기만 내려온 경우에도 기존 완료 경기를 삭제하지 않습니다. 이미 고정된 과거 스코어와 충돌하는 입력은 거부합니다.

## 예측 모델

### Elo 기대값

두 팀의 Elo가 $R_A$, $R_B$일 때:

$$
E_A = \frac{1}{1 + 10^{(R_B-R_A)/400}}
$$

`E_A`는 문자 그대로의 승률이 아니라 Elo 기대값입니다. 무승부 확률은 득점 모델에서 별도로 생성됩니다.

프로젝트는 완료 경기가 반영된 최신 Elo를 내려받습니다. 내려받은 Elo에 로컬 K-factor를 다시 중복 적용하지 않습니다.

### 예상 득점

운영 계수는 `src/model_config.py`에서 관리합니다.

$$
\lambda_A = 1.35 \left(\frac{E_A}{1-E_A}\right)^{0.25}
$$

$$
\lambda_B = 1.35 \left(\frac{1-E_A}{E_A}\right)^{0.25}
$$

2018·2022 조별리그의 경기 전 Elo를 복원한 회고 평가를 바탕으로 지수를 기존 `0.376`에서 `0.25`로 낮췄습니다. 다중 클래스 log loss는 `1.0709`에서 `1.0163`으로, Brier score는 `0.6124`에서 `0.5984`로 감소했습니다. 현재 2026 조별리그 스냅샷에서도 같은 비교가 log loss `0.9575`에서 `0.9244`, Brier score `0.5531`에서 `0.5471`로 개선됩니다. 이 값은 모델 진단이며 미래 성능을 보장하지 않습니다.

선택형 grid search는 득점 기준값을 고정하고 의도적으로 표본 내 진단으로 표시합니다. 과거 대회가 두 개뿐이라 연도별 최적값이 불안정하므로 운영 계수를 자동으로 덮어쓰지 않습니다.

재현 가능한 보고서:

```bash
python evaluate_model.py
python evaluate_model.py --grid-search
```

### 완전한 승·무·패 확률

`src/poisson.py`는 필요한 득점 범위를 자동으로 확장하고 최종 승·무·패 벡터를 정규화합니다. Elo 차이가 매우 커도 합계가 정확히 100%이며, 10골 초과 확률을 버리지 않습니다.

가장 가능성이 높은 스코어는 두 포아송 분포의 최빈값으로 분석적으로 계산합니다. 따라서 CLI와 대시보드가 이 값을 위해 10만~100만 회 무작위 표본을 만들지 않습니다.

### 경기 환경 보정

- 공동 개최국: USA·Mexico·Canada가 비개최국을 상대하면 `+40` Elo
- 휴식: 하루당 `+5` Elo, 최대 `+30`
- 조별리그 3차전 로테이션: 2승 팀의 공격력 20% 감소
- 이동: 인접 권역 1.5%, 장거리 권역 3% 공격 감소
- 결장: 스쿼드 포지션 가치 비중과 HHI 집중도를 이용한 공격·수비 배율

조별리그, 토너먼트, CLI, 대시보드는 모두 `get_adjusted_ratings`와 `get_expected_goals`를 사용하므로 보정 방식이 서로 다르지 않습니다.

## 일정과 브래킷

`src/schedule.py`는 `data/schedule.json`에서 경기 번호, 날짜, 단계, 도시, 권역을 직접 읽습니다. `src/bracket.py`는 `Winner Match 74` 같은 참가 팀 출처를 해석하므로 별도의 하드코딩 브래킷을 유지하지 않습니다.

대표 연결:

- M89 = M74 승자 vs M77 승자
- M90 = M73 승자 vs M75 승자
- M97 = M89 승자 vs M90 승자

각 팀의 실제 직전 경기 날짜와 도시를 다음 라운드로 넘겨 휴식일 및 이동 피로를 계산합니다.

## 설치

패키지 메타데이터는 Python 3.11~3.14를 허용하며, CI는 현재 3.11·3.12·3.13을 검증합니다.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt`에는 직접 사용하는 런타임 의존성만 있으며 `pyproject.toml`과 일치합니다.

## 실행 방법

```bash
# 저장된 모든 데이터 검증
python validate_data.py

# 최신 Elo, 결과, 징계 갱신
python fetch_data.py

# 네트워크 없이 기존 결과의 경기 번호 검증 또는 마이그레이션
python fetch_data.py --backfill-match-numbers --check-only
python fetch_data.py --backfill-match-numbers

# 전체 대회 전망
python main.py --iterations 10000

# 1대1 예측: 선택 입력은 휴식일 차이, A 피로도, B 피로도
python predict_match.py "South Korea" "Mexico" 1 0.015 0

# 대시보드
streamlit run app.py
```

## 검증

회귀 테스트는 확률 합계, 극단 입력, 분석적 최빈 스코어, 주입형 테스트 난수 생성기, FIFA 타이브레이커, Annex C, 전체 브래킷 그래프, M89·M90, 일정 메타데이터, 현재 브래킷, 실제 결과 고정, 동점 준결승 판정, 원자적 저장, 부분 응답, 데이터 스키마, 몬테카를로 신뢰구간을 다룹니다.

```bash
python -m unittest discover -s tests -v
python validate_data.py
python -m compileall -q . -x '(^|/)(venv|\.git|scratch)/'
```

GitHub Actions는 Python 3.11, 3.12, 3.13에서 데이터 검증, 전체 테스트, 의존성 검사, 소스 컴파일, 애플리케이션 import를 실행합니다.

표시되는 Wilson 구간은 유한한 몬테카를로 표본 오차만 나타냅니다. Elo, 모델 계수, 부상 정보 및 기타 입력 가정의 불확실성은 포함하지 않습니다.
