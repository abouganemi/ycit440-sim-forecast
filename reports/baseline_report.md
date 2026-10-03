# Baseline report: rolling-origin validation, 2021–2024

Team 4, YCIT 440 capstone, Project B (fire-service operations). Branch `feat/baseline-validation`.
Every number here comes from [`baseline_validation.json`](baseline_validation.json), written by
`uv run python -m ycit440_sim_forecast.evaluate baselines` from
`data/processed/division_day.parquet` and the anomaly calendar of the EDA.

**Purpose.** Set the scores every model must beat, and fix the validation procedure that will
score them. Models, features and the ensemble come later and are scored with the same code.

---

## 1. Key findings

1. **The two daily baselines tie.** For the v2 target (day *D*, data through *D−3*) the
   same-weekday mean over 13 weeks scores **MAE 8.48** and the plain 28-day mean **8.49**
   interventions per day. The EDA preview on 2024 alone had the plain mean ahead (8.03 against
   8.16); over four years that advantage disappears.
2. **13 weeks is the same-weekday window.** Of 4, 8, 13, 26 and 52 weeks, 13 has the lowest
   validation MAE for every setting. 52 weeks lags behind level changes (bias −2.29).
3. **On weekly totals same-weekday wins clearly**: MAE **35.1** against 37.4 interventions per
   week, about 6% lower, and lower in five of six divisions.
4. **Storm days carry the error tail.** Excluding windows that contain an anomaly day (96 of 8,760
   division-days) brings daily MAE to 7.72 for both baselines, and bias turns from slightly
   negative to positive: the baselines miss surges and slightly overshoot ordinary days.
5. **April is the hardest month** (MAE 12.1–13.7, the 2023 ice storm); March and November are
   the easiest (6.3–7.1).
6. **The plain mean degrades with lead time; same-weekday does not until lead 6.** From lead 1
   to lead 7 the plain mean goes from 8.49 to 8.75. Same-weekday stays at 8.48 through lead 5
   (its lag-7 value is still before the cutoff) and rises to 8.58 at leads 6–7.

What this means for modelling: the room left above the baselines is small. The EDA estimated a
noise floor near 7.3–7.5 for daily counts under the *D−3* cutoff, so a good model should expect
gains of a few percent on daily MAE, with more room on weekly totals.

---

## 2. Procedure

- **Forecast setting.** `ForecastSpec(publication_lag_days=2, lead_days=1, window_days=1)` is the
  v2 target: issued on *D−1*, cutoff *D−3*, target *D*. The same code runs weekly totals
  (`window_days=7`, issued the day before a Monday-to-Sunday week) and leads 2–7.
- **Rolling origin.** For each issue date the harness cuts the history at that date's cutoff and
  only then asks the forecaster for a forecast, so no forecast can use a later count. Forecasters
  are refitted at the start of each month (the baselines have nothing to fit).
- **Period.** Target days 2021-01-01 to 2024-12-31. 2020 serves only as lag history. The test
  period (2025-01-01 onward) is refused by the code unless explicitly allowed, and was not used.
- **Scoring.** MAE and bias (forecast minus actual) per division and per calendar month, with
  and without windows that touch an anomaly day. 2024-12-31, missing from the source files, is
  never scored. Daily results cover 8,760 division-days; weekly results 1,248 division-weeks
  (208 non-overlapping weeks).
- **Baselines.** Same-weekday: for each target day, the mean of the 13 most recent same-weekday
  counts on or before the cutoff, summed over the window. Plain: the mean of the 28 days ending at
  the cutoff, times the window length. Both need 70% of their values present.
- **Checks.** Automated tests prove no forecaster sees data past the cutoff in any setting, and
  that setting every later count to 10,000 changes no forecast. On 2024 the harness reproduces
  the EDA preview exactly.

---

## 3. Results

### Daily, v2 target (MAE in interventions per day)

| Division | Same weekday, 13 wk | Plain 28-day | Same weekday, no anomalies | Plain, no anomalies |
| -------- | ------------------: | -----------: | -------------------------: | ------------------: |
| 1        |                6.32 |         6.37 |                       5.31 |                5.39 |
| 2        |                8.77 |         8.74 |                       8.10 |                8.05 |
| 3        |                9.71 |         9.86 |                       8.56 |                8.67 |
| 4        |                8.84 |         8.73 |                       8.22 |                8.09 |
| 5        |                9.36 |         9.35 |                       8.54 |                8.51 |
| 6        |                7.88 |         7.89 |                       7.60 |                7.60 |
| **All**  |            **8.48** |     **8.49** |                   **7.72** |            **7.72** |
| Bias     |               −0.48 |        −0.16 |                      +0.36 |               +0.70 |

### Candidate windows (daily, all divisions)

| Same weekday, 4 wk | 8 wk | 13 wk | 26 wk | 52 wk | Plain 28-day |
| -----------------: | ---: | ----: | ----: | ----: | -----------: |
| 9.20 | 8.64 | **8.48** | 8.51 | 8.57 | 8.49 |

### Weekly totals (MAE in interventions per week)

| Division | Same weekday, 13 wk | Plain 28-day |
| -------- | ------------------: | -----------: |
| 1        |                26.4 |         28.3 |
| 2        |                36.6 |         38.3 |
| 3        |                39.2 |         43.8 |
| 4        |                34.0 |         37.3 |
| 5        |                40.4 |         43.2 |
| 6        |                34.1 |         33.3 |
| **All**  |            **35.1** |         37.4 |

Excluding anomaly weeks: 29.9 against 31.6. On weekly totals the 4-week same-weekday window
equals the plain 28-day mean exactly, because both average the same 28 days; the code was checked
against that identity.

### By lead time (daily, all divisions)

| Lead (days after issue) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| ----------------------- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Days from cutoff        | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| Same weekday, 13 wk     | 8.48 | 8.48 | 8.48 | 8.48 | 8.48 | 8.58 | 8.58 |
| Plain 28-day            | 8.49 | 8.55 | 8.60 | 8.65 | 8.69 | 8.72 | 8.75 |

### By calendar month (daily, all divisions)

| Month | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
| ----- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| Same weekday | 7.84 | 8.47 | 6.63 | 12.12 | 8.78 | 10.10 | 8.47 | 8.53 | 7.98 | 7.46 | 7.12 | 8.38 |
| Plain 28-day | 7.45 | 8.97 | 6.32 | 13.74 | 8.48 | 8.94 | 9.09 | 8.04 | 7.90 | 7.37 | 7.06 | 8.69 |

Bias by month swings between about −6 (April, same weekday) and +2.8 (March): both baselines
under-forecast April to June and over-forecast January, March and November. Monthly calendar
features should help here.

---

## 4. Limits

- **No significance test yet.** The daily difference (0.01) is far inside any plausible noise
  band. Paired bootstrap intervals will be computed when models are compared with the baselines.
- **Anomaly days were flagged from the full series.** They serve only as a reporting slice, never
  as a model input, so this does not leak into forecasts.
- **Validation only.** Test-period scores (2025 onward, including the late-2025 drop in
  first-responder calls) will be computed once, after model selection.

## 5. Decisions for the next chunks

- Report both baselines side by side for every setting; neither is the clear winner on daily.
- The models must beat **8.48** (daily) and **35.1** (weekly) on 2021–2024 rolling-origin
  validation, with intervals, before any claim of improvement.
- Features: recent levels ending at the cutoff, calendar month (the monthly bias pattern),
  weekday, holidays, and Division 6 recent-level features.
