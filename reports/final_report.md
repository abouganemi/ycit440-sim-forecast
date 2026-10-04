# Final evaluation report: one-day-ahead SIM forecast by division

Team 4, YCIT 440 capstone, Project B (fire-service operations). Branch `feat/evaluation-report`.
Every number here comes from the JSON reports in this folder:

- [`ensemble_validation.json`](ensemble_validation.json): validation of the ensembles, 2022–2024;
- [`holdout_test.json`](holdout_test.json): the single test run, 2025-01-01 → 2026-09-24;
- [`horizon_comparison.json`](horizon_comparison.json) and the per-setting
  `models_validation_<setting>.json` / `ensemble_validation_<setting>.json`: the horizon
  comparison, validation data only.

Earlier stages are documented in [`eda_report.md`](eda_report.md) and
[`baseline_report.md`](baseline_report.md).

**Purpose.** State how well the final model forecasts the number of published SIM interventions
per division and day, compared with the agreed baseline, on data it never saw while it was built.

---

## 1. Key findings

1. **The final model beats the agreed baseline on the test period.** `ens_mean` scores **MAE
   8.00** interventions per day against **8.79** for the same-weekday mean over 13 weeks: 0.79
   lower (95% CI 0.50 to 1.08), about **9%**. It is lower in every division. This meets the
   framing's success rule.
2. **Against the plain 28-day mean the gain is small and not significant overall**: 0.20 lower,
   95% CI from 0.43 lower to 0.03 higher (about 2.5%). It becomes significant after the late-2025
   break (0.27 lower, CI 0.01 to 0.52).
3. **The model held up through the level break.** After 2025-12-20, when first-responder calls
   fell 25–30% in Divisions 1–5 and rose in Division 6, `ens_mean`'s MAE went *down* (8.12 →
   7.84) while the same-weekday baseline's went up (8.57 → 9.08). The baseline kept forecasting
   the old level (bias +1.0); the ensemble's level-tracking members followed the drop faster.
4. **The test result is in line with validation.** On 2022–2024 `ens_mean` was 0.63 below the
   same-weekday baseline (7%); on the test it is 0.79 below (9%). The test MAE of 8.00 is still
   above the noise floor of about 7.3–7.5 estimated for the D−3 cutoff
   ([`baseline_report.md`](baseline_report.md)).
5. **The models help only for the next few days.** On validation data the ensemble's advantage
   over the same-weekday mean shrinks from 0.63 at lead 1 to 0.29 at lead 7, and disappears at
   leads 14 and 30. On weekly totals it ties the baseline. The v2 daily target is where the
   models add value.
6. **Volume is still mostly unpredictable at this cutoff.** Even the final model misses by about
   8 interventions a day on a median of about 53 (WAPE 15% in validation). Storm and shock days,
   which no input describes, carry the largest errors.

---

## 2. What is forecast

- **Target:** the number of SIM interventions published in the Montréal open data for each of
  Divisions 1–6 on calendar day *D* (all intervention types; the 600 `DIVISION = 0` rows are
  excluded).
- **Timing:** issued on *D−1*, using completed daily counts through *D−3*, since the portal
  publishes about two days late. Calendar features of *D* are allowed; nothing describing the
  incidents of *D* is.
- **Metric:** MAE in interventions per day, with bias (forecast minus actual), median error and
  the share of days forecast below the actual, per division and per month.
- **Success rule (framing):** lower MAE than the same-weekday baseline. The plain 28-day mean is an
  extra, harder check we added.
- **Scope:** recorded volume, not workload. Nothing here says anything about staffing, vehicles or
  response times.

---

## 3. Method

**Data split.** 2020 serves only as lag history (COVID). Models and the baseline window were
chosen on 2021–2024 by rolling origin. 2025-01-01 → 2026-09-24 was held back and scored once,
after the final model was fixed.

**Rolling origin.** For each issue date the history is cut at that date's cutoff (*D−3*) before
the model sees it, and models are refitted at the first issue date of every month. Tests check
that no forecast can change when every count after the cutoff is replaced by 10,000. No k-fold
cross-validation: it would train on days after the ones it predicts.

**Baselines.** `same_wd_13w`: mean of the 13 most recent same-weekday counts known at the cutoff
(window chosen on 2021–2024 among 4, 8, 13, 26 and 52 weeks). `plain_28d`: mean of the last 28
known days.

**Features.** Recent levels ending at *D−3* (lags 0–6, 7/14/28/91-day means, 28-day spread,
level ratios, same-weekday means), the first-responder share of past calls, and the calendar of
*D* (weekday, month, day of year, Québec holidays and the days next to them). No linear time
trend.

**Models.** 13 configurations were validated: LightGBM, XGBoost and CatBoost with Poisson,
Tweedie and L1 losses, with and without the 28-day level as an offset; a negative-binomial GLM;
ETS per division; and top-down versions forecasting the city total and splitting it by recent
shares. A seasonal AR model with calendar regressors was also tried and dropped (its constant
pulled every forecast towards a mean that included the 2020 dip).

**Ensemble.** Five candidates whose errors differ: `lgbm_poisson_raw` and `lgbm_poisson`
(LightGBM Poisson without and with the level offset), `cat_tweedie`, `ets_weekly` and
`topdown_ets`. Three combinations were validated on 2022–2024 with weights refitted monthly on
earlier data only: the mean, the median, and MAE-optimal weights. **`ens_mean`, the plain
mean, was chosen before the test**: it scored best and learns nothing.

**Uncertainty.** Every comparison is a paired block bootstrap: whole ISO weeks are resampled,
all divisions together, because errors are correlated in time and across divisions on shock
days. 2,000 resamples, 95% percentile intervals.

---

## 4. Validation results (2022–2024, 6,570 division-days)

| Model                        | MAE       | Bias  | WAPE  | `ens_mean` minus model, 95% CI |
| ---------------------------- | --------- | ----- | ----- | ------------------------------ |
| **ens_mean**                 | **8.257** | −1.29 | 0.153 |                                |
| ens_weighted                 | 8.282     | −1.19 | 0.154 |                                |
| ens_median                   | 8.330     | −1.64 | 0.155 |                                |
| lgbm_poisson_raw\*           | 8.534     | −2.07 | 0.158 | −0.28 [−0.41, −0.15]           |
| same_wd_13w                  | 8.889     | −0.56 | 0.165 | −0.63 [−0.83, −0.45]           |
| plain_28d                    | 9.002     | −0.21 | 0.167 | −0.75 [−1.08, −0.45]           |

\* Best single candidate, picked in hindsight on the scored period.

Without windows that contain an anomaly day, `ens_mean` scores 7.35 against 7.96 and 8.05 for
the two baselines.

---

## 5. Test results (2025-01-01 → 2026-09-24, 3,792 division-days)

Run once, configuration `20c87f4ffadd`. Each `ens_mean` value was checked against a
recomputation from the five candidates' stored forecasts.

### Overall

| Model                               | MAE       | Bias  | Median error | Share under |
| ----------------------------------- | --------- | ----- | ------------ | ----------- |
| **ens_mean**                        | **7.998** | −0.22 | +0.56        | 0.48        |
| cat_tweedie                         | 8.118     | −0.13 |              |             |
| ets_weekly                          | 8.170     | −0.64 |              |             |
| plain_28d                           | 8.202     | +0.12 | +1.04        | 0.46        |
| topdown_ets                         | 8.239     | −0.14 |              |             |
| lgbm_poisson                        | 8.360     | +0.05 |              |             |
| lgbm_poisson_raw\*                  | 8.477     | −0.22 |              |             |
| same_wd_13w                         | 8.792     | +0.24 | +1.00        | 0.46        |

\* Best single candidate on validation.

Without anomaly windows (32 division-days removed): `ens_mean` 7.67, `plain_28d` 7.87,
`same_wd_13w` 8.48.

### Paired bootstrap, `ens_mean` minus reference (95% CI)

| Reference        | Whole test           | Before 2025-12-20    | From 2025-12-20      |
| ---------------- | -------------------- | -------------------- | -------------------- |
| same_wd_13w      | −0.79 [−1.08, −0.50] | −0.45 [−0.84, −0.04] | −1.23 [−1.61, −0.87] |
| plain_28d        | −0.20 [−0.43, +0.03] | −0.15 [−0.49, +0.21] | −0.27 [−0.52, −0.01] |
| lgbm_poisson_raw | −0.48 [−0.75, −0.26] | −0.50 [−0.96, −0.15] | −0.45 [−0.69, −0.22] |

Every candidate alone also beats `same_wd_13w` on the test, but only `cat_tweedie` (−0.67),
`ets_weekly` (−0.62) and `topdown_ets` (−0.55) do so significantly; the two LightGBM models'
intervals reach zero. `lgbm_poisson_raw`, the best single model in validation, is the worst
candidate on the test. That was expected: without the level offset a tree model cannot forecast
outside the range it was trained on, and the test period has level shifts. Averaging protected
against picking it alone.

### By division (MAE, whole test)

| Division | ens_mean | same_wd_13w | plain_28d |
| -------- | -------- | ----------- | --------- |
| 1        | 5.92     | 6.00        | 5.91      |
| 2        | 8.12     | 8.91        | 8.34      |
| 3        | 8.77     | 9.38        | 9.10      |
| 4        | 8.08     | 8.87        | 7.99      |
| 5        | 8.68     | 9.84        | 8.95      |
| 6        | 8.41     | 9.75        | 8.92      |

`ens_mean` beats `same_wd_13w` in all six divisions. It is within 0.1 of `plain_28d` in
Divisions 1 and 4 and lower in the other four.

### Drift: before and after the late-2025 break

The split date, 2025-12-20, is the start of the first-responder drop marked in the EDA; it was
fixed before the test run.

| Period                     | Division-days | ens_mean MAE (bias) | same_wd_13w   | plain_28d     |
| -------------------------- | ------------- | ------------------- | ------------- | ------------- |
| 2025-01-01 → 2025-12-19    | 2,118         | 8.12 (−0.63)        | 8.57 (−0.37)  | 8.27 (−0.08)  |
| 2025-12-20 → 2026-09-24    | 1,674         | 7.84 (+0.30)        | 9.08 (+1.00)  | 8.11 (+0.37)  |

- After the break every model over-forecasts a little (positive bias), as expected after a
  drop. The 13-week same-weekday mean reaches 13 weeks back and adapts slowest: its bias in
  Divisions 2–5 is +1.2 to +2.1 after the break.
- **Division 6 moved the other way** (rising) and is the one division `ens_mean` under-forecasts
  after the break (bias −0.96). Its MAE there is 8.94, against 9.51 for `plain_28d` and 10.65 for
  `same_wd_13w`.
- By month, `ens_mean` has the lowest MAE of the three in 15 of 21 months. Its worst month is
  April 2025 (9.44 against 7.49 for `plain_28d`).

---

## 6. Horizon comparison (validation, 2022–2024)

The same five candidates and the mean, re-run for other horizons with the same D−3 cutoff.
Official target stays v2 daily; this is the planned comparison. **The horizons were not run on
the test period.**

| Setting (n)            | ens_mean MAE | WAPE  | vs same_wd_13w       | vs plain_28d         |
| ---------------------- | ------------ | ----- | -------------------- | -------------------- |
| Day ahead, v2 (6,570)  | 8.26         | 0.153 | −0.63 [−0.83, −0.45] | −0.75 [−1.08, −0.45] |
| 3 days ahead (6,570)   | 8.40         | 0.156 | −0.49 [−0.68, −0.31] | −0.74 [−1.08, −0.42] |
| 7 days ahead (6,570)   | 8.69         | 0.161 | −0.29 [−0.49, −0.07] | −0.62 [−0.96, −0.32] |
| 14 days ahead (6,570)  | 9.00         | 0.167 | −0.08 [−0.35, +0.20] | −0.49 [−0.85, −0.17] |
| 30 days ahead (6,570)  | 9.42         | 0.175 | +0.31 [−0.10, +0.80] | +0.02 [−0.36, +0.41] |
| Weekly total (936)     | 36.42        | 0.096 | −0.79 [−3.29, +1.78] | −5.01 [−8.67, −1.54] |

"Days ahead" counts from the issue date; the gap to the last known day is two days more.
Weekly totals are scored on non-overlapping weeks (156 weeks × 6 divisions), so their intervals
are wide.

- The models' value comes from recent dynamics, which fade within about a week. Beyond two weeks
  a 13-week same-weekday mean is as good.
- Bias grows more negative with lead (−1.3 at lead 1, −2.6 at lead 30, −11 per week on weekly
  totals), driven by the tree models.
- Weekly totals are easier in relative terms (WAPE 10%) because daily noise averages out, but the
  ensemble does not beat the same-weekday baseline on them. Making weekly the official target
  would not gain anything from the models.

---

## 7. Reading the bias

MAE is minimised by the median, and daily counts are right-skewed, so a good MAE forecast shows a
slightly negative mean bias. Bias alone makes the best models look worse than they are; we report
the median error and the share of days under-forecast next to it.

- On validation, `ens_mean`'s bias is −1.29; about 0.8 of that comes from anomaly days.
- On the test before the break, `ens_mean` is centred on the median: median error −0.12, 50% of
  days under. After the break it over-forecasts slightly (median error +1.34, 45% under) while
  the level settles.

---

## 8. Limits

- **Candidate selection overlaps validation.** The five candidates were picked from 2021–2024
  results, which include the 2022–2024 scoring period, so the validation advantage is somewhat
  optimistic. The test period is the clean check, and it agrees.
- **Choosing the mean among three combinations** on the same period is also a choice, but the
  mean is the a-priori reference and learns nothing.
- **The publication cutoff is simulated.** The portal keeps no history of what was published
  when, so D−3 is assumed throughout. A live forecast would need a fallback (a longer lag) on
  days the portal is late.
- **First-responder share is a predictor built from past intervention types**, all at or before
  D−3. It is not a leak, but it goes beyond the framing's initial predictor set of counts and
  calendar.
- **The open data are about 94% of the totals SIM reports.** The model forecasts the published
  count.
- **The cause of the late-2025 drop is not documented.** The model followed it, but a reversal or
  a new change in dispatch rules would hit it the same way.
- **No weather or event inputs.** Storm days carry most of the large errors; the v2 design
  excludes external data.
- **The anomaly calendar** from the EDA is used only to split the reporting, never as a model
  input.

---

## 9. Reproducing

```bash
uv run python -m ycit440_sim_forecast.models validate                 # 2021–2024, all models, v2
uv run python -m ycit440_sim_forecast.ensemble validate               # 2022–2024 ensembles, v2
for h in weekly lead3 lead7 lead14 lead30; do
  uv run python -m ycit440_sim_forecast.models validate --horizon "$h" --models candidates
  uv run python -m ycit440_sim_forecast.ensemble validate --horizon "$h"
done
uv run python -m ycit440_sim_forecast.horizons compare
uv run python -m ycit440_sim_forecast.holdout run                     # once; refuses to overwrite
```

The holdout input hash and configuration are stored in `holdout_test.json`.
