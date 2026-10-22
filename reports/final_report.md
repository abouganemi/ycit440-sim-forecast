# Final evaluation report: one-day-ahead SIM forecast by division

Team 4, YCIT 440 capstone, Project B (fire-service operations).
Every number here comes from the JSON reports in this folder:

- [`ensemble_validation.json`](ensemble_validation.json): validation of the ensembles, 2022–2024;
- [`holdout_test.json`](holdout_test.json): the single test run, 2025-01-01 → 2026-09-24;
- [`horizon_comparison.json`](horizon_comparison.json) and the per-setting
  `models_validation_<setting>.json` / `ensemble_validation_<setting>.json`: the horizon
  comparison, validation data only;
- [`tuning_validation.json`](tuning_validation.json): the Optuna tuning of the boosters, tuned on
  2021 and compared on 2022–2024, validation data only.

Earlier stages are documented in [`eda_report.md`](eda_report.md) and
[`baseline_report.md`](baseline_report.md).

**Purpose.** State how well the final model forecasts the number of published SIM interventions
per division and day, compared with the agreed baseline, on data it never saw while it was built.

**How to read this report.** Section 0 is a plain-language summary. Sections 1 to 3 give the
findings, what is forecast and how. Sections 4 to 6 hold the result tables; each table has a
sentence on how to read it. Section 8 lists the limits. A glossary of terms is at the end
(section 10). Technical terms are explained where they first appear.

---

## 0. Summary in plain language

**What is forecast.** For each of the six fire-service divisions, the model predicts how many
SIM interventions (incidents the fire service responded to and published in the Montréal open
data) will be recorded on a given day. The forecast is made the day before, using counts that are
known up to three days earlier, because the open-data portal publishes about two days late.

**How well it works.** On a test period the model never saw (2025-01-01 → 2026-09-24), the
final model (an ensemble, `ens_mean`, explained in §3) is off by **8.00 interventions per
division-day on average**. The agreed baseline, a simple rule that predicts the average of the
same weekday over the last 13 weeks (`same_wd_13w`, "the same-weekday baseline"), is off by
**8.79**. That is an average error 0.79 smaller per division-day, about **9%** better. The interval around the 0.79
(95% CI 0.50 to 1.08) does not include 0, so we can say the model is better and the gap is not
just luck.

**What this means in practice.**

- On a typical division-day the forecast is still about 8 interventions away from what was
  recorded, on a median of about 53 per day. Day-to-day counts are partly random, and a perfect
  model would still miss by about 7.3–7.5. The model is about 0.5–0.7 above that floor, so there is
  limited room left to improve.
- The improvement is real but modest. Against a second, harder baseline (the plain average of the
  last 28 days, `plain_28d`) the gain is small and not significant overall.
- The model helps only for the next few days. Beyond about two weeks ahead, the simple baseline
  is as good.
- Storm days and other shocks are not predictable from the inputs and cause the largest misses.
- This forecasts **recorded volume**, not workload. Nothing here says anything about staffing,
  vehicles or response times.

---

## 1. Key findings

1. **The final model beats the agreed baseline on the test period.** `ens_mean` (the ensemble)
   scores **MAE 8.00** interventions per day against **8.79** for the same-weekday mean over 13
   weeks (`same_wd_13w`): 0.79 lower (95% CI 0.50 to 1.08), about **9%**. MAE is the average size
   of the miss, so "8.00" means the forecast is off by 8.00 interventions per division-day on
   average. It is lower in every division. This meets the framing's success rule.
2. **Against the plain 28-day mean the gain is small and not significant overall**: 0.20 lower,
   95% CI from 0.43 lower to 0.03 higher (about 2.5%). "Not significant" here means the interval
   includes 0, so we cannot say the ensemble is better than this baseline over the whole test. It
   becomes significant after the late-2025 break (0.27 lower, CI 0.01 to 0.52).
3. **The model held up through the level break.** After 2025-12-20, when first-responder calls
   fell 25–30% in Divisions 1–5 and rose in Division 6, `ens_mean`'s MAE went *down* (8.12 →
   7.84) while the same-weekday baseline's went up (8.57 → 9.08). The baseline kept forecasting
   the old level (bias +1.0, meaning it forecast about 1.0 too high on average); the ensemble's
   level-tracking members followed the drop faster.
4. **The test result is in line with validation.** On 2022–2024 `ens_mean` was 0.63 below the
   same-weekday baseline (7%); on the test it is 0.79 below (9%). The test MAE of 8.00 is still
   above the noise floor of about 7.3–7.5 estimated for the D−3 cutoff
   ([`baseline_report.md`](baseline_report.md)). The noise floor is the error even a perfect model
   would make, because daily counts are partly random.
5. **The models help only for the next few days.** On validation data the ensemble's advantage
   over the same-weekday mean shrinks from 0.63 at lead 1 (forecasting one day ahead) to 0.29 at
   lead 7, and disappears at leads 14 and 30. On weekly totals it ties the baseline. The v2 daily
   target is where the models add value.
6. **Volume is still mostly unpredictable at this cutoff.** Even the final model misses by about
   8 interventions a day on a median of about 53 (WAPE 15% in validation; WAPE is the total miss
   as a share of the total count). Storm and shock days, which no input describes, carry the
   largest errors.
7. **Hyperparameter tuning did not improve the ensemble.** Tuning means searching for better
   settings of the models. The final models use fixed, untuned settings. An Optuna search (an
   automatic search tool) on the three boosters, tuned on 2021, gave an ensemble with MAE
   **8.318** on 2022–2024 against **8.257** untuned: 0.06 worse, 95% CI −0.01 to +0.13. The
   interval includes 0, so the difference is not significant, but there is no sign of a gain. The
   untuned `ens_mean` stays the final model (§4).

---

## 2. What is forecast

- **Target:** the number of SIM interventions published in the Montréal open data for each of
  Divisions 1–6 on calendar day *D* (all intervention types; the 600 `DIVISION = 0` rows are
  excluded).
- **Timing:** issued on *D−1*, using completed daily counts through *D−3*, since the portal
  publishes about two days late. This "D−3 cutoff" means the model never sees the last two days
  before the forecast is issued. Calendar features of *D* are allowed; nothing describing the
  incidents of *D* is.
- **Metric:** MAE (mean absolute error, the average size of the miss, in interventions per day),
  with bias (average of forecast minus actual; negative means forecasting too low), median error
  (the typical signed miss, less affected by a few very bad days), and the share of days forecast
  below the actual ("share under"), per division and per month.
- **Success rule (framing):** lower MAE than the same-weekday baseline. The plain 28-day mean is an
  extra, harder check we added.
- **Scope:** recorded volume, not workload. Nothing here says anything about staffing, vehicles or
  response times.

---

## 3. Method

**Data split.** 2020 serves only as lag history (COVID). Models and the baseline window were
chosen on 2021–2024 by rolling origin. The validation period is the data used to choose and
compare models (2021–2024; scoring on 2022–2024). 2025-01-01 → 2026-09-24 is the test period: it
was held back and scored once, after the final model was fixed. Because the test was run only
once, it gives one honest check, and it cannot be repeated with changed settings.

**Rolling origin.** This mimics real use: for each issue date the history is cut at that date's
cutoff (*D−3*) before the model sees it, and models are refitted at the first issue date of
every month using only the data available at the time. Tests check that no forecast can change
when every count after the cutoff is replaced by 10,000. We do not use k-fold cross-validation
(which shuffles days into folds): it would train on days after the ones it predicts, which is
information a real forecast never has.

**Baselines.** A baseline is a simple rule the model must beat. `same_wd_13w` (the same-weekday
baseline): mean of the 13 most recent same-weekday counts known at the cutoff (window chosen on
2021–2024 among 4, 8, 13, 26 and 52 weeks). `plain_28d` (the 28-day baseline): mean of the last
28 known days.

**Features.** Features are the inputs given to the models. Recent levels ending at *D−3* (lags
0–6, 7/14/28/91-day means, 28-day spread, level ratios, same-weekday means), the first-responder
share of past calls, and the calendar of *D* (weekday, month, day of year, Québec holidays and
the days next to them). No linear time trend.

**Models.** 13 configurations were validated: LightGBM, XGBoost and CatBoost (three
decision-tree libraries) with Poisson, Tweedie and L1 losses (three ways of scoring errors during
training), with and without the 28-day level as an offset (a starting point, see the table
below); a negative-binomial GLM (a classical count-data regression); ETS per division
(exponential smoothing); and top-down versions forecasting the city total and splitting it by
recent shares. A seasonal AR model with calendar regressors was also tried and dropped (its
constant pulled every forecast towards a mean that included the 2020 dip).

**Ensemble.** An ensemble combines several models into one forecast. Five candidates whose errors
differ: `lgbm_poisson_raw` and `lgbm_poisson` (LightGBM Poisson without and with the level
offset), `cat_tweedie`, `ets_weekly` and `topdown_ets`. Three combinations were validated on
2022–2024 with weights refitted monthly on earlier data only: the mean, the median, and
MAE-optimal weights. **`ens_mean`, the plain mean, was chosen before the test**: it scored best
and learns nothing.

The five members:

| Name | What it is, in plain words | What it brings |
| ---- | -------------------------- | -------------- |
| `lgbm_poisson_raw` | LightGBM (a gradient-boosted decision-tree model) predicting the count from recent counts and the calendar, with no level offset. | It must learn the level itself, so it can't forecast outside the range it was trained on. |
| `lgbm_poisson` | The same model, but starting from the recent 28-day level (the "offset"), so the trees only learn the adjustment (for example, how much a given weekday, month or holiday usually differs from the recent level). | It follows level shifts automatically. |
| `cat_tweedie` | CatBoost, a different tree library with a different tree shape and loss, also starting from the 28-day level. | Different errors from the LightGBM models. |
| `ets_weekly` | Exponential smoothing per division, a classical time-series method that keeps a smoothed level plus a weekly pattern. No trees, no features. | A simple method with a different kind of error. |
| `topdown_ets` | The same smoothing on the city-wide total, then split among divisions by each one's share of the last 28 days. | The city total is less noisy than a single division. |

`ens_mean` (the ensemble) is the simple average of these five: each counts for one fifth, and no
weights are learned. All five are refitted monthly with only the data available at the time.

Why averaging helps: the members make different mistakes, and in an average these partly cancel.
The test results show this (§5): `lgbm_poisson_raw`, the best single model on
validation, was the worst on the test, and averaging diluted its weakness.

**Hyperparameters.** Hyperparameters are the settings of a model chosen before training (for
example, number of trees). The boosters use fixed settings (400 trees, learning rate 0.05,
shallow trees), not tuned values. A tuning experiment was run afterwards on validation data only
(§4); it did not change the final model.

**Uncertainty.** A confidence interval (CI) is a range of plausible values for a difference. Every
comparison is a paired block bootstrap: the same days are scored for both models, and whole ISO
weeks are resampled (2,000 resamples), all divisions together, because errors are correlated in
time and across divisions on shock days. The intervals are 95% percentile intervals. **If the
interval includes 0, we can't say one model is better than the other.**

---

## 4. Validation results (2022–2024, 6,570 division-days)

How to read this table: lower MAE is better. Bias is the average forecast minus actual (negative
means too low). WAPE is the total miss as a share of the total count. The last column is
`ens_mean` minus the model on that row, with its 95% CI; negative numbers mean the ensemble made
smaller errors, and an interval that includes 0 means no clear difference.

| Model                        | MAE       | Bias  | WAPE  | `ens_mean` minus model, 95% CI |
| ---------------------------- | --------- | ----- | ----- | ------------------------------ |
| **ens_mean**                 | **8.257** | −1.29 | 0.153 |                                |
| ens_weighted                 | 8.282     | −1.19 | 0.154 |                                |
| ens_median                   | 8.330     | −1.64 | 0.155 |                                |
| lgbm_poisson_raw\*           | 8.534     | −2.07 | 0.158 | −0.28 [−0.41, −0.15]           |
| same_wd_13w                  | 8.889     | −0.56 | 0.165 | −0.63 [−0.83, −0.45]           |
| plain_28d                    | 9.002     | −0.21 | 0.167 | −0.75 [−1.08, −0.45]           |

\* Best single candidate, picked in hindsight on the scored period.

Without windows that contain an anomaly day (an unusual day, such as a storm, flagged in the
EDA), `ens_mean` scores 7.35 against 7.96 and 8.05 for the two baselines.

### Tuned versus untuned ensemble

Tuning means searching for better model settings. The three boosters in the ensemble
(`lgbm_poisson_raw`, `lgbm_poisson`, `cat_tweedie`) were tuned with Optuna (a tool that tries
many settings automatically): 40 trials each, TPE sampler with a fixed seed, the default settings
as the first trial. The search covered the number of trees, learning rate, tree size, leaf
minimum, regularisation and row/column subsampling. Each trial was scored by rolling origin on
**2021 only**, so the tuning never saw the 2022–2024 period used to compare the ensembles.
`ets_weekly` and `topdown_ets` have nothing to tune and were kept as they are.
`ens_mean_tuned` is the plain mean of the three tuned boosters and the two unchanged ETS models.
The test period was not used. Configuration `c7977b696fd3`.

How to read this table: the middle columns show MAE before and after tuning (lower is better).
The 2021 column is the year the settings were tuned on, so it always improves and proves
nothing on its own; the 2022–2024 column is the fair test, on years the tuning never saw. The
last column is tuned minus untuned on 2022–2024; positive numbers mean tuning made things
worse, negative numbers mean it helped, and an interval that includes 0 means no clear
difference.

| Model                  | Tuning MAE, 2021 (default → best) | MAE 2022–2024 (untuned → tuned) | Tuned minus untuned, 95% CI |
| ---------------------- | --------------------------------- | ------------------------------- | --------------------------- |
| lgbm_poisson_raw       | 7.47 → 6.91                       | 8.534 → 8.397                   | −0.14 [−0.21, −0.07]        |
| lgbm_poisson           | 8.01 → 6.92                       | 8.618 → 8.681                   | +0.06 [−0.13, +0.27]        |
| cat_tweedie            | 7.55 → 7.03                       | 8.544 → 8.672                   | +0.13 [−0.06, +0.32]        |
| **ensemble (mean)**    |                                   | **8.257 → 8.318**               | **+0.06 [−0.01, +0.13]**    |

By year the tuned ensemble wins 2022 (7.40 against 7.43) and loses 2023 (9.89 against 9.75) and
2024 (7.66 against 7.59). Without anomaly windows: 7.39 against 7.35.

**Why tuning didn't help.**

- **The gains on the tuning year did not carry over.** Tuning cut each booster's 2021 MAE by 0.5
  to 1.1, yet only `lgbm_poisson_raw` improved on 2022–2024. The search had only one year of
  data to work with (about 2,190 division-days), and with that little data it ends up fitting
  that year's random ups and downs instead of patterns that repeat. This is overfitting: the
  settings look good on the year they were tuned on and not on other years. 2021 was also an
  easier year than 2022–2024, so settings that suited it were a poor guide to the harder years.
- **The two level-offset models were tuned towards doing very little.** Their best 2021 settings
  learn slowly and stop early (about 100 trees at learning rate 0.01), so their forecasts stay
  close to the recent 28-day level. That suited 2021 but over-forecasts later: `lgbm_poisson`'s
  median error went from −0.06 to +0.73 and `cat_tweedie`'s from −0.34 to +0.72.
- **The ensemble already absorbs most of what tuning can fix.** Averaging five models with
  different errors smooths the individual settings out, and the remaining error sits close to
  the noise floor of about 7.3–7.5 at this cutoff.

Keeping only the tuned `lgbm_poisson_raw` might gain a little. But that choice would be made from
these 2022–2024 results, and the test period has already been used, so it could not be checked
cleanly. **The untuned `ens_mean` stays the final model.**

---

## 5. Test results (2025-01-01 → 2026-09-24, 3,792 division-days)

Run once, configuration `20c87f4ffadd`. Each `ens_mean` value was checked against a
recomputation from the five candidates' stored forecasts.

### Overall

How to read this table: lower MAE is better, and the rows are sorted by MAE. Bias is the average
forecast minus actual (negative means too low). Median error is the typical signed miss. Share
under is the fraction of days the forecast was below the actual (0.50 would be balanced). Blank
cells were not computed for that model.

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

How to read this table: each cell is `ens_mean` MAE minus the reference's MAE, with the 95% CI in
brackets. Negative numbers mean the ensemble made smaller errors. If the bracket includes 0 (for
example [−0.43, +0.03]), the difference is not significant. The columns split the test at
2025-12-20 (see the drift section below).

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

How to read this table: each cell is the MAE for that division (lower is better). Compare across
a row to see which method did best in that division.

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

Drift is a shift in the level of the data over time. The split date, 2025-12-20, is the start of
the first-responder drop marked in the EDA; it was fixed before the test run.

How to read this table: each cell is MAE with the bias in brackets. A positive bias means the
method forecast too high on average, a negative bias too low.

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

The horizon (or lead) is how many days ahead the forecast looks. The same five candidates and the
mean, re-run for other horizons with the same D−3 cutoff. Official target stays v2 daily; this is
the planned comparison. **The horizons were not run on the test period.**

How to read this table: each row is a different horizon. The last two columns are `ens_mean` MAE
minus the baseline's MAE with the 95% CI; negative means the ensemble is better, and an interval
that includes 0 (as for 14 and 30 days ahead) means no clear difference.

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

MAE is minimised by the median, and daily counts are right-skewed (a few very high days pull the
average up), so a good MAE forecast shows a slightly negative mean bias. Bias alone makes the
best models look worse than they are; we report the median error and the share of days
under-forecast next to it.

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
  D−3. It is not a leak (the model does not see information from after the cutoff), but it goes
  beyond the framing's initial predictor set of counts and calendar.
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
uv run python -m ycit440_sim_forecast.tuning run --trials 40         # about 20 min; validation only
uv run python -m ycit440_sim_forecast.holdout run                     # once; refuses to overwrite
```

The holdout input hash and configuration are stored in `holdout_test.json`.

---

## 10. Glossary

- **Anomaly days:** unusual days, such as storms, flagged in the EDA. Some tables are also shown
  without windows that contain them.
- **Baseline:** a simple rule the model must beat. `same_wd_13w` is the mean of the same weekday
  over the last 13 weeks (the agreed baseline). `plain_28d` is the mean of the last 28 days.
- **Bias:** the average of forecast minus actual. Negative means the forecast is too low on
  average, positive too high.
- **D−3 cutoff:** the forecast for day *D* uses only counts through day *D−3*, because the portal
  publishes about two days late.
- **Ensemble:** one forecast built from several models. `ens_mean` is the plain average of five.
- **Horizon / lead:** how many days ahead the forecast looks.
- **Hyperparameters / tuning / Optuna:** hyperparameters are model settings chosen before
  training. Tuning is the search for better ones. Optuna is the tool that runs the search.
- **Level offset:** starting the model from the recent 28-day average, so it only has to learn
  the adjustment on top of it.
- **MAE (mean absolute error):** the average size of the miss, in interventions per division-day.
  Lower is better.
- **Median error:** the typical signed miss (the middle value), not distorted by a few very bad
  days.
- **95% CI / paired bootstrap:** a range of plausible values for a difference, found by
  re-scoring many resampled sets of weeks (the same days for both models). If the interval
  includes 0, we can't say one model is better than the other.
- **Noise floor (about 7.3–7.5):** the error you would get even from a perfect model, because
  day-to-day counts are partly random.
- **Rolling origin:** testing by replaying history one issue date at a time, giving the model only
  what was known then. Used instead of k-fold, which would let the model train on days after the
  ones it predicts.
- **Share under:** the fraction of days on which the forecast was below the actual.
- **Validation vs test period:** validation (2021–2024) is the data used to choose and compare
  models. The test period (2025-01-01 → 2026-09-24) was held back and scored once, after the
  final model was fixed.
- **WAPE:** the total miss divided by the total actual count, a percentage-like error that allows
  comparison across different sizes (0.153 means about 15%).
