# EDA report: SIM interventions, 2020 to 2026-09-24

Team 4, YCIT 440 capstone, Project B (fire-service operations). Branch `feat/initial-eda`.
Source notebook: [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb). Every number here comes from
[`eda_numbers.json`](eda_numbers.json), which the notebook writes.

**Framing (team-accepted v2).** Forecast, one calendar day ahead, the number of SIM interventions
**published in the open data** for each operational Division 1–6. The forecast for day *D* is
prepared on *D−1* using completed daily counts through *D−3*. The baseline is the mean of recent
same-weekday values, with the window chosen on validation data. The primary metric is MAE in
interventions per day, reported with bias. The forecast measures volume, not workload.

**Method.** The notebook follows the ten steps of the course *Dataset Investigation Guide*, then
standard EDA. It recomputes every number stated in the v2 documents (`DEFINITIVE` report, v2
Dataset Investigation Guide, `verification/09_FINAL_VERIFIED.md`) from the local files and stops if
one does not reproduce. **All of them reproduce.** Structure that could influence modelling
(seasonality, autocorrelation, dispersion, baseline windows) is estimated on **2021–2024 only**.
The proposed test period, 2025 onward, is only inspected for level changes.

---

## 1. Key findings

1. **The target can be built, and it is dense.** There are 14,748 division-days. The median is 53
   interventions per day, and no division has a zero day. **2024-12-31 is absent from both files**,
   so it is treated as missing, not as zero. The 600 `DIVISION = 0` rows are undocumented and
   excluded.
2. **The open data are a subset of SIM activity.** Published counts are 93.8%, 94.3% and 93.5% of
   the totals SIM reports for 2022, 2023 and 2024. About 6–7% of each year's incident numbers
   never appear. The target must be called "published interventions".
3. **The level is not stable.** Mean volume went from 227/day (2020, COVID) to 358/day (2025),
   and was flat in 2023–24.
   - **Spring 2020.** First-responder records almost vanish: 56 days between 2020-04-07 and
     2020-07-07 have fewer than 5.
   - **Late 2025.** A drop starts and is concentrated in first-responder calls. For Jan 1 – Sep 24,
     2026 against the same span of 2025:
     - first-responder calls fell 25–30% in Divisions 1–5 and rose 9.5% in Division 6;
     - all other interventions were flat or up (+1% to +16%, except Division 4 at −1.2%).
   - The cause is not documented.
4. **Calendar structure is weak.** The weekday index stays within ±9% of each division's mean.
   STL weekly-seasonal strength is 0.06, against 0.33 for trend. Division 6 (Ville-Marie, Plateau)
   is the exception, with a weekend peak.
5. **The D−3 cutoff removes most short-term persistence.**
   - Lag-1 autocorrelation is 0.36–0.59, but lag 3, the first usable one, is only 0.16–0.38.
   - All divisions are over-dispersed relative to Poisson (variance/mean 2.8–7.0 on 2021–24).
   - ADF rejects a unit root and KPSS rejects a constant level: the series revert to a slowly
     moving level.
6. **Surge days are weather days.** Examples are the ice storm (1,712 interventions on
   2023-04-06), the Debby rain (Aug 2024) and the Feb 2023 cold snap. Their top types are
   electrical problems and flooding. No calendar feature predicts them, and they dominate the MAE
   tail.
7. **Baseline preview (2024 validation year).**
   - Same-weekday windows of 13 and 26 weeks tie at MAE 8.16/day, and 4 weeks is worst (8.89).
   - **The plain 28-day mean ending at D−3 is better overall (8.03).** It is best or tied in
     Divisions 1, 2, 4 and 6; a same-weekday window is slightly better in Divisions 3 and 5.
   - This is feasibility evidence, not the final window selection.
8. **CASERNE is not a sub-unit of DIVISION.** 30 of 66 casernes report into more than one of
   Divisions 1–6. Caserne-days are sparse: median 4, and 8.96% are zero. DIVISION is used as
   published and is never rebuilt from casernes.

---

## 2. Dataset Reference and Limitations (draft for the executive report)

> This project uses *Interventions des pompiers de Montréal*, published by the Ville de Montréal
> on its open-data portal (CC BY 4.0, updated daily) and extracted from the SIM's
> computer-assisted dispatch system (RAO). We use the 2020–2024 file and the current two-year file
> (retrieved 2026-09-26). They share one 13-column layout and cover consecutive periods, so they
> are appended, not joined: 754,413 records from 2020-01-01 to 2026-09-24. No external dataset is
> joined.
>
> Each row is one intervention event as published, with its final classification. `INCIDENT_NBR`
> restarts every year; combined with the calendar year it is unique. `CREATION_DATE_TIME` gives
> the event date used to build daily counts. `DIVISION` is the SIM division responsible for the
> territory where the event occurred. It is not the unit that responded, and `CASERNE` does not
> nest inside it. Our target is the daily count of published interventions for each of
> Divisions 1–6. The 600 `DIVISION = 0` records are undocumented and excluded.
>
> Only information available before the forecast is used. A forecast for day *D* is prepared on
> *D−1* from counts through *D−3*, matching the observed two-day publication lag, plus calendar
> variables. The type, territory and deployed vehicles (`NOMBRE_UNITES`) of future events are
> never predictors. Because the portal keeps no historical snapshots, this cutoff is simulated, not
> replayed.
>
> Material quality issues:
>
> - 2024-12-31 is missing from both files and is treated as missing, not zero.
> - 8 records have a date without a time; they are still usable for daily counts.
> - `DESCRIPTION_GROUPE` is blank in 1,788 records and `NOMBRE_UNITES` in 156; 7 records carry the
>   undocumented group `NOUVEAU`.
> - Incident types can be revised after the initial report, and the type vocabulary changes by
>   10–33 types a year.
> - Locations are moved to an intersection of the street segment for privacy.
> - The files hold about 94% of the interventions SIM reports, and the reason is undocumented.
>
> The level shifts twice: first-responder records almost disappear in spring 2020, and fall
> 25–30% from late December 2025 in Divisions 1–5. We therefore use 2020 only as lag history,
> estimate structure on 2021–2024, and evaluate on 2025 onward, reporting error per division and
> by month.
>
> The data do not measure incident duration, response or arrival times, which units responded,
> staffing, crew size, unit availability, costs, or unpublished and unmet demand. The forecasts
> estimate **recorded intervention volume**. They are not estimates of workload, staffing, vehicle
> needs or response-time performance, and we make no claims about those.

---

## 3. Working Dataset Summary (course guide §2)

| Field | Interventions des pompiers de Montréal (2020–2024 + current files) |
|---|---|
| **Dataset and source** | Ville de Montréal, donnees.montreal.ca, `interventions-service-securite-incendie-montreal`. Files `donneesouvertes-interventions-sim2020.csv` (539,203 rows) and `donneesouvertes-interventions-sim.csv` (215,210 rows). Retrieved 2026-09-26. sha256 recorded in `data/manifest.json`. CC BY 4.0. |
| **Purpose and coverage** | Extract of SIM's RAO dispatch system, produced for provincial reporting. Covers 2020-01-01 to 2026-09-24 (2-day lag, daily refresh) across the Montréal agglomeration: Montréal plus 14 linked municipalities. |
| **Unit represented by a row** | One published intervention event with its final classification. There is one vehicle count per event. |
| **Main entities** | Event; type and group; territorial division (`DIVISION`) and station territory (`CASERNE`); municipality and borough; displaced location. Responding units, crews and times are absent. |
| **Identifiers and joins** | Key `(year, INCIDENT_NBR)`: 0 duplicates. The two files are appended: identical header, no date overlap, no key overlap. No external join. CASERNE→DIVISION does not nest: 94.6% of rows fall in the caserne's modal division, and 30 of 66 casernes span several divisions. |
| **Relevant variables** | `CREATION_DATE_TIME` (target date, calendar), `DIVISION` (target grouping). `DESCRIPTION_GROUPE`, `INCIDENT_TYPE_DESC` and `NOMBRE_UNITES` are used for diagnostics only. |
| **Privacy transformations** | Locations moved to an intersection of the street segment: 99% of rows share a coordinate with another row, across 54,308 distinct points. About 6% of dispatch numbers are unpublished. Types may be revised after the initial report. |
| **Known limitations** | 2024-12-31 missing. 8 date-only rows. 1,788 blank groups, 156 blank unit counts, 7 `NOUVEAU`. 600 `DIVISION = 0` rows. Type vocabulary churn. First-responder breaks in 2020 and late 2025. No durations, response times, responding units, staffing or costs. |
| **Implications for the project** | Supports daily division-level counts and calendar features. The training window is 2021–2024, with 2020 as lag history. Level drift requires recent-level features and evaluation per division and per month. Does not support workload, staffing, vehicle, coverage or response-time claims. |

---

## 4. Variable reference (course guide §3)

| Variable | Verified meaning | Type / unit | Availability and quality | Project role or limitation |
|---|---|---|---|---|
| `CREATION_DATE_TIME` | Event creation time, local civil time: no records in the skipped spring-forward hour, no midnight heaping | datetime | Known at call creation; 8 date-only rows (7 in the 2020–24 file, 1 in the current file) | **Builds the target** (date) and calendar features |
| `DIVISION` | Division responsible for the event's territory | int, 0–6 | Complete; `0` = 600 rows, 59–100 per year, undocumented | **Target grouping**, Divisions 1–6 |
| `INCIDENT_NBR` | Event number; restarts every year | int | Unique with the year; 93–94% of issued numbers published; a second range at 500,000 and above since 2022 (9–109 rows per year) | Key and de-duplication only |
| `DESCRIPTION_GROUPE` | Broad group: 6 official groups plus `NOUVEAU` | category | 1,788 blank, mostly fire-type (mean 9.7 vehicles); 7 `NOUVEAU` | Drift diagnostics; never a predictor for day *D* |
| `INCIDENT_TYPE_DESC` | Detailed type; may be revised after the initial report | category | 117–143 types a year, of which 10–33 are new and 11–22 dropped | Diagnostics only |
| `CASERNE` | Station responsible for the territory, **not** the responder | int | 1 row = 0; does not nest in DIVISION | Not used: caserne-days are sparse (median 4, 8.96% zero) |
| `NOMBRE_UNITES` | Vehicles deployed; 1 vehicle ≈ 3–5 firefighters | int, vehicles | 156 blank; median 1, max 275 | Not a predictor (known only after the event) and not the target (volume ≠ workload) |
| `NOM_VILLE`, `NOM_ARROND` | Municipality and borough; `Indéterminé` = linked municipality | text | Complete | Context only |
| `LATITUDE`/`LONGITUDE`, `MTM8_X`/`MTM8_Y` | Location displaced to an intersection | float | Complete; inside the agglomeration's bounding box | Not used |

---

## 5. Findings by investigation step

### Steps 1–2 · Source, version and row unit

| Year | Published | Highest issued number | Share of issued | SIM-reported total | Share of SIM total |
|---|---:|---:|---:|---:|---:|
| 2022 | 111,552 | 118,786 | 93.9% | 118,916 | 93.8% |
| 2023 | 121,514 | 128,993 | 94.2% | 128,902 | 94.3% |
| 2024 | 121,432 | 129,523 | 93.8% | 129,822 | 93.5% |
| 2025 | 130,791 | 140,402 | 93.2% | n/a | n/a |

The highest issued incident number almost equals SIM's reported total, and about 6% of numbers
never reach the open data. That is why the target says "published".

### Step 3 · Entities: counts and vehicles tell different stories

![Share of incidents vs share of vehicles by group; vehicles per incident](figures/04_groups_units.png)

First-responder calls (1-REPOND) are 61% of incidents but only 36% of vehicle deployments. Building
fires are 1% of incidents and 7% of vehicles. Counting incidents is a deliberate choice: it
measures volume, not effort.

### Step 5 · Timing

The last record is 2026-09-24 23:59, retrieved 2026-09-26, a publication lag of **2 days**. That
supports the D−3 cutoff. Timestamps are local civil time. Demand bottoms at 04:00 and stays on a
plateau from 10:00 to 19:00.

![Hour of day](figures/05_hour_of_day.png)

| Variable | Known when | Allowed use |
|---|---|---|
| Calendar of *D* | always | predictor |
| Daily counts up to *D−3* | about 2 days after the day ends | lagged predictor |
| Type, group, caserne, units of events on *D* | during or after *D* | never a predictor |
| Revised type labels | unknown delay | hindsight labels; not used |

### Step 7 · Coverage, level and composition

![Citywide daily interventions with 28-day mean](figures/07_city_daily.png)

| Year | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (to Sep 24) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean interventions per day | 227.2 | 278.2 | 305.6 | 332.9 | 332.7 | 358.3 | 316.2 |

![Weekly mean of daily counts, one panel per division](figures/07_division_weekly.png)

![Composition by group](figures/07_group_composition.png)

**2026 vs 2025, Jan 1 – Sep 24, % change:**

| Division | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| First-responder (1-REPOND) | −27.8 | −28.5 | −24.9 | −29.8 | −27.0 | **+9.5** |
| All other groups | +16.0 | +12.4 | +1.0 | −1.2 | +2.8 | +7.6 |

**Density by grain (why DIVISION × day):**

| Grain | Cells | Median | Zero cells | Cells ≤ 5 | Variance/mean |
|---|---:|---:|---:|---:|---:|
| Citywide × day | 2,458 | 307 | 0% | 0% | 18.7 |
| **Division × day** | **14,748** | **53** | **0%** | **0.14%** | 7.2 |
| Caserne × day | 162,228 | 4 | 8.96% | 65.04% | 2.8 |

![Division-day distribution, letter-value plot](figures/07_division_box.png)

### Step 7 · Seasonality and dynamics (2021–2024)

![Weekday by month index heatmap per division](figures/07_seasonality.png)

![STL decomposition, citywide](figures/07_stl_citywide.png)

| Division | Mean/day | Variance/mean | ACF lag 1 | ACF lag 3 | ACF lag 7 | ACF lag 28 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 29.5 | 7.00 | 0.59 | 0.24 | 0.07 | 0.06 |
| 2 | 55.2 | 3.43 | 0.41 | 0.19 | 0.17 | 0.06 |
| 3 | 62.9 | 4.96 | 0.51 | 0.20 | 0.15 | 0.06 |
| 4 | 61.6 | 2.77 | 0.36 | 0.16 | 0.12 | 0.01 |
| 5 | 58.9 | 4.05 | 0.47 | 0.22 | 0.15 | 0.07 |
| 6 | 43.9 | 3.54 | 0.56 | 0.38 | 0.43 | 0.33 |

For every division, ADF p ≤ 0.0013 and KPSS p ≤ 0.01.

![ACF and PACF, Division 3](figures/07_acf_pacf.png)

### Steps 8–9 · Privacy and what the data do not measure

Locations are snapped to intersections: 99% of rows share a coordinate. About 6% of dispatch
numbers are unpublished. Labels may be revised. There is no personal data.

The data do not include response or arrival times, durations, responding units, crew size,
staffing, unit availability, costs, call priority, outcomes, or unmet demand.

### Step 10 · Fitness: baseline preview on the 2024 validation year

MAE in interventions per day. Forecasts use only data at or before *D−3*.

| Division | Same weekday, 4 wk | 8 wk | 13 wk | 26 wk | 52 wk | Plain 28-day mean |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 6.58 | 6.01 | 5.94 | 6.01 | 6.23 | **5.92** |
| 2 | 8.85 | 8.42 | 8.27 | 8.21 | 8.43 | **8.04** |
| 3 | 11.05 | 10.05 | 9.95 | **9.61** | 9.81 | 9.73 |
| 4 | 9.08 | 8.31 | 8.21 | 8.17 | 8.25 | **7.88** |
| 5 | 9.54 | 8.89 | **8.66** | 8.81 | 8.95 | 8.67 |
| 6 | 8.23 | **7.91** | 7.93 | 8.12 | 8.13 | **7.91** |
| **All** | 8.89 | 8.26 | 8.16 | 8.16 | 8.30 | **8.03** |
| Bias, all | −0.17 | −0.22 | −0.39 | −0.69 | −0.63 | −0.18 |

---

## 6. Implications for modelling

- **Split.**
  - 2020 serves as lag history only (COVID).
  - Fit and select on 2021–2024 with rolling-origin validation.
  - Test on 2025-01-01 → 2026-09-24. The late-2025 break falls inside the test period, which makes
    it a real drift test.
- **Baseline.** Keep the v2 same-weekday baseline, with the window chosen on validation data
  (13 and 26 weeks tie on 2024). Also report the **plain 28-day mean** as the secondary reference
  recommended in `09_FINAL_VERIFIED.md` §7: it was the stronger comparator in 2024.
- **Features.**
  - Recent levels (rolling means ending at D−3) and calendar features: weekday, month, holidays.
  - Drop the linear time trend; the red-team review showed it over-predicts after Dec 2025.
  - Consider a first-responder share feature to track composition drift.
- **Loss and intervals.** Counts are over-dispersed, so use negative-binomial or quasi-Poisson
  models, or quantile intervals, rather than Poisson.
- **Evaluation.** Report MAE and bias per division and by month, with and without the top 1% of
  (weather) days.
- **Open questions for the team.**
  - What changed in first-responder dispatch in late 2025?
  - What is `DIVISION = 0`?
  - Should public weather forecasts be added? Surges are weather-driven, but the v2 design
    excludes external data.

## 7. Differences from the v2 documents

No v2 number is contradicted. These results are new or sharper:

- **Plain 28-day mean beats the same-weekday family on 2024** (8.03 vs 8.16). The red team found
  that a simple model beats the plain mean by only about 1% on the test period, with a confidence
  interval including zero. Here the plain mean already beats the chosen baseline family on
  validation data, before any modelling.
- **Division 6 behaves differently.** First-responder calls rose while Divisions 1–5 fell, it has
  the only strong weekend peak, and it keeps persistent autocorrelation at lag 28.
- **COVID gap as defined here:** 56 days with fewer than 5 first-responder records between
  2020-04-07 and 2020-07-07. The red team gives 31 Mar – 8 Jul under a different definition.
- **Share of SIM-reported totals** (93.5–94.3%) is now computed from the files rather than quoted.

## Reproduce

```bash
uv sync --locked
uv run python -m ycit440_sim_forecast.manifest verify   # same input bytes
uv run --with nbconvert jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output-dir /tmp
```
