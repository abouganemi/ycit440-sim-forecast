# Publication lag and the D−3 cutoff (v2 verification, section 2.1)

> Copied verbatim from `ml-capstone-project/2_understanding_and_framing_the_capstone_project/team_contributions/team_accepted_proposed_business_framing/v2/verification/03_design_stress_test.md`, lines 55–99, on 2026-09-28. Evidence produced by `scripts/03_timing_publication_lag.py` from the portal's CKAN activity stream.

### 2.1 Timing

**C1. "Completed daily counts through D−3 when available … to account for the observed publication lag".** The wording appears in Report §2 and §4, DIG "Check timing", Worksheet Q8 and Summary §3. **Verdict: ✓.**

Evidence:
- [FACT] The portal metadata was checked on 2026‑09‑28 00:08 UTC via `https://donnees.montreal.ca/api/3/action/package_show?id=interventions-service-securite-incendie-montreal`. It gives `update_frequency: daily`. The current CSV's `last_modified` is 2026‑09‑26T03:02:50 UTC. The HTTP header on the GCS object reads `Last-Modified: Sat, 26 Sep 2026 03:02:51 GMT`, and its md5 is identical to `Originals/donneesouvertes-interventions-sim.csv`.
- [FINDING] The latest record in that file is dated 2026‑09‑24. The file was published at 23:02 on 25 September, Montréal time.
- [FINDING] Method (`03_timing_publication_lag.py`):
  - The CKAN activity stream (`package_activity_list`, 1,300 entries back to October 2025) gives the size of every upload.
  - The current file is not sorted by date, so I summed its bytes by record date. For every 2026 upload, the size equals the cumulative byte total through a single date X, within ±5.4% of one day's bytes.
  - That date X is always the local day before publication (237 of 237 uploads).
  - Upload times: 22:xx local on 58 uploads and 23:xx on 178. The single exception was a manual reload at 12:41 on 24 September.
- [FINDING] For a forecast prepared at 09:00 or 17:00 on D−1 (268 days, January to September 2026):

  | Latest available date | Share of days |
  |---|---|
  | D−3 | 88.1% |
  | D−4 | 3.4% |
  | D−5 or earlier | 8.5% |

  The worst cases are 3–19 March 2026, when the latest date was D−5 to D−21. Days with no upload include 1–18 March, 28–30 August, 19–23 September and 26 September. The missed days are not only weekends, so these are outages rather than a weekday-only publishing schedule.

Fix: none required. OPTIONAL: add "(on 88% of days in 2026)" to show where the D−3 claim comes from.

**C2. "The latest available complete day".** **Verdict: ⚠** (the rule is right but not operationally defined).
- [FINDING] The last date in each upload was essentially complete when it was published: file size is within ±5.4% of one day's bytes of the final total, and the median difference is +2%. Part of that difference comes from later edits, which fits the documented reclassification of incident types. So "complete" is approximately true of the last date in the file.
- [INFERENCE] The documents do not say how the baseline behaves when the lag exceeds 7 days, which happened on 14 days in 2026. In that case the most recent same-weekday observation is itself missing.

Fix (SHOULD): define it as "the latest calendar date present in the most recent file published before the forecast is prepared; every lagged feature, including the same-weekday baseline, uses only dates up to that date."

**C3. The caveat about historical publication dates.** It appears in Report §4, DIG and Summary §3. **Verdict: ✓ (adequate).**
- [FINDING] I replaced the fixed D−3 cutoff with the real 2026 availability (`03_realistic_availability.py`, 16 January to 24 September 2026, n = 1,512 division-days). The MAE changes are:
  - baseline k = 6: 8.53 → 8.50;
  - baseline k = 12: 8.84 → 8.87;
  - baseline k = 26: 9.66 → 9.69;
  - model: 8.06 → 8.05.
- On the 180 division-days with lag > 3, the model's MAE is 8.40 under real availability.
- The caveat is correct: records are not versioned. It slightly understates what is available, because the portal's activity log records upload times and sizes back to at least October 2025.

OPTIONAL fix: "Portal upload logs show a one-day publication delay on 88% of 2026 days; a simulation using the actual 2026 upload record changed MAE by less than 0.05."

**C4. Does the D−3 cutoff limit the baseline?** **Verdict: ✓ (worth one clarifying sentence).**
- [FINDING] A same-weekday baseline uses D−7, D−14 and earlier, so the D−3 cutoff never binds for it unless the lag exceeds 7 days. The cutoff only constrains the model's recent-lag features.
- [FINDING] Cost of the cutoff for the model: test MAE 7.98 with data through D−1, 8.07 through D−2, 8.13 through D−3, 8.16 through D−4, and 8.25 through D−7.
