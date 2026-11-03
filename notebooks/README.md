# Notebooks

## `01_eda.ipynb` keeps its outputs

Like the presentation notebooks below, `01_eda.ipynb` is committed **with its outputs** (figures, tables,
checks), so the team can read it on GitHub without the data or an environment. It opts in with
`"keep_output": true` in its notebook metadata, which the `nbstripout` hook honours.

**After any change to the notebook, refresh the outputs before committing:**

```bash
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

Run it from the repo root. It re-runs every cell top to bottom and also rewrites
`reports/figures/*.png`, `reports/eda_numbers.json` and `data/processed/division_day.parquet`, so
commit the report files together with the notebook. On unchanged data the figures and numbers come
out byte-identical; the notebook itself still shows as modified until the commit hook strips the
execution timestamps nbconvert adds.

Skipping this leaves outputs in git that no longer match the code. Running cells by hand in VS Code
works too, but only if you use **Run All** on a fresh kernel.

Keeping outputs is only for public-data exploration like this. Notebooks that touch private or
production data must stay stripped.

## Presentation notebooks (02–05)

These four notebooks explain the forecast to readers without a programming background: plain
language, a glossary at the top, a short explanation above every code cell, and charts. They
call the project's own code (`ycit440_sim_forecast`) and never re-implement the forecasting.
Read them in order:

1. `02_data_and_features.ipynb`: the data, the forecast timeline (data up to D−3, forecast made
   on D−1 for day D) and what the models see.
2. `03_models_and_ensemble.ipynb`: the five models and how averaging them gives the final
   ensemble (`ens_mean`), checked on 2021–2024 validation.
3. `04_test_results.ipynb`: how the ensemble did on the untouched 2025-01-01 → 2026-09-24 test,
   before and after the late-2025 break, and how sure we are.
4. `05_forecast_tomorrow.ipynb`: trains the ensemble on the latest data and makes a real
   forecast per division (a few seconds).

**`RERUN` in 03.** By default (`RERUN = False`) notebook 03 loads the saved validation results
and runs in seconds. Set `RERUN = True` to re-fit the models in memory instead (a few
minutes); it checks the result against the saved report and writes nothing.

**04 is read-only.** It only reads `reports/holdout_test.json` and
`data/processed/holdout_predictions_v2.parquet`. It never re-scores the test set, which is
used once (`uv run python -m ycit440_sim_forecast.holdout run`).

**Outputs.** Like 01, these notebooks are committed with their outputs (`"keep_output": true`).
Unlike 01, they write no files. After changing one, refresh its outputs from the repo root:

```bash
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb
```

## Step-by-step modelling notebooks (`modelling/`)

Cristal's six notebooks, `1_…` to `6_…`, build a forecast from scratch in plain pandas, one step
per notebook: data preparation, the same-weekday baseline, feature engineering, the monthly
validation schedule, LightGBM and CatBoost training, and a comparison with linear regression.
They were written in Colab and are kept as written; only the file paths were changed and the
`%pip install` cells replaced (LightGBM and CatBoost are already project dependencies). They are
not part of the `ycit440_sim_forecast` pipeline, so ruff and basedpyright skip this folder.

Each notebook reads the previous one's output, so run them in order. Notebook 1 reads the raw
CSVs in `data/raw/`, and notebooks 1 to 5 write their CSV outputs to `data/interim/modelling/`
(committed, so any notebook can be opened on its own). Like 01, they keep their outputs. To
refresh them, run from the repo root (about 20 s in total):

```bash
for f in notebooks/modelling/[1-6]_*.ipynb; do
  uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace "$f"
done
```

On the committed data the outputs match the original Colab run; the model predictions differ
only by floating-point rounding (about 1e-14).
