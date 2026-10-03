# Notebooks

## `01_eda.ipynb` keeps its outputs

Unlike every other notebook here, `01_eda.ipynb` is committed **with its outputs** (figures, tables,
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
