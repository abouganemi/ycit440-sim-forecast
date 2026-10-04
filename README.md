# ycit440-sim-forecast

Team 4, YCIT 440 capstone. Built on [python-template](https://github.com/abouganemi/python-template) (uv, ruff, basedpyright, pytest, prek).

This README gets you from a fresh clone to running everything that exists so far, by hand.

## What the project is

A one-day-ahead forecast of the number of recorded SIM (Service de sécurité incendie de Montréal) interventions for each operational division, 1 to 6. The forecast for day D is made on D-1 and may only use data through D-3, because the open data is published with a lag.

Validation covers 2021-2024. Everything from 2025-01-01 on is a held-out test set. The code refuses to score it unless `allow_test=True` is passed explicitly. Do not change that, and do not pass it while choosing models.

## Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). It installs the right Python (3.13 or 3.14, see `requires-python` in `pyproject.toml`) and the dependencies for you. You do not need to install Python or create a virtual environment yourself.

From the repo root:

```bash
uv sync --no-group gpu
```

This creates `.venv/` with everything except the `gpu` group. That group is ROCm PyTorch (about 6 GB of wheels), is a default group, and nothing in the pipeline below uses it. Plain `uv sync` installs it.

`uv run` re-syncs to the default groups before every command, so it would download the `gpu` group again on your next run. To stop that, export one of these in your shell before you work:

```bash
export UV_NO_GROUP=gpu   # keeps syncing, but never installs the gpu group
```

CI instead sets `UV_NO_SYNC=1`, which stops `uv run` from syncing at all. Then you must run `uv sync --no-group gpu` yourself after pulling dependency changes.

Dependency groups: `dev` (pytest, ruff, basedpyright, prek), `eda` (matplotlib, seaborn), `gpu` (torch). The lock file only covers Linux on x86_64.

VS Code: open the repo folder itself (not a parent folder), or run "Python: Select Interpreter" and pick `.venv/bin/python`. Notebooks use the `.venv` kernel.

## Get the data

The raw data is not in git (`data/raw/` is gitignored). The two CSV files come from the Montréal open data portal, dataset "Interventions des pompiers de Montréal":
<https://donnees.montreal.ca/dataset/interventions-service-securite-incendie-montreal>

Put these two files in `data/raw/`, with exactly these names:

| File | Covers |
| --- | --- |
| `donneesouvertes-interventions-sim2020.csv` | "Interventions du SIM - 2020 à 2024" |
| `donneesouvertes-interventions-sim.csv` | "Interventions du SIM - courant (2 dernières années)" |

The portal answers `403 RBAC: access denied` to clients without a browser user agent. Download from the dataset page in a browser, or use a browser user agent with curl. The direct links are listed in `docs/metadata_review/provenance_and_sources.md`.

```bash
mkdir -p data/raw
curl -L -A "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0" \
  -o data/raw/donneesouvertes-interventions-sim.csv \
  "https://donnees.montreal.ca/dataset/2fc8a2b9-1556-410e-a118-c46e97e9f19e/resource/71e86320-e35c-4b4c-878a-e52124294355/download/donneesouvertes-interventions-sim.csv"
curl -L -A "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0" \
  -o data/raw/donneesouvertes-interventions-sim2020.csv \
  "https://donnees.montreal.ca/dataset/2fc8a2b9-1556-410e-a118-c46e97e9f19e/resource/4a46d93f-9fd9-4cce-8952-424918edeafe/download/donneesouvertes-interventions-sim2020.csv"
```

Check the files against the committed manifest (`data/manifest.json`, sha256 and size of each file):

```bash
uv run python -m ycit440_sim_forecast.manifest verify
```

It prints `data matches manifest` or lists what is missing or changed.

**A fresh download will probably fail this check.** The analysis and every number in `reports/` are pinned to the files saved on 2026-09-26. The portal replaces the "courant" file as new days are published, so today's copy is larger and its hash differs. The 2020-2024 file rarely changes. If the check fails, the results you get may differ slightly from `reports/`. Ask a teammate for the exact files if you need identical numbers. Do not run `manifest write` to make the check pass: that overwrites the record of what the reports were built from.

## Run the pipeline

Run these in order, from the repo root. Each step needs the output of the one before it.

### 1. EDA notebook (writes the processed data)

```bash
uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

Runs `notebooks/01_eda.ipynb` top to bottom. It writes:

- `data/processed/division_day.parquet` (daily counts per division) and `data/processed/anomaly_days.parquet`. Both CLIs below need them.
- `reports/eda_numbers.json` and `reports/figures/*.png`.

The notebook stops with an error if a number from the project's earlier framing documents does not reproduce from your data.

This notebook is the one exception to "notebook outputs are stripped": it is committed with its outputs. Re-running rewrites those outputs, plus the figures and numbers, in tracked files. If you only wanted the parquet files, discard the changes afterwards:

```bash
git restore notebooks/01_eda.ipynb reports/eda_numbers.json reports/figures
```

The parquet files are gitignored, so they stay. See `notebooks/README.md` for when to commit the refreshed outputs.

### 2. Baseline validation (about 80 s)

```bash
uv run python -m ycit440_sim_forecast.evaluate baselines
```

Scores the two baselines (same-weekday mean and plain 28-day mean) with a rolling origin over 2021-2024: for each issue date it cuts the history at D-3, forecasts, and compares with the actual count. Writes `reports/baseline_validation.json` and prints one MAE line per setting.

Options: `--history`, `--anomalies`, `--out` (paths), `--horizon` (daily leads 1..N, default 7), `--start` and `--end` (`YYYY-MM-DD`, default 2021-01-01 and 2024-12-31). An `--end` in 2025 or later is refused.

### 3. Model validation (about 5-6 min)

```bash
uv run python -m ycit440_sim_forecast.models validate
```

Scores both baselines and the models (CatBoost, LightGBM, XGBoost, a negative-binomial GLM, ETS, and top-down variants) over the same 2021-2024 period, with the same rules. Writes `reports/models_validation.json` and `data/processed/model_predictions_v2.parquet`, and prints MAE and bias per model.

Options: `--history`, `--anomalies`, `--out`, `--predictions` (paths), `--start`, `--end` (as above). It overwrites the tracked `reports/models_validation.json`; run `git restore reports/models_validation.json` if you do not want to keep the new copy.

### Optional: refresh the portal metadata snapshot

```bash
uv run python -m ycit440_sim_forecast.metadata snapshot
```

Downloads the portal's official metadata and data dictionary into `docs/source/` (`--out` changes the folder). It needs network access, rewrites tracked files, and is only needed when the raw files are re-downloaded. You do not need it to run the pipeline.

## Run the checks

CI runs the pre-commit hooks, basedpyright and pytest. Run the same locally:

```bash
uv run pytest
uv run ruff format
uv run ruff check
uv run basedpyright
uv run prek run --all-files
```

`uv run pytest` takes about 1 min. The suite fails if total coverage of `src/` is below 90% (`fail_under = 90`). `prek` runs the hooks in `.pre-commit-config.yaml` (ruff, nbstripout, uv-lock, whitespace and YAML checks, large-file and private-key checks, conventional-commit message check). It only sees files git tracks, so `git add` new files first.

Run one test file or a subset like this:

```bash
uv run pytest tests/test_manifest.py --no-cov
uv run pytest -k "manifest or seed" --no-cov
uv run pytest tests/test_manifest.py::test_write_then_verify --no-cov
```

Use `--no-cov` for partial runs: the coverage gate counts the whole of `src/`, so a partial run always fails it even when every test passes. Several files on one command line work too.

If pytest collects the wrong tests for the paths you gave (the whole suite for one file, or only the last of several files), the `.pytest_cache` folder is stale. Clear it once:

```bash
uv run pytest --cache-clear --co -q --no-cov tests/test_manifest.py
```

`--co` only lists what would run; check the count, then run normally.

Commit messages must follow Conventional Commits (`feat:`, `fix:`, `docs:`, ...).

## Read the results

| File | Contents |
| --- | --- |
| `reports/eda_report.md` | Written EDA findings; numbers come from `eda_numbers.json` |
| `reports/eda_numbers.json` | Key numbers computed by the notebook |
| `reports/figures/` | Figures made by the notebook |
| `reports/baseline_report.md` | Written report on the baselines; numbers come from `baseline_validation.json` |
| `reports/baseline_validation.json` | Baseline scores: overall, per division, per month, per setting |
| `reports/models_validation.json` | Model scores: `overall`, `overall_without_anomalies`, `by_division`, `by_month`, `by_year`, bootstrap comparisons |

Print overall MAE per model:

```bash
uv run python -c "
import json
for r in json.load(open('reports/models_validation.json'))['overall']:
    print(f\"{r['model']:<26} MAE {r['mae']:.3f}\")
"
```

Each row of `overall` also has `bias`, `median_error` and `under_share`. Errors are forecast minus actual, in interventions per division per day. MAE is the mean absolute error (lower is better). Bias is the mean error: negative means the model forecasts too low on average. `median_error` is the median error, and `under_share` is the share of days where the forecast was below the actual count. The counts are right-skewed, so a well-centred model has a slightly negative bias and an `under_share` near 0.5.

## Troubleshooting

**`data/processed/division_day.parquet not found; run notebooks/01_eda.ipynb first`.** The two CLIs read parquet files that only the notebook writes. Run step 1 first.

**`uv sync` or `uv run` downloads gigabytes.** That is the `gpu` group (ROCm torch). Use `uv sync --no-group gpu` and `export UV_NO_GROUP=gpu` (see Setup).

**Coverage fails on a partial run.** Add `--no-cov` (see Run the checks).

**pytest runs the wrong tests for the files I named.** The `.pytest_cache` folder is stale; add `--cache-clear` once (see Run the checks).

**VS Code does not see the packages or the kernel.** Open the repo folder itself, then select `.venv/bin/python` as the interpreter and kernel. Run `uv sync --no-group gpu` first if `.venv/` does not exist.

**Download returns `403 RBAC: access denied`.** The portal blocks non-browser user agents. Use a browser, or curl with `-A` as shown in "Get the data".

**`raw file not found: data/raw/...`.** The CSV file names must match the table in "Get the data".

## Repo layout

| Path | Contents |
| --- | --- |
| `src/ycit440_sim_forecast/` | reusable code, tested (coverage gate 90%) |
| `notebooks/` | exploration: VS Code notebooks and `# %%` scripts; outside the coverage gate |
| `data/raw/` | source data, never edited; gitignored |
| `data/interim/`, `data/processed/` | derived data (parquet), rebuilt from raw; gitignored |
| `models/` | trained artifacts; gitignored |
| `data/manifest.json` | committed sha256 + size of the inputs a result was built from |
| `reports/` | written reports and the JSON they quote |
| `docs/` | official portal documentation snapshot and the framing-stage metadata review |

Notebook outputs are stripped by the `nbstripout` hook, except `notebooks/01_eda.ipynb`, which opts in to keeping them with `"keep_output": true` in its metadata (see `notebooks/README.md`). Keep that for public-data exploration only; notebooks that touch private or production data must stay stripped.

Reproducibility: call `ycit440_sim_forecast.seed.seed_everything()` at the start of a run and pass the returned seed as `random_state`/`seed` to scikit-learn, XGBoost, LightGBM and CatBoost.

## GPU, Docker and k3s (not needed for the steps above)

The `gpu` group installs PyTorch for AMD ROCm. Nothing in the pipeline uses it. `scripts/check_gpu.py` only prints the GPU and runs a matmul (`uv run --env-file .env python scripts/check_gpu.py`; `.env` pins `HIP_VISIBLE_DEVICES`). The `Dockerfile` and the manifests in `k8s/` build and run that same GPU check; there is no training entrypoint yet. You can skip the rest of this section.

Needs Docker Engine (not a VM-based runtime such as Colima, which can't pass the GPU through). The image is ~15 GB, almost all torch.

```bash
docker build -t ycit440-sim-forecast:0.1.0 .
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --security-opt seccomp=unconfined \
  --ipc=host --env-file .env \
  -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" \
  ycit440-sim-forecast:0.1.0
```

The k3s cluster (`~/Projects/vulcan-k3s`) runs pods in the `ml` namespace under the `restricted` Pod Security level, which forbids mounting host directories, so data and models live on two PVCs:

```bash
docker save ycit440-sim-forecast:0.1.0 | sudo k3s ctr images import -

# once: create the PVCs and copy the data in
kubectl apply -f k8s/storage.yaml -f k8s/data-loader.yaml
kubectl wait -n ml --for=condition=Ready pod/ycit440-sim-forecast-loader
kubectl cp data/raw ml/ycit440-sim-forecast-loader:/data
kubectl delete -f k8s/data-loader.yaml

# train
kubectl apply -f k8s/train-job.yaml && kubectl logs -n ml -f job/ycit440-sim-forecast-train

# fetch the models (start the loader again, copy, delete it)
kubectl apply -f k8s/data-loader.yaml
kubectl wait -n ml --for=condition=Ready pod/ycit440-sim-forecast-loader
kubectl cp ml/ycit440-sim-forecast-loader:/models models
kubectl delete -f k8s/data-loader.yaml
```

**Deleting a PVC deletes its data** (the `local-path` storage class has reclaim policy `Delete`), so never `kubectl delete -f k8s/` or `kubectl delete -f k8s/storage.yaml` unless the data and models are safe elsewhere. Deleting the Job or the loader pod is always safe.

The image tag is the project version at scaffold time; when it changes, update the tag in the build commands and in `k8s/train-job.yaml` together. Replace the `check_gpu.py` command in the Job with the training entrypoint once it exists.
