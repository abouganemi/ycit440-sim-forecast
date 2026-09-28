# ycit440-sim-forecast

Data-science project built on [python-template](https://github.com/abouganemi/python-template) (uv, ruff, basedpyright, pytest, prek).

## Setup

```bash
uv sync --all-groups --locked
```

## Layout

| Path | Contents |
|---|---|
| `src/ycit440_sim_forecast/` | reusable code, tested (coverage gate 90%) |
| `notebooks/` | exploration: VS Code notebooks and `# %%` scripts; outside the coverage gate |
| `data/raw/` | source data, never edited; gitignored |
| `data/interim/`, `data/processed/` | derived data (e.g. parquet), rebuilt from raw; gitignored |
| `models/` | trained artifacts; gitignored |
| `data/manifest.json` | committed sha256 + size of the inputs a result was built from |

Notebook outputs are stripped by the `nbstripout` hook. Open notebooks in VS Code and pick the `.venv` kernel.

## Data

```bash
uv run python -m ycit440_sim_forecast.manifest write data/raw   # after adding or changing inputs
uv run python -m ycit440_sim_forecast.manifest verify           # fails if inputs changed
```

## Reproducibility

Call `ycit440_sim_forecast.seed.seed_everything()` at the start of a run and pass the returned seed as `random_state`/`seed` to scikit-learn, XGBoost, LightGBM and CatBoost.

## Checks

```bash
uv run prek run --all-files   # only sees files git tracks: `git add` new files first
uv run basedpyright
uv run pytest
```

## GPU (AMD ROCm)

PyTorch comes from the PyTorch ROCm index (see `[tool.uv.index]` in `pyproject.toml`) and lives in the `gpu` dependency group, which is a default group, so every `uv sync` installs it. CI runs `uv sync --all-groups --no-group gpu --locked` to skip the ~14 GB wheel; nothing in `src/` may require torch at import time for CI to pass.

```bash
uv run --env-file .env python scripts/check_gpu.py   # prints the GPU and runs a matmul
```

`.env` sets `HIP_VISIBLE_DEVICES` to the discrete GPU. The CPU's integrated GPU is disabled in the BIOS, so this is a safety net: ROCm would list it again if it were re-enabled. VS Code loads `.env` automatically; on the command line pass `--env-file .env`.

Keep the project on the same filesystem as `~/.cache/uv` so uv hardlinks torch instead of copying it.

## Training in a container / k3s

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
