"""Tune the boosted ensemble members with Optuna and re-score ``ens_mean``.

Each booster of ``TUNABLE`` is tuned on its own, minimising its rolling-origin
MAE on 2021 only, a validation year outside the 2022-2024 period the ensemble
is scored on, so the comparison never sees the data the search did. Trial 0
runs the untuned settings of ``default_models`` (see ``default_trial_params``).
The rolling origin is run one issue month at a time (``issue_month_folds``),
which gives the same forecasts as a single run, and the running MAE after each
month is reported so ``MedianPruner`` can stop poor trials early.

The tuned boosters are then refitted by rolling origin over 2021-2024, like
``models validate``, and averaged with the untuned ``ets_weekly`` and
``topdown_ets`` forecasts read from the out-of-fold predictions into
``ens_mean_tuned``, which is compared with ``ens_mean`` on 2022-2024. The test
period is never read. Usage::

    uv run python -m ycit440_sim_forecast.tuning run --trials 40
    uv run python -m ycit440_sim_forecast.tuning compare
"""

import datetime as dt
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

import optuna
import polars as pl
import typer
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from ycit440_sim_forecast.ensemble import DEFAULT_OUT as ENSEMBLE_REPORT
from ycit440_sim_forecast.ensemble import (
    FINAL_MODEL,
    KEYS,
    OUTPUT_SCHEMA,
    SCORE_START,
    EnsembleConfig,
    combine_oof,
)
from ycit440_sim_forecast.evaluate import (
    DEFAULT_ANOMALIES,
    DEFAULT_HISTORY,
    TEST_START,
    VALIDATION_END,
    VALIDATION_START,
    flag_anomalies,
    issue_dates,
    paired_bootstrap,
    rolling_origin,
    score,
)
from ycit440_sim_forecast.manifest import sha256_file
from ycit440_sim_forecast.models import (
    DEFAULT_PREDICTIONS,
    ENSEMBLE_CANDIDATES,
    REFERENCES,
    BoostedForecaster,
    LevelMode,
    cat_params,
    default_models,
    lgbm_params,
    rounded,
)
from ycit440_sim_forecast.seed import DEFAULT_SEED
from ycit440_sim_forecast.spec import V2_DAILY, ForecastSpec

TUNE_START = dt.date(2021, 1, 1)
TUNE_END = dt.date(2021, 12, 31)
"""Tuning period: the validation year before the ensemble's scored period."""

DEFAULT_TRIALS = 40
TUNED_SUFFIX = "_tuned"
TUNED_ENSEMBLE = f"{FINAL_MODEL}{TUNED_SUFFIX}"
"""Name of the mean of the tuned boosters and the untuned other candidates."""

PRUNER_STARTUP_TRIALS = 5
PRUNER_WARMUP_STEPS = 3
"""Issue months before pruning may start; the first holds a single issue date."""
LGBM_REG_LAMBDA_FLOOR = 1e-3
"""Lower end of the ``reg_lambda`` search; trial 0 uses it for LightGBM's 0."""

DEFAULT_OUT = Path("reports/tuning_validation.json")
DEFAULT_TUNED_PREDICTIONS = Path("data/processed/model_predictions_v2_tuned.parquet")
DEFAULT_TRIALS_OUT = Path("reports/tuning_trials.parquet")
_CAT_LOSS = "Tweedie:variance_power=1.5"

TRIALS_SCHEMA = pl.Schema(
    {
        "model": pl.String,
        "number": pl.Int64,
        "state": pl.String,
        "value": pl.Float64,
        "seconds": pl.Float64,
        "n_steps": pl.Int64,
        "params": pl.String,
    }
)
"""One row per Optuna trial; ``params`` is canonical JSON."""

type SearchSpace = Callable[[optuna.trial.BaseTrial], dict[str, Any]]

app = typer.Typer(help=__doc__, no_args_is_help=True)


def lgbm_space(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    """LightGBM settings drawn from ``trial``.

    Args:
        trial: Optuna trial (or ``FixedTrial``) to draw from.

    Returns:
        ``n_estimators``, ``learning_rate``, ``num_leaves``,
        ``min_child_samples``, ``reg_lambda``, ``subsample`` and
        ``colsample_bytree``.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", LGBM_REG_LAMBDA_FLOOR, 10.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }


def cat_space(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    """CatBoost settings drawn from ``trial``.

    Args:
        trial: Optuna trial (or ``FixedTrial``) to draw from.

    Returns:
        ``iterations``, ``learning_rate``, ``depth``, ``l2_leaf_reg``, ``rsm``
        and ``random_strength``.
    """
    return {
        "iterations": trial.suggest_int("iterations", 100, 1000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
    }


SEARCH_SPACES: Mapping[str, SearchSpace] = {
    "lgbm_poisson_raw": lgbm_space,
    "lgbm_poisson": lgbm_space,
    "cat_tweedie": cat_space,
}
"""Search space of each tunable candidate."""

TUNABLE: tuple[str, ...] = tuple(SEARCH_SPACES)
"""Boosted ensemble candidates that can be tuned, in ``ENSEMBLE_CANDIDATES`` order."""

_LEVELS: Mapping[str, LevelMode] = {
    "lgbm_poisson_raw": "none",
    "lgbm_poisson": "offset",
    "cat_tweedie": "offset",
}


def tuned_name(name: str) -> str:
    """Result-table name of a tuned candidate.

    Args:
        name: Candidate name.

    Returns:
        ``name`` with ``TUNED_SUFFIX``.
    """
    return f"{name}{TUNED_SUFFIX}"


def default_trial_params(name: str) -> dict[str, Any]:
    """Untuned ``default_models`` settings, as the search space names them.

    They are enqueued as trial 0. Every value equals the default model's
    except LightGBM's ``reg_lambda``: its default 0 is outside the log-scale
    search, so the floor ``1e-3`` stands in. ``subsample`` 1.0 with the fixed
    ``subsample_freq`` 1 turns bagging off, as the default does.

    Args:
        name: Tunable candidate.

    Returns:
        One value per search-space parameter.

    Raises:
        ValueError: If ``name`` is not in ``TUNABLE``.
    """
    if name not in SEARCH_SPACES:
        msg = f"{name!r} is not tunable; expected one of {TUNABLE}"
        raise ValueError(msg)
    if SEARCH_SPACES[name] is cat_space:
        cat = cat_params()
        return {
            "iterations": cat["iterations"],
            "learning_rate": cat["learning_rate"],
            "depth": cat["depth"],
            "l2_leaf_reg": 3.0,  # CatBoost's default
            "rsm": 1.0,
            "random_strength": 1.0,
        }
    lgbm = lgbm_params()
    return {
        "n_estimators": lgbm["n_estimators"],
        "learning_rate": lgbm["learning_rate"],
        "num_leaves": lgbm["num_leaves"],
        "min_child_samples": lgbm["min_child_samples"],
        "reg_lambda": LGBM_REG_LAMBDA_FLOOR,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    }


def build_forecaster(
    name: str,
    params: Mapping[str, Any],
    seed: int = DEFAULT_SEED,
    label: str | None = None,
) -> BoostedForecaster:
    """The candidate ``name`` with its search-space settings replaced by ``params``.

    The fixed settings (objective, level mode, seed, threads, deterministic
    options, logging) are those of ``default_models``; LightGBM also gets
    ``subsample_freq`` 1 so ``subsample`` takes effect.

    Args:
        name: Tunable candidate.
        params: Search-space values, e.g. from ``SEARCH_SPACES[name]``.
        seed: Random seed passed to the library.
        label: Result-table name; ``name`` when None.

    Returns:
        An unfitted forecaster.

    Raises:
        ValueError: If ``name`` is not tunable or ``params`` sets a key outside
            its search space.
    """
    if name not in SEARCH_SPACES:
        msg = f"{name!r} is not tunable; expected one of {TUNABLE}"
        raise ValueError(msg)
    extra = sorted(set(params) - set(default_trial_params(name)))
    if extra:
        msg = f"{name}: parameters outside the search space: {extra}"
        raise ValueError(msg)
    label = name if label is None else label
    if SEARCH_SPACES[name] is cat_space:
        estimator = CatBoostRegressor(
            loss_function=_CAT_LOSS, **{**cat_params(seed), **params}
        )
        return BoostedForecaster(label, estimator, level=_LEVELS[name])
    settings = {**lgbm_params(seed), "subsample_freq": 1, **params}
    estimator = LGBMRegressor(objective="poisson", **settings)
    return BoostedForecaster(label, estimator, level=_LEVELS[name])


@dataclass(frozen=True, kw_only=True)
class TuningConfig:
    """Settings of one tuning and comparison run.

    Attributes:
        spec: Forecast setting.
        candidates: Boosters to tune, each on its own.
        tune_start: First target day of the tuning objective.
        tune_end: Last target day of the tuning objective; must be before
            ``score_start``.
        refit_start: First target day of the tuned boosters' refit, which
            must cover the out-of-fold predictions they are combined with.
        score_start: First target day of the comparison.
        score_end: Last target day of the comparison and of the refit; must
            be before ``TEST_START``.
        n_trials: Optuna trials per booster, trial 0 included.
        n_boot: Bootstrap resamples per comparison.
        seed: Seed of the sampler, the libraries and the bootstrap.
    """

    spec: ForecastSpec = V2_DAILY
    candidates: tuple[str, ...] = TUNABLE
    tune_start: dt.date = TUNE_START
    tune_end: dt.date = TUNE_END
    refit_start: dt.date = VALIDATION_START
    score_start: dt.date = SCORE_START
    score_end: dt.date = VALIDATION_END
    n_trials: int = DEFAULT_TRIALS
    n_boot: int = 2000
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        """Reject settings that leak into the comparison or touch the test period.

        Raises:
            ValueError: If candidates are empty, repeated or not tunable, a
                period is reversed, the tuning period reaches
                ``score_start``, the refit starts after the tuning or scored
                period, ``score_end`` reaches ``TEST_START``, or ``n_trials``
                or ``n_boot`` is below 1.
        """
        if not self.candidates or len(set(self.candidates)) != len(self.candidates):
            msg = f"candidates must be non-empty and unique, got {self.candidates}"
            raise ValueError(msg)
        unknown = sorted(set(self.candidates) - set(TUNABLE))
        if unknown:
            msg = f"candidates {unknown} are not tunable; expected some of {TUNABLE}"
            raise ValueError(msg)
        if self.tune_start > self.tune_end:
            msg = f"tune_start {self.tune_start} is after tune_end {self.tune_end}"
            raise ValueError(msg)
        if self.score_start > self.score_end:
            msg = f"score_start {self.score_start} is after score_end {self.score_end}"
            raise ValueError(msg)
        if self.tune_end >= self.score_start:
            msg = (
                f"tune_end {self.tune_end} must be before score_start "
                f"{self.score_start}, or tuning leaks into the comparison"
            )
            raise ValueError(msg)
        if self.refit_start > self.tune_start:
            msg = (
                f"refit_start {self.refit_start} is after tune_start {self.tune_start}"
            )
            raise ValueError(msg)
        if self.score_end >= TEST_START:
            msg = f"score_end {self.score_end} reaches the test period ({TEST_START}+)"
            raise ValueError(msg)
        if self.n_trials < 1:
            msg = f"n_trials must be >= 1, got {self.n_trials}"
            raise ValueError(msg)
        if self.n_boot < 1:
            msg = f"n_boot must be >= 1, got {self.n_boot}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready settings.

        Returns:
            Every field, dates as ISO strings and the spec as its fields and key.
        """
        return {
            "spec": {
                "key": self.spec.key,
                "publication_lag_days": self.spec.publication_lag_days,
                "lead_days": self.spec.lead_days,
                "window_days": self.spec.window_days,
            },
            "candidates": list(self.candidates),
            "tune_start": self.tune_start.isoformat(),
            "tune_end": self.tune_end.isoformat(),
            "refit_start": self.refit_start.isoformat(),
            "score_start": self.score_start.isoformat(),
            "score_end": self.score_end.isoformat(),
            "n_trials": self.n_trials,
            "n_boot": self.n_boot,
            "seed": self.seed,
        }

    @property
    def config_id(self) -> str:
        """First 12 hex digits of the SHA-256 of the canonical ``to_dict`` JSON."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def ensemble_config(self, candidates: Sequence[str]) -> EnsembleConfig:
        """``EnsembleConfig`` of a plain mean over ``candidates`` on the scored period.

        Args:
            candidates: Member names, in column order.

        Returns:
            The settings ``combine_oof`` needs.
        """
        return EnsembleConfig(
            spec=self.spec,
            candidates=tuple(candidates),
            methods=("mean",),
            score_start=self.score_start,
            score_end=self.score_end,
            n_boot=self.n_boot,
            seed=self.seed,
        )


@dataclass(frozen=True)
class TuneResult:
    """Outcome of tuning one booster.

    Attributes:
        name: Candidate name.
        best_params: Search-space values of the best completed trial.
        best_mae: Tuning-period MAE of the best trial.
        trial0_mae: Tuning-period MAE of trial 0 (``default_trial_params``).
        default_mae: Tuning-period MAE of the exact ``default_models`` booster.
        n_complete: Completed trials.
        n_pruned: Pruned trials.
        trials: One row per trial with ``TRIALS_SCHEMA``.
    """

    name: str
    best_params: dict[str, Any]
    best_mae: float
    trial0_mae: float
    default_mae: float
    n_complete: int
    n_pruned: int
    trials: pl.DataFrame = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready fields.

        Returns:
            Every field but ``name`` and ``trials``, plus the mean seconds of a
            completed trial; ``best_params`` is not rounded.
        """
        done = self.trials.filter(pl.col("state") == "COMPLETE")
        return {
            "best_params": dict(self.best_params),
            "best_mae": self.best_mae,
            "trial0_mae": self.trial0_mae,
            "default_mae": self.default_mae,
            "n_complete": self.n_complete,
            "n_pruned": self.n_pruned,
            "mean_seconds_complete": done["seconds"].mean() if done.height else None,
        }


def issue_month_folds(
    start: dt.date, end: dt.date, spec: ForecastSpec
) -> list[tuple[dt.date, dt.date]]:
    """Target periods that split the rolling origin into one call per issue month.

    ``rolling_origin`` refits at the first issue date of each month, so
    running it on each of these periods in turn makes the same fits and
    forecasts as one run on ``[start, end]``.

    Args:
        start: First target day.
        end: Last target day.
        spec: Forecast setting.

    Returns:
        ``(first target day, last target day)`` of each issue month, in order.

    Raises:
        ValueError: From ``issue_dates``.
    """
    months: dict[tuple[int, int], list[dt.date]] = {}
    for issue in issue_dates(start, end, spec):
        months.setdefault((issue.year, issue.month), []).append(issue)
    return [
        (spec.target_start(dates[0]), spec.target_end(dates[-1]))
        for dates in months.values()
    ]


def tuning_mae(
    history: pl.DataFrame,
    forecaster: BoostedForecaster,
    config: TuningConfig,
    trial: optuna.Trial | None = None,
) -> float:
    """Rolling-origin MAE of ``forecaster`` on the tuning period.

    Args:
        history: ``date, DIVISION, n, n_fr`` on the full calendar.
        forecaster: Model to score.
        config: Run settings; ``spec``, ``tune_start`` and ``tune_end`` are used.
        trial: When given, the running MAE after each issue month is reported
            to it as step ``k`` and the trial is pruned if Optuna says so.

    Returns:
        MAE over every scored window of the tuning period.

    Raises:
        optuna.TrialPruned: If ``trial`` should be pruned.
        ValueError: From ``rolling_origin``, or if no window is scored.
    """
    frames: list[pl.DataFrame] = []
    for step, (start, end) in enumerate(
        issue_month_folds(config.tune_start, config.tune_end, config.spec)
    ):
        frames.append(rolling_origin(history, [forecaster], config.spec, start, end))
        if trial is None:
            continue
        table = score(pl.concat(frames))
        if table.height:
            trial.report(float(table["mae"].item()), step)
            if trial.should_prune():
                raise optuna.TrialPruned
    table = score(pl.concat(frames)) if frames else pl.DataFrame()
    if table.is_empty():
        msg = f"{forecaster.name}: no scored window in the tuning period"
        raise ValueError(msg)
    return float(table["mae"].item())


def trials_frame(name: str, study: optuna.Study) -> pl.DataFrame:
    """One row per trial of ``study``.

    Args:
        name: Candidate the study tuned.
        study: Finished study.

    Returns:
        Rows with ``TRIALS_SCHEMA`` in trial order; ``value`` is the last
        reported MAE for a pruned trial.
    """
    rows = [
        {
            "model": name,
            "number": t.number,
            "state": t.state.name,
            "value": (
                t.value
                if t.value is not None
                else t.intermediate_values.get(t.last_step)
                if t.last_step is not None
                else None
            ),
            "seconds": (
                (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_complete is not None and t.datetime_start is not None
                else None
            ),
            "n_steps": len(t.intermediate_values),
            "params": json.dumps(t.params, sort_keys=True),
        }
        for t in study.trials
    ]
    return pl.DataFrame(rows, schema=TRIALS_SCHEMA, orient="row")


def tune(history: pl.DataFrame, name: str, config: TuningConfig) -> TuneResult:
    """Tune one booster on the tuning period with TPE and median pruning.

    Args:
        history: ``date, DIVISION, n, n_fr`` on the full calendar.
        name: Tunable candidate.
        config: Run settings.

    Returns:
        The best settings, their MAE, trial 0's and the exact default's MAE,
        and every trial.

    Raises:
        ValueError: If ``name`` is not tunable, or from ``tuning_mae``.
    """
    space = SEARCH_SPACES.get(name)
    if space is None:
        msg = f"{name!r} is not tunable; expected one of {TUNABLE}"
        raise ValueError(msg)

    def objective(trial: optuna.Trial) -> float:
        forecaster = build_forecaster(name, space(trial), config.seed)
        return tuning_mae(history, forecaster, config, trial)

    study = optuna.create_study(
        study_name=name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=PRUNER_STARTUP_TRIALS,
            n_warmup_steps=PRUNER_WARMUP_STEPS,
        ),
    )
    study.enqueue_trial(default_trial_params(name))
    study.optimize(objective, n_trials=config.n_trials, n_jobs=1)
    states = [t.state for t in study.trials]
    trial0 = study.trials[0].value
    if trial0 is None:
        msg = f"{name}: trial 0 (the defaults) did not complete"
        raise ValueError(msg)
    return TuneResult(
        name=name,
        best_params=dict(study.best_params),
        best_mae=float(study.best_value),
        trial0_mae=float(trial0),
        default_mae=tuning_mae(history, default_booster(name, config.seed), config),
        n_complete=states.count(optuna.trial.TrialState.COMPLETE),
        n_pruned=states.count(optuna.trial.TrialState.PRUNED),
        trials=trials_frame(name, study),
    )


def default_booster(name: str, seed: int = DEFAULT_SEED) -> BoostedForecaster:
    """The candidate ``name`` exactly as ``default_models`` builds it.

    Args:
        name: Tunable candidate.
        seed: Random seed.

    Returns:
        The untuned forecaster (LightGBM ``reg_lambda`` 0 included).

    Raises:
        ValueError: If ``name`` is not tunable.
    """
    if name not in SEARCH_SPACES:
        msg = f"{name!r} is not tunable; expected one of {TUNABLE}"
        raise ValueError(msg)
    model = next(m for m in default_models(seed) if m.name == name)
    if not isinstance(model, BoostedForecaster):  # pragma: no cover - registry bug
        msg = f"default model {name!r} is not a BoostedForecaster"
        raise TypeError(msg)
    return model


def tuned_predictions(
    history: pl.DataFrame,
    anomalies: pl.DataFrame,
    best_params: Mapping[str, Mapping[str, Any]],
    config: TuningConfig,
) -> pl.DataFrame:
    """Refit the tuned boosters by rolling origin from ``refit_start`` to ``score_end``.

    Args:
        history: ``date, DIVISION, n, n_fr`` on the full calendar.
        anomalies: ``date, DIVISION, anomaly`` (the EDA anomaly calendar).
        best_params: Search-space values per candidate.
        config: Run settings.

    Returns:
        Rows with ``OUTPUT_SCHEMA``, models named ``tuned_name(candidate)``.

    Raises:
        ValueError: If a candidate has no parameters, or from
            ``build_forecaster`` or ``rolling_origin``.
    """
    missing = sorted(set(config.candidates) - set(best_params))
    if missing:
        msg = f"no tuned parameters for {missing}"
        raise ValueError(msg)
    forecasters = [
        build_forecaster(name, best_params[name], config.seed, tuned_name(name))
        for name in config.candidates
    ]
    results = rolling_origin(
        history, forecasters, config.spec, config.refit_start, config.score_end
    )
    return (
        flag_anomalies(results, anomalies)
        .select(OUTPUT_SCHEMA.names())
        .cast(OUTPUT_SCHEMA)
    )


def tuned_members(config: TuningConfig) -> tuple[str, ...]:
    """``ENSEMBLE_CANDIDATES`` with each tuned candidate replaced by its tuned name.

    Args:
        config: Run settings.

    Returns:
        Member names of ``ens_mean_tuned``, in candidate order.
    """
    return tuple(
        tuned_name(name) if name in config.candidates else name
        for name in ENSEMBLE_CANDIDATES
    )


def comparison(
    oof: pl.DataFrame, tuned: pl.DataFrame, config: TuningConfig
) -> tuple[dict[str, Any], pl.DataFrame]:
    """Score ``ens_mean_tuned`` against ``ens_mean`` and the members on the same rows.

    Args:
        oof: Out-of-fold predictions with ``OUTPUT_SCHEMA`` holding every
            ``ENSEMBLE_CANDIDATES`` model and ``REFERENCES``.
        tuned: Output of ``tuned_predictions`` on the same keys.
        config: Run settings.

    Returns:
        Score tables (overall, without anomalies, by division, by year), the
        paired bootstraps of ``ens_mean_tuned`` against ``ens_mean`` and
        ``REFERENCES`` and of each tuned booster against its untuned version,
        floats rounded to 3; and every scored row of those models.

    Raises:
        ValueError: From ``combine_oof``, or if a single model does not cover
            every scored key.
    """
    current, _ = combine_oof(oof, config.ensemble_config(ENSEMBLE_CANDIDATES))
    members = tuned_members(config)
    combined = pl.concat(
        [oof.select(OUTPUT_SCHEMA.names()), tuned.select(OUTPUT_SCHEMA.names())]
    ).cast(OUTPUT_SCHEMA)
    new, _ = combine_oof(combined, config.ensemble_config(members))
    new = new.with_columns(model=pl.lit(TUNED_ENSEMBLE))
    keys = current.select(KEYS).unique()
    singles = list(dict.fromkeys((*ENSEMBLE_CANDIDATES, *members, *REFERENCES)))
    others = combined.filter(pl.col("model").is_in(singles)).join(
        keys, on=list(KEYS), how="semi"
    )
    counts = dict(others.group_by("model").len().iter_rows())
    short = sorted(name for name in singles if counts.get(name) != keys.height)
    if short:
        msg = f"{short} do not cover the {keys.height} scored keys"
        raise ValueError(msg)
    results = pl.concat([current, new, others]).sort("model", *KEYS)
    with_year = results.with_columns(year=pl.col("target_start").dt.year())

    def table(frame: pl.DataFrame, by: Sequence[str] = ()) -> list[dict[str, Any]]:
        return score(frame, by).to_dicts()

    def boot(model: str, reference: str) -> dict[str, Any]:
        return paired_bootstrap(results, model, reference, config.n_boot, config.seed)

    report = {
        "members": {"current": list(ENSEMBLE_CANDIDATES), "tuned": list(members)},
        "overall": table(results),
        "overall_without_anomalies": table(results.filter(~pl.col("anomaly"))),
        "by_division": table(results, ["DIVISION"]),
        "by_year": table(with_year, ["year"]),
        "bootstrap": {
            "ensemble": [
                boot(TUNED_ENSEMBLE, ref) for ref in (FINAL_MODEL, *REFERENCES)
            ],
            "boosters": [boot(tuned_name(name), name) for name in config.candidates],
        },
    }
    return rounded(report), results


def tuning_report(
    config: TuningConfig,
    tuned: Mapping[str, Mapping[str, Any]],
    compared: Mapping[str, Any],
    input_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Assemble the JSON report.

    Args:
        config: Run settings.
        tuned: ``TuneResult.to_dict`` per candidate (``best_params`` unrounded).
        compared: First output of ``comparison``.
        input_sha256: SHA-256 of each input file, by role.

    Returns:
        Config, config_id, input hashes, periods, library versions, the tuning
        results and the comparison; floats rounded to 3 except the best
        parameters, which are kept exact so ``compare`` can reuse them.
    """
    tuning = {
        name: {
            **rounded({k: v for k, v in result.items() if k != "best_params"}),
            "best_params": dict(result["best_params"]),
        }
        for name, result in tuned.items()
    }
    return {
        "config": config.to_dict(),
        "config_id": config.config_id,
        "input_sha256": dict(input_sha256),
        "periods": {
            "tune": [config.tune_start.isoformat(), config.tune_end.isoformat()],
            "refit": [config.refit_start.isoformat(), config.score_end.isoformat()],
            "score": [config.score_start.isoformat(), config.score_end.isoformat()],
        },
        "versions": {lib: version(lib) for lib in ("optuna", "lightgbm", "catboost")},
        "tuning": tuning,
        "comparison": dict(compared),
    }


def config_from_dict(data: Mapping[str, Any]) -> TuningConfig:
    """Rebuild a ``TuningConfig`` from its ``to_dict`` output.

    Args:
        data: Output of ``TuningConfig.to_dict``.

    Returns:
        The equal config (same ``config_id``).

    Raises:
        ValueError: From ``TuningConfig``.
    """
    spec = data["spec"]
    return TuningConfig(
        spec=ForecastSpec(
            publication_lag_days=spec["publication_lag_days"],
            lead_days=spec["lead_days"],
            window_days=spec["window_days"],
        ),
        candidates=tuple(data["candidates"]),
        tune_start=dt.date.fromisoformat(data["tune_start"]),
        tune_end=dt.date.fromisoformat(data["tune_end"]),
        refit_start=dt.date.fromisoformat(data["refit_start"]),
        score_start=dt.date.fromisoformat(data["score_start"]),
        score_end=dt.date.fromisoformat(data["score_end"]),
        n_trials=data["n_trials"],
        n_boot=data["n_boot"],
        seed=data["seed"],
    )


def _refuse_overwrite(
    outputs: Sequence[Path], inputs: Sequence[Path], *, force: bool
) -> None:
    """Exit before any work if an output exists or would replace an input.

    Args:
        outputs: Paths the run writes.
        inputs: Paths the run reads.
        force: Allow replacing existing outputs (never inputs).

    Raises:
        typer.Exit: If an output is an input, or exists without ``force``.
    """
    resolved = {path.resolve() for path in inputs}
    for path in outputs:
        if path.resolve() in resolved:
            typer.echo(f"{path} is also an input; refusing to overwrite it", err=True)
            raise typer.Exit(code=1)
        if path.exists() and not force:
            typer.echo(
                f"{path} already exists. Refusing to overwrite it; pass --force "
                "only if you mean to replace that result.",
                err=True,
            )
            raise typer.Exit(code=1)
    if len({path.resolve() for path in outputs}) != len(outputs):
        typer.echo("output paths must differ", err=True)
        raise typer.Exit(code=1)


def _check_inputs(paths: Sequence[Path]) -> None:
    """Exit if an input file is missing.

    Args:
        paths: Files the run reads.

    Raises:
        typer.Exit: If one is not a file.
    """
    for path in paths:
        if not path.is_file():
            typer.echo(f"{path} not found", err=True)
            raise typer.Exit(code=1)


def _check_current_mean(compared: Mapping[str, Any], ensemble_report: Path) -> None:
    """Exit if the recomputed ``ens_mean`` MAE differs from the ensemble report's.

    Args:
        compared: First output of ``comparison``.
        ensemble_report: ``ensemble validate`` report; skipped when absent.

    Raises:
        typer.Exit: If both MAEs exist and differ.
    """
    if not ensemble_report.is_file():
        typer.echo(f"{ensemble_report} not found; ens_mean not cross-checked")
        return
    expected = {
        row["model"]: row["mae"]
        for row in json.loads(ensemble_report.read_text())["overall"]
    }.get(FINAL_MODEL)
    got = {row["model"]: row["mae"] for row in compared["overall"]}[FINAL_MODEL]
    if expected != got:
        typer.echo(
            f"recomputed {FINAL_MODEL} MAE {got} differs from {expected} in "
            f"{ensemble_report}; the predictions or period do not match",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(f"{FINAL_MODEL} MAE {got} matches {ensemble_report}")


def _write_and_echo(
    report: dict[str, Any], rows: pl.DataFrame, out: Path, tuned_out: Path
) -> None:
    """Write the report and the tuned predictions, then print a summary.

    Args:
        report: Output of ``tuning_report``.
        rows: Tuned booster predictions.
        out: JSON report path.
        tuned_out: Parquet path of the tuned predictions.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    tuned_out.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(tuned_out)
    typer.echo(f"config {report['config_id']}")
    for name, result in report["tuning"].items():
        typer.echo(
            f"{name:<18} tuning MAE default {result['default_mae']:.3f}  "
            f"trial0 {result['trial0_mae']:.3f}  best {result['best_mae']:.3f}  "
            f"({result['n_complete']} complete, {result['n_pruned']} pruned)"
        )
    compared = report["comparison"]
    for row in compared["overall"]:
        typer.echo(
            f"{row['model']:<24} MAE {row['mae']:6.3f}  bias {row['bias']:+6.3f}"
        )
    for row in (*compared["bootstrap"]["ensemble"], *compared["bootstrap"]["boosters"]):
        typer.echo(
            f"{row['model']:<24} vs {row['reference']:<18} diff {row['diff']:+6.3f}  "
            f"95% CI [{row['ci_low']:+6.3f}, {row['ci_high']:+6.3f}]"
        )
    typer.echo(f"wrote {out} and {tuned_out}")


@app.callback()
def main() -> None:
    """Tune the boosted ensemble members and compare the ensembles."""


@app.command()
def run(
    history_path: Annotated[
        Path, typer.Option("--history", help="Division-day parquet.")
    ] = DEFAULT_HISTORY,
    anomalies_path: Annotated[
        Path, typer.Option("--anomalies", help="Anomaly calendar parquet.")
    ] = DEFAULT_ANOMALIES,
    predictions: Annotated[
        Path, typer.Option(help="Out-of-fold predictions of the untuned models.")
    ] = DEFAULT_PREDICTIONS,
    ensemble_report: Annotated[
        Path, typer.Option(help="Ensemble report to cross-check ens_mean against.")
    ] = ENSEMBLE_REPORT,
    out: Annotated[Path, typer.Option(help="JSON report path.")] = DEFAULT_OUT,
    tuned_predictions_path: Annotated[
        Path,
        typer.Option("--tuned-predictions", help="Parquet path of the tuned boosters."),
    ] = DEFAULT_TUNED_PREDICTIONS,
    trials_out: Annotated[
        Path, typer.Option(help="Parquet path of every trial.")
    ] = DEFAULT_TRIALS_OUT,
    trials: Annotated[
        int, typer.Option(min=1, help="Optuna trials per booster.")
    ] = DEFAULT_TRIALS,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite existing outputs.")
    ] = False,
) -> None:
    """Tune each booster on 2021, then compare the ensembles on 2022-2024.

    Args:
        history_path: Division-day parquet.
        anomalies_path: Anomaly calendar parquet.
        predictions: Out-of-fold predictions of the untuned models (read only).
        ensemble_report: Ensemble report whose ``ens_mean`` MAE must match.
        out: JSON report path; must not exist unless ``force``.
        tuned_predictions_path: Parquet path of the tuned boosters'
            predictions; must not exist unless ``force``.
        trials_out: Parquet path of every trial; must not exist unless
            ``force``.
        trials: Optuna trials per booster, trial 0 (the defaults) included.
        force: Overwrite existing outputs.

    Raises:
        typer.Exit: If an output exists without ``--force`` or is an input,
            an input file is missing, or ``ens_mean`` does not match.
    """
    inputs = [history_path, anomalies_path, predictions, ensemble_report]
    _refuse_overwrite([out, tuned_predictions_path, trials_out], inputs, force=force)
    _check_inputs([history_path, anomalies_path, predictions])
    config = TuningConfig(n_trials=trials)
    history = pl.read_parquet(history_path)
    results = [tune(history, name, config) for name in config.candidates]
    best = {r.name: r.best_params for r in results}
    rows = tuned_predictions(history, pl.read_parquet(anomalies_path), best, config)
    compared, _ = comparison(pl.read_parquet(predictions), rows, config)
    _check_current_mean(compared, ensemble_report)
    report = tuning_report(
        config,
        {r.name: r.to_dict() for r in results},
        compared,
        {
            "history": sha256_file(history_path),
            "anomalies": sha256_file(anomalies_path),
            "predictions": sha256_file(predictions),
        },
    )
    _write_and_echo(report, rows, out, tuned_predictions_path)
    trials_out.parent.mkdir(parents=True, exist_ok=True)
    pl.concat([r.trials for r in results]).write_parquet(trials_out)
    typer.echo(f"wrote {trials_out}")


@app.command()
def compare(
    tuning: Annotated[
        Path, typer.Option(help="Report of an earlier run holding the best params.")
    ] = DEFAULT_OUT,
    history_path: Annotated[
        Path, typer.Option("--history", help="Division-day parquet.")
    ] = DEFAULT_HISTORY,
    anomalies_path: Annotated[
        Path, typer.Option("--anomalies", help="Anomaly calendar parquet.")
    ] = DEFAULT_ANOMALIES,
    predictions: Annotated[
        Path, typer.Option(help="Out-of-fold predictions of the untuned models.")
    ] = DEFAULT_PREDICTIONS,
    ensemble_report: Annotated[
        Path, typer.Option(help="Ensemble report to cross-check ens_mean against.")
    ] = ENSEMBLE_REPORT,
    out: Annotated[Path, typer.Option(help="JSON report path.")] = Path(
        "reports/tuning_comparison.json"
    ),
    tuned_predictions_path: Annotated[
        Path,
        typer.Option("--tuned-predictions", help="Parquet path of the tuned boosters."),
    ] = DEFAULT_TUNED_PREDICTIONS,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite existing outputs.")
    ] = False,
) -> None:
    """Refit with the best params of an earlier run and redo only the comparison.

    The settings come from the earlier report's ``config``, so the periods,
    seed and candidates match the run that found the parameters.

    Args:
        tuning: JSON report written by ``run``.
        history_path: Division-day parquet.
        anomalies_path: Anomaly calendar parquet.
        predictions: Out-of-fold predictions of the untuned models (read only).
        ensemble_report: Ensemble report whose ``ens_mean`` MAE must match.
        out: JSON report path; must not exist unless ``force``.
        tuned_predictions_path: Parquet path of the tuned boosters'
            predictions; must not exist unless ``force``.
        force: Overwrite existing outputs.

    Raises:
        typer.Exit: If an output exists without ``--force`` or is an input,
            an input file is missing, or ``ens_mean`` does not match.
    """
    inputs = [tuning, history_path, anomalies_path, predictions, ensemble_report]
    _refuse_overwrite([out, tuned_predictions_path], inputs, force=force)
    _check_inputs([tuning, history_path, anomalies_path, predictions])
    earlier = json.loads(tuning.read_text())
    config = config_from_dict(earlier["config"])
    best = {name: result["best_params"] for name, result in earlier["tuning"].items()}
    history = pl.read_parquet(history_path)
    rows = tuned_predictions(history, pl.read_parquet(anomalies_path), best, config)
    compared, _ = comparison(pl.read_parquet(predictions), rows, config)
    _check_current_mean(compared, ensemble_report)
    report = tuning_report(
        config,
        earlier["tuning"],
        compared,
        {
            "tuning": sha256_file(tuning),
            "history": sha256_file(history_path),
            "anomalies": sha256_file(anomalies_path),
            "predictions": sha256_file(predictions),
        },
    )
    _write_and_echo(report, rows, out, tuned_predictions_path)


if __name__ == "__main__":
    app()
